	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_gfx1201:     ; @attention_fp8_e4m3_fa2_gqa_gfx1201
.Lfunc_begin0:
	.file	0 "/home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt" "/home/kaden/ClaudeCode/warpfront/wt-fapkt/kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0xcbcbc8b99da4d3b92241381d394891d4
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.file	1 "/home/kaden/ClaudeCode/warpfront/wt-fapkt" "kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0xcbcbc8b99da4d3b92241381d394891d4
	.loc	1 1334 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1334:17
	s_load_b128 s[4:7], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	1 1334 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1334:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_89
; %bb.1:
	.loc	1 1337 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1337:14
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB0_89
; %bb.2:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0
	s_load_b32 s22, s[0:1], 0x38
	.loc	1 1339 35 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1339:35
	s_lshl_b32 s3, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 1340 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1340:16
	s_cmp_ge_i32 s3, s7
	s_cbranch_scc1 .LBB0_89
; %bb.3:
.Ltmp0:
	.loc	1 957 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:957:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 0
	v_lshrrev_b32_e32 v156, 4, v0
	v_lshrrev_b32_e32 v157, 3, v0
	v_mov_b32_e32 v147, 0
	s_mov_b32 s23, 0
	.loc	1 974 9                         ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:974:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_5
; %bb.4:
	.loc	1 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshrrev_b32_e32 v1, 4, v0
	.loc	1 976 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:976:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v145, v0, 7, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_b32_e32 v1, 6, v1
	.loc	1 978 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_lo_u32 v3, v145, 24
	.loc	1 977 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:977:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_gt_i32_e32 vcc_lo, s7, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	1 975 25                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[1:2], null, ttmp7, 6, v[1:2]
	.loc	1 975 45 is_stmt 0              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshrrev_b32_e32 v2, 3, v0
	s_and_b32 s23, vcc_lo, exec_lo
	.loc	1 975 39                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v147, v2, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 978 55 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_lshl_u32 v2, v147, v3, 8
	.loc	1 1150 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1150:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b32_e32 v3, 0
.Ltmp1:
.LBB0_5:
	.loc	1 0 38 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:38
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1334 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1334:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_dual_mov_b32 v5, -1 :: v_dual_and_b32 v6, 31, v0
	v_bfrev_b32_e32 v7, -2
.Ltmp2:
	.loc	1 991 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:991:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 8, v6
	s_cbranch_execz .LBB0_9
; %bb.6:
	.loc	1 992 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:992:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v4, s3, v6
	v_bfrev_b32_e32 v7, -2
	v_mov_b32_e32 v5, -1
	.loc	1 993 16                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:993:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v4
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, vcc_lo, s0, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	.loc	1 994 27 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:994:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b32 v5, v[4:5], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v7, v5
.LBB0_8:
	.loc	1 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
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
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshrrev_b32_e32 v161, 5, v0
.Ltmp5:
	.file	2 "/opt/rocm/core/include/hip/amd_detail" "amd_warp_functions.h" md5 0x78b3d571ca85d7d16c60115ed6cb4554
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v1, 16, v11
.Ltmp6:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v3, 4, v11
	v_cndmask_b32_e64 v15, 0, v2, s23
	v_ashrrev_i32_e32 v146, 31, v145
	v_lshrrev_b32_e32 v2, 1, v6
.Ltmp7:
	.loc	2 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
.Ltmp8:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v17, 1, v11
	v_lshl_add_u32 v160, v6, 3, 0
	v_lshlrev_b64_e32 v[129:130], 2, v[145:146]
	s_mov_b32 s17, 0
.Ltmp9:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v11, v1, vcc_lo
	s_mul_i32 s16, s7, 0x1800
	s_mov_b32 s4, ttmp7
	s_ashr_i32 s5, ttmp7, 31
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[18:19], s[8:9], s[16:17]
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v158, 2, v1
.Ltmp10:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v1, 8, v11
	s_lshl_b64 s[4:5], s[4:5], 8
	s_lshl_b32 s6, ttmp7, 1
	s_add_nc_u64 s[20:21], s[10:11], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s7, s6, 31
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	s_mov_b32 s16, s17
	s_wait_alu depctr_va_vcc(0)
	v_dual_mov_b32 v168, 0xff800000 :: v_dual_cndmask_b32 v9, v11, v1
.Ltmp11:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v4, v158, v5
.Ltmp12:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v8, v158, v7
.Ltmp13:
	.loc	2 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_mov_b32_e32 v1, 0
.Ltmp14:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v9, 2, v9
	v_and_b32_e32 v146, 8, v2
	v_lshlrev_b32_e32 v162, 7, v159
	v_lshlrev_b32_e32 v18, 4, v159
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v2, v1
.Ltmp15:
	.file	3 "/opt/rocm/core-10.0/lib/llvm/lib/clang/23/include" "__clang_hip_math.h" md5 0x036a896c80a276b79275b247e57f4652
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v4, v5, v4
.Ltmp16:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v7, v8
	v_lshrrev_b32_e32 v7, 1, v0
	v_lshlrev_b32_e32 v8, 3, v0
.Ltmp17:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v11, v3, vcc_lo
.Ltmp18:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v12, v9, v4
.Ltmp19:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v13, v9, v5
	v_mov_b32_e32 v167, 0
	v_and_b32_e32 v19, 0xf8, v8
.Ltmp20:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v163, 8, v7
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v7, v1
.Ltmp21:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_or_b32 v133, v161, 8, v19
	v_mad_co_u64_u32 v[9:10], null, v145, 24, v[147:148]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v10, v1 :: v_dual_add_nc_u32 v165, 0, v133
	v_lshlrev_b64_e32 v[131:132], 2, v[9:10]
.Ltmp22:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v12, v4, v12
.Ltmp23:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v13, v5, v13
.Ltmp24:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_xor_b32_e32 v4, 2, v11
	v_mov_b32_e32 v5, v1
	v_add_co_u32 v148, s3, s20, v18
.Ltmp25:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v16, v0, v12
.Ltmp26:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v0, v0, v13
.Ltmp27:
	.loc	2 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v149, null, s21, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[10:11], s[6:7]
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v11, v4, vcc_lo
.Ltmp28:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v17
	v_mov_b32_e32 v4, v1
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, v11, v17, vcc_lo
.Ltmp29:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v134, vcc_lo, v15, v146
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, 0, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp30:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v89, 2, v66
.Ltmp31:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_lshlrev_b32_e32 v20, 2, v8
	v_mov_b32_e32 v8, v1
.Ltmp32:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v65, v12, v16
.Ltmp33:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v13, v0
	v_add_co_u32 v150, vcc_lo, s0, v129
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v11, v3
	v_mov_b32_e32 v13, v5
.Ltmp34:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v67, v20, v65
.Ltmp35:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v68, v20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v151, null, s1, v130, vcc_lo
.Ltmp36:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v136, v65, v67
.Ltmp38:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v68
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v49, v1
.Ltmp39:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	ds_bpermute_b32 v137, v89, v136
.Ltmp40:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
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
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v131, v136, v137
.Ltmp42:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v138
.Ltmp43:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v130, vcc_lo
	v_dual_mov_b32 v15, v7 :: v_dual_lshlrev_b32 v164, 4, v161
	.loc	1 1009 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1009:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1010 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1010:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_readfirstlane_b32 s25, v0
	v_mov_b32_e32 v134, v6
	v_mov_b32_e32 v132, v4
	v_mov_b32_e32 v130, v2
	s_add_nc_u64 s[18:19], s[12:13], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[6:7]
	s_branch .LBB0_13
.LBB0_10:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	1 0 22 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:22
	v_mov_b32_e32 v168, v2
.LBB0_11:                               ;   in Loop: Header=BB0_13 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s27
.Ltmp44:
	.file	4 "/opt/rocm/core/include/hip/amd_detail" "amd_device_functions.h" md5 0xa1a79f59a27f4196ae31454cacd6cd14
	.loc	4 701 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp45:
.LBB0_12:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	4 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
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
	.loc	1 1015 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1015:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_lshl_b32 s26, s16, 6
	.loc	1 1016 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1016:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s26, s24
	s_cselect_b32 s11, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.14:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
	s_mov_b32 s0, 8
	s_branch .LBB0_16
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1032 49 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_b32_e32 v7, 0xe0, v6
	.loc	1 1024 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 1032 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v7, v7, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1032 54 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:54 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v7, v0, 15, v7
	.loc	1 1024 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v0, 8, v0
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	.loc	1 1024 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc1 .LBB0_18
.LBB0_16:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 9                           ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s17, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	.loc	1 1033 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1033:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s24, v7
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	.loc	1 1037 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1037:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v7, v[148:149]
	.loc	1 1038 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1038:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	1 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v6, s26, v163
	v_dual_mov_b32 v7, v164 :: v_dual_mov_b32 v8, v161
	s_movk_i32 s0, 0xc000
	s_branch .LBB0_22
.LBB0_19:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB0_20:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_21:                               ;   in Loop: Header=BB0_22 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1088 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v0, s0, v165
	.loc	1 1049 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 1088 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_store_b64 v0, v[2:3] offset:32768
	.loc	1 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc1 .LBB0_38
.LBB0_22:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1058 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1058:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	.loc	1 1057 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1057:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_and_or_b32 v0, 0xf0, v7, v159
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1062 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v2, 7, v137
	.loc	1 1062 27 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1065 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1065:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, s[18:19]
	.loc	1 1067 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1067:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1073 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1073:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v139.l, v5.h
	.loc	1 1066 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1066:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	1 1068 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	.loc	1 1068 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	.loc	1 1069 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1069:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	.loc	1 1070 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	.loc	1 1072 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	.loc	1 1068 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	1 1070 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or3_b32 v0, v0, v2, v3
	.loc	1 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	.loc	1 1074 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	.loc	1 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or3_b32 v0, v0, 0, 0
	.loc	1 1074 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB0_24:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_21
; %bb.25:                               ;   in Loop: Header=BB0_22 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	.loc	1 1075 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1075:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s24, v137
	s_cbranch_execz .LBB0_20
; %bb.26:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1077 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1077:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v4, s4, s18, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s19, 0, s4
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e32 v0.h, 0
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmpx_gt_i32_e64 s24, v137
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1078 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB0_28:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 2, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB0_30:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 3, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	.loc	1 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_u8 v0, v[138:139], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB0_32:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 4, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB0_34:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 5, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	1 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v3, v0, v3
.LBB0_36:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, 6, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s24, v0
	s_cbranch_execz .LBB0_19
; %bb.37:                               ;   in Loop: Header=BB0_22 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v0, v[4:5]
	.loc	1 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e32 v0.l, 0
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_d16_hi_u8 v0, v[4:5], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB0_19
.LBB0_38:                               ;   in Loop: Header=BB0_13 Depth=1
.Ltmp46:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp47:
	.loc	1 1096 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1096:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_11
; %bb.39:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b32_e32 v0, s22
	.loc	1 1111 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1111:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	1 1112 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1112:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b32 v0, v[152:153], off
	.loc	1 1113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s22, v0
.LBB0_41:                               ;   in Loop: Header=BB0_13 Depth=1
	.loc	1 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 1098 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_or_b32 s0, s26, 16
	v_mov_b32_e32 v4, v160
	.loc	1 1098 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s24
	s_mov_b32 s28, 0
	s_cselect_b32 s29, -1, 0
	s_branch .LBB0_44
.LBB0_42:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v168, v2
	.loc	1 1189 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v168
	.loc	1 1209 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v172, null, v3, v3, v140
	.loc	1 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v174, null, v3, v3, v139
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v177, null, v3, v3, v137
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v176, null, v3, v3, v8
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v173, v172
	.loc	1 1251 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1251:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_add_u32 v188, s28, 12, v160
	v_exp_f32_e32 v143, v143
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v180, v177
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v179, v176
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v175, -v172, v173, 1.0
	.loc	1 1189 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_fmac_f32 v173, v175, v173 :: v_dual_mul_f32 v144, v166, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1226 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v166, null, v3, v3, v141
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v134, v134, v6
	v_dual_mul_f32 v130, v130, v6 :: v_dual_fmac_f32 v5, v167, v143
	v_mul_f32_e32 v128, v128, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v143, null, v3, v3, v142
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v168, v166
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v126, v126, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v124, v124, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v122, v122, v6
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v170, -v166, v168, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v25, v25, v6
	v_dual_mul_f32 v129, v129, v6 :: v_dual_mul_f32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v143, v144, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v33, v33, v6 :: v_dual_fmac_f32 v168, v170, v168
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v170, s0, v141, v3, v141
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v127, v127, v6 :: v_dual_mul_f32 v118, v118, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v144, v167, v144
	v_div_scale_f32 v167, vcc_lo, v142, v3, v142
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v16, v16, v6 :: v_dual_mul_f32 v125, v125, v6
	v_dual_mul_f32 v116, v116, v6 :: v_dual_mul_f32 v123, v123, v6
	v_dual_mul_f32 v114, v114, v6 :: v_dual_mul_f32 v169, v167, v144
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v112, v112, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v110, v110, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v171, -v143, v169, v167
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v117, v117, v6 :: v_dual_mul_f32 v108, v108, v6
	v_dual_mul_f32 v115, v115, v6 :: v_dual_mul_f32 v106, v106, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v169, v171, v144
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v171, v170, v168
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v104, v104, v6
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v102, v102, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v143, -v143, v169, v167
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v166, v171, v170
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v98, v98, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v171, v167, v168
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v167, s1, v140, v3, v140
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v144, v174
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v143, v143, v3, v142
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v142, -v166, v171, v170
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v166, v167, v173
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v170, null, v3, v3, v7
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v96, v96, v6
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v168, v142, v168, v171
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v171, -v172, v166, v167
	s_mov_b32 vcc_lo, s1
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v175, v170
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v174, v144, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v19, v19, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v166, v171, v173
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v142.h, 0
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v142.l, v1.l
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	.loc	1 1242 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v172, v166, v167
	.loc	1 1245 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v178, -v170, v175, 1.0
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v168, v168, v3, v141
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v141.h, v142.h
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b16_e64 v141.l, v142.l
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v167, v173, v166
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v167, null, v3, v3, v138
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v173, -v176, v179, 1.0
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v141.l, v143, v168
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v140, v166, v3, v140
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v166, -v177, v180, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v15, v15, v6
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v179, v173, v179
	v_div_scale_f32 v173, s3, v8, v3, v8
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v94, v94, v6
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v180, v166, v180
	v_div_scale_f32 v166, s4, v137, v3, v137
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v10, v10, v6 :: v_dual_mul_f32 v101, v101, v6
	v_dual_mul_f32 v92, v92, v6 :: v_dual_mul_f32 v99, v99, v6
	v_mul_f32_e32 v90, v90, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v182, v166, v180
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v175, v178, v175
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v171, v169, v144
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_rcp_f32_e32 v178, v167
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v88, v88, v6
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v86, v86, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v172, -v174, v171, v169
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v84, v84, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v82, v82, v6
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v171, v172, v144
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_scale_f32 v172, s1, v7, v3, v7
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v181, -v167, v178, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v17, v17, v6
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v80, v80, v6
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v174, v171, v169
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v174, v172, v175
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v178, v181, v178
	v_div_scale_f32 v181, s0, v138, v3, v138
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v78, v78, v6
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v171, v173, v179
	.loc	1 1245 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v170, v174, v172
	.loc	1 1247 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v183, v181, v178
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v139, v144, v3, v139
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v144, -v176, v171, v173
	.loc	1 1245 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v174, v169, v175
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v169, -v177, v182, v166
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v76, v76, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v171, v144, v179
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v144, -v167, v183, v181
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v170, -v170, v174, v172
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v182, v169, v180
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v74, v74, v6
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fmac_f32_e32 v183, v144, v178
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	.loc	1 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v170, -v176, v171, v173
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v144, -v177, v182, v166
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v167, -v167, v183, v181
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v7, v169, v3, v7
	.loc	1 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v170, v179, v171
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v72, v72, v6
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	.loc	1 1247 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1245 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v8, v166, v3, v8
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v167, v167, v178, v183
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v70, v70, v6
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v137, v144, v3, v137
	.loc	1 1244 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1244:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v138, v167, v3, v138
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[7:8], v188 offset:16384
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v68, v68, v6
	.loc	1 1246 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1246:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[137:138], v188 offset:16640
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[139:140], v188 offset:16896
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[143:144], v188 offset:17152
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[166:167], v188 offset:17408
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[168:169], v188 offset:17664
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[170:171], v188 offset:17920
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[172:173], v188 offset:18176
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[174:175], v188 offset:18432
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[176:177], v188 offset:18688
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[178:179], v188 offset:18944
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[180:181], v188 offset:19200
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[182:183], v188 offset:19456
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[184:185], v188 offset:19712
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[186:187], v188 offset:19968
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_b64 v[188:189], v188 offset:20224
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1253 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1253:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v168, v2
	.loc	1 1116 40 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_add_co_i32 s28, s28, 1
	.loc	1 1116 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s28, 4
	.loc	1 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc0 .LBB0_10
.LBB0_44:                               ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_49 Depth 3
	.loc	1 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s28, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB0_47
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=2
	s_cmp_eq_u32 s28, 1
	s_mov_b32 s0, s29
	s_cbranch_scc1 .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 1118 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cmp_eq_u32 s28, 2
	.loc	1 1118 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s24, s0
	s_cselect_b32 s0, -1, 0
.LBB0_47:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
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
	.loc	1 1151 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1151:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	.loc	1 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	.loc	1 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	1 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	.loc	1 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_addk_co_i32 s0, 0x1000
	.loc	1 1159 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1159:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	.loc	1 1155 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1155:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1139 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_cbranch_scc1 .LBB0_49
; %bb.50:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 1161 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1161:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_lshl4_add_u32 s0, s28, s26
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	.loc	1 1162 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	.loc	1 1162 49 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s25
	s_cselect_b32 s30, -1, 0
	s_cmp_gt_i32 s1, s25
	s_cselect_b32 s1, -1, 0
	.loc	1 1164 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s23
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 1164 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	global_load_b32 v170, v[150:151], off
.LBB0_52:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v146
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp48:
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[5:6], null, 0x408, v2, s[20:21]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.Ltmp49:
.LBB0_54:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v5, 1, v2
.Ltmp50:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_gt_i32_e64 s0, s24, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[6:7], null, 0x408, v5, s[20:21]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.Ltmp51:
.LBB0_56:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp52:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s1, s24, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v6, s[20:21]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.Ltmp53:
.LBB0_58:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp54:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s3, s24, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v7, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.Ltmp55:
.LBB0_60:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp56:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s4, s24, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v8, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.Ltmp57:
.LBB0_62:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp58:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s5, s24, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v169, s[20:21]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.Ltmp59:
.LBB0_64:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp60:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s6, s24, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[180:181], null, 0x408, v171, s[20:21]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.Ltmp61:
.LBB0_66:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
.Ltmp62:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_cmp_ge_i32_e64 s7, s24, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[180:181], null, 0x408, v172, s[20:21]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.Ltmp63:
.LBB0_68:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
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
	.loc	1 1178 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s30, s9
	s_or_b32 s8, s30, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
.Ltmp65:
	.loc	1 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s23, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s23, s9
.Ltmp66:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v143, v170, v142, v140
.Ltmp67:
	.loc	1 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp68:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v175, v143, v173, v144
.Ltmp69:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1186:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ]
	ds_bpermute_b32 v176, v158, v175
.Ltmp70:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB0_70:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB0_72:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB0_74:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB0_76:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB0_78:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB0_85
; %bb.79:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB0_86
.LBB0_80:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 14                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB0_82
.LBB0_81:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.Ltmp71:
.LBB0_82:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp72:
	.loc	3 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1187:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v168, v175, v176
.Ltmp73:
	.loc	1 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v7, v137
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v141, v137, v170
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v140, v138, v143
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
.Ltmp74:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v6, v142, 0, v141
.Ltmp75:
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v170, v3
.Ltmp76:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
.Ltmp77:
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_f32_e32 v137, v5, v174
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp78:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v6, v6, v7, v8
.Ltmp79:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp80:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max3_num_f32 v144, v6, v137, v138
.Ltmp81:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v3, v5, v3
.Ltmp82:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ]
	ds_bpermute_b32 v169, v158, v144
.Ltmp83:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_add_f32_e32 v5, v143, v3
.Ltmp84:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ] ]
	ds_bpermute_b32 v6, v158, v5
.Ltmp85:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v166
.Ltmp86:
	.loc	1 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB0_42
; %bb.83:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 1223 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB0_42
.Ltmp88:
.LBB0_84:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v168
	s_branch .LBB0_43
.LBB0_85:                               ;   in Loop: Header=BB0_44 Depth=2
.Ltmp89:
	.loc	1 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB0_80
.LBB0_86:                               ;   in Loop: Header=BB0_44 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	v_mad_co_u64_u32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB0_81
	s_branch .LBB0_82
.Ltmp90:
.LBB0_87:
	.loc	1 1267 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1267:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_and_saveexec_b32 s0, s23
	s_cbranch_execz .LBB0_89
; %bb.88:
	.loc	1 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1272 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1272:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mul_lo_u32 v4, 0x1800, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v1, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	.loc	1 1272 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1272:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshl_add_u32 v2, v147, 8, v4
	.loc	1 1270 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v167
	.loc	1 1278 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_mov_b32_e32 v1, 0
	.loc	1 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_div_fixup_f32 v3, v0, v167, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1274 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1274:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_or_b32_e32 v0, v2, v146
	.loc	1 1270 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1270:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 1278 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v139, v166, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s15, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 1278 60 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x7
	global_store_b128 v[137:138], v[0:3], off
	global_store_b128 v[137:138], v[4:7], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	.loc	1 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	v_dual_mul_f32 v0, v89, v139 :: v_dual_mul_f32 v3, v92, v139
	v_dual_mul_f32 v1, v90, v139 :: v_dual_mul_f32 v2, v91, v139
	v_dual_mul_f32 v5, v94, v139 :: v_dual_mul_f32 v4, v93, v139
	v_dual_mul_f32 v7, v96, v139 :: v_dual_mul_f32 v6, v95, v139
	v_dual_mul_f32 v81, v81, v139 :: v_dual_mul_f32 v82, v82, v139
	v_dual_mul_f32 v83, v83, v139 :: v_dual_mul_f32 v84, v84, v139
	v_dual_mul_f32 v85, v85, v139 :: v_dual_mul_f32 v86, v86, v139
	v_dual_mul_f32 v87, v87, v139 :: v_dual_mul_f32 v88, v88, v139
	.loc	1 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[101:104], off offset:272
	global_store_b128 v[137:138], v[0:3], off offset:320
	global_store_b128 v[137:138], v[4:7], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	.loc	1 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:448
	global_store_b128 v[137:138], v[4:7], off offset:464
	global_store_b128 v[137:138], v[65:68], off offset:512
	global_store_b128 v[137:138], v[69:72], off offset:528
	global_store_b128 v[137:138], v[57:60], off offset:576
	global_store_b128 v[137:138], v[61:64], off offset:592
	.loc	1 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:640
	global_store_b128 v[137:138], v[4:7], off offset:656
	global_store_b128 v[137:138], v[41:44], off offset:704
	global_store_b128 v[137:138], v[45:48], off offset:720
	global_store_b128 v[137:138], v[33:36], off offset:768
	global_store_b128 v[137:138], v[37:40], off offset:784
	.loc	1 1278 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
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
	.loc	1 1278 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1278:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1342:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[0:3], off offset:832
	global_store_b128 v[137:138], v[4:7], off offset:848
	global_store_b128 v[137:138], v[17:20], off offset:896
	global_store_b128 v[137:138], v[21:24], off offset:912
	global_store_b128 v[137:138], v[8:11], off offset:960
	global_store_b128 v[137:138], v[12:15], off offset:976
.Ltmp91:
.LBB0_89:
	.loc	1 1345 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1345:1
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
	.file	5 "/opt/rocm/core/include/hip/amd_detail" "amd_hip_runtime.h" md5 0xe1e2844c29b46b290fe1912868fc4e78
	.loc	5 248 59 prologue_end           ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1423:26 ] ]
	s_load_b32 s2, s[0:1], 0x20
.Ltmp93:
	.loc	1 1424 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1424:26
	v_lshrrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1426 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1426:39
	v_lshl_or_b32 v1, ttmp9, 2, v1
	.loc	1 1427 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1427:25
	s_wait_kmcnt 0x0
	s_mul_i32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	.loc	1 1427 11 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1427:11
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB1_2
; %bb.1:
	.loc	1 1429 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1429:22
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
.Ltmp94:
	.loc	5 248 59                        ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1423:26 ] ]
	s_load_b128 s[0:3], s[0:1], 0x0
.Ltmp95:
	.loc	1 1431 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:25
	v_mov_b32_e32 v9, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1431 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:48
	v_mov_b32_e32 v11, v9
	.loc	1 1429 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1429:22
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, v2, v3
	.loc	1 1430 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1430:21
	v_mul_lo_u32 v3, v2, 24
	.loc	1 1431 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:40
	v_mul_lo_u32 v8, 0x1800, v2
	.loc	1 1433 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:24
	v_lshlrev_b32_e32 v2, 3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v12, 0xf8, v2
	.loc	1 1430 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1430:21
	v_sub_nc_u32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 1433 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:16
	v_lshlrev_b32_e32 v4, 2, v12
	.loc	1 1431 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:62
	v_lshlrev_b32_e32 v10, 8, v1
	.loc	1 1431 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:25
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]
	.loc	1 1432 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:24
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]
	.loc	1 1451 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:5
	v_lshlrev_b32_e32 v12, 1, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	1 1431 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:48
	v_lshlrev_b64_e32 v[2:3], 2, v[10:11]
	.loc	1 1432 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:47
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]
	.loc	1 1431 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:25
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1431 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1431:48
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1433 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:16
	v_add_co_u32 v4, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v1, vcc_lo
	.loc	1 1432 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:24
	v_add_co_u32 v8, vcc_lo, s2, v8
	.loc	1 1433 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1433:16
	s_clause 0x1
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[4:5], off offset:16
	.loc	1 1432 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v9, vcc_lo
	.loc	1 1432 47 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1432:47
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	.loc	1 1447 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1447:22
	s_wait_loadcnt 0x1
	v_cvt_f16_f32_e32 v0.l, v0
	.loc	1 1447 43 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1447:43
	v_cvt_f16_f32_e32 v0.h, v1
	.loc	1 1448 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1448:22
	v_cvt_f16_f32_e32 v1.l, v2
	.loc	1 1449 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1449:22
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v2.l, v4
	.loc	1 1451 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:5
	v_add_co_u32 v4, vcc_lo, v8, v12
	.loc	1 1448 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1448:43
	v_cvt_f16_f32_e32 v1.h, v3
	.loc	1 1449 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1449:43
	v_cvt_f16_f32_e32 v2.h, v5
	.loc	1 1450 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1450:22
	v_cvt_f16_f32_e32 v3.l, v6
	.loc	1 1450 43 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1450:43
	v_cvt_f16_f32_e32 v3.h, v7
	.loc	1 1451 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v9, vcc_lo
	.loc	1 1451 32 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1451:32
	global_store_b128 v[4:5], v[0:3], off
.LBB1_2:
	.loc	1 1452 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1452:1
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
	.loc	5 248 59 prologue_end           ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1471:26 ] ]
	s_load_b32 s2, s[0:1], 0x18
.Ltmp97:
	.loc	1 1472 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1472:26
	v_lshrrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1474 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1474:39
	v_lshl_or_b32 v1, ttmp9, 2, v1
	.loc	1 1475 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1475:25
	s_wait_kmcnt 0x0
	s_mul_i32 s2, s2, 24
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	.loc	1 1475 11 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1475:11
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.1:
	.loc	1 1477 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1477:22
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
.Ltmp98:
	.loc	5 248 59                        ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1471:26 ] ]
	s_load_b128 s[8:11], s[0:1], 0x0
.Ltmp99:
	.loc	2 0 0 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:0 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:522:14 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ] ]
	v_mbcnt_lo_u32_b32 v12, -1, 0
.Ltmp100:
	.loc	1 1473 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1473:26
	v_and_b32_e32 v14, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp101:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	v_xor_b32_e32 v13, 8, v12
.Ltmp102:
	.loc	1 1477 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1477:22
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1479 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	v_dual_mov_b32 v3, 0 :: v_dual_add_nc_u32 v2, v2, v3
	.loc	1 1478 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1478:21
	v_mul_lo_u32 v4, v2, 24
	.loc	1 1479 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:40
	v_mul_lo_u32 v2, 0x1800, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 1478 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1478:21
	v_sub_nc_u32_e32 v6, v1, v4
	.loc	1 1479 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	v_lshlrev_b64_e32 v[4:5], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1479 62 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:62
	v_lshlrev_b32_e32 v2, 8, v6
	.loc	1 1479 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s8, v4
	.loc	1 1480 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1480:22
	v_lshlrev_b32_e32 v4, 5, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1479 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:48
	v_lshlrev_b64_e32 v[6:7], 2, v[2:3]
	.loc	1 1479 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:25
	v_add_co_ci_u32_e64 v2, null, s9, v5, vcc_lo
	.loc	1 1479 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1479:48
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v7, vcc_lo
	.loc	1 1480 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1480:22
	v_add_co_u32 v8, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[4:7], v[8:9], off
	global_load_b128 v[8:11], v[8:9], off offset:16
.Ltmp103:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1485:9 ]
	s_wait_loadcnt 0x1
	v_max_num_f32_e64 v0, |v5|, |v5|
	v_max_num_f32_e64 v2, |v4|, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v2, v0
.Ltmp104:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	v_xor_b32_e32 v2, 16, v12
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v12, v2, vcc_lo
.Ltmp105:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v13, v12, v13 :: v_dual_lshlrev_b32 v2, 2, v2
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	v_lshlrev_b32_e32 v13, 2, v13
.Ltmp106:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1487:9 ]
	v_max3_num_f32 v0, v0, |v6|, |v7|
.Ltmp107:
	.loc	3 454 44 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1489:9 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v0, v0, |v8|, |v9|
.Ltmp108:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1491:9 ]
	v_max3_num_f32 v0, v0, |v10|, |v11|
.Ltmp109:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:18 ] ]
	ds_bpermute_b32 v2, v2, v0
.Ltmp110:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1492:9 ]
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
.Ltmp111:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:18 ] ]
	ds_bpermute_b32 v2, v13, v0
.Ltmp112:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	v_xor_b32_e32 v13, 4, v12
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
.Ltmp113:
	.loc	3 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1493:9 ]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_lshlrev_b32 v13, 2, v13
	v_max_num_f32_e32 v0, v0, v2
.Ltmp114:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:18 ] ]
	ds_bpermute_b32 v2, v13, v0
.Ltmp115:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	v_xor_b32_e32 v13, 2, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v12, v13, vcc_lo
.Ltmp116:
	.loc	3 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1494:9 ]
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v2, v2, v2 :: v_dual_lshlrev_b32 v13, 2, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
.Ltmp117:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:18 ] ]
	ds_bpermute_b32 v2, v13, v0
.Ltmp118:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	v_xor_b32_e32 v13, 1, v12
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	v_lshlrev_b32_e32 v12, 2, v12
.Ltmp119:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1495:9 ]
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	v_max_num_f32_e32 v0, v0, v2
.Ltmp120:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:18 ] ]
	ds_bpermute_b32 v2, v12, v0
.Ltmp121:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1496:9 ]
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v2, v2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v2
.Ltmp122:
	.loc	1 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
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
	.loc	1 1480 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1480:30
	v_lshlrev_b32_e32 v15, 3, v14
	.loc	1 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v2, v2, v12, v13
	v_cmp_neq_f32_e32 vcc_lo, 0, v0
	.loc	1 1499 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:10
	v_mov_b16_e32 v13.l, v3.l
	.loc	1 1500 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:10
	v_mov_b16_e32 v13.h, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	v_div_fixup_f32 v2, v2, 0x43e00000, v0
	.loc	1 1499 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:10
	v_mov_b16_e32 v12.l, v13.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	1 1500 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:10
	v_mov_b16_e32 v12.h, v13.h
	.loc	1 1497 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1497:21
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 1.0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_div_scale_f32 v30, null, v0, v0, v11
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_div_scale_f32 v16, null, v0, v0, v4
	.loc	1 1499 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_div_scale_f32 v18, null, v0, v0, v5
	.loc	1 1500 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_div_scale_f32 v20, null, v0, v0, v6
	.loc	1 1500 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_div_scale_f32 v22, null, v0, v0, v7
	.loc	1 1503 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_rcp_f32_e32 v38, v30
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_rcp_f32_e32 v31, v16
	.loc	1 1499 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_rcp_f32_e32 v32, v18
	.loc	1 1502 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_div_scale_f32 v24, null, v0, v0, v8
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_rcp_f32_e32 v33, v20
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_div_scale_f32 v26, null, v0, v0, v9
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_rcp_f32_e32 v34, v22
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_div_scale_f32 v39, s8, v11, v0, v11
	.loc	1 1503 45 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_div_scale_f32 v28, null, v0, v0, v10
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fma_f32 v47, -v30, v38, 1.0
	.loc	1 1502 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_rcp_f32_e32 v35, v24
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_fma_f32 v40, -v16, v31, 1.0
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_rcp_f32_e32 v36, v26
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fma_f32 v41, -v18, v32, 1.0
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fmac_f32_e32 v38, v47, v38
	.loc	1 1503 45 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_rcp_f32_e32 v37, v28
	.loc	1 1500 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fma_f32 v42, -v20, v33, 1.0
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_div_scale_f32 v17, vcc_lo, v4, v0, v4
	v_fmac_f32_e32 v31, v40, v31
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_mul_f32_e32 v47, v39, v38
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_fma_f32 v43, -v22, v34, 1.0
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_div_scale_f32 v19, s2, v5, v0, v5
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fma_f32 v44, -v24, v35, 1.0
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_div_scale_f32 v21, s3, v6, v0, v6
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fma_f32 v55, -v30, v47, v39
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_dual_fmac_f32 v32, v41, v32 :: v_dual_fmac_f32 v33, v42, v33
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fma_f32 v45, -v26, v36, 1.0
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_mul_f32_e32 v40, v17, v31
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_div_scale_f32 v23, s4, v7, v0, v7
	v_fmac_f32_e32 v34, v43, v34
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fma_f32 v46, -v28, v37, 1.0
	.loc	1 1503 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fmac_f32_e32 v47, v55, v38
	.loc	1 1499 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_mul_f32_e32 v41, v19, v32
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_div_scale_f32 v25, s5, v8, v0, v8
	v_dual_fmac_f32 v35, v44, v35 :: v_dual_mul_f32 v42, v21, v33
	.loc	1 1502 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_div_scale_f32 v27, s6, v9, v0, v9
	v_fmac_f32_e32 v36, v45, v36
	.loc	1 1499 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_fma_f32 v48, -v16, v40, v17
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_mul_f32_e32 v43, v23, v34
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_div_scale_f32 v29, s7, v10, v0, v10
	v_fmac_f32_e32 v37, v46, v37
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fma_f32 v49, -v18, v41, v19
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_mul_f32_e32 v44, v25, v35
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fma_f32 v50, -v20, v42, v21
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_dual_mul_f32 v45, v27, v36 :: v_dual_fmac_f32 v40, v48, v31
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_fma_f32 v51, -v22, v43, v23
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_mul_f32_e32 v46, v29, v37
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fmac_f32_e32 v41, v49, v32
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fma_f32 v52, -v24, v44, v25
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fmac_f32_e32 v42, v50, v33
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fma_f32 v53, -v26, v45, v27
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_fma_f32 v16, -v16, v40, v17
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_dual_fmac_f32 v43, v51, v34 :: v_dual_lshlrev_b32 v2, 8, v1
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fma_f32 v54, -v28, v46, v29
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_fma_f32 v17, -v18, v41, v19
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fmac_f32_e32 v44, v52, v35
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_fma_f32 v18, -v20, v42, v21
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fmac_f32_e32 v45, v53, v36
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v31, v40
	.loc	1 1499 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	s_mov_b32 vcc_lo, s2
	.loc	1 1500 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_fma_f32 v19, -v22, v43, v23
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fmac_f32_e32 v46, v54, v37
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v32, v41
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	s_mov_b32 vcc_lo, s3
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_fma_f32 v20, -v24, v44, v25
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v33, v42
	.loc	1 1500 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	s_mov_b32 vcc_lo, s4
	.loc	1 1502 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_fma_f32 v21, -v26, v45, v27
	.loc	1 1499 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:45
	v_div_fixup_f32 v4, v16, v0, v4
	.loc	1 1500 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v16, v19, v34, v43
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	s_mov_b32 vcc_lo, s5
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_fma_f32 v22, -v28, v46, v29
	.loc	1 1499 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:53
	v_div_fixup_f32 v5, v17, v0, v5
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v20, v35, v44
	.loc	1 1502 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	s_mov_b32 vcc_lo, s6
	.loc	1 1503 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_fma_f32 v23, -v30, v47, v39
	.loc	1 1500 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:45
	v_div_fixup_f32 v6, v18, v0, v6
	.loc	1 1502 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v21, v36, v45
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	s_mov_b32 vcc_lo, s7
	.loc	1 1504 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1504:31
	v_add_co_u32 v2, s2, s10, v2
	.loc	1 1503 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v22, v37, v46
	.loc	1 1503 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	s_mov_b32 vcc_lo, s8
	.loc	1 1500 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:53
	v_div_fixup_f32 v7, v16, v0, v7
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v20, v23, v38, v47
	.loc	1 1502 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:45
	v_div_fixup_f32 v8, v17, v0, v8
	.loc	1 1502 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:53
	v_div_fixup_f32 v9, v18, v0, v9
	.loc	1 1503 45 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:45
	v_div_fixup_f32 v10, v19, v0, v10
	.loc	1 1504 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1504:31
	v_add_co_ci_u32_e64 v16, null, s11, 0, s2
	.loc	1 1503 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:53
	v_div_fixup_f32 v11, v20, v0, v11
	.loc	1 1499 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1499:10
	v_cvt_pk_fp8_f32 v12.l, v4, v5
	.loc	1 1505 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1505:9
	v_add_co_u32 v4, vcc_lo, v2, v15
	.loc	1 1500 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1500:10
	v_cvt_pk_fp8_f32 v12.h, v6, v7
	.loc	1 1502 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1502:10
	v_cvt_pk_fp8_f32 v13.l, v8, v9
	.loc	1 1503 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1503:10
	v_cvt_pk_fp8_f32 v13.h, v10, v11
	.loc	1 1505 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1505:9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v16, vcc_lo
	.loc	1 1508 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1508:14
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	.loc	1 1507 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1507:30
	global_store_b64 v[4:5], v[12:13], off
	.loc	1 1508 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1508:14
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.2:
.Ltmp123:
	.loc	5 248 59                        ; /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:248:59 @[ /opt/rocm/core/include/hip/amd_detail/amd_hip_runtime.h:299:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1471:26 ] ]
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
	.loc	1 1509 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1509:46
	global_store_b32 v[1:2], v0, off
.LBB2_3:
	.loc	1 1510 1                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1510:1
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
	.loc	1 1569 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1569:17
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 1569 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1569:23
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
	.loc	1 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 1574 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1574:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB3_93
; %bb.2:
	.loc	1 1576 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1576:35
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 1577 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1577:16
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB3_93
; %bb.3:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 1580 15 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1580:15
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB3_93
; %bb.4:
.Ltmp126:
	.loc	1 957 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:957:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_lshrrev_b32_e32 v157, 4, v0
	v_lshrrev_b32_e32 v158, 3, v0
	s_mov_b32 s19, 0
	.loc	1 974 9                         ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:974:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_6
; %bb.5:
	.loc	1 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v1, 4, v0
	.loc	1 976 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:976:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v2, v0, 7, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 975 33                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v1, 6, v1
	.loc	1 977 24                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:977:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_gt_i32_e32 vcc_lo, s7, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	.loc	1 975 25                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_u64_u32 v[3:4], null, s3, 6, v[1:2]
	.loc	1 975 45 is_stmt 0              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v1, 3, v0
	.loc	1 978 31 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v4, v2, 24
	s_and_b32 s19, vcc_lo, exec_lo
	.loc	1 975 39                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:975:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v5, v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 978 55                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:978:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_lshl_u32 v3, v5, v4, 8
	.loc	1 1150 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1150:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b32_e32 v4, 0
.Ltmp127:
.LBB3_6:
	.loc	1 0 38 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:38
	s_or_b32 exec_lo, exec_lo, s5
	.loc	1 1569 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1569:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
.Ltmp128:
	.loc	1 954 26                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:954:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v9, 31, v0
	v_mov_b32_e32 v7, -1
	v_bfrev_b32_e32 v8, -2
	.loc	1 991 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:991:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v9
	s_cbranch_execz .LBB3_10
; %bb.7:
	.loc	1 992 31                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:992:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, s4, v9
	v_bfrev_b32_e32 v8, -2
	v_mov_b32_e32 v7, -1
	.loc	1 993 16                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:993:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v6
	s_cbranch_execz .LBB3_9
; %bb.8:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v7, 31, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_kmcnt 0x0
	v_add_co_u32 v6, vcc_lo, s0, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	.loc	1 994 27 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:994:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b32 v7, v[6:7], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v8, v7
.LBB3_9:
	.loc	1 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp129:
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_lshrrev_b32_e32 v13, 1, v0
.Ltmp130:
	.loc	1 956 25 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:956:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v160, 4, v9
	v_dual_mov_b32 v154, 0xff800000 :: v_dual_and_b32 v161, 15, v0
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp131:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v1, 16, v10
.Ltmp132:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v12, 8, v10
	v_lshl_add_u32 v162, v9, 3, 0
	v_cndmask_b32_e64 v9, 0, v4, s19
.Ltmp133:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v163, 5, v0
.Ltmp134:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
.Ltmp135:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v15, 1, v10
.Ltmp136:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_cvt_f32_u32 s4, s17
	s_add_co_i32 s6, s17, 0x1ff
	s_mov_b32 s5, 0
.Ltmp137:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v10, v1 :: v_dual_lshlrev_b32 v14, 3, v0
.Ltmp138:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
.Ltmp139:
	.loc	1 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s20, s4
	s_and_b32 s6, s6, 0xffff
	v_dual_mov_b32 v156, 1.0 :: v_dual_lshlrev_b32 v159, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s6
	v_and_b32_e32 v166, 8, v13
	v_lshlrev_b32_e32 v164, 3, v160
.Ltmp140:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v6, v159, v7
.Ltmp141:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v11, v159, v8
.Ltmp142:
	.loc	1 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s20, s6, s20
	s_delay_alu instid0(SALU_CYCLE_3)
	s_trunc_f32 s20, s20
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[145:146], null, v2, 24, v[5:6]
.Ltmp143:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v10, v12, vcc_lo
	v_mov_b32_e32 v1, 0
.Ltmp144:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max_i32_e32 v4, v7, v6
.Ltmp145:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v7, 4, v10
	v_cndmask_b32_e64 v12, 0, v3, s19
.Ltmp146:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v0, 2, v5
.Ltmp147:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v8, v11
.Ltmp148:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v11, 2, v10
.Ltmp149:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_ashrrev_i32_e32 v3, 31, v2
.Ltmp150:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v6, v0, v4
.Ltmp151:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v0, v0, v5
	v_and_b32_e32 v13, 0xf8, v14
.Ltmp152:
	.loc	2 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v10, v7, vcc_lo
.Ltmp153:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_lshlrev_b64_e32 v[129:130], 2, v[2:3]
	v_mov_b32_e32 v3, v1
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v165, 7, v161
.Ltmp154:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v7, 2, v7
.Ltmp155:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, v10, v11, vcc_lo
.Ltmp156:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_lshlrev_b32_e32 v133, 4, v161
	v_dual_mov_b32 v8, v1 :: v_dual_lshlrev_b32 v167, 4, v163
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v146, v1
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v134, v10, v15, vcc_lo
.Ltmp157:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v136, vcc_lo, v12, v164
.Ltmp158:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v41, v4, v6
.Ltmp159:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v5, v0
	v_mov_b32_e32 v6, v1
.Ltmp160:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_or_b32 v135, v163, 8, v13
	v_mov_b32_e32 v4, v1
.Ltmp161:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v42, v7, v41
.Ltmp162:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v43, v7, v0
	v_mov_b32_e32 v5, v1
	v_dual_mov_b32 v7, v1 :: v_dual_add_nc_u32 v168, 0, v135
.Ltmp163:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v65, 2, v44
	v_lshlrev_b64_e32 v[131:132], 2, v[145:146]
.Ltmp164:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v9, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v146, vcc_lo, s0, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, s1, v130, vcc_lo
.Ltmp165:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v129, 2, v134
.Ltmp166:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_xor_b32 s0, s20, 0x80000000
.Ltmp167:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v135, vcc_lo, s8, v136
.Ltmp168:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s6, s0, s4
	s_cvt_u32_f32 s0, s20
.Ltmp169:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, s9, v137, vcc_lo
.Ltmp170:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v138, v41, v42
.Ltmp171:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v43
	v_mov_b32_e32 v48, v8
.Ltmp172:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s6, 31
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v56, v8
.Ltmp173:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v139, v65, v138
.Ltmp174:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v140, v65, v0
.Ltmp175:
	.loc	1 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s6, s4
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v72, v8
	s_add_co_ci_u32 s6, s0, 0
	s_lshl_b32 s4, s3, 8
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v88, v8
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[4:5]
	s_add_nc_u64 s[20:21], s[12:13], s[4:5]
	s_mul_i32 s4, s7, 0x1800
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v96, v8
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v104, v8
.Ltmp176:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v138, v139
.Ltmp177:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v140
.Ltmp178:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v148, vcc_lo, v135, 48
.Ltmp179:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_and_b32 s22, s6, 0xffff
.Ltmp180:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v134, v129, v130
.Ltmp181:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v129, v129, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[8:9], s[4:5]
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v112, v8
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v120, v8
.Ltmp182:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, 0, v136, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v152, vcc_lo, s6, v131
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v128, v8
	v_add_co_u32 v150, s0, s0, v133
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
.Ltmp183:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v130, v134
.Ltmp184:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v129
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_mov_b32_e32 v121, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s7, v132, vcc_lo
.Ltmp185:
	.loc	1 1009 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1009:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1010 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1010:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_readfirstlane_b32 s27, v0
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v132, v4
	v_dual_mov_b32 v134, v6 :: v_dual_mov_b32 v133, v5
	v_dual_mov_b32 v130, v2 :: v_dual_mov_b32 v131, v3
	v_mov_b32_e32 v129, v1
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mul_i32 s24, s18, s22
	s_lshl_b32 s4, s3, 1
.Ltmp186:
	.loc	1 1584 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1584:23
	s_add_co_i32 s25, s24, s22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
.Ltmp187:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl_b32 s11, s24, 6
	s_branch .LBB3_14
.LBB3_11:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_mov_b32_e32 v154, v2
.LBB3_12:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s30
.Ltmp188:
	.loc	4 701 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp189:
.LBB3_13:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	4 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
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
	.loc	1 1015 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1015:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl_b32 s29, s24, 6
	.loc	1 1016 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1016:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s29, s26
	s_cselect_b32 s28, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_13
; %bb.15:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_mov_b32_e32 v0, v157
	v_mov_b32_e32 v6, v158
	s_mov_b32 s0, 8
	s_branch .LBB3_17
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1032 49 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v7, 0xe0, v6
	.loc	1 1024 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 1032 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v7, v7, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1032 54 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1032:54 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v7, v0, 15, v7
	.loc	1 1024 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 8, v0
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	.loc	1 1024 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1024:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc1 .LBB3_19
.LBB3_17:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 9                           ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s11, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	.loc	1 1033 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1033:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s26, v7
	s_cbranch_execz .LBB3_16
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
	.loc	1 1037 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1037:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v7, v[150:151]
	.loc	1 1038 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1038:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB3_16
.LBB3_19:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, s29, v166
	v_dual_mov_b32 v7, v167 :: v_dual_mov_b32 v8, v163
	s_movk_i32 s0, 0xc000
	s_branch .LBB3_23
.LBB3_20:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_21:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB3_22:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1088 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, s0, v168
	.loc	1 1049 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 1088 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1088:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_store_b64 v0, v[2:3] offset:32768
	.loc	1 1049 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1049:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc1 .LBB3_39
.LBB3_23:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 9 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1058 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1058:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	.loc	1 1057 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1057:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_and_or_b32 v0, 0xf0, v7, v161
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1062 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v2, 7, v137
	.loc	1 1062 27 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1062:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB3_25
; %bb.24:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1065 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1065:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, s[20:21]
	.loc	1 1067 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1067:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1073 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1073:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v139.l, v5.h
	.loc	1 1066 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1066:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	1 1068 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	.loc	1 1068 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	.loc	1 1069 53 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1069:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	.loc	1 1070 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	.loc	1 1072 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	.loc	1 1068 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1068:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	1 1070 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1070:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or3_b32 v0, v0, v2, v3
	.loc	1 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	.loc	1 1074 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	.loc	1 1072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or3_b32 v0, v0, 0, 0
	.loc	1 1074 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1074:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB3_25:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB3_22
; %bb.26:                               ;   in Loop: Header=BB3_23 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	.loc	1 1075 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1075:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s26, v137
	s_cbranch_execz .LBB3_21
; %bb.27:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1077 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1077:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v4, s4, s20, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s21, 0, s4
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e32 v0.h, 0
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmpx_gt_i32_e64 s26, v137
	s_cbranch_execz .LBB3_29
; %bb.28:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1078 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB3_29:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 2, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_31
; %bb.30:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB3_31:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 3, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_33
; %bb.32:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	.loc	1 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_u8 v0, v[138:139], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB3_33:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 4, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_35
; %bb.34:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB3_35:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 5, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_37
; %bb.36:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	.loc	1 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v3, v0, v3
.LBB3_37:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1078 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1078:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v0, 6, v137
	.loc	1 1080 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1080:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_20
; %bb.38:                               ;   in Loop: Header=BB3_23 Depth=2
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v0, v[4:5]
	.loc	1 1085 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1085:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e32 v0.l, 0
	.loc	1 1081 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1081:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_d16_hi_u8 v0, v[4:5], off
	.loc	1 1084 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1084:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB3_20
.LBB3_39:                               ;   in Loop: Header=BB3_14 Depth=1
.Ltmp190:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp191:
	.loc	1 1096 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1096:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB3_12
; %bb.40:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b32_e32 v0, s16
	.loc	1 1111 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1111:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB3_42
; %bb.41:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 1112 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1112:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b32 v0, v[152:153], off
	.loc	1 1113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s16, v0
.LBB3_42:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 1098 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_or_b32 s0, s29, 16
	v_mov_b32_e32 v4, v162
	.loc	1 1098 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1098:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s26
	s_mov_b32 s31, 0
	s_cselect_b32 s33, -1, 0
	s_branch .LBB3_45
.LBB3_43:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v154, v2
	.loc	1 1189 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v154
	.loc	1 1209 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v172, null, v3, v3, v140
	.loc	1 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v174, null, v3, v3, v139
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v177, null, v3, v3, v137
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v176, null, v3, v3, v8
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v173, v172
	.loc	1 1251 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1251:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_add_u32 v189, s31, 12, v162
	v_exp_f32_e32 v143, v143
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v180, v177
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v179, v176
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v175, -v172, v173, 1.0
	.loc	1 1189 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_fmac_f32 v173, v175, v173 :: v_dual_mul_f32 v144, v156, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1226 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, null, v3, v3, v144
	v_div_scale_f32 v170, vcc_lo, v144, v3, v144
	v_rcp_f32_e32 v156, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v169, -v154, v156, 1.0
	v_fmac_f32_e32 v156, v169, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v169, v170, v156
	v_fma_f32 v171, -v154, v169, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v169, v171, v156
	v_fma_f32 v154, -v154, v169, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v154, v154, v156, v169
	v_div_fixup_f32 v6, v154, v3, v144
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, null, v3, v3, v141
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v134, v134, v6
	v_dual_mul_f32 v130, v130, v6 :: v_dual_fmac_f32 v5, v155, v143
	v_mul_f32_e32 v128, v128, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v143, null, v3, v3, v142
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v156, v154
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v126, v126, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v124, v124, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v122, v122, v6
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v154, v156, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v25, v25, v6
	v_dual_mul_f32 v129, v129, v6 :: v_dual_mul_f32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v143, v144, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v33, v33, v6 :: v_dual_fmac_f32 v156, v170, v156
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v170, s0, v141, v3, v141
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v127, v127, v6 :: v_dual_mul_f32 v118, v118, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v144, v155, v144
	v_div_scale_f32 v155, vcc_lo, v142, v3, v142
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v16, v16, v6 :: v_dual_mul_f32 v125, v125, v6
	v_dual_mul_f32 v116, v116, v6 :: v_dual_mul_f32 v123, v123, v6
	v_dual_mul_f32 v114, v114, v6 :: v_dual_mul_f32 v169, v155, v144
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v112, v112, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v110, v110, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v171, -v143, v169, v155
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v117, v117, v6 :: v_dual_mul_f32 v108, v108, v6
	v_dual_mul_f32 v115, v115, v6 :: v_dual_mul_f32 v106, v106, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v169, v171, v144
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v170, v156
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v104, v104, v6
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v102, v102, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v143, -v143, v169, v155
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v154, v171, v170
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v98, v98, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v155, v156
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v155, s1, v140, v3, v140
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v144, v174
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v143, v143, v3, v142
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v142, -v154, v171, v170
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v154, v155, v173
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v170, null, v3, v3, v7
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v96, v96, v6
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v156, v142, v156, v171
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v171, -v172, v154, v155
	s_mov_b32 vcc_lo, s1
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v175, v170
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v174, v144, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v19, v19, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v154, v171, v173
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v142.l, v1.l
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v156, v156, v3, v141
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	.loc	1 1242 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v172, v154, v155
	.loc	1 1245 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v178, -v170, v175, 1.0
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v141.l, v142.l
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v142.h, 0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v94, v94, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v155, v173, v154
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v155, null, v3, v3, v138
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v173, -v176, v179, 1.0
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v141.l, v143, v156
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v140, v154, v3, v140
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v154, -v177, v180, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v15, v15, v6
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_fmac_f32 v179, v173, v179 :: v_dual_mov_b32 v156, v3
	v_div_scale_f32 v173, s3, v8, v3, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v180, v154, v180
	v_div_scale_f32 v154, s4, v137, v3, v137
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v10, v10, v6
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v141.h, v142.h
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v101, v101, v6 :: v_dual_mul_f32 v92, v92, v6
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v182, v154, v180
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v175, v178, v175
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v169, v144
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v178, v155
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v99, v99, v6 :: v_dual_mul_f32 v90, v90, v6
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v88, v88, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v172, -v174, v171, v169
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v86, v86, v6
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v84, v84, v6
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v172, v144
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v172, s1, v7, v3, v7
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v181, -v155, v178, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v17, v17, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v82, v82, v6
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v174, v171, v169
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v174, v172, v175
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v178, v181, v178
	v_div_scale_f32 v181, s0, v138, v3, v138
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v80, v80, v6
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v173, v179
	.loc	1 1245 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v170, v174, v172
	.loc	1 1247 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v183, v181, v178
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v139, v144, v3, v139
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v176, v171, v173
	.loc	1 1245 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v174, v169, v175
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v177, v182, v154
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v78, v78, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v144, v179
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v155, v183, v181
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v170, v174, v172
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v182, v169, v180
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v76, v76, v6
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v183, v144, v178
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	.loc	1 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v176, v171, v173
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v177, v182, v154
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v155, v183, v181
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v7, v169, v3, v7
	.loc	1 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v170, v179, v171
	.loc	1 1247 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v74, v74, v6
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	.loc	1 1247 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1245 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v8, v154, v3, v8
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v155, v155, v178, v183
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v72, v72, v6
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v137, v144, v3, v137
	.loc	1 1244 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1244:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v138, v155, v3, v138
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[7:8], v189 offset:16384
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v70, v70, v6
	.loc	1 1246 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1246:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[137:138], v189 offset:16640
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[139:140], v189 offset:16896
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[143:144], v189 offset:17152
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[154:155], v189 offset:17408
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[169:170], v189 offset:17664
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[171:172], v189 offset:17920
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[173:174], v189 offset:18176
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[175:176], v189 offset:18432
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[177:178], v189 offset:18688
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[179:180], v189 offset:18944
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[181:182], v189 offset:19200
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[183:184], v189 offset:19456
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[185:186], v189 offset:19712
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[187:188], v189 offset:19968
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[189:190], v189 offset:20224
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v68, v68, v6
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
	.loc	1 1253 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1253:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
.LBB3_44:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v154, v2
	.loc	1 1116 40 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_add_co_i32 s31, s31, 1
	.loc	1 1116 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s31, 4
	.loc	1 1116 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1116:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc0 .LBB3_11
.LBB3_45:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_50 Depth 3
	.loc	1 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s31, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_48
; %bb.46:                               ;   in Loop: Header=BB3_45 Depth=2
	s_cmp_eq_u32 s31, 1
	s_mov_b32 s0, s33
	s_cbranch_scc1 .LBB3_48
; %bb.47:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 1118 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cmp_eq_u32 s31, 2
	.loc	1 1118 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1118:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s26, s0
	s_cselect_b32 s0, -1, 0
.LBB3_48:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
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
	.loc	1 1151 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1151:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	.loc	1 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	.loc	1 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	1 1154 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1154:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	.loc	1 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_addk_co_i32 s0, 0x1000
	.loc	1 1159 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1159:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1139 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	.loc	1 1155 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1155:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1139 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1139:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_cbranch_scc1 .LBB3_50
; %bb.51:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 1161 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1161:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl4_add_u32 s0, s31, s29
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	.loc	1 1162 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	.loc	1 1162 49 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1162:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s27
	s_cselect_b32 s34, -1, 0
	s_cmp_gt_i32 s1, s27
	s_cselect_b32 s1, -1, 0
	.loc	1 1164 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB3_53
; %bb.52:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 1164 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1164:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_load_b32 v170, v[146:147], off
.LBB3_53:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v164
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp192:
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_55
; %bb.54:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[5:6], null, 0x408, v2, s[22:23]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.Ltmp193:
.LBB3_55:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v5, 1, v2
.Ltmp194:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_i32_e64 s0, s26, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_57
; %bb.56:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[6:7], null, 0x408, v5, s[22:23]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.Ltmp195:
.LBB3_57:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp196:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s1, s26, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB3_59
; %bb.58:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v6, s[22:23]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.Ltmp197:
.LBB3_59:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp198:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s3, s26, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_61
; %bb.60:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v7, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.Ltmp199:
.LBB3_61:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp200:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s4, s26, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_63
; %bb.62:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v8, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.Ltmp201:
.LBB3_63:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp202:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s5, s26, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_65
; %bb.64:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v169, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.Ltmp203:
.LBB3_65:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp204:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s6, s26, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB3_67
; %bb.66:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[180:181], null, 0x408, v171, s[22:23]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.Ltmp205:
.LBB3_67:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
.Ltmp206:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_ge_i32_e64 s7, s26, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_69
; %bb.68:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[180:181], null, 0x408, v172, s[22:23]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1172:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.Ltmp207:
.LBB3_69:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
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
	.loc	1 1178 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
.Ltmp208:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s34, s9
	s_or_b32 s8, s34, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
.Ltmp209:
	.loc	1 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s19, s9
.Ltmp210:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v143, v170, v142, v140
.Ltmp211:
	.loc	1 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp212:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v175, v143, v173, v144
.Ltmp213:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1186:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v176, v159, v175
.Ltmp214:
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB3_71
; %bb.70:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB3_71:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB3_73
; %bb.72:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB3_73:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_75
; %bb.74:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB3_75:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB3_77
; %bb.76:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB3_77:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB3_79
; %bb.78:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB3_79:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	.loc	1 757 14 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB3_86
; %bb.80:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB3_87
.LBB3_81:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 14                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB3_83
.LBB3_82:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.Ltmp215:
.LBB3_83:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp216:
	.loc	3 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1187:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v154, v175, v176
.Ltmp217:
	.loc	1 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v7, v137
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v141, v137, v170
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v140, v138, v143
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
.Ltmp218:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v142, 0, v141
.Ltmp219:
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v170, v3
.Ltmp220:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
.Ltmp221:
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v137, v5, v174
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp222:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v6, v7, v8
.Ltmp223:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp224:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v144, v6, v137, v138
.Ltmp225:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v5, v3
.Ltmp226:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v169, v159, v144
.Ltmp227:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v5, v143, v3
.Ltmp228:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v6, v159, v5
.Ltmp229:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v156
.Ltmp230:
	.loc	1 1222 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1222:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB3_43
; %bb.84:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 1223 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
.Ltmp231:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB3_43
.Ltmp232:
.LBB3_85:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v154
	s_branch .LBB3_44
.LBB3_86:                               ;   in Loop: Header=BB3_45 Depth=2
.Ltmp233:
	.loc	1 760 23 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB3_81
.LBB3_87:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 760 23                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:760:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_mad_co_i64_i32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	.loc	1 763 19                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:763:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	.loc	1 757 14                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:757:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1198:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB3_82
	s_branch .LBB3_83
.Ltmp234:
.LBB3_88:
	.loc	1 1267 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1267:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB3_93
; %bb.89:
	.loc	1 1282 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1282:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_eq_u32_e32 vcc_lo, 0, v160
	s_and_b32 s1, vcc_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
; %bb.90:
	.loc	1 1286 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1286:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	.loc	1 1288 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1288:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1287 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1287:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	1 1288 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1288:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	.loc	1 1290 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1290:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b64 v[0:1], v[154:155], off
.LBB3_91:
	.loc	1 0 28 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:28
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 1293 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1293:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB3_93
; %bb.92:
	.loc	1 1296 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1296:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v6, v156, v130 :: v_dual_mul_f32 v9, v9, v156
	v_mul_f32_e32 v8, v122, v156
	v_mul_f32_e32 v98, v98, v156
	v_dual_mul_f32 v90, v90, v156 :: v_dual_mul_f32 v83, v83, v156
	v_dual_mul_f32 v82, v82, v156 :: v_dual_mul_f32 v75, v75, v156
	.loc	1 1297 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1297:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	1 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v74, v74, v156
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v67, v67, v156 :: v_dual_mul_f32 v66, v66, v156
	v_dual_mul_f32 v59, v59, v156 :: v_dual_mul_f32 v42, v42, v156
	v_mul_f32_e32 v35, v35, v156
	v_add_nc_u32_e32 v2, v0, v164
	v_mul_f32_e32 v5, v156, v129
	.loc	1 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[3:4], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v7, v121, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v113, v113, v156 :: v_dual_add_nc_u32 v0, 2, v2
	v_mul_f32_e32 v105, v105, v156
	v_mul_f32_e32 v97, v97, v156
	v_mul_f32_e32 v89, v89, v156
	v_mul_f32_e32 v81, v81, v156
	.loc	1 1304 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[137:138], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v73, v73, v156 :: v_dual_add_nc_u32 v0, 18, v2
	v_mul_f32_e32 v65, v65, v156
	v_mul_f32_e32 v57, v57, v156
	.loc	1 1298 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v3, vcc_lo, s14, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[139:140], 2, v[0:1]
	.loc	1 1304 32 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 34, v2
	.loc	1 1298 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s15, v4, vcc_lo
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v137, vcc_lo, v3, v137
	v_lshlrev_b64_e32 v[141:142], 2, v[0:1]
	.loc	1 1304 32 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 50, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, v4, v138, vcc_lo
	v_add_co_u32 v139, vcc_lo, v3, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[143:144], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x42, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, v4, v140, vcc_lo
	v_add_co_u32 v141, vcc_lo, v3, v141
	v_lshlrev_b64_e32 v[145:146], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x52, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, v4, v142, vcc_lo
	v_add_co_u32 v143, vcc_lo, v3, v143
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[147:148], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x62, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v144, null, v4, v144, vcc_lo
	v_add_co_u32 v145, vcc_lo, v3, v145
	v_lshlrev_b64_e32 v[149:150], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x72, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v146, null, v4, v146, vcc_lo
	v_add_co_u32 v147, vcc_lo, v3, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[151:152], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v49, v49, v156 :: v_dual_add_nc_u32 v0, 0x82, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v148, null, v4, v148, vcc_lo
	v_add_co_u32 v149, vcc_lo, v3, v149
	v_lshlrev_b64_e32 v[153:154], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v41, v41, v156 :: v_dual_add_nc_u32 v0, 0x92, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v150, null, v4, v150, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v33, v33, v156
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[157:158], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xa2, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v151, vcc_lo, v3, v151
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v25, v25, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v152, null, v4, v152, vcc_lo
	v_lshlrev_b64_e32 v[159:160], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xb2, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v153, vcc_lo, v3, v153
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v154, null, v4, v154, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[161:162], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xc2, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v157, vcc_lo, v3, v157
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b64 v[137:138], v[5:6], off
	global_store_b64 v[139:140], v[7:8], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[163:164], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v17, v17, v156 :: v_dual_add_nc_u32 v0, 0xd2, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v158, null, v4, v158, vcc_lo
	v_add_co_u32 v159, vcc_lo, v3, v159
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[165:166], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xe2, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v160, null, v4, v160, vcc_lo
	v_add_co_u32 v161, vcc_lo, v3, v161
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xf2, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v162, null, v4, v162, vcc_lo
	v_add_co_u32 v129, vcc_lo, v3, v163
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_add_nc_u32_e32 v0, 4, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, v4, v164, vcc_lo
	v_add_co_u32 v121, vcc_lo, v3, v165
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, v4, v166, vcc_lo
	v_add_co_u32 v5, vcc_lo, v3, v5
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v58, v58, v156 :: v_dual_mul_f32 v51, v51, v156
	v_dual_mul_f32 v50, v50, v156 :: v_dual_mul_f32 v43, v43, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_store_b64 v[145:146], v[97:98], off
	global_store_b64 v[147:148], v[89:90], off
	global_store_b64 v[149:150], v[81:82], off
	global_store_b64 v[151:152], v[73:74], off
	global_store_b64 v[153:154], v[65:66], off
	global_store_b64 v[157:158], v[57:58], off
	global_store_b64 v[159:160], v[49:50], off
	global_store_b64 v[161:162], v[41:42], off
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v34, v34, v156 :: v_dual_mul_f32 v19, v19, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[41:42], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 20, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v7, vcc_lo, v3, v7
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v26, v26, v156
	v_mul_f32_e32 v18, v18, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v4, v8, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v10, v10, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x3
	global_store_b64 v[129:130], v[33:34], off
	global_store_b64 v[121:122], v[25:26], off
	global_store_b64 v[5:6], v[17:18], off
	global_store_b64 v[7:8], v[9:10], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_co_u32 v8, vcc_lo, v3, v41
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v5, v156, v131 :: v_dual_add_nc_u32 v0, 36, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v4, v42, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v33, vcc_lo, v3, v6
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v6, v156, v132
	v_dual_mul_f32 v114, v114, v156 :: v_dual_mul_f32 v17, v123, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v4, v7, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v18, v124, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b64 v[8:9], v[5:6], off
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v9, v27, v156
	v_mul_f32_e32 v27, v109, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[25:26], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v7, v115, v156 :: v_dual_add_nc_u32 v0, 52, v2
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b64 v[141:142], v[113:114], off
	global_store_b64 v[33:34], v[17:18], off
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v33, v101, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[41:42], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v53, v53, v156 :: v_dual_add_nc_u32 v0, 0x44, v2
	v_mul_f32_e32 v45, v45, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v25, vcc_lo, v3, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[57:58], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x54, v2
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v106, v106, v156 :: v_dual_mul_f32 v49, v107, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v26, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x64, v2
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v8, v116, v156 :: v_dual_mul_f32 v37, v37, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v41, vcc_lo, v3, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[81:82], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x74, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v42, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v50, v108, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b64 v[25:26], v[7:8], off
	global_store_b64 v[143:144], v[105:106], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[89:90], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x84, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v57, vcc_lo, v3, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v4, v58, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[97:98], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x94, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v65, vcc_lo, v3, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	v_lshlrev_b64_e32 v[105:106], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xa4, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v81, vcc_lo, v3, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, v4, v82, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[113:114], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xb4, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v89, vcc_lo, v3, v89
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v90, null, v4, v90, vcc_lo
	v_lshlrev_b64_e32 v[121:122], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xc4, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v97, vcc_lo, v3, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v98, null, v4, v98, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[129:130], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xd4, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v105, vcc_lo, v3, v105
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v10, v99, v156
	v_mul_f32_e32 v7, v11, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[137:138], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v11, v100, v156 :: v_dual_add_nc_u32 v0, 0xe4, v2
	v_mul_f32_e32 v73, v91, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v106, null, v4, v106, vcc_lo
	v_add_co_u32 v113, vcc_lo, v3, v113
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v74, v92, v156
	v_mul_f32_e32 v84, v84, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v114, null, v4, v114, vcc_lo
	v_add_co_u32 v121, vcc_lo, v3, v121
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v76, v76, v156
	v_dual_mul_f32 v68, v68, v156 :: v_dual_mul_f32 v55, v55, v156
	v_dual_mul_f32 v60, v60, v156 :: v_dual_mul_f32 v47, v47, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, v4, v122, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v52, v52, v156 :: v_dual_mul_f32 v39, v39, v156
	v_dual_mul_f32 v44, v44, v156 :: v_dual_mul_f32 v31, v31, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_store_b64 v[57:58], v[10:11], off
	global_store_b64 v[65:66], v[73:74], off
	global_store_b64 v[81:82], v[83:84], off
	global_store_b64 v[89:90], v[75:76], off
	global_store_b64 v[97:98], v[67:68], off
	global_store_b64 v[105:106], v[59:60], off
	global_store_b64 v[113:114], v[51:52], off
	global_store_b64 v[121:122], v[43:44], off
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v60, v69, v156
	v_mul_f32_e32 v69, v61, v156
	v_mul_f32_e32 v61, v70, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xf4, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v129, vcc_lo, v3, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, v4, v130, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v15, v15, v156 :: v_dual_add_nc_u32 v0, 6, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v123, vcc_lo, v3, v137
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v124, null, v4, v138, vcc_lo
	v_add_co_u32 v5, vcc_lo, v3, v5
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v36, v36, v156
	v_mul_f32_e32 v8, v12, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 22, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v17, vcc_lo, v3, v17
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v10, v28, v156
	v_mul_f32_e32 v20, v20, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v18, vcc_lo
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x3
	global_store_b64 v[129:130], v[35:36], off
	global_store_b64 v[123:124], v[9:10], off
	global_store_b64 v[5:6], v[19:20], off
	global_store_b64 v[17:18], v[7:8], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 38, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v8, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 54, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v17, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v7, vcc_lo
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x46, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v25, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x56, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v19, vcc_lo, v3, v19
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b64 v[41:42], v[49:50], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v20, vcc_lo
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x66, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v41, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[43:44], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x76, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v49, vcc_lo, v3, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v4, v35, vcc_lo
	v_lshlrev_b64_e32 v[51:52], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x86, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v43, vcc_lo, v3, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, v4, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x96, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v51, vcc_lo, v3, v51
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, v4, v52, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xa6, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v67, vcc_lo, v3, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, v4, v59, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xb6, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v65, vcc_lo, v3, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	v_lshlrev_b64_e32 v[73:74], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xc6, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v75, vcc_lo, v3, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, v4, v59, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v5, v156, v133 :: v_dual_add_nc_u32 v0, 0xd6, v2
	v_mul_f32_e32 v6, v156, v134
	v_dual_mul_f32 v10, v125, v156 :: v_dual_mul_f32 v11, v126, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[81:82], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xe6, v2
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b64 v[8:9], v[5:6], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v73, vcc_lo, v3, v73
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v7, v117, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xf6, v2
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v8, v118, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, v4, v74, vcc_lo
	v_add_co_u32 v83, vcc_lo, v3, v58
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v28, v110, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b64 v[17:18], v[10:11], off
	global_store_b64 v[25:26], v[7:8], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, v4, v59, vcc_lo
	v_add_co_u32 v81, vcc_lo, v3, v81
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v34, v102, v156
	v_add_nc_u32_e32 v0, 8, v2
	v_mul_f32_e32 v12, v93, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, v4, v82, vcc_lo
	v_add_co_u32 v5, vcc_lo, v3, v5
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v7, v13, v156
	v_mul_f32_e32 v13, v94, v156
	v_dual_mul_f32 v35, v85, v156 :: v_dual_mul_f32 v36, v86, v156
	v_mul_f32_e32 v57, v77, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v25, vcc_lo, v3, v10
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v58, v78, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v11, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v70, v62, v156
	v_mul_f32_e32 v54, v54, v156
	v_mul_f32_e32 v46, v46, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_store_b64 v[41:42], v[33:34], off
	global_store_b64 v[49:50], v[12:13], off
	global_store_b64 v[43:44], v[35:36], off
	global_store_b64 v[51:52], v[57:58], off
	global_store_b64 v[67:68], v[60:61], off
	global_store_b64 v[65:66], v[69:70], off
	global_store_b64 v[75:76], v[53:54], off
	global_store_b64 v[73:74], v[45:46], off
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v38, v38, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v9, v29, v156 :: v_dual_add_nc_u32 v0, 24, v2
	v_dual_mul_f32 v10, v30, v156 :: v_dual_mul_f32 v17, v21, v156
	v_mul_f32_e32 v18, v22, v156
	v_mul_f32_e32 v8, v14, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x3
	global_store_b64 v[83:84], v[37:38], off
	global_store_b64 v[81:82], v[9:10], off
	global_store_b64 v[5:6], v[17:18], off
	global_store_b64 v[25:26], v[7:8], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 40, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v8, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 56, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v13, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v4, v7, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x48, v2
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b64 v[19:20], v[27:28], off
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v19, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x58, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v25, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x68, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v29, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x78, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v35, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, v4, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x88, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v41, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0x98, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v45, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, v4, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[49:50], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xa8, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v51, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[53:54], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xb8, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v49, vcc_lo, v3, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v4, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[57:58], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xc8, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v53, vcc_lo, v3, v53
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v5, v156, v135
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v54, null, v4, v54, vcc_lo
	v_lshlrev_b64_e32 v[59:60], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xd8, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v57, vcc_lo, v3, v57
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v6, v156, v136
	v_mul_f32_e32 v10, v127, v156
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[61:62], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v11, v128, v156 :: v_dual_add_nc_u32 v0, 0xe8, v2
	v_mul_f32_e32 v17, v119, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v4, v58, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	.loc	1 1304 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_nc_u32_e32 v0, 0xf8, v2
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v59, vcc_lo, v3, v59
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v18, v120, v156 :: v_dual_mul_f32 v21, v111, v156
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v22, v112, v156 :: v_dual_mul_f32 v27, v103, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, v4, v60, vcc_lo
	v_add_co_u32 v61, vcc_lo, v3, v61
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v28, v104, v156 :: v_dual_mul_f32 v33, v95, v156
	v_mul_f32_e32 v37, v87, v156
	v_mul_f32_e32 v43, v79, v156
	v_dual_mul_f32 v7, v71, v156 :: v_dual_mul_f32 v34, v96, v156
	v_mul_f32_e32 v38, v88, v156
	v_mul_f32_e32 v44, v80, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_store_b64 v[8:9], v[5:6], off
	global_store_b64 v[13:14], v[10:11], off
	global_store_b64 v[19:20], v[17:18], off
	global_store_b64 v[25:26], v[21:22], off
	global_store_b64 v[29:30], v[27:28], off
	global_store_b64 v[35:36], v[33:34], off
	global_store_b64 v[41:42], v[37:38], off
	global_store_b64 v[45:46], v[43:44], off
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v8, v72, v156
	v_mul_f32_e32 v12, v63, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, v4, v62, vcc_lo
	v_add_co_u32 v65, vcc_lo, v3, v65
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v13, v64, v156
	v_mul_f32_e32 v56, v56, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	v_add_co_u32 v0, vcc_lo, v3, v0
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v48, v48, v156
	v_mul_f32_e32 v40, v40, v156
	v_mul_f32_e32 v2, v23, v156
	.loc	1 1304 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v4, v1, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v32, v32, v156
	v_mul_f32_e32 v3, v24, v156
	v_mul_f32_e32 v16, v16, v156
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x7
	global_store_b64 v[51:52], v[7:8], off
	global_store_b64 v[49:50], v[12:13], off
	global_store_b64 v[53:54], v[55:56], off
	global_store_b64 v[57:58], v[47:48], off
	global_store_b64 v[59:60], v[39:40], off
	global_store_b64 v[61:62], v[31:32], off
	global_store_b64 v[65:66], v[2:3], off
	global_store_b64 v[0:1], v[15:16], off
.Ltmp235:
.LBB3_93:
	.loc	1 1588 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1588:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp236:
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
; codeLenInByte = 10640
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
	.loc	1 1648 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1648:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	1 1648 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1648:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB4_10
; %bb.1:
	.loc	1 1653 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1653:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	1 1656 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1656:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1655 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1655:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	1 1656 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1656:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB4_10
; %bb.2:
	.loc	1 1648 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1648:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	1 1659 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:5
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
	.loc	1 1662 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:22
	global_load_b32 v3, v[6:7], off
.Ltmp237:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp238:
	.loc	1 1659 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp239:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp240:
	.loc	1 1659 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:5
	s_cbranch_scc0 .LBB4_3
; %bb.4:
	.loc	1 1654 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1654:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	1 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
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
	.loc	1 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1676 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
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
	.loc	1 1664 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	1 1676 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	1 1675 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1675:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	1 1664 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	1 1676 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	1 1675 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1675:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	1 1676 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	1 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	s_or_b32 s1, vcc_lo, s1
	.loc	1 1676 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1676:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	1 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	1 1675 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1675:44
	global_store_b32 v[5:6], v11, off
	.loc	1 1664 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1664:5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB4_10
.LBB4_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_8 Depth 2
	.loc	1 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB4_8
.LBB4_7:                                ;   in Loop: Header=BB4_8 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	.loc	1 1673 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1673:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	1 1667 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1667:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	1 1673 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1673:24
	global_load_b32 v15, v[15:16], off
	.loc	1 1672 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1672:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	1 1667 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1667:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 1673 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1673:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	1 1667 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1667:9
	s_cbranch_scc1 .LBB4_5
.LBB4_8:                                ;   Parent Loop BB4_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 1671 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	1 1671 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:17
	s_mov_b32 s2, exec_lo
	.loc	1 1671 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	1 1671 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:17
	s_cbranch_execz .LBB4_7
; %bb.9:                                ;   in Loop: Header=BB4_8 Depth=2
	.loc	1 1671 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:41
	global_load_b32 v14, v[5:6], off
	.loc	1 1671 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp241:
	.loc	3 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	3 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB4_7
.Ltmp242:
.LBB4_10:
	.loc	1 1678 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1678:1
	s_endpgm
.Ltmp243:
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
	.loc	1 2115 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2115:17
	s_load_b128 s[12:15], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	1 2115 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2115:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s14, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_58
; %bb.1:
	.loc	1 2118 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2118:14
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB5_58
; %bb.2:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0
	s_load_b32 s22, s[0:1], 0x38
	.loc	1 2120 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2120:30
	s_lshl_b32 s13, ttmp9, 7
	.loc	1 2121 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:23
	s_mul_i32 s3, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2121 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:18
	s_cmp_ge_i32 s13, s3
	s_cbranch_scc1 .LBB5_58
; %bb.3:
.Ltmp244:
	.loc	1 1765 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1765:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshrrev_b32_e32 v5, 5, v0
	.loc	1 1767 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_and_b32_e32 v6, 15, v0
.Ltmp245:
	.loc	1 2115 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2115:17
	s_load_b256 s[4:11], s[0:1], 0x0
.Ltmp246:
	.loc	1 1768 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1768:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_bfe_u32 v10, v0, 4, 1
.Ltmp247:
	.loc	1 2115 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2115:17
	s_load_b64 s[18:19], s[0:1], 0x20
	v_dual_mov_b32 v8, -1 :: v_dual_lshlrev_b32 v7, 4, v5
.Ltmp248:
	.loc	1 1766 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1766:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_and_b32_e32 v11, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshlrev_b32_e32 v183, 3, v10
	v_bfrev_b32_e32 v9, -2
	.loc	1 1776 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v1, v7, v6
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1776 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_nc_u32_e32 v4, s13, v1
	.loc	1 1777 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1777:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mul_hi_i32 v1, 0x2aaaaaab, v4
	.loc	1 1779 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmp_le_i32_e64 s2, s3, v4
	v_cmp_gt_i32_e64 s0, s3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1777 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1777:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshrrev_b32_e32 v2, 31, v1
	v_add_nc_u32_e32 v142, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1778 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mul_lo_u32 v1, v142, 6
	v_sub_nc_u32_e32 v1, v4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1778 31 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[144:145], null, ttmp7, 6, v[1:2]
	v_mad_co_u64_u32 v[1:2], null, v142, 24, v[144:145]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 8, v1
	.loc	1 1792 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, v2, 0, s2
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, s12, s4, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s5, 0, s12
	s_mov_b32 s12, 0
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_co_u32 v2, vcc_lo, v2, v183
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	1 1797 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1797:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0xf
	global_load_b64 v[145:146], v[2:3], off
	global_load_b64 v[147:148], v[2:3], off offset:16
	global_load_b64 v[149:150], v[2:3], off offset:32
	global_load_b64 v[151:152], v[2:3], off offset:48
	global_load_b64 v[153:154], v[2:3], off offset:64
	global_load_b64 v[155:156], v[2:3], off offset:80
	global_load_b64 v[157:158], v[2:3], off offset:96
	global_load_b64 v[159:160], v[2:3], off offset:112
	global_load_b64 v[161:162], v[2:3], off offset:128
	global_load_b64 v[163:164], v[2:3], off offset:144
	global_load_b64 v[165:166], v[2:3], off offset:160
	global_load_b64 v[167:168], v[2:3], off offset:176
	global_load_b64 v[169:170], v[2:3], off offset:192
	global_load_b64 v[171:172], v[2:3], off offset:208
	global_load_b64 v[173:174], v[2:3], off offset:224
	global_load_b64 v[175:176], v[2:3], off offset:240
	.loc	1 1792 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b32_e32 v2, 0
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_gt_u32_e32 22, v11
	s_cbranch_execz .LBB5_7
; %bb.4:
	.loc	1 1805 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mul_hi_i32 s14, s13, 0x2aaaaaab
	v_bfrev_b32_e32 v9, -2
	s_lshr_b32 s16, s14, 31
	v_mov_b32_e32 v8, -1
	.loc	1 1805 37 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add3_u32 v3, s14, s16, v11
	.loc	1 1806 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1806:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s14, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v3
	s_cbranch_execz .LBB5_6
; %bb.5:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v4, 31, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s18, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s19, v4, vcc_lo
	.loc	1 1807 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1807:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	global_load_b32 v8, v[3:4], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
.LBB5_6:
	.loc	1 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
.LBB5_7:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
.Ltmp249:
	v_mbcnt_lo_u32_b32 v15, -1, 0
	v_lshlrev_b32_e32 v18, 3, v11
	v_lshl_add_u32 v186, v11, 4, 0
.Ltmp250:
	.loc	1 1772 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1772:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshl_add_u32 v16, v5, 11, 0
	v_lshlrev_b32_e32 v185, 3, v5
.Ltmp251:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_xor_b32_e32 v3, 16, v15
.Ltmp252:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_xor_b32_e32 v4, 8, v15
	v_and_or_b32 v20, v5, 4, v10
	v_and_or_b32 v184, v7, 48, v6
.Ltmp253:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_xor_b32_e32 v30, 2, v15
.Ltmp254:
	.loc	2 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
.Ltmp255:
	.loc	1 1827 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1827:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cndmask_b32_e64 v1, v1, 0, s2
	v_and_b32_e32 v17, 0x7f, v0
	v_bfe_u32 v12, v0, 5, 1
	v_dual_mov_b32 v216, 1.0 :: v_dual_and_b32 v19, 16, v0
.Ltmp256:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v15, v3, vcc_lo
.Ltmp257:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v4
	v_or_b32_e32 v22, 0x100, v0
	v_or_b32_e32 v23, 0x200, v0
	v_or_b32_e32 v24, 0x300, v0
.Ltmp258:
	.loc	1 1825 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_addk_co_i32 s13, 0x7f
.Ltmp259:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v15, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	v_mov_b32_e32 v5, v2
	v_ashrrev_i32_e32 v143, 31, v142
	v_mov_b32_e32 v4, v2
	s_delay_alu instid0(VALU_DEP_4)
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_lshlrev_b32_e32 v14, 2, v14
.Ltmp260:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v11, v3, v8
	v_mov_b32_e32 v6, v2
.Ltmp261:
	.loc	1 1825 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s3
	v_cmp_eq_u32_e64 s3, v10, v12
	v_lshrrev_b32_e32 v33, 5, v24
	v_dual_mov_b32 v219, 0xff800000 :: v_dual_lshlrev_b32 v200, 3, v20
.Ltmp262:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_xor_b32_e32 v31, 1, v15
	s_mov_b32 s14, ttmp7
.Ltmp263:
	.loc	1 1826 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mul_i32 s16, s15, 0x1800
	.loc	1 1825 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cselect_b32 s24, -1, 0
	s_ashr_i32 s15, ttmp7, 31
	.loc	1 1826 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s17, s12
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[14:15], 8
	s_add_nc_u64 s[20:21], s[4:5], s[16:17]
	.loc	1 1828 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[16:17], s[6:7], s[14:15]
	s_add_nc_u64 s[14:15], s[8:9], s[14:15]
	s_lshl_b32 s4, ttmp7, 1
	v_cmp_gt_u32_e64 s1, 64, v0
	v_lshl_add_u32 v187, v0, 1, 0
.Ltmp264:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v26, v8, v11
	v_mov_b32_e32 v8, v2
.Ltmp265:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v13, v3, v9
.Ltmp266:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_xor_b32_e32 v11, 4, v15
	v_mov_b32_e32 v3, v2
.Ltmp267:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v28, v14, v26
	v_xad_u32 v188, 0x120, v18, v16
	v_xad_u32 v189, 0x124, v18, v16
.Ltmp268:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_lshlrev_b32_e32 v21, 4, v0
	v_xad_u32 v190, 0x240, v18, v16
	v_xad_u32 v191, 0x244, v18, v16
	v_xad_u32 v192, 0x360, v18, v16
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v32, v15, v11 :: v_dual_mov_b32 v7, v2
.Ltmp269:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v30
	v_lshlrev_b32_e32 v25, 6, v0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp270:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_lshlrev_b32_e32 v32, 2, v32
	v_add_nc_u32_e32 v1, 0, v19
	v_lshrrev_b32_e32 v19, 5, v22
	v_and_or_b32 v22, 0x180, v22, v17
.Ltmp271:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x1
	v_min_i32_e32 v27, v9, v13
	v_xad_u32 v193, 0x364, v18, v16
	v_xad_u32 v194, 0x520, v18, v16
.Ltmp272:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v26, v26, v28
	v_lshrrev_b32_e32 v28, 5, v23
.Ltmp273:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v29, v14, v27
	v_and_or_b32 v23, 0x280, v23, v17
	v_and_or_b32 v17, 0x380, v24, v17
.Ltmp274:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v24, v15, v30 :: v_dual_mov_b32 v9, v2
	v_lshlrev_b64_e32 v[13:14], 2, v[142:143]
	v_add_nc_u32_e32 v143, v16, v18
	v_xad_u32 v195, 0x524, v18, v16
	v_xad_u32 v196, 0x640, v18, v16
	v_xad_u32 v197, 0x644, v18, v16
	v_xad_u32 v198, 0x760, v18, v16
	v_xad_u32 v199, 0x764, v18, v16
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_lshlrev_b32_e32 v16, 2, v24
.Ltmp275:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v31
	v_and_b32_e32 v25, 0x3000, v25
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v177, s13, s14, v18
	v_and_or_b32 v18, v19, 12, v10
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v15, v31, vcc_lo
	v_mov_b32_e32 v93, v9
.Ltmp276:
	.loc	3 1334 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v27, v27, v29
.Ltmp277:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v29, v32, v26
	v_dual_mov_b32 v88, v4 :: v_dual_mov_b32 v109, v9
	v_lshlrev_b32_e32 v19, 4, v22
.Ltmp278:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v32, v32, v27
	v_lshlrev_b32_e32 v22, 4, v23
	v_dual_mov_b32 v220, 0 :: v_dual_add_nc_u32 v23, v186, v25
	v_lshlrev_b32_e32 v210, 3, v18
.Ltmp279:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_lshlrev_b32_e32 v15, 2, v15
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v133, v9 :: v_dual_add_nc_u32 v208, 0x4c00, v23
	v_mov_b32_e32 v130, v6
	v_add_co_u32 v179, vcc_lo, s20, v11
	v_dual_mov_b32 v86, v2 :: v_dual_add_nc_u32 v201, 0, v21
	v_and_or_b32 v21, v28, 20, v10
	v_dual_mov_b32 v128, v4 :: v_dual_lshlrev_b32 v17, 4, v17
.Ltmp280:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v24, v26, v29
	v_and_or_b32 v10, v33, 28, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v180, null, s21, v12, vcc_lo
.Ltmp281:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v26, v27, v32
.Ltmp282:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v20, v16, v24
	v_add_co_u32 v181, vcc_lo, s18, v13
	v_dual_mov_b32 v125, v9 :: v_dual_mov_b32 v108, v8
.Ltmp283:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v16, v16, v26
	v_dual_mov_b32 v107, v7 :: v_dual_add_nc_u32 v202, 0x4000, v23
	v_dual_mov_b32 v105, v5 :: v_dual_add_nc_u32 v204, 0x4400, v23
	v_dual_mov_b32 v103, v3 :: v_dual_add_nc_u32 v206, 0x4800, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v182, null, s19, v14, vcc_lo
	v_dual_mov_b32 v126, v2 :: v_dual_add_nc_u32 v203, 0x4200, v23
	v_add_nc_u32_e32 v205, 0x4600, v23
	v_add_nc_u32_e32 v207, 0x4a00, v23
	v_add_nc_u32_e32 v209, 0x4e00, v23
	v_dual_mov_b32 v131, v7 :: v_dual_lshlrev_b32 v212, 3, v21
.Ltmp284:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v20, v24, v20
	v_dual_mov_b32 v129, v5 :: v_dual_add_nc_u32 v214, 0, v17
	v_add_nc_u32_e32 v215, 0, v22
.Ltmp285:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v16, v26, v16
.Ltmp286:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v24, v15, v20
	v_dual_mov_b32 v132, v8 :: v_dual_add_nc_u32 v211, 0, v19
	v_mov_b32_e32 v117, v9
.Ltmp287:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	ds_bpermute_b32 v15, v15, v16
	v_mov_b32_e32 v101, v9
	v_mov_b32_e32 v85, v9
	v_mov_b32_e32 v77, v9
	v_mov_b32_e32 v69, v9
	v_mov_b32_e32 v61, v9
	v_mov_b32_e32 v53, v9
	v_mov_b32_e32 v45, v9
	v_dual_mov_b32 v37, v9 :: v_dual_mov_b32 v92, v8
	v_dual_mov_b32 v91, v7 :: v_dual_mov_b32 v122, v6
	v_dual_mov_b32 v90, v6 :: v_dual_mov_b32 v89, v5
	v_dual_mov_b32 v120, v4 :: v_dual_mov_b32 v87, v3
	v_mov_b32_e32 v118, v2
.Ltmp288:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v11, v20, v24
	v_dual_mov_b32 v29, v9 :: v_dual_mov_b32 v124, v8
.Ltmp289:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v12, v16, v15
	v_mov_b32_e32 v21, v9
	v_dual_mov_b32 v123, v7 :: v_dual_mov_b32 v106, v6
	v_dual_mov_b32 v121, v5 :: v_dual_mov_b32 v104, v4
	v_dual_mov_b32 v119, v3 :: v_dual_mov_b32 v102, v2
	v_dual_mov_b32 v100, v8 :: v_dual_lshlrev_b32 v213, 3, v10
.Ltmp290:
	.loc	1 1822 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1822:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_readfirstlane_b32 s18, v11
	.loc	1 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_readfirstlane_b32 s19, v12
	v_dual_mov_b32 v127, v3 :: v_dual_mov_b32 v96, v4
	v_dual_mov_b32 v116, v8 :: v_dual_mov_b32 v115, v7
	v_mov_b32_e32 v84, v8
	v_dual_mov_b32 v114, v6 :: v_dual_mov_b32 v113, v5
	v_mov_b32_e32 v82, v6
	v_dual_mov_b32 v112, v4 :: v_dual_mov_b32 v111, v3
	v_mov_b32_e32 v80, v4
	v_dual_mov_b32 v110, v2 :: v_dual_mov_b32 v99, v7
	v_mov_b32_e32 v76, v8
	v_dual_mov_b32 v98, v6 :: v_dual_mov_b32 v97, v5
	v_dual_mov_b32 v74, v6 :: v_dual_mov_b32 v95, v3
	v_mov_b32_e32 v72, v4
	v_dual_mov_b32 v94, v2 :: v_dual_mov_b32 v83, v7
	v_dual_mov_b32 v68, v8 :: v_dual_mov_b32 v81, v5
	v_dual_mov_b32 v66, v6 :: v_dual_mov_b32 v79, v3
	v_mov_b32_e32 v64, v4
	v_dual_mov_b32 v78, v2 :: v_dual_mov_b32 v75, v7
	v_dual_mov_b32 v60, v8 :: v_dual_mov_b32 v73, v5
	v_dual_mov_b32 v58, v6 :: v_dual_mov_b32 v71, v3
	v_mov_b32_e32 v56, v4
	v_dual_mov_b32 v70, v2 :: v_dual_mov_b32 v67, v7
	v_dual_mov_b32 v52, v8 :: v_dual_mov_b32 v65, v5
	v_dual_mov_b32 v50, v6 :: v_dual_mov_b32 v63, v3
	v_mov_b32_e32 v48, v4
	v_dual_mov_b32 v62, v2 :: v_dual_mov_b32 v59, v7
	v_dual_mov_b32 v44, v8 :: v_dual_mov_b32 v57, v5
	v_dual_mov_b32 v42, v6 :: v_dual_mov_b32 v55, v3
	v_mov_b32_e32 v40, v4
	v_dual_mov_b32 v54, v2 :: v_dual_mov_b32 v51, v7
	v_dual_mov_b32 v36, v8 :: v_dual_mov_b32 v49, v5
	v_dual_mov_b32 v34, v6 :: v_dual_mov_b32 v47, v3
	v_mov_b32_e32 v32, v4
	v_dual_mov_b32 v46, v2 :: v_dual_mov_b32 v43, v7
	v_dual_mov_b32 v28, v8 :: v_dual_mov_b32 v41, v5
	v_dual_mov_b32 v26, v6 :: v_dual_mov_b32 v39, v3
	v_mov_b32_e32 v24, v4
	v_dual_mov_b32 v38, v2 :: v_dual_mov_b32 v35, v7
	v_dual_mov_b32 v20, v8 :: v_dual_mov_b32 v33, v5
	v_dual_mov_b32 v18, v6 :: v_dual_mov_b32 v31, v3
	v_mov_b32_e32 v16, v4
	v_dual_mov_b32 v30, v2 :: v_dual_mov_b32 v27, v7
	v_mov_b32_e32 v25, v5
	v_dual_mov_b32 v23, v3 :: v_dual_mov_b32 v22, v2
	v_mov_b32_e32 v19, v7
	v_mov_b32_e32 v17, v5
	v_dual_mov_b32 v15, v3 :: v_dual_mov_b32 v14, v2
	v_mov_b32_e32 v13, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v178, null, s15, 0, s13
	v_dual_mov_b32 v12, v8 :: v_dual_mov_b32 v11, v7
	v_dual_mov_b32 v10, v6 :: v_dual_mov_b32 v9, v5
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v2
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s23, 0x76543210
	.loc	1 1828 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[6:7], s[4:5]
	s_add_nc_u64 s[8:9], s[8:9], s[4:5]
	s_mov_b32 s5, 0
	s_branch .LBB5_10
.LBB5_8:                                ;   in Loop: Header=BB5_10 Depth=1
.Ltmp291:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2054:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2054:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v219, v3
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2054:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp292:
.LBB5_9:                                ;   in Loop: Header=BB5_10 Depth=1
	.loc	4 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s5, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 0x3fffffff
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s20, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_56
.LBB5_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_47 Depth 2
	.loc	1 1829 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s21, s5, 6
	.loc	1 1830 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1830:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s21, s18
	s_cselect_b32 s20, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_9
; %bb.11:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v3, s21, v184
                                        ; implicit-def: $vgpr134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[138:139], null, 0x408, v3, s[16:17]
	v_cmp_ge_i32_e32 vcc_lo, s18, v3
	.loc	1 1843 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s13, exec_lo, s4
	s_cbranch_execz .LBB5_13
; %bb.12:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1850 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_co_u32 v3, s4, v138, v200
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s4
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
	v_add_co_u32 v3, s4, v138, v210
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s4
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v201, v[134:137]
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
.LBB5_13:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s13
	s_cbranch_execz .LBB5_15
; %bb.14:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v137, 0
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v136, v137
	v_dual_mov_b32 v135, v137 :: v_dual_mov_b32 v134, v137
	ds_store_b128 v201, v[2:5]
.LBB5_15:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v211, v[134:137]
                                        ; implicit-def: $vgpr134
	.loc	1 1843 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB5_17
; %bb.16:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_co_u32 v3, vcc_lo, v138, v212
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
	v_add_co_u32 v3, vcc_lo, v138, v213
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v215, v[134:137]
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
.LBB5_17:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB5_19
; %bb.18:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v4, v2
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v134, v137
	v_dual_mov_b32 v136, v137 :: v_dual_mov_b32 v135, v137
	ds_store_b128 v215, v[2:5]
.LBB5_19:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v214, v[134:137]
	v_dual_mov_b32 v134, 0 :: v_dual_add_nc_u32 v5, s21, v185
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v3, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_21
; %bb.20:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v5, v[177:178]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_21:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v136, 0x8000, v143
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v136, v3, v4 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_gt_i32_e64 s18, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_23
; %bb.22:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1862 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v3, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[177:178]
	global_load_b64 v[134:135], v[3:4], off
.LBB5_23:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v138, 2, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v188, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b32 v189, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_25
; %bb.24:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v138, v[177:178]
	global_load_b64 v[136:137], v[134:135], off
.LBB5_25:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v134, 3, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v190, v136 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b32 v191, v137 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v134
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_27
; %bb.26:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v134, v[177:178]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_27:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v138, 4, v5
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v192, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b32 v193, v4 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_29
; %bb.28:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v138, v[177:178]
	global_load_b64 v[136:137], v[3:4], off
.LBB5_29:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v3, 5, v5
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_nc_u32_e32 v4, 0x8400, v143
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v4, v136, v137 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v3
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_31
; %bb.30:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[177:178]
	global_load_b64 v[134:135], v[3:4], off
.LBB5_31:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v138, 6, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v194, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b32 v195, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_33
; %bb.32:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v138, v[177:178]
	global_load_b64 v[136:137], v[134:135], off
.LBB5_33:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v5, 7, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v196, v136 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b32 v197, v137 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_ge_i32_e64 s18, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_execz .LBB5_35
; %bb.34:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v5, v[177:178]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_35:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1871 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v198, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b32 v199, v4 offset:32768
.Ltmp293:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp294:
	.loc	1 1888 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1888:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_37
; %bb.36:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1904 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s15, s12
	s_mov_b32 s13, s12
	s_mov_b32 s14, s12
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v137, s15 :: v_dual_mov_b32 v136, s14
	v_dual_mov_b32 v135, s13 :: v_dual_mov_b32 v134, s12
	ds_store_b128 v202, v[134:137]
	ds_store_b128 v203, v[134:137]
	ds_store_b128 v204, v[134:137]
	ds_store_b128 v205, v[134:137]
	ds_store_b128 v206, v[134:137]
	ds_store_b128 v207, v[134:137]
	ds_store_b128 v208, v[134:137]
	ds_store_b128 v209, v[134:137]
.LBB5_37:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1910 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB5_41
; %bb.38:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1911 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1911:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v4, s21, v0
	v_mov_b32_e32 v3, 0
	.loc	1 1913 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1913:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s18, v4
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1918 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v4, s[6:7]
	.loc	1 1920 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, s[8:9]
	.loc	1 1918 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	global_load_d16_b16 v3, v[134:135], off offset:1024
	.loc	1 1920 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	global_load_d16_hi_b16 v3, v[4:5], off offset:1024
	.loc	1 1918 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v4.h, 8, v3.l
	.loc	1 1920 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshrrev_b16 v4.l, 8, v3.h
	.loc	1 1920 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_and_b16 v5.h, 0xff, v3.l
	v_and_b16 v5.l, 0xff, v3.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1921 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1921:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_pk_lshlrev_b16 v3, 8, v4 op_sel_hi:[0,1]
	.loc	1 1920 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v3, v3, v5
.LBB5_40:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	.loc	1 1923 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b16_d16_hi v187, v3 offset:49152
	.loc	1 1924 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1924:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_store_b16 v187, v3 offset:49280
.LBB5_41:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp295:
	.loc	4 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp296:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b32_e32 v217, s22
.Ltmp297:
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	s_barrier_wait -1
	.loc	4 703 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp298:
	.loc	1 1936 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1936:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB5_43
; %bb.42:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1937 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	global_load_b32 v3, v[179:180], off
	.loc	1 1938 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v217, s22, v3
.LBB5_43:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1834 35 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1834:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_or_b32 s4, s21, 63
	v_mov_b32_e32 v218, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s19
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s13, s24, s4
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s14, s2, s13
	.loc	1 1940 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s14
	s_cbranch_execz .LBB5_45
; %bb.44:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1940 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	global_load_b32 v218, v[181:182], off
.LBB5_45:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1930 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_or_b32 s4, s21, 16
	s_mov_b32 s14, 0
	.loc	1 1930 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s18
	s_cselect_b32 s15, -1, 0
	s_branch .LBB5_47
.LBB5_46:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_sub_f32 v222, v219, v3 :: v_dual_add_f32 v5, v5, v221
	.loc	1 1993 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1993:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v219
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v222, 0x3fb8aa3b, v222
	v_exp_f32_e32 v222, v222
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v219, 0, v222, vcc_lo
	.loc	1 2017 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mul_f32_e32 v216, v216, v219
	.loc	1 2008 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2008:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_fmac_f32_e32 v5, v220, v219
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2017 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v219, null, v4, v4, v216
	v_rcp_f32_e32 v220, v219
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v221, -v219, v220, 1.0
	v_fmac_f32_e32 v220, v221, v220
	v_div_scale_f32 v221, vcc_lo, v216, v4, v216
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v222, v221, v220
	v_fma_f32 v223, -v219, v222, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v222, v223, v220
	v_fma_f32 v219, -v219, v222, v221
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v219, v219, v220, v222
	v_div_fixup_f32 v216, v219, v4, v216
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 2020 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2020:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mul_f32 v93, v93, v216 :: v_dual_mul_f32 v92, v92, v216
	v_dual_mul_f32 v91, v91, v216 :: v_dual_mul_f32 v90, v90, v216
	v_dual_mul_f32 v89, v89, v216 :: v_dual_mul_f32 v88, v88, v216
	v_dual_mul_f32 v87, v87, v216 :: v_dual_mul_f32 v86, v86, v216
	v_dual_mul_f32 v125, v125, v216 :: v_dual_mul_f32 v124, v124, v216
	v_dual_mul_f32 v123, v123, v216 :: v_dual_mul_f32 v122, v122, v216
	v_dual_mul_f32 v121, v121, v216 :: v_dual_mul_f32 v120, v120, v216
	v_dual_mul_f32 v119, v119, v216 :: v_dual_mul_f32 v118, v118, v216
	v_dual_mul_f32 v109, v109, v216 :: v_dual_mul_f32 v108, v108, v216
	v_dual_mul_f32 v107, v107, v216 :: v_dual_mul_f32 v106, v106, v216
	v_dual_mul_f32 v105, v105, v216 :: v_dual_mul_f32 v104, v104, v216
	v_dual_mul_f32 v103, v103, v216 :: v_dual_mul_f32 v102, v102, v216
	v_dual_mul_f32 v133, v133, v216 :: v_dual_mul_f32 v132, v132, v216
	v_dual_mul_f32 v131, v131, v216 :: v_dual_mul_f32 v130, v130, v216
	v_dual_mul_f32 v129, v129, v216 :: v_dual_mul_f32 v128, v128, v216
	v_dual_mul_f32 v127, v127, v216 :: v_dual_mul_f32 v126, v126, v216
	v_dual_mul_f32 v117, v117, v216 :: v_dual_mul_f32 v116, v116, v216
	v_dual_mul_f32 v115, v115, v216 :: v_dual_mul_f32 v114, v114, v216
	v_dual_mul_f32 v113, v113, v216 :: v_dual_mul_f32 v112, v112, v216
	v_dual_mul_f32 v111, v111, v216 :: v_dual_mul_f32 v110, v110, v216
	v_dual_mul_f32 v101, v101, v216 :: v_dual_mul_f32 v100, v100, v216
	v_dual_mul_f32 v99, v99, v216 :: v_dual_mul_f32 v98, v98, v216
	v_dual_mul_f32 v97, v97, v216 :: v_dual_mul_f32 v96, v96, v216
	v_dual_mul_f32 v95, v95, v216 :: v_dual_mul_f32 v94, v94, v216
	v_dual_mul_f32 v85, v85, v216 :: v_dual_mul_f32 v84, v84, v216
	v_dual_mul_f32 v83, v83, v216 :: v_dual_mul_f32 v82, v82, v216
	v_dual_mul_f32 v81, v81, v216 :: v_dual_mul_f32 v80, v80, v216
	v_dual_mul_f32 v79, v79, v216 :: v_dual_mul_f32 v78, v78, v216
	v_dual_mul_f32 v77, v77, v216 :: v_dual_mul_f32 v76, v76, v216
	v_dual_mul_f32 v75, v75, v216 :: v_dual_mul_f32 v74, v74, v216
	v_dual_mul_f32 v73, v73, v216 :: v_dual_mul_f32 v72, v72, v216
	v_dual_mul_f32 v71, v71, v216 :: v_dual_mul_f32 v70, v70, v216
	v_dual_mul_f32 v69, v69, v216 :: v_dual_mul_f32 v68, v68, v216
	v_dual_mul_f32 v67, v67, v216 :: v_dual_mul_f32 v66, v66, v216
	v_dual_mul_f32 v65, v65, v216 :: v_dual_mul_f32 v64, v64, v216
	v_dual_mul_f32 v63, v63, v216 :: v_dual_mul_f32 v62, v62, v216
	v_dual_mul_f32 v61, v61, v216 :: v_dual_mul_f32 v60, v60, v216
	v_dual_mul_f32 v59, v59, v216 :: v_dual_mul_f32 v58, v58, v216
	v_dual_mul_f32 v57, v57, v216 :: v_dual_mul_f32 v56, v56, v216
	v_dual_mul_f32 v55, v55, v216 :: v_dual_mul_f32 v54, v54, v216
	v_dual_mul_f32 v53, v53, v216 :: v_dual_mul_f32 v52, v52, v216
	v_dual_mul_f32 v51, v51, v216 :: v_dual_mul_f32 v50, v50, v216
	v_dual_mul_f32 v49, v49, v216 :: v_dual_mul_f32 v48, v48, v216
	v_dual_mul_f32 v47, v47, v216 :: v_dual_mul_f32 v46, v46, v216
	v_dual_mul_f32 v45, v45, v216 :: v_dual_mul_f32 v44, v44, v216
	v_dual_mul_f32 v43, v43, v216 :: v_dual_mul_f32 v42, v42, v216
	v_dual_mul_f32 v41, v41, v216 :: v_dual_mul_f32 v40, v40, v216
	v_dual_mul_f32 v39, v39, v216 :: v_dual_mul_f32 v38, v38, v216
	v_dual_mul_f32 v37, v37, v216 :: v_dual_mul_f32 v36, v36, v216
	v_dual_mul_f32 v35, v35, v216 :: v_dual_mul_f32 v34, v34, v216
	v_dual_mul_f32 v33, v33, v216 :: v_dual_mul_f32 v32, v32, v216
	v_dual_mul_f32 v31, v31, v216 :: v_dual_mul_f32 v30, v30, v216
	v_dual_mul_f32 v29, v29, v216 :: v_dual_mul_f32 v28, v28, v216
	v_dual_mul_f32 v27, v27, v216 :: v_dual_mul_f32 v26, v26, v216
	v_dual_mul_f32 v25, v25, v216 :: v_dual_mul_f32 v24, v24, v216
	v_dual_mul_f32 v23, v23, v216 :: v_dual_mul_f32 v22, v22, v216
	v_dual_mul_f32 v21, v21, v216 :: v_dual_mul_f32 v20, v20, v216
	v_dual_mul_f32 v19, v19, v216 :: v_dual_mul_f32 v18, v18, v216
	v_dual_mul_f32 v17, v17, v216 :: v_dual_mul_f32 v16, v16, v216
	v_dual_mul_f32 v15, v15, v216 :: v_dual_mul_f32 v14, v14, v216
	v_dual_mul_f32 v13, v13, v216 :: v_dual_mul_f32 v12, v12, v216
	v_dual_mul_f32 v11, v11, v216 :: v_dual_mul_f32 v10, v10, v216
	v_dual_mul_f32 v9, v9, v216 :: v_dual_mul_f32 v8, v8, v216
	v_dual_mul_f32 v7, v7, v216 :: v_dual_mul_f32 v6, v6, v216
	.loc	1 2023 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2023:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v216, null, v4, v4, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v219, v216
	v_fma_f32 v220, -v216, v219, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v219, v220, v219
	v_div_scale_f32 v220, vcc_lo, v141, v4, v141
	v_mul_f32_e32 v221, v220, v219
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v222, -v216, v221, v220
	v_fmac_f32_e32 v221, v222, v219
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v216, v221, v220
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v216, v216, v219, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v216, v216, v4, v141
	.loc	1 2023 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2023:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v141, null, v4, v4, v140
	v_rcp_f32_e32 v219, v141
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v220, -v141, v219, 1.0
	v_fmac_f32_e32 v219, v220, v219
	v_div_scale_f32 v220, vcc_lo, v140, v4, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v221, v220, v219
	v_fma_f32 v222, -v141, v221, v220
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v221, v222, v219
	v_fma_f32 v141, -v141, v221, v220
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v141, v141, v219, v221
	v_div_fixup_f32 v219, v141, v4, v140
	.loc	1 2022 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b16_e64 v141.l, v2.l
	.loc	1 2024 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b16_e64 v141.h, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2022 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b16_e64 v140.l, v141.l
	.loc	1 2024 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b16_e64 v140.h, v141.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 2022 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cvt_pk_fp8_f32 v140.l, v216, v219
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v216, null, v4, v4, v139
	v_rcp_f32_e32 v219, v216
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v220, -v216, v219, 1.0
	v_fmac_f32_e32 v219, v220, v219
	v_div_scale_f32 v220, vcc_lo, v139, v4, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v221, v220, v219
	v_fma_f32 v222, -v216, v221, v220
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v221, v222, v219
	v_fma_f32 v216, -v216, v221, v220
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v216, v216, v219, v221
	v_div_fixup_f32 v139, v216, v4, v139
	.loc	1 2025 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v216, null, v4, v4, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v219, v216
	v_fma_f32 v220, -v216, v219, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v219, v220, v219
	v_div_scale_f32 v220, vcc_lo, v138, v4, v138
	v_mul_f32_e32 v221, v220, v219
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v222, -v216, v221, v220
	v_fmac_f32_e32 v221, v222, v219
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v216, v221, v220
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v216, v216, v219, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v138, v216, v4, v138
	.loc	1 2024 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cvt_pk_fp8_f32 v140.h, v139, v138
	.loc	1 2028 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v138, null, v4, v4, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v139, v138
	v_fma_f32 v216, -v138, v139, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v139, v216, v139
	v_div_scale_f32 v216, vcc_lo, v137, v4, v137
	v_mul_f32_e32 v219, v216, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v220, -v138, v219, v216
	v_fmac_f32_e32 v219, v220, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v138, v219, v216
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v138, v138, v139, v219
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v137, v138, v4, v137
	.loc	1 2028 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v138, null, v4, v4, v136
	v_rcp_f32_e32 v139, v138
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v138, v139, 1.0
	v_fmac_f32_e32 v139, v216, v139
	v_div_scale_f32 v216, vcc_lo, v136, v4, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v219, v216, v139
	v_fma_f32 v220, -v138, v219, v216
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v219, v220, v139 :: v_dual_mov_b32 v220, v5
	v_fma_f32 v138, -v138, v219, v216
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v138, v138, v139, v219
	v_div_fixup_f32 v136, v138, v4, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 2027 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cvt_pk_fp8_f32 v141.l, v137, v136
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v136, null, v4, v4, v135
	v_rcp_f32_e32 v137, v136
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v136, v137, 1.0
	v_fmac_f32_e32 v137, v138, v137
	v_div_scale_f32 v138, vcc_lo, v135, v4, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, v138, v137
	v_fma_f32 v216, -v136, v139, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v139, v216, v137
	v_fma_f32 v136, -v136, v139, v138
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v136, v136, v137, v139
	v_div_fixup_f32 v135, v136, v4, v135
	.loc	1 2030 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v136, null, v4, v4, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v137, v136
	v_fma_f32 v138, -v136, v137, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v137, v138, v137
	v_div_scale_f32 v138, vcc_lo, v134, v4, v134
	v_mul_f32_e32 v139, v138, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v136, v139, v138
	v_fmac_f32_e32 v139, v216, v137
	v_mov_b32_e32 v216, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v136, -v136, v139, v138
	.loc	1 2035 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshl_add_u32 v138, s14, 12, v186
	.loc	1 2030 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v136, v136, v137, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v134, v136, v4, v134
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cvt_pk_fp8_f32 v141.h, v135, v134
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:16384
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[86:93], v[134:135], v[140:141], v[86:93]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[136:137], v[140:141], v[118:125]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:16896
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[102:109], v[134:135], v[140:141], v[102:109]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[126:133], v[136:137], v[140:141], v[126:133]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:17408
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[110:117], v[134:135], v[140:141], v[110:117]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[94:101], v[136:137], v[140:141], v[94:101]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:17920
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[78:85], v[134:135], v[140:141], v[78:85]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[70:77], v[136:137], v[140:141], v[70:77]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:18432
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[62:69], v[134:135], v[140:141], v[62:69]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[54:61], v[136:137], v[140:141], v[54:61]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:18944
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[134:135], v[140:141], v[46:53]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[38:45], v[136:137], v[140:141], v[38:45]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:19456
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[30:37], v[134:135], v[140:141], v[30:37]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[22:29], v[136:137], v[140:141], v[22:29]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[134:137], v138 offset:19968
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[14:21], v[134:135], v[140:141], v[14:21]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[6:13], v[136:137], v[140:141], v[6:13]
	v_mov_b32_e32 v219, v3
	.loc	1 1946 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_add_co_i32 s14, s14, 1
	.loc	1 1946 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s14, 4
	.loc	1 1946 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_scc1 .LBB5_8
.LBB5_47:                               ;   Parent Loop BB5_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s14, 1
	s_mov_b32 s4, -1
	s_cbranch_scc1 .LBB5_50
; %bb.48:                               ;   in Loop: Header=BB5_47 Depth=2
	s_cmp_eq_u32 s14, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s4, s15
	s_cbranch_scc1 .LBB5_50
; %bb.49:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 1948 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cmp_eq_u32 s14, 2
	.loc	1 1948 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cselect_b32 s4, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s18, s4
	s_cselect_b32 s4, -1, 0
.LBB5_50:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_55
; %bb.51:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 1959 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1959:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshl_add_u32 v3, s14, 9, v186
	.loc	1 1971 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_lshl_b32 s4, s14, 4
	.loc	1 1971 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s21
	.loc	1 1972 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, s4, 15
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[221:224], v3
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[225:228], v3 offset:2048
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[229:232], v3 offset:4096
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1972 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s25, s19
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[221:222], v[145:146], 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[223:224], v[147:148], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[221:224], v3 offset:6144
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[225:226], v[149:150], v[134:141]
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[227:228], v[151:152], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[225:228], v3 offset:8192
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[229:230], v[153:154], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[231:232], v[155:156], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[229:232], v3 offset:10240
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[233:236], v3 offset:12288
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[221:222], v[157:158], v[134:141]
	v_lshl_add_u32 v221, s14, 5, v1
	v_mov_b32_e32 v222, 0xff800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[223:224], v[159:160], v[134:141]
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[225:226], v[161:162], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[227:228], v[163:164], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b128 v[224:227], v3 offset:14336
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1976 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_b96 v[3:5], v221 offset:49154
	ds_load_u16_d16 v223, v221 offset:49166
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[229:230], v[165:166], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[231:232], v[167:168], v[134:141]
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[233:234], v[169:170], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[235:236], v[171:172], v[134:141]
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[224:225], v[173:174], v[134:141]
	v_or_b32_e32 v224, s4, v183
	.loc	1 1972 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s13, s4
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[226:227], v[175:176], v[134:141]
	.loc	1 1981 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s25, s0
	s_cbranch_execz .LBB5_53
; %bb.52:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 1976 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_u16_d16 v222, v221 offset:49152
	v_mul_f32_e32 v134, v217, v134
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e32 vcc_lo, v224, v218
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v134, v222, v134, neg(0) op_sel_hi:[1,0,0]
	.loc	1 1983 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1983:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v222, 0xff800000, v134, vcc_lo
.LBB5_53:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s25
	v_mul_f32_e32 v134, v217, v135
	v_or_b32_e32 v135, 2, v224
	s_wait_loadcnt 0x0
	v_cmp_ge_i32_e32 vcc_lo, v224, v218
	s_xor_b32 s25, s4, -1
	v_mul_f32_e32 v136, v217, v136
	v_cmp_gt_i32_e64 s4, v135, v218
	v_or_b32_e32 v135, 3, v224
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s26, s25, vcc_lo
	.loc	1 1981 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s2, s26
	s_and_b32 s4, s25, s4
	v_cmp_gt_i32_e32 vcc_lo, v135, v218
	v_mul_f32_e32 v135, v217, v137
	s_wait_dscnt 0x1
	v_fma_mix_f32 v134, v3, v134, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v3, v136, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v136, 4, v224
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	v_cndmask_b32_e64 v134, v134, 0xff800000, s26
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v135, v217, v138
	s_and_b32 s4, s25, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136, v218
	v_or_b32_e32 v136, 5, v224
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v217, v139
	s_and_b32 s4, s25, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136, v218
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	v_or_b32_e32 v136, 7, v224
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v5, v4, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v4, 6, v224
	s_and_b32 s4, s25, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, v3, 0xff800000, s4
	v_cmp_gt_i32_e32 vcc_lo, v4, v218
	v_mul_f32_e32 v3, v217, v140
	v_cmp_gt_i32_e64 s4, v136, v218
	v_mul_f32_e32 v4, v217, v141
.Ltmp299:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1989:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v136, v222, 0xff800000, v134
	s_and_b32 s26, s25, vcc_lo
	v_fma_mix_f32 v3, v5, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s4, s25, s4
	s_wait_dscnt 0x0
	v_fma_mix_f32 v4, v223, v4, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v136, v137, v138
.Ltmp300:
	.loc	1 1981 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, s2, s26
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v136, v3, 0xff800000, s25
	v_cndmask_b32_e64 v4, v4, 0xff800000, s4
.Ltmp301:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1989:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v3, v5, v135, v139
.Ltmp302:
	.loc	1 2015 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2015:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp303:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1989:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v3, v3, v136, v4
.Ltmp304:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_mov_b32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v5, v5, s23, 0xfedcba98
.Ltmp305:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1991:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v3, v219, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_sub_f32 v139, v139, v3 :: v_dual_add_nc_u32 v140, 0xc080, v221
	v_dual_sub_f32 v4, v4, v3 :: v_dual_sub_f32 v5, v222, v3
	v_sub_f32_e32 v134, v134, v3
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v3
	v_dual_sub_f32 v135, v135, v3 :: v_dual_sub_f32 v136, v136, v3
	v_dual_mul_f32 v4, 0x3fb8aa3b, v4 :: v_dual_mul_f32 v5, 0x3fb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v136, 0x3fb8aa3b, v136
	v_exp_f32_e32 v141, v4
	v_dual_sub_f32 v4, v137, v3 :: v_dual_sub_f32 v137, v138, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v138, v5
	v_exp_f32_e32 v139, v139
	v_exp_f32_e32 v136, v136
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v134, 0x3fb8aa3b, v134
	v_mul_f32_e32 v223, 0x3fb8aa3b, v4
.Ltmp306:
	.loc	1 1997 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1997:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_2addr_b32 v[4:5], v140 offset1:1
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v134, v134
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_cndmask_b32_e64 v224, v139, 0, vcc_lo
	v_cndmask_b32_e64 v226, v136, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e64 v225, v137, 0, vcc_lo
	.loc	1 1997 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1997:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_nc_u32_e32 v140, 0xc088, v221
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cndmask_b32_e64 v134, v134, 0, vcc_lo
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	.loc	1 1997 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1997:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	ds_load_2addr_b32 v[221:222], v140 offset1:1
	v_exp_f32_e32 v140, v223
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cndmask_b32_e64 v223, v141, 0, vcc_lo
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v141, v4, v138, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cndmask_b32_e64 v137, v140, 0, vcc_lo
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_fma_mix_f32 v140, v4, v134, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_f32_e32 v134, v138, v134
	v_exp_f32_e32 v135, v135
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_fma_mix_f32 v138, v5, v225, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v139, v5, v137, neg(0) op_sel_hi:[1,0,0]
.Ltmp307:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v5, v141, 0, v140
.Ltmp308:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_f32_e32 v134, v137, v134
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v136, v221, v224, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
.Ltmp309:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v5, v5, v139, v138
.Ltmp310:
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cndmask_b32_e64 v4, v135, 0, vcc_lo
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_fma_mix_f32 v135, v222, v226, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v137, v221, v4, neg(0) op_sel_hi:[1,0,0]
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_f32_e32 v221, v225, v134
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_fma_mix_f32 v134, v222, v223, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp311:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v5, v5, v137, v136
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp312:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_add_f32_e32 v4, v4, v221
.Ltmp313:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max3_num_f32 v222, v5, v135, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp314:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_add_f32 v4, v224, v4 :: v_dual_mov_b32 v221, v222
	v_add_f32_e32 v4, v226, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp315:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_permlanex16_b32 v221, v221, s23, 0xfedcba98
.Ltmp316:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_add_f32 v5, v223, v4 :: v_dual_max_num_f32 v4, v221, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp317:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2007:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_dual_mov_b32 v221, v5 :: v_dual_max_num_f32 v222, v222, v4
	v_permlanex16_b32 v221, v221, s23, 0xfedcba98
	v_mov_b32_e32 v4, v216
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp318:
	.loc	1 2015 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2015:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmpx_lt_f32_e32 0, v222
	s_cbranch_execz .LBB5_46
; %bb.54:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 2016 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v4, null, 0x43e00000, 0x43e00000, v222
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v223, v4
	v_fma_f32 v224, -v4, v223, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v223, v224, v223
	v_div_scale_f32 v224, vcc_lo, v222, 0x43e00000, v222
	v_mul_f32_e32 v225, v224, v223
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v226, -v4, v225, v224
	v_fmac_f32_e32 v225, v226, v223
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v225, v224
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v4, v4, v223, v225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v4, v4, 0x43e00000, v222
.Ltmp319:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ] ]
	v_max_num_f32_e32 v4, 0x1f800000, v4
	s_branch .LBB5_46
.Ltmp320:
.LBB5_55:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v3, v219
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v219, v3
	.loc	1 1946 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_add_co_i32 s14, s14, 1
	.loc	1 1946 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s14, 4
	.loc	1 1946 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_cbranch_scc0 .LBB5_47
	s_branch .LBB5_8
.LBB5_56:
	.loc	1 2058 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_58
; %bb.57:
	.loc	1 2059 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_scale_f32 v0, null, v220, v220, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v220, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v1, v0
	v_fma_f32 v2, -v0, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v2, v1
	v_mul_f32_e32 v2, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v0, v2, v3
	v_fmac_f32_e32 v2, v4, v1
	.loc	1 2061 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2061:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mul_lo_u32 v4, 0x1800, v142
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 2059 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v1, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	.loc	1 2061 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2061:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshl_add_u32 v2, v144, 8, v4
	.loc	1 2059 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v220
	.loc	1 2067 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_mov_b32_e32 v1, 0
	.loc	1 2059 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_div_fixup_f32 v3, v0, v220, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 2063 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_or_b32_e32 v0, v2, v183
	.loc	1 2059 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2067 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v136, v216, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v134, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 2067 56 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mul_f32 v0, v86, v136 :: v_dual_mul_f32 v3, v89, v136
	v_dual_mul_f32 v1, v87, v136 :: v_dual_mul_f32 v2, v88, v136
	v_dual_mul_f32 v87, v91, v136 :: v_dual_mul_f32 v86, v90, v136
	v_dual_mul_f32 v89, v93, v136 :: v_dual_mul_f32 v88, v92, v136
	v_dual_mul_f32 v91, v119, v136 :: v_dual_mul_f32 v90, v118, v136
	v_dual_mul_f32 v93, v121, v136 :: v_dual_mul_f32 v92, v120, v136
	v_dual_mul_f32 v119, v123, v136 :: v_dual_mul_f32 v118, v122, v136
	v_dual_mul_f32 v121, v125, v136 :: v_dual_mul_f32 v120, v124, v136
	v_dual_mul_f32 v103, v103, v136 :: v_dual_mul_f32 v124, v128, v136
	v_dual_mul_f32 v128, v132, v136 :: v_dual_mul_f32 v111, v111, v136
	v_dual_mul_f32 v110, v110, v136 :: v_dual_mul_f32 v113, v113, v136
	v_dual_mul_f32 v112, v112, v136 :: v_dual_mul_f32 v115, v115, v136
	v_dual_mul_f32 v102, v102, v136 :: v_dual_mul_f32 v105, v105, v136
	v_dual_mul_f32 v104, v104, v136 :: v_dual_mul_f32 v107, v107, v136
	v_dual_mul_f32 v114, v114, v136 :: v_dual_mul_f32 v117, v117, v136
	v_mul_f32_e32 v116, v116, v136
	v_dual_mul_f32 v106, v106, v136 :: v_dual_mul_f32 v109, v109, v136
	v_dual_mul_f32 v108, v108, v136 :: v_dual_mul_f32 v123, v127, v136
	v_dual_mul_f32 v122, v126, v136 :: v_dual_mul_f32 v125, v129, v136
	v_dual_mul_f32 v126, v130, v136 :: v_dual_mul_f32 v129, v133, v136
	v_mul_f32_e32 v127, v131, v136
	.loc	1 2067 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x7
	global_store_b128 v[134:135], v[0:3], off
	global_store_b128 v[134:135], v[86:89], off offset:16
	global_store_b128 v[134:135], v[90:93], off offset:64
	global_store_b128 v[134:135], v[118:121], off offset:80
	global_store_b128 v[134:135], v[102:105], off offset:128
	global_store_b128 v[134:135], v[106:109], off offset:144
	global_store_b128 v[134:135], v[122:125], off offset:192
	global_store_b128 v[134:135], v[126:129], off offset:208
	.loc	1 2067 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mul_f32 v0, v94, v136 :: v_dual_mul_f32 v3, v97, v136
	v_dual_mul_f32 v1, v95, v136 :: v_dual_mul_f32 v2, v96, v136
	v_dual_mul_f32 v87, v99, v136 :: v_dual_mul_f32 v86, v98, v136
	v_dual_mul_f32 v89, v101, v136 :: v_dual_mul_f32 v88, v100, v136
	v_dual_mul_f32 v79, v79, v136 :: v_dual_mul_f32 v78, v78, v136
	v_dual_mul_f32 v81, v81, v136 :: v_dual_mul_f32 v80, v80, v136
	v_dual_mul_f32 v83, v83, v136 :: v_dual_mul_f32 v82, v82, v136
	v_dual_mul_f32 v85, v85, v136 :: v_dual_mul_f32 v84, v84, v136
	.loc	1 2067 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[110:113], off offset:256
	global_store_b128 v[134:135], v[114:117], off offset:272
	global_store_b128 v[134:135], v[0:3], off offset:320
	global_store_b128 v[134:135], v[86:89], off offset:336
	global_store_b128 v[134:135], v[78:81], off offset:384
	global_store_b128 v[134:135], v[82:85], off offset:400
	.loc	1 2067 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mul_f32 v0, v70, v136 :: v_dual_mul_f32 v3, v73, v136
	v_dual_mul_f32 v1, v71, v136 :: v_dual_mul_f32 v2, v72, v136
	v_dual_mul_f32 v71, v75, v136 :: v_dual_mul_f32 v70, v74, v136
	v_dual_mul_f32 v73, v77, v136 :: v_dual_mul_f32 v72, v76, v136
	v_dual_mul_f32 v63, v63, v136 :: v_dual_mul_f32 v62, v62, v136
	v_dual_mul_f32 v65, v65, v136 :: v_dual_mul_f32 v64, v64, v136
	v_dual_mul_f32 v67, v67, v136 :: v_dual_mul_f32 v66, v66, v136
	v_dual_mul_f32 v69, v69, v136 :: v_dual_mul_f32 v68, v68, v136
	v_dual_mul_f32 v55, v55, v136 :: v_dual_mul_f32 v54, v54, v136
	v_dual_mul_f32 v57, v57, v136 :: v_dual_mul_f32 v56, v56, v136
	v_dual_mul_f32 v59, v59, v136 :: v_dual_mul_f32 v58, v58, v136
	v_dual_mul_f32 v61, v61, v136 :: v_dual_mul_f32 v60, v60, v136
	.loc	1 2067 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[0:3], off offset:448
	global_store_b128 v[134:135], v[70:73], off offset:464
	global_store_b128 v[134:135], v[62:65], off offset:512
	global_store_b128 v[134:135], v[66:69], off offset:528
	global_store_b128 v[134:135], v[54:57], off offset:576
	global_store_b128 v[134:135], v[58:61], off offset:592
	.loc	1 2067 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mul_f32 v0, v46, v136 :: v_dual_mul_f32 v3, v49, v136
	v_dual_mul_f32 v1, v47, v136 :: v_dual_mul_f32 v2, v48, v136
	v_dual_mul_f32 v47, v51, v136 :: v_dual_mul_f32 v46, v50, v136
	v_dual_mul_f32 v49, v53, v136 :: v_dual_mul_f32 v48, v52, v136
	v_dual_mul_f32 v39, v39, v136 :: v_dual_mul_f32 v38, v38, v136
	v_dual_mul_f32 v41, v41, v136 :: v_dual_mul_f32 v40, v40, v136
	v_dual_mul_f32 v43, v43, v136 :: v_dual_mul_f32 v42, v42, v136
	v_dual_mul_f32 v45, v45, v136 :: v_dual_mul_f32 v44, v44, v136
	v_dual_mul_f32 v31, v31, v136 :: v_dual_mul_f32 v30, v30, v136
	v_dual_mul_f32 v33, v33, v136 :: v_dual_mul_f32 v32, v32, v136
	v_dual_mul_f32 v35, v35, v136 :: v_dual_mul_f32 v34, v34, v136
	v_dual_mul_f32 v37, v37, v136 :: v_dual_mul_f32 v36, v36, v136
	.loc	1 2067 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[0:3], off offset:640
	global_store_b128 v[134:135], v[46:49], off offset:656
	global_store_b128 v[134:135], v[38:41], off offset:704
	global_store_b128 v[134:135], v[42:45], off offset:720
	global_store_b128 v[134:135], v[30:33], off offset:768
	global_store_b128 v[134:135], v[34:37], off offset:784
	.loc	1 2067 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	v_dual_mul_f32 v0, v22, v136 :: v_dual_mul_f32 v3, v25, v136
	v_dual_mul_f32 v1, v23, v136 :: v_dual_mul_f32 v2, v24, v136
	v_dual_mul_f32 v23, v27, v136 :: v_dual_mul_f32 v22, v26, v136
	v_dual_mul_f32 v25, v29, v136 :: v_dual_mul_f32 v24, v28, v136
	v_dual_mul_f32 v15, v15, v136 :: v_dual_mul_f32 v14, v14, v136
	v_dual_mul_f32 v17, v17, v136 :: v_dual_mul_f32 v16, v16, v136
	v_dual_mul_f32 v19, v19, v136 :: v_dual_mul_f32 v18, v18, v136
	v_dual_mul_f32 v21, v21, v136 :: v_dual_mul_f32 v20, v20, v136
	v_dual_mul_f32 v5, v7, v136 :: v_dual_mul_f32 v4, v6, v136
	v_dual_mul_f32 v7, v9, v136 :: v_dual_mul_f32 v6, v8, v136
	v_dual_mul_f32 v9, v11, v136 :: v_dual_mul_f32 v8, v10, v136
	v_dual_mul_f32 v11, v13, v136 :: v_dual_mul_f32 v10, v12, v136
	.loc	1 2067 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[0:3], off offset:832
	global_store_b128 v[134:135], v[22:25], off offset:848
	global_store_b128 v[134:135], v[14:17], off offset:896
	global_store_b128 v[134:135], v[18:21], off offset:912
	global_store_b128 v[134:135], v[4:7], off offset:960
	global_store_b128 v[134:135], v[8:11], off offset:976
.Ltmp321:
.LBB5_58:
	.loc	1 2126 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2126:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp322:
.Lfunc_end5:
	.size	attention_fp8_e4m3_fa2_gqa_packet_gfx1201, .Lfunc_end5-attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_gfx1201
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
		.amdhsa_next_free_vgpr 237
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_vgpr, 237
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
; codeLenInByte = 8164
; TotalNumSgprs: 29
; NumVgprs: 237
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 237
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
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
	.loc	1 2141 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:17
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2141 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:23
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
	s_cbranch_vccnz .LBB6_61
; %bb.1:
	.loc	1 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2146 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2146:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB6_61
; %bb.2:
	.loc	1 2148 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2148:30
	s_lshl_b32 s14, ttmp9, 7
	.loc	1 2149 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2149:23
	s_mul_i32 s13, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2149 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2149:18
	s_cmp_ge_i32 s14, s13
	s_cbranch_scc1 .LBB6_61
; %bb.3:
	.loc	1 0 18                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2152 15 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2152:15
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB6_61
; %bb.4:
.Ltmp323:
	.loc	1 1765 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1765:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshrrev_b32_e32 v5, 5, v0
	.loc	1 1767 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_and_b32_e32 v8, 15, v0
.Ltmp324:
	.loc	1 2141 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:17
	s_load_b256 s[4:11], s[0:1], 0x0
.Ltmp325:
	.loc	1 1778 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mul_i32 s2, s3, 6
	.loc	1 1768 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1768:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_bfe_u32 v184, v0, 4, 1
	.loc	1 1776 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b32_e32 v10, 4, v5
.Ltmp326:
	.loc	1 2141 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:17
	s_load_b64 s[20:21], s[0:1], 0x20
.Ltmp327:
	.loc	1 1766 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1766:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_and_b32_e32 v11, 31, v0
	v_bfrev_b32_e32 v7, -2
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b32_e32 v182, 3, v184
	.loc	1 1776 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v1, v10, v8
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v6, -1 :: v_dual_add_nc_u32 v1, s14, v1
	.loc	1 1777 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1777:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 31, v2
	v_add_nc_u32_e32 v9, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1778 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_lo_u32 v2, v9, 6
	v_mul_lo_u32 v3, v9, 24
	v_sub_nc_u32_e32 v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v183, v2, s2, v3
	.loc	1 1779 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmp_gt_i32_e64 s2, s13, v1
	v_lshlrev_b32_e32 v2, 8, v183
	.loc	1 1792 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, 0, v2, s2
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, s12, s4, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s5, 0, s12
	s_mov_b32 s12, 0
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v1, vcc_lo, v1, v182
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	.loc	1 1797 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1797:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0xf
	global_load_b64 v[142:143], v[1:2], off
	global_load_b64 v[144:145], v[1:2], off offset:16
	global_load_b64 v[146:147], v[1:2], off offset:32
	global_load_b64 v[148:149], v[1:2], off offset:48
	global_load_b64 v[150:151], v[1:2], off offset:64
	global_load_b64 v[152:153], v[1:2], off offset:80
	global_load_b64 v[154:155], v[1:2], off offset:96
	global_load_b64 v[156:157], v[1:2], off offset:112
	global_load_b64 v[158:159], v[1:2], off offset:128
	global_load_b64 v[160:161], v[1:2], off offset:144
	global_load_b64 v[162:163], v[1:2], off offset:160
	global_load_b64 v[164:165], v[1:2], off offset:176
	global_load_b64 v[166:167], v[1:2], off offset:192
	global_load_b64 v[168:169], v[1:2], off offset:208
	global_load_b64 v[170:171], v[1:2], off offset:224
	global_load_b64 v[172:173], v[1:2], off offset:240
	.loc	1 1792 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b32_e32 v2, 0
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_gt_u32_e32 22, v11
	s_cbranch_execz .LBB6_8
; %bb.5:
	.loc	1 1805 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mul_hi_i32 s1, s14, 0x2aaaaaab
	v_bfrev_b32_e32 v7, -2
	s_lshr_b32 s19, s1, 31
	v_mov_b32_e32 v6, -1
	.loc	1 1805 37 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add3_u32 v3, s1, s19, v11
	.loc	1 1806 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1806:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v3
	s_cbranch_execz .LBB6_7
; %bb.6:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v4, 31, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	.loc	1 1807 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1807:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	global_load_b32 v6, v[3:4], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v7, v6
.LBB6_7:
	.loc	1 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.LBB6_8:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp328:
	v_mbcnt_lo_u32_b32 v13, -1, 0
	v_and_or_b32 v186, v10, 48, v8
	v_lshlrev_b32_e32 v16, 3, v11
	v_lshl_add_u32 v188, v11, 4, 0
.Ltmp329:
	.loc	1 1772 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1772:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshl_add_u32 v14, v5, 11, 0
.Ltmp330:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_xor_b32_e32 v3, 16, v13
	v_lshlrev_b32_e32 v187, 3, v5
	v_and_or_b32 v18, v5, 4, v184
.Ltmp331:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_xor_b32_e32 v28, 4, v13
.Ltmp332:
	.loc	1 1827 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1827:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cndmask_b32_e64 v1, 0, v183, s2
.Ltmp333:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_bfe_u32 v12, v0, 5, 1
	v_dual_mov_b32 v180, 0xff800000 :: v_dual_and_b32 v17, 16, v0
.Ltmp334:
	.loc	1 2154 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_add_co_i32 s1, s17, 0x1ff
.Ltmp335:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v3, v13, v3 :: v_dual_mov_b32 v4, v2
.Ltmp336:
	.loc	1 2154 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
.Ltmp337:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_xor_b32_e32 v29, 2, v13
.Ltmp338:
	.loc	1 2154 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s22, s1
.Ltmp339:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_lshlrev_b32_e32 v3, 2, v3
	v_cmp_eq_u32_e64 s1, v184, v12
	v_or_b32_e32 v20, 0x100, v0
	v_or_b32_e32 v21, 0x200, v0
	v_or_b32_e32 v22, 0x300, v0
	ds_bpermute_b32 v8, v3, v6
.Ltmp340:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_xor_b32_e32 v30, 1, v13
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshrrev_b32_e32 v31, 5, v21
.Ltmp341:
	.loc	1 2154 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_cvt_f32_u32 s19, s17
.Ltmp342:
	.loc	1 1826 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mul_i32 s24, s15, 0x1800
	.loc	1 1826 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s25, s12
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_add_nc_u64 s[4:5], s[4:5], s[24:25]
.Ltmp343:
	.loc	1 2154 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s24, s19
	s_mov_b32 s15, s12
	v_cmp_gt_u32_e64 s0, 64, v0
	v_lshl_add_u32 v189, v0, 1, 0
	s_mov_b32 s23, s12
	v_mov_b32_e32 v185, 1.0
	s_delay_alu instid0(TRANS32_DEP_1)
	s_mul_f32 s24, s22, s24
.Ltmp344:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v25, v6, v8
	v_mov_b32_e32 v6, v2
.Ltmp345:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v11, v3, v7
.Ltmp346:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_xor_b32_e32 v3, 8, v13
	v_mov_b32_e32 v8, v2
	s_delay_alu instid0(VALU_DEP_2)
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	.loc	2 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v24, v13, v3 :: v_dual_and_b32 v15, 0x7f, v0
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v3, v2
.Ltmp347:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v28
	v_add_nc_u32_e32 v190, v14, v16
.Ltmp348:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_lshlrev_b32_e32 v24, 2, v24
	v_xad_u32 v191, 0x120, v16, v14
	v_xad_u32 v192, 0x124, v16, v14
	v_xad_u32 v193, 0x240, v16, v14
.Ltmp349:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v26, v7, v11
.Ltmp350:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v27, v24, v25
	v_mov_b32_e32 v7, v2
	v_lshlrev_b32_e32 v23, 6, v0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
.Ltmp351:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v24, v24, v26
	v_xad_u32 v194, 0x244, v16, v14
	v_xad_u32 v195, 0x360, v16, v14
	v_xad_u32 v196, 0x364, v16, v14
	v_xad_u32 v197, 0x520, v16, v14
	v_xad_u32 v198, 0x524, v16, v14
	v_xad_u32 v199, 0x640, v16, v14
	v_xad_u32 v200, 0x644, v16, v14
	v_xad_u32 v201, 0x760, v16, v14
	v_xad_u32 v202, 0x764, v16, v14
	v_and_b32_e32 v14, 0x3000, v23
	v_add_nc_u32_e32 v1, 0, v17
.Ltmp352:
	.loc	2 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v13, v28, vcc_lo
	v_lshrrev_b32_e32 v28, 5, v20
	v_and_or_b32 v20, 0x180, v20, v15
	v_add_nc_u32_e32 v14, v188, v14
.Ltmp353:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v25, v25, v27
.Ltmp354:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_lshlrev_b32_e32 v17, 2, v17
	v_and_or_b32 v21, 0x280, v21, v15
	v_and_or_b32 v15, 0x380, v22, v15
.Ltmp355:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v24, v26, v24
	v_lshrrev_b32_e32 v26, 5, v22
.Ltmp356:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v22, v17, v25
.Ltmp357:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v29
	v_lshlrev_b32_e32 v20, 4, v20
.Ltmp358:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v17, v17, v24
	v_add_nc_u32_e32 v204, 0x4200, v14
	v_and_or_b32 v26, v26, 28, v184
.Ltmp359:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v27, v13, v29, vcc_lo
.Ltmp360:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v30
	v_add_nc_u32_e32 v206, 0x4600, v14
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp361:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_lshlrev_b32_e32 v23, 2, v27
.Ltmp362:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v13, v30, vcc_lo
	v_add_co_u32 v174, vcc_lo, s4, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v175, null, s5, v12, vcc_lo
	v_add_co_u32 v176, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v177, null, s21, v10, vcc_lo
.Ltmp363:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v22, v25, v22
.Ltmp364:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_lshlrev_b32_e32 v9, 2, v13
.Ltmp365:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v17, v24, v17
.Ltmp366:
	.loc	1 2154 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_trunc_f32 s4, s24
	v_and_or_b32 v24, v28, 12, v184
.Ltmp367:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v27, v23, v22
	v_and_or_b32 v25, v31, 20, v184
.Ltmp368:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v23, v23, v17
.Ltmp369:
	.loc	1 2154 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, s4, 0x80000000
	s_cvt_u32_f32 s4, s4
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s22, s5, s19
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_bitset0_b32 s22, 31
	s_cmp_ge_f32 s22, s19
	s_add_co_ci_u32 s4, s4, 0
.Ltmp370:
	.loc	1 1825 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_addk_co_i32 s14, 0x7f
.Ltmp371:
	.loc	1 2154 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s4, 0xffff
	.loc	1 2155 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2155:26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s19, s18, s4
.Ltmp372:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v10, v22, v27
.Ltmp373:
	.loc	1 2156 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2156:23
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s20, s19, s4
.Ltmp374:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v11, v17, v23
.Ltmp375:
	.loc	1 1825 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cmp_lt_i32 s14, s13
.Ltmp376:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v12, v9, v10
.Ltmp377:
	.loc	1 1825 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cselect_b32 s21, -1, 0
.Ltmp378:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	ds_bpermute_b32 v9, v9, v11
	s_lshl_b32 s14, s3, 8
	s_lshl_b32 s22, s3, 1
.Ltmp379:
	.loc	1 1828 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[6:7], s[14:15]
	s_add_nc_u64 s[14:15], s[8:9], s[14:15]
	s_add_nc_u64 s[6:7], s[6:7], s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v178, s3, s14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v179, null, s15, 0, s3
	s_add_nc_u64 s[8:9], s[8:9], s[22:23]
	s_mov_b32 s22, 0x76543210
.Ltmp380:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v10, v10, v12
.Ltmp381:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v9, v11, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp382:
	.loc	1 1822 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1822:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_readfirstlane_b32 s24, v10
	.loc	1 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_readfirstlane_b32 s25, v9
	v_mov_b32_e32 v9, v2
	v_lshlrev_b32_e32 v19, 4, v0
	v_lshlrev_b32_e32 v21, 4, v21
	v_lshlrev_b32_e32 v15, 4, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v125, v9 :: v_dual_add_nc_u32 v208, 0x4a00, v14
	v_mov_b32_e32 v109, v9
	v_dual_mov_b32 v102, v2 :: v_dual_add_nc_u32 v203, 0x4000, v14
	v_add_nc_u32_e32 v205, 0x4400, v14
	v_add_nc_u32_e32 v207, 0x4800, v14
	v_add_nc_u32_e32 v209, 0x4c00, v14
	v_dual_mov_b32 v123, v7 :: v_dual_add_nc_u32 v210, 0x4e00, v14
	v_lshlrev_b32_e32 v211, 3, v18
	v_dual_mov_b32 v121, v5 :: v_dual_lshlrev_b32 v212, 3, v24
	v_add_nc_u32_e32 v213, 0, v20
	v_dual_mov_b32 v119, v3 :: v_dual_lshlrev_b32 v214, 3, v25
	v_dual_mov_b32 v124, v8 :: v_dual_lshlrev_b32 v215, 3, v26
	v_dual_mov_b32 v106, v6 :: v_dual_add_nc_u32 v217, 0, v19
	v_dual_mov_b32 v105, v5 :: v_dual_add_nc_u32 v218, 0, v21
	v_mov_b32_e32 v104, v4
	v_dual_mov_b32 v107, v7 :: v_dual_add_nc_u32 v216, 0, v15
	v_mov_b32_e32 v93, v9
	v_mov_b32_e32 v77, v9
	v_mov_b32_e32 v133, v9
	v_mov_b32_e32 v117, v9
	v_mov_b32_e32 v101, v9
	v_mov_b32_e32 v85, v9
	v_mov_b32_e32 v69, v9
	v_mov_b32_e32 v61, v9
	v_mov_b32_e32 v53, v9
	v_mov_b32_e32 v45, v9
	v_mov_b32_e32 v37, v9
	v_mov_b32_e32 v29, v9
	v_dual_mov_b32 v21, v9 :: v_dual_mov_b32 v122, v6
	v_mov_b32_e32 v120, v4
	v_mov_b32_e32 v118, v2
	v_dual_mov_b32 v108, v8 :: v_dual_mov_b32 v103, v3
	v_mov_b32_e32 v132, v8
	v_dual_mov_b32 v92, v8 :: v_dual_mov_b32 v91, v7
	v_mov_b32_e32 v128, v4
	v_dual_mov_b32 v90, v6 :: v_dual_mov_b32 v89, v5
	v_mov_b32_e32 v126, v2
	v_dual_mov_b32 v88, v4 :: v_dual_mov_b32 v87, v3
	v_mov_b32_e32 v116, v8
	v_mov_b32_e32 v86, v2
	v_dual_mov_b32 v76, v8 :: v_dual_mov_b32 v75, v7
	v_mov_b32_e32 v112, v4
	v_dual_mov_b32 v74, v6 :: v_dual_mov_b32 v73, v5
	v_mov_b32_e32 v110, v2
	v_dual_mov_b32 v72, v4 :: v_dual_mov_b32 v71, v3
	v_mov_b32_e32 v100, v8
	v_dual_mov_b32 v70, v2 :: v_dual_mov_b32 v181, 0
	v_dual_mov_b32 v98, v6 :: v_dual_mov_b32 v131, v7
	v_mov_b32_e32 v94, v2
	v_dual_mov_b32 v130, v6 :: v_dual_mov_b32 v129, v5
	v_dual_mov_b32 v84, v8 :: v_dual_mov_b32 v127, v3
	v_dual_mov_b32 v82, v6 :: v_dual_mov_b32 v115, v7
	v_mov_b32_e32 v78, v2
	v_dual_mov_b32 v114, v6 :: v_dual_mov_b32 v113, v5
	v_dual_mov_b32 v68, v8 :: v_dual_mov_b32 v111, v3
	v_dual_mov_b32 v66, v6 :: v_dual_mov_b32 v99, v7
	v_dual_mov_b32 v62, v2 :: v_dual_mov_b32 v97, v5
	v_mov_b32_e32 v60, v8
	v_dual_mov_b32 v96, v4 :: v_dual_mov_b32 v95, v3
	v_dual_mov_b32 v58, v6 :: v_dual_mov_b32 v83, v7
	v_dual_mov_b32 v54, v2 :: v_dual_mov_b32 v81, v5
	v_mov_b32_e32 v52, v8
	v_dual_mov_b32 v80, v4 :: v_dual_mov_b32 v79, v3
	v_dual_mov_b32 v50, v6 :: v_dual_mov_b32 v67, v7
	v_dual_mov_b32 v46, v2 :: v_dual_mov_b32 v65, v5
	v_mov_b32_e32 v44, v8
	v_dual_mov_b32 v64, v4 :: v_dual_mov_b32 v63, v3
	v_dual_mov_b32 v42, v6 :: v_dual_mov_b32 v59, v7
	v_dual_mov_b32 v38, v2 :: v_dual_mov_b32 v57, v5
	v_mov_b32_e32 v36, v8
	v_dual_mov_b32 v56, v4 :: v_dual_mov_b32 v55, v3
	v_dual_mov_b32 v34, v6 :: v_dual_mov_b32 v51, v7
	v_dual_mov_b32 v30, v2 :: v_dual_mov_b32 v49, v5
	v_mov_b32_e32 v28, v8
	v_dual_mov_b32 v48, v4 :: v_dual_mov_b32 v47, v3
	v_dual_mov_b32 v26, v6 :: v_dual_mov_b32 v43, v7
	v_dual_mov_b32 v22, v2 :: v_dual_mov_b32 v41, v5
	v_mov_b32_e32 v20, v8
	v_dual_mov_b32 v40, v4 :: v_dual_mov_b32 v39, v3
	v_dual_mov_b32 v18, v6 :: v_dual_mov_b32 v35, v7
	v_dual_mov_b32 v14, v2 :: v_dual_mov_b32 v33, v5
	v_dual_mov_b32 v32, v4 :: v_dual_mov_b32 v31, v3
	v_mov_b32_e32 v27, v7
	v_dual_mov_b32 v25, v5 :: v_dual_mov_b32 v24, v4
	v_mov_b32_e32 v23, v3
	v_mov_b32_e32 v19, v7
	v_dual_mov_b32 v17, v5 :: v_dual_mov_b32 v16, v4
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v13, v9
	v_dual_mov_b32 v12, v8 :: v_dual_mov_b32 v11, v7
	v_dual_mov_b32 v10, v6 :: v_dual_mov_b32 v9, v5
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v2
	s_branch .LBB6_11
.LBB6_9:                                ;   in Loop: Header=BB6_11 Depth=1
.Ltmp383:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2054:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2054:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v180, v3
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2054:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp384:
.LBB6_10:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	4 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s19, s19, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s19, s20
	s_cselect_b32 s3, -1, 0
	s_xor_b32 s13, s23, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s13, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_57
.LBB6_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_48 Depth 2
	.loc	1 1829 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_lshl_b32 s26, s19, 6
	.loc	1 1830 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1830:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s26, s24
	s_cselect_b32 s23, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_10
; %bb.12:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v3, s26, v186
                                        ; implicit-def: $vgpr134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[138:139], null, 0x408, v3, s[4:5]
	v_cmp_ge_i32_e32 vcc_lo, s24, v3
	.loc	1 1843 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s13, exec_lo, s3
	s_cbranch_execz .LBB6_14
; %bb.13:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1850 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v3, s3, v138, v211
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s3
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
	v_add_co_u32 v3, s3, v138, v212
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s3
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v217, v[134:137]
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
.LBB6_14:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s13
	s_cbranch_execz .LBB6_16
; %bb.15:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v137, 0
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v136, v137
	v_dual_mov_b32 v135, v137 :: v_dual_mov_b32 v134, v137
	ds_store_b128 v217, v[2:5]
.LBB6_16:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v213, v[134:137]
                                        ; implicit-def: $vgpr134
	.loc	1 1843 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v3, vcc_lo, v138, v214
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
	v_add_co_u32 v3, vcc_lo, v138, v215
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v218, v[134:137]
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
.LBB6_18:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v4, v2
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v134, v137
	v_dual_mov_b32 v136, v137 :: v_dual_mov_b32 v135, v137
	ds_store_b128 v218, v[2:5]
.LBB6_20:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v216, v[134:137]
	v_dual_mov_b32 v134, 0 :: v_dual_add_nc_u32 v5, s26, v187
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v3, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[178:179]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_22:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v136, 0x8000, v190
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v136, v3, v4 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_gt_i32_e64 s24, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1862 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v3, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[178:179]
	global_load_b64 v[134:135], v[3:4], off
.LBB6_24:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v138, 2, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v191, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b32 v192, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v138, v[178:179]
	global_load_b64 v[136:137], v[134:135], off
.LBB6_26:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v134, 3, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v193, v136 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b32 v194, v137 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v134
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_28
; %bb.27:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v134, v[178:179]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_28:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v138, 4, v5
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v195, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b32 v196, v4 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v138, v[178:179]
	global_load_b64 v[136:137], v[3:4], off
.LBB6_30:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v3, 5, v5
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v4, 0x8400, v190
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v4, v136, v137 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v3
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_32
; %bb.31:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[178:179]
	global_load_b64 v[134:135], v[3:4], off
.LBB6_32:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v138, 6, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v197, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b32 v198, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_34
; %bb.33:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v138, v[178:179]
	global_load_b64 v[136:137], v[134:135], off
.LBB6_34:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v5, 7, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v199, v136 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b32 v200, v137 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_ge_i32_e64 s24, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_execz .LBB6_36
; %bb.35:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[178:179]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_36:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1871 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v201, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b32 v202, v4 offset:32768
.Ltmp385:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp386:
	.loc	1 1888 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1888:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1904 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s15, s12
	s_mov_b32 s13, s12
	s_mov_b32 s14, s12
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v137, s15 :: v_dual_mov_b32 v136, s14
	v_dual_mov_b32 v135, s13 :: v_dual_mov_b32 v134, s12
	ds_store_b128 v203, v[134:137]
	ds_store_b128 v204, v[134:137]
	ds_store_b128 v205, v[134:137]
	ds_store_b128 v206, v[134:137]
	ds_store_b128 v207, v[134:137]
	ds_store_b128 v208, v[134:137]
	ds_store_b128 v209, v[134:137]
	ds_store_b128 v210, v[134:137]
.LBB6_38:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1910 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB6_42
; %bb.39:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1911 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1911:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v4, s26, v0
	v_mov_b32_e32 v3, 0
	.loc	1 1913 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1913:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s24, v4
	s_cbranch_execz .LBB6_41
; %bb.40:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1918 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v4, s[6:7]
	.loc	1 1920 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, s[8:9]
	.loc	1 1918 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	global_load_d16_b16 v3, v[134:135], off offset:1024
	.loc	1 1920 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	global_load_d16_hi_b16 v3, v[4:5], off offset:1024
	.loc	1 1918 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v4.h, 8, v3.l
	.loc	1 1920 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshrrev_b16 v4.l, 8, v3.h
	.loc	1 1920 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_and_b16 v5.h, 0xff, v3.l
	v_and_b16 v5.l, 0xff, v3.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1921 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1921:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_pk_lshlrev_b16 v3, 8, v4 op_sel_hi:[0,1]
	.loc	1 1920 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1920:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_or_b32_e32 v3, v3, v5
.LBB6_41:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	.loc	1 1923 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b16_d16_hi v189, v3 offset:49152
	.loc	1 1924 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1924:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_store_b16 v189, v3 offset:49280
.LBB6_42:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp387:
	.loc	4 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp388:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b32_e32 v219, s16
.Ltmp389:
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	s_barrier_wait -1
	.loc	4 703 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp390:
	.loc	1 1936 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1936:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_44
; %bb.43:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1937 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	global_load_b32 v3, v[174:175], off
	.loc	1 1938 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v219, s16, v3
.LBB6_44:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1834 35 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1834:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_or_b32 s3, s26, 63
	v_mov_b32_e32 v220, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s25
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s13, s21, s3
	.loc	1 1940 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s13, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s14
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1940 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	global_load_b32 v220, v[176:177], off
.LBB6_46:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1930 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_or_b32 s3, s26, 16
	s_mov_b32 s14, 0
	.loc	1 1930 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s24
	s_cselect_b32 s15, -1, 0
	s_branch .LBB6_48
.LBB6_47:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_sub_f32 v222, v180, v3 :: v_dual_add_f32 v5, v5, v221
	.loc	1 1993 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1993:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v180
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v222, 0x3fb8aa3b, v222
	v_exp_f32_e32 v222, v222
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v180, 0, v222, vcc_lo
	.loc	1 2008 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2008:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_fmac_f32_e32 v5, v181, v180
	.loc	1 2017 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v180, v185, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2017 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v181, null, v4, v4, v180
	v_rcp_f32_e32 v185, v181
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v221, -v181, v185, 1.0
	v_fmac_f32_e32 v185, v221, v185
	v_div_scale_f32 v221, vcc_lo, v180, v4, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v222, v221, v185
	v_fma_f32 v223, -v181, v222, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v222, v223, v185
	v_fma_f32 v181, -v181, v222, v221
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v181, v181, v185, v222
	v_div_fixup_f32 v180, v181, v4, v180
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 2020 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2020:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v125, v125, v180 :: v_dual_mul_f32 v124, v124, v180
	v_mul_f32_e32 v121, v121, v180
	v_dual_mul_f32 v123, v123, v180 :: v_dual_mul_f32 v122, v122, v180
	v_dual_mul_f32 v119, v119, v180 :: v_dual_mul_f32 v120, v120, v180
	v_dual_mul_f32 v109, v109, v180 :: v_dual_mul_f32 v118, v118, v180
	v_dual_mul_f32 v107, v107, v180 :: v_dual_mul_f32 v108, v108, v180
	v_dual_mul_f32 v105, v105, v180 :: v_dual_mul_f32 v106, v106, v180
	v_dual_mul_f32 v103, v103, v180 :: v_dual_mul_f32 v104, v104, v180
	v_dual_mul_f32 v93, v93, v180 :: v_dual_mul_f32 v102, v102, v180
	v_dual_mul_f32 v91, v91, v180 :: v_dual_mul_f32 v92, v92, v180
	v_dual_mul_f32 v89, v89, v180 :: v_dual_mul_f32 v90, v90, v180
	v_dual_mul_f32 v87, v87, v180 :: v_dual_mul_f32 v88, v88, v180
	v_dual_mul_f32 v77, v77, v180 :: v_dual_mul_f32 v86, v86, v180
	v_dual_mul_f32 v75, v75, v180 :: v_dual_mul_f32 v76, v76, v180
	v_dual_mul_f32 v73, v73, v180 :: v_dual_mul_f32 v74, v74, v180
	v_dual_mul_f32 v71, v71, v180 :: v_dual_mul_f32 v72, v72, v180
	v_dual_mul_f32 v133, v133, v180 :: v_dual_mul_f32 v70, v70, v180
	v_dual_mul_f32 v131, v131, v180 :: v_dual_mul_f32 v132, v132, v180
	v_dual_mul_f32 v129, v129, v180 :: v_dual_mul_f32 v130, v130, v180
	v_dual_mul_f32 v127, v127, v180 :: v_dual_mul_f32 v128, v128, v180
	v_dual_mul_f32 v117, v117, v180 :: v_dual_mul_f32 v126, v126, v180
	v_dual_mul_f32 v115, v115, v180 :: v_dual_mul_f32 v116, v116, v180
	v_dual_mul_f32 v113, v113, v180 :: v_dual_mul_f32 v114, v114, v180
	v_dual_mul_f32 v111, v111, v180 :: v_dual_mul_f32 v112, v112, v180
	v_dual_mul_f32 v101, v101, v180 :: v_dual_mul_f32 v110, v110, v180
	v_dual_mul_f32 v99, v99, v180 :: v_dual_mul_f32 v100, v100, v180
	v_dual_mul_f32 v97, v97, v180 :: v_dual_mul_f32 v98, v98, v180
	v_dual_mul_f32 v95, v95, v180 :: v_dual_mul_f32 v96, v96, v180
	v_dual_mul_f32 v85, v85, v180 :: v_dual_mul_f32 v94, v94, v180
	v_dual_mul_f32 v83, v83, v180 :: v_dual_mul_f32 v84, v84, v180
	v_dual_mul_f32 v81, v81, v180 :: v_dual_mul_f32 v82, v82, v180
	v_dual_mul_f32 v79, v79, v180 :: v_dual_mul_f32 v80, v80, v180
	v_dual_mul_f32 v69, v69, v180 :: v_dual_mul_f32 v78, v78, v180
	v_dual_mul_f32 v67, v67, v180 :: v_dual_mul_f32 v68, v68, v180
	v_dual_mul_f32 v65, v65, v180 :: v_dual_mul_f32 v66, v66, v180
	v_dual_mul_f32 v63, v63, v180 :: v_dual_mul_f32 v64, v64, v180
	v_dual_mul_f32 v61, v61, v180 :: v_dual_mul_f32 v62, v62, v180
	v_dual_mul_f32 v59, v59, v180 :: v_dual_mul_f32 v60, v60, v180
	v_dual_mul_f32 v57, v57, v180 :: v_dual_mul_f32 v58, v58, v180
	v_dual_mul_f32 v55, v55, v180 :: v_dual_mul_f32 v56, v56, v180
	v_dual_mul_f32 v53, v53, v180 :: v_dual_mul_f32 v54, v54, v180
	v_dual_mul_f32 v51, v51, v180 :: v_dual_mul_f32 v52, v52, v180
	v_dual_mul_f32 v49, v49, v180 :: v_dual_mul_f32 v50, v50, v180
	v_dual_mul_f32 v47, v47, v180 :: v_dual_mul_f32 v48, v48, v180
	v_dual_mul_f32 v45, v45, v180 :: v_dual_mul_f32 v46, v46, v180
	v_dual_mul_f32 v43, v43, v180 :: v_dual_mul_f32 v44, v44, v180
	v_dual_mul_f32 v41, v41, v180 :: v_dual_mul_f32 v42, v42, v180
	v_dual_mul_f32 v39, v39, v180 :: v_dual_mul_f32 v40, v40, v180
	v_dual_mul_f32 v37, v37, v180 :: v_dual_mul_f32 v38, v38, v180
	v_dual_mul_f32 v35, v35, v180 :: v_dual_mul_f32 v36, v36, v180
	v_dual_mul_f32 v33, v33, v180 :: v_dual_mul_f32 v34, v34, v180
	v_dual_mul_f32 v31, v31, v180 :: v_dual_mul_f32 v32, v32, v180
	v_dual_mul_f32 v29, v29, v180 :: v_dual_mul_f32 v30, v30, v180
	v_dual_mul_f32 v27, v27, v180 :: v_dual_mul_f32 v28, v28, v180
	v_dual_mul_f32 v25, v25, v180 :: v_dual_mul_f32 v26, v26, v180
	v_dual_mul_f32 v23, v23, v180 :: v_dual_mul_f32 v24, v24, v180
	v_dual_mul_f32 v21, v21, v180 :: v_dual_mul_f32 v22, v22, v180
	v_dual_mul_f32 v19, v19, v180 :: v_dual_mul_f32 v20, v20, v180
	v_dual_mul_f32 v17, v17, v180 :: v_dual_mul_f32 v18, v18, v180
	v_dual_mul_f32 v15, v15, v180 :: v_dual_mul_f32 v16, v16, v180
	v_dual_mul_f32 v13, v13, v180 :: v_dual_mul_f32 v14, v14, v180
	v_dual_mul_f32 v11, v11, v180 :: v_dual_mul_f32 v12, v12, v180
	v_dual_mul_f32 v9, v9, v180 :: v_dual_mul_f32 v10, v10, v180
	v_dual_mul_f32 v7, v7, v180 :: v_dual_mul_f32 v8, v8, v180
	v_mul_f32_e32 v6, v6, v180
	.loc	1 2023 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2023:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v180, null, v4, v4, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v181, v180
	v_fma_f32 v185, -v180, v181, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v185, v181
	v_div_scale_f32 v185, vcc_lo, v141, v4, v141
	v_mul_f32_e32 v221, v185, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v222, -v180, v221, v185
	v_fmac_f32_e32 v221, v222, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v180, v221, v185
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v180, v180, v181, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v180, v180, v4, v141
	.loc	1 2023 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2023:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v141, null, v4, v4, v140
	v_rcp_f32_e32 v181, v141
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v185, -v141, v181, 1.0
	v_fmac_f32_e32 v181, v185, v181
	v_div_scale_f32 v185, vcc_lo, v140, v4, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v221, v185, v181
	v_fma_f32 v222, -v141, v221, v185
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v221, v222, v181
	v_fma_f32 v141, -v141, v221, v185
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v141, v141, v181, v221
	v_div_fixup_f32 v181, v141, v4, v140
	.loc	1 2022 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b16_e64 v141.l, v2.l
	.loc	1 2024 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b16_e64 v141.h, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2022 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b16_e64 v140.l, v141.l
	.loc	1 2024 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b16_e64 v140.h, v141.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 2022 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cvt_pk_fp8_f32 v140.l, v180, v181
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v180, null, v4, v4, v139
	v_rcp_f32_e32 v181, v180
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v185, -v180, v181, 1.0
	v_fmac_f32_e32 v181, v185, v181
	v_div_scale_f32 v185, vcc_lo, v139, v4, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v221, v185, v181
	v_fma_f32 v222, -v180, v221, v185
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v221, v222, v181
	v_fma_f32 v180, -v180, v221, v185
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v180, v180, v181, v221
	v_div_fixup_f32 v139, v180, v4, v139
	.loc	1 2025 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v180, null, v4, v4, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v181, v180
	v_fma_f32 v185, -v180, v181, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v185, v181
	v_div_scale_f32 v185, vcc_lo, v138, v4, v138
	v_mul_f32_e32 v221, v185, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v222, -v180, v221, v185
	v_fmac_f32_e32 v221, v222, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v180, v221, v185
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v180, v180, v181, v221
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v138, v180, v4, v138
	.loc	1 2024 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cvt_pk_fp8_f32 v140.h, v139, v138
	.loc	1 2028 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v138, null, v4, v4, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v139, v138
	v_fma_f32 v180, -v138, v139, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v139, v180, v139
	v_div_scale_f32 v180, vcc_lo, v137, v4, v137
	v_mul_f32_e32 v181, v180, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v185, -v138, v181, v180
	v_fmac_f32_e32 v181, v185, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v138, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v138, v138, v139, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v137, v138, v4, v137
	.loc	1 2028 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v138, null, v4, v4, v136
	v_rcp_f32_e32 v139, v138
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v138, v139, 1.0
	v_fmac_f32_e32 v139, v180, v139
	v_div_scale_f32 v180, vcc_lo, v136, v4, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v181, v180, v139
	v_fma_f32 v185, -v138, v181, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v181, v185, v139
	v_mov_b32_e32 v185, v4
	v_fma_f32 v138, -v138, v181, v180
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v138, v138, v139, v181
	v_mov_b32_e32 v181, v5
	v_div_fixup_f32 v136, v138, v4, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 2027 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cvt_pk_fp8_f32 v141.l, v137, v136
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v136, null, v4, v4, v135
	v_rcp_f32_e32 v137, v136
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v138, -v136, v137, 1.0
	v_fmac_f32_e32 v137, v138, v137
	v_div_scale_f32 v138, vcc_lo, v135, v4, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, v138, v137
	v_fma_f32 v180, -v136, v139, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v139, v180, v137
	v_fma_f32 v136, -v136, v139, v138
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v136, v136, v137, v139
	v_div_fixup_f32 v135, v136, v4, v135
	.loc	1 2030 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v136, null, v4, v4, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v137, v136
	v_fma_f32 v138, -v136, v137, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v137, v138, v137
	v_div_scale_f32 v138, vcc_lo, v134, v4, v134
	v_mul_f32_e32 v139, v138, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v136, v139, v138
	v_fmac_f32_e32 v139, v180, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v136, -v136, v139, v138
	.loc	1 2035 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshl_add_u32 v138, s14, 12, v188
	.loc	1 2030 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v136, v136, v137, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v134, v136, v4, v134
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cvt_pk_fp8_f32 v141.h, v135, v134
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:16384
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[134:135], v[140:141], v[118:125]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[102:109], v[136:137], v[140:141], v[102:109]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:16896
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[86:93], v[134:135], v[140:141], v[86:93]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[70:77], v[136:137], v[140:141], v[70:77]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:17408
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[126:133], v[134:135], v[140:141], v[126:133]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[110:117], v[136:137], v[140:141], v[110:117]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:17920
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[94:101], v[134:135], v[140:141], v[94:101]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[78:85], v[136:137], v[140:141], v[78:85]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:18432
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[62:69], v[134:135], v[140:141], v[62:69]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[54:61], v[136:137], v[140:141], v[54:61]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:18944
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[134:135], v[140:141], v[46:53]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[38:45], v[136:137], v[140:141], v[38:45]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:19456
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[30:37], v[134:135], v[140:141], v[30:37]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[22:29], v[136:137], v[140:141], v[22:29]
	.loc	1 2037 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[134:137], v138 offset:19968
	.loc	1 2046 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2041 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2041:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[14:21], v[134:135], v[140:141], v[14:21]
	.loc	1 2044 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[6:13], v[136:137], v[140:141], v[6:13]
	v_mov_b32_e32 v180, v3
	.loc	1 1946 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_add_co_i32 s14, s14, 1
	.loc	1 1946 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s14, 4
	.loc	1 1946 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_scc1 .LBB6_9
.LBB6_48:                               ;   Parent Loop BB6_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s14, 1
	s_mov_b32 s3, -1
	s_cbranch_scc1 .LBB6_51
; %bb.49:                               ;   in Loop: Header=BB6_48 Depth=2
	s_cmp_eq_u32 s14, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s3, s15
	s_cbranch_scc1 .LBB6_51
; %bb.50:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 1948 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cmp_eq_u32 s14, 2
	.loc	1 1948 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1948:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cselect_b32 s3, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s24, s3
	s_cselect_b32 s3, -1, 0
.LBB6_51:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_56
; %bb.52:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 1959 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1959:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshl_add_u32 v3, s14, 9, v188
	.loc	1 1971 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_lshl_b32 s3, s14, 4
	.loc	1 1971 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s26
	.loc	1 1972 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s27, s3, 15
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[221:224], v3
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[225:228], v3 offset:2048
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[229:232], v3 offset:4096
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1972 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s27, s25
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[221:222], v[142:143], 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[223:224], v[144:145], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[221:224], v3 offset:6144
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[225:226], v[146:147], v[134:141]
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[227:228], v[148:149], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[225:228], v3 offset:8192
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[229:230], v[150:151], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[231:232], v[152:153], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[229:232], v3 offset:10240
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[233:236], v3 offset:12288
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[221:222], v[154:155], v[134:141]
	v_lshl_add_u32 v221, s14, 5, v1
	v_mov_b32_e32 v222, 0xff800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[223:224], v[156:157], v[134:141]
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[225:226], v[158:159], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[227:228], v[160:161], v[134:141]
	.loc	1 1961 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b128 v[224:227], v3 offset:14336
	.loc	1 1968 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1968:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1976 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_b96 v[3:5], v221 offset:49154
	ds_load_u16_d16 v223, v221 offset:49166
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[229:230], v[162:163], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[231:232], v[164:165], v[134:141]
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[233:234], v[166:167], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[235:236], v[168:169], v[134:141]
	.loc	1 1964 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[224:225], v[170:171], v[134:141]
	v_or_b32_e32 v224, s3, v182
	.loc	1 1972 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s27, s13, s3
	.loc	1 1966 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[226:227], v[172:173], v[134:141]
	.loc	1 1981 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_54
; %bb.53:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 1976 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_u16_d16 v222, v221 offset:49152
	v_mul_f32_e32 v134, v219, v134
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e32 vcc_lo, v224, v220
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s27, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v134, v222, v134, neg(0) op_sel_hi:[1,0,0]
	.loc	1 1983 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1983:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v222, 0xff800000, v134, vcc_lo
.LBB6_54:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v134, v219, v135
	v_or_b32_e32 v135, 2, v224
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, v224, v220
	v_mul_f32_e32 v136, v219, v136
	s_wait_dscnt 0x1
	v_fma_mix_f32 v134, v3, v134, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s3, v135, v220
	v_or_b32_e32 v135, 3, v224
	s_or_b32 s28, s27, vcc_lo
	v_fma_mix_f32 v3, v3, v136, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 1981 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_b32 vcc_lo, s2, s28
	s_or_b32 s3, s27, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v134, 0xff800000, v134, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135, v220
	v_mul_f32_e32 v135, v219, v137
	v_or_b32_e32 v136, 4, v224
	s_and_b32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v135, v219, v138
	s_or_b32 s3, s27, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v220
	v_or_b32_e32 v136, 5, v224
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v219, v139
	s_or_b32 s3, s27, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v220
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, s3
	v_or_b32_e32 v136, 7, v224
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v5, v4, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v4, 6, v224
	s_or_b32 s3, s27, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s2, s3
	v_cmp_le_i32_e64 s3, v136, v220
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v139, 0xff800000, v3, vcc_lo
	v_mul_f32_e32 v3, v219, v140
	v_cmp_le_i32_e32 vcc_lo, v4, v220
	v_mul_f32_e32 v4, v219, v141
.Ltmp391:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1989:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v136, v222, 0xff800000, v134
	s_or_b32 s3, s27, s3
	v_fma_mix_f32 v3, v5, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_or_b32 s28, s27, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v4, v223, v4, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v136, v137, v138
.Ltmp392:
	.loc	1 1981 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_b32 vcc_lo, s2, s28
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v136, 0xff800000, v3, vcc_lo
	s_and_b32 vcc_lo, s2, s3
.Ltmp393:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1989:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v3, v5, v135, v139
.Ltmp394:
	.loc	1 1981 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v4, 0xff800000, v4, vcc_lo
	.loc	1 2015 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2015:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp395:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1989:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v3, v3, v136, v4
.Ltmp396:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1990:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_mov_b32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v5, v5, s22, 0xfedcba98
.Ltmp397:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1991:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v3, v180, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v135, v135, v3 :: v_dual_add_nc_u32 v140, 0xc080, v221
	v_dual_sub_f32 v5, v222, v3 :: v_dual_sub_f32 v4, v4, v3
	v_dual_sub_f32 v134, v134, v3 :: v_dual_sub_f32 v139, v139, v3
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v3
	v_dual_mul_f32 v5, 0x3fb8aa3b, v5 :: v_dual_mul_f32 v4, 0x3fb8aa3b, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v141, v4
	v_sub_f32_e32 v4, v137, v3
	v_dual_mul_f32 v134, 0x3fb8aa3b, v134 :: v_dual_sub_f32 v137, v138, v3
	v_exp_f32_e32 v138, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v223, 0x3fb8aa3b, v4
.Ltmp398:
	.loc	1 1997 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1997:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_2addr_b32 v[4:5], v140 offset1:1
	v_exp_f32_e32 v134, v134
	v_mul_f32_e32 v137, 0x3fb8aa3b, v137
	v_add_nc_u32_e32 v140, 0xc088, v221
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v137, v137
	.loc	1 1997 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1997:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	ds_load_2addr_b32 v[221:222], v140 offset1:1
	v_exp_f32_e32 v140, v223
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cndmask_b32_e64 v134, v134, 0, vcc_lo
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_cndmask_b32_e64 v223, v141, 0, vcc_lo
	v_cndmask_b32_e64 v225, v137, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cndmask_b32_e64 v137, v140, 0, vcc_lo
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v140, v4, v134, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v134, v138, v134
	v_exp_f32_e32 v135, v135
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_fma_mix_f32 v141, v4, v138, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v138, v5, v225, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v134, v137, v134
	v_sub_f32_e32 v136, v136, v3
	v_mul_f32_e32 v139, 0x3fb8aa3b, v139
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cndmask_b32_e64 v4, v135, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	v_exp_f32_e32 v139, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v136, v136
	v_cndmask_b32_e64 v224, v139, 0, vcc_lo
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_fma_mix_f32 v139, v5, v137, neg(0) op_sel_hi:[1,0,0]
.Ltmp399:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v5, v141, 0, v140
.Ltmp400:
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v137, v221, v4, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	1 2001 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2001:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cndmask_b32_e64 v226, v136, 0, vcc_lo
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_fma_mix_f32 v136, v221, v224, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v221, v225, v134
.Ltmp401:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v5, v5, v139, v138
.Ltmp402:
	.loc	1 2005 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_fma_mix_f32 v134, v222, v223, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v135, v222, v226, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v4, v4, v221
.Ltmp403:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v5, v5, v137, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp404:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v4, v224, v4
.Ltmp405:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max3_num_f32 v222, v5, v135, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp406:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v4, v226, v4
.Ltmp407:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_mov_b32_e32 v221, v222
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp408:
	.loc	1 2004 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2004:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_f32_e32 v5, v223, v4
.Ltmp409:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_permlanex16_b32 v221, v221, s22, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp410:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max_num_f32_e32 v4, v221, v221
.Ltmp411:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2007:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_dual_mov_b32 v221, v5 :: v_dual_max_num_f32 v222, v222, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v221, v221, s22, 0xfedcba98
	v_mov_b32_e32 v4, v185
.Ltmp412:
	.loc	1 2015 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2015:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmpx_lt_f32_e32 0, v222
	s_cbranch_execz .LBB6_47
; %bb.55:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 2016 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_div_scale_f32 v4, null, 0x43e00000, 0x43e00000, v222
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v223, v4
	v_fma_f32 v224, -v4, v223, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v223, v224, v223
	v_div_scale_f32 v224, vcc_lo, v222, 0x43e00000, v222
	v_mul_f32_e32 v225, v224, v223
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v226, -v4, v225, v224
	v_fmac_f32_e32 v225, v226, v223
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v225, v224
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v4, v4, v223, v225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v4, v4, 0x43e00000, v222
.Ltmp413:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ] ]
	v_max_num_f32_e32 v4, 0x1f800000, v4
	s_branch .LBB6_47
.Ltmp414:
.LBB6_56:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v3, v180
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v180, v3
	.loc	1 1946 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_add_co_i32 s14, s14, 1
	.loc	1 1946 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s14, 4
	.loc	1 1946 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1946:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_cbranch_scc0 .LBB6_48
	s_branch .LBB6_9
.LBB6_57:
	.loc	1 2071 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_cmp_eq_u32_e32 vcc_lo, 0, v184
	s_and_b32 s1, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_59
; %bb.58:
	.loc	1 2075 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_u64_u32 v[0:1], null, s17, v183, s[18:19]
	.loc	1 2077 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2076 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	1 2077 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	.loc	1 2079 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	global_store_b64 v[0:1], v[180:181], off
.LBB6_59:
	.loc	1 0 24 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 2082 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB6_61
; %bb.60:
	.loc	1 2085 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mad_co_u64_u32 v[0:1], null, s17, v183, s[18:19]
	.loc	1 2087 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v102, v102, v185
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v110, v110, v185
	v_dual_mul_f32 v86, v86, v185 :: v_dual_mul_f32 v103, v103, v185
	v_dual_mul_f32 v70, v70, v185 :: v_dual_mul_f32 v87, v87, v185
	.loc	1 2086 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v5, v118, v185
	v_dual_mul_f32 v118, v126, v185 :: v_dual_mul_f32 v71, v71, v185
	v_mul_f32_e32 v126, v185, v6
	v_dual_mul_f32 v6, v119, v185 :: v_dual_mul_f32 v17, v17, v185
	v_dual_mul_f32 v119, v127, v185 :: v_dual_add_nc_u32 v2, v0, v182
	.loc	1 2087 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[3:4], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v94, v94, v185 :: v_dual_mul_f32 v95, v95, v185
	v_dual_mul_f32 v78, v78, v185 :: v_dual_mul_f32 v79, v79, v185
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v111, v111, v185 :: v_dual_add_nc_u32 v0, 2, v2
	.loc	1 2087 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v3, vcc_lo, s10, v3
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v47, v47, v185
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 2093 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[134:135], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 18, v2
	.loc	1 2087 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s11, v4, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v39, v39, v185 :: v_dual_mul_f32 v62, v62, v185
	v_mul_f32_e32 v63, v63, v185
	.loc	1 2093 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[136:137], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v31, v31, v185 :: v_dual_add_nc_u32 v0, 34, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v134, vcc_lo, v3, v134
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, v4, v135, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[138:139], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v23, v23, v185 :: v_dual_add_nc_u32 v0, 50, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v136, vcc_lo, v3, v136
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v15, v15, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[140:141], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x42, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, v4, v137, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v127, v185, v7
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v138, vcc_lo, v3, v138
	s_wait_loadcnt 0xf
	v_lshlrev_b64_e32 v[142:143], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v7, v120, v185 :: v_dual_add_nc_u32 v0, 0x52, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v139, null, v4, v139, vcc_lo
	v_add_co_u32 v140, vcc_lo, v3, v140
	s_wait_loadcnt 0xe
	v_lshlrev_b64_e32 v[144:145], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x62, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v141, null, v4, v141, vcc_lo
	v_add_co_u32 v142, vcc_lo, v3, v142
	s_wait_loadcnt 0xd
	v_lshlrev_b64_e32 v[146:147], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x72, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v143, null, v4, v143, vcc_lo
	v_add_co_u32 v144, vcc_lo, v3, v144
	s_wait_loadcnt 0xc
	v_lshlrev_b64_e32 v[148:149], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x82, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v145, null, v4, v145, vcc_lo
	v_add_co_u32 v146, vcc_lo, v3, v146
	s_wait_loadcnt 0xb
	v_lshlrev_b64_e32 v[150:151], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x92, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, v4, v147, vcc_lo
	v_add_co_u32 v148, vcc_lo, v3, v148
	s_wait_loadcnt 0xa
	v_lshlrev_b64_e32 v[152:153], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xa2, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, v4, v149, vcc_lo
	v_add_co_u32 v150, vcc_lo, v3, v150
	s_wait_loadcnt 0x9
	v_lshlrev_b64_e32 v[154:155], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xb2, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v151, null, v4, v151, vcc_lo
	v_add_co_u32 v152, vcc_lo, v3, v152
	s_wait_loadcnt 0x8
	v_lshlrev_b64_e32 v[156:157], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xc2, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, v4, v153, vcc_lo
	v_add_co_u32 v154, vcc_lo, v3, v154
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, v4, v155, vcc_lo
	v_add_co_u32 v156, vcc_lo, v3, v156
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v54, v54, v185 :: v_dual_mul_f32 v55, v55, v185
	v_mul_f32_e32 v46, v46, v185
	v_dual_mul_f32 v38, v38, v185 :: v_dual_mul_f32 v81, v81, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v157, null, v4, v157, vcc_lo
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0xb
	global_store_b64 v[134:135], v[5:6], off
	global_store_b64 v[136:137], v[102:103], off
	global_store_b64 v[138:139], v[86:87], off
	global_store_b64 v[140:141], v[70:71], off
	global_store_b64 v[142:143], v[118:119], off
	global_store_b64 v[144:145], v[110:111], off
	global_store_b64 v[146:147], v[94:95], off
	global_store_b64 v[148:149], v[78:79], off
	global_store_b64 v[150:151], v[62:63], off
	global_store_b64 v[152:153], v[54:55], off
	global_store_b64 v[154:155], v[46:47], off
	global_store_b64 v[156:157], v[38:39], off
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v55, v73, v185
	v_mul_f32_e32 v73, v97, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x7
	v_lshlrev_b64_e32 v[158:159], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v65, v65, v185 :: v_dual_add_nc_u32 v0, 0xd2, v2
	v_mul_f32_e32 v49, v49, v185
	v_dual_mul_f32 v25, v25, v185 :: v_dual_mul_f32 v30, v30, v185
	v_mul_f32_e32 v57, v57, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_loadcnt 0x6
	v_lshlrev_b64_e32 v[160:161], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xe2, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v158, vcc_lo, v3, v158
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v159, null, v4, v159, vcc_lo
	s_wait_loadcnt 0x5
	v_lshlrev_b64_e32 v[162:163], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xf2, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v160, vcc_lo, v3, v160
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v161, null, v4, v161, vcc_lo
	s_wait_loadcnt 0x4
	v_lshlrev_b64_e32 v[164:165], 2, v[0:1]
	v_add_nc_u32_e32 v0, 4, v2
	v_add_co_u32 v162, vcc_lo, v3, v162
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v22, v22, v185 :: v_dual_mul_f32 v41, v41, v185
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 20, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v163, null, v4, v163, vcc_lo
	v_add_co_u32 v164, vcc_lo, v3, v164
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v14, v14, v185 :: v_dual_mul_f32 v33, v33, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v165, null, v4, v165, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v39, v89, v185
	v_mul_f32_e32 v89, v185, v9
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x3
	global_store_b64 v[158:159], v[30:31], off
	global_store_b64 v[160:161], v[22:23], off
	global_store_b64 v[162:163], v[14:15], off
	global_store_b64 v[164:165], v[126:127], off
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 36, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[30:31], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 52, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v14, vcc_lo, v3, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, v4, v15, vcc_lo
	v_lshlrev_b64_e32 v[46:47], 2, v[0:1]
	v_add_co_u32 v30, vcc_lo, v3, v30
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v38, v88, v185
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x44, v2
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v88, v185, v8 :: v_dual_mul_f32 v13, v185, v13
	v_mul_f32_e32 v8, v121, v185
	v_mul_f32_e32 v22, v104, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, v4, v31, vcc_lo
	v_add_co_u32 v46, vcc_lo, v3, v46
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v23, v105, v185 :: v_dual_mul_f32 v54, v72, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, v4, v47, vcc_lo
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x3
	global_store_b64 v[5:6], v[7:8], off
	global_store_b64 v[14:15], v[22:23], off
	global_store_b64 v[30:31], v[38:39], off
	global_store_b64 v[46:47], v[54:55], off
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v55, v99, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[62:63], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v67, v67, v185 :: v_dual_add_nc_u32 v0, 0x54, v2
	v_mul_f32_e32 v51, v51, v185
	v_mul_f32_e32 v35, v35, v185
	v_mul_f32_e32 v19, v19, v185
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[71:72], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x64, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v62, vcc_lo, v3, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, v4, v63, vcc_lo
	v_lshlrev_b64_e32 v[86:87], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x74, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v94, vcc_lo, v3, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v95, null, v4, v72, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[102:103], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x84, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v86, vcc_lo, v3, v86
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v87, null, v4, v87, vcc_lo
	v_lshlrev_b64_e32 v[110:111], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x94, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v102, vcc_lo, v3, v102
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v53, v53, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v103, null, v4, v103, vcc_lo
	v_lshlrev_b64_e32 v[118:119], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xa4, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v110, vcc_lo, v3, v110
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v111, null, v4, v111, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[126:127], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xb4, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v118, vcc_lo, v3, v118
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v119, null, v4, v119, vcc_lo
	v_lshlrev_b64_e32 v[134:135], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xc4, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v126, vcc_lo, v3, v126
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v127, null, v4, v127, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[136:137], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xd4, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v134, vcc_lo, v3, v134
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, v4, v135, vcc_lo
	v_lshlrev_b64_e32 v[138:139], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xe4, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v136, vcc_lo, v3, v136
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, v4, v137, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[140:141], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xf4, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v138, vcc_lo, v3, v138
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v139, null, v4, v139, vcc_lo
	v_lshlrev_b64_e32 v[142:143], 2, v[0:1]
	v_add_nc_u32_e32 v0, 6, v2
	v_add_co_u32 v140, vcc_lo, v3, v140
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v70, v128, v185 :: v_dual_mul_f32 v59, v59, v185
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 22, v2
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v32, v32, v185 :: v_dual_mul_f32 v71, v129, v185
	v_mul_f32_e32 v78, v112, v185
	v_dual_mul_f32 v24, v24, v185 :: v_dual_mul_f32 v45, v45, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 38, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v141, null, v4, v141, vcc_lo
	v_add_co_u32 v104, vcc_lo, v3, v142
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v79, v113, v185 :: v_dual_mul_f32 v72, v96, v185
	v_dual_mul_f32 v27, v27, v185 :: v_dual_mul_f32 v16, v16, v185
	v_dual_mul_f32 v37, v37, v185 :: v_dual_mul_f32 v80, v80, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v105, null, v4, v143, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v64, v64, v185
	v_mul_f32_e32 v56, v56, v185
	v_mul_f32_e32 v48, v48, v185
	v_mul_f32_e32 v40, v40, v185
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0xb
	global_store_b64 v[62:63], v[70:71], off
	global_store_b64 v[94:95], v[78:79], off
	global_store_b64 v[86:87], v[72:73], off
	global_store_b64 v[102:103], v[80:81], off
	global_store_b64 v[110:111], v[64:65], off
	global_store_b64 v[118:119], v[56:57], off
	global_store_b64 v[126:127], v[48:49], off
	global_store_b64 v[134:135], v[40:41], off
	global_store_b64 v[136:137], v[32:33], off
	global_store_b64 v[138:139], v[24:25], off
	global_store_b64 v[140:141], v[16:17], off
	global_store_b64 v[104:105], v[88:89], off
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 54, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[24:25], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x46, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v22, vcc_lo, v3, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, v4, v9, vcc_lo
	v_lshlrev_b64_e32 v[32:33], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x56, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v30, vcc_lo, v3, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, v4, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[39:40], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x66, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v24, vcc_lo, v3, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v4, v25, vcc_lo
	v_lshlrev_b64_e32 v[46:47], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x76, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v32, vcc_lo, v3, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v4, v33, vcc_lo
	v_add_co_u32 v48, vcc_lo, v3, v39
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v4, v40, vcc_lo
	v_lshlrev_b64_e32 v[39:40], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x86, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v46, vcc_lo, v3, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, v4, v47, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[62:63], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x96, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v64, vcc_lo, v3, v39
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, v4, v40, vcc_lo
	v_lshlrev_b64_e32 v[39:40], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xa6, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v62, vcc_lo, v3, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, v4, v63, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[70:71], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xb6, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v72, vcc_lo, v3, v39
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, v4, v40, vcc_lo
	v_lshlrev_b64_e32 v[39:40], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xc6, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v70, vcc_lo, v3, v70
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v71, null, v4, v71, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[78:79], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xd6, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v80, vcc_lo, v3, v39
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, v4, v40, vcc_lo
	v_lshlrev_b64_e32 v[39:40], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xe6, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v78, vcc_lo, v3, v78
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v7, v122, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, v4, v79, vcc_lo
	v_lshlrev_b64_e32 v[86:87], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xf6, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v88, vcc_lo, v3, v39
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v8, v123, v185
	v_mul_f32_e32 v14, v106, v185
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[94:95], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v15, v107, v185 :: v_dual_add_nc_u32 v0, 8, v2
	v_mul_f32_e32 v9, v90, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v89, null, v4, v40, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v40, v185, v10
	v_mul_f32_e32 v10, v91, v185
	v_dual_mul_f32 v16, v74, v185 :: v_dual_mul_f32 v17, v75, v185
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x3
	global_store_b64 v[5:6], v[7:8], off
	global_store_b64 v[22:23], v[14:15], off
	global_store_b64 v[30:31], v[9:10], off
	global_store_b64 v[24:25], v[16:17], off
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 24, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v86, vcc_lo, v3, v86
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v87, null, v4, v87, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 40, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v90, vcc_lo, v3, v94
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v4, v95, vcc_lo
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 56, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v5, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v16, vcc_lo, v3, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v4, v9, vcc_lo
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x48, v2
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v38, v130, v185 :: v_dual_mul_f32 v39, v131, v185
	v_mul_f32_e32 v41, v114, v185
	v_mul_f32_e32 v74, v42, v185
	v_mul_f32_e32 v42, v115, v185
	v_mul_f32_e32 v54, v98, v185
	v_mul_f32_e32 v34, v34, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x58, v2
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v56, v82, v185
	v_dual_mul_f32 v26, v26, v185 :: v_dual_mul_f32 v57, v83, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v14, vcc_lo, v3, v14
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v66, v66, v185
	v_mul_f32_e32 v18, v18, v185
	v_mul_f32_e32 v58, v58, v185
	v_dual_mul_f32 v50, v50, v185 :: v_dual_mul_f32 v75, v43, v185
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x7
	global_store_b64 v[32:33], v[38:39], off
	global_store_b64 v[48:49], v[41:42], off
	global_store_b64 v[46:47], v[54:55], off
	global_store_b64 v[64:65], v[56:57], off
	global_store_b64 v[62:63], v[66:67], off
	global_store_b64 v[72:73], v[58:59], off
	global_store_b64 v[70:71], v[50:51], off
	global_store_b64 v[80:81], v[74:75], off
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v41, v185, v11
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, v4, v15, vcc_lo
	v_add_co_u32 v25, vcc_lo, v3, v8
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x3
	global_store_b64 v[78:79], v[34:35], off
	global_store_b64 v[88:89], v[26:27], off
	global_store_b64 v[86:87], v[18:19], off
	global_store_b64 v[90:91], v[40:41], off
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v9, vcc_lo
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x68, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v30, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x78, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v34, vcc_lo, v3, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v4, v9, vcc_lo
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x88, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v40, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v41, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[23:24], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0x98, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v46, vcc_lo, v3, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, v4, v9, vcc_lo
	v_lshlrev_b64_e32 v[48:49], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xa8, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v50, vcc_lo, v3, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v4, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[54:55], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xb8, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v48, vcc_lo, v3, v48
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v49, null, v4, v49, vcc_lo
	v_lshlrev_b64_e32 v[56:57], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xc8, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v54, vcc_lo, v3, v54
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v7, v124, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v55, null, v4, v55, vcc_lo
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xd8, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v56, vcc_lo, v3, v56
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v8, v125, v185
	v_mul_f32_e32 v10, v108, v185
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[62:63], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v11, v109, v185 :: v_dual_add_nc_u32 v0, 0xe8, v2
	v_mul_f32_e32 v18, v92, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, v4, v57, vcc_lo
	v_lshlrev_b64_e32 v[64:65], 2, v[0:1]
	.loc	1 2093 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_nc_u32_e32 v0, 0xf8, v2
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v58, vcc_lo, v3, v58
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v19, v93, v185 :: v_dual_mul_f32 v22, v76, v185
	s_delay_alu instid0(VALU_DEP_3)
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v23, v77, v185
	v_mul_f32_e32 v27, v132, v185
	v_mul_f32_e32 v24, v60, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v4, v59, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v60, v28, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_add_co_u32 v62, vcc_lo, v3, v62
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v28, v133, v185
	v_mul_f32_e32 v32, v116, v185
	v_mul_f32_e32 v38, v100, v185
	v_mul_f32_e32 v42, v84, v185
	v_mul_f32_e32 v9, v68, v185
	v_mul_f32_e32 v33, v117, v185
	v_mul_f32_e32 v39, v101, v185
	v_mul_f32_e32 v43, v85, v185
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x7
	global_store_b64 v[5:6], v[7:8], off
	global_store_b64 v[16:17], v[10:11], off
	global_store_b64 v[14:15], v[18:19], off
	global_store_b64 v[25:26], v[22:23], off
	global_store_b64 v[30:31], v[27:28], off
	global_store_b64 v[34:35], v[32:33], off
	global_store_b64 v[40:41], v[38:39], off
	global_store_b64 v[46:47], v[42:43], off
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v10, v69, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, v4, v63, vcc_lo
	v_add_co_u32 v64, vcc_lo, v3, v64
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_dual_mul_f32 v25, v61, v185 :: v_dual_mul_f32 v52, v52, v185
	v_mul_f32_e32 v44, v44, v185
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, v4, v65, vcc_lo
	v_add_co_u32 v0, vcc_lo, v3, v0
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v36, v36, v185
	v_mul_f32_e32 v2, v20, v185
	v_mul_f32_e32 v12, v185, v12
	.loc	1 2093 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v4, v1, vcc_lo
	.loc	1 2093 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	v_mul_f32_e32 v61, v29, v185
	v_mul_f32_e32 v3, v21, v185
	.loc	1 2093 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2093:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:5 ]
	s_clause 0x7
	global_store_b64 v[50:51], v[9:10], off
	global_store_b64 v[48:49], v[24:25], off
	global_store_b64 v[54:55], v[52:53], off
	global_store_b64 v[56:57], v[44:45], off
	global_store_b64 v[58:59], v[36:37], off
	global_store_b64 v[62:63], v[60:61], off
	global_store_b64 v[64:65], v[2:3], off
	global_store_b64 v[0:1], v[12:13], off
.Ltmp415:
.LBB6_61:
	.loc	1 2160 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2160:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp416:
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
		.amdhsa_next_free_vgpr 237
		.amdhsa_next_free_sgpr 29
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_vgpr, 237
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 29
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10796
; TotalNumSgprs: 31
; NumVgprs: 237
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 237
; Occupancy: 6
; WaveLimiterHint : 0
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
	.loc	1 2171 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2171:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	1 2171 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2171:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB7_10
; %bb.1:
	.loc	1 2176 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2176:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	1 2179 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2179:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2178 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2178:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	1 2179 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2179:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_10
; %bb.2:
	.loc	1 2171 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2171:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	1 2182 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2182:5
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
	.loc	1 2185 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2185:22
	global_load_b32 v3, v[6:7], off
.Ltmp417:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2185:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp418:
	.loc	1 2182 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2182:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp419:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2185:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp420:
	.loc	1 2182 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2182:5
	s_cbranch_scc0 .LBB7_3
; %bb.4:
	.loc	1 2177 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2177:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	1 2187 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5
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
	.loc	1 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 2199 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:13
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
	.loc	1 2187 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	1 2199 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	1 2198 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2198:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	1 2187 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	1 2199 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	1 2198 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2198:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	1 2199 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	1 2187 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5
	s_or_b32 s1, vcc_lo, s1
	.loc	1 2199 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	1 2187 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	1 2198 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2198:44
	global_store_b32 v[5:6], v11, off
	.loc	1 2187 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB7_10
.LBB7_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_8 Depth 2
	.loc	1 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_mov_b32 s0, s7
	s_branch .LBB7_8
.LBB7_7:                                ;   in Loop: Header=BB7_8 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	.loc	1 2196 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2196:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	1 2190 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	1 2196 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2196:24
	global_load_b32 v15, v[15:16], off
	.loc	1 2195 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2195:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	1 2190 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 2196 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2196:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	1 2190 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:9
	s_cbranch_scc1 .LBB7_5
.LBB7_8:                                ;   Parent Loop BB7_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 2194 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	1 2194 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:17
	s_mov_b32 s2, exec_lo
	.loc	1 2194 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	1 2194 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:17
	s_cbranch_execz .LBB7_7
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=2
	.loc	1 2194 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:41
	global_load_b32 v14, v[5:6], off
	.loc	1 2194 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp421:
	.loc	3 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	3 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB7_7
.Ltmp422:
.LBB7_10:
	.loc	1 2201 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:1
	s_endpgm
.Ltmp423:
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
	.type	__hip_cuid_885084fbc1a7b25e,@object ; @__hip_cuid_885084fbc1a7b25e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_885084fbc1a7b25e
__hip_cuid_885084fbc1a7b25e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_885084fbc1a7b25e, 1

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
	.byte	1                               ; Abbrev [1] 0xc:0x9f8 DW_TAG_compile_unit
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
	.byte	1                               ; DW_AT_call_file
	.short	1342                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4e:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x58:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp4-.Ltmp3                   ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x67:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x71:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x85:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp13-.Ltmp12                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x93:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_low_pc
	.long	.Ltmp16-.Ltmp15                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa1:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	4                               ; DW_AT_low_pc
	.long	.Ltmp17-.Ltmp16                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xaf:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_low_pc
	.long	.Ltmp20-.Ltmp19                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xbd:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	6                               ; DW_AT_low_pc
	.long	.Ltmp23-.Ltmp22                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xcb:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	7                               ; DW_AT_low_pc
	.long	.Ltmp24-.Ltmp23                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xd9:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	5                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xe3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_low_pc
	.long	.Ltmp27-.Ltmp26                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xf1:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_low_pc
	.long	.Ltmp33-.Ltmp32                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xff:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_low_pc
	.long	.Ltmp34-.Ltmp33                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x10d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	11                              ; DW_AT_low_pc
	.long	.Ltmp36-.Ltmp35                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x11b:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	12                              ; DW_AT_low_pc
	.long	.Ltmp38-.Ltmp37                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x129:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	13                              ; DW_AT_low_pc
	.long	.Ltmp39-.Ltmp38                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x137:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	14                              ; DW_AT_low_pc
	.long	.Ltmp41-.Ltmp40                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x145:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	15                              ; DW_AT_low_pc
	.long	.Ltmp42-.Ltmp41                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x153:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	16                              ; DW_AT_low_pc
	.long	.Ltmp43-.Ltmp42                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x161:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1263                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x16f:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x17d:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	17                              ; DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x18d:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1090                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x19b:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1a9:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x1b9:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	6                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1172                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x1c3:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	7                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1185                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x1cd:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_AT_low_pc
	.long	.Ltmp70-.Ltmp69                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1186                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1db:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_AT_low_pc
	.long	.Ltmp70-.Ltmp69                 ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x1ea:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	8                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1198                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1f4:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	20                              ; DW_AT_low_pc
	.long	.Ltmp73-.Ltmp72                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1187                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x202:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	9                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1219                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x20c:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	21                              ; DW_AT_low_pc
	.long	.Ltmp83-.Ltmp82                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x21a:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	21                              ; DW_AT_low_pc
	.long	.Ltmp83-.Ltmp82                 ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x229:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	22                              ; DW_AT_low_pc
	.long	.Ltmp85-.Ltmp84                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1209                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x237:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	22                              ; DW_AT_low_pc
	.long	.Ltmp85-.Ltmp84                 ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x246:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	23                              ; DW_AT_low_pc
	.long	.Ltmp86-.Ltmp85                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x254:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	24                              ; DW_AT_low_pc
	.long	.Ltmp88-.Ltmp87                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
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
	.byte	1                               ; DW_AT_call_file
	.short	1423                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x279:0xa DW_TAG_inlined_subroutine
	.long	612                             ; DW_AT_abstract_origin
	.byte	10                              ; DW_AT_ranges
	.byte	5                               ; DW_AT_call_file
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
	.byte	1                               ; DW_AT_call_file
	.short	1471                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x296:0xa DW_TAG_inlined_subroutine
	.long	612                             ; DW_AT_abstract_origin
	.byte	11                              ; DW_AT_ranges
	.byte	5                               ; DW_AT_call_file
	.short	299                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x2a1:0x24 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	12                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1492                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x2ab:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	12                              ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x2b5:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	27                              ; DW_AT_low_pc
	.long	.Ltmp100-.Ltmp99                ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x2c5:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	13                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1493                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x2cf:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	13                              ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x2da:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_low_pc
	.long	.Ltmp104-.Ltmp103               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1485                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x2e8:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	29                              ; DW_AT_low_pc
	.long	.Ltmp107-.Ltmp106               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1487                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x2f6:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	30                              ; DW_AT_low_pc
	.long	.Ltmp108-.Ltmp107               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1489                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x304:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	31                              ; DW_AT_low_pc
	.long	.Ltmp109-.Ltmp108               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1491                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x312:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	32                              ; DW_AT_low_pc
	.long	.Ltmp111-.Ltmp110               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1492                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x320:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	14                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1494                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x32a:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	14                              ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x335:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	33                              ; DW_AT_low_pc
	.long	.Ltmp114-.Ltmp113               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1493                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x343:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	15                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1495                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x34d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	15                              ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x358:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_low_pc
	.long	.Ltmp117-.Ltmp116               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1494                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x366:0x15 DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	16                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1496                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x370:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	16                              ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x37b:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_low_pc
	.long	.Ltmp120-.Ltmp119               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1495                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x389:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_low_pc
	.long	.Ltmp122-.Ltmp121               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
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
	.byte	1                               ; DW_AT_call_file
	.short	1585                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x3ab:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	18                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3b5:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_low_pc
	.long	.Ltmp130-.Ltmp129               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x3c4:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x3ce:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	20                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3d8:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_low_pc
	.long	.Ltmp142-.Ltmp141               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3e6:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_low_pc
	.long	.Ltmp145-.Ltmp144               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	999                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x3f4:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	21                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x3fe:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_low_pc
	.long	.Ltmp148-.Ltmp147               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1004                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x40c:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	22                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x416:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_low_pc
	.long	.Ltmp152-.Ltmp151               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x424:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_low_pc
	.long	.Ltmp159-.Ltmp158               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x432:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_low_pc
	.long	.Ltmp160-.Ltmp159               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x440:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_low_pc
	.long	.Ltmp163-.Ltmp162               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x44e:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_low_pc
	.long	.Ltmp171-.Ltmp170               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x45c:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_low_pc
	.long	.Ltmp172-.Ltmp171               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x46a:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_low_pc
	.long	.Ltmp175-.Ltmp174               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x478:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_low_pc
	.long	.Ltmp177-.Ltmp176               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x486:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_low_pc
	.long	.Ltmp178-.Ltmp177               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x494:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_low_pc
	.long	.Ltmp182-.Ltmp181               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4a2:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_low_pc
	.long	.Ltmp184-.Ltmp183               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4b0:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_low_pc
	.long	.Ltmp185-.Ltmp184               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4be:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp189-.Ltmp188               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1263                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4cc:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp189-.Ltmp188               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4da:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp189-.Ltmp188               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x4ea:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp191-.Ltmp190               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1090                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4f8:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp191-.Ltmp190               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x506:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp191-.Ltmp190               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x516:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	23                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1172                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x520:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	24                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1185                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x52a:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_low_pc
	.long	.Ltmp214-.Ltmp213               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1186                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x538:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_low_pc
	.long	.Ltmp214-.Ltmp213               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x547:0xa DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	25                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1198                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x551:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	57                              ; DW_AT_low_pc
	.long	.Ltmp217-.Ltmp216               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1187                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x55f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	26                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1219                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x569:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_low_pc
	.long	.Ltmp227-.Ltmp226               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x577:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_low_pc
	.long	.Ltmp227-.Ltmp226               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x586:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_low_pc
	.long	.Ltmp229-.Ltmp228               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1209                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x594:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_low_pc
	.long	.Ltmp229-.Ltmp228               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x5a3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	60                              ; DW_AT_low_pc
	.long	.Ltmp230-.Ltmp229               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x5b1:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	61                              ; DW_AT_low_pc
	.long	.Ltmp232-.Ltmp231               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
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
	.byte	1                               ; DW_AT_call_file
	.short	1662                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x5d4:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	63                              ; DW_AT_low_pc
	.long	.Ltmp242-.Ltmp241               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1671                            ; DW_AT_call_line
	.byte	34                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x5e3:0x2 DW_TAG_subprogram
	.byte	17                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x5e5:0x2 DW_TAG_subprogram
	.byte	18                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x5e7:0x1f8 DW_TAG_subprogram
	.byte	64                              ; DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	25                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x5ee:0x1f0 DW_TAG_inlined_subroutine
	.long	1507                            ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2123                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5f8:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	29                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x602:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	65                              ; DW_AT_low_pc
	.long	.Ltmp250-.Ltmp249               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x611:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	30                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x61b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	31                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x625:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	32                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x62f:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	66                              ; DW_AT_low_pc
	.long	.Ltmp265-.Ltmp264               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x63d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	67                              ; DW_AT_low_pc
	.long	.Ltmp266-.Ltmp265               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x64b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	33                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x655:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	68                              ; DW_AT_low_pc
	.long	.Ltmp272-.Ltmp271               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x663:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	69                              ; DW_AT_low_pc
	.long	.Ltmp273-.Ltmp272               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x671:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	70                              ; DW_AT_low_pc
	.long	.Ltmp274-.Ltmp273               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x67f:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	71                              ; DW_AT_low_pc
	.long	.Ltmp277-.Ltmp276               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x68d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	72                              ; DW_AT_low_pc
	.long	.Ltmp279-.Ltmp278               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x69b:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	73                              ; DW_AT_low_pc
	.long	.Ltmp281-.Ltmp280               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6a9:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	74                              ; DW_AT_low_pc
	.long	.Ltmp282-.Ltmp281               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6b7:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	75                              ; DW_AT_low_pc
	.long	.Ltmp284-.Ltmp283               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6c5:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	76                              ; DW_AT_low_pc
	.long	.Ltmp285-.Ltmp284               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6d3:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	77                              ; DW_AT_low_pc
	.long	.Ltmp286-.Ltmp285               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6e1:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	78                              ; DW_AT_low_pc
	.long	.Ltmp288-.Ltmp287               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6ef:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	79                              ; DW_AT_low_pc
	.long	.Ltmp289-.Ltmp288               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6fd:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	80                              ; DW_AT_low_pc
	.long	.Ltmp290-.Ltmp289               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x70b:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp292-.Ltmp291               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2054                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x719:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp292-.Ltmp291               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x727:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp292-.Ltmp291               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x737:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1876                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x745:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x753:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x763:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1926                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x76d:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x777:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x783:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1989                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x78d:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp305-.Ltmp304               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1990                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x79b:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp306-.Ltmp305               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1991                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7a9:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2012                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7b3:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp316-.Ltmp315               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2013                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7c1:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	86                              ; DW_AT_low_pc
	.long	.Ltmp318-.Ltmp317               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2007                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7cf:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	87                              ; DW_AT_low_pc
	.long	.Ltmp320-.Ltmp319               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2016                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x7df:0x2 DW_TAG_subprogram
	.byte	19                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x7e1:0x202 DW_TAG_subprogram
	.byte	88                              ; DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	26                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x7e8:0x1fa DW_TAG_inlined_subroutine
	.long	2015                            ; DW_AT_abstract_origin
	.byte	37                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2157                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7f2:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7fc:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	89                              ; DW_AT_low_pc
	.long	.Ltmp329-.Ltmp328               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x80b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x815:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x81f:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x829:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	90                              ; DW_AT_low_pc
	.long	.Ltmp345-.Ltmp344               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x837:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	91                              ; DW_AT_low_pc
	.long	.Ltmp346-.Ltmp345               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x845:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x84f:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	92                              ; DW_AT_low_pc
	.long	.Ltmp350-.Ltmp349               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x85d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	93                              ; DW_AT_low_pc
	.long	.Ltmp352-.Ltmp351               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x86b:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	94                              ; DW_AT_low_pc
	.long	.Ltmp354-.Ltmp353               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x879:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	95                              ; DW_AT_low_pc
	.long	.Ltmp356-.Ltmp355               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x887:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	96                              ; DW_AT_low_pc
	.long	.Ltmp359-.Ltmp358               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x895:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	97                              ; DW_AT_low_pc
	.long	.Ltmp364-.Ltmp363               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8a3:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	98                              ; DW_AT_low_pc
	.long	.Ltmp366-.Ltmp365               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8b1:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	99                              ; DW_AT_low_pc
	.long	.Ltmp369-.Ltmp368               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8bf:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	100                             ; DW_AT_low_pc
	.long	.Ltmp373-.Ltmp372               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8cd:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	101                             ; DW_AT_low_pc
	.long	.Ltmp375-.Ltmp374               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8db:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	102                             ; DW_AT_low_pc
	.long	.Ltmp379-.Ltmp378               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8e9:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	103                             ; DW_AT_low_pc
	.long	.Ltmp381-.Ltmp380               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8f7:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	104                             ; DW_AT_low_pc
	.long	.Ltmp382-.Ltmp381               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x905:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp384-.Ltmp383               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2054                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x913:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp384-.Ltmp383               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x921:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp384-.Ltmp383               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x931:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp386-.Ltmp385               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1876                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x93f:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp386-.Ltmp385               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x94d:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp386-.Ltmp385               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x95d:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1926                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x967:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x971:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x97d:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1989                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x987:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp397-.Ltmp396               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1990                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x995:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp398-.Ltmp397               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1991                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9a3:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2012                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9ad:0xa DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2013                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9b7:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp411-.Ltmp410               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2013                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9c5:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	110                             ; DW_AT_low_pc
	.long	.Ltmp412-.Ltmp411               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2007                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9d3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp414-.Ltmp413               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2016                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0x9e3:0x20 DW_TAG_subprogram
	.byte	112                             ; DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	27                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0x9ea:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2185                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9f4:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	113                             ; DW_AT_low_pc
	.long	.Ltmp422-.Ltmp421               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2194                            ; DW_AT_call_line
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
	.long	48                              ; Offset entry count
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
	.uleb128 .Ltmp142-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp143-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp166-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp167-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp168-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp169-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp172-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp173-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp175-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp176-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp179-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp180-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp186-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp187-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp235-.Lfunc_begin0         ;   ending offset
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
	.uleb128 .Ltmp143-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp144-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp146-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp147-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp150-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp151-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges20:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp135-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp136-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp156-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp157-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp165-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp166-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp180-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp181-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges21:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp145-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp146-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp149-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp150-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp152-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp153-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp154-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp155-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp161-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp162-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges22:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp148-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp149-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp153-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp154-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp155-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp156-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp163-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp164-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp173-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp174-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges23:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp192-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp193-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp194-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp195-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp196-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp197-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp198-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp199-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp200-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp201-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp202-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp203-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp204-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp205-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp206-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp207-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges24:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp208-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp209-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp210-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp211-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp212-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp213-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges25:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp214-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp215-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp233-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp234-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges26:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp218-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp219-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp220-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp221-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp222-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp223-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp224-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp225-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges27:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp237-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp238-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp239-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp240-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges28:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp244-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp245-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp246-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp247-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp248-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges29:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp249-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp250-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp251-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp255-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp256-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp257-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp260-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp261-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges30:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp257-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp258-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp259-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp260-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp268-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges31:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp269-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp282-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp283-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges32:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp262-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp263-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp276-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp279-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp280-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp286-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp287-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges33:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp266-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp268-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp269-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp271-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp278-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges34:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp295-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp296-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp297-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp298-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges35:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp299-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp300-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp301-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp302-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp303-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp304-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges36:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp307-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp308-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp309-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp310-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp311-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp312-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp313-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp314-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges37:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp324-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp334-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp338-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp339-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp342-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp343-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp344-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp366-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp367-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp369-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp372-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp373-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp374-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp415-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges38:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp328-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp329-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp330-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp333-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp334-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp339-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges39:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp347-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp348-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp352-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp353-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp354-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp355-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp356-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp357-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges40:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp338-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp357-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp358-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp359-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp360-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp361-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp362-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp367-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp368-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges41:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp360-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp361-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp362-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp363-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp364-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp365-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp376-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp377-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges42:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp346-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp347-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp348-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp349-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp350-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp351-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges43:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp387-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp388-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp389-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp390-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges44:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp391-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp392-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp393-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp394-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp395-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp396-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges45:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp399-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp400-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp401-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp402-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp403-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp404-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp405-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp406-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges46:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp407-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp408-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp409-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp410-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges47:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp417-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp418-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp419-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp420-.Lfunc_begin0         ;   ending offset
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
	.asciz	"/home/kaden/ClaudeCode/warpfront/wt-fapkt/kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" ; string offset=112 ; /home/kaden/ClaudeCode/warpfront/wt-fapkt/kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip
.Linfo_string2:
	.asciz	"/home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt" ; string offset=201 ; /home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt
.Linfo_string3:
	.asciz	"fa2_stageb_nbody<false>"       ; string offset=268 ; fa2_stageb_nbody<false>
.Linfo_string4:
	.asciz	"__lane_id"                     ; string offset=292 ; __lane_id
.Linfo_string5:
	.asciz	"__shfl_xor"                    ; string offset=302 ; __shfl_xor
.Linfo_string6:
	.asciz	"max"                           ; string offset=313 ; max
.Linfo_string7:
	.asciz	"min"                           ; string offset=317 ; min
.Linfo_string8:
	.asciz	"__work_group_barrier"          ; string offset=321 ; __work_group_barrier
.Linfo_string9:
	.asciz	"__barrier"                     ; string offset=342 ; __barrier
.Linfo_string10:
	.asciz	"__syncthreads"                 ; string offset=352 ; __syncthreads
.Linfo_string11:
	.asciz	"fa2_scale_n"                   ; string offset=366 ; fa2_scale_n
.Linfo_string12:
	.asciz	"fmaxf"                         ; string offset=378 ; fmaxf
.Linfo_string13:
	.asciz	"__hip_get_thread_idx_x"        ; string offset=384 ; __hip_get_thread_idx_x
.Linfo_string14:
	.asciz	"__get_x"                       ; string offset=407 ; __get_x
.Linfo_string15:
	.asciz	"fa2_stageb_nbody<true>"        ; string offset=415 ; fa2_stageb_nbody<true>
.Linfo_string16:
	.asciz	"__expf"                        ; string offset=438 ; __expf
.Linfo_string17:
	.asciz	"fa2_stageb_packet_body<false>" ; string offset=445 ; fa2_stageb_packet_body<false>
.Linfo_string18:
	.asciz	"fa2_pkt_xor16"                 ; string offset=475 ; fa2_pkt_xor16
.Linfo_string19:
	.asciz	"fa2_stageb_packet_body<true>"  ; string offset=489 ; fa2_stageb_packet_body<true>
.Linfo_string20:
	.asciz	"attention_fp8_e4m3_fa2_gqa_gfx1201" ; string offset=518 ; attention_fp8_e4m3_fa2_gqa_gfx1201
.Linfo_string21:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201" ; string offset=553 ; attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
.Linfo_string22:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201" ; string offset=601 ; attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
.Linfo_string23:
	.asciz	"attention_fp8_e4m3_fa2_gqa_partial_gfx1201" ; string offset=649 ; attention_fp8_e4m3_fa2_gqa_partial_gfx1201
.Linfo_string24:
	.asciz	"attention_fp8_e4m3_fa2_gqa_merge_gfx1201" ; string offset=692 ; attention_fp8_e4m3_fa2_gqa_merge_gfx1201
.Linfo_string25:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_gfx1201" ; string offset=733 ; attention_fp8_e4m3_fa2_gqa_packet_gfx1201
.Linfo_string26:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201" ; string offset=775 ; attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
.Linfo_string27:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201" ; string offset=825 ; attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
	.quad	.Ltmp144
	.quad	.Ltmp147
	.quad	.Ltmp151
	.quad	.Ltmp158
	.quad	.Ltmp159
	.quad	.Ltmp162
	.quad	.Ltmp170
	.quad	.Ltmp171
	.quad	.Ltmp174
	.quad	.Ltmp176
	.quad	.Ltmp177
	.quad	.Ltmp181
	.quad	.Ltmp183
	.quad	.Ltmp184
	.quad	.Ltmp188
	.quad	.Ltmp190
	.quad	.Ltmp213
	.quad	.Ltmp216
	.quad	.Ltmp226
	.quad	.Ltmp228
	.quad	.Ltmp229
	.quad	.Ltmp231
	.quad	.Lfunc_begin4
	.quad	.Ltmp241
	.quad	.Lfunc_begin5
	.quad	.Ltmp249
	.quad	.Ltmp264
	.quad	.Ltmp265
	.quad	.Ltmp271
	.quad	.Ltmp272
	.quad	.Ltmp273
	.quad	.Ltmp276
	.quad	.Ltmp278
	.quad	.Ltmp280
	.quad	.Ltmp281
	.quad	.Ltmp283
	.quad	.Ltmp284
	.quad	.Ltmp285
	.quad	.Ltmp287
	.quad	.Ltmp288
	.quad	.Ltmp289
	.quad	.Ltmp291
	.quad	.Ltmp293
	.quad	.Ltmp304
	.quad	.Ltmp305
	.quad	.Ltmp315
	.quad	.Ltmp317
	.quad	.Ltmp319
	.quad	.Lfunc_begin6
	.quad	.Ltmp328
	.quad	.Ltmp344
	.quad	.Ltmp345
	.quad	.Ltmp349
	.quad	.Ltmp351
	.quad	.Ltmp353
	.quad	.Ltmp355
	.quad	.Ltmp358
	.quad	.Ltmp363
	.quad	.Ltmp365
	.quad	.Ltmp368
	.quad	.Ltmp372
	.quad	.Ltmp374
	.quad	.Ltmp378
	.quad	.Ltmp380
	.quad	.Ltmp381
	.quad	.Ltmp383
	.quad	.Ltmp385
	.quad	.Ltmp396
	.quad	.Ltmp397
	.quad	.Ltmp410
	.quad	.Ltmp411
	.quad	.Ltmp413
	.quad	.Lfunc_begin7
	.quad	.Ltmp421
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_885084fbc1a7b25e
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
    .max_flat_workgroup_size: 256
    .name:           attention_fp8_e4m3_fa2_gqa_packet_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     237
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
    .max_flat_workgroup_size: 256
    .name:           attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     237
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
