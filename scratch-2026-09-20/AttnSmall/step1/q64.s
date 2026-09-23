	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_gfx1201:     ; @attention_fp8_e4m3_fa2_gqa_gfx1201
.Lfunc_begin0:
	.file	0 "/home/kaden/ClaudeCode/warpfront/wt-attnsmall" "kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0x11d1a80c7a7e46a78a3be04958999a56
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
	.loc	0 2183 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:17
	s_load_b128 s[4:7], s[0:1], 0x30
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	0 2183 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_85
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2186 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2186:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB5_85
; %bb.2:
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_load_b32 s5, s[0:1], 0x40
	.loc	0 2190 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:30
	s_lshr_b32 s2, ttmp7, 8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, 0xfffe00
	.loc	0 2190 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:41
	s_cmp_gt_i32 s7, 0x200
	.loc	0 2190 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:30
	s_cselect_b32 s2, s2, 0
	.loc	0 2191 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2191:22
	s_cmp_le_i32 s7, s2
	s_cbranch_scc1 .LBB5_85
; %bb.3:
	.loc	0 2193 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2193:46
	s_sub_co_i32 s4, s7, s2
	.loc	0 2194 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2194:29
	s_lshl_b32 s6, ttmp9, 6
.Ltmp241:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2193:26 ]
	s_min_i32 s7, s4, 0x200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
.Ltmp242:
	.loc	0 2195 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2195:23
	s_mul_i32 s4, s7, 6
	.loc	0 2195 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2195:18
	s_cmp_ge_i32 s6, s4
	s_cbranch_scc1 .LBB5_85
; %bb.4:
.Ltmp243:
	.loc	0 1767 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	0 1769 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1769:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_and_b32_e32 v4, 15, v0
.Ltmp244:
	.loc	0 2183 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x8
	s_load_b64 s[18:19], s[0:1], 0x28
.Ltmp245:
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v88, s5 :: v_dual_lshlrev_b32 v87, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1778 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v2, v87, v4
	.loc	0 1778 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_nc_u32_e32 v5, s6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1779 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mul_hi_i32 v2, 0x2aaaaaab, v5
	.loc	0 1782 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1782:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmp_le_i32_e64 s1, s4, v5
	v_cmp_gt_i32_e64 s0, s4, v5
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshrrev_b32_e32 v3, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v3, v2, v3
	.loc	0 1781 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mul_lo_u32 v2, v3, 6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v2, v5, v2
	.loc	0 1781 31 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[81:82], null, s3, 6, v[2:3]
	.loc	0 1780 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1780:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_nc_u32_e32 v82, s2, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, v82, 24, v[81:82]
	.loc	0 1832 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1832:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s16, s0
	s_cbranch_execz .LBB5_6
; %bb.5:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_mov_b32_e32 v3, 0
	s_mul_i32 s20, s7, 0x1800
	s_mov_b32 s21, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_add_nc_u64 s[20:21], s[8:9], s[20:21]
	v_lshlrev_b64_e32 v[5:6], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s20, v5
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	.loc	0 1835 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1835:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	global_load_b32 v3, v[5:6], off
	.loc	0 1987 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1987:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v88, s5, v3
.LBB5_6:
	.loc	0 0 47 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:47
	s_or_b32 exec_lo, exec_lo, s16
	.loc	0 1841 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_or_b32 s5, s6, 63
	.loc	0 1841 60 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_co_i32 s7, s4, -1
	.loc	0 1840 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1840:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mul_hi_i32 s16, s6, 0x2aaaaaab
.Ltmp246:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s7, s5, s7
	v_dual_mov_b32 v6, -1 :: v_dual_and_b32 v3, 31, v0
.Ltmp247:
	.loc	0 1842 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_i32 s6, s7, 0x2aaaaaab
	.loc	0 1840 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1840:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_lshr_b32 s7, s16, 31
	.loc	0 1842 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s17, s6, 31
	.loc	0 1840 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1840:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_co_i32 s16, s16, s7
	.loc	0 1842 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_co_i32 s6, s6, s17
	v_bfrev_b32_e32 v5, -2
	.loc	0 1842 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s6, s6, s16
	.loc	0 1845 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1845:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_ge_i32_e32 vcc_lo, s6, v3
	s_lshr_b32 s6, ttmp7, 16
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB5_8
; %bb.7:
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	v_add3_u32 v5, s2, s16, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s18, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, vcc_lo
	.loc	0 1847 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v6, v5
.LBB5_8:
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
.Ltmp248:
	v_mbcnt_lo_u32_b32 v7, -1, 0
	v_dual_mov_b32 v14, 0x6020400 :: v_dual_lshlrev_b32 v11, 3, v3
	v_and_or_b32 v89, v87, 48, v4
	v_lshrrev_b32_e32 v4, 2, v4
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp249:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_xor_b32_e32 v8, 16, v7
.Ltmp250:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_xor_b32_e32 v22, 4, v7
.Ltmp251:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_xor_b32_e32 v23, 2, v7
.Ltmp252:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_xor_b32_e32 v24, 1, v7
	v_ashrrev_i32_e32 v83, 31, v82
.Ltmp253:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
.Ltmp254:
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshlrev_b32_e32 v9, 11, v1
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v12, 3, v0
.Ltmp255:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v7, v8 :: v_dual_lshlrev_b32 v17, 8, v2
.Ltmp256:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_xor_b32_e32 v2, 8, v7
.Ltmp257:
	.loc	0 1770 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1770:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshrrev_b32_e32 v10, 4, v3
	v_lshl_or_b32 v4, v12, 6, v4
.Ltmp258:
	.loc	0 2188 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2188:46
	s_and_b32 s20, s6, 1
	v_dual_mov_b32 v15, 0x5040100 :: v_dual_lshlrev_b32 v8, 2, v8
.Ltmp259:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_and_b32_e32 v13, 1, v0
	v_or_b32_e32 v28, 0x108, v4
.Ltmp260:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v16, v8, v6
.Ltmp261:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v8, v8, v5
.Ltmp262:
	.loc	1 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v7, v2, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v13
	v_lshlrev_b32_e32 v19, 4, v3
.Ltmp263:
	.loc	0 1864 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cmp_lt_i32 s5, s4
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s7, 0
	.loc	0 1864 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cselect_b32 s21, -1, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v92, 0x3070105, v14, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v12
	v_and_b32_e32 v18, 16, v0
	s_lshl_b32 s6, s3, 8
	v_or_b32_e32 v29, 12, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[12:13], s[6:7]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v93, 0x3020706, v15, vcc_lo
.Ltmp264:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v22
.Ltmp265:
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_nc_u32_e32 v21, 0, v9
	v_or_b32_e32 v30, 0x10c, v4
	v_or_b32_e32 v31, 16, v4
	v_or_b32_e32 v32, 0x110, v4
.Ltmp266:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v7, v22, vcc_lo
.Ltmp267:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v23
	v_lshlrev_b32_e32 v20, 4, v0
.Ltmp268:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v16
.Ltmp269:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v8
.Ltmp270:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v23, v7, v23 :: v_dual_lshlrev_b32 v14, 2, v14
.Ltmp271:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v24
.Ltmp272:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v8, 3, v12
	v_dual_mov_b32 v151, 0xff800000 :: v_dual_add_nc_u32 v96, v21, v11
.Ltmp273:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v7, v24, vcc_lo
.Ltmp274:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v16, v2, v6
.Ltmp275:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v25, v2, v5
.Ltmp276:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_lshlrev_b32_e32 v98, 2, v23
	v_lshlrev_b64_e32 v[2:3], 2, v[82:83]
.Ltmp277:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_lshlrev_b32_e32 v110, 2, v7
	v_or_b32_e32 v7, 0x114, v4
	v_add_co_u32 v83, s5, s22, v11
	v_xad_u32 v97, 0x120, v11, v21
	v_add_co_u32 v85, vcc_lo, s18, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v86, null, s19, v3, vcc_lo
	v_xor_b32_e32 v3, v28, v8
	v_xad_u32 v99, 0x124, v11, v21
	v_xad_u32 v100, 0x240, v11, v21
	v_xad_u32 v101, 0x244, v11, v21
	v_xad_u32 v102, 0x360, v11, v21
	v_lshl_add_u32 v113, v3, 2, v21
	v_xad_u32 v103, 0x364, v11, v21
.Ltmp278:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v16
.Ltmp279:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v25
	v_xad_u32 v104, 0x520, v11, v21
	v_xad_u32 v105, 0x524, v11, v21
	v_xad_u32 v106, 0x640, v11, v21
.Ltmp280:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v27, v14, v6
.Ltmp281:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v14, v14, v5
	v_xad_u32 v107, 0x644, v11, v21
	v_xad_u32 v108, 0x760, v11, v21
	v_xad_u32 v109, 0x764, v11, v21
	v_lshlrev_b32_e32 v11, 2, v4
	v_or_b32_e32 v33, 20, v4
	v_dual_mov_b32 v149, 1.0 :: v_dual_lshlrev_b32 v12, 5, v12
	v_lshlrev_b32_e32 v91, 3, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v84, null, s23, 0, s5
	v_add_nc_u32_e32 v94, 0, v18
	v_add_nc_u32_e32 v95, 0, v19
	v_or_b32_e32 v15, 4, v10
	v_or_b32_e32 v18, 12, v10
	v_or_b32_e32 v22, 16, v10
	v_or_b32_e32 v16, 20, v10
	v_or_b32_e32 v25, 24, v10
	v_or_b32_e32 v26, 28, v10
.Ltmp282:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v27
.Ltmp283:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v14
	v_or_b32_e32 v27, 8, v4
	v_add3_u32 v111, v21, v11, v12
.Ltmp284:
	.loc	0 1865 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_nc_u64 s[16:17], s[10:11], s[6:7]
.Ltmp285:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v14, v98, v6
.Ltmp286:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v23, v98, v5
	s_lshl_b32 s6, s3, 1
	v_cmp_eq_u32_e64 s3, 0, v10
	v_cmp_ne_u32_e64 s4, 0, v10
	v_dual_mov_b32 v150, 0 :: v_dual_lshlrev_b32 v145, 3, v25
	v_cmp_gt_u32_e64 s2, 64, v0
	v_lshl_add_u32 v90, v0, 1, 0
.Ltmp287:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v14
.Ltmp288:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v23
	v_or_b32_e32 v23, 0x118, v4
	v_or_b32_e32 v14, 24, v4
.Ltmp289:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v34, v110, v6
.Ltmp290:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v35, v110, v5
.Ltmp291:
	.loc	2 1334 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v5, v35
	v_or_b32_e32 v5, 40, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp292:
	.loc	0 1862 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_readfirstlane_b32 s19, v3
	v_or_b32_e32 v3, 0x11c, v4
	v_xor_b32_e32 v3, v3, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v123, v3, 2, v21
	v_or_b32_e32 v3, 0x12c, v4
	v_xor_b32_e32 v3, v3, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v127, v3, 2, v21
	v_or_b32_e32 v3, 56, v4
	v_xor_b32_e32 v3, v3, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v132, v3, 2, v21
	v_mov_b32_e32 v3, v1
	v_xor_b32_e32 v2, v27, v8
	v_xor_b32_e32 v5, v5, v8
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v11, v29, v8
	v_xor_b32_e32 v12, v30, v8
	v_lshl_add_u32 v112, v2, 2, v21
.Ltmp293:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max_i32_e32 v2, v6, v34
	v_or_b32_e32 v6, 0x128, v4
	v_lshl_add_u32 v124, v5, 2, v21
	v_or_b32_e32 v5, 48, v4
	v_lshl_add_u32 v119, v7, 2, v21
.Ltmp294:
	.loc	0 1861 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_readfirstlane_b32 s18, v2
	v_or_b32_e32 v2, 28, v4
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v7, v23, v8
	v_xor_b32_e32 v5, v5, v8
	v_xor_b32_e32 v27, v31, v8
	v_xor_b32_e32 v2, v2, v8
	v_lshl_add_u32 v125, v6, 2, v21
	v_or_b32_e32 v6, 0x130, v4
	v_lshl_add_u32 v121, v7, 2, v21
	v_or_b32_e32 v7, 52, v4
	v_lshl_add_u32 v122, v2, 2, v21
	v_or_b32_e32 v2, 44, v4
	v_xor_b32_e32 v6, v6, v8
	v_lshl_add_u32 v128, v5, 2, v21
	v_or_b32_e32 v5, 0x138, v4
	v_xor_b32_e32 v7, v7, v8
	v_xor_b32_e32 v2, v2, v8
	v_lshl_add_u32 v129, v6, 2, v21
	v_or_b32_e32 v6, 60, v4
	v_xor_b32_e32 v5, v5, v8
	v_xor_b32_e32 v28, v32, v8
	v_lshl_add_u32 v126, v2, 2, v21
	v_or_b32_e32 v2, 0x134, v4
	v_or_b32_e32 v4, 0x13c, v4
	v_xor_b32_e32 v6, v6, v8
	v_xor_b32_e32 v29, v33, v8
	v_xor_b32_e32 v14, v14, v8
	v_xor_b32_e32 v2, v2, v8
	v_xor_b32_e32 v4, v4, v8
	v_lshl_add_u32 v130, v7, 2, v21
	v_lshl_add_u32 v133, v5, 2, v21
	v_lshl_add_u32 v134, v6, 2, v21
	v_lshl_add_u32 v131, v2, 2, v21
	v_mov_b32_e32 v2, v1
	.loc	0 1791 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1791:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cndmask_b32_e64 v13, v17, 0, s1
	v_or_b32_e32 v17, 8, v10
	v_lshl_add_u32 v135, v4, 2, v21
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v13, s5, s8, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v24, null, s9, 0, s5
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_mov_b32_e32 v8, v1
	v_add_nc_u32_e32 v9, v21, v9
	v_add_co_u32 v137, vcc_lo, v13, v91
	v_lshl_add_u32 v114, v11, 2, v21
	v_lshl_add_u32 v115, v12, 2, v21
	v_lshl_add_u32 v116, v27, 2, v21
	v_lshl_add_u32 v117, v28, 2, v21
	v_lshl_add_u32 v118, v29, 2, v21
	v_lshl_add_u32 v120, v14, 2, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, 0, v24, vcc_lo
	v_lshlrev_b32_e32 v139, 3, v10
	v_lshlrev_b32_e32 v140, 3, v15
	v_lshlrev_b32_e32 v141, 3, v17
	v_lshlrev_b32_e32 v142, 3, v18
	v_lshlrev_b32_e32 v143, 3, v22
	v_lshlrev_b32_e32 v144, 3, v16
	v_lshlrev_b32_e32 v146, 3, v26
	v_add_nc_u32_e32 v147, 0, v20
	v_add_nc_u32_e32 v148, v9, v19
	v_mov_b32_e32 v16, v8
	v_mov_b32_e32 v24, v8
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v67, v3
	v_lshl_add_u32 v136, s20, 11, v95
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v9, v1
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_mov_b32_e32 v17, v1
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_mov_b32_e32 v31, v7
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v68, v4
	v_mov_b32_e32 v66, v2
	.loc	0 1865 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[10:11], s[6:7]
	s_add_nc_u64 s[10:11], s[12:13], s[6:7]
	s_mov_b32 s6, 0x76543210
	s_branch .LBB5_11
.LBB5_9:                                ;   in Loop: Header=BB5_11 Depth=1
.Ltmp295:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v151, v2
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp296:
.LBB5_10:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	3 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0x3fffffff
	s_cselect_b32 s5, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s12, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_83
.LBB5_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_72 Depth 2
                                        ;       Child Loop BB5_77 Depth 3
	.loc	0 1866 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_lshl_b32 s13, s7, 6
	.loc	0 1867 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1867:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s13, s18
	s_cselect_b32 s12, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s12
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_10
; %bb.12:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v2, s13, v89
                                        ; implicit-def: $vgpr5
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[73:74], null, 0x408, v2, s[16:17]
	v_cmp_ge_i32_e32 vcc_lo, s18, v2
	.loc	0 1880 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s22, exec_lo, s5
	s_cbranch_execz .LBB5_14
; %bb.13:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1887 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_co_u32 v2, s5, v73, v139
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, s5
	v_add_co_u32 v7, s5, v73, v140
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, s5
	s_clause 0x3
	global_load_b64 v[75:76], v[2:3], off
	global_load_b64 v[77:78], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[75:78]
.LBB5_14:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s22
	s_cbranch_execz .LBB5_16
; %bb.15:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4]
.LBB5_16:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:2048
                                        ; implicit-def: $vgpr5
	.loc	0 1880 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s22, exec_lo, s5
	s_cbranch_execz .LBB5_18
; %bb.17:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1887 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_co_u32 v2, s5, v73, v141
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, s5
	v_add_co_u32 v7, s5, v73, v142
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, s5
	s_clause 0x3
	global_load_b64 v[75:76], v[2:3], off
	global_load_b64 v[77:78], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[75:78] offset:4096
.LBB5_18:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s22
	s_cbranch_execz .LBB5_20
; %bb.19:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4] offset:4096
.LBB5_20:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:6144
                                        ; implicit-def: $vgpr5
	.loc	0 1880 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s22, exec_lo, s5
	s_cbranch_execz .LBB5_22
; %bb.21:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1887 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_co_u32 v2, s5, v73, v143
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, s5
	v_add_co_u32 v7, s5, v73, v144
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, s5
	s_clause 0x3
	global_load_b64 v[75:76], v[2:3], off
	global_load_b64 v[77:78], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[75:78] offset:8192
.LBB5_22:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s22
	s_cbranch_execz .LBB5_24
; %bb.23:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4] offset:8192
.LBB5_24:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:10240
                                        ; implicit-def: $vgpr5
	.loc	0 1880 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB5_26
; %bb.25:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1887 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_co_u32 v2, vcc_lo, v73, v145
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, vcc_lo
	v_add_co_u32 v7, vcc_lo, v73, v146
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, vcc_lo
	s_clause 0x3
	global_load_b64 v[73:74], v[2:3], off
	global_load_b64 v[75:76], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[73:76] offset:12288
.LBB5_26:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s5, s5
	s_cbranch_execz .LBB5_28
; %bb.27:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4] offset:12288
.LBB5_28:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_dual_mov_b32 v74, 0 :: v_dual_add_nc_u32 v75, s13, v87
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 0
	.loc	0 1906 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:14336
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_30
; %bb.29:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v75, v[83:84]
	global_load_b64 v[2:3], v[2:3], off
.LBB5_30:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v76, 0x8000, v96
	.loc	0 1906 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v76, v2, v3 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_gt_i32_e64 s18, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_32
; %bb.31:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1903 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v2, 1, v75
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, v[83:84]
	global_load_b64 v[73:74], v[2:3], off
.LBB5_32:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v6, 2, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v97, v73 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v99, v74 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_34
; %bb.33:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v6, v[83:84]
	global_load_b64 v[4:5], v[4:5], off
.LBB5_34:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v6, 3, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v100, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v101, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_36
; %bb.35:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v6, v[83:84]
	global_load_b64 v[2:3], v[2:3], off
.LBB5_36:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v8, 4, v75
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v102, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v103, v3 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v8
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_38
; %bb.37:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v8, v[83:84]
	global_load_b64 v[6:7], v[2:3], off
.LBB5_38:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v2, 5, v75
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_nc_u32_e32 v74, 0x8400, v96
	.loc	0 1906 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v74, v6, v7 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v2
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, v[83:84]
	global_load_b64 v[4:5], v[2:3], off
.LBB5_40:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v8, 6, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v104, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v105, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v8
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_42
; %bb.41:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v8, v[83:84]
	global_load_b64 v[6:7], v[4:5], off
.LBB5_42:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v4, 7, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v106, v6 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v107, v7 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v4
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_44
; %bb.43:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v4, v[83:84]
	global_load_b64 v[2:3], v[2:3], off
.LBB5_44:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1912 62 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v108, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v109, v3 offset:32768
.Ltmp297:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_barrier_signal -1
	v_add_nc_u32_e32 v8, 0x8000, v111
	v_add_nc_u32_e32 v73, 0x8400, v111
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp298:
	.loc	0 1930 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, s3
	s_cbranch_execz .LBB5_46
; %bb.45:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset1:4
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset1:4
.Ltmp299:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v110, v2
.Ltmp300:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v110, v4
.Ltmp301:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v110, v3
.Ltmp302:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v110, v5
.Ltmp303:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v77, v3, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp304:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v3, v98, v2
.Ltmp305:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v4
.Ltmp306:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v6
.Ltmp307:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp308:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v6, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:16384
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v112 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v113 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v114 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v115 offset:32768
.Ltmp309:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp310:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp311:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v110, v4
.Ltmp312:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v110, v5
.Ltmp313:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp314:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp315:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp316:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v4
.Ltmp317:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp318:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:16896
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v116 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v117 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v118 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v119 offset:32768
.Ltmp319:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp320:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp321:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v110, v4
.Ltmp322:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v110, v5
.Ltmp323:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp324:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp325:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp326:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v4
.Ltmp327:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp328:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:17408
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v120 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v121 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v122 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v123 offset:32768
.Ltmp329:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp330:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp331:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v110, v4
.Ltmp332:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v110, v5
.Ltmp333:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp334:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp335:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp336:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v4
.Ltmp337:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp338:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:17920
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset0:32 offset1:36
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset0:32 offset1:36
.Ltmp339:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v110, v2
.Ltmp340:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v110, v4
.Ltmp341:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v110, v3
.Ltmp342:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v110, v5
.Ltmp343:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v77, v3, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp344:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v3, v98, v2
.Ltmp345:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v4
.Ltmp346:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v6
.Ltmp347:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp348:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v6, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:18432
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v124 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v125 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v126 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v127 offset:32768
.Ltmp349:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp350:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp351:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v110, v4
.Ltmp352:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v110, v5
.Ltmp353:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp354:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp355:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp356:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v4
.Ltmp357:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp358:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:18944
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v128 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v129 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v130 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v131 offset:32768
.Ltmp359:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp360:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp361:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v110, v4
.Ltmp362:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v110, v5
.Ltmp363:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp364:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp365:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp366:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v4
.Ltmp367:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp368:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:19456
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v132 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v133 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v134 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v135 offset:32768
.Ltmp369:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp370:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp371:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v110, v4
.Ltmp372:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v110, v5
.Ltmp373:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v92
.Ltmp374:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp375:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp376:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v77, v98, v4
.Ltmp377:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v78, v98, v5
.Ltmp378:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:19968
.LBB5_46:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 51 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:51
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp379:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v6, 8, v75
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
.Ltmp380:
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
.Ltmp381:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp382:
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_48
; %bb.47:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v6, v[83:84]
	global_load_b64 v[2:3], v[2:3], off
.LBB5_48:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v6, 9, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v76, v2, v3 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_50
; %bb.49:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v6, v[83:84]
	global_load_b64 v[4:5], v[2:3], off
.LBB5_50:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v76, 10, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v97, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v99, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v76
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_52
; %bb.51:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v76, v[83:84]
	global_load_b64 v[6:7], v[4:5], off
.LBB5_52:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v4, 11, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v100, v6 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v101, v7 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v4
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_54
; %bb.53:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v4, v[83:84]
	global_load_b64 v[2:3], v[2:3], off
.LBB5_54:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v76, 12, v75
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v102, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v103, v3 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v76
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_56
; %bb.55:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v76, v[83:84]
	global_load_b64 v[6:7], v[2:3], off
.LBB5_56:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v2, 13, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v74, v6, v7 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v2
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_58
; %bb.57:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v2, v[83:84]
	global_load_b64 v[4:5], v[2:3], off
.LBB5_58:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v74, 14, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v104, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v105, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v74
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_60
; %bb.59:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v74, v[83:84]
	global_load_b64 v[6:7], v[4:5], off
.LBB5_60:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v4, 15, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v106, v6 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v107, v7 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_ge_i32_e64 s18, v4
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_execz .LBB5_62
; %bb.61:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[2:3], null, 0x408, v4, v[83:84]
	global_load_b64 v[2:3], v[2:3], off
.LBB5_62:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1912 62 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v108, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b32 v109, v3 offset:32768
.Ltmp383:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp384:
	.loc	0 1930 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB5_64
; %bb.63:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset1:4
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset1:4
.Ltmp385:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v110, v2
.Ltmp386:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v110, v4
.Ltmp387:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v74, v110, v3
.Ltmp388:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v75, v110, v5
.Ltmp389:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v74, v3, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v92
.Ltmp390:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v3, v98, v2
.Ltmp391:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v4
.Ltmp392:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v74, v98, v6
.Ltmp393:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v75, v98, v5
.Ltmp394:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v6, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:16384
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v112 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v113 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v114 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v115 offset:32768
.Ltmp395:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp396:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp397:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v74, v110, v4
.Ltmp398:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v75, v110, v5
.Ltmp399:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v92
.Ltmp400:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp401:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp402:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v74, v98, v4
.Ltmp403:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v75, v98, v5
.Ltmp404:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:16896
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v116 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v117 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v118 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v119 offset:32768
.Ltmp405:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp406:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp407:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v74, v110, v4
.Ltmp408:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v75, v110, v5
.Ltmp409:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v92
.Ltmp410:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp411:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp412:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v74, v98, v4
.Ltmp413:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v75, v98, v5
.Ltmp414:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:17408
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v120 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v121 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v122 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v123 offset:32768
.Ltmp415:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp416:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp417:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v74, v110, v4
.Ltmp418:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v75, v110, v5
.Ltmp419:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v92
.Ltmp420:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp421:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp422:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v74, v98, v4
.Ltmp423:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v75, v98, v5
.Ltmp424:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:17920
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset0:32 offset1:36
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset0:32 offset1:36
.Ltmp425:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v110, v2
.Ltmp426:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v110, v4
.Ltmp427:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v8, v110, v3
.Ltmp428:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v73, v110, v5
.Ltmp429:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v8, v3, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v92
.Ltmp430:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v3, v98, v2
.Ltmp431:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v4
.Ltmp432:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v8, v98, v6
.Ltmp433:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v73, v98, v5
.Ltmp434:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v6, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:18432
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v124 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v125 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v126 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v127 offset:32768
.Ltmp435:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp436:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp437:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v110, v4
.Ltmp438:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v73, v110, v5
.Ltmp439:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v92
.Ltmp440:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp441:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp442:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v8, v98, v4
.Ltmp443:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v73, v98, v5
.Ltmp444:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:18944
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v128 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v129 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v130 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v131 offset:32768
.Ltmp445:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp446:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp447:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v110, v4
.Ltmp448:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v73, v110, v5
.Ltmp449:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v92
.Ltmp450:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp451:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp452:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v8, v98, v4
.Ltmp453:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v73, v98, v5
.Ltmp454:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:19456
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v2, v132 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v3, v133 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v4, v134 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b32 v5, v135 offset:32768
.Ltmp455:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v110, v2
.Ltmp456:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v110, v3
.Ltmp457:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v110, v4
.Ltmp458:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v73, v110, v5
.Ltmp459:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v92
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v92
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v92
.Ltmp460:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v6, v98, v2
.Ltmp461:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v7, v98, v3
.Ltmp462:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v8, v98, v4
.Ltmp463:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	ds_bpermute_b32 v73, v98, v5
.Ltmp464:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v93
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v93
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v93
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b128 v148, v[2:5] offset:19968
.LBB5_64:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 51 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:51
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp465:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp466:
	.loc	0 1964 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB5_68
; %bb.65:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1965 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1965:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v3, s13, v0
	v_mov_b32_e32 v2, 0
	.loc	0 1967 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1967:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s22, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s18, v3
	s_cbranch_execz .LBB5_67
; %bb.66:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1972 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v3, s[8:9]
	.loc	0 1974 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mad_co_u64_u32 v[6:7], null, 0x408, v3, s[10:11]
	.loc	0 1972 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	global_load_d16_b16 v2, v[4:5], off offset:1024
	.loc	0 1974 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	global_load_d16_hi_b16 v2, v[6:7], off offset:1024
	.loc	0 1972 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v3.h, 8, v2.l
	.loc	0 1974 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshrrev_b16 v3.l, 8, v2.h
	.loc	0 1974 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_and_b16 v4.h, 0xff, v2.l
	v_and_b16 v4.l, 0xff, v2.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1975 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1975:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_pk_lshlrev_b16 v2, 8, v3 op_sel_hi:[0,1]
	.loc	0 1974 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_or_b32_e32 v2, v2, v4
.LBB5_67:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	.loc	0 1977 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b16_d16_hi v90, v2 offset:40960
	.loc	0 1978 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1978:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_store_b16 v90, v2 offset:41088
.LBB5_68:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
.Ltmp467:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp468:
	.loc	0 1871 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_or_b32 s5, s13, 63
	v_mov_b32_e32 v5, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s5, s19
	s_cselect_b32 s5, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s22, s21, s5
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s23, s1, s22
.Ltmp469:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp470:
	.loc	0 1988 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s23
	s_cbranch_execz .LBB5_70
; %bb.69:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 1988 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	global_load_b32 v5, v[85:86], off
.LBB5_70:                               ;   in Loop: Header=BB5_11 Depth=1
	.loc	0 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1984 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_or_b32 s5, s13, 16
	s_mov_b32 s23, 0
	.loc	0 1984 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s5, s18
	s_cselect_b32 s24, -1, 0
	s_branch .LBB5_72
.LBB5_71:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_sub_f32_e32 v79, v151, v2
	.loc	0 2055 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v151
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2069 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_dual_add_f32 v4, v4, v78 :: v_dual_mul_f32 v79, 0x3fb8aa3b, v79
	v_exp_f32_e32 v79, v79
	.loc	0 2055 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v79, 0, v79, vcc_lo
	.loc	0 2079 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mul_f32_e32 v78, v149, v79
	.loc	0 2070 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_fmac_f32_e32 v4, v150, v79
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2079 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v79, null, v3, v3, v78
	v_rcp_f32_e32 v80, v79
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v79, v80, 1.0
	v_fmac_f32_e32 v80, v149, v80
	v_div_scale_f32 v149, vcc_lo, v78, v3, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v80
	v_fma_f32 v151, -v79, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v80
	v_fma_f32 v79, -v79, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v79, v79, v80, v150
	v_div_fixup_f32 v78, v79, v3, v78
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_dual_mul_f32 v72, v72, v78 :: v_dual_mul_f32 v67, v67, v78
	v_dual_mul_f32 v71, v71, v78 :: v_dual_mul_f32 v70, v70, v78
	v_mul_f32_e32 v65, v65, v78
	v_dual_mul_f32 v69, v69, v78 :: v_dual_mul_f32 v68, v68, v78
	v_dual_mul_f32 v63, v63, v78 :: v_dual_mul_f32 v66, v66, v78
	v_dual_mul_f32 v61, v61, v78 :: v_dual_mul_f32 v64, v64, v78
	v_dual_mul_f32 v59, v59, v78 :: v_dual_mul_f32 v62, v62, v78
	v_dual_mul_f32 v57, v57, v78 :: v_dual_mul_f32 v60, v60, v78
	v_dual_mul_f32 v55, v55, v78 :: v_dual_mul_f32 v58, v58, v78
	v_dual_mul_f32 v53, v53, v78 :: v_dual_mul_f32 v56, v56, v78
	v_dual_mul_f32 v51, v51, v78 :: v_dual_mul_f32 v54, v54, v78
	v_dual_mul_f32 v49, v49, v78 :: v_dual_mul_f32 v52, v52, v78
	v_dual_mul_f32 v47, v47, v78 :: v_dual_mul_f32 v50, v50, v78
	v_dual_mul_f32 v45, v45, v78 :: v_dual_mul_f32 v48, v48, v78
	v_dual_mul_f32 v43, v43, v78 :: v_dual_mul_f32 v46, v46, v78
	v_dual_mul_f32 v41, v41, v78 :: v_dual_mul_f32 v44, v44, v78
	v_dual_mul_f32 v39, v39, v78 :: v_dual_mul_f32 v42, v42, v78
	v_dual_mul_f32 v37, v37, v78 :: v_dual_mul_f32 v40, v40, v78
	v_dual_mul_f32 v35, v35, v78 :: v_dual_mul_f32 v38, v38, v78
	v_dual_mul_f32 v33, v33, v78 :: v_dual_mul_f32 v36, v36, v78
	v_dual_mul_f32 v31, v31, v78 :: v_dual_mul_f32 v34, v34, v78
	v_dual_mul_f32 v29, v29, v78 :: v_dual_mul_f32 v32, v32, v78
	v_dual_mul_f32 v27, v27, v78 :: v_dual_mul_f32 v30, v30, v78
	v_dual_mul_f32 v25, v25, v78 :: v_dual_mul_f32 v28, v28, v78
	v_dual_mul_f32 v23, v23, v78 :: v_dual_mul_f32 v26, v26, v78
	v_dual_mul_f32 v21, v21, v78 :: v_dual_mul_f32 v24, v24, v78
	v_dual_mul_f32 v19, v19, v78 :: v_dual_mul_f32 v22, v22, v78
	v_dual_mul_f32 v17, v17, v78 :: v_dual_mul_f32 v20, v20, v78
	v_dual_mul_f32 v15, v15, v78 :: v_dual_mul_f32 v18, v18, v78
	v_dual_mul_f32 v13, v13, v78 :: v_dual_mul_f32 v16, v16, v78
	v_dual_mul_f32 v11, v11, v78 :: v_dual_mul_f32 v14, v14, v78
	v_dual_mul_f32 v9, v9, v78 :: v_dual_mul_f32 v12, v12, v78
	v_mul_f32_e32 v10, v10, v78
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v78, null, v3, v3, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v79, v78
	v_fma_f32 v80, -v78, v79, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v79, v80, v79
	v_div_scale_f32 v80, vcc_lo, v77, v3, v77
	v_mul_f32_e32 v149, v80, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v78, v149, v80
	v_fmac_f32_e32 v149, v150, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v78, -v78, v149, v80
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v78, v78, v79, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v79, v78, v3, v77
	.loc	0 2085 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v77, null, v3, v3, v76
	v_rcp_f32_e32 v78, v77
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v80, -v77, v78, 1.0
	v_fmac_f32_e32 v78, v80, v78
	v_div_scale_f32 v80, vcc_lo, v76, v3, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v149, v80, v78
	v_fma_f32 v150, -v77, v149, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v78
	v_fma_f32 v77, -v77, v149, v80
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v77, v77, v78, v149
	.loc	0 2084 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b16_e32 v78.l, v1.l
	.loc	0 2086 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b16_e32 v78.h, 0
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_fixup_f32 v76, v77, v3, v76
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 2084 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b16_e32 v77.l, v78.l
	.loc	0 2086 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b16_e32 v77.h, v78.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2084 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cvt_pk_fp8_f32 v77.l, v79, v76
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v76, null, v3, v3, v75
	v_rcp_f32_e32 v79, v76
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v80, -v76, v79, 1.0
	v_fmac_f32_e32 v79, v80, v79
	v_div_scale_f32 v80, vcc_lo, v75, v3, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v149, v80, v79
	v_fma_f32 v150, -v76, v149, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v79
	v_fma_f32 v76, -v76, v149, v80
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v76, v76, v79, v149
	v_div_fixup_f32 v75, v76, v3, v75
	.loc	0 2087 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v76, null, v3, v3, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v79, v76
	v_fma_f32 v80, -v76, v79, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v79, v80, v79
	v_div_scale_f32 v80, vcc_lo, v74, v3, v74
	v_mul_f32_e32 v149, v80, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v76, v149, v80
	v_dual_fmac_f32 v149, v150, v79 :: v_dual_mov_b32 v150, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v76, -v76, v149, v80
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v76, v76, v79, v149
	v_mov_b32_e32 v149, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v74, v76, v3, v74
	.loc	0 2086 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cvt_pk_fp8_f32 v77.h, v75, v74
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v74, null, v3, v3, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v75, v74
	v_fma_f32 v76, -v74, v75, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v75, v76, v75
	v_div_scale_f32 v76, vcc_lo, v73, v3, v73
	v_mul_f32_e32 v79, v76, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v80, -v74, v79, v76
	v_fmac_f32_e32 v79, v80, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v74, -v74, v79, v76
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v74, v74, v75, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v73, v74, v3, v73
	.loc	0 2090 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v74, null, v3, v3, v8
	v_rcp_f32_e32 v75, v74
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v76, -v74, v75, 1.0
	v_fmac_f32_e32 v75, v76, v75
	v_div_scale_f32 v76, vcc_lo, v8, v3, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v79, v76, v75
	v_fma_f32 v80, -v74, v79, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v79, v80, v75
	v_fma_f32 v74, -v74, v79, v76
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v74, v74, v75, v79
	v_div_fixup_f32 v8, v74, v3, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2089 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2089:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cvt_pk_fp8_f32 v78.l, v73, v8
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v8, null, v3, v3, v7
	v_rcp_f32_e32 v73, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v74, -v8, v73, 1.0
	v_fmac_f32_e32 v73, v74, v73
	v_div_scale_f32 v74, vcc_lo, v7, v3, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v75, v74, v73
	v_fma_f32 v76, -v8, v75, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v75, v76, v73
	v_fma_f32 v8, -v8, v75, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v8, v8, v73, v75
	v_div_fixup_f32 v7, v8, v3, v7
	.loc	0 2092 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v8, null, v3, v3, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v73, v8
	v_fma_f32 v74, -v8, v73, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v73, v74, v73
	v_div_scale_f32 v74, vcc_lo, v6, v3, v6
	v_mul_f32_e32 v75, v74, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v76, -v8, v75, v74
	v_fmac_f32_e32 v75, v76, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v8, -v8, v75, v74
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v8, v8, v73, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v6, v8, v3, v6
	.loc	0 2091 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cvt_pk_fp8_f32 v78.h, v7, v6
	.loc	0 2098 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshl_add_u32 v6, s23, 12, v136
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b128 v[73:76], v6 offset:16384
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[73:74], v[77:78], v[65:72]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[77:78], v[57:64]
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b128 v[73:76], v6 offset:16896
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[73:74], v[77:78], v[49:56]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[75:76], v[77:78], v[41:48]
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b128 v[73:76], v6 offset:17408
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[73:74], v[77:78], v[33:40]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[75:76], v[77:78], v[25:32]
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b128 v[73:76], v6 offset:17920
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[73:74], v[77:78], v[17:24]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[75:76], v[77:78], v[9:16]
	v_mov_b32_e32 v151, v2
	.loc	0 1994 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_co_i32 s23, s23, 1
	.loc	0 1994 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s23, 4
	.loc	0 1994 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_scc1 .LBB5_9
.LBB5_72:                               ;   Parent Loop BB5_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_77 Depth 3
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s23, 1
	s_mov_b32 s5, -1
	s_cbranch_scc1 .LBB5_75
; %bb.73:                               ;   in Loop: Header=BB5_72 Depth=2
	s_cmp_eq_u32 s23, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s5, s24
	s_cbranch_scc1 .LBB5_75
; %bb.74:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 1996 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1996:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cmp_eq_u32 s23, 2
	.loc	0 1996 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1996:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cselect_b32 s5, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s5, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s18, s5
	s_cselect_b32 s5, -1, 0
.LBB5_75:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s5
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_82
; %bb.76:                               ;   in Loop: Header=BB5_72 Depth=2
	v_mov_b32_e32 v73, 0
	.loc	0 2006 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshl_add_u32 v2, s23, 9, v95
	s_mov_b32 s5, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v74, v73 :: v_dual_mov_b32 v75, v73
	v_dual_mov_b32 v76, v73 :: v_dual_mov_b32 v77, v73
	v_dual_mov_b32 v78, v73 :: v_dual_mov_b32 v79, v73
	v_mov_b32_e32 v80, v73
.LBB5_77:                               ;   Parent Loop BB5_11 Depth=1
                                        ;     Parent Loop BB5_72 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 2018 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2018:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s25, s5, 6
	.loc	0 2011 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2011:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshl_add_u32 v8, s5, 12, v2
	.loc	0 2021 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, vcc_lo, v137, s25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v138, vcc_lo
	.loc	0 2013 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b128 v[152:155], v8
	ds_load_b128 v[156:159], v8 offset:2048
	.loc	0 2006 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_co_i32 s5, s5, 1
	.loc	0 2021 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_clause 0x3
	global_load_b64 v[6:7], v[3:4], off
	global_load_b64 v[160:161], v[3:4], off offset:16
	global_load_b64 v[162:163], v[3:4], off offset:32
	global_load_b64 v[3:4], v[3:4], off offset:48
	.loc	0 2030 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 4
	.loc	0 2024 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[152:153], v[6:7], v[73:80]
	.loc	0 2027 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[154:155], v[160:161], v[73:80]
	.loc	0 2024 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[156:157], v[162:163], v[73:80]
	.loc	0 2027 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[158:159], v[3:4], v[73:80]
	.loc	0 2006 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_scc1 .LBB5_77
; %bb.78:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_lshl_add_u32 v6, s23, 5, v94
	.loc	0 2033 48 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_lshl_b32 s5, s23, 4
	v_mov_b32_e32 v7, 0xff800000
	.loc	0 2033 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s5, s13
	.loc	0 2038 41 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2038:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_b96 v[2:4], v6 offset:40962
	ds_load_u16_d16 v8, v6 offset:40974
	.loc	0 2034 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, s5, 15
	v_or_b32_e32 v152, s5, v91
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s25, s19
	s_cselect_b32 s5, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s22, s5
	.loc	0 2043 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s25, s0
	s_cbranch_execz .LBB5_80
; %bb.79:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 2038 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2038:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_u16_d16 v7, v6 offset:40960
	v_mul_f32_e32 v73, v88, v73
	v_cmp_le_i32_e32 vcc_lo, v152, v5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s5, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v7, v7, v73, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2045 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2045:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v7, 0xff800000, v7, vcc_lo
.LBB5_80:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s25
	v_mul_f32_e32 v73, v88, v74
	v_or_b32_e32 v74, 2, v152
	v_cmp_ge_i32_e32 vcc_lo, v152, v5
	s_xor_b32 s25, s5, -1
	v_mul_f32_e32 v75, v88, v75
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s5, v74, v5
	v_or_b32_e32 v74, 3, v152
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s26, s25, vcc_lo
	.loc	0 2043 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s1, s26
	s_and_b32 s5, s25, s5
	v_cmp_gt_i32_e32 vcc_lo, v74, v5
	v_mul_f32_e32 v74, v88, v76
	s_wait_dscnt 0x1
	v_fma_mix_f32 v73, v2, v73, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v2, v2, v75, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v75, 4, v152
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s1, s5
	v_cndmask_b32_e64 v73, v73, 0xff800000, s26
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v76, v2, 0xff800000, s5
	v_fma_mix_f32 v2, v3, v74, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v74, v88, v77
	s_and_b32 s5, s25, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v75, v5
	v_or_b32_e32 v75, 5, v152
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s1, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v77, v2, 0xff800000, s5
	v_fma_mix_f32 v2, v3, v74, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v3, v88, v78
	s_and_b32 s5, s25, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v75, v5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s1, s5
	v_or_b32_e32 v75, 7, v152
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v74, v2, 0xff800000, s5
	v_fma_mix_f32 v2, v4, v3, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v3, 6, v152
	s_and_b32 s5, s25, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s1, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v78, v2, 0xff800000, s5
	v_cmp_gt_i32_e32 vcc_lo, v3, v5
	v_mul_f32_e32 v2, v88, v79
	v_cmp_gt_i32_e64 s5, v75, v5
	v_mul_f32_e32 v3, v88, v80
.Ltmp471:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v75, v7, 0xff800000, v73
	s_and_b32 s26, s25, vcc_lo
	v_fma_mix_f32 v2, v4, v2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s5, s25, s5
	s_wait_dscnt 0x0
	v_fma_mix_f32 v3, v8, v3, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v4, v75, v76, v77
.Ltmp472:
	.loc	0 2043 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, s1, s26
	s_or_b32 s5, s1, s5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v8, v2, 0xff800000, s25
	v_cndmask_b32_e64 v3, v3, 0xff800000, s5
.Ltmp473:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v2, v4, v74, v78
.Ltmp474:
	.loc	0 2077 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp475:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v2, v2, v8, v3
.Ltmp476:
	.loc	0 1739 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1739:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2052:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_mov_b32_e32 v4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v4, v4, s6, 0xfedcba98
.Ltmp477:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2053:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v2, v151, v2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_sub_f32 v4, v7, v2 :: v_dual_sub_f32 v7, v73, v2
	v_sub_f32_e32 v3, v3, v2
	v_add_nc_u32_e32 v73, 0xa080, v6
	v_sub_f32_e32 v78, v78, v2
	v_dual_mul_f32 v4, 0x3fb8aa3b, v4 :: v_dual_mul_f32 v7, 0x3fb8aa3b, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v3, 0x3fb8aa3b, v3
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_exp_f32_e32 v7, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v75, v3
	v_dual_sub_f32 v3, v76, v2 :: v_dual_sub_f32 v76, v77, v2
	v_exp_f32_e32 v77, v4
	v_mul_f32_e32 v76, 0x3fb8aa3b, v76
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v80, 0x3fb8aa3b, v3
.Ltmp478:
	.loc	0 2059 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[3:4], v73 offset1:1
	v_sub_f32_e32 v73, v74, v2
	v_mul_f32_e32 v74, 0x3fb8aa3b, v78
	v_exp_f32_e32 v76, v76
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v73, 0x3fb8aa3b, v73
	v_exp_f32_e32 v74, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v73, v73
	v_cndmask_b32_e64 v153, v76, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_2)
	v_cndmask_b32_e64 v152, v74, 0, vcc_lo
	.loc	0 2059 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_nc_u32_e32 v6, 0xa088, v6
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v76, v3, v7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v74, v4, v153, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2059 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	ds_load_2addr_b32 v[78:79], v6 offset1:1
	v_exp_f32_e32 v6, v80
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cndmask_b32_e64 v80, v75, 0, vcc_lo
	v_cndmask_b32_e64 v75, v77, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v8, v8, v2 :: v_dual_add_f32 v7, v75, v7
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_fma_mix_f32 v77, v3, v75, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cndmask_b32_e64 v3, v73, 0, vcc_lo
	v_mul_f32_e32 v8, 0x3fb8aa3b, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v6, v6, 0, vcc_lo
	v_exp_f32_e32 v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_fma_mix_f32 v75, v4, v6, neg(0) op_sel_hi:[1,0,0]
.Ltmp479:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v4, v77, 0, v76
.Ltmp480:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_f32_e32 v6, v6, v7
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v73, v78, v3, neg(0) op_sel_hi:[1,0,0]
.Ltmp481:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v4, v4, v75, v74
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
.Ltmp482:
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cndmask_b32_e64 v154, v8, 0, vcc_lo
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_fma_mix_f32 v8, v78, v152, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_f32_e32 v78, v153, v6
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_fma_mix_f32 v6, v79, v80, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v7, v79, v154, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp483:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v4, v4, v73, v8
.Ltmp484:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_add_f32_e32 v3, v3, v78
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp485:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max3_num_f32 v79, v4, v7, v6
.Ltmp486:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_dual_add_f32 v3, v152, v3 :: v_dual_mov_b32 v78, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v154, v3
.Ltmp487:
	.loc	0 1739 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1739:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_permlanex16_b32 v78, v78, s6, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp488:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_dual_add_f32 v4, v80, v3 :: v_dual_max_num_f32 v3, v78, v78
.Ltmp489:
	.loc	0 1739 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1739:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_dual_mov_b32 v78, v4 :: v_dual_max_num_f32 v79, v79, v3
	v_mov_b32_e32 v3, v149
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v78, v78, s6, 0xfedcba98
.Ltmp490:
	.loc	0 2077 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmpx_lt_f32_e32 0, v79
	s_cbranch_execz .LBB5_71
; %bb.81:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	0 2078 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v80, v3
	v_fma_f32 v152, -v3, v80, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v80, v152, v80
	v_div_scale_f32 v152, vcc_lo, v79, 0x43e00000, v79
	v_mul_f32_e32 v153, v152, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v3, v153, v152
	v_fmac_f32_e32 v153, v154, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v153, v152
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v80, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x43e00000, v79
.Ltmp491:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ] ]
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB5_71
.Ltmp492:
.LBB5_82:                               ;   in Loop: Header=BB5_72 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v151
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v151, v2
	.loc	0 1994 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_add_co_i32 s23, s23, 1
	.loc	0 1994 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s23, 4
	.loc	0 1994 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_cbranch_scc0 .LBB5_72
	s_branch .LBB5_9
.LBB5_83:
	.loc	0 2121 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_85
; %bb.84:
	.loc	0 2122 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2122:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_scale_f32 v0, null, v150, v150, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v150, 1.0
	s_lshl_b32 s0, s20, 7
	v_rcp_f32_e32 v1, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v3, v1
	v_fma_f32 v4, -v0, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v4, v1
	.loc	0 2124 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2124:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mul_lo_u32 v4, 0x1800, v82
	.loc	0 2122 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2122:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v0, v0, v1, v2
	.loc	0 2124 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2124:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshl_add_u32 v2, v81, 8, v4
	.loc	0 2122 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2122:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v150
	.loc	0 2131 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	.loc	0 2122 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2122:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_div_fixup_f32 v3, v0, v150, 1.0
	.loc	0 2126 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2126:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v0, v2, s0, v91
	.loc	0 2122 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2122:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 2131 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v75, v149, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v73, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, s15, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 2131 56 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_dual_mul_f32 v0, v65, v75 :: v_dual_mul_f32 v3, v68, v75
	v_dual_mul_f32 v1, v66, v75 :: v_dual_mul_f32 v2, v67, v75
	v_dual_mul_f32 v5, v70, v75 :: v_dual_mul_f32 v4, v69, v75
	v_dual_mul_f32 v7, v72, v75 :: v_dual_mul_f32 v6, v71, v75
	v_dual_mul_f32 v57, v57, v75 :: v_dual_mul_f32 v58, v58, v75
	v_dual_mul_f32 v59, v59, v75 :: v_dual_mul_f32 v60, v60, v75
	v_dual_mul_f32 v61, v61, v75 :: v_dual_mul_f32 v62, v62, v75
	v_dual_mul_f32 v63, v63, v75 :: v_dual_mul_f32 v64, v64, v75
	v_dual_mul_f32 v49, v49, v75 :: v_dual_mul_f32 v50, v50, v75
	v_dual_mul_f32 v51, v51, v75 :: v_dual_mul_f32 v52, v52, v75
	v_dual_mul_f32 v53, v53, v75 :: v_dual_mul_f32 v54, v54, v75
	v_dual_mul_f32 v55, v55, v75 :: v_dual_mul_f32 v56, v56, v75
	v_dual_mul_f32 v41, v41, v75 :: v_dual_mul_f32 v42, v42, v75
	v_dual_mul_f32 v43, v43, v75 :: v_dual_mul_f32 v44, v44, v75
	v_dual_mul_f32 v45, v45, v75 :: v_dual_mul_f32 v46, v46, v75
	v_dual_mul_f32 v47, v47, v75 :: v_dual_mul_f32 v48, v48, v75
	v_dual_mul_f32 v33, v33, v75 :: v_dual_mul_f32 v34, v34, v75
	v_dual_mul_f32 v35, v35, v75 :: v_dual_mul_f32 v36, v36, v75
	v_dual_mul_f32 v37, v37, v75 :: v_dual_mul_f32 v38, v38, v75
	v_dual_mul_f32 v39, v39, v75 :: v_dual_mul_f32 v40, v40, v75
	.loc	0 2131 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_clause 0x9
	global_store_b128 v[73:74], v[0:3], off
	global_store_b128 v[73:74], v[4:7], off offset:16
	global_store_b128 v[73:74], v[57:60], off offset:64
	global_store_b128 v[73:74], v[61:64], off offset:80
	global_store_b128 v[73:74], v[49:52], off offset:128
	global_store_b128 v[73:74], v[53:56], off offset:144
	global_store_b128 v[73:74], v[41:44], off offset:192
	global_store_b128 v[73:74], v[45:48], off offset:208
	global_store_b128 v[73:74], v[33:36], off offset:256
	global_store_b128 v[73:74], v[37:40], off offset:272
	.loc	0 2131 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	v_dual_mul_f32 v0, v25, v75 :: v_dual_mul_f32 v3, v28, v75
	v_dual_mul_f32 v1, v26, v75 :: v_dual_mul_f32 v2, v27, v75
	v_dual_mul_f32 v5, v30, v75 :: v_dual_mul_f32 v4, v29, v75
	v_dual_mul_f32 v7, v32, v75 :: v_dual_mul_f32 v6, v31, v75
	v_dual_mul_f32 v17, v17, v75 :: v_dual_mul_f32 v18, v18, v75
	v_dual_mul_f32 v19, v19, v75 :: v_dual_mul_f32 v20, v20, v75
	v_dual_mul_f32 v21, v21, v75 :: v_dual_mul_f32 v22, v22, v75
	v_dual_mul_f32 v23, v23, v75 :: v_dual_mul_f32 v24, v24, v75
	v_dual_mul_f32 v8, v9, v75 :: v_dual_mul_f32 v9, v10, v75
	v_dual_mul_f32 v10, v11, v75 :: v_dual_mul_f32 v11, v12, v75
	v_dual_mul_f32 v12, v13, v75 :: v_dual_mul_f32 v13, v14, v75
	v_dual_mul_f32 v14, v15, v75 :: v_dual_mul_f32 v15, v16, v75
	.loc	0 2131 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2197:5 ]
	s_clause 0x5
	global_store_b128 v[73:74], v[0:3], off offset:320
	global_store_b128 v[73:74], v[4:7], off offset:336
	global_store_b128 v[73:74], v[17:20], off offset:384
	global_store_b128 v[73:74], v[21:24], off offset:400
	global_store_b128 v[73:74], v[8:11], off offset:448
	global_store_b128 v[73:74], v[12:15], off offset:464
.Ltmp493:
.LBB5_85:
	.loc	0 2201 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp494:
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
		.amdhsa_next_free_vgpr 164
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_vgpr, 164
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
; codeLenInByte = 11644
; TotalNumSgprs: 29
; NumVgprs: 164
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 164
; Occupancy: 9
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
	.loc	0 2216 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2216:17
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2216 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2216:23
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
	s_cbranch_vccnz .LBB6_87
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s24, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2221 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2221:14
	s_cmp_gt_i32 s24, 3
	s_cbranch_scc1 .LBB6_87
; %bb.2:
	.loc	0 2223 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2223:29
	s_lshl_b32 s2, ttmp9, 6
	.loc	0 2224 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2224:23
	s_mul_i32 s14, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2224 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2224:18
	s_cmp_ge_i32 s2, s14
	s_cbranch_scc1 .LBB6_87
; %bb.3:
	.loc	0 2227 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2227:39
	s_lshr_b32 s12, ttmp7, 17
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2228 15                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2228:15
	s_cmp_ge_i32 s12, s17
	s_cbranch_scc1 .LBB6_87
; %bb.4:
.Ltmp495:
	.loc	0 1767 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	0 1769 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1769:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_and_b32_e32 v3, 15, v0
.Ltmp496:
	.loc	0 2216 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2216:17
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x20
.Ltmp497:
	.loc	0 1781 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mul_i32 s1, s24, 6
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mov_b32 v89, s16 :: v_dual_lshlrev_b32 v88, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1778 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v2, v88, v3
	.loc	0 1778 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_nc_u32_e32 v4, s2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1779 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_hi_i32 v2, 0x2aaaaaab, v4
	.loc	0 1782 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1782:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmp_gt_i32_e64 s0, s14, v4
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshrrev_b32_e32 v5, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, v2, v5
	.loc	0 1781 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_lo_u32 v5, v2, 6
	v_mul_lo_u32 v6, v2, 24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v5, v4, v5
	v_add3_u32 v81, v5, s1, v6
	.loc	0 1832 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1832:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_6
; %bb.5:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_mov_b32_e32 v82, 0
	s_mul_i32 s20, s15, 0x1800
	s_mov_b32 s21, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[20:21], s[4:5], s[20:21]
	v_lshlrev_b64_e32 v[4:5], 2, v[81:82]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, s20, v4
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	.loc	0 1835 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1835:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	global_load_b32 v4, v[4:5], off
	.loc	0 1987 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1987:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v89, s16, v4
.LBB6_6:
	.loc	0 0 47 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:47
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 1841 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_or_b32 s15, s2, 63
	.loc	0 1841 60 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_co_i32 s1, s14, -1
	.loc	0 1840 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1840:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mul_hi_i32 s3, s2, 0x2aaaaaab
.Ltmp498:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_alu depctr_sa_sdst(0)
	s_min_i32 s1, s15, s1
.Ltmp499:
	.loc	0 1840 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1840:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_lshr_b32 s2, s3, 31
	.loc	0 1842 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_i32 s1, s1, 0x2aaaaaab
	.loc	0 1768 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1768:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_and_b32_e32 v4, 31, v0
	.loc	0 1842 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s13, s1, 31
	.loc	0 1840 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1840:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_co_i32 s3, s3, s2
	.loc	0 1842 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_co_i32 s1, s1, s13
	v_mov_b32_e32 v6, -1
	.loc	0 1842 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1842:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s1, s1, s3
	v_bfrev_b32_e32 v5, -2
	.loc	0 1845 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1845:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_ge_i32_e32 vcc_lo, s1, v4
	s_lshr_b32 s1, ttmp7, 16
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB6_8
; %bb.7:
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	v_add_nc_u32_e32 v5, s3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s18, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, vcc_lo
	.loc	0 1847 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v6, v5
.LBB6_8:
	.loc	0 0 23 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
.Ltmp500:
	v_mbcnt_lo_u32_b32 v7, -1, 0
.Ltmp501:
	.loc	0 1770 25 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1770:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshrrev_b32_e32 v90, 4, v4
	v_lshlrev_b32_e32 v10, 3, v4
	v_dual_mov_b32 v16, 0x5040100 :: v_dual_and_b32 v11, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp502:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_xor_b32_e32 v8, 16, v7
.Ltmp503:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_xor_b32_e32 v18, 8, v7
.Ltmp504:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_xor_b32_e32 v22, 4, v7
	v_lshlrev_b32_e32 v20, 4, v4
	v_lshrrev_b32_e32 v12, 2, v3
.Ltmp505:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
.Ltmp506:
	.loc	0 2230 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_cvt_f32_u32 s16, s17
	v_and_or_b32 v91, v88, 48, v3
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_or_b32 v12, v11, 6, v12
.Ltmp507:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v7, v8 :: v_dual_lshlrev_b32 v9, 11, v1
.Ltmp508:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v18
	v_and_b32_e32 v13, 1, v0
	v_or_b32_e32 v28, 0x108, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v15, 0x6020400 :: v_dual_lshlrev_b32 v8, 2, v8
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v7, v18, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v13
	v_lshlrev_b32_e32 v17, 8, v81
.Ltmp509:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v14, v8, v6
.Ltmp510:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v8, v5
	s_wait_alu depctr_va_vcc(0)
	v_dual_mov_b32 v1, 0 :: v_dual_cndmask_b32 v94, 0x3070105, v15
	v_cmp_gt_u32_e32 vcc_lo, 2, v11
.Ltmp511:
	.loc	0 1774 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_nc_u32_e32 v18, 0, v9
	.loc	0 1791 10                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1791:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cndmask_b32_e64 v13, 0, v17, s0
.Ltmp512:
	.loc	0 2230 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_add_co_i32 s2, s17, 0x1ff
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s22, s16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v95, 0x3020706, v16, vcc_lo
.Ltmp513:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v22
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
.Ltmp514:
	.loc	0 2230 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_and_b32 s2, s2, 0xffff
	s_wait_kmcnt 0x0
	v_add_co_u32 v13, s4, s4, v13
.Ltmp515:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v17, v7, v22 :: v_dual_mov_b32 v86, 0xff800000
.Ltmp516:
	.loc	0 2230 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s20, s2
	.loc	0 2226 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2226:46
	s_and_b32 s13, s1, 1
	v_or_b32_e32 v27, 8, v12
.Ltmp517:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1851:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v14
.Ltmp518:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v5, v8
.Ltmp519:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_xor_b32_e32 v8, 2, v7
.Ltmp520:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_xor_b32_e32 v14, 1, v7
.Ltmp521:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_lshlrev_b32_e32 v17, 2, v17
	s_mov_b32 s21, 0
	v_or_b32_e32 v29, 12, v12
.Ltmp522:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	v_lshlrev_b32_e32 v23, 3, v11
	v_and_b32_e32 v19, 16, v0
	v_lshlrev_b32_e32 v21, 4, v0
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v7, v8 :: v_dual_lshlrev_b32 v93, 3, v90
.Ltmp523:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v14
.Ltmp524:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_lshlrev_b32_e32 v4, 2, v4
	v_add_nc_u32_e32 v97, 0, v20
	v_add_nc_u32_e32 v98, v18, v10
.Ltmp525:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v7, v14 :: v_dual_lshlrev_b32 v106, 2, v8
.Ltmp526:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v24, v4, v6
.Ltmp527:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v4, v4, v5
	v_add_co_u32 v82, vcc_lo, s18, v2
	v_dual_mov_b32 v149, 1.0 :: v_dual_lshlrev_b32 v112, 2, v7
	v_xor_b32_e32 v7, v28, v23
	v_add_co_ci_u32_e64 v14, null, s5, 0, s4
.Ltmp528:
	.loc	0 2230 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s4, s20, s22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, s19, v3, vcc_lo
	v_lshl_add_u32 v115, v7, 2, v18
	v_mov_b32_e32 v7, v1
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s4, s4
	v_or_b32_e32 v30, 0x10c, v12
	v_or_b32_e32 v22, 16, v90
	v_or_b32_e32 v25, 20, v90
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, s4, 0x80000000
	s_cvt_u32_f32 s4, s4
.Ltmp529:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1852:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v6, v6, v24
.Ltmp530:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v5, v4
.Ltmp531:
	.loc	0 2230 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s20, s5, s16
	v_or_b32_e32 v24, 28, v90
	v_or_b32_e32 v26, 24, v90
.Ltmp532:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v5, v17, v6
.Ltmp533:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v17, v17, v4
.Ltmp534:
	.loc	0 2230 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2230:42
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s20, 31
	v_xad_u32 v99, 0x120, v10, v18
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s20, s16
	v_xad_u32 v100, 0x124, v10, v18
	v_xad_u32 v101, 0x240, v10, v18
	v_xad_u32 v102, 0x244, v10, v18
	s_add_co_ci_u32 s4, s4, 0
	v_xad_u32 v103, 0x360, v10, v18
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s4, 0xffff
	v_xad_u32 v104, 0x364, v10, v18
	.loc	0 2231 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2231:26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s16, s12, s4
	v_xad_u32 v105, 0x520, v10, v18
	.loc	0 2232 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2232:23
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s22, s16, s4
.Ltmp535:
	.loc	0 1864 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cmp_lt_i32 s15, s14
	v_xad_u32 v107, 0x524, v10, v18
	s_cselect_b32 s23, -1, 0
	s_lshl_b32 s20, s24, 8
	v_xad_u32 v108, 0x640, v10, v18
.Ltmp536:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1853:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v5, v6, v5
.Ltmp537:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v4, v4, v17
.Ltmp538:
	.loc	0 1865 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[6:7], s[20:21]
	s_add_nc_u64 s[4:5], s[8:9], s[20:21]
	s_lshl_b32 s20, s24, 1
.Ltmp539:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v5
.Ltmp540:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v106, v4
	v_xad_u32 v109, 0x644, v10, v18
	v_xad_u32 v110, 0x760, v10, v18
	v_xad_u32 v111, 0x764, v10, v18
	v_or_b32_e32 v15, 4, v90
	v_or_b32_e32 v16, 8, v90
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v84, s4, s4, v10
	v_cmp_gt_u32_e64 s1, 64, v0
	v_lshl_add_u32 v92, v0, 1, 0
	v_cmp_eq_u32_e64 s2, 0, v90
	v_cmp_ne_u32_e64 s3, 0, v90
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v85, null, s5, 0, s4
.Ltmp541:
	.loc	0 1865 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_nc_u64 s[18:19], s[6:7], s[20:21]
	s_add_nc_u64 s[8:9], s[8:9], s[20:21]
	s_mov_b32 s20, 0x76543210
	v_mov_b32_e32 v87, 0
.Ltmp542:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1854:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v5, v6
.Ltmp543:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v4, v8
	v_xor_b32_e32 v6, v27, v23
	v_xor_b32_e32 v8, v29, v23
.Ltmp544:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v4, v112, v2
.Ltmp545:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v5, v112, v3
	v_lshl_add_u32 v114, v6, 2, v18
	v_or_b32_e32 v6, 24, v12
	v_lshl_add_u32 v116, v8, 2, v18
	v_mov_b32_e32 v8, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v6, v6, v23
	v_lshl_add_u32 v122, v6, 2, v18
	v_or_b32_e32 v6, 0x128, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v6, v6, v23
.Ltmp546:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v2, v2, v4
.Ltmp547:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v5
	v_or_b32_e32 v4, 20, v12
	v_or_b32_e32 v5, 0x114, v12
	v_lshl_add_u32 v127, v6, 2, v18
.Ltmp548:
	.loc	0 1861 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_readfirstlane_b32 s24, v2
	.loc	0 1862 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_readfirstlane_b32 s25, v3
	v_or_b32_e32 v2, 16, v12
	v_or_b32_e32 v3, 0x110, v12
	v_xor_b32_e32 v4, v4, v23
	v_xor_b32_e32 v5, v5, v23
	v_or_b32_e32 v6, 52, v12
	v_xor_b32_e32 v2, v2, v23
	v_xor_b32_e32 v3, v3, v23
	v_lshl_add_u32 v120, v4, 2, v18
	v_lshl_add_u32 v121, v5, 2, v18
	v_or_b32_e32 v4, 0x11c, v12
	v_lshl_add_u32 v118, v2, 2, v18
	v_lshl_add_u32 v119, v3, 2, v18
	v_or_b32_e32 v2, 0x118, v12
	v_or_b32_e32 v3, 28, v12
	v_or_b32_e32 v5, 40, v12
	v_xor_b32_e32 v4, v4, v23
	v_xor_b32_e32 v6, v6, v23
	v_xor_b32_e32 v2, v2, v23
	v_xor_b32_e32 v3, v3, v23
	v_xor_b32_e32 v5, v5, v23
	v_lshl_add_u32 v125, v4, 2, v18
	v_or_b32_e32 v4, 48, v12
	v_lshl_add_u32 v123, v2, 2, v18
	v_lshl_add_u32 v124, v3, 2, v18
	v_lshl_add_u32 v126, v5, 2, v18
	v_or_b32_e32 v2, 44, v12
	v_or_b32_e32 v3, 0x12c, v12
	v_or_b32_e32 v5, 0x130, v12
	v_xor_b32_e32 v4, v4, v23
	v_lshl_add_u32 v132, v6, 2, v18
	v_xor_b32_e32 v2, v2, v23
	v_xor_b32_e32 v3, v3, v23
	v_xor_b32_e32 v5, v5, v23
	v_lshl_add_u32 v130, v4, 2, v18
	v_or_b32_e32 v4, 0x138, v12
	v_lshl_add_u32 v128, v2, 2, v18
	v_lshl_add_u32 v129, v3, 2, v18
	v_lshl_add_u32 v131, v5, 2, v18
	v_or_b32_e32 v2, 0x134, v12
	v_or_b32_e32 v3, 56, v12
	v_or_b32_e32 v5, 60, v12
	v_or_b32_e32 v6, 0x13c, v12
	v_xor_b32_e32 v4, v4, v23
	v_xor_b32_e32 v2, v2, v23
	v_xor_b32_e32 v3, v3, v23
	v_xor_b32_e32 v5, v5, v23
	v_xor_b32_e32 v6, v6, v23
	v_lshl_add_u32 v135, v4, 2, v18
	v_lshl_add_u32 v133, v2, 2, v18
	v_lshl_add_u32 v134, v3, 2, v18
	v_lshl_add_u32 v136, v5, 2, v18
	v_lshl_add_u32 v137, v6, 2, v18
	v_mov_b32_e32 v2, v1
	v_dual_mov_b32 v3, v1 :: v_dual_add_nc_u32 v96, 0, v19
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_lshlrev_b32 v11, 5, v11
	v_lshlrev_b32_e32 v17, 2, v12
	v_or_b32_e32 v19, 12, v90
	v_add_nc_u32_e32 v9, v18, v9
	v_lshlrev_b32_e32 v143, 3, v22
	v_lshlrev_b32_e32 v146, 3, v24
	v_add3_u32 v113, v18, v17, v11
	v_xor_b32_e32 v11, v30, v23
	v_lshlrev_b32_e32 v142, 3, v19
	v_add_nc_u32_e32 v147, 0, v21
	v_add_nc_u32_e32 v148, v9, v20
	v_lshlrev_b32_e32 v144, 3, v25
	v_lshl_add_u32 v117, v11, 2, v18
	v_dual_mov_b32 v24, v8 :: v_dual_lshlrev_b32 v145, 3, v26
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v37, v5
	v_mov_b32_e32 v48, v8
	v_add_co_u32 v138, vcc_lo, v13, v93
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v56, v8
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v64, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v139, null, 0, v14, vcc_lo
	v_lshlrev_b32_e32 v140, 3, v15
	v_dual_mov_b32 v16, v8 :: v_dual_lshlrev_b32 v141, 3, v16
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v72, v8
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v9, v1
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v46, v6 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v44, v4 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v52, v4 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v60, v4 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v70, v6 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v68, v4 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v69, v5
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v65, v1
	s_branch .LBB6_11
.LBB6_9:                                ;   in Loop: Header=BB6_11 Depth=1
.Ltmp549:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v86, v2
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp550:
.LBB6_10:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	3 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s16, s22
	s_cselect_b32 s4, -1, 0
	s_xor_b32 s5, s21, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_83
.LBB6_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_72 Depth 2
                                        ;       Child Loop BB6_77 Depth 3
	.loc	0 1866 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_lshl_b32 s26, s16, 6
	.loc	0 1867 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1867:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s26, s24
	s_cselect_b32 s21, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_10
; %bb.12:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v2, s26, v91
                                        ; implicit-def: $vgpr5
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[73:74], null, 0x408, v2, s[14:15]
	v_cmp_ge_i32_e32 vcc_lo, s24, v2
	.loc	0 1880 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s4
	s_cbranch_execz .LBB6_14
; %bb.13:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	v_lshlrev_b32_e32 v2, 3, v90
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1887 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_co_u32 v2, s4, v73, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, s4
	v_add_co_u32 v7, s4, v73, v140
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, s4
	s_clause 0x3
	global_load_b64 v[75:76], v[2:3], off
	global_load_b64 v[77:78], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[75:78]
.LBB6_14:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .LBB6_16
; %bb.15:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4]
.LBB6_16:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:2048
                                        ; implicit-def: $vgpr5
	.loc	0 1880 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s4
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1887 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_co_u32 v2, s4, v73, v141
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, s4
	v_add_co_u32 v7, s4, v73, v142
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, s4
	s_clause 0x3
	global_load_b64 v[75:76], v[2:3], off
	global_load_b64 v[77:78], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[75:78] offset:4096
.LBB6_18:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4] offset:4096
.LBB6_20:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:6144
                                        ; implicit-def: $vgpr5
	.loc	0 1880 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s4
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1887 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_co_u32 v2, s4, v73, v143
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, s4
	v_add_co_u32 v7, s4, v73, v144
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, s4
	s_clause 0x3
	global_load_b64 v[75:76], v[2:3], off
	global_load_b64 v[77:78], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[75:78] offset:8192
.LBB6_22:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4] offset:8192
.LBB6_24:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:10240
                                        ; implicit-def: $vgpr5
	.loc	0 1880 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1880:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1887 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1887:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_co_u32 v2, vcc_lo, v73, v145
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v74, vcc_lo
	v_add_co_u32 v7, vcc_lo, v73, v146
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v74, vcc_lo
	s_clause 0x3
	global_load_b64 v[73:74], v[2:3], off
	global_load_b64 v[75:76], v[2:3], off offset:16
	global_load_b64 v[5:6], v[7:8], off
	global_load_b64 v[7:8], v[7:8], off offset:16
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v147, v[73:76] offset:12288
.LBB6_26:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB6_28
; %bb.27:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v3, v1
	.loc	0 1895 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b32_e32 v2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	ds_store_b128 v147, v[1:4] offset:12288
.LBB6_28:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v74, 0 :: v_dual_add_nc_u32 v75, s26, v88
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 0
	.loc	0 1906 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1895 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1895:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v147, v[5:8] offset:14336
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v75, v[84:85]
	global_load_b64 v[2:3], v[2:3], off
.LBB6_30:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v76, 0x8000, v98
	.loc	0 1906 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v76, v2, v3 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_gt_i32_e64 s24, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_32
; %bb.31:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1903 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v2, 1, v75
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[84:85]
	global_load_b64 v[73:74], v[2:3], off
.LBB6_32:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v6, 2, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v99, v73 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v100, v74 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_34
; %bb.33:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v6, v[84:85]
	global_load_b64 v[4:5], v[4:5], off
.LBB6_34:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v6, 3, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v101, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v102, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_36
; %bb.35:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v6, v[84:85]
	global_load_b64 v[2:3], v[2:3], off
.LBB6_36:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v8, 4, v75
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v103, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v104, v3 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v8
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v8, v[84:85]
	global_load_b64 v[6:7], v[2:3], off
.LBB6_38:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v2, 5, v75
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_nc_u32_e32 v74, 0x8400, v98
	.loc	0 1906 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v74, v6, v7 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v2
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_40
; %bb.39:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[84:85]
	global_load_b64 v[4:5], v[2:3], off
.LBB6_40:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v8, 6, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v105, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v107, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v8
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_42
; %bb.41:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v8, v[84:85]
	global_load_b64 v[6:7], v[4:5], off
.LBB6_42:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v4, 7, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v108, v6 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v109, v7 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v4
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_44
; %bb.43:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v4, v[84:85]
	global_load_b64 v[2:3], v[2:3], off
.LBB6_44:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1912 62 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v110, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v111, v3 offset:32768
.Ltmp551:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_barrier_signal -1
	v_add_nc_u32_e32 v8, 0x8000, v113
	v_add_nc_u32_e32 v73, 0x8400, v113
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp552:
	.loc	0 1930 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset1:4
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset1:4
.Ltmp553:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v112, v2
.Ltmp554:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v112, v4
.Ltmp555:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v112, v3
.Ltmp556:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v112, v5
.Ltmp557:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v77, v3, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp558:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v3, v106, v2
.Ltmp559:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v4
.Ltmp560:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v6
.Ltmp561:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp562:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v6, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:16384
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v114 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v115 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v116 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v117 offset:32768
.Ltmp563:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp564:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp565:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v112, v4
.Ltmp566:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v112, v5
.Ltmp567:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp568:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp569:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp570:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v4
.Ltmp571:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp572:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:16896
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v118 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v119 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v120 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v121 offset:32768
.Ltmp573:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp574:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp575:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v112, v4
.Ltmp576:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v112, v5
.Ltmp577:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp578:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp579:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp580:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v4
.Ltmp581:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp582:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:17408
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v122 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v123 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v124 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v125 offset:32768
.Ltmp583:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp584:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp585:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v112, v4
.Ltmp586:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v112, v5
.Ltmp587:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp588:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp589:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp590:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v4
.Ltmp591:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp592:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:17920
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset0:32 offset1:36
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset0:32 offset1:36
.Ltmp593:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v112, v2
.Ltmp594:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v112, v4
.Ltmp595:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v112, v3
.Ltmp596:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v112, v5
.Ltmp597:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v77, v3, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp598:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v3, v106, v2
.Ltmp599:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v4
.Ltmp600:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v6
.Ltmp601:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp602:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v6, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:18432
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v126 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v127 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v128 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v129 offset:32768
.Ltmp603:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp604:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp605:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v112, v4
.Ltmp606:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v112, v5
.Ltmp607:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp608:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp609:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp610:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v4
.Ltmp611:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp612:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:18944
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v130 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v131 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v132 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v133 offset:32768
.Ltmp613:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp614:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp615:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v112, v4
.Ltmp616:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v112, v5
.Ltmp617:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp618:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp619:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp620:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v4
.Ltmp621:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp622:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:19456
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v134 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v135 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v136 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v137 offset:32768
.Ltmp623:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp624:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp625:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v77, v112, v4
.Ltmp626:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v78, v112, v5
.Ltmp627:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v94
.Ltmp628:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp629:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp630:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v77, v106, v4
.Ltmp631:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v78, v106, v5
.Ltmp632:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v77, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v78, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:19968
.LBB6_46:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 51 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:51
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp633:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v6, 8, v75
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
.Ltmp634:
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
.Ltmp635:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp636:
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_48
; %bb.47:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v6, v[84:85]
	global_load_b64 v[2:3], v[2:3], off
.LBB6_48:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v6, 9, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v76, v2, v3 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v6
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_50
; %bb.49:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v6, v[84:85]
	global_load_b64 v[4:5], v[2:3], off
.LBB6_50:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v76, 10, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v99, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v100, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v76
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_52
; %bb.51:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v76, v[84:85]
	global_load_b64 v[6:7], v[4:5], off
.LBB6_52:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v4, 11, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v101, v6 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v102, v7 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v4
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_54
; %bb.53:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v4, v[84:85]
	global_load_b64 v[2:3], v[2:3], off
.LBB6_54:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v76, 12, v75
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v103, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v104, v3 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v76
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_56
; %bb.55:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v76, v[84:85]
	global_load_b64 v[6:7], v[2:3], off
.LBB6_56:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v2, 13, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v74, v6, v7 offset1:1
	.loc	0 1904 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v2
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_58
; %bb.57:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[84:85]
	global_load_b64 v[4:5], v[2:3], off
.LBB6_58:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v74, 14, v75
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v105, v4 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v107, v5 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v74
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_60
; %bb.59:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v74, v[84:85]
	global_load_b64 v[6:7], v[4:5], off
.LBB6_60:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 55 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v4, 15, v75
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v108, v6 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v109, v7 offset:32768
	.loc	0 1904 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_ge_i32_e64 s24, v4
	.loc	0 1906 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_execz .LBB6_62
; %bb.61:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1910 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[2:3], null, 0x408, v4, v[84:85]
	global_load_b64 v[2:3], v[2:3], off
.LBB6_62:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1912 62 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v110, v2 offset:32768
	.loc	0 1914 66                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:66 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b32 v111, v3 offset:32768
.Ltmp637:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp638:
	.loc	0 1930 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1930:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_64
; %bb.63:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset1:4
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset1:4
.Ltmp639:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v112, v2
.Ltmp640:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v112, v4
.Ltmp641:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v74, v112, v3
.Ltmp642:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v75, v112, v5
.Ltmp643:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v74, v3, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v94
.Ltmp644:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v3, v106, v2
.Ltmp645:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v4
.Ltmp646:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v74, v106, v6
.Ltmp647:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v75, v106, v5
.Ltmp648:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v6, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:16384
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v114 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v115 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v116 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v117 offset:32768
.Ltmp649:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp650:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp651:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v74, v112, v4
.Ltmp652:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v75, v112, v5
.Ltmp653:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v94
.Ltmp654:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp655:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp656:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v74, v106, v4
.Ltmp657:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v75, v106, v5
.Ltmp658:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:16896
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v118 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v119 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v120 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v121 offset:32768
.Ltmp659:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp660:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp661:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v74, v112, v4
.Ltmp662:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v75, v112, v5
.Ltmp663:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v94
.Ltmp664:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp665:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp666:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v74, v106, v4
.Ltmp667:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v75, v106, v5
.Ltmp668:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:17408
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v122 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v123 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v124 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v125 offset:32768
.Ltmp669:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp670:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp671:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v74, v112, v4
.Ltmp672:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v75, v112, v5
.Ltmp673:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v94
.Ltmp674:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp675:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp676:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v74, v106, v4
.Ltmp677:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v75, v106, v5
.Ltmp678:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v74, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v75, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:17920
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[2:3], v8 offset0:32 offset1:36
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[4:5], v73 offset0:32 offset1:36
.Ltmp679:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v6, v112, v2
.Ltmp680:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v7, v112, v4
.Ltmp681:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v112, v3
.Ltmp682:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v73, v112, v5
.Ltmp683:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v7, v4, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v6, v8, v3, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v94
.Ltmp684:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v3, v106, v2
.Ltmp685:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v4
.Ltmp686:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v106, v6
.Ltmp687:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v73, v106, v5
.Ltmp688:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v3, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v4, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v6, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:18432
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v126 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v127 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v128 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v129 offset:32768
.Ltmp689:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp690:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp691:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v112, v4
.Ltmp692:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v73, v112, v5
.Ltmp693:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v94
.Ltmp694:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp695:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp696:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v106, v4
.Ltmp697:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v73, v106, v5
.Ltmp698:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:18944
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v130 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v131 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v132 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v133 offset:32768
.Ltmp699:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp700:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp701:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v112, v4
.Ltmp702:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v73, v112, v5
.Ltmp703:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v94
.Ltmp704:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp705:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp706:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v106, v4
.Ltmp707:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v73, v106, v5
.Ltmp708:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:19456
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v2, v134 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v3, v135 offset:32768
	.loc	0 1938 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v4, v136 offset:32768
	.loc	0 1939 49                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b32 v5, v137 offset:32768
.Ltmp709:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v6, v112, v2
.Ltmp710:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v112, v3
.Ltmp711:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v112, v4
.Ltmp712:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v73, v112, v5
.Ltmp713:
	.loc	0 1942 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v94
	.loc	0 1942 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v94
	.loc	0 1945 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v94
.Ltmp714:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v6, v106, v2
.Ltmp715:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v7, v106, v3
.Ltmp716:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v8, v106, v4
.Ltmp717:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	ds_bpermute_b32 v73, v106, v5
.Ltmp718:
	.loc	0 1949 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v2, v6, v2, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v3, v7, v3, v95
	.loc	0 1949 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1949:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v8, v4, v95
	.loc	0 1953 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v5, v73, v5, v95
	.loc	0 1957 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b128 v148, v[2:5] offset:19968
.LBB6_64:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 51 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:51
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp719:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1961:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp720:
	.loc	0 1964 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB6_68
; %bb.65:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1965 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1965:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v3, s26, v0
	v_mov_b32_e32 v2, 0
	.loc	0 1967 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1967:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s24, v3
	s_cbranch_execz .LBB6_67
; %bb.66:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1972 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v3, s[18:19]
	.loc	0 1974 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_i64_i32 v[6:7], null, 0x408, v3, s[8:9]
	.loc	0 1972 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	global_load_d16_b16 v2, v[4:5], off offset:1024
	.loc	0 1974 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	global_load_d16_hi_b16 v2, v[6:7], off offset:1024
	.loc	0 1972 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v3.h, 8, v2.l
	.loc	0 1974 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshrrev_b16 v3.l, 8, v2.h
	.loc	0 1974 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_and_b16 v4.h, 0xff, v2.l
	v_and_b16 v4.l, 0xff, v2.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1975 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1975:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_pk_lshlrev_b16 v2, 8, v3 op_sel_hi:[0,1]
	.loc	0 1974 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1974:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v2, v2, v4
.LBB6_67:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1977 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b16_d16_hi v92, v2 offset:40960
	.loc	0 1978 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1978:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_store_b16 v92, v2 offset:41088
.LBB6_68:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp721:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp722:
	.loc	0 1871 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_or_b32 s4, s26, 63
	v_mov_b32_e32 v5, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s25
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s27, s23, s4
	.loc	0 1988 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, s27, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s0, s4
.Ltmp723:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp724:
	.loc	0 1988 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB6_70
; %bb.69:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 1988 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	global_load_b32 v5, v[82:83], off
.LBB6_70:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	0 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1984 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_or_b32 s4, s26, 16
	s_mov_b32 s28, 0
	.loc	0 1984 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1984:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s24
	s_cselect_b32 s29, -1, 0
	s_branch .LBB6_72
.LBB6_71:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_or_b32 exec_lo, exec_lo, s4
	v_sub_f32_e32 v79, v86, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	.loc	0 2085 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v153, null, v8, v8, v76
	.loc	0 2055 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v86
	.loc	0 2069 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_f32_e32 v157, v77, v78
	v_mul_f32_e32 v79, 0x3fb8aa3b, v79
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v154, v153
	v_exp_f32_e32 v79, v79
	s_delay_alu instid0(TRANS32_DEP_2)
	v_fma_f32 v77, -v153, v154, 1.0
	.loc	0 2055 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v79, 0, v79 :: v_dual_fmac_f32 v154, v77, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2079 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v80, v149, v79
	.loc	0 2079 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v86, null, v8, v8, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v149, v86
	v_fma_f32 v150, -v86, v149, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v149
	v_div_scale_f32 v151, vcc_lo, v80, v8, v80
	v_mul_f32_e32 v150, v151, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v86, v150, v151
	v_fmac_f32_e32 v150, v152, v149
	.loc	0 2085 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v152, null, v8, v8, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 2079 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v86, -v86, v150, v151
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v151, v152
	.loc	0 2079 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_div_fmas_f32 v86, v86, v149, v150
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v149, null, v8, v8, v74
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v78, vcc_lo, v76, v8, v76
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v150, null, v8, v8, v73
	.loc	0 2079 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v158, v86, v8, v80
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v80, v149
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v86, -v152, v151, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v156, v150
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v16, v16, v158 :: v_dual_mul_f32 v71, v71, v158
	v_mul_f32_e32 v70, v70, v158
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v151, v86, v151
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v69, v69, v158 :: v_dual_mul_f32 v68, v68, v158
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v155, -v149, v80, 1.0
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v67, v67, v158 :: v_dual_mul_f32 v66, v66, v158
	v_dual_mul_f32 v72, v72, v158 :: v_dual_mul_f32 v65, v65, v158
	v_mul_f32_e32 v64, v64, v158
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v80, v155, v80
	v_div_scale_f32 v155, s5, v74, v8, v74
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v77, v78, v154
	.loc	0 2070 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v157, v87, v79
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v79, s4, v75, v8, v75
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v63, v63, v158 :: v_dual_mul_f32 v62, v62, v158
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v86, -v153, v77, v78
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v61, v61, v158 :: v_dual_mul_f32 v60, v60, v158
	v_dual_mul_f32 v59, v59, v158 :: v_dual_mul_f32 v58, v58, v158
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v77, v86, v154
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v57, v57, v158 :: v_dual_mul_f32 v56, v56, v158
	v_dual_mul_f32 v55, v55, v158 :: v_dual_mul_f32 v54, v54, v158
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v78, -v153, v77, v78
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v153, -v150, v156, 1.0
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v87, v79, v151
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v53, v53, v158 :: v_dual_mul_f32 v52, v52, v158
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v77, v78, v154, v77
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v156, v153, v156
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v86, -v152, v87, v79
	s_mov_b32 vcc_lo, s4
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v51, v51, v158 :: v_dual_mul_f32 v50, v50, v158
	.loc	0 2085 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v159, v77, v8, v76
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2085 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_fmac_f32 v87, v86, v151 :: v_dual_mul_f32 v86, v155, v80
	.loc	0 2090 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v76, null, v8, v8, v3
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v49, v49, v158 :: v_dual_mul_f32 v48, v48, v158
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v78, -v152, v87, v79
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v79, -v149, v86, v155
	.loc	0 2087 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v152, s6, v73, v8, v73
	.loc	0 2082 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v47, v47, v158 :: v_dual_mul_f32 v46, v46, v158
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v78, v78, v151, v87
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v86, v79, v80
	s_mov_b32 vcc_lo, s5
	.loc	0 2084 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b16_e32 v87.l, v1.l
	.loc	0 2086 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b16_e32 v87.h, 0
	.loc	0 2085 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2085:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v160, v78, v8, v75
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v75, -v149, v86, v155
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v78, v76
	.loc	0 2090 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v149, null, v8, v8, v4
	.loc	0 2082 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v45, v45, v158 :: v_dual_mul_f32 v44, v44, v158
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v75, v75, v80, v86
	.loc	0 2087 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 vcc_lo, s6
	.loc	0 2084 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b16_e32 v86.l, v87.l
	.loc	0 2086 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b16_e32 v86.h, v87.h
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v43, v43, v158 :: v_dual_mul_f32 v42, v42, v158
	.loc	0 2087 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v161, v75, v8, v74
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v74, null, v8, v8, v6
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v80, -v76, v78, 1.0
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v41, v41, v158 :: v_dual_mul_f32 v40, v40, v158
	v_dual_mul_f32 v39, v39, v158 :: v_dual_mul_f32 v38, v38, v158
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v151, v74
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v78, v80, v78
	v_div_scale_f32 v80, s4, v3, v8, v3
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v37, v37, v158 :: v_dual_mul_f32 v36, v36, v158
	v_dual_mul_f32 v35, v35, v158 :: v_dual_mul_f32 v34, v34, v158
	v_dual_mul_f32 v33, v33, v158 :: v_dual_mul_f32 v32, v32, v158
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v154, -v74, v151, 1.0
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v31, v31, v158 :: v_dual_mul_f32 v30, v30, v158
	v_dual_mul_f32 v12, v12, v158 :: v_dual_mul_f32 v29, v29, v158
	v_dual_mul_f32 v28, v28, v158 :: v_dual_fmac_f32 v151, v154, v151
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v77, v152, v156
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v154, s6, v6, v8, v6
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v15, v15, v158
	v_dual_mul_f32 v27, v27, v158 :: v_dual_mul_f32 v26, v26, v158
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v79, -v150, v77, v152
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v25, v25, v158 :: v_dual_mul_f32 v24, v24, v158
	v_dual_mul_f32 v23, v23, v158 :: v_dual_mul_f32 v22, v22, v158
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v77, v79, v156
	.loc	0 2090 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v79, v149
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v10, v10, v158 :: v_dual_mul_f32 v21, v21, v158
	v_mul_f32_e32 v20, v20, v158
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v75, -v150, v77, v152
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v150, null, v8, v8, v7
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v19, v19, v158 :: v_dual_mul_f32 v18, v18, v158
	v_dual_mul_f32 v17, v17, v158 :: v_dual_mul_f32 v14, v14, v158
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_rcp_f32_e32 v153, v150
	s_delay_alu instid0(TRANS32_DEP_2)
	.loc	0 2090 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v152, -v149, v79, 1.0
	.loc	0 2087 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v75, v75, v156, v77
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v77, v80, v78 :: v_dual_mul_f32 v156, v154, v151
	s_mov_b32 vcc_lo, s4
	.loc	0 2090 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v79, v152, v79
	v_div_scale_f32 v152, s5, v4, v8, v4
	.loc	0 2087 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v162, v75, v8, v73
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v73, -v150, v153, 1.0
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v75, -v76, v77, v80
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v13, v13, v158
	.loc	0 2090 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v155, v152, v79
	.loc	0 2084 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cvt_pk_fp8_f32 v86.l, v159, v160
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v153, v73, v153
	v_div_scale_f32 v73, s7, v7, v8, v7
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v77, v75, v78
	.loc	0 2090 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v75, -v149, v155, v152
	.loc	0 2086 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2086:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cvt_pk_fp8_f32 v86.h, v161, v162
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v11, v11, v158
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v163, v73, v153
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v76, -v76, v77, v80
	.loc	0 2090 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v155, v75, v79
	.loc	0 2092 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v75, -v74, v156, v154
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v9, v9, v158
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v80, -v150, v163, v73
	.loc	0 2090 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v164, v76, v78, v77
	.loc	0 2090 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v76, -v149, v155, v152
	.loc	0 2092 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v156, v75, v151
	v_lshl_add_u32 v75, s13, 11, v97
	.loc	0 2092 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fmac_f32_e32 v163, v80, v153
	.loc	0 2090 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 vcc_lo, s5
	.loc	0 2090 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v3, v164, v8, v3
	.loc	0 2092 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v74, -v74, v156, v154
	.loc	0 2098 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshl_add_u32 v154, s28, 12, v75
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_f32 v73, -v150, v163, v73
	.loc	0 2090 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v165, v76, v79, v155
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v166, v74, v151, v156
	.loc	0 2092 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 vcc_lo, s7
	.loc	0 2090 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v4, v165, v8, v4
	.loc	0 2092 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v163, v73, v153, v163
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b128 v[73:76], v154 offset:16384
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b128 v[77:80], v154 offset:16896
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b128 v[149:152], v154 offset:17408
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2100 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b128 v[153:156], v154 offset:17920
	.loc	0 2092 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v6, v166, v8, v6
	.loc	0 2092 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_fixup_f32 v7, v163, v8, v7
	.loc	0 2089 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2089:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cvt_pk_fp8_f32 v87.l, v3, v4
	.loc	0 2109 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2109:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2091 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cvt_pk_fp8_f32 v87.h, v6, v7
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[73:74], v[86:87], v[65:72]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[86:87], v[57:64]
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[77:78], v[86:87], v[49:56]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[79:80], v[86:87], v[41:48]
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[149:150], v[86:87], v[33:40]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[151:152], v[86:87], v[25:32]
	.loc	0 2104 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2104:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[153:154], v[86:87], v[17:24]
	.loc	0 2107 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2107:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[155:156], v[86:87], v[9:16]
	v_mov_b32_e32 v87, v157
	v_dual_mov_b32 v149, v8 :: v_dual_mov_b32 v86, v2
	.loc	0 1994 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_co_i32 s28, s28, 1
	.loc	0 1994 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s28, 4
	.loc	0 1994 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_scc1 .LBB6_9
.LBB6_72:                               ;   Parent Loop BB6_11 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB6_77 Depth 3
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s28, 1
	s_mov_b32 s4, -1
	s_cbranch_scc1 .LBB6_75
; %bb.73:                               ;   in Loop: Header=BB6_72 Depth=2
	s_cmp_eq_u32 s28, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s4, s29
	s_cbranch_scc1 .LBB6_75
; %bb.74:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 1996 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1996:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cmp_eq_u32 s28, 2
	.loc	0 1996 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1996:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cselect_b32 s4, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s24, s4
	s_cselect_b32 s4, -1, 0
.LBB6_75:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_82
; %bb.76:                               ;   in Loop: Header=BB6_72 Depth=2
	v_mov_b32_e32 v73, 0
	.loc	0 2006 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshl_add_u32 v2, s28, 9, v97
	s_mov_b32 s4, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v74, v73 :: v_dual_mov_b32 v75, v73
	v_dual_mov_b32 v76, v73 :: v_dual_mov_b32 v77, v73
	v_dual_mov_b32 v78, v73 :: v_dual_mov_b32 v79, v73
	v_mov_b32_e32 v80, v73
.LBB6_77:                               ;   Parent Loop BB6_11 Depth=1
                                        ;     Parent Loop BB6_72 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 2018 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2018:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s5, s4, 6
	.loc	0 2011 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2011:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshl_add_u32 v8, s4, 12, v2
	.loc	0 2021 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, vcc_lo, v138, s5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	.loc	0 2013 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b128 v[150:153], v8
	ds_load_b128 v[154:157], v8 offset:2048
	.loc	0 2006 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_co_i32 s4, s4, 1
	.loc	0 2021 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_clause 0x3
	global_load_b64 v[6:7], v[3:4], off
	global_load_b64 v[158:159], v[3:4], off offset:16
	global_load_b64 v[160:161], v[3:4], off offset:32
	global_load_b64 v[3:4], v[3:4], off offset:48
	.loc	0 2030 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 4
	.loc	0 2024 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[150:151], v[6:7], v[73:80]
	.loc	0 2027 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[152:153], v[158:159], v[73:80]
	.loc	0 2024 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2024:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[154:155], v[160:161], v[73:80]
	.loc	0 2027 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[156:157], v[3:4], v[73:80]
	.loc	0 2006 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_scc1 .LBB6_77
; %bb.78:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_lshl_add_u32 v6, s28, 5, v96
	.loc	0 2033 48 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_lshl_b32 s4, s28, 4
	v_mov_b32_e32 v7, 0xff800000
	.loc	0 2033 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s26
	.loc	0 2038 41 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2038:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_b96 v[2:4], v6 offset:40962
	ds_load_u16_d16 v8, v6 offset:40974
	.loc	0 2034 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s4, 15
	v_or_b32_e32 v150, s4, v93
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s5, s25
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s27, s4
	.loc	0 2043 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB6_80
; %bb.79:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 2038 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2038:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_u16_d16 v7, v6 offset:40960
	v_mul_f32_e32 v73, v89, v73
	v_cmp_le_i32_e32 vcc_lo, v150, v5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s5, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v7, v7, v73, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2045 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2045:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v7, 0xff800000, v7, vcc_lo
.LBB6_80:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_mul_f32_e32 v73, v89, v74
	v_or_b32_e32 v74, 2, v150
	v_cmp_lt_i32_e32 vcc_lo, v150, v5
	v_mul_f32_e32 v75, v89, v75
	s_wait_dscnt 0x1
	v_fma_mix_f32 v73, v2, v73, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s4, v74, v5
	v_or_b32_e32 v74, 3, v150
	s_or_b32 s6, s5, vcc_lo
	v_fma_mix_f32 v2, v2, v75, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2043 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_b32 vcc_lo, s0, s6
	s_or_b32 s4, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v73, 0xff800000, v73, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v74, v5
	v_mul_f32_e32 v74, v89, v76
	v_or_b32_e32 v75, 4, v150
	s_and_b32 s4, s0, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v76, 0xff800000, v2, s4
	v_fma_mix_f32 v2, v3, v74, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v74, v89, v77
	s_or_b32 s4, s5, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v75, v5
	v_or_b32_e32 v75, 5, v150
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s0, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v77, 0xff800000, v2, s4
	v_fma_mix_f32 v2, v3, v74, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v3, v89, v78
	s_or_b32 s4, s5, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v75, v5
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s0, s4
	v_or_b32_e32 v75, 7, v150
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v74, 0xff800000, v2, s4
	v_fma_mix_f32 v2, v4, v3, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v3, 6, v150
	s_or_b32 s4, s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s4
	v_cmp_le_i32_e64 s4, v75, v5
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v78, 0xff800000, v2, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v3, v5
	v_mul_f32_e32 v2, v89, v79
.Ltmp725:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v75, v7, 0xff800000, v73
	s_or_b32 s4, s5, s4
	s_or_b32 s6, s5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v4, v2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp726:
	.loc	0 2043 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_b32 vcc_lo, s0, s6
	v_mul_f32_e32 v3, v89, v80
.Ltmp727:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v4, v75, v76, v77
	s_wait_dscnt 0x0
	v_fma_mix_f32 v3, v8, v3, neg(0) op_sel_hi:[1,0,0]
.Ltmp728:
	.loc	0 2043 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v8, 0xff800000, v2, vcc_lo
	s_and_b32 vcc_lo, s0, s4
.Ltmp729:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v2, v4, v74, v78
.Ltmp730:
	.loc	0 2077 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 2043 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v3, 0xff800000, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp731:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v2, v2, v8, v3
.Ltmp732:
	.loc	0 1739 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1739:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2052:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_mov_b32_e32 v4, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v4, v4, s20, 0xfedcba98
.Ltmp733:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2053:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v2, v86, v2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v4, v7, v2 :: v_dual_sub_f32 v7, v73, v2
	v_dual_sub_f32 v3, v3, v2 :: v_dual_sub_f32 v8, v8, v2
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_dual_mul_f32 v4, 0x3fb8aa3b, v4 :: v_dual_mul_f32 v3, 0x3fb8aa3b, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v8, 0x3fb8aa3b, v8
	v_exp_f32_e32 v75, v3
	v_dual_sub_f32 v3, v76, v2 :: v_dual_sub_f32 v76, v77, v2
	v_mul_f32_e32 v7, 0x3fb8aa3b, v7
	v_exp_f32_e32 v77, v4
	v_exp_f32_e32 v8, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v76, 0x3fb8aa3b, v76
	v_exp_f32_e32 v79, v7
	v_dual_sub_f32 v7, v78, v2 :: v_dual_mul_f32 v78, 0x3fb8aa3b, v3
.Ltmp734:
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v80, v75, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v77, v77, 0, vcc_lo
	v_cndmask_b32_e64 v8, v8, 0, vcc_lo
	v_exp_f32_e32 v78, v78
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e64 v79, v79, 0, vcc_lo
	v_cndmask_b32_e64 v78, v78, 0, vcc_lo
	v_add_nc_u32_e32 v73, 0xa080, v6
	.loc	0 2059 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_nc_u32_e32 v6, 0xa088, v6
	ds_load_2addr_b32 v[3:4], v73 offset1:1
	v_sub_f32_e32 v73, v74, v2
	v_exp_f32_e32 v74, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v73, 0x3fb8aa3b, v73
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cndmask_b32_e64 v151, v74, 0, vcc_lo
	v_mul_f32_e32 v7, 0x3fb8aa3b, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v76, v7
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v75, v3, v79, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v74, v4, v78, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cndmask_b32_e64 v152, v73, 0, vcc_lo
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_mix_f32 v73, v4, v151, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2063 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cndmask_b32_e64 v150, v76, 0, vcc_lo
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_fma_mix_f32 v76, v3, v77, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_f32_e32 v3, v77, v79
	.loc	0 2059 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	ds_load_2addr_b32 v[6:7], v6 offset1:1
.Ltmp735:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v77, v76, 0, v75
.Ltmp736:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_f32_e32 v78, v78, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp737:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v77, v77, v74, v73
.Ltmp738:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_f32_e32 v78, v151, v78
	.loc	0 2067 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v3, v6, v152, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v4, v6, v150, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v6, v7, v8, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v7, v7, v80, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp739:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max3_num_f32 v77, v77, v3, v4
	v_max3_num_f32 v79, v77, v6, v7
.Ltmp740:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_f32_e32 v78, v152, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v77, v150, v78 :: v_dual_mov_b32 v78, v79
	v_add_f32_e32 v8, v8, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp741:
	.loc	0 1739 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1739:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2075:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_permlanex16_b32 v78, v78, s20, 0xfedcba98
.Ltmp742:
	.loc	0 2066 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_add_f32 v77, v80, v8 :: v_dual_max_num_f32 v8, v78, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
.Ltmp743:
	.loc	0 1739 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1739:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_dual_mov_b32 v78, v77 :: v_dual_max_num_f32 v79, v79, v8
	v_mov_b32_e32 v8, v149
	v_permlanex16_b32 v78, v78, s20, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp744:
	.loc	0 2077 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_cmpx_lt_f32_e32 0, v79
	s_cbranch_execz .LBB6_71
; %bb.81:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	0 2078 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_div_scale_f32 v8, null, 0x43e00000, 0x43e00000, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v80, v8
	v_fma_f32 v150, -v8, v80, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v80, v150, v80
	v_div_scale_f32 v150, vcc_lo, v79, 0x43e00000, v79
	v_mul_f32_e32 v151, v150, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v8, v151, v150
	v_fmac_f32_e32 v151, v152, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v8, -v8, v151, v150
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v8, v8, v80, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v8, v8, 0x43e00000, v79
.Ltmp745:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ] ]
	v_max_num_f32_e32 v8, 0x1f800000, v8
	s_branch .LBB6_71
.Ltmp746:
.LBB6_82:                               ;   in Loop: Header=BB6_72 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v86, v2
	.loc	0 1994 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_add_co_i32 s28, s28, 1
	.loc	0 1994 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s28, 4
	.loc	0 1994 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_cbranch_scc0 .LBB6_72
	s_branch .LBB6_9
.LBB6_83:
	.loc	0 2135 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2135:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_or_b32_e32 v0, s13, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_and_b32 s2, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB6_85
; %bb.84:
	.loc	0 2139 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2139:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_u64_u32 v[0:1], null, s17, v81, s[12:13]
	.loc	0 2141 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2140 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2140:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	0 2141 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	.loc	0 2143 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2143:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	global_store_b64 v[0:1], v[86:87], off
.LBB6_85:
	.loc	0 0 24 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 2146 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2146:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_87
; %bb.86:
	.loc	0 2149 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2149:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mad_co_u64_u32 v[1:2], null, s17, v81, s[12:13]
	.loc	0 2151 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mov_b32 v3, 0 :: v_dual_mul_f32 v0, v149, v65
	.loc	0 2158 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshlrev_b32_e32 v6, 2, v93
	.loc	0 2158 53 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v4, v57, v149 :: v_dual_mul_f32 v5, v58, v149
	v_mul_f32_e32 v8, v49, v149
	v_dual_mul_f32 v41, v41, v149 :: v_dual_mul_f32 v42, v42, v149
	.loc	0 2150 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2150:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_lo_u32 v2, 0x102, v1
	.loc	0 2158 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v1, v149, v66
	.loc	0 2158 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshl_or_b32 v6, s13, 9, v6
	.loc	0 2158 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_dual_mul_f32 v33, v33, v149 :: v_dual_mul_f32 v34, v34, v149
	v_dual_mul_f32 v25, v25, v149 :: v_dual_mul_f32 v26, v26, v149
	v_mul_f32_e32 v49, v9, v149
	.loc	0 2151 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	.loc	0 2158 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v9, v50, v149
	v_mul_f32_e32 v50, v10, v149
	v_dual_mul_f32 v10, v51, v149 :: v_dual_mul_f32 v7, v60, v149
	v_dual_mul_f32 v35, v35, v149 :: v_dual_mul_f32 v44, v44, v149
	.loc	0 2151 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_co_u32 v2, vcc_lo, s10, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s11, v3, vcc_lo
	.loc	0 2158 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v43, v43, v149
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	0 2158 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_add_co_u32 v65, vcc_lo, v2, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, 0, v3, vcc_lo
	.loc	0 2158 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v2, v149, v67
	v_dual_mul_f32 v6, v59, v149 :: v_dual_mul_f32 v3, v149, v68
	v_dual_mul_f32 v27, v27, v149 :: v_dual_mul_f32 v36, v36, v149
	v_dual_mul_f32 v19, v19, v149 :: v_dual_mul_f32 v28, v28, v149
	v_dual_mul_f32 v51, v11, v149 :: v_dual_mul_f32 v20, v20, v149
	v_mul_f32_e32 v11, v52, v149
	v_dual_mul_f32 v52, v12, v149 :: v_dual_mul_f32 v53, v53, v149
	v_dual_mul_f32 v57, v149, v69 :: v_dual_mul_f32 v58, v149, v70
	v_mul_f32_e32 v12, v61, v149
	v_dual_mul_f32 v45, v45, v149 :: v_dual_mul_f32 v54, v54, v149
	v_dual_mul_f32 v37, v37, v149 :: v_dual_mul_f32 v46, v46, v149
	v_dual_mul_f32 v29, v29, v149 :: v_dual_mul_f32 v38, v38, v149
	v_dual_mul_f32 v21, v21, v149 :: v_dual_mul_f32 v30, v30, v149
	v_dual_mul_f32 v61, v13, v149 :: v_dual_mul_f32 v22, v22, v149
	v_mul_f32_e32 v13, v62, v149
	v_dual_mul_f32 v62, v14, v149 :: v_dual_mul_f32 v59, v149, v71
	v_mul_f32_e32 v14, v63, v149
	v_dual_mul_f32 v55, v55, v149 :: v_dual_mul_f32 v60, v149, v72
	v_dual_mul_f32 v47, v47, v149 :: v_dual_mul_f32 v56, v56, v149
	v_dual_mul_f32 v39, v39, v149 :: v_dual_mul_f32 v48, v48, v149
	v_dual_mul_f32 v31, v31, v149 :: v_dual_mul_f32 v40, v40, v149
	v_dual_mul_f32 v23, v23, v149 :: v_dual_mul_f32 v32, v32, v149
	v_dual_mul_f32 v63, v15, v149 :: v_dual_mul_f32 v24, v24, v149
	v_mul_f32_e32 v15, v64, v149
	v_dual_mul_f32 v17, v17, v149 :: v_dual_mul_f32 v18, v18, v149
	.loc	0 2158 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_clause 0x9
	global_store_b128 v[65:66], v[0:3], off offset:8
	global_store_b128 v[65:66], v[57:60], off offset:24
	global_store_b128 v[65:66], v[4:7], off offset:72
	global_store_b128 v[65:66], v[12:15], off offset:88
	global_store_b128 v[65:66], v[8:11], off offset:136
	global_store_b128 v[65:66], v[53:56], off offset:152
	global_store_b128 v[65:66], v[41:44], off offset:200
	global_store_b128 v[65:66], v[45:48], off offset:216
	global_store_b128 v[65:66], v[33:36], off offset:264
	global_store_b128 v[65:66], v[37:40], off offset:280
	.loc	0 2158 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	v_mul_f32_e32 v64, v16, v149
	.loc	0 2158 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2158:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2233:5 ]
	s_clause 0x5
	global_store_b128 v[65:66], v[25:28], off offset:328
	global_store_b128 v[65:66], v[29:32], off offset:344
	global_store_b128 v[65:66], v[17:20], off offset:392
	global_store_b128 v[65:66], v[21:24], off offset:408
	global_store_b128 v[65:66], v[49:52], off offset:456
	global_store_b128 v[65:66], v[61:64], off offset:472
.Ltmp747:
.LBB6_87:
	.loc	0 2237 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2237:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp748:
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
		.amdhsa_next_free_vgpr 167
		.amdhsa_next_free_sgpr 30
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_vgpr, 167
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 30
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11652
; TotalNumSgprs: 32
; NumVgprs: 167
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 167
; Occupancy: 9
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
	.loc	0 2248 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2248:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	0 2248 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2248:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB7_10
; %bb.1:
	.loc	0 2253 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2253:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	0 2256 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2255 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2255:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	0 2256 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_10
; %bb.2:
	.loc	0 2248 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2248:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	0 2259 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:5
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
	.loc	0 2262 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2262:22
	global_load_b32 v3, v[6:7], off
.Ltmp749:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2262:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp750:
	.loc	0 2259 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp751:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2262:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp752:
	.loc	0 2259 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:5
	s_cbranch_scc0 .LBB7_3
; %bb.4:
	.loc	0 2254 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	0 2264 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:5
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
	.loc	0 2276 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2276:13
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
	.loc	0 2264 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	0 2276 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2276:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	0 2275 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2275:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	0 2264 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	0 2276 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2276:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 2275 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2275:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	0 2276 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2276:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	0 2264 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:5
	s_or_b32 s1, vcc_lo, s1
	.loc	0 2276 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2276:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	0 2264 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	0 2275 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2275:44
	global_store_b32 v[5:6], v11, off
	.loc	0 2264 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2264:5
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
	.loc	0 2273 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2273:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	0 2267 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2267:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	0 2273 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2273:24
	global_load_b32 v15, v[15:16], off
	.loc	0 2272 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2272:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	0 2267 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2267:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 2273 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2273:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	0 2267 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2267:9
	s_cbranch_scc1 .LBB7_5
.LBB7_8:                                ;   Parent Loop BB7_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 2271 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	0 2271 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:17
	s_mov_b32 s2, exec_lo
	.loc	0 2271 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	0 2271 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:17
	s_cbranch_execz .LBB7_7
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=2
	.loc	0 2271 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:41
	global_load_b32 v14, v[5:6], off
	.loc	0 2271 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp753:
	.loc	2 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	2 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2271:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB7_7
.Ltmp754:
.LBB7_10:
	.loc	0 2278 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2278:1
	s_endpgm
.Ltmp755:
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
	.type	__hip_cuid_671dadf8546874d7,@object ; @__hip_cuid_671dadf8546874d7
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_671dadf8546874d7
__hip_cuid_671dadf8546874d7:
	.byte	0                               ; 0x0
	.size	__hip_cuid_671dadf8546874d7, 1

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
	.byte	1                               ; Abbrev [1] 0xc:0xa90 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x5e7:0x250 DW_TAG_subprogram
	.byte	64                              ; DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	25                              ; DW_AT_name
	.byte	5                               ; Abbrev [5] 0x5ee:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	65                              ; DW_AT_low_pc
	.long	.Ltmp242-.Ltmp241               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2193                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5fc:0x23a DW_TAG_inlined_subroutine
	.long	1507                            ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2197                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x606:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	66                              ; DW_AT_low_pc
	.long	.Ltmp247-.Ltmp246               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1841                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x614:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	29                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1851                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x61e:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	67                              ; DW_AT_low_pc
	.long	.Ltmp249-.Ltmp248               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x62d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	30                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1853                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x637:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	31                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1854                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x641:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	32                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x64b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	33                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1852                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x655:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	68                              ; DW_AT_low_pc
	.long	.Ltmp262-.Ltmp261               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x663:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	69                              ; DW_AT_low_pc
	.long	.Ltmp269-.Ltmp268               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1851                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x671:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	70                              ; DW_AT_low_pc
	.long	.Ltmp270-.Ltmp269               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x67f:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	71                              ; DW_AT_low_pc
	.long	.Ltmp276-.Ltmp275               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x68d:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	72                              ; DW_AT_low_pc
	.long	.Ltmp279-.Ltmp278               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1852                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x69b:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	73                              ; DW_AT_low_pc
	.long	.Ltmp280-.Ltmp279               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6a9:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	74                              ; DW_AT_low_pc
	.long	.Ltmp282-.Ltmp281               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6b7:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	75                              ; DW_AT_low_pc
	.long	.Ltmp283-.Ltmp282               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1853                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6c5:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	76                              ; DW_AT_low_pc
	.long	.Ltmp284-.Ltmp283               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6d3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	77                              ; DW_AT_low_pc
	.long	.Ltmp287-.Ltmp286               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6e1:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	78                              ; DW_AT_low_pc
	.long	.Ltmp288-.Ltmp287               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1854                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6ef:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	79                              ; DW_AT_low_pc
	.long	.Ltmp289-.Ltmp288               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6fd:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	80                              ; DW_AT_low_pc
	.long	.Ltmp291-.Ltmp290               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x70b:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp292-.Ltmp291               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x719:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x727:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp296-.Ltmp295               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2117                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x735:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp296-.Ltmp295               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x743:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp296-.Ltmp295               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x753:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1917                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x75d:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x767:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x773:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1941                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x77d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1944                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x787:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	37                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1947                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x791:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1951                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x79b:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1961                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7a5:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7af:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x7bb:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1980                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7c5:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7cf:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x7db:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2051                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7e5:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp477-.Ltmp476               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2052                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7f3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp478-.Ltmp477               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2053                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x801:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2074                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x80b:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	86                              ; DW_AT_low_pc
	.long	.Ltmp488-.Ltmp487               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2075                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x819:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	87                              ; DW_AT_low_pc
	.long	.Ltmp490-.Ltmp489               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2069                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x827:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	88                              ; DW_AT_low_pc
	.long	.Ltmp492-.Ltmp491               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2078                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x837:0x2 DW_TAG_subprogram
	.byte	19                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x839:0x242 DW_TAG_subprogram
	.byte	89                              ; DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	26                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x840:0x23a DW_TAG_inlined_subroutine
	.long	2103                            ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2233                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x84a:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	90                              ; DW_AT_low_pc
	.long	.Ltmp499-.Ltmp498               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1841                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x858:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1851                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x862:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	91                              ; DW_AT_low_pc
	.long	.Ltmp501-.Ltmp500               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x871:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1852                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x87b:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1853                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x885:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	92                              ; DW_AT_low_pc
	.long	.Ltmp511-.Ltmp510               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x893:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	93                              ; DW_AT_low_pc
	.long	.Ltmp518-.Ltmp517               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1851                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8a1:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	94                              ; DW_AT_low_pc
	.long	.Ltmp519-.Ltmp518               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x8af:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1854                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x8b9:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8c3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	95                              ; DW_AT_low_pc
	.long	.Ltmp528-.Ltmp527               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8d1:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	96                              ; DW_AT_low_pc
	.long	.Ltmp530-.Ltmp529               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1852                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8df:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	97                              ; DW_AT_low_pc
	.long	.Ltmp531-.Ltmp530               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8ed:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	98                              ; DW_AT_low_pc
	.long	.Ltmp534-.Ltmp533               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8fb:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	99                              ; DW_AT_low_pc
	.long	.Ltmp537-.Ltmp536               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1853                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x909:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	100                             ; DW_AT_low_pc
	.long	.Ltmp538-.Ltmp537               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x917:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	101                             ; DW_AT_low_pc
	.long	.Ltmp541-.Ltmp540               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x925:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	102                             ; DW_AT_low_pc
	.long	.Ltmp543-.Ltmp542               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1854                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x933:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	103                             ; DW_AT_low_pc
	.long	.Ltmp544-.Ltmp543               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x941:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	104                             ; DW_AT_low_pc
	.long	.Ltmp546-.Ltmp545               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x94f:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp547-.Ltmp546               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x95d:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp548-.Ltmp547               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x96b:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp550-.Ltmp549               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2117                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x979:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp550-.Ltmp549               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x987:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp550-.Ltmp549               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x997:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1917                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9a1:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9ab:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x9b7:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1941                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9c1:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1944                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9cb:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1947                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9d5:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1951                            ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9df:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1961                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9e9:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9f3:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x9ff:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1980                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xa09:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa13:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xa1f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2051                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa29:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp733-.Ltmp732               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2052                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa37:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp734-.Ltmp733               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2053                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa45:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	57                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2074                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa4f:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	110                             ; DW_AT_low_pc
	.long	.Ltmp742-.Ltmp741               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2075                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa5d:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp744-.Ltmp743               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2069                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa6b:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	112                             ; DW_AT_low_pc
	.long	.Ltmp746-.Ltmp745               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2078                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0xa7b:0x20 DW_TAG_subprogram
	.byte	113                             ; DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	27                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0xa82:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2262                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa8c:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	114                             ; DW_AT_low_pc
	.long	.Ltmp754-.Ltmp753               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2271                            ; DW_AT_call_line
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
	.long	59                              ; Offset entry count
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
	.uleb128 .Ltmp258-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp259-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp493-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges29:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp248-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp250-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp255-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp256-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp260-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp261-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges30:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp250-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp251-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp264-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp265-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp266-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp280-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp281-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges31:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp251-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp268-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp271-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp276-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp285-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp286-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges32:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp271-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp272-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp273-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp278-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp289-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp290-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges33:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp256-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp257-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp259-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp260-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp262-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp263-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp272-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp273-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges34:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp297-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp298-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp383-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp384-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges35:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp299-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp300-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp301-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp302-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp309-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp310-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp311-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp312-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp319-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp329-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp330-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp339-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp342-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp349-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp350-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp351-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp352-.Lfunc_begin0         ;   ending offset
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
	.uleb128 .Ltmp385-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp386-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp387-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp388-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp395-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp396-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp397-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp398-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp405-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp406-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp407-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp408-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp415-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp416-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp417-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp418-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp425-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp426-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp427-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp428-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp435-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp436-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp437-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp438-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp445-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp446-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp447-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp448-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp455-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp456-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp457-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp458-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges36:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp300-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp301-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp302-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp303-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp310-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp311-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp312-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp313-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp330-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp333-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp342-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp343-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp350-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp351-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp352-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp353-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp360-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp361-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp362-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp363-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp372-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp373-.Lfunc_begin0         ;   ending offset
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp436-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp437-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp438-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp439-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp446-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp447-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp448-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp449-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp456-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp457-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp458-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp459-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges37:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp304-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp305-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp306-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp307-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp314-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp315-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp316-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp317-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp324-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp334-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp344-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp345-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp346-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp347-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp354-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp355-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp356-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp357-.Lfunc_begin0         ;   ending offset
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
	.uleb128 .Ltmp390-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp391-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp392-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp393-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp400-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp401-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp402-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp403-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp410-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp411-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp412-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp413-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp420-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp421-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp422-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp423-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp430-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp431-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp432-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp433-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp440-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp441-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp442-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp443-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp450-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp451-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp452-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp453-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp460-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp461-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp462-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp463-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges38:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp305-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp306-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp307-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp308-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp315-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp316-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp317-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp318-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp328-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp338-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp345-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp346-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp347-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp348-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp355-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp356-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp357-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp358-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp365-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp366-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp367-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp368-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp375-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp376-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp377-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp378-.Lfunc_begin0         ;   ending offset
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp441-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp442-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp443-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp444-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp451-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp452-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp453-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp454-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp461-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp462-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp463-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp464-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges39:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp379-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp380-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp381-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp382-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp465-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp466-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges40:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp467-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp468-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp469-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp470-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges41:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp471-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp472-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp473-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp474-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp475-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp476-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges42:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp479-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp480-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp482-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp483-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp484-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp485-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp486-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges43:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp495-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp496-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp497-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp506-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp507-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp512-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp513-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp514-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp515-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp516-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp517-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp528-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp529-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp531-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp532-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp534-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp535-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp747-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges44:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp500-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp501-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp502-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp503-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp505-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp506-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp507-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp508-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp509-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp510-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges45:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp503-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp504-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp508-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp509-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp524-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp525-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp526-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp527-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges46:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp504-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp505-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp513-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp514-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp515-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp516-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp521-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp522-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp532-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp533-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges47:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp519-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp520-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp522-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp523-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp539-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp540-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges48:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp520-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp521-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp523-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp524-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp525-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp526-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp544-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp545-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges49:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp551-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp552-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp637-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp638-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges50:
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp603-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp604-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp605-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp606-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp613-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp614-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp615-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp616-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp623-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp624-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp625-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp626-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp639-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp640-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp641-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp642-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp649-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp650-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp651-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp652-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp659-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp660-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp661-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp662-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp669-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp670-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp671-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp672-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp679-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp680-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp681-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp682-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp689-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp690-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp691-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp692-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp699-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp700-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp701-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp702-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp709-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp710-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp711-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp712-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges51:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp554-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp555-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp556-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp557-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp564-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp565-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp566-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp567-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp574-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp575-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp576-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp577-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp584-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp585-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp586-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp587-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp594-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp595-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp596-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp597-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp604-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp605-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp606-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp607-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp614-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp615-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp616-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp617-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp624-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp625-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp626-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp627-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp640-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp641-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp642-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp643-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp650-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp651-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp652-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp653-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp660-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp661-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp662-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp663-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp670-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp671-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp672-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp673-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp680-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp681-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp682-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp683-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp690-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp691-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp692-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp693-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp700-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp701-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp702-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp703-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp710-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp711-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp712-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp713-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges52:
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp598-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp599-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp600-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp601-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp608-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp609-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp610-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp611-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp618-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp619-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp620-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp621-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp628-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp629-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp630-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp631-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp644-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp645-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp646-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp647-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp654-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp655-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp656-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp657-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp664-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp665-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp666-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp667-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp674-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp675-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp676-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp677-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp684-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp685-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp686-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp687-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp694-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp695-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp696-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp697-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp704-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp705-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp706-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp707-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp714-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp715-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp716-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp717-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges53:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp559-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp560-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp561-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp562-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp569-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp570-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp571-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp572-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp579-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp580-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp581-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp582-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp589-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp590-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp591-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp592-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp599-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp600-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp601-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp602-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp609-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp610-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp611-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp612-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp619-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp620-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp621-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp622-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp629-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp630-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp631-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp632-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp645-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp646-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp647-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp648-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp655-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp656-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp657-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp658-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp665-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp666-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp667-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp668-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp675-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp676-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp677-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp678-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp685-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp686-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp687-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp688-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp695-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp696-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp697-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp698-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp705-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp706-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp707-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp708-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp715-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp716-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp717-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp718-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges54:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp633-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp634-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp635-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp636-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp719-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp720-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges55:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp721-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp722-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp723-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp724-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges56:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp725-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp726-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp727-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp728-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp729-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp730-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp731-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp732-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges57:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp735-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp736-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp737-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp738-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp739-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp740-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges58:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp749-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp750-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp751-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp752-.Lfunc_begin0         ;   ending offset
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
	.asciz	"/home/kaden/ClaudeCode/warpfront/wt-attnsmall" ; string offset=159 ; /home/kaden/ClaudeCode/warpfront/wt-attnsmall
.Linfo_string3:
	.asciz	"fa2_stageb_nbody<false>"       ; string offset=205 ; fa2_stageb_nbody<false>
.Linfo_string4:
	.asciz	"__lane_id"                     ; string offset=229 ; __lane_id
.Linfo_string5:
	.asciz	"__shfl_xor"                    ; string offset=239 ; __shfl_xor
.Linfo_string6:
	.asciz	"max"                           ; string offset=250 ; max
.Linfo_string7:
	.asciz	"min"                           ; string offset=254 ; min
.Linfo_string8:
	.asciz	"__work_group_barrier"          ; string offset=258 ; __work_group_barrier
.Linfo_string9:
	.asciz	"__barrier"                     ; string offset=279 ; __barrier
.Linfo_string10:
	.asciz	"__syncthreads"                 ; string offset=289 ; __syncthreads
.Linfo_string11:
	.asciz	"fa2_scale_n"                   ; string offset=303 ; fa2_scale_n
.Linfo_string12:
	.asciz	"fmaxf"                         ; string offset=315 ; fmaxf
.Linfo_string13:
	.asciz	"__hip_get_thread_idx_x"        ; string offset=321 ; __hip_get_thread_idx_x
.Linfo_string14:
	.asciz	"__get_x"                       ; string offset=344 ; __get_x
.Linfo_string15:
	.asciz	"fa2_stageb_nbody<true>"        ; string offset=352 ; fa2_stageb_nbody<true>
.Linfo_string16:
	.asciz	"__expf"                        ; string offset=375 ; __expf
.Linfo_string17:
	.asciz	"fa2_stageb_packet_body<false, false>" ; string offset=382 ; fa2_stageb_packet_body<false, false>
.Linfo_string18:
	.asciz	"fa2_pkt_xor16"                 ; string offset=419 ; fa2_pkt_xor16
.Linfo_string19:
	.asciz	"fa2_stageb_packet_body<true, false>" ; string offset=433 ; fa2_stageb_packet_body<true, false>
.Linfo_string20:
	.asciz	"attention_fp8_e4m3_fa2_gqa_gfx1201" ; string offset=469 ; attention_fp8_e4m3_fa2_gqa_gfx1201
.Linfo_string21:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201" ; string offset=504 ; attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
.Linfo_string22:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201" ; string offset=552 ; attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
.Linfo_string23:
	.asciz	"attention_fp8_e4m3_fa2_gqa_partial_gfx1201" ; string offset=600 ; attention_fp8_e4m3_fa2_gqa_partial_gfx1201
.Linfo_string24:
	.asciz	"attention_fp8_e4m3_fa2_gqa_merge_gfx1201" ; string offset=643 ; attention_fp8_e4m3_fa2_gqa_merge_gfx1201
.Linfo_string25:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_gfx1201" ; string offset=684 ; attention_fp8_e4m3_fa2_gqa_packet_gfx1201
.Linfo_string26:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201" ; string offset=726 ; attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
.Linfo_string27:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201" ; string offset=776 ; attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
	.quad	.Ltmp246
	.quad	.Ltmp248
	.quad	.Ltmp261
	.quad	.Ltmp268
	.quad	.Ltmp269
	.quad	.Ltmp275
	.quad	.Ltmp278
	.quad	.Ltmp279
	.quad	.Ltmp281
	.quad	.Ltmp282
	.quad	.Ltmp283
	.quad	.Ltmp286
	.quad	.Ltmp287
	.quad	.Ltmp288
	.quad	.Ltmp290
	.quad	.Ltmp291
	.quad	.Ltmp293
	.quad	.Ltmp295
	.quad	.Ltmp476
	.quad	.Ltmp477
	.quad	.Ltmp487
	.quad	.Ltmp489
	.quad	.Ltmp491
	.quad	.Lfunc_begin6
	.quad	.Ltmp498
	.quad	.Ltmp500
	.quad	.Ltmp510
	.quad	.Ltmp517
	.quad	.Ltmp518
	.quad	.Ltmp527
	.quad	.Ltmp529
	.quad	.Ltmp530
	.quad	.Ltmp533
	.quad	.Ltmp536
	.quad	.Ltmp537
	.quad	.Ltmp540
	.quad	.Ltmp542
	.quad	.Ltmp543
	.quad	.Ltmp545
	.quad	.Ltmp546
	.quad	.Ltmp547
	.quad	.Ltmp549
	.quad	.Ltmp732
	.quad	.Ltmp733
	.quad	.Ltmp741
	.quad	.Ltmp743
	.quad	.Ltmp745
	.quad	.Lfunc_begin7
	.quad	.Ltmp753
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_671dadf8546874d7
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
    .max_flat_workgroup_size: 128
    .name:           attention_fp8_e4m3_fa2_gqa_packet_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     164
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
    .name:           attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     167
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
