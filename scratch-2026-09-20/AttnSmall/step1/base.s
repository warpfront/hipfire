	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_gfx1201:     ; @attention_fp8_e4m3_fa2_gqa_gfx1201
.Lfunc_begin0:
	.file	0 "/home/kaden/ClaudeCode/warpfront/wt-lloyd" "kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0xfbbb642c938a57c8453ff559be1d830f
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
	.loc	0 2171 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2171:17
	s_load_b128 s[4:7], s[0:1], 0x30
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	0 2171 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2171:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_63
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s21, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2174 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2174:14
	s_cmp_gt_i32 s21, 3
	s_cbranch_scc1 .LBB5_63
; %bb.2:
	.loc	0 0 14 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:14
	s_load_b32 s25, s[0:1], 0x40
	s_lshr_b32 s2, ttmp7, 7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, 0x1fffe00
	.loc	0 2176 41 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2176:41
	s_cmp_gt_i32 s7, 0x200
	.loc	0 2176 30 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2176:30
	s_cselect_b32 s23, s2, 0
	.loc	0 2177 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2177:22
	s_cmp_le_i32 s7, s23
	s_cbranch_scc1 .LBB5_63
; %bb.3:
	.loc	0 2179 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2179:46
	s_sub_co_i32 s2, s7, s23
	.loc	0 2180 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2180:30
	s_lshl_b32 s22, ttmp9, 7
.Ltmp241:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2179:26 ]
	s_min_i32 s24, s2, 0x200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
.Ltmp242:
	.loc	0 2181 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2181:23
	s_mul_i32 s20, s24, 6
	.loc	0 2181 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2181:18
	s_cmp_ge_i32 s22, s20
	s_cbranch_scc1 .LBB5_63
; %bb.4:
.Ltmp243:
	.loc	0 1767 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshrrev_b32_e32 v8, 5, v0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v151, s25 :: v_dual_and_b32 v10, 15, v0
.Ltmp244:
	.loc	0 2171 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2171:17
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b128 s[16:19], s[0:1], 0x20
.Ltmp245:
	.loc	0 1778 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshlrev_b32_e32 v11, 4, v8
	.loc	0 1770 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1770:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_bfe_u32 v7, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1778 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v1, v11, v10
	.loc	0 1778 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_nc_u32_e32 v3, s22, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1779 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_hi_i32 v1, 0x2aaaaaab, v3
	.loc	0 1782 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1782:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmp_le_i32_e64 s1, s20, v3
	v_cmp_gt_i32_e64 s0, s20, v3
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshrrev_b32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, v1, v2
	.loc	0 1781 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_lo_u32 v1, v2, 6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v3, v1
	.loc	0 1781 31 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[145:146], null, s21, 6, v[1:2]
	.loc	0 1780 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1780:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_nc_u32_e32 v146, s23, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v146, 24, v[145:146]
	v_lshlrev_b32_e32 v9, 8, v1
	.loc	0 1800 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1800:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s26, s0
	s_cbranch_execz .LBB5_8
; %bb.5:
	.loc	0 1801 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1801:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b32_e32 v2, 0
.Ltmp246:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_mov_b32 s2, 0x76543210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
.Ltmp247:
	.loc	0 1801 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1801:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshlrev_b64_e32 v[136:137], 10, v[1:2]
	.loc	0 1805 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshlrev_b32_e32 v1, 5, v7
	.loc	0 1801 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1801:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_kmcnt 0x0
	v_add_co_u32 v3, vcc_lo, s8, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v137, vcc_lo
	.loc	0 1805 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v132, vcc_lo, v3, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v133, null, 0, v4, vcc_lo
	s_clause 0x1f
	global_load_b128 v[3:6], v[132:133], off
	global_load_b128 v[12:15], v[132:133], off offset:16
	global_load_b128 v[16:19], v[132:133], off offset:64
	global_load_b128 v[20:23], v[132:133], off offset:80
	global_load_b128 v[24:27], v[132:133], off offset:128
	global_load_b128 v[28:31], v[132:133], off offset:144
	global_load_b128 v[32:35], v[132:133], off offset:192
	global_load_b128 v[36:39], v[132:133], off offset:208
	global_load_b128 v[40:43], v[132:133], off offset:256
	global_load_b128 v[44:47], v[132:133], off offset:272
	global_load_b128 v[48:51], v[132:133], off offset:320
	global_load_b128 v[52:55], v[132:133], off offset:336
	global_load_b128 v[56:59], v[132:133], off offset:384
	global_load_b128 v[60:63], v[132:133], off offset:400
	global_load_b128 v[64:67], v[132:133], off offset:448
	global_load_b128 v[68:71], v[132:133], off offset:464
	global_load_b128 v[72:75], v[132:133], off offset:512
	global_load_b128 v[76:79], v[132:133], off offset:528
	global_load_b128 v[80:83], v[132:133], off offset:576
	global_load_b128 v[84:87], v[132:133], off offset:592
	global_load_b128 v[88:91], v[132:133], off offset:640
	global_load_b128 v[92:95], v[132:133], off offset:656
	global_load_b128 v[96:99], v[132:133], off offset:704
	global_load_b128 v[100:103], v[132:133], off offset:720
	global_load_b128 v[104:107], v[132:133], off offset:768
	global_load_b128 v[108:111], v[132:133], off offset:784
	global_load_b128 v[112:115], v[132:133], off offset:832
	global_load_b128 v[116:119], v[132:133], off offset:848
	global_load_b128 v[120:123], v[132:133], off offset:896
	global_load_b128 v[124:127], v[132:133], off offset:912
	global_load_b128 v[128:131], v[132:133], off offset:960
	global_load_b128 v[132:135], v[132:133], off offset:976
.Ltmp248:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1f
	v_max3_num_f32 v1, |v3|, 0, |v4|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp249:
	.loc	2 454 44 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v5|, |v6|
.Ltmp250:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1e
	v_max3_num_f32 v1, v1, |v12|, |v13|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp251:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v14|, |v15|
.Ltmp252:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1d
	v_max3_num_f32 v1, v1, |v16|, |v17|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp253:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v18|, |v19|
.Ltmp254:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1c
	v_max3_num_f32 v1, v1, |v20|, |v21|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp255:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v22|, |v23|
.Ltmp256:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1b
	v_max3_num_f32 v1, v1, |v24|, |v25|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp257:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v26|, |v27|
.Ltmp258:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1a
	v_max3_num_f32 v1, v1, |v28|, |v29|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp259:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v30|, |v31|
.Ltmp260:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x19
	v_max3_num_f32 v1, v1, |v32|, |v33|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp261:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v34|, |v35|
.Ltmp262:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x18
	v_max3_num_f32 v1, v1, |v36|, |v37|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp263:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v38|, |v39|
.Ltmp264:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x17
	v_max3_num_f32 v1, v1, |v40|, |v41|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp265:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v42|, |v43|
.Ltmp266:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x16
	v_max3_num_f32 v1, v1, |v44|, |v45|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp267:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v46|, |v47|
.Ltmp268:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x15
	v_max3_num_f32 v1, v1, |v48|, |v49|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp269:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v50|, |v51|
.Ltmp270:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x14
	v_max3_num_f32 v1, v1, |v52|, |v53|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp271:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v54|, |v55|
.Ltmp272:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x13
	v_max3_num_f32 v1, v1, |v56|, |v57|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp273:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v58|, |v59|
.Ltmp274:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x12
	v_max3_num_f32 v1, v1, |v60|, |v61|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp275:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v62|, |v63|
.Ltmp276:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x11
	v_max3_num_f32 v1, v1, |v64|, |v65|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp277:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v66|, |v67|
.Ltmp278:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x10
	v_max3_num_f32 v1, v1, |v68|, |v69|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp279:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v70|, |v71|
.Ltmp280:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0xf
	v_max3_num_f32 v1, v1, |v72|, |v73|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp281:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v74|, |v75|
.Ltmp282:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0xe
	v_max3_num_f32 v1, v1, |v76|, |v77|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp283:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v78|, |v79|
.Ltmp284:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0xd
	v_max3_num_f32 v1, v1, |v80|, |v81|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp285:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v82|, |v83|
.Ltmp286:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0xc
	v_max3_num_f32 v1, v1, |v84|, |v85|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp287:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v86|, |v87|
.Ltmp288:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0xb
	v_max3_num_f32 v1, v1, |v88|, |v89|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp289:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v90|, |v91|
.Ltmp290:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0xa
	v_max3_num_f32 v1, v1, |v92|, |v93|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp291:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v94|, |v95|
.Ltmp292:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x9
	v_max3_num_f32 v1, v1, |v96|, |v97|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp293:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v98|, |v99|
.Ltmp294:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x8
	v_max3_num_f32 v1, v1, |v100|, |v101|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp295:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v102|, |v103|
.Ltmp296:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x7
	v_max3_num_f32 v1, v1, |v104|, |v105|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp297:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v106|, |v107|
.Ltmp298:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x6
	v_max3_num_f32 v1, v1, |v108|, |v109|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp299:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v110|, |v111|
.Ltmp300:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x5
	v_max3_num_f32 v1, v1, |v112|, |v113|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp301:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v114|, |v115|
.Ltmp302:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x4
	v_max3_num_f32 v1, v1, |v116|, |v117|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp303:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v118|, |v119|
.Ltmp304:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x3
	v_max3_num_f32 v1, v1, |v120|, |v121|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp305:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v122|, |v123|
.Ltmp306:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x2
	v_max3_num_f32 v1, v1, |v124|, |v125|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp307:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v126|, |v127|
.Ltmp308:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1808:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x1
	v_max3_num_f32 v1, v1, |v128|, |v129|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
.Ltmp309:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1810:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v130|, |v131|
.Ltmp310:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_loadcnt 0x0
	v_max3_num_f32 v1, v1, |v132|, |v133|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp311:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v1, v1, |v134|, |v135|
.Ltmp312:
	.loc	0 1740 12 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_mov_b32_e32 v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v3, v3, s2, 0xfedcba98
.Ltmp313:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v1, v1, v3
.Ltmp314:
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v1
	v_div_scale_f32 v6, vcc_lo, v1, 0x43e00000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v4, v3
	v_fma_f32 v5, -v3, v4, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v4
	v_mul_f32_e32 v5, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v3, v5, v6
	v_fmac_f32_e32 v5, v12, v4
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_or_b32 v12, v7, 5, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v3, -v3, v5, v6
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_or_b32 v6, v7, 3, v9
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v5, s2, s10, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s11, 0, s2
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v13, v3, 0x43e00000, v1
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v3, vcc_lo, s8, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s9, v137, vcc_lo
	.loc	0 1817 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0, v1
	s_mov_b32 s9, 16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 1.0, v13, vcc_lo
	.loc	0 1819 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v5, vcc_lo, v5, 4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
.LBB5_6:                                ; =>This Inner Loop Header: Depth=1
	.loc	0 1820 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x1
	global_load_b128 v[12:15], v[3:4], off
	global_load_b128 v[16:19], v[3:4], off offset:16
	.loc	0 1819 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v3, vcc_lo, v3, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	.loc	0 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e32 v21.l, v2.l
	.loc	0 1825 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e32 v21.h, 0
	.loc	0 1819 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s9, 0
	.loc	0 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e32 v20.l, v21.l
	.loc	0 1825 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e32 v20.h, v21.h
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x1
	v_div_scale_f32 v22, null, v1, v1, v12
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v24, null, v1, v1, v13
	.loc	0 1826 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v26, null, v1, v1, v14
	.loc	0 1826 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v28, null, v1, v1, v15
	.loc	0 1829 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	v_div_scale_f32 v30, null, v1, v1, v16
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v38, v22
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v39, v24
	.loc	0 1826 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v40, v26
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v32, null, v1, v1, v17
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v41, v28
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v34, null, v1, v1, v18
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v42, v30
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v36, null, v1, v1, v19
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v43, v32
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v46, -v22, v38, 1.0
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v47, -v24, v39, 1.0
	.loc	0 1831 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v44, v34
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v48, -v26, v40, 1.0
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v23, vcc_lo, v12, v1, v12
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v45, v36
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_fmac_f32 v38, v46, v38 :: v_dual_fmac_f32 v39, v47, v39
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v49, -v28, v41, 1.0
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v25, s2, v13, v1, v13
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v50, -v30, v42, 1.0
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v27, s3, v14, v1, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v40, v48, v40 :: v_dual_fmac_f32 v41, v49, v41
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v51, -v32, v43, 1.0
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v46, v23, v38 :: v_dual_mul_f32 v47, v25, v39
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v29, s4, v15, v1, v15
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v52, -v34, v44, 1.0
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v31, s5, v16, v1, v16
	v_dual_fmac_f32 v42, v50, v42 :: v_dual_fmac_f32 v43, v51, v43
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v53, -v36, v45, 1.0
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v48, v27, v40 :: v_dual_mul_f32 v49, v29, v41
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v33, s6, v17, v1, v17
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v54, -v22, v46, v23
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v35, s7, v18, v1, v18
	v_dual_fmac_f32 v44, v52, v44 :: v_dual_fmac_f32 v45, v53, v45
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v55, -v24, v47, v25
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v50, v31, v42 :: v_dual_mul_f32 v51, v33, v43
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v37, s8, v19, v1, v19
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v56, -v26, v48, v27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_fmac_f32 v46, v54, v38 :: v_dual_fmac_f32 v47, v55, v39
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v57, -v28, v49, v29
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v52, v35, v44 :: v_dual_mul_f32 v53, v37, v45
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v58, -v30, v50, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_fmac_f32 v48, v56, v40 :: v_dual_fmac_f32 v49, v57, v41
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v59, -v32, v51, v33
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v22, -v22, v46, v23
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v60, -v34, v52, v35
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v23, -v24, v47, v25
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_fmac_f32 v50, v58, v42 :: v_dual_fmac_f32 v51, v59, v43
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v61, -v36, v53, v37
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v24, -v26, v48, v27
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v22, v22, v38, v46
	.loc	0 1824 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s2
	.loc	0 1826 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v25, -v28, v49, v29
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_fmac_f32 v52, v60, v44 :: v_dual_fmac_f32 v53, v61, v45
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v23, v23, v39, v47
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v26, -v30, v50, v31
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v24, v24, v40, v48
	.loc	0 1826 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 1829 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v27, -v32, v51, v33
	.loc	0 1824 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v12, v22, v1, v12
	.loc	0 1826 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v22, v25, v41, v49
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s5
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v28, -v34, v52, v35
	.loc	0 1824 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1824:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v13, v23, v1, v13
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v23, v26, v42, v50
	.loc	0 1829 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s6
	.loc	0 1831 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v29, -v36, v53, v37
	.loc	0 1826 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v14, v24, v1, v14
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v24, v27, v43, v51
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s7
	.loc	0 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v20.l, v12, v13
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v12, v28, v44, v52
	.loc	0 1831 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s8
	.loc	0 1826 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v15, v22, v1, v15
	.loc	0 1829 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v13, v23, v1, v16
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v16, v29, v45, v53
	.loc	0 1829 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v17, v24, v1, v17
	.loc	0 1831 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v12, v12, v1, v18
	.loc	0 1825 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v20.h, v14, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 1831 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1831:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v14, v16, v1, v19
	.loc	0 1828 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v21.l, v13, v17
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1830 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1830:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v21.h, v12, v14
	.loc	0 1835 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1835:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	global_store_b64 v[5:6], v[20:21], off offset:-4
	.loc	0 1819 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v5, vcc_lo, v5, 16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	0 1819 13 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_scc0 .LBB5_6
; %bb.7:
	.loc	0 1979 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1979:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v151, s25, v1
.LBB5_8:
	.loc	0 0 47 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:47
	s_or_b32 exec_lo, exec_lo, s26
	v_and_b32_e32 v4, 31, v0
	v_mov_b32_e32 v2, -1
	v_bfrev_b32_e32 v3, -2
	.loc	0 1847 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 22, v4
	s_cbranch_execz .LBB5_12
; %bb.9:
	.loc	0 1848 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v1, s23, v4
	.loc	0 1848 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mul_hi_i32 s3, s22, 0x2aaaaaab
	v_bfrev_b32_e32 v3, -2
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s4, s3, 31
	v_mov_b32_e32 v2, -1
	.loc	0 1848 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add3_u32 v1, s3, s4, v1
	.loc	0 1849 31 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_add_co_i32 s24, s24, s23
	.loc	0 1849 16 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s3, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s24, v1
	s_cbranch_execz .LBB5_11
; %bb.10:
	.loc	0 0 16                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s18, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s19, v2, vcc_lo
	.loc	0 1850 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	global_load_b32 v2, v[1:2], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, v2
.LBB5_11:
	.loc	0 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_or_b32 exec_lo, exec_lo, s3
.LBB5_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
.Ltmp315:
	v_mbcnt_lo_u32_b32 v5, -1, 0
	v_dual_mov_b32 v17, 0 :: v_dual_and_b32 v14, 1, v0
	v_and_b32_e32 v13, 3, v0
	v_dual_mov_b32 v15, 0x6020400 :: v_dual_lshlrev_b32 v154, 3, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
.Ltmp316:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_xor_b32_e32 v1, 16, v5
.Ltmp317:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_xor_b32_e32 v20, 8, v5
.Ltmp318:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_xor_b32_e32 v29, 1, v5
	v_dual_mov_b32 v16, 0x5040100 :: v_dual_lshlrev_b32 v153, 3, v8
.Ltmp319:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
.Ltmp320:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_xor_b32_e32 v26, 4, v5
	v_lshl_add_u32 v155, v4, 4, 0
.Ltmp321:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_xor_b32_e32 v28, 2, v5
.Ltmp322:
	.loc	1 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v5, v1 :: v_dual_lshlrev_b32 v12, 3, v4
.Ltmp323:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v20
.Ltmp324:
	.loc	0 1791 10 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1791:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cndmask_b32_e64 v4, v9, 0, s1
	v_and_b32_e32 v9, 16, v0
	v_and_or_b32 v152, v11, 48, v10
.Ltmp325:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v20, v5, v20 :: v_dual_lshlrev_b32 v1, 2, v1
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	v_lshlrev_b32_e32 v24, 6, v0
	v_bfe_u32 v18, v0, 5, 1
.Ltmp326:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v19, v1, v2
.Ltmp327:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_lshlrev_b32_e32 v20, 2, v20
.Ltmp328:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v1, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v157, 0x3070105, v15, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v13
	v_or_b32_e32 v21, 0x100, v0
	v_or_b32_e32 v22, 0x200, v0
	v_or_b32_e32 v23, 0x300, v0
	v_cmp_eq_u32_e64 s3, v7, v18
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v158, 0x3020706, v16 :: v_dual_add_nc_u32 v159, 0, v9
.Ltmp329:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v26
	v_and_b32_e32 v11, 0x7f, v0
	v_lshrrev_b32_e32 v16, 5, v21
.Ltmp330:
	.loc	0 1868 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_addk_co_i32 s22, 0x7f
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_add_u32 v6, v8, 11, 0
.Ltmp331:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v5, v26, vcc_lo
.Ltmp332:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v28
	v_lshlrev_b32_e32 v30, 3, v13
	v_and_or_b32 v18, 0x180, v21, v11
.Ltmp333:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v19, v2, v19
.Ltmp334:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_lshlrev_b32_e32 v9, 2, v14
.Ltmp335:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v1
	v_and_or_b32 v21, 0x280, v22, v11
	v_and_or_b32 v11, 0x380, v23, v11
.Ltmp336:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v27, v20, v19
.Ltmp337:
	.loc	0 1868 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s22, s20
	v_ashrrev_i32_e32 v147, 31, v146
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_kmcnt 0x0
	s_mov_b32 s9, 0
	.loc	0 1868 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cselect_b32 s20, -1, 0
	s_lshl_b32 s8, s21, 8
	v_add_co_u32 v4, s4, s10, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s11, 0, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[14:15], s[8:9]
	v_lshrrev_b32_e32 v10, 2, v10
	v_and_or_b32 v8, v8, 4, v7
	v_lshlrev_b64_e32 v[1:2], 2, v[146:147]
	v_add_nc_u32_e32 v160, v6, v12
	v_xad_u32 v161, 0x120, v12, v6
	v_xad_u32 v162, 0x124, v12, v6
	v_xad_u32 v163, 0x240, v12, v6
	v_xad_u32 v164, 0x244, v12, v6
	v_xad_u32 v165, 0x360, v12, v6
.Ltmp338:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v14, v19, v27
	v_lshrrev_b32_e32 v19, 5, v22
	v_lshrrev_b32_e32 v22, 5, v23
.Ltmp339:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v23, v5, v28, vcc_lo
.Ltmp340:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v29
	v_xad_u32 v166, 0x364, v12, v6
	v_xad_u32 v167, 0x520, v12, v6
	v_xad_u32 v168, 0x524, v12, v6
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v5, v5, v29 :: v_dual_lshlrev_b32 v172, 2, v23
.Ltmp341:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v20, v20, v3
	v_xad_u32 v169, 0x640, v12, v6
	v_xad_u32 v170, 0x644, v12, v6
	v_xad_u32 v171, 0x760, v12, v6
	v_dual_mov_b32 v213, 0xff800000 :: v_dual_lshlrev_b32 v174, 2, v5
	v_xad_u32 v173, 0x764, v12, v6
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v147, s4, s4, v12
	v_and_or_b32 v12, v16, 12, v7
	v_and_or_b32 v26, v19, 20, v7
	v_and_or_b32 v7, v22, 28, v7
	v_lshl_or_b32 v10, v13, 6, v10
	v_add_co_u32 v149, vcc_lo, s18, v1
	v_lshlrev_b32_e32 v25, 4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v150, null, s19, v2, vcc_lo
	v_or_b32_e32 v19, 8, v10
	v_or_b32_e32 v28, 0x110, v10
.Ltmp342:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v20
.Ltmp343:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v20, v9, v14
	v_lshlrev_b32_e32 v16, 4, v18
	v_xor_b32_e32 v1, v19, v30
	v_xor_b32_e32 v19, v28, v30
.Ltmp344:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v9, v9, v3
	v_dual_mov_b32 v211, 1.0 :: v_dual_lshlrev_b32 v18, 2, v10
	v_lshl_add_u32 v176, v1, 2, v6
	v_lshl_add_u32 v181, v19, 2, v6
	v_mov_b32_e32 v19, v17
	v_or_b32_e32 v31, 0x114, v10
	v_or_b32_e32 v32, 24, v10
	v_or_b32_e32 v29, 20, v10
	v_add_co_u32 v200, vcc_lo, v4, v154
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v201, null, 0, v15, vcc_lo
	v_cmp_gt_u32_e64 s2, 64, v0
	v_lshl_add_u32 v156, v0, 1, 0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v148, null, s5, 0, s4
.Ltmp345:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v14, v14, v20
.Ltmp346:
	.loc	0 1869 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_add_nc_u64 s[10:11], s[12:13], s[8:9]
	s_lshl_b32 s8, s21, 1
.Ltmp347:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v9
.Ltmp348:
	.loc	0 1869 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[12:13], s[8:9]
.Ltmp349:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v20, v172, v14
.Ltmp350:
	.loc	0 1869 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_add_nc_u64 s[14:15], s[14:15], s[8:9]
	s_mov_b32 s8, 0x76543210
	v_mov_b32_e32 v212, 0
.Ltmp351:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v5, v14, v20
	v_or_b32_e32 v14, 0x108, v10
	v_or_b32_e32 v20, 12, v10
.Ltmp352:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v22, v174, v5
	v_xor_b32_e32 v2, v14, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v177, v2, 2, v6
	v_xor_b32_e32 v2, v32, v30
	v_lshl_add_u32 v184, v2, 2, v6
	v_or_b32_e32 v2, 28, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v2, v2, v30
.Ltmp353:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v1, v5, v22
	v_dual_mov_b32 v22, v17 :: v_dual_lshlrev_b32 v27, 4, v21
	v_or_b32_e32 v21, 0x10c, v10
	v_or_b32_e32 v5, 40, v10
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp354:
	.loc	0 1865 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_readfirstlane_b32 s18, v1
	v_xor_b32_e32 v1, v31, v30
	v_lshl_add_u32 v186, v2, 2, v6
	v_xor_b32_e32 v14, v21, v30
	v_mov_b32_e32 v21, v17
.Ltmp355:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v23, v172, v3
	v_lshl_add_u32 v183, v1, 2, v6
	v_or_b32_e32 v1, 0x118, v10
	v_xor_b32_e32 v5, v5, v30
	v_or_b32_e32 v2, 0x12c, v10
	v_lshl_add_u32 v179, v14, 2, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v1, v1, v30
	v_lshl_add_u32 v188, v5, 2, v6
	v_or_b32_e32 v5, 0x130, v10
	v_xor_b32_e32 v2, v2, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v185, v1, 2, v6
	v_or_b32_e32 v1, 44, v10
	v_xor_b32_e32 v5, v5, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v191, v2, 2, v6
	v_or_b32_e32 v2, 56, v10
	v_xor_b32_e32 v1, v1, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v193, v5, 2, v6
.Ltmp356:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v23
	v_or_b32_e32 v5, 60, v10
	v_xor_b32_e32 v2, v2, v30
	v_lshl_add_u32 v190, v1, 2, v6
	v_or_b32_e32 v1, 0x134, v10
.Ltmp357:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v23, v174, v3
	v_xor_b32_e32 v5, v5, v30
	v_lshl_add_u32 v196, v2, 2, v6
	v_xor_b32_e32 v1, v1, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v198, v5, 2, v6
	v_lshl_add_u32 v195, v1, 2, v6
.Ltmp358:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v3, v3, v23
	v_mov_b32_e32 v23, v17
	v_lshlrev_b32_e32 v9, 5, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
.Ltmp359:
	.loc	0 1866 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_readfirstlane_b32 s19, v3
	v_or_b32_e32 v3, 0x11c, v10
	v_add3_u32 v175, v6, v18, v9
	v_xor_b32_e32 v9, v20, v30
	v_xor_b32_e32 v20, v29, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v30
	v_lshl_add_u32 v178, v9, 2, v6
	v_or_b32_e32 v9, 0x128, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v182, v20, 2, v6
	v_lshl_add_u32 v187, v3, 2, v6
	v_or_b32_e32 v3, 48, v10
	v_mov_b32_e32 v20, v17
	v_xor_b32_e32 v9, v9, v30
	v_and_b32_e32 v13, 0x3000, v24
	v_or_b32_e32 v24, 16, v10
	v_xor_b32_e32 v3, v3, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v189, v9, 2, v6
	v_or_b32_e32 v9, 52, v10
	v_xor_b32_e32 v18, v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v192, v3, 2, v6
	v_or_b32_e32 v3, 0x138, v10
	v_mov_b32_e32 v24, v17
	v_xor_b32_e32 v9, v9, v30
	v_lshl_add_u32 v180, v18, 2, v6
	v_mov_b32_e32 v18, v17
	v_xor_b32_e32 v3, v3, v30
	v_lshlrev_b32_e32 v11, 4, v11
	v_lshl_add_u32 v194, v9, 2, v6
	v_or_b32_e32 v9, 0x13c, v10
	v_lshlrev_b32_e32 v202, 3, v8
	v_lshl_add_u32 v197, v3, 2, v6
	v_lshlrev_b32_e32 v206, 3, v7
	v_lshlrev_b32_e32 v203, 3, v12
	v_xor_b32_e32 v9, v9, v30
	v_add_nc_u32_e32 v204, 0, v16
	v_lshlrev_b32_e32 v205, 3, v26
	v_dual_mov_b32 v40, v24 :: v_dual_add_nc_u32 v209, 0, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v199, v9, 2, v6
	v_dual_mov_b32 v1, v17 :: v_dual_mov_b32 v4, v20
	v_add_nc_u32_e32 v208, v155, v13
	v_dual_mov_b32 v8, v24 :: v_dual_add_nc_u32 v207, 0, v11
	v_dual_mov_b32 v9, v17 :: v_dual_mov_b32 v6, v22
	v_dual_mov_b32 v7, v23 :: v_dual_add_nc_u32 v210, 0, v27
	v_dual_mov_b32 v13, v21 :: v_dual_mov_b32 v32, v24
	v_mov_b32_e32 v27, v19
	v_dual_mov_b32 v35, v19 :: v_dual_mov_b32 v48, v24
	v_dual_mov_b32 v43, v19 :: v_dual_mov_b32 v56, v24
	v_dual_mov_b32 v51, v19 :: v_dual_mov_b32 v64, v24
	v_dual_mov_b32 v59, v19 :: v_dual_mov_b32 v72, v24
	v_dual_mov_b32 v67, v19 :: v_dual_mov_b32 v80, v24
	v_dual_mov_b32 v75, v19 :: v_dual_mov_b32 v88, v24
	v_dual_mov_b32 v83, v19 :: v_dual_mov_b32 v96, v24
	v_dual_mov_b32 v91, v19 :: v_dual_mov_b32 v104, v24
	v_dual_mov_b32 v99, v19 :: v_dual_mov_b32 v112, v24
	v_dual_mov_b32 v107, v19 :: v_dual_mov_b32 v120, v24
	v_dual_mov_b32 v115, v19 :: v_dual_mov_b32 v128, v24
	v_dual_mov_b32 v123, v19 :: v_dual_mov_b32 v136, v24
	v_dual_mov_b32 v2, v18 :: v_dual_mov_b32 v3, v19
	v_dual_mov_b32 v5, v21 :: v_dual_mov_b32 v10, v18
	v_dual_mov_b32 v11, v19 :: v_dual_mov_b32 v12, v20
	v_dual_mov_b32 v14, v22 :: v_dual_mov_b32 v15, v23
	v_mov_b32_e32 v30, v22
	v_dual_mov_b32 v16, v24 :: v_dual_mov_b32 v31, v23
	v_dual_mov_b32 v28, v20 :: v_dual_mov_b32 v29, v21
	v_dual_mov_b32 v26, v18 :: v_dual_mov_b32 v25, v17
	v_dual_mov_b32 v38, v22 :: v_dual_mov_b32 v39, v23
	v_dual_mov_b32 v36, v20 :: v_dual_mov_b32 v37, v21
	v_dual_mov_b32 v34, v18 :: v_dual_mov_b32 v33, v17
	v_dual_mov_b32 v46, v22 :: v_dual_mov_b32 v47, v23
	v_dual_mov_b32 v44, v20 :: v_dual_mov_b32 v45, v21
	v_dual_mov_b32 v42, v18 :: v_dual_mov_b32 v41, v17
	v_dual_mov_b32 v54, v22 :: v_dual_mov_b32 v55, v23
	v_dual_mov_b32 v52, v20 :: v_dual_mov_b32 v53, v21
	v_dual_mov_b32 v50, v18 :: v_dual_mov_b32 v49, v17
	v_dual_mov_b32 v62, v22 :: v_dual_mov_b32 v63, v23
	v_dual_mov_b32 v60, v20 :: v_dual_mov_b32 v61, v21
	v_dual_mov_b32 v58, v18 :: v_dual_mov_b32 v57, v17
	v_dual_mov_b32 v70, v22 :: v_dual_mov_b32 v71, v23
	v_dual_mov_b32 v68, v20 :: v_dual_mov_b32 v69, v21
	v_dual_mov_b32 v66, v18 :: v_dual_mov_b32 v65, v17
	v_dual_mov_b32 v78, v22 :: v_dual_mov_b32 v79, v23
	v_dual_mov_b32 v76, v20 :: v_dual_mov_b32 v77, v21
	v_dual_mov_b32 v74, v18 :: v_dual_mov_b32 v73, v17
	v_dual_mov_b32 v86, v22 :: v_dual_mov_b32 v87, v23
	v_dual_mov_b32 v84, v20 :: v_dual_mov_b32 v85, v21
	v_dual_mov_b32 v82, v18 :: v_dual_mov_b32 v81, v17
	v_dual_mov_b32 v94, v22 :: v_dual_mov_b32 v95, v23
	v_dual_mov_b32 v92, v20 :: v_dual_mov_b32 v93, v21
	v_dual_mov_b32 v90, v18 :: v_dual_mov_b32 v89, v17
	v_dual_mov_b32 v102, v22 :: v_dual_mov_b32 v103, v23
	v_dual_mov_b32 v100, v20 :: v_dual_mov_b32 v101, v21
	v_dual_mov_b32 v98, v18 :: v_dual_mov_b32 v97, v17
	v_dual_mov_b32 v110, v22 :: v_dual_mov_b32 v111, v23
	v_dual_mov_b32 v108, v20 :: v_dual_mov_b32 v109, v21
	v_dual_mov_b32 v106, v18 :: v_dual_mov_b32 v105, v17
	v_dual_mov_b32 v118, v22 :: v_dual_mov_b32 v119, v23
	v_dual_mov_b32 v116, v20 :: v_dual_mov_b32 v117, v21
	v_dual_mov_b32 v114, v18 :: v_dual_mov_b32 v113, v17
	v_dual_mov_b32 v126, v22 :: v_dual_mov_b32 v127, v23
	v_dual_mov_b32 v124, v20 :: v_dual_mov_b32 v125, v21
	v_dual_mov_b32 v122, v18 :: v_dual_mov_b32 v121, v17
	v_dual_mov_b32 v134, v22 :: v_dual_mov_b32 v135, v23
	v_dual_mov_b32 v132, v20 :: v_dual_mov_b32 v133, v21
	v_dual_mov_b32 v130, v18 :: v_dual_mov_b32 v131, v19
	v_mov_b32_e32 v129, v17
	s_branch .LBB5_15
.LBB5_13:                               ;   in Loop: Header=BB5_15 Depth=1
.Ltmp360:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2108:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2108:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v213, v18
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2108:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp361:
.LBB5_14:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	3 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s9, s9, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s9, 0x3fffffff
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s21, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_61
.LBB5_15:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_50 Depth 2
                                        ;       Child Loop BB5_55 Depth 3
	.loc	0 1870 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1870:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_lshl_b32 s22, s9, 6
	.loc	0 1871 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s22, s18
	s_cselect_b32 s21, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_14
; %bb.16:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v18, s22, v152
                                        ; implicit-def: $vgpr21
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[137:138], null, 0x408, v18, s[10:11]
	v_cmp_ge_i32_e32 vcc_lo, s18, v18
	.loc	0 1884 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1884:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, exec_lo, s4
	s_cbranch_execz .LBB5_18
; %bb.17:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1891 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1891:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v18, s4, v137, v202
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v19, null, 0, v138, s4
	v_add_co_u32 v23, s4, v137, v203
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v24, null, 0, v138, s4
	s_clause 0x3
	global_load_b64 v[139:140], v[18:19], off
	global_load_b64 v[141:142], v[18:19], off offset:16
	global_load_b64 v[21:22], v[23:24], off
	global_load_b64 v[23:24], v[23:24], off offset:16
	.loc	0 1899 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v209, v[139:142]
.LBB5_18:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .LBB5_20
; %bb.19:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v19, v17
	.loc	0 1899 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b32_e32 v18, v17
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v20, v17 :: v_dual_mov_b32 v21, v24
	v_dual_mov_b32 v23, v24 :: v_dual_mov_b32 v22, v24
	ds_store_b128 v209, v[17:20]
.LBB5_20:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1899 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v204, v[21:24]
                                        ; implicit-def: $vgpr21
	.loc	0 1884 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1884:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB5_22
; %bb.21:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1891 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1891:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_co_u32 v18, vcc_lo, v137, v205
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, 0, v138, vcc_lo
	v_add_co_u32 v23, vcc_lo, v137, v206
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, 0, v138, vcc_lo
	s_clause 0x3
	global_load_b64 v[137:138], v[18:19], off
	global_load_b64 v[139:140], v[18:19], off offset:16
	global_load_b64 v[21:22], v[23:24], off
	global_load_b64 v[23:24], v[23:24], off offset:16
	.loc	0 1899 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v210, v[137:140]
.LBB5_22:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB5_24
; %bb.23:                               ;   in Loop: Header=BB5_15 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v19, v17
	.loc	0 1899 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b32_e32 v18, v17
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v20, v17 :: v_dual_mov_b32 v21, v24
	v_dual_mov_b32 v23, v24 :: v_dual_mov_b32 v22, v24
	ds_store_b128 v210, v[17:20]
.LBB5_24:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v138, 0 :: v_dual_add_nc_u32 v139, s22, v153
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v18, 0
	v_mov_b32_e32 v19, 0
	.loc	0 1906 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1899 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v207, v[21:24]
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_26
; %bb.25:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[18:19], null, 0x408, v139, v[147:148]
	global_load_b64 v[18:19], v[18:19], off
.LBB5_26:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v20, 0x8000, v160
	.loc	0 1906 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v20, v18, v19 offset1:1
	.loc	0 1904 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_gt_i32_e64 s18, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_28
; %bb.27:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1903 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v18, 1, v139
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[18:19], null, 0x408, v18, v[147:148]
	global_load_b64 v[137:138], v[18:19], off
.LBB5_28:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v22, 2, v139
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v161, v137 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b32 v162, v138 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v22
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_30
; %bb.29:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[20:21], null, 0x408, v22, v[147:148]
	global_load_b64 v[20:21], v[20:21], off
.LBB5_30:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v22, 3, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v163, v20 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b32 v164, v21 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v22
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_32
; %bb.31:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[18:19], null, 0x408, v22, v[147:148]
	global_load_b64 v[18:19], v[18:19], off
.LBB5_32:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v24, 4, v139
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v18 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b32 v166, v19 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v24
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_34
; %bb.33:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[18:19], null, 0x408, v24, v[147:148]
	global_load_b64 v[22:23], v[18:19], off
.LBB5_34:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v18, 5, v139
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_nc_u32_e32 v19, 0x8400, v160
	.loc	0 1906 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v19, v22, v23 offset1:1
	.loc	0 1904 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v18
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_36
; %bb.35:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[18:19], null, 0x408, v18, v[147:148]
	global_load_b64 v[20:21], v[18:19], off
.LBB5_36:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v24, 6, v139
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v20 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b32 v168, v21 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v24
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_38
; %bb.37:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[20:21], null, 0x408, v24, v[147:148]
	global_load_b64 v[22:23], v[20:21], off
.LBB5_38:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v20, 7, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v22 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b32 v170, v23 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_ge_i32_e64 s18, v20
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[18:19], null, 0x408, v20, v[147:148]
	global_load_b64 v[18:19], v[18:19], off
.LBB5_40:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1912 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v171, v18 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b32 v173, v19 offset:32768
.Ltmp362:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp363:
	.loc	0 1929 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1929:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_42
; %bb.41:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_nc_u32_e32 v22, 0x8000, v175
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_nc_u32_e32 v23, 0x8400, v175
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_2addr_b32 v[18:19], v22 offset1:4
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_2addr_b32 v[20:21], v23 offset1:4
.Ltmp364:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v24, v174, v18
.Ltmp365:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v137, v174, v20
.Ltmp366:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v138, v174, v19
.Ltmp367:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v139, v174, v21
.Ltmp368:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v20, v137, v20, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v24, v138, v19, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v157
.Ltmp369:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v19, v172, v18
.Ltmp370:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v20
.Ltmp371:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v138, v172, v24
.Ltmp372:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v139, v172, v21
.Ltmp373:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v19, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v20, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v24, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:16384
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v18, v176 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v19, v177 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v20, v178 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v21, v179 offset:32768
.Ltmp374:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v24, v174, v18
.Ltmp375:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v174, v19
.Ltmp376:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v174, v20
.Ltmp377:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v174, v21
.Ltmp378:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v19, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v20, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v157
.Ltmp379:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v18
.Ltmp380:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v19
.Ltmp381:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v138, v172, v20
.Ltmp382:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v139, v172, v21
.Ltmp383:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v19, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v20, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:16896
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v18, v180 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v19, v181 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v20, v182 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v21, v183 offset:32768
.Ltmp384:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v24, v174, v18
.Ltmp385:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v174, v19
.Ltmp386:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v174, v20
.Ltmp387:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v174, v21
.Ltmp388:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v19, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v20, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v157
.Ltmp389:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v18
.Ltmp390:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v19
.Ltmp391:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v138, v172, v20
.Ltmp392:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v139, v172, v21
.Ltmp393:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v19, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v20, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:17408
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v18, v184 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v19, v185 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v20, v186 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v21, v187 offset:32768
.Ltmp394:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v24, v174, v18
.Ltmp395:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v174, v19
.Ltmp396:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v174, v20
.Ltmp397:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v174, v21
.Ltmp398:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v19, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v20, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v157
.Ltmp399:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v18
.Ltmp400:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v19
.Ltmp401:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v138, v172, v20
.Ltmp402:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v139, v172, v21
.Ltmp403:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v24, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v137, v19, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v138, v20, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v139, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:17920
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_2addr_b32 v[18:19], v22 offset0:32 offset1:36
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_2addr_b32 v[20:21], v23 offset0:32 offset1:36
.Ltmp404:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v22, v174, v18
.Ltmp405:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v23, v174, v20
.Ltmp406:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v174, v19
.Ltmp407:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v174, v21
.Ltmp408:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v20, v23, v20, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v22, v24, v19, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v157
.Ltmp409:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v19, v172, v18
.Ltmp410:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v23, v172, v20
.Ltmp411:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v22
.Ltmp412:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v21
.Ltmp413:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v19, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v20, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v22, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:18432
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v18, v188 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v19, v189 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v20, v190 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v21, v191 offset:32768
.Ltmp414:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v22, v174, v18
.Ltmp415:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v23, v174, v19
.Ltmp416:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v24, v174, v20
.Ltmp417:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v174, v21
.Ltmp418:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v19, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v20, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v157
.Ltmp419:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v22, v172, v18
.Ltmp420:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v23, v172, v19
.Ltmp421:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v20
.Ltmp422:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v21
.Ltmp423:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v19, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v20, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:18944
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v18, v192 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v19, v193 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v20, v194 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v21, v195 offset:32768
.Ltmp424:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v22, v174, v18
.Ltmp425:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v23, v174, v19
.Ltmp426:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v24, v174, v20
.Ltmp427:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v174, v21
.Ltmp428:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v19, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v20, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v157
.Ltmp429:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v22, v172, v18
.Ltmp430:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v23, v172, v19
.Ltmp431:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v20
.Ltmp432:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v21
.Ltmp433:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v19, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v20, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:19456
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v18, v196 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v19, v197 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v20, v198 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b32 v21, v199 offset:32768
.Ltmp434:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v22, v174, v18
.Ltmp435:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v23, v174, v19
.Ltmp436:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v24, v174, v20
.Ltmp437:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v174, v21
.Ltmp438:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v19, v157
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v20, v157
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v157
.Ltmp439:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v22, v172, v18
.Ltmp440:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v23, v172, v19
.Ltmp441:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v24, v172, v20
.Ltmp442:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	ds_bpermute_b32 v137, v172, v21
.Ltmp443:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v18, v22, v18, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v19, v23, v19, v158
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v20, v24, v20, v158
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v21, v137, v21, v158
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b128 v208, v[18:21] offset:19968
.LBB5_42:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1956 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1956:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB5_46
; %bb.43:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1957 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v19, s22, v0
	v_mov_b32_e32 v18, 0
	.loc	0 1959 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1959:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s18, v19
	s_cbranch_execz .LBB5_45
; %bb.44:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1964 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[20:21], null, 0x408, v19, s[12:13]
	.loc	0 1966 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mad_co_u64_u32 v[22:23], null, 0x408, v19, s[14:15]
	.loc	0 1964 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	global_load_d16_b16 v18, v[20:21], off offset:1024
	.loc	0 1966 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	global_load_d16_hi_b16 v18, v[22:23], off offset:1024
	.loc	0 1964 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v19.h, 8, v18.l
	.loc	0 1966 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshrrev_b16 v19.l, 8, v18.h
	.loc	0 1966 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_and_b16 v20.h, 0xff, v18.l
	v_and_b16 v20.l, 0xff, v18.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1967 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1967:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_pk_lshlrev_b16 v18, 8, v19 op_sel_hi:[0,1]
	.loc	0 1966 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v18, v18, v20
.LBB5_45:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	0 1969 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b16_d16_hi v156, v18 offset:49152
	.loc	0 1970 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1970:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_store_b16 v156, v18 offset:49280
.LBB5_46:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp444:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp445:
	.loc	0 1875 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1875:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_or_b32 s4, s22, 63
	v_mov_b32_e32 v21, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s19
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s23, s20, s4
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s5, s1, s23
.Ltmp446:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp447:
	.loc	0 1980 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s5
	s_cbranch_execz .LBB5_48
; %bb.47:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 1980 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	global_load_b32 v21, v[149:150], off
.LBB5_48:                               ;   in Loop: Header=BB5_15 Depth=1
	.loc	0 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1976 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_or_b32 s4, s22, 16
	s_mov_b32 s24, 0
	.loc	0 1976 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s18
	s_cselect_b32 s25, -1, 0
	s_branch .LBB5_50
.LBB5_49:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_sub_f32_e32 v143, v213, v18
	.loc	0 2047 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v213
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v217, null, v19, v19, v140
	.loc	0 2079 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v219, null, v19, v19, v139
	.loc	0 2076 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e64 v236.l, v17.l
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	0 2078 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e64 v236.h, 0
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v218, v217
	.loc	0 2089 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2089:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_add_u32 v231, s24, 12, v155
	.loc	0 2076 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e64 v235.l, v236.l
	v_exp_f32_e32 v143, v143
	.loc	0 2078 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b16_e64 v235.h, v236.h
	.loc	0 2061 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2061:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_f32_e32 v20, v20, v22
	s_delay_alu instid0(TRANS32_DEP_2)
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v220, -v217, v218, 1.0
	.loc	0 2047 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v143, 0, v143 :: v_dual_fmac_f32 v218, v220, v218
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v220, null, v19, v19, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2071 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v144, v211, v143
	.loc	0 2071 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v211, null, v19, v19, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v213, v211
	v_fma_f32 v214, -v211, v213, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v213, v214, v213
	v_div_scale_f32 v215, vcc_lo, v144, v19, v144
	v_mul_f32_e32 v214, v215, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v211, v214, v215
	v_fmac_f32_e32 v214, v216, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v211, -v211, v214, v215
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v211, v211, v213, v214
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v22, v211, v19, v144
	.loc	0 2062 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v20, v212, v143
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v143, null, v19, v19, v142
	.loc	0 2077 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v211, null, v19, v19, v141
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v135, v135, v22 :: v_dual_mul_f32 v136, v136, v22
	v_mul_f32_e32 v133, v133, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	0 2077 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v213, v211
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v134, v134, v22 :: v_dual_mul_f32 v131, v131, v22
	v_dual_mul_f32 v132, v132, v22 :: v_dual_mul_f32 v129, v129, v22
	v_dual_mul_f32 v130, v130, v22 :: v_dual_mul_f32 v127, v127, v22
	v_dual_mul_f32 v128, v128, v22 :: v_dual_mul_f32 v125, v125, v22
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v212, -v143, v144, 1.0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v15, v15, v22
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v215, -v211, v213, 1.0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v16, v16, v22
	v_dual_mul_f32 v126, v126, v22 :: v_dual_mul_f32 v123, v123, v22
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v144, v212, v144
	v_div_scale_f32 v212, vcc_lo, v142, v19, v142
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v11, v11, v22
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v213, v215, v213
	v_div_scale_f32 v215, s4, v141, v19, v141
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v124, v124, v22 :: v_dual_mul_f32 v121, v121, v22
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v214, v212, v144
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v122, v122, v22 :: v_dual_mul_f32 v119, v119, v22
	v_dual_mul_f32 v120, v120, v22 :: v_dual_mul_f32 v117, v117, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v216, -v143, v214, v212
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v118, v118, v22 :: v_dual_mul_f32 v115, v115, v22
	v_dual_mul_f32 v116, v116, v22 :: v_dual_mul_f32 v113, v113, v22
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v214, v216, v144
	.loc	0 2077 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v216, v215, v213
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v114, v114, v22 :: v_dual_mul_f32 v111, v111, v22
	v_dual_mul_f32 v112, v112, v22 :: v_dual_mul_f32 v109, v109, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v143, -v143, v214, v212
	.loc	0 2077 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v212, -v211, v216, v215
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v110, v110, v22 :: v_dual_mul_f32 v107, v107, v22
	v_dual_mul_f32 v108, v108, v22 :: v_dual_mul_f32 v105, v105, v22
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v214
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v144, v219
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v106, v106, v22 :: v_dual_mul_f32 v103, v103, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v142, v143, v19, v142
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v104, v104, v22 :: v_dual_mul_f32 v101, v101, v22
	v_dual_mul_f32 v102, v102, v22 :: v_dual_mul_f32 v99, v99, v22
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v214, -v219, v144, 1.0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v100, v100, v22 :: v_dual_mul_f32 v97, v97, v22
	v_dual_mul_f32 v98, v98, v22 :: v_dual_mul_f32 v95, v95, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v144, v214, v144
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v216, v212, v213
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v212, s5, v140, v19, v140
	.loc	0 2079 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v214, s4, v139, v19, v139
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v96, v96, v22 :: v_dual_mul_f32 v93, v93, v22
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v143, -v211, v216, v215
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v211, v212, v218
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v94, v94, v22 :: v_dual_mul_f32 v91, v91, v22
	v_dual_mul_f32 v92, v92, v22 :: v_dual_mul_f32 v89, v89, v22
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v213, v216
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v213, null, v19, v19, v138
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v215, -v217, v211, v212
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v216, null, v19, v19, v137
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s5
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v141, v143, v19, v141
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v143, v213
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v211, v215, v218
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v90, v90, v22 :: v_dual_mul_f32 v87, v87, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 2076 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v235.l, v142, v141
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v88, v88, v22 :: v_dual_mul_f32 v85, v85, v22
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v141, -v217, v211, v212
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v217, v216
	s_delay_alu instid0(TRANS32_DEP_2)
	.loc	0 2082 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v212, -v213, v143, 1.0
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v6, v6, v22
	v_dual_mul_f32 v86, v86, v22 :: v_dual_mul_f32 v83, v83, v22
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v218, v211
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v143, v212, v143
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v211, v220
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v212, -v216, v217, 1.0
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v140, v141, v19, v140
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v141, null, v19, v19, v24
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v84, v84, v22 :: v_dual_mul_f32 v81, v81, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v217, v212, v217
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v8, v8, v22 :: v_dual_mul_f32 v215, v214, v144
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v212, s6, v137, v19, v137
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v221, -v220, v211, 1.0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v82, v82, v22 :: v_dual_mul_f32 v79, v79, v22
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v142, -v219, v215, v214
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v3, v3, v22
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v211, v221, v211
	v_div_scale_f32 v221, s4, v23, v19, v23
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v80, v80, v22 :: v_dual_mul_f32 v77, v77, v22
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v215, v142, v144
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v142, s5, v138, v19, v138
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v4, v4, v22
	v_dual_mul_f32 v78, v78, v22 :: v_dual_mul_f32 v75, v75, v22
	s_delay_alu instid0(VALU_DEP_4)
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v214, -v219, v215, v214
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_rcp_f32_e32 v219, v141
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v218, v142, v143 :: v_dual_mul_f32 v1, v1, v22
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v76, v76, v22 :: v_dual_mul_f32 v73, v73, v22
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v214, v144, v215
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v215, v212, v217
	.loc	0 2082 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v214, -v213, v218, v142
	s_mov_b32 vcc_lo, s5
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	0 2084 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v222, -v141, v219, 1.0
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v139, v144, v19, v139
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v144, -v216, v215, v212
	.loc	0 2082 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v218, v214, v143
	.loc	0 2084 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v214, v221, v211
	.loc	0 2084 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v219, v222, v219
	v_div_scale_f32 v222, s7, v24, v19, v24
	.loc	0 2082 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v215, v144, v217
	.loc	0 2082 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v142, -v213, v218, v142
	.loc	0 2084 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v144, -v220, v214, v221
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v2, v2, v22
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_f32_e32 v213, v222, v219
	.loc	0 2078 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v235.h, v140, v139
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v143, v218
	.loc	0 2082 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v143, -v216, v215, v212
	.loc	0 2084 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v214, v144, v211
	.loc	0 2084 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v212, -v141, v213, v222
	.loc	0 2082 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s6
	.loc	0 2082 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v138, v142, v19, v138
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v217, v215
	.loc	0 2084 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v142, -v220, v214, v221
	.loc	0 2084 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fmac_f32_e32 v213, v212, v219
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v74, v74, v22 :: v_dual_mul_f32 v71, v71, v22
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v211, v214
	.loc	0 2084 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v141, -v141, v213, v222
	s_mov_b32 vcc_lo, s7
	.loc	0 2082 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v137, v143, v19, v137
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v72, v72, v22 :: v_dual_mul_f32 v69, v69, v22
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v219, v213
	.loc	0 2084 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v23, v142, v19, v23
	.loc	0 2081 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v236.l, v138, v137
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[137:140], v231 offset:16384
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v24, v141, v19, v24
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[141:144], v231 offset:16896
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[211:214], v231 offset:17408
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[215:218], v231 offset:17920
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[219:222], v231 offset:18432
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[223:226], v231 offset:18944
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[227:230], v231 offset:19456
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[231:234], v231 offset:19968
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v70, v70, v22 :: v_dual_mul_f32 v67, v67, v22
	v_dual_mul_f32 v68, v68, v22 :: v_dual_mul_f32 v65, v65, v22
	v_dual_mul_f32 v66, v66, v22 :: v_dual_mul_f32 v63, v63, v22
	v_dual_mul_f32 v64, v64, v22 :: v_dual_mul_f32 v61, v61, v22
	v_dual_mul_f32 v62, v62, v22 :: v_dual_mul_f32 v59, v59, v22
	v_dual_mul_f32 v60, v60, v22 :: v_dual_mul_f32 v57, v57, v22
	v_dual_mul_f32 v58, v58, v22 :: v_dual_mul_f32 v55, v55, v22
	v_dual_mul_f32 v56, v56, v22 :: v_dual_mul_f32 v53, v53, v22
	v_dual_mul_f32 v54, v54, v22 :: v_dual_mul_f32 v51, v51, v22
	v_dual_mul_f32 v52, v52, v22 :: v_dual_mul_f32 v49, v49, v22
	v_dual_mul_f32 v50, v50, v22 :: v_dual_mul_f32 v47, v47, v22
	v_dual_mul_f32 v48, v48, v22 :: v_dual_mul_f32 v45, v45, v22
	v_dual_mul_f32 v46, v46, v22 :: v_dual_mul_f32 v43, v43, v22
	v_dual_mul_f32 v44, v44, v22 :: v_dual_mul_f32 v41, v41, v22
	v_dual_mul_f32 v42, v42, v22 :: v_dual_mul_f32 v39, v39, v22
	v_dual_mul_f32 v40, v40, v22 :: v_dual_mul_f32 v37, v37, v22
	v_dual_mul_f32 v38, v38, v22 :: v_dual_mul_f32 v35, v35, v22
	v_dual_mul_f32 v36, v36, v22 :: v_dual_mul_f32 v33, v33, v22
	v_dual_mul_f32 v34, v34, v22 :: v_dual_mul_f32 v31, v31, v22
	v_dual_mul_f32 v32, v32, v22 :: v_dual_mul_f32 v29, v29, v22
	v_dual_mul_f32 v30, v30, v22 :: v_dual_mul_f32 v27, v27, v22
	v_dual_mul_f32 v28, v28, v22 :: v_dual_mul_f32 v25, v25, v22
	v_dual_mul_f32 v26, v26, v22 :: v_dual_mul_f32 v13, v13, v22
	v_dual_mul_f32 v14, v14, v22 :: v_dual_mul_f32 v9, v9, v22
	v_dual_mul_f32 v12, v12, v22 :: v_dual_mul_f32 v7, v7, v22
	v_dual_mul_f32 v10, v10, v22 :: v_dual_mul_f32 v5, v5, v22
	.loc	0 2083 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cvt_pk_fp8_f32 v236.h, v23, v24
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x7
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[137:138], v[235:236], v[129:136]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[139:140], v[235:236], v[121:128]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[141:142], v[235:236], v[113:120]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[235:236], v[105:112]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[211:212], v[235:236], v[97:104]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[213:214], v[235:236], v[89:96]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[215:216], v[235:236], v[81:88]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[217:218], v[235:236], v[73:80]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[219:220], v[235:236], v[65:72]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[221:222], v[235:236], v[57:64]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[223:224], v[235:236], v[49:56]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[225:226], v[235:236], v[41:48]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[227:228], v[235:236], v[33:40]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[229:230], v[235:236], v[25:32]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[231:232], v[235:236], v[9:16]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[233:234], v[235:236], v[1:8]
	v_dual_mov_b32 v212, v20 :: v_dual_mov_b32 v211, v19
	v_mov_b32_e32 v213, v18
	.loc	0 1986 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_add_co_i32 s24, s24, 1
	.loc	0 1986 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s24, 4
	.loc	0 1986 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_scc1 .LBB5_13
.LBB5_50:                               ;   Parent Loop BB5_15 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB5_55 Depth 3
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s24, 1
	s_mov_b32 s4, -1
	s_cbranch_scc1 .LBB5_53
; %bb.51:                               ;   in Loop: Header=BB5_50 Depth=2
	s_cmp_eq_u32 s24, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s4, s25
	s_cbranch_scc1 .LBB5_53
; %bb.52:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 1988 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cmp_eq_u32 s24, 2
	.loc	0 1988 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cselect_b32 s4, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s22
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s18, s4
	s_cselect_b32 s4, -1, 0
.LBB5_53:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_60
; %bb.54:                               ;   in Loop: Header=BB5_50 Depth=2
	v_mov_b32_e32 v137, 0
	.loc	0 1998 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_add_u32 v18, s24, 9, v155
	s_mov_b32 s4, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB5_55:                               ;   Parent Loop BB5_15 Depth=1
                                        ;     Parent Loop BB5_50 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 2010 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s5, s4, 6
	.loc	0 2003 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2003:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_add_u32 v24, s4, 12, v18
	.loc	0 2013 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v19, vcc_lo, v200, s5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, 0, v201, vcc_lo
	.loc	0 2005 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b128 v[214:217], v24
	ds_load_b128 v[218:221], v24 offset:2048
	.loc	0 1998 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_add_co_i32 s4, s4, 1
	.loc	0 2013 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x3
	global_load_b64 v[22:23], v[19:20], off
	global_load_b64 v[222:223], v[19:20], off offset:16
	global_load_b64 v[224:225], v[19:20], off offset:32
	global_load_b64 v[19:20], v[19:20], off offset:48
	.loc	0 2022 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1998 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 4
	.loc	0 2016 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[214:215], v[22:23], v[137:144]
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[216:217], v[222:223], v[137:144]
	.loc	0 2016 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[218:219], v[224:225], v[137:144]
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[220:221], v[19:20], v[137:144]
	.loc	0 1998 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_scc1 .LBB5_55
; %bb.56:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_lshl_add_u32 v22, s24, 5, v159
	.loc	0 2025 48 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_lshl_b32 s4, s24, 4
	v_mov_b32_e32 v23, 0xff800000
	.loc	0 2025 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s22
	.loc	0 2030 41 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_b96 v[18:20], v22 offset:49154
	ds_load_u16_d16 v24, v22 offset:49166
	.loc	0 2026 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2026:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s4, 15
	v_or_b32_e32 v214, s4, v154
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s5, s19
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s23, s4
	.loc	0 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB5_58
; %bb.57:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 2030 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_u16_d16 v23, v22 offset:49152
	v_mul_f32_e32 v137, v151, v137
	v_cmp_le_i32_e32 vcc_lo, v214, v21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v23, v23, v137, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2037 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v23, 0xff800000, v23, vcc_lo
.LBB5_58:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mul_f32_e32 v137, v151, v138
	v_or_b32_e32 v138, 2, v214
	v_cmp_ge_i32_e32 vcc_lo, v214, v21
	s_xor_b32 s5, s4, -1
	v_mul_f32_e32 v139, v151, v139
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s4, v138, v21
	v_or_b32_e32 v138, 3, v214
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s6, s5, vcc_lo
	.loc	0 2035 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s1, s6
	s_and_b32 s4, s5, s4
	v_cmp_gt_i32_e32 vcc_lo, v138, v21
	v_mul_f32_e32 v138, v151, v140
	s_wait_dscnt 0x1
	v_fma_mix_f32 v137, v18, v137, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v18, v18, v139, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v139, 4, v214
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	v_cndmask_b32_e64 v137, v137, 0xff800000, s6
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, v18, 0xff800000, s4
	v_fma_mix_f32 v18, v19, v138, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v138, v151, v141
	s_and_b32 s4, s5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v139, v21
	v_or_b32_e32 v139, 5, v214
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, v18, 0xff800000, s4
	v_fma_mix_f32 v18, v19, v138, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v19, v151, v142
	s_and_b32 s4, s5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v139, v21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	v_or_b32_e32 v139, 7, v214
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, v18, 0xff800000, s4
	v_fma_mix_f32 v18, v20, v19, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v19, 6, v214
	s_and_b32 s4, s5, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, v18, 0xff800000, s4
	v_cmp_gt_i32_e32 vcc_lo, v19, v21
	v_mul_f32_e32 v18, v151, v143
	v_cmp_gt_i32_e64 s4, v139, v21
	v_mul_f32_e32 v19, v151, v144
.Ltmp448:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v139, v23, 0xff800000, v137
	s_and_b32 s6, s5, vcc_lo
	v_fma_mix_f32 v18, v20, v18, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s4, s5, s4
	s_wait_dscnt 0x0
	v_fma_mix_f32 v19, v24, v19, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v20, v139, v140, v141
.Ltmp449:
	.loc	0 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s5, s1, s6
	s_or_b32 s4, s1, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v24, v18, 0xff800000, s5
	v_cndmask_b32_e64 v19, v19, 0xff800000, s4
.Ltmp450:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v18, v20, v138, v142
.Ltmp451:
	.loc	0 2069 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp452:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v18, v18, v24, v19
.Ltmp453:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_mov_b32_e32 v20, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v20, v20, s8, 0xfedcba98
.Ltmp454:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2045:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v18, v213, v18, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v20, v23, v18 :: v_dual_sub_f32 v23, v137, v18
	v_dual_sub_f32 v19, v19, v18 :: v_dual_sub_f32 v24, v24, v18
	v_add_nc_u32_e32 v137, 0xc080, v22
	v_mul_f32_e32 v20, 0x3fb8aa3b, v20
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v19, 0x3fb8aa3b, v19 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v24
	v_exp_f32_e32 v139, v19
	v_dual_sub_f32 v19, v140, v18 :: v_dual_sub_f32 v140, v141, v18
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	v_exp_f32_e32 v141, v20
	v_exp_f32_e32 v144, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v143, v23
	v_dual_sub_f32 v23, v142, v18 :: v_dual_mul_f32 v142, 0x3fb8aa3b, v19
.Ltmp455:
	.loc	0 2051 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	ds_load_2addr_b32 v[19:20], v137 offset1:1
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v214, v139, 0, vcc_lo
	v_cndmask_b32_e64 v139, v141, 0, vcc_lo
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	v_sub_f32_e32 v137, v138, v18
	v_exp_f32_e32 v138, v140
	v_cndmask_b32_e64 v143, v143, 0, vcc_lo
	.loc	0 2051 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_nc_u32_e32 v22, 0xc088, v22
	v_exp_f32_e32 v140, v23
	v_mul_f32_e32 v137, 0x3fb8aa3b, v137
	ds_load_2addr_b32 v[23:24], v22 offset1:1
	v_exp_f32_e32 v22, v142
	v_exp_f32_e32 v137, v137
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cndmask_b32_e64 v216, v138, 0, vcc_lo
	v_cndmask_b32_e64 v215, v140, 0, vcc_lo
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v142, v19, v139, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v19, v143, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cndmask_b32_e64 v22, v22, 0, vcc_lo
	v_cndmask_b32_e64 v19, v137, 0, vcc_lo
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_f32_e32 v137, v139, v143
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_mix_f32 v139, v20, v216, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_mix_f32 v140, v20, v22, neg(0) op_sel_hi:[1,0,0]
.Ltmp456:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v20, v142, 0, v141
.Ltmp457:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_f32_e32 v22, v22, v137
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v138, v23, v19, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v137, v23, v215, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp458:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v20, v20, v140, v139
.Ltmp459:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_f32_e32 v22, v216, v22
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_mix_f32 v23, v24, v143, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v24, v24, v214, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp460:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v20, v20, v138, v137
.Ltmp461:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_f32_e32 v19, v19, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp462:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max3_num_f32 v144, v20, v23, v24
.Ltmp463:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_add_f32 v19, v215, v19 :: v_dual_mov_b32 v22, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v19, v143, v19
.Ltmp464:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_permlanex16_b32 v22, v22, s8, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp465:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_add_f32_e32 v20, v214, v19
.Ltmp466:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max_num_f32_e32 v19, v22, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_num_f32_e32 v143, v144, v19
.Ltmp467:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2061:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_dual_mov_b32 v22, v20 :: v_dual_mov_b32 v19, v211
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v22, v22, s8, 0xfedcba98
.Ltmp468:
	.loc	0 2069 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB5_49
; %bb.59:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	0 2070 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v19, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v19
	v_fma_f32 v214, -v19, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v214, v144
	v_div_scale_f32 v214, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v215, v214, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v19, v215, v214
	v_fmac_f32_e32 v215, v216, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v19, -v19, v215, v214
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v19, v19, v144, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v19, v19, 0x43e00000, v143
.Ltmp469:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ] ]
	v_max_num_f32_e32 v19, 0x1f800000, v19
	s_branch .LBB5_49
.Ltmp470:
.LBB5_60:                               ;   in Loop: Header=BB5_50 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v18, v213
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v213, v18
	.loc	0 1986 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_add_co_i32 s24, s24, 1
	.loc	0 1986 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s24, 4
	.loc	0 1986 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_cbranch_scc0 .LBB5_50
	s_branch .LBB5_13
.LBB5_61:
	.loc	0 2112 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2112:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_63
; %bb.62:
	.loc	0 2113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_scale_f32 v0, null, v212, v212, 1.0
	v_div_scale_f32 v19, vcc_lo, 1.0, v212, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v17, v0
	v_fma_f32 v18, -v0, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v19, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v0, v18, v19
	v_fmac_f32_e32 v18, v20, v17
	.loc	0 2115 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2115:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mul_lo_u32 v20, 0x1800, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	0 2113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_fma_f32 v0, -v0, v18, v19
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v0, v0, v17, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	.loc	0 2115 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2115:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshl_add_u32 v17, v145, 8, v20
	.loc	0 2113 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2113:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v212
	.loc	0 2121 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_mov_b32_e32 v18, 0
	.loc	0 2113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_div_fixup_f32 v0, v0, v212, 1.0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 2117 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2117:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_or_b32_e32 v17, v17, v154
	.loc	0 2113 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2113:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	0 2121 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_mul_f32_e32 v139, v211, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v137, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 2121 56 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v17, v129, v139 :: v_dual_mul_f32 v18, v130, v139
	v_mul_f32_e32 v21, v133, v139
	v_dual_mul_f32 v19, v131, v139 :: v_dual_mul_f32 v20, v132, v139
	v_dual_mul_f32 v23, v135, v139 :: v_dual_mul_f32 v22, v134, v139
	v_dual_mul_f32 v121, v121, v139 :: v_dual_mul_f32 v24, v136, v139
	v_dual_mul_f32 v123, v123, v139 :: v_dual_mul_f32 v122, v122, v139
	v_dual_mul_f32 v125, v125, v139 :: v_dual_mul_f32 v124, v124, v139
	v_dual_mul_f32 v127, v127, v139 :: v_dual_mul_f32 v126, v126, v139
	v_dual_mul_f32 v113, v113, v139 :: v_dual_mul_f32 v128, v128, v139
	v_dual_mul_f32 v115, v115, v139 :: v_dual_mul_f32 v110, v110, v139
	v_dual_mul_f32 v97, v97, v139 :: v_dual_mul_f32 v112, v112, v139
	v_dual_mul_f32 v99, v99, v139 :: v_dual_mul_f32 v98, v98, v139
	v_dual_mul_f32 v101, v101, v139 :: v_dual_mul_f32 v100, v100, v139
	v_dual_mul_f32 v103, v103, v139 :: v_dual_mul_f32 v114, v114, v139
	v_dual_mul_f32 v117, v117, v139 :: v_dual_mul_f32 v116, v116, v139
	v_dual_mul_f32 v119, v119, v139 :: v_dual_mul_f32 v102, v102, v139
	v_mul_f32_e32 v104, v104, v139
	v_dual_mul_f32 v118, v118, v139 :: v_dual_mul_f32 v105, v105, v139
	v_dual_mul_f32 v120, v120, v139 :: v_dual_mul_f32 v107, v107, v139
	v_dual_mul_f32 v106, v106, v139 :: v_dual_mul_f32 v109, v109, v139
	v_dual_mul_f32 v108, v108, v139 :: v_dual_mul_f32 v111, v111, v139
	.loc	0 2121 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x7
	global_store_b128 v[137:138], v[17:20], off
	global_store_b128 v[137:138], v[21:24], off offset:16
	global_store_b128 v[137:138], v[121:124], off offset:64
	global_store_b128 v[137:138], v[125:128], off offset:80
	global_store_b128 v[137:138], v[113:116], off offset:128
	global_store_b128 v[137:138], v[117:120], off offset:144
	global_store_b128 v[137:138], v[105:108], off offset:192
	global_store_b128 v[137:138], v[109:112], off offset:208
	.loc	0 2121 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v17, v89, v139 :: v_dual_mul_f32 v18, v90, v139
	v_mul_f32_e32 v21, v93, v139
	v_dual_mul_f32 v19, v91, v139 :: v_dual_mul_f32 v20, v92, v139
	v_dual_mul_f32 v23, v95, v139 :: v_dual_mul_f32 v22, v94, v139
	v_dual_mul_f32 v81, v81, v139 :: v_dual_mul_f32 v24, v96, v139
	v_dual_mul_f32 v83, v83, v139 :: v_dual_mul_f32 v82, v82, v139
	v_dual_mul_f32 v85, v85, v139 :: v_dual_mul_f32 v84, v84, v139
	v_dual_mul_f32 v87, v87, v139 :: v_dual_mul_f32 v86, v86, v139
	v_mul_f32_e32 v88, v88, v139
	.loc	0 2121 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[97:100], off offset:256
	global_store_b128 v[137:138], v[101:104], off offset:272
	global_store_b128 v[137:138], v[17:20], off offset:320
	global_store_b128 v[137:138], v[21:24], off offset:336
	global_store_b128 v[137:138], v[81:84], off offset:384
	global_store_b128 v[137:138], v[85:88], off offset:400
	.loc	0 2121 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v17, v73, v139 :: v_dual_mul_f32 v18, v74, v139
	v_mul_f32_e32 v21, v77, v139
	v_dual_mul_f32 v19, v75, v139 :: v_dual_mul_f32 v20, v76, v139
	v_dual_mul_f32 v23, v79, v139 :: v_dual_mul_f32 v22, v78, v139
	v_dual_mul_f32 v65, v65, v139 :: v_dual_mul_f32 v24, v80, v139
	v_dual_mul_f32 v67, v67, v139 :: v_dual_mul_f32 v66, v66, v139
	v_dual_mul_f32 v69, v69, v139 :: v_dual_mul_f32 v68, v68, v139
	v_dual_mul_f32 v71, v71, v139 :: v_dual_mul_f32 v70, v70, v139
	v_dual_mul_f32 v57, v57, v139 :: v_dual_mul_f32 v72, v72, v139
	v_dual_mul_f32 v59, v59, v139 :: v_dual_mul_f32 v58, v58, v139
	v_dual_mul_f32 v61, v61, v139 :: v_dual_mul_f32 v60, v60, v139
	v_dual_mul_f32 v63, v63, v139 :: v_dual_mul_f32 v62, v62, v139
	v_mul_f32_e32 v64, v64, v139
	.loc	0 2121 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[17:20], off offset:448
	global_store_b128 v[137:138], v[21:24], off offset:464
	global_store_b128 v[137:138], v[65:68], off offset:512
	global_store_b128 v[137:138], v[69:72], off offset:528
	global_store_b128 v[137:138], v[57:60], off offset:576
	global_store_b128 v[137:138], v[61:64], off offset:592
	.loc	0 2121 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v17, v49, v139 :: v_dual_mul_f32 v18, v50, v139
	v_mul_f32_e32 v21, v53, v139
	v_dual_mul_f32 v19, v51, v139 :: v_dual_mul_f32 v20, v52, v139
	v_dual_mul_f32 v23, v55, v139 :: v_dual_mul_f32 v22, v54, v139
	v_dual_mul_f32 v41, v41, v139 :: v_dual_mul_f32 v24, v56, v139
	v_dual_mul_f32 v43, v43, v139 :: v_dual_mul_f32 v42, v42, v139
	v_dual_mul_f32 v45, v45, v139 :: v_dual_mul_f32 v44, v44, v139
	v_dual_mul_f32 v47, v47, v139 :: v_dual_mul_f32 v46, v46, v139
	v_dual_mul_f32 v33, v33, v139 :: v_dual_mul_f32 v48, v48, v139
	v_dual_mul_f32 v35, v35, v139 :: v_dual_mul_f32 v34, v34, v139
	v_dual_mul_f32 v37, v37, v139 :: v_dual_mul_f32 v36, v36, v139
	v_dual_mul_f32 v39, v39, v139 :: v_dual_mul_f32 v38, v38, v139
	v_mul_f32_e32 v40, v40, v139
	.loc	0 2121 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[17:20], off offset:640
	global_store_b128 v[137:138], v[21:24], off offset:656
	global_store_b128 v[137:138], v[41:44], off offset:704
	global_store_b128 v[137:138], v[45:48], off offset:720
	global_store_b128 v[137:138], v[33:36], off offset:768
	global_store_b128 v[137:138], v[37:40], off offset:784
	.loc	0 2121 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	v_dual_mul_f32 v17, v25, v139 :: v_dual_mul_f32 v18, v26, v139
	v_mul_f32_e32 v21, v29, v139
	v_dual_mul_f32 v19, v27, v139 :: v_dual_mul_f32 v20, v28, v139
	v_dual_mul_f32 v23, v31, v139 :: v_dual_mul_f32 v22, v30, v139
	v_dual_mul_f32 v9, v9, v139 :: v_dual_mul_f32 v24, v32, v139
	v_dual_mul_f32 v11, v11, v139 :: v_dual_mul_f32 v10, v10, v139
	v_dual_mul_f32 v13, v13, v139 :: v_dual_mul_f32 v12, v12, v139
	v_dual_mul_f32 v15, v15, v139 :: v_dual_mul_f32 v14, v14, v139
	v_mul_f32_e32 v16, v16, v139
	v_dual_mul_f32 v0, v1, v139 :: v_dual_mul_f32 v1, v2, v139
	v_dual_mul_f32 v2, v3, v139 :: v_dual_mul_f32 v3, v4, v139
	v_dual_mul_f32 v4, v5, v139 :: v_dual_mul_f32 v5, v6, v139
	v_dual_mul_f32 v6, v7, v139 :: v_dual_mul_f32 v7, v8, v139
	.loc	0 2121 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2121:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[17:20], off offset:832
	global_store_b128 v[137:138], v[21:24], off offset:848
	global_store_b128 v[137:138], v[9:12], off offset:896
	global_store_b128 v[137:138], v[13:16], off offset:912
	global_store_b128 v[137:138], v[0:3], off offset:960
	global_store_b128 v[137:138], v[4:7], off offset:976
.Ltmp471:
.LBB5_63:
	.loc	0 2186 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2186:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp472:
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
; codeLenInByte = 12216
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
	.loc	0 2201 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:17
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2201 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:23
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
	s_cbranch_vccnz .LBB6_63
; %bb.1:
	.loc	0 0 23                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:23
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2206 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2206:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB6_63
; %bb.2:
	.loc	0 2208 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2208:30
	s_lshl_b32 s20, ttmp9, 7
	.loc	0 2209 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2209:23
	s_mul_i32 s14, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2209 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2209:18
	s_cmp_ge_i32 s20, s14
	s_cbranch_scc1 .LBB6_63
; %bb.3:
	.loc	0 0 18                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	s_lshr_b32 s12, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 2212 15 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2212:15
	s_cmp_ge_i32 s12, s17
	s_cbranch_scc1 .LBB6_63
; %bb.4:
.Ltmp473:
	.loc	0 1767 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshrrev_b32_e32 v4, 5, v0
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mov_b32 v152, s16 :: v_dual_and_b32 v5, 15, v0
.Ltmp474:
	.loc	0 2201 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:17
	s_clause 0x1
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x20
.Ltmp475:
	.loc	0 1781 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mul_i32 s1, s3, 6
	.loc	0 1778 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshlrev_b32_e32 v7, 4, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1778 30 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v1, v7, v5
	.loc	0 1778 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_nc_u32_e32 v2, s20, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	0 1779 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_hi_i32 v1, 0x2aaaaaab, v2
	.loc	0 1782 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1782:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmp_gt_i32_e64 s0, s14, v2
	.loc	0 1779 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshrrev_b32_e32 v3, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, v1, v3
	.loc	0 1781 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1781:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_lo_u32 v3, v1, 6
	v_mul_lo_u32 v6, v1, 24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v3, v2, v3
	v_add3_u32 v145, v3, s1, v6
	.loc	0 1838 16                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1838:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_6
; %bb.5:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_mov_b32_e32 v146, 0
	s_mul_i32 s22, s15, 0x1800
	s_mov_b32 s23, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[22:23], s[4:5], s[22:23]
	v_lshlrev_b64_e32 v[2:3], 2, v[145:146]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s22, v2
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo
	.loc	0 1841 23 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1841:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_load_b32 v2, v[2:3], off
	.loc	0 1979 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1979:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v152, s16, v2
.LBB6_6:
	.loc	0 0 47 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:47
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v3, -1 :: v_dual_and_b32 v8, 31, v0
	v_bfrev_b32_e32 v6, -2
	.loc	0 1847 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1847:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_u32_e32 22, v8
	s_cbranch_execz .LBB6_10
; %bb.7:
	.loc	0 1848 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mul_hi_i32 s2, s20, 0x2aaaaaab
	v_bfrev_b32_e32 v6, -2
	s_lshr_b32 s13, s2, 31
	v_mov_b32_e32 v3, -1
	.loc	0 1848 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1848:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add3_u32 v2, s2, s13, v8
	.loc	0 1849 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1849:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v2
	s_cbranch_execz .LBB6_9
; %bb.8:
	.loc	0 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v3, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	v_add_co_u32 v2, vcc_lo, s18, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v3, null, s19, v3, vcc_lo
	.loc	0 1850 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_load_b32 v3, v[2:3], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v6, v3
.LBB6_9:
	.loc	0 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
.LBB6_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.Ltmp476:
	v_mbcnt_lo_u32_b32 v9, -1, 0
	v_and_or_b32 v154, v7, 48, v5
.Ltmp477:
	.loc	0 1770 25 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1770:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshrrev_b32_e32 v153, 4, v8
	v_dual_mov_b32 v17, 0x5040100 :: v_dual_and_b32 v12, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp478:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_xor_b32_e32 v2, 16, v9
	v_lshlrev_b32_e32 v11, 3, v8
	v_lshl_add_u32 v156, v8, 4, 0
.Ltmp479:
	.loc	1 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_xor_b32_e32 v8, 8, v9
	v_dual_mov_b32 v16, 0x6020400 :: v_dual_lshlrev_b32 v155, 3, v4
.Ltmp480:
	.loc	1 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_bfe_u32 v18, v0, 5, 1
.Ltmp481:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_add_co_i32 s2, s17, 0x1ff
.Ltmp482:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_xor_b32_e32 v25, 2, v9
.Ltmp483:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v2, v9, v2 :: v_dual_and_b32 v7, 0x7f, v0
.Ltmp484:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	v_lshlrev_b32_e32 v19, 8, v145
.Ltmp485:
	.loc	0 2214 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s2, 0xffff
	v_dual_mov_b32 v97, 0 :: v_dual_lshlrev_b32 v2, 2, v2
.Ltmp486:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v9, v8, vcc_lo
.Ltmp487:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s15, s2
	v_cmp_eq_u32_e64 s2, v153, v18
	v_mov_b32_e32 v99, v97
.Ltmp488:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v14, v2, v3
.Ltmp489:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v15, v2, v6
.Ltmp490:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_lshlrev_b32_e32 v8, 2, v8
.Ltmp491:
	.loc	1 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_xor_b32_e32 v26, 1, v9
	v_and_b32_e32 v20, 16, v0
	v_or_b32_e32 v23, 0x200, v0
.Ltmp492:
	.loc	0 1774 32                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1774:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshl_add_u32 v10, v4, 11, 0
.Ltmp493:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_cvt_f32_u32 s13, s17
	v_dual_mov_b32 v98, v97 :: v_dual_lshlrev_b32 v27, 3, v12
	v_dual_mov_b32 v100, v97 :: v_dual_add_nc_u32 v161, 0, v20
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s16, s13
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshrrev_b32_e32 v5, 2, v5
	v_or_b32_e32 v22, 0x100, v0
	v_or_b32_e32 v24, 0x300, v0
	s_mov_b32 s21, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshl_or_b32 v5, v12, 6, v5
.Ltmp494:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1855:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v14
.Ltmp495:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1860:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v6, v6, v15
.Ltmp496:
	.loc	1 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_xor_b32_e32 v14, 4, v9
	v_and_b32_e32 v13, 1, v0
.Ltmp497:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_mul_f32 s16, s15, s16
.Ltmp498:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v15, v8, v3
.Ltmp499:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v8, v8, v6
	v_and_or_b32 v4, v4, 4, v153
	v_cmp_eq_u32_e32 vcc_lo, 0, v13
.Ltmp500:
	.loc	0 1791 10 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1791:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cndmask_b32_e64 v13, 0, v19, s0
	v_lshrrev_b32_e32 v19, 5, v24
	v_dual_mov_b32 v213, 1.0 :: v_dual_lshlrev_b32 v158, 3, v153
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v159, 0x3070105, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v12
	v_mov_b32_e32 v103, v97
	s_wait_kmcnt 0x0
	v_add_co_u32 v13, s4, s4, v13
	v_lshrrev_b32_e32 v16, 5, v22
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v160, 0x3020706, v17, vcc_lo
.Ltmp501:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v14
	v_and_or_b32 v17, 0x280, v23, v7
	v_add_nc_u32_e32 v162, v10, v11
	v_dual_mov_b32 v104, v97 :: v_dual_mov_b32 v101, v97
	.loc	1 524 11 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v9, v14, vcc_lo
.Ltmp502:
	.loc	2 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1856:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v3, v15
.Ltmp503:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1861:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v6, v6, v8
.Ltmp504:
	.loc	1 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v25
	v_lshlrev_b32_e32 v21, 4, v0
.Ltmp505:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_lshlrev_b32_e32 v14, 2, v14
	v_dual_mov_b32 v102, v97 :: v_dual_lshlrev_b32 v17, 4, v17
.Ltmp506:
	.loc	1 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v20, v9, v25, vcc_lo
.Ltmp507:
	.loc	1 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v26
.Ltmp508:
	.loc	1 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v18, v14, v3
.Ltmp509:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v14, v14, v6
	v_and_or_b32 v8, 0x180, v22, v7
.Ltmp510:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_lshlrev_b32_e32 v175, 2, v20
.Ltmp511:
	.loc	1 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v9, v26, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v20, null, s5, 0, s4
.Ltmp512:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_wait_alu depctr_sa_sdst(0)
	s_trunc_f32 s4, s16
	v_add_co_u32 v146, vcc_lo, s18, v1
.Ltmp513:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_lshlrev_b32_e32 v176, 2, v9
.Ltmp514:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s5, s4, 0x80000000
	v_lshrrev_b32_e32 v15, 5, v23
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s15, s5, s13
	v_and_or_b32 v7, 0x380, v24, v7
	v_or_b32_e32 v23, 8, v5
	v_or_b32_e32 v9, 0x108, v5
	v_or_b32_e32 v24, 12, v5
	v_or_b32_e32 v25, 0x10c, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, s19, v2, vcc_lo
.Ltmp515:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v6, v6, v14
.Ltmp516:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1857:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max_i32_e32 v3, v3, v18
.Ltmp517:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s15, 31
	s_cvt_u32_f32 s4, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s15, s13
.Ltmp518:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v18, v175, v6
.Ltmp519:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v14, v175, v3
	v_xad_u32 v163, 0x120, v11, v10
.Ltmp520:
	.loc	0 2214 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_add_co_ci_u32 s4, s4, 0
.Ltmp521:
	.loc	0 1868 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_addk_co_i32 s20, 0x7f
.Ltmp522:
	.loc	0 2214 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2214:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s4, 0xffff
	v_xad_u32 v164, 0x124, v11, v10
	.loc	0 2215 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2215:26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s12, s4
	v_xad_u32 v165, 0x240, v11, v10
	.loc	0 2216 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2216:23
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s16, s13, s4
.Ltmp523:
	.loc	0 1868 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1868:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cmp_lt_i32 s20, s14
	v_xad_u32 v166, 0x244, v11, v10
	s_cselect_b32 s22, -1, 0
	s_lshl_b32 s20, s3, 8
	v_xad_u32 v167, 0x360, v11, v10
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[8:9], s[20:21]
	v_xad_u32 v168, 0x364, v11, v10
	v_xad_u32 v169, 0x520, v11, v10
	v_xad_u32 v170, 0x524, v11, v10
	v_xad_u32 v171, 0x640, v11, v10
.Ltmp524:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	v_min_i32_e32 v6, v6, v18
.Ltmp525:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v3, v3, v14
	v_xad_u32 v172, 0x644, v11, v10
	v_xad_u32 v173, 0x760, v11, v10
	v_xad_u32 v174, 0x764, v11, v10
.Ltmp526:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v18, v176, v6
.Ltmp527:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v14, v176, v3
.Ltmp528:
	.loc	0 1869 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_add_nc_u64 s[14:15], s[6:7], s[20:21]
	s_lshl_b32 s20, s3, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v148, s3, s4, v11
	v_or_b32_e32 v11, 24, v5
	v_lshlrev_b32_e32 v22, 2, v5
	v_lshlrev_b32_e32 v8, 4, v8
	v_dual_mov_b32 v150, 0xff800000 :: v_dual_lshlrev_b32 v7, 4, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v11, v11, v27
	v_lshlrev_b32_e32 v12, 5, v12
	v_and_or_b32 v16, v16, 12, v153
	v_and_or_b32 v15, v15, 20, v153
	v_and_or_b32 v19, v19, 28, v153
	v_lshl_add_u32 v186, v11, 2, v10
	v_or_b32_e32 v11, 0x128, v5
	v_add_co_u32 v202, vcc_lo, v13, v158
.Ltmp529:
	.loc	2 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1864:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	v_min_i32_e32 v2, v6, v18
.Ltmp530:
	.loc	2 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1859:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v1, v3, v14
	v_xor_b32_e32 v3, v9, v27
	v_xor_b32_e32 v6, v24, v27
	v_xor_b32_e32 v9, v25, v27
.Ltmp531:
	.loc	0 1866 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1866:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_readfirstlane_b32 s24, v2
	v_xor_b32_e32 v2, v23, v27
	v_lshl_add_u32 v179, v3, 2, v10
	v_lshl_add_u32 v180, v6, 2, v10
	v_lshl_add_u32 v181, v9, 2, v10
	v_or_b32_e32 v3, 0x110, v5
	v_lshl_add_u32 v178, v2, 2, v10
	v_or_b32_e32 v2, 16, v5
	v_or_b32_e32 v6, 20, v5
	v_or_b32_e32 v9, 0x114, v5
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v11, v11, v27
	v_xor_b32_e32 v2, v2, v27
	v_xor_b32_e32 v6, v6, v27
	v_xor_b32_e32 v9, v9, v27
	v_lshl_add_u32 v183, v3, 2, v10
	v_or_b32_e32 v3, 28, v5
	v_lshl_add_u32 v182, v2, 2, v10
	v_lshl_add_u32 v184, v6, 2, v10
	v_lshl_add_u32 v185, v9, 2, v10
	v_or_b32_e32 v2, 0x118, v5
	v_or_b32_e32 v6, 0x11c, v5
	v_or_b32_e32 v9, 40, v5
	v_xor_b32_e32 v3, v3, v27
	.loc	0 1865 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_readfirstlane_b32 s23, v1
	v_xor_b32_e32 v2, v2, v27
	v_xor_b32_e32 v6, v6, v27
	v_xor_b32_e32 v9, v9, v27
	v_lshl_add_u32 v188, v3, 2, v10
	v_or_b32_e32 v3, 0x12c, v5
	v_lshl_add_u32 v187, v2, 2, v10
	v_lshl_add_u32 v189, v6, 2, v10
	v_lshl_add_u32 v190, v9, 2, v10
	v_or_b32_e32 v2, 44, v5
	v_or_b32_e32 v6, 48, v5
	v_or_b32_e32 v9, 0x130, v5
	v_xor_b32_e32 v3, v3, v27
	v_lshlrev_b32_e32 v1, 6, v0
	v_xor_b32_e32 v2, v2, v27
	v_xor_b32_e32 v6, v6, v27
	v_xor_b32_e32 v9, v9, v27
	v_lshl_add_u32 v191, v11, 2, v10
	v_or_b32_e32 v11, 52, v5
	v_lshl_add_u32 v192, v2, 2, v10
	v_lshl_add_u32 v193, v3, 2, v10
	v_lshl_add_u32 v194, v6, 2, v10
	v_lshl_add_u32 v195, v9, 2, v10
	v_or_b32_e32 v2, 0x134, v5
	v_or_b32_e32 v3, 56, v5
	v_or_b32_e32 v6, 0x138, v5
	v_or_b32_e32 v9, 60, v5
	v_or_b32_e32 v5, 0x13c, v5
	v_and_b32_e32 v1, 0x3000, v1
	v_xor_b32_e32 v2, v2, v27
	v_xor_b32_e32 v3, v3, v27
	v_xor_b32_e32 v6, v6, v27
	v_xor_b32_e32 v5, v5, v27
	v_xor_b32_e32 v11, v11, v27
	v_xor_b32_e32 v9, v9, v27
	v_lshl_add_u32 v197, v2, 2, v10
	v_lshl_add_u32 v198, v3, 2, v10
	v_lshl_add_u32 v199, v6, 2, v10
	v_lshl_add_u32 v201, v5, 2, v10
	v_lshlrev_b32_e32 v204, 3, v4
	v_dual_mov_b32 v151, 0 :: v_dual_add_nc_u32 v206, 0, v8
	v_add_nc_u32_e32 v209, 0, v7
	v_dual_mov_b32 v1, v97 :: v_dual_add_nc_u32 v210, v156, v1
	v_lshl_add_u32 v196, v11, 2, v10
	v_lshl_add_u32 v200, v9, 2, v10
	v_dual_mov_b32 v6, v102 :: v_dual_lshlrev_b32 v205, 3, v16
	v_dual_mov_b32 v8, v104 :: v_dual_lshlrev_b32 v207, 3, v15
	v_dual_mov_b32 v3, v99 :: v_dual_lshlrev_b32 v208, 3, v19
	v_add3_u32 v177, v10, v22, v12
	v_mov_b32_e32 v9, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v203, null, 0, v20, vcc_lo
	v_dual_mov_b32 v12, v100 :: v_dual_add_nc_u32 v211, 0, v21
	v_dual_mov_b32 v13, v101 :: v_dual_add_nc_u32 v212, 0, v17
	v_dual_mov_b32 v16, v104 :: v_dual_mov_b32 v17, v97
	v_dual_mov_b32 v24, v104 :: v_dual_mov_b32 v25, v97
	v_dual_mov_b32 v32, v104 :: v_dual_mov_b32 v33, v97
	v_dual_mov_b32 v40, v104 :: v_dual_mov_b32 v41, v97
	v_dual_mov_b32 v48, v104 :: v_dual_mov_b32 v49, v97
	v_dual_mov_b32 v56, v104 :: v_dual_mov_b32 v57, v97
	v_dual_mov_b32 v64, v104 :: v_dual_mov_b32 v65, v97
	v_dual_mov_b32 v72, v104 :: v_dual_mov_b32 v73, v97
	v_dual_mov_b32 v80, v104 :: v_dual_mov_b32 v81, v97
	v_dual_mov_b32 v88, v104 :: v_dual_mov_b32 v89, v97
	v_dual_mov_b32 v112, v104 :: v_dual_mov_b32 v109, v101
	v_dual_mov_b32 v120, v104 :: v_dual_mov_b32 v117, v101
	v_dual_mov_b32 v128, v104 :: v_dual_mov_b32 v125, v101
	v_dual_mov_b32 v136, v104 :: v_dual_mov_b32 v133, v101
	v_cmp_gt_u32_e64 s1, 64, v0
	v_lshl_add_u32 v157, v0, 1, 0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v149, null, s5, 0, s3
	v_mov_b32_e32 v2, v98
	v_dual_mov_b32 v4, v100 :: v_dual_mov_b32 v5, v101
	v_dual_mov_b32 v7, v103 :: v_dual_mov_b32 v10, v98
	v_dual_mov_b32 v11, v99 :: v_dual_mov_b32 v14, v102
	v_dual_mov_b32 v15, v103 :: v_dual_mov_b32 v18, v98
	v_dual_mov_b32 v19, v99 :: v_dual_mov_b32 v20, v100
	v_dual_mov_b32 v21, v101 :: v_dual_mov_b32 v22, v102
	v_dual_mov_b32 v23, v103 :: v_dual_mov_b32 v26, v98
	v_dual_mov_b32 v27, v99 :: v_dual_mov_b32 v28, v100
	v_dual_mov_b32 v29, v101 :: v_dual_mov_b32 v30, v102
	v_dual_mov_b32 v31, v103 :: v_dual_mov_b32 v34, v98
	v_dual_mov_b32 v35, v99 :: v_dual_mov_b32 v36, v100
	v_dual_mov_b32 v37, v101 :: v_dual_mov_b32 v38, v102
	v_dual_mov_b32 v39, v103 :: v_dual_mov_b32 v42, v98
	v_dual_mov_b32 v43, v99 :: v_dual_mov_b32 v44, v100
	v_dual_mov_b32 v45, v101 :: v_dual_mov_b32 v46, v102
	v_dual_mov_b32 v47, v103 :: v_dual_mov_b32 v50, v98
	v_dual_mov_b32 v51, v99 :: v_dual_mov_b32 v52, v100
	v_dual_mov_b32 v53, v101 :: v_dual_mov_b32 v54, v102
	v_dual_mov_b32 v55, v103 :: v_dual_mov_b32 v58, v98
	v_dual_mov_b32 v59, v99 :: v_dual_mov_b32 v60, v100
	v_dual_mov_b32 v61, v101 :: v_dual_mov_b32 v62, v102
	v_dual_mov_b32 v63, v103 :: v_dual_mov_b32 v66, v98
	v_dual_mov_b32 v67, v99 :: v_dual_mov_b32 v68, v100
	v_dual_mov_b32 v69, v101 :: v_dual_mov_b32 v70, v102
	v_dual_mov_b32 v71, v103 :: v_dual_mov_b32 v74, v98
	v_dual_mov_b32 v75, v99 :: v_dual_mov_b32 v76, v100
	v_dual_mov_b32 v77, v101 :: v_dual_mov_b32 v78, v102
	v_dual_mov_b32 v79, v103 :: v_dual_mov_b32 v82, v98
	v_dual_mov_b32 v83, v99 :: v_dual_mov_b32 v84, v100
	v_dual_mov_b32 v85, v101 :: v_dual_mov_b32 v86, v102
	v_dual_mov_b32 v87, v103 :: v_dual_mov_b32 v90, v98
	v_dual_mov_b32 v91, v99 :: v_dual_mov_b32 v92, v100
	v_dual_mov_b32 v93, v101 :: v_dual_mov_b32 v94, v102
	v_dual_mov_b32 v95, v103 :: v_dual_mov_b32 v96, v104
	v_dual_mov_b32 v111, v103 :: v_dual_mov_b32 v110, v102
	v_dual_mov_b32 v107, v99 :: v_dual_mov_b32 v108, v100
	v_dual_mov_b32 v105, v97 :: v_dual_mov_b32 v106, v98
	v_dual_mov_b32 v119, v103 :: v_dual_mov_b32 v118, v102
	v_dual_mov_b32 v115, v99 :: v_dual_mov_b32 v116, v100
	v_dual_mov_b32 v113, v97 :: v_dual_mov_b32 v114, v98
	v_dual_mov_b32 v127, v103 :: v_dual_mov_b32 v126, v102
	v_dual_mov_b32 v123, v99 :: v_dual_mov_b32 v124, v100
	v_dual_mov_b32 v121, v97 :: v_dual_mov_b32 v122, v98
	v_dual_mov_b32 v135, v103 :: v_dual_mov_b32 v134, v102
	v_dual_mov_b32 v131, v99 :: v_dual_mov_b32 v132, v100
	v_dual_mov_b32 v129, v97 :: v_dual_mov_b32 v130, v98
	.loc	0 1869 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_add_nc_u64 s[18:19], s[6:7], s[20:21]
	s_add_nc_u64 s[8:9], s[8:9], s[20:21]
	s_mov_b32 s7, 0x76543210
	s_branch .LBB6_13
.LBB6_11:                               ;   in Loop: Header=BB6_13 Depth=1
.Ltmp532:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2108:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2108:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v150, v98
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2108:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp533:
.LBB6_12:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	3 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s13, s13, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s13, s16
	s_cselect_b32 s3, -1, 0
	s_xor_b32 s4, s20, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_59
.LBB6_13:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_48 Depth 2
                                        ;       Child Loop BB6_53 Depth 3
	.loc	0 1870 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1870:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_lshl_b32 s21, s13, 6
	.loc	0 1871 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s21, s23
	s_cselect_b32 s20, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_12
; %bb.14:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v98, s21, v154
                                        ; implicit-def: $vgpr101
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[137:138], null, 0x408, v98, s[14:15]
	v_cmp_ge_i32_e32 vcc_lo, s23, v98
	.loc	0 1884 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1884:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s3
	s_cbranch_execz .LBB6_16
; %bb.15:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	0 1891 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1891:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_co_u32 v98, s3, v137, v204
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v99, null, 0, v138, s3
	v_add_co_u32 v103, s3, v137, v205
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v104, null, 0, v138, s3
	s_clause 0x3
	global_load_b64 v[139:140], v[98:99], off
	global_load_b64 v[141:142], v[98:99], off offset:16
	global_load_b64 v[101:102], v[103:104], off
	global_load_b64 v[103:104], v[103:104], off offset:16
	.loc	0 1899 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v211, v[139:142]
.LBB6_16:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s4
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_13 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v99, v97
	.loc	0 1899 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b32_e32 v98, v97
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v100, v97 :: v_dual_mov_b32 v101, v104
	v_dual_mov_b32 v103, v104 :: v_dual_mov_b32 v102, v104
	ds_store_b128 v211, v[97:100]
.LBB6_18:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1899 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v206, v[101:104]
                                        ; implicit-def: $vgpr101
	.loc	0 1884 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1884:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1891 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1891:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_co_u32 v98, vcc_lo, v137, v207
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, 0, v138, vcc_lo
	v_add_co_u32 v103, vcc_lo, v137, v208
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v104, null, 0, v138, vcc_lo
	s_clause 0x3
	global_load_b64 v[137:138], v[98:99], off
	global_load_b64 v[139:140], v[98:99], off offset:16
	global_load_b64 v[101:102], v[103:104], off
	global_load_b64 v[103:104], v[103:104], off offset:16
	.loc	0 1899 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v212, v[137:140]
.LBB6_20:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_13 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v99, v97
	.loc	0 1899 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b32_e32 v98, v97
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v100, v97 :: v_dual_mov_b32 v101, v104
	v_dual_mov_b32 v103, v104 :: v_dual_mov_b32 v102, v104
	ds_store_b128 v212, v[97:100]
.LBB6_22:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v138, 0 :: v_dual_add_nc_u32 v139, s21, v155
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v98, 0
	v_mov_b32_e32 v99, 0
	.loc	0 1906 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 1899 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v209, v[101:104]
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[98:99], null, 0x408, v139, v[148:149]
	global_load_b64 v[98:99], v[98:99], off
.LBB6_24:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v100, 0x8000, v162
	.loc	0 1906 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v100, v98, v99 offset1:1
	.loc	0 1904 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_gt_i32_e64 s23, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1903 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v98, 1, v139
	s_delay_alu instid0(VALU_DEP_1)
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[98:99], null, 0x408, v98, v[148:149]
	global_load_b64 v[137:138], v[98:99], off
.LBB6_26:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v102, 2, v139
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v101, 0
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v163, v137 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b32 v164, v138 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v102
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_28
; %bb.27:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[100:101], null, 0x408, v102, v[148:149]
	global_load_b64 v[100:101], v[100:101], off
.LBB6_28:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v102, 3, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v165, v100 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b32 v166, v101 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v102
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[98:99], null, 0x408, v102, v[148:149]
	global_load_b64 v[98:99], v[98:99], off
.LBB6_30:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v104, 4, v139
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v103, 0
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v167, v98 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b32 v168, v99 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v104
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_32
; %bb.31:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[98:99], null, 0x408, v104, v[148:149]
	global_load_b64 v[102:103], v[98:99], off
.LBB6_32:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v98, 5, v139
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_nc_u32_e32 v99, 0x8400, v162
	.loc	0 1906 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v99, v102, v103 offset1:1
	.loc	0 1904 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v98
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_34
; %bb.33:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[98:99], null, 0x408, v98, v[148:149]
	global_load_b64 v[100:101], v[98:99], off
.LBB6_34:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v104, 6, v139
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v103, 0
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v169, v100 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b32 v170, v101 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v104
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_36
; %bb.35:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[100:101], null, 0x408, v104, v[148:149]
	global_load_b64 v[102:103], v[100:101], off
.LBB6_36:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1903 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v100, 7, v139
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 1912 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v171, v102 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b32 v172, v103 offset:32768
	.loc	0 1904 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_ge_i32_e64 s23, v100
	.loc	0 1906 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1910 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1910:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[98:99], null, 0x408, v100, v[148:149]
	global_load_b64 v[98:99], v[98:99], off
.LBB6_38:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1912 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1912:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v173, v98 offset:32768
	.loc	0 1914 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1914:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b32 v174, v99 offset:32768
.Ltmp534:
	.loc	3 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1917:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp535:
	.loc	0 1929 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1929:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_40
; %bb.39:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_nc_u32_e32 v102, 0x8000, v177
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_nc_u32_e32 v103, 0x8400, v177
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_2addr_b32 v[98:99], v102 offset1:4
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_2addr_b32 v[100:101], v103 offset1:4
.Ltmp536:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v104, v176, v98
.Ltmp537:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v137, v176, v100
.Ltmp538:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v138, v176, v99
.Ltmp539:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v139, v176, v101
.Ltmp540:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v100, v137, v100, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v104, v138, v99, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v159
.Ltmp541:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v99, v175, v98
.Ltmp542:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v100
.Ltmp543:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v138, v175, v104
.Ltmp544:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v139, v175, v101
.Ltmp545:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v99, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v100, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v104, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:16384
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v98, v178 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v99, v179 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v100, v180 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v101, v181 offset:32768
.Ltmp546:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v104, v176, v98
.Ltmp547:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v176, v99
.Ltmp548:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v176, v100
.Ltmp549:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v176, v101
.Ltmp550:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v99, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v100, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v159
.Ltmp551:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v98
.Ltmp552:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v99
.Ltmp553:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v138, v175, v100
.Ltmp554:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v139, v175, v101
.Ltmp555:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v99, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v100, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:16896
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v98, v182 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v99, v183 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v100, v184 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v101, v185 offset:32768
.Ltmp556:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v104, v176, v98
.Ltmp557:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v176, v99
.Ltmp558:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v176, v100
.Ltmp559:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v176, v101
.Ltmp560:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v99, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v100, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v159
.Ltmp561:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v98
.Ltmp562:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v99
.Ltmp563:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v138, v175, v100
.Ltmp564:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v139, v175, v101
.Ltmp565:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v99, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v100, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:17408
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v98, v186 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v99, v187 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v100, v188 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v101, v189 offset:32768
.Ltmp566:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v104, v176, v98
.Ltmp567:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v176, v99
.Ltmp568:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v176, v100
.Ltmp569:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v176, v101
.Ltmp570:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v99, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v100, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v159
.Ltmp571:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v98
.Ltmp572:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v99
.Ltmp573:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v138, v175, v100
.Ltmp574:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v139, v175, v101
.Ltmp575:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v104, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v137, v99, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v138, v100, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v139, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:17920
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_2addr_b32 v[98:99], v102 offset0:32 offset1:36
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_2addr_b32 v[100:101], v103 offset0:32 offset1:36
.Ltmp576:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v102, v176, v98
.Ltmp577:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v103, v176, v100
.Ltmp578:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v176, v99
.Ltmp579:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v176, v101
.Ltmp580:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v100, v103, v100, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v102, v104, v99, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v159
.Ltmp581:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v99, v175, v98
.Ltmp582:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v103, v175, v100
.Ltmp583:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v102
.Ltmp584:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v101
.Ltmp585:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v99, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v100, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v102, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:18432
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v98, v190 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v99, v191 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v100, v192 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v101, v193 offset:32768
.Ltmp586:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v102, v176, v98
.Ltmp587:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v103, v176, v99
.Ltmp588:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v104, v176, v100
.Ltmp589:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v176, v101
.Ltmp590:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v99, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v100, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v159
.Ltmp591:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v102, v175, v98
.Ltmp592:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v103, v175, v99
.Ltmp593:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v100
.Ltmp594:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v101
.Ltmp595:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v99, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v100, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:18944
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v98, v194 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v99, v195 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v100, v196 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v101, v197 offset:32768
.Ltmp596:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v102, v176, v98
.Ltmp597:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v103, v176, v99
.Ltmp598:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v104, v176, v100
.Ltmp599:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v176, v101
.Ltmp600:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v99, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v100, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v159
.Ltmp601:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v102, v175, v98
.Ltmp602:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v103, v175, v99
.Ltmp603:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v100
.Ltmp604:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v101
.Ltmp605:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v99, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v100, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:19456
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v98, v198 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v99, v199 offset:32768
	.loc	0 1937 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1937:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v100, v200 offset:32768
	.loc	0 1938 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1938:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b32 v101, v201 offset:32768
.Ltmp606:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v102, v176, v98
.Ltmp607:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v103, v176, v99
.Ltmp608:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1939:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v104, v176, v100
.Ltmp609:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v176, v101
.Ltmp610:
	.loc	0 1940 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v99, v159
	.loc	0 1940 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1940:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v100, v159
	.loc	0 1942 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v159
.Ltmp611:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v102, v175, v98
.Ltmp612:
	.loc	1 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v103, v175, v99
.Ltmp613:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v104, v175, v100
.Ltmp614:
	.loc	1 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	ds_bpermute_b32 v137, v175, v101
.Ltmp615:
	.loc	0 1944 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v98, v102, v98, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v99, v103, v99, v160
	.loc	0 1944 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1944:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v100, v104, v100, v160
	.loc	0 1947 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1947:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v101, v137, v101, v160
	.loc	0 1950 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1950:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b128 v210, v[98:101] offset:19968
.LBB6_40:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1956 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1956:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB6_44
; %bb.41:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1957 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1957:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v99, s21, v0
	v_mov_b32_e32 v98, 0
	.loc	0 1959 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1959:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s23, v99
	s_cbranch_execz .LBB6_43
; %bb.42:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1964 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[100:101], null, 0x408, v99, s[18:19]
	.loc	0 1966 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_i64_i32 v[102:103], null, 0x408, v99, s[8:9]
	.loc	0 1964 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_load_d16_b16 v98, v[100:101], off offset:1024
	.loc	0 1966 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_load_d16_hi_b16 v98, v[102:103], off offset:1024
	.loc	0 1964 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v99.h, 8, v98.l
	.loc	0 1966 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshrrev_b16 v99.l, 8, v98.h
	.loc	0 1966 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_and_b16 v100.h, 0xff, v98.l
	v_and_b16 v100.l, 0xff, v98.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 1967 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1967:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_pk_lshlrev_b16 v98, 8, v99 op_sel_hi:[0,1]
	.loc	0 1966 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_or_b32_e32 v98, v98, v100
.LBB6_43:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	0 1969 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b16_d16_hi v157, v98 offset:49152
	.loc	0 1970 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1970:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_store_b16 v157, v98 offset:49280
.LBB6_44:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp616:
	.loc	3 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp617:
	.loc	0 1875 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1875:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_or_b32 s3, s21, 63
	v_mov_b32_e32 v101, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s24
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s25, s22, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	0 1980 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_xor_b32 s3, s25, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s0, s3
.Ltmp618:
	.loc	3 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	s_barrier_wait -1
	.loc	3 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1972:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp619:
	.loc	0 1980 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 1980 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1980:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_load_b32 v101, v[146:147], off
.LBB6_46:                               ;   in Loop: Header=BB6_13 Depth=1
	.loc	0 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	0 1976 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_or_b32 s3, s21, 16
	s_mov_b32 s26, 0
	.loc	0 1976 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s23
	s_cselect_b32 s27, -1, 0
	s_branch .LBB6_48
.LBB6_47:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_or_b32 exec_lo, exec_lo, s3
	v_sub_f32_e32 v143, v150, v98
	.loc	0 2047 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v150
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v217, null, v99, v99, v140
	.loc	0 2079 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v219, null, v99, v99, v139
	.loc	0 2082 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v221, null, v99, v99, v137
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v222, null, v99, v99, v103
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v218, v217
	.loc	0 2089 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2089:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshl_add_u32 v233, s26, 12, v156
	.loc	0 2061 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2061:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_f32_e32 v100, v100, v102
	v_exp_f32_e32 v143, v143
	s_delay_alu instid0(TRANS32_DEP_2)
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v220, -v217, v218, 1.0
	.loc	0 2047 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2047:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v143, 0, v143 :: v_dual_fmac_f32 v218, v220, v218
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2071 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v144, v213, v143
	.loc	0 2071 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2071:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v150, null, v99, v99, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v213, v150
	v_fma_f32 v214, -v150, v213, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v213, v214, v213
	v_div_scale_f32 v215, vcc_lo, v144, v99, v144
	v_mul_f32_e32 v214, v215, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v150, v214, v215
	v_fmac_f32_e32 v214, v216, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v214, v215
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v213, v214
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v102, v150, v99, v144
	.loc	0 2062 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2062:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v100, v151, v143
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v143, null, v99, v99, v142
	.loc	0 2077 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v150, null, v99, v99, v141
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v133, v133, v102 :: v_dual_mul_f32 v136, v136, v102
	v_mul_f32_e32 v135, v135, v102
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	0 2077 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v213, v150
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v134, v134, v102 :: v_dual_mul_f32 v131, v131, v102
	v_dual_mul_f32 v132, v132, v102 :: v_dual_mul_f32 v129, v129, v102
	v_dual_mul_f32 v130, v130, v102 :: v_dual_mul_f32 v127, v127, v102
	v_dual_mul_f32 v128, v128, v102 :: v_dual_mul_f32 v125, v125, v102
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v151, -v143, v144, 1.0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v13, v13, v102
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v215, -v150, v213, 1.0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v16, v16, v102
	v_dual_mul_f32 v126, v126, v102 :: v_dual_mul_f32 v123, v123, v102
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v144, v151, v144
	v_div_scale_f32 v151, vcc_lo, v142, v99, v142
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v9, v9, v102
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v213, v215, v213
	v_div_scale_f32 v215, s3, v141, v99, v141
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v124, v124, v102 :: v_dual_mul_f32 v121, v121, v102
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v214, v151, v144
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v122, v122, v102 :: v_dual_mul_f32 v119, v119, v102
	v_dual_mul_f32 v120, v120, v102 :: v_dual_mul_f32 v117, v117, v102
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v216, -v143, v214, v151
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v118, v118, v102 :: v_dual_mul_f32 v115, v115, v102
	v_dual_mul_f32 v116, v116, v102 :: v_dual_mul_f32 v113, v113, v102
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v214, v216, v144
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v114, v114, v102 :: v_dual_mul_f32 v111, v111, v102
	v_dual_mul_f32 v112, v112, v102 :: v_dual_mul_f32 v109, v109, v102
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v143, -v143, v214, v151
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v110, v110, v102 :: v_dual_mul_f32 v107, v107, v102
	v_dual_mul_f32 v108, v108, v102 :: v_dual_mul_f32 v105, v105, v102
	.loc	0 2077 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v214
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v214, s4, v140, v99, v140
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v216, v215, v213
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v144, v219
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 2077 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v142, v143, v99, v142
	.loc	0 2079 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v220, s3, v139, v99, v139
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v151, -v150, v216, v215
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v106, v106, v102 :: v_dual_mul_f32 v95, v95, v102
	v_dual_mul_f32 v96, v96, v102 :: v_dual_mul_f32 v93, v93, v102
	s_delay_alu instid0(VALU_DEP_3)
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v216, v151, v213
	.loc	0 2076 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b16_e64 v151.l, v97.l
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v3, v3, v102
	.loc	0 2078 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b16_e64 v151.h, 0
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v94, v94, v102 :: v_dual_mul_f32 v91, v91, v102
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v143, -v150, v216, v215
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v215, v214, v218
	.loc	0 2079 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v150, -v219, v144, 1.0
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v1, v1, v102 :: v_dual_mul_f32 v92, v92, v102
	v_mul_f32_e32 v89, v89, v102
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v213, v216
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v216, -v217, v215, v214
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v213, null, v99, v99, v138
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v144, v150, v144
	.loc	0 2076 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b16_e64 v150.l, v151.l
	.loc	0 2077 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2077:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v141, v143, v99, v141
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v215, v216, v218
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v143, v213
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v216, v220, v144
	.loc	0 2079 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 2076 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cvt_pk_fp8_f32 v150.l, v142, v141
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v141, -v217, v215, v214
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v217, v221
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v142, -v219, v216, v220
	.loc	0 2078 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b16_e64 v150.h, v151.h
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v90, v90, v102 :: v_dual_mul_f32 v87, v87, v102
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v218, v215
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v214, -v213, v143, 1.0
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v216, v142, v144
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v8, v8, v102
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v142, s4, v138, v99, v138
	.loc	0 2079 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v140, v141, v99, v140
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v141, null, v99, v99, v104
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v143, v214, v143
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v218, -v219, v216, v220
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v214, v222
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v215, -v221, v217, 1.0
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_rcp_f32_e32 v220, v141
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v219, v142, v143
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v88, v88, v102 :: v_dual_mul_f32 v85, v85, v102
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v217, v215, v217
	v_div_scale_f32 v215, s5, v137, v99, v137
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v218, v144, v216
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v216, -v213, v219, v142
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v224, -v141, v220, 1.0
	.loc	0 2084 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v223, -v222, v214, 1.0
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v6, v6, v102
	.loc	0 2079 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2079:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v139, v144, v99, v139
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v219, v216, v143
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v220, v224, v220
	v_div_scale_f32 v224, s6, v104, v99, v104
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v218, v215, v217
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v214, v223, v214
	v_div_scale_f32 v223, s3, v103, v99, v103
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v142, -v213, v219, v142
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v213, v224, v220
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v144, -v221, v218, v215
	.loc	0 2082 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s4
	.loc	0 2084 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v216, v223, v214
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v143, v219
	.loc	0 2082 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s5
	v_fmac_f32_e32 v218, v144, v217
	.loc	0 2078 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2078:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cvt_pk_fp8_f32 v150.h, v140, v139
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v144, -v222, v216, v223
	.loc	0 2082 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v138, v142, v99, v138
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v86, v86, v102 :: v_dual_mul_f32 v83, v83, v102
	.loc	0 2082 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v143, -v221, v218, v215
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v215, -v141, v213, v224
	.loc	0 2084 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v216, v144, v214
	.loc	0 2074 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v84, v84, v102 :: v_dual_mul_f32 v81, v81, v102
	v_dual_mul_f32 v82, v82, v102 :: v_dual_mul_f32 v79, v79, v102
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fmac_f32_e32 v213, v215, v220
	.loc	0 2084 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v142, -v222, v216, v223
	.loc	0 2082 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v143, v143, v217, v218
	.loc	0 2084 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s3
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v80, v80, v102 :: v_dual_mul_f32 v77, v77, v102
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_f32 v141, -v141, v213, v224
	.loc	0 2084 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v142, v142, v214, v216
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 vcc_lo, s6
	.loc	0 2082 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v137, v143, v99, v137
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v78, v78, v102 :: v_dual_mul_f32 v75, v75, v102
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v141, v141, v220, v213
	.loc	0 2084 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v103, v142, v99, v103
	.loc	0 2081 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cvt_pk_fp8_f32 v151.l, v138, v137
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[137:140], v233 offset:16384
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2084 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_fixup_f32 v104, v141, v99, v104
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[141:144], v233 offset:16896
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[213:216], v233 offset:17408
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[217:220], v233 offset:17920
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[221:224], v233 offset:18432
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[225:228], v233 offset:18944
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[229:232], v233 offset:19456
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2091 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[233:236], v233 offset:19968
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v76, v76, v102 :: v_dual_mul_f32 v73, v73, v102
	v_dual_mul_f32 v74, v74, v102 :: v_dual_mul_f32 v71, v71, v102
	v_dual_mul_f32 v72, v72, v102 :: v_dual_mul_f32 v69, v69, v102
	v_dual_mul_f32 v70, v70, v102 :: v_dual_mul_f32 v67, v67, v102
	v_dual_mul_f32 v68, v68, v102 :: v_dual_mul_f32 v65, v65, v102
	v_dual_mul_f32 v66, v66, v102 :: v_dual_mul_f32 v63, v63, v102
	v_dual_mul_f32 v64, v64, v102 :: v_dual_mul_f32 v61, v61, v102
	v_dual_mul_f32 v62, v62, v102 :: v_dual_mul_f32 v59, v59, v102
	v_dual_mul_f32 v60, v60, v102 :: v_dual_mul_f32 v57, v57, v102
	v_dual_mul_f32 v58, v58, v102 :: v_dual_mul_f32 v55, v55, v102
	v_dual_mul_f32 v56, v56, v102 :: v_dual_mul_f32 v53, v53, v102
	v_dual_mul_f32 v54, v54, v102 :: v_dual_mul_f32 v51, v51, v102
	v_dual_mul_f32 v52, v52, v102 :: v_dual_mul_f32 v49, v49, v102
	v_dual_mul_f32 v50, v50, v102 :: v_dual_mul_f32 v47, v47, v102
	v_dual_mul_f32 v48, v48, v102 :: v_dual_mul_f32 v45, v45, v102
	v_dual_mul_f32 v46, v46, v102 :: v_dual_mul_f32 v43, v43, v102
	v_dual_mul_f32 v44, v44, v102 :: v_dual_mul_f32 v41, v41, v102
	v_dual_mul_f32 v42, v42, v102 :: v_dual_mul_f32 v39, v39, v102
	v_dual_mul_f32 v40, v40, v102 :: v_dual_mul_f32 v37, v37, v102
	v_dual_mul_f32 v38, v38, v102 :: v_dual_mul_f32 v35, v35, v102
	v_dual_mul_f32 v36, v36, v102 :: v_dual_mul_f32 v33, v33, v102
	v_dual_mul_f32 v34, v34, v102 :: v_dual_mul_f32 v31, v31, v102
	v_dual_mul_f32 v32, v32, v102 :: v_dual_mul_f32 v29, v29, v102
	v_dual_mul_f32 v30, v30, v102 :: v_dual_mul_f32 v27, v27, v102
	v_dual_mul_f32 v28, v28, v102 :: v_dual_mul_f32 v25, v25, v102
	v_dual_mul_f32 v26, v26, v102 :: v_dual_mul_f32 v23, v23, v102
	v_dual_mul_f32 v24, v24, v102 :: v_dual_mul_f32 v21, v21, v102
	v_dual_mul_f32 v22, v22, v102 :: v_dual_mul_f32 v19, v19, v102
	v_dual_mul_f32 v20, v20, v102 :: v_dual_mul_f32 v17, v17, v102
	v_dual_mul_f32 v18, v18, v102 :: v_dual_mul_f32 v15, v15, v102
	v_dual_mul_f32 v14, v14, v102 :: v_dual_mul_f32 v11, v11, v102
	v_dual_mul_f32 v12, v12, v102 :: v_dual_mul_f32 v7, v7, v102
	v_dual_mul_f32 v10, v10, v102 :: v_dual_mul_f32 v5, v5, v102
	.loc	0 2083 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2083:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cvt_pk_fp8_f32 v151.h, v103, v104
	.loc	0 2074 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2074:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v4, v4, v102
	v_mul_f32_e32 v2, v2, v102
	.loc	0 2100 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2100:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[137:138], v[150:151], v[129:136]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[139:140], v[150:151], v[121:128]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[141:142], v[150:151], v[113:120]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[150:151], v[105:112]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[213:214], v[150:151], v[89:96]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[215:216], v[150:151], v[81:88]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[217:218], v[150:151], v[73:80]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[219:220], v[150:151], v[65:72]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[221:222], v[150:151], v[57:64]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[223:224], v[150:151], v[49:56]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[225:226], v[150:151], v[41:48]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[227:228], v[150:151], v[33:40]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[229:230], v[150:151], v[25:32]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[231:232], v[150:151], v[17:24]
	.loc	0 2095 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2095:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[233:234], v[150:151], v[9:16]
	.loc	0 2098 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[235:236], v[150:151], v[1:8]
	v_mov_b32_e32 v151, v100
	v_dual_mov_b32 v213, v99 :: v_dual_mov_b32 v150, v98
	.loc	0 1986 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_add_co_i32 s26, s26, 1
	.loc	0 1986 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s26, 4
	.loc	0 1986 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_scc1 .LBB6_11
.LBB6_48:                               ;   Parent Loop BB6_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB6_53 Depth 3
	.loc	0 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s26, 1
	s_mov_b32 s3, -1
	s_cbranch_scc1 .LBB6_51
; %bb.49:                               ;   in Loop: Header=BB6_48 Depth=2
	s_cmp_eq_u32 s26, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s3, s27
	s_cbranch_scc1 .LBB6_51
; %bb.50:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 1988 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cmp_eq_u32 s26, 2
	.loc	0 1988 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cselect_b32 s3, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s23, s3
	s_cselect_b32 s3, -1, 0
.LBB6_51:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_58
; %bb.52:                               ;   in Loop: Header=BB6_48 Depth=2
	v_mov_b32_e32 v137, 0
	.loc	0 1998 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshl_add_u32 v98, s26, 9, v156
	s_mov_b32 s3, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB6_53:                               ;   Parent Loop BB6_13 Depth=1
                                        ;     Parent Loop BB6_48 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	.loc	0 2010 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s3, 6
	.loc	0 2003 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2003:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshl_add_u32 v104, s3, 12, v98
	.loc	0 2013 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v99, vcc_lo, v202, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v100, null, 0, v203, vcc_lo
	.loc	0 2005 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2005:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b128 v[214:217], v104
	ds_load_b128 v[218:221], v104 offset:2048
	.loc	0 1998 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_add_co_i32 s3, s3, 1
	.loc	0 2013 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_clause 0x3
	global_load_b64 v[102:103], v[99:100], off
	global_load_b64 v[222:223], v[99:100], off offset:16
	global_load_b64 v[224:225], v[99:100], off offset:32
	global_load_b64 v[99:100], v[99:100], off offset:48
	.loc	0 2022 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	0 1998 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s3, 4
	.loc	0 2016 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[214:215], v[102:103], v[137:144]
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[216:217], v[222:223], v[137:144]
	.loc	0 2016 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2016:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[218:219], v[224:225], v[137:144]
	.loc	0 2019 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2019:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[220:221], v[99:100], v[137:144]
	.loc	0 1998 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_scc1 .LBB6_53
; %bb.54:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 0 17 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:17
	v_lshl_add_u32 v102, s26, 5, v161
	.loc	0 2025 48 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_lshl_b32 s3, s26, 4
	v_mov_b32_e32 v103, 0xff800000
	.loc	0 2025 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s21
	.loc	0 2030 41 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_b96 v[98:100], v102 offset:49154
	ds_load_u16_d16 v104, v102 offset:49166
	.loc	0 2026 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2026:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s3, 15
	v_or_b32_e32 v214, s3, v158
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s24
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s25, s3
	.loc	0 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB6_56
; %bb.55:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 2030 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_u16_d16 v103, v102 offset:49152
	v_mul_f32_e32 v137, v152, v137
	v_cmp_le_i32_e32 vcc_lo, v214, v101
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v103, v103, v137, neg(0) op_sel_hi:[1,0,0]
	.loc	0 2037 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2037:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v103, 0xff800000, v103, vcc_lo
.LBB6_56:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v137, v152, v138
	v_or_b32_e32 v138, 2, v214
	v_cmp_lt_i32_e32 vcc_lo, v214, v101
	v_mul_f32_e32 v139, v152, v139
	s_wait_dscnt 0x1
	v_fma_mix_f32 v137, v98, v137, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s3, v138, v101
	v_or_b32_e32 v138, 3, v214
	s_or_b32 s5, s4, vcc_lo
	v_fma_mix_f32 v98, v98, v139, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2035 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	s_or_b32 s3, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v137, 0xff800000, v137, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v138, v101
	v_mul_f32_e32 v138, v152, v140
	v_or_b32_e32 v139, 4, v214
	s_and_b32 s3, s0, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, 0xff800000, v98, s3
	v_fma_mix_f32 v98, v99, v138, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v138, v152, v141
	s_or_b32 s3, s4, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v139, v101
	v_or_b32_e32 v139, 5, v214
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s0, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v98, s3
	v_fma_mix_f32 v98, v99, v138, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v99, v152, v142
	s_or_b32 s3, s4, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v139, v101
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s0, s3
	v_or_b32_e32 v139, 7, v214
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v98, s3
	v_fma_mix_f32 v98, v100, v99, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v99, 6, v214
	s_or_b32 s3, s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s3
	v_cmp_le_i32_e64 s3, v139, v101
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v142, 0xff800000, v98, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v99, v101
	v_mul_f32_e32 v98, v152, v143
.Ltmp620:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v139, v103, 0xff800000, v137
	s_or_b32 s3, s4, s3
	s_or_b32 s5, s4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_mix_f32 v98, v100, v98, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp621:
	.loc	0 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s0, s5
	v_mul_f32_e32 v99, v152, v144
.Ltmp622:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v100, v139, v140, v141
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_mix_f32 v99, v104, v99, neg(0) op_sel_hi:[1,0,0]
.Ltmp623:
	.loc	0 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v104, 0xff800000, v98, vcc_lo
	s_and_b32 vcc_lo, s0, s3
.Ltmp624:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v98, v100, v138, v142
.Ltmp625:
	.loc	0 2069 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_mov_b32 s3, exec_lo
	.loc	0 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v99, 0xff800000, v99, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp626:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2043:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v98, v98, v104, v99
.Ltmp627:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2044:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_mov_b32_e32 v100, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v100, v100, s7, 0xfedcba98
.Ltmp628:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2045:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v98, v150, v98, v100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v100, v103, v98 :: v_dual_sub_f32 v103, v137, v98
	v_dual_sub_f32 v99, v99, v98 :: v_dual_sub_f32 v104, v104, v98
	v_add_nc_u32_e32 v137, 0xc080, v102
	v_mul_f32_e32 v100, 0x3fb8aa3b, v100
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v99, 0x3fb8aa3b, v99 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v104
	v_exp_f32_e32 v139, v99
	v_dual_sub_f32 v99, v140, v98 :: v_dual_sub_f32 v140, v141, v98
	v_mul_f32_e32 v103, 0x3fb8aa3b, v103
	v_exp_f32_e32 v141, v100
	v_exp_f32_e32 v144, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v143, v103
	v_dual_sub_f32 v103, v142, v98 :: v_dual_mul_f32 v142, 0x3fb8aa3b, v99
.Ltmp629:
	.loc	0 2051 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	ds_load_2addr_b32 v[99:100], v137 offset1:1
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v214, v139, 0, vcc_lo
	v_cndmask_b32_e64 v139, v141, 0, vcc_lo
	v_mul_f32_e32 v103, 0x3fb8aa3b, v103
	v_sub_f32_e32 v137, v138, v98
	v_exp_f32_e32 v138, v140
	v_cndmask_b32_e64 v143, v143, 0, vcc_lo
	.loc	0 2051 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_nc_u32_e32 v102, 0xc088, v102
	v_exp_f32_e32 v140, v103
	v_mul_f32_e32 v137, 0x3fb8aa3b, v137
	ds_load_2addr_b32 v[103:104], v102 offset1:1
	v_exp_f32_e32 v102, v142
	v_exp_f32_e32 v137, v137
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cndmask_b32_e64 v216, v138, 0, vcc_lo
	v_cndmask_b32_e64 v215, v140, 0, vcc_lo
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v142, v99, v139, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v99, v143, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cndmask_b32_e64 v102, v102, 0, vcc_lo
	v_cndmask_b32_e64 v99, v137, 0, vcc_lo
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_f32_e32 v137, v139, v143
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_mix_f32 v139, v100, v216, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	0 2055 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2055:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_mix_f32 v140, v100, v102, neg(0) op_sel_hi:[1,0,0]
.Ltmp630:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v100, v142, 0, v141
.Ltmp631:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_f32_e32 v102, v102, v137
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v138, v103, v99, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v137, v103, v215, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp632:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v100, v100, v140, v139
.Ltmp633:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_f32_e32 v102, v216, v102
	.loc	0 2059 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_fma_mix_f32 v103, v104, v143, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v104, v104, v214, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp634:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v100, v100, v138, v137
.Ltmp635:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_f32_e32 v99, v99, v102
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp636:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max3_num_f32 v144, v100, v103, v104
.Ltmp637:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_add_f32 v99, v215, v99 :: v_dual_mov_b32 v102, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v99, v143, v99
.Ltmp638:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_permlanex16_b32 v102, v102, s7, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp639:
	.loc	0 2058 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2058:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_f32_e32 v100, v214, v99
.Ltmp640:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2067:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max_num_f32_e32 v99, v102, v102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_num_f32_e32 v143, v144, v99
.Ltmp641:
	.loc	0 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2061:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_dual_mov_b32 v102, v100 :: v_dual_mov_b32 v99, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v102, v102, s7, 0xfedcba98
.Ltmp642:
	.loc	0 2069 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2069:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB6_47
; %bb.57:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	0 2070 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_div_scale_f32 v99, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v99
	v_fma_f32 v214, -v99, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v214, v144
	v_div_scale_f32 v214, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v215, v214, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v216, -v99, v215, v214
	v_fmac_f32_e32 v215, v216, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v99, -v99, v215, v214
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v99, v99, v144, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v99, v99, 0x43e00000, v143
.Ltmp643:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2070:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ] ]
	v_max_num_f32_e32 v99, 0x1f800000, v99
	s_branch .LBB6_47
.Ltmp644:
.LBB6_58:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	2 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v98, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v150, v98
	.loc	0 1986 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_add_co_i32 s26, s26, 1
	.loc	0 1986 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s26, 4
	.loc	0 1986 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_cbranch_scc0 .LBB6_48
	s_branch .LBB6_11
.LBB6_59:
	.loc	0 2125 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_cmp_eq_u32_e32 vcc_lo, 0, v153
	s_and_b32 s2, vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB6_61
; %bb.60:
	.loc	0 2129 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2129:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_u64_u32 v[97:98], null, s17, v145, s[12:13]
	.loc	0 2131 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mov_b32_e32 v98, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2130 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2130:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_lo_u32 v97, 0x102, v97
	.loc	0 2131 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshlrev_b64_e32 v[97:98], 2, v[97:98]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v97, vcc_lo, s10, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v98, null, s11, v98, vcc_lo
	.loc	0 2133 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2133:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_store_b64 v[97:98], v[150:151], off
.LBB6_61:
	.loc	0 0 24 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:24
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	0 2136 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2136:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_63
; %bb.62:
	.loc	0 2139 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2139:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mad_co_u64_u32 v[98:99], null, s17, v145, s[12:13]
	.loc	0 2141 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mov_b32 v99, 0 :: v_dual_mul_f32 v0, v213, v129
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v97, v121, v213 :: v_dual_lshlrev_b32 v100, 2, v158
	v_mul_f32_e32 v101, v113, v213
	v_dual_mul_f32 v105, v105, v213 :: v_dual_mul_f32 v102, v114, v213
	.loc	0 2140 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2140:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_lo_u32 v98, 0x102, v98
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v113, v1, v213 :: v_dual_mul_f32 v10, v10, v213
	v_dual_mul_f32 v1, v213, v130 :: v_dual_mul_f32 v114, v2, v213
	v_dual_mul_f32 v2, v213, v131 :: v_dual_mul_f32 v103, v115, v213
	v_mul_f32_e32 v130, v213, v134
	.loc	0 2141 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_lshlrev_b64_e32 v[98:99], 2, v[98:99]
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v115, v3, v213 :: v_dual_mul_f32 v28, v28, v213
	v_dual_mul_f32 v3, v213, v132 :: v_dual_mul_f32 v20, v20, v213
	v_dual_mul_f32 v89, v89, v213 :: v_dual_mul_f32 v106, v106, v213
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	0 2141 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2141:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_co_u32 v98, vcc_lo, s10, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, s11, v99, vcc_lo
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v81, v81, v213 :: v_dual_mul_f32 v90, v90, v213
	.loc	0 2147 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_add_co_u32 v137, vcc_lo, v98, v100
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v138, null, 0, v99, vcc_lo
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v73, v73, v213 :: v_dual_mul_f32 v82, v82, v213
	v_dual_mul_f32 v65, v65, v213 :: v_dual_mul_f32 v74, v74, v213
	v_dual_mul_f32 v57, v57, v213 :: v_dual_mul_f32 v66, v66, v213
	v_dual_mul_f32 v49, v49, v213 :: v_dual_mul_f32 v58, v58, v213
	v_dual_mul_f32 v41, v41, v213 :: v_dual_mul_f32 v50, v50, v213
	v_dual_mul_f32 v33, v33, v213 :: v_dual_mul_f32 v42, v42, v213
	v_dual_mul_f32 v25, v25, v213 :: v_dual_mul_f32 v34, v34, v213
	v_dual_mul_f32 v17, v17, v213 :: v_dual_mul_f32 v26, v26, v213
	v_dual_mul_f32 v9, v9, v213 :: v_dual_mul_f32 v18, v18, v213
	v_dual_mul_f32 v98, v122, v213 :: v_dual_mul_f32 v99, v123, v213
	v_dual_mul_f32 v107, v107, v213 :: v_dual_mul_f32 v132, v213, v136
	v_dual_mul_f32 v91, v91, v213 :: v_dual_mul_f32 v100, v124, v213
	v_dual_mul_f32 v83, v83, v213 :: v_dual_mul_f32 v104, v116, v213
	v_dual_mul_f32 v75, v75, v213 :: v_dual_mul_f32 v108, v108, v213
	v_dual_mul_f32 v67, v67, v213 :: v_dual_mul_f32 v92, v92, v213
	v_dual_mul_f32 v59, v59, v213 :: v_dual_mul_f32 v84, v84, v213
	v_dual_mul_f32 v51, v51, v213 :: v_dual_mul_f32 v76, v76, v213
	v_dual_mul_f32 v43, v43, v213 :: v_dual_mul_f32 v68, v68, v213
	v_dual_mul_f32 v35, v35, v213 :: v_dual_mul_f32 v60, v60, v213
	v_dual_mul_f32 v27, v27, v213 :: v_dual_mul_f32 v52, v52, v213
	v_dual_mul_f32 v19, v19, v213 :: v_dual_mul_f32 v44, v44, v213
	v_dual_mul_f32 v11, v11, v213 :: v_dual_mul_f32 v36, v36, v213
	v_mul_f32_e32 v129, v213, v133
	v_dual_mul_f32 v131, v213, v135 :: v_dual_mul_f32 v12, v12, v213
	.loc	0 2147 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	global_store_b128 v[137:138], v[0:3], off offset:8
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_dual_mul_f32 v116, v4, v213 :: v_dual_mul_f32 v109, v109, v213
	v_dual_mul_f32 v0, v125, v213 :: v_dual_mul_f32 v1, v126, v213
	v_dual_mul_f32 v4, v117, v213 :: v_dual_mul_f32 v3, v128, v213
	v_dual_mul_f32 v93, v93, v213 :: v_dual_mul_f32 v2, v127, v213
	v_dual_mul_f32 v85, v85, v213 :: v_dual_mul_f32 v110, v110, v213
	v_dual_mul_f32 v77, v77, v213 :: v_dual_mul_f32 v94, v94, v213
	v_dual_mul_f32 v69, v69, v213 :: v_dual_mul_f32 v86, v86, v213
	v_dual_mul_f32 v61, v61, v213 :: v_dual_mul_f32 v78, v78, v213
	v_dual_mul_f32 v53, v53, v213 :: v_dual_mul_f32 v70, v70, v213
	v_dual_mul_f32 v45, v45, v213 :: v_dual_mul_f32 v62, v62, v213
	v_dual_mul_f32 v37, v37, v213 :: v_dual_mul_f32 v54, v54, v213
	v_dual_mul_f32 v29, v29, v213 :: v_dual_mul_f32 v46, v46, v213
	v_dual_mul_f32 v21, v21, v213 :: v_dual_mul_f32 v38, v38, v213
	v_dual_mul_f32 v13, v13, v213 :: v_dual_mul_f32 v30, v30, v213
	v_dual_mul_f32 v117, v5, v213 :: v_dual_mul_f32 v22, v22, v213
	v_mul_f32_e32 v5, v118, v213
	v_dual_mul_f32 v14, v14, v213 :: v_dual_mul_f32 v111, v111, v213
	v_dual_mul_f32 v118, v6, v213 :: v_dual_mul_f32 v95, v95, v213
	v_mul_f32_e32 v6, v119, v213
	v_dual_mul_f32 v87, v87, v213 :: v_dual_mul_f32 v112, v112, v213
	v_dual_mul_f32 v79, v79, v213 :: v_dual_mul_f32 v96, v96, v213
	v_dual_mul_f32 v71, v71, v213 :: v_dual_mul_f32 v88, v88, v213
	v_dual_mul_f32 v63, v63, v213 :: v_dual_mul_f32 v80, v80, v213
	v_dual_mul_f32 v55, v55, v213 :: v_dual_mul_f32 v72, v72, v213
	v_dual_mul_f32 v47, v47, v213 :: v_dual_mul_f32 v64, v64, v213
	v_dual_mul_f32 v39, v39, v213 :: v_dual_mul_f32 v56, v56, v213
	v_dual_mul_f32 v31, v31, v213 :: v_dual_mul_f32 v48, v48, v213
	v_dual_mul_f32 v23, v23, v213 :: v_dual_mul_f32 v40, v40, v213
	v_dual_mul_f32 v15, v15, v213 :: v_dual_mul_f32 v32, v32, v213
	v_dual_mul_f32 v119, v7, v213 :: v_dual_mul_f32 v24, v24, v213
	v_mul_f32_e32 v7, v120, v213
	v_mul_f32_e32 v16, v16, v213
	.loc	0 2147 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_clause 0x18
	global_store_b128 v[137:138], v[129:132], off offset:24
	global_store_b128 v[137:138], v[97:100], off offset:72
	global_store_b128 v[137:138], v[0:3], off offset:88
	global_store_b128 v[137:138], v[101:104], off offset:136
	global_store_b128 v[137:138], v[4:7], off offset:152
	global_store_b128 v[137:138], v[105:108], off offset:200
	global_store_b128 v[137:138], v[109:112], off offset:216
	global_store_b128 v[137:138], v[89:92], off offset:264
	global_store_b128 v[137:138], v[93:96], off offset:280
	global_store_b128 v[137:138], v[81:84], off offset:328
	global_store_b128 v[137:138], v[85:88], off offset:344
	global_store_b128 v[137:138], v[73:76], off offset:392
	global_store_b128 v[137:138], v[77:80], off offset:408
	global_store_b128 v[137:138], v[65:68], off offset:456
	global_store_b128 v[137:138], v[69:72], off offset:472
	global_store_b128 v[137:138], v[57:60], off offset:520
	global_store_b128 v[137:138], v[61:64], off offset:536
	global_store_b128 v[137:138], v[49:52], off offset:584
	global_store_b128 v[137:138], v[53:56], off offset:600
	global_store_b128 v[137:138], v[41:44], off offset:648
	global_store_b128 v[137:138], v[45:48], off offset:664
	global_store_b128 v[137:138], v[33:36], off offset:712
	global_store_b128 v[137:138], v[37:40], off offset:728
	global_store_b128 v[137:138], v[25:28], off offset:776
	global_store_b128 v[137:138], v[29:32], off offset:792
	.loc	0 2147 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	v_mul_f32_e32 v120, v8, v213
	.loc	0 2147 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2147:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2217:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[17:20], off offset:840
	global_store_b128 v[137:138], v[21:24], off offset:856
	global_store_b128 v[137:138], v[9:12], off offset:904
	global_store_b128 v[137:138], v[13:16], off offset:920
	global_store_b128 v[137:138], v[113:116], off offset:968
	global_store_b128 v[137:138], v[117:120], off offset:984
.Ltmp645:
.LBB6_63:
	.loc	0 2220 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2220:1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp646:
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
		.amdhsa_next_free_sgpr 28
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 28
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9972
; TotalNumSgprs: 30
; NumVgprs: 237
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 237
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
	.loc	0 2231 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2231:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	0 2231 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2231:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB7_10
; %bb.1:
	.loc	0 2236 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2236:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	0 2239 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2239:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	0 2238 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2238:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	0 2239 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2239:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_10
; %bb.2:
	.loc	0 2231 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2231:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	0 2242 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2242:5
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
	.loc	0 2245 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2245:22
	global_load_b32 v3, v[6:7], off
.Ltmp647:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2245:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp648:
	.loc	0 2242 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2242:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp649:
	.loc	2 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2245:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp650:
	.loc	0 2242 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2242:5
	s_cbranch_scc0 .LBB7_3
; %bb.4:
	.loc	0 2237 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2237:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	0 2247 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2247:5
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
	.loc	0 2259 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:13
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
	.loc	0 2247 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2247:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	0 2259 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	0 2258 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2258:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	0 2247 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2247:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	0 2259 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	0 2258 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2258:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	0 2259 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	0 2247 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2247:5
	s_or_b32 s1, vcc_lo, s1
	.loc	0 2259 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2259:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	0 2247 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2247:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	0 2258 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2258:44
	global_store_b32 v[5:6], v11, off
	.loc	0 2247 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2247:5
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
	.loc	0 2256 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	0 2250 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2250:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	0 2256 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:24
	global_load_b32 v15, v[15:16], off
	.loc	0 2255 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2255:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	0 2250 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2250:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	0 2256 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2256:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	0 2250 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2250:9
	s_cbranch_scc1 .LBB7_5
.LBB7_8:                                ;   Parent Loop BB7_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	0 2254 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	0 2254 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:17
	s_mov_b32 s2, exec_lo
	.loc	0 2254 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	0 2254 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:17
	s_cbranch_execz .LBB7_7
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=2
	.loc	0 2254 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:41
	global_load_b32 v14, v[5:6], off
	.loc	0 2254 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp651:
	.loc	2 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	2 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2254:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB7_7
.Ltmp652:
.LBB7_10:
	.loc	0 2261 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2261:1
	s_endpgm
.Ltmp653:
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
	.type	__hip_cuid_995e48e1443ce375,@object ; @__hip_cuid_995e48e1443ce375
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_995e48e1443ce375
__hip_cuid_995e48e1443ce375:
	.byte	0                               ; 0x0
	.size	__hip_cuid_995e48e1443ce375, 1

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
	.byte	1                               ; Abbrev [1] 0xc:0xaa8 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x5e7:0x27c DW_TAG_subprogram
	.byte	64                              ; DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	25                              ; DW_AT_name
	.byte	5                               ; Abbrev [5] 0x5ee:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	65                              ; DW_AT_low_pc
	.long	.Ltmp242-.Ltmp241               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2179                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5fc:0x266 DW_TAG_inlined_subroutine
	.long	1507                            ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2183                            ; DW_AT_call_line
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
	.byte	4                               ; Abbrev [4] 0x646:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x650:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	67                              ; DW_AT_low_pc
	.long	.Ltmp316-.Ltmp315               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x65f:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x669:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x673:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	37                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x67d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x687:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	68                              ; DW_AT_low_pc
	.long	.Ltmp329-.Ltmp328               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x695:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	69                              ; DW_AT_low_pc
	.long	.Ltmp334-.Ltmp333               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6a3:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	70                              ; DW_AT_low_pc
	.long	.Ltmp336-.Ltmp335               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6b1:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	71                              ; DW_AT_low_pc
	.long	.Ltmp339-.Ltmp338               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6bf:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	72                              ; DW_AT_low_pc
	.long	.Ltmp342-.Ltmp341               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6cd:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	73                              ; DW_AT_low_pc
	.long	.Ltmp343-.Ltmp342               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6db:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	74                              ; DW_AT_low_pc
	.long	.Ltmp345-.Ltmp344               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6e9:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	75                              ; DW_AT_low_pc
	.long	.Ltmp346-.Ltmp345               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6f7:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	76                              ; DW_AT_low_pc
	.long	.Ltmp348-.Ltmp347               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x705:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	77                              ; DW_AT_low_pc
	.long	.Ltmp352-.Ltmp351               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x713:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	78                              ; DW_AT_low_pc
	.long	.Ltmp354-.Ltmp353               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x721:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	79                              ; DW_AT_low_pc
	.long	.Ltmp356-.Ltmp355               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x72f:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	80                              ; DW_AT_low_pc
	.long	.Ltmp357-.Ltmp356               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x73d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp358-.Ltmp357               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x74b:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp359-.Ltmp358               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x759:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp361-.Ltmp360               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2108                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x767:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp361-.Ltmp360               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x775:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp361-.Ltmp360               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x785:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp363-.Ltmp362               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1917                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x793:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp363-.Ltmp362               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7a1:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp363-.Ltmp362               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x7b1:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1939                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7bb:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1941                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7c5:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1943                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7cf:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1945                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7d9:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1972                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x7e3:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7ed:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x7f9:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2043                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x803:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp454-.Ltmp453               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2044                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x811:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	86                              ; DW_AT_low_pc
	.long	.Ltmp455-.Ltmp454               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2045                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x81f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2066                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x829:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	87                              ; DW_AT_low_pc
	.long	.Ltmp465-.Ltmp464               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2067                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x837:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	88                              ; DW_AT_low_pc
	.long	.Ltmp467-.Ltmp466               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2067                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x845:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	89                              ; DW_AT_low_pc
	.long	.Ltmp468-.Ltmp467               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2061                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x853:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	90                              ; DW_AT_low_pc
	.long	.Ltmp470-.Ltmp469               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2070                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x863:0x2 DW_TAG_subprogram
	.byte	19                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x865:0x22e DW_TAG_subprogram
	.byte	91                              ; DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	26                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x86c:0x226 DW_TAG_inlined_subroutine
	.long	2147                            ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2217                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x876:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x880:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	92                              ; DW_AT_low_pc
	.long	.Ltmp477-.Ltmp476               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x88f:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x899:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8a3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	93                              ; DW_AT_low_pc
	.long	.Ltmp490-.Ltmp489               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x8b1:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8bb:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	94                              ; DW_AT_low_pc
	.long	.Ltmp495-.Ltmp494               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1855                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8c9:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	95                              ; DW_AT_low_pc
	.long	.Ltmp496-.Ltmp495               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1860                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x8d7:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8e1:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	96                              ; DW_AT_low_pc
	.long	.Ltmp500-.Ltmp499               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8ef:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	97                              ; DW_AT_low_pc
	.long	.Ltmp503-.Ltmp502               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1856                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8fd:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	98                              ; DW_AT_low_pc
	.long	.Ltmp504-.Ltmp503               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1861                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x90b:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	99                              ; DW_AT_low_pc
	.long	.Ltmp510-.Ltmp509               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x919:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	100                             ; DW_AT_low_pc
	.long	.Ltmp516-.Ltmp515               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1862                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x927:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	101                             ; DW_AT_low_pc
	.long	.Ltmp517-.Ltmp516               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1857                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x935:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	102                             ; DW_AT_low_pc
	.long	.Ltmp519-.Ltmp518               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x943:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	103                             ; DW_AT_low_pc
	.long	.Ltmp525-.Ltmp524               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x951:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	104                             ; DW_AT_low_pc
	.long	.Ltmp526-.Ltmp525               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1858                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x95f:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp527-.Ltmp526               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x96d:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp530-.Ltmp529               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1864                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x97b:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp531-.Ltmp530               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1859                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x989:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp533-.Ltmp532               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2108                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x997:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp533-.Ltmp532               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9a5:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp533-.Ltmp532               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x9b5:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp535-.Ltmp534               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	1917                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x9c3:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp535-.Ltmp534               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9d1:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp535-.Ltmp534               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x9e1:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1939                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9eb:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1941                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9f5:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1943                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9ff:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1945                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xa09:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	1972                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xa13:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa1d:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0xa29:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	57                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2043                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa33:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	110                             ; DW_AT_low_pc
	.long	.Ltmp628-.Ltmp627               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2044                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa41:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp629-.Ltmp628               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2045                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa4f:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2066                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa59:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	112                             ; DW_AT_low_pc
	.long	.Ltmp639-.Ltmp638               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2067                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa67:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	113                             ; DW_AT_low_pc
	.long	.Ltmp641-.Ltmp640               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2067                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa75:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	114                             ; DW_AT_low_pc
	.long	.Ltmp642-.Ltmp641               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2061                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa83:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	115                             ; DW_AT_low_pc
	.long	.Ltmp644-.Ltmp643               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2070                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0xa93:0x20 DW_TAG_subprogram
	.byte	116                             ; DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	27                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0xa9a:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.short	2245                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xaa4:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	117                             ; DW_AT_low_pc
	.long	.Ltmp652-.Ltmp651               ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.short	2254                            ; DW_AT_call_line
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
	.long	60                              ; Offset entry count
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
	.uleb128 .Ltmp471-.Lfunc_begin0         ;   ending offset
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
	.uleb128 .Ltmp315-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp317-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp319-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges35:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp317-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp318-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp324-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp328-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges36:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp318-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp319-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp352-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp353-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges37:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp329-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp330-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp334-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp343-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp344-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges38:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp333-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp339-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp349-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp350-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges39:
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
	.uleb128 .Ltmp384-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp385-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp386-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp387-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp394-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp395-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp396-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp397-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp404-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp405-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp406-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp407-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp414-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp415-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp416-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp417-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp424-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp425-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp426-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp427-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp434-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp435-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp436-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp437-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges40:
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
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges41:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp369-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp372-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp379-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp380-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp381-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp382-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp389-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp390-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp391-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp392-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp399-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp400-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp401-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp402-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp409-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp410-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp411-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp412-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp419-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp420-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp421-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp422-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp429-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp430-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp431-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp432-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp439-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp440-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp441-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp442-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges42:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp372-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp373-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp380-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp381-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp382-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp383-.Lfunc_begin0         ;   ending offset
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
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges43:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp444-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp445-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp446-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp447-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges44:
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
.Ldebug_ranges45:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp456-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp457-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp458-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp459-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp460-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp461-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp462-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp463-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges46:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp473-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp474-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp475-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp482-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp485-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp486-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp487-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp488-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp493-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp494-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp497-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp498-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp512-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp513-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp514-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp515-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp517-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp518-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp520-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp521-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp522-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp523-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp645-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges47:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp476-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp477-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp478-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp479-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp480-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp483-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp484-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp488-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp489-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges48:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp479-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp480-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp484-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp485-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp486-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp487-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp490-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp491-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp498-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp499-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges49:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp482-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp483-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp504-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp505-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp506-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp507-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp510-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp511-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp519-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp520-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges50:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp491-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp492-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp507-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp508-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp511-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp512-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp513-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp514-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp527-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp528-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges51:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp496-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp497-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp501-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp502-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp505-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp506-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp508-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp509-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges52:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp536-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp537-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp538-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp539-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp546-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp547-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp548-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp549-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp556-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp557-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp558-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp559-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp566-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp567-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp568-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp569-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp576-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp577-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp578-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp579-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp586-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp587-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp588-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp589-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp596-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp597-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp598-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp599-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp606-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp607-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp608-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp609-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges53:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp537-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp538-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp539-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp540-.Lfunc_begin0         ;   ending offset
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp597-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp598-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp599-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp600-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp607-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp608-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp609-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp610-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges54:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp541-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp542-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp543-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp544-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp551-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp552-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp553-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp554-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp561-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp562-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp563-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp564-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp571-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp572-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp573-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp574-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp581-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp582-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp583-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp584-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp591-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp592-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp593-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp594-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp601-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp602-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp603-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp604-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp611-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp612-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp613-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp614-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges55:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp542-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp543-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp544-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp545-.Lfunc_begin0         ;   ending offset
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp602-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp603-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp604-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp605-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp612-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp613-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp614-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp615-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges56:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp616-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp617-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp618-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp619-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges57:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp620-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp621-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp622-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp623-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp624-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp625-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp626-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp627-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges58:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp630-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp631-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp632-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp633-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp634-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp635-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp636-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp637-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges59:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp647-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp648-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp649-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp650-.Lfunc_begin0         ;   ending offset
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
	.asciz	"/home/kaden/ClaudeCode/warpfront/wt-lloyd" ; string offset=159 ; /home/kaden/ClaudeCode/warpfront/wt-lloyd
.Linfo_string3:
	.asciz	"fa2_stageb_nbody<false>"       ; string offset=201 ; fa2_stageb_nbody<false>
.Linfo_string4:
	.asciz	"__lane_id"                     ; string offset=225 ; __lane_id
.Linfo_string5:
	.asciz	"__shfl_xor"                    ; string offset=235 ; __shfl_xor
.Linfo_string6:
	.asciz	"max"                           ; string offset=246 ; max
.Linfo_string7:
	.asciz	"min"                           ; string offset=250 ; min
.Linfo_string8:
	.asciz	"__work_group_barrier"          ; string offset=254 ; __work_group_barrier
.Linfo_string9:
	.asciz	"__barrier"                     ; string offset=275 ; __barrier
.Linfo_string10:
	.asciz	"__syncthreads"                 ; string offset=285 ; __syncthreads
.Linfo_string11:
	.asciz	"fa2_scale_n"                   ; string offset=299 ; fa2_scale_n
.Linfo_string12:
	.asciz	"fmaxf"                         ; string offset=311 ; fmaxf
.Linfo_string13:
	.asciz	"__hip_get_thread_idx_x"        ; string offset=317 ; __hip_get_thread_idx_x
.Linfo_string14:
	.asciz	"__get_x"                       ; string offset=340 ; __get_x
.Linfo_string15:
	.asciz	"fa2_stageb_nbody<true>"        ; string offset=348 ; fa2_stageb_nbody<true>
.Linfo_string16:
	.asciz	"__expf"                        ; string offset=371 ; __expf
.Linfo_string17:
	.asciz	"fa2_stageb_packet_body<false, true>" ; string offset=378 ; fa2_stageb_packet_body<false, true>
.Linfo_string18:
	.asciz	"fa2_pkt_xor16"                 ; string offset=414 ; fa2_pkt_xor16
.Linfo_string19:
	.asciz	"fa2_stageb_packet_body<true, false>" ; string offset=428 ; fa2_stageb_packet_body<true, false>
.Linfo_string20:
	.asciz	"attention_fp8_e4m3_fa2_gqa_gfx1201" ; string offset=464 ; attention_fp8_e4m3_fa2_gqa_gfx1201
.Linfo_string21:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201" ; string offset=499 ; attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
.Linfo_string22:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201" ; string offset=547 ; attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
.Linfo_string23:
	.asciz	"attention_fp8_e4m3_fa2_gqa_partial_gfx1201" ; string offset=595 ; attention_fp8_e4m3_fa2_gqa_partial_gfx1201
.Linfo_string24:
	.asciz	"attention_fp8_e4m3_fa2_gqa_merge_gfx1201" ; string offset=638 ; attention_fp8_e4m3_fa2_gqa_merge_gfx1201
.Linfo_string25:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_gfx1201" ; string offset=679 ; attention_fp8_e4m3_fa2_gqa_packet_gfx1201
.Linfo_string26:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201" ; string offset=721 ; attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
.Linfo_string27:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201" ; string offset=771 ; attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
	.quad	.Ltmp328
	.quad	.Ltmp333
	.quad	.Ltmp335
	.quad	.Ltmp338
	.quad	.Ltmp341
	.quad	.Ltmp342
	.quad	.Ltmp344
	.quad	.Ltmp345
	.quad	.Ltmp347
	.quad	.Ltmp351
	.quad	.Ltmp353
	.quad	.Ltmp355
	.quad	.Ltmp356
	.quad	.Ltmp357
	.quad	.Ltmp358
	.quad	.Ltmp360
	.quad	.Ltmp362
	.quad	.Ltmp453
	.quad	.Ltmp454
	.quad	.Ltmp464
	.quad	.Ltmp466
	.quad	.Ltmp467
	.quad	.Ltmp469
	.quad	.Lfunc_begin6
	.quad	.Ltmp476
	.quad	.Ltmp489
	.quad	.Ltmp494
	.quad	.Ltmp495
	.quad	.Ltmp499
	.quad	.Ltmp502
	.quad	.Ltmp503
	.quad	.Ltmp509
	.quad	.Ltmp515
	.quad	.Ltmp516
	.quad	.Ltmp518
	.quad	.Ltmp524
	.quad	.Ltmp525
	.quad	.Ltmp526
	.quad	.Ltmp529
	.quad	.Ltmp530
	.quad	.Ltmp532
	.quad	.Ltmp534
	.quad	.Ltmp627
	.quad	.Ltmp628
	.quad	.Ltmp638
	.quad	.Ltmp640
	.quad	.Ltmp641
	.quad	.Ltmp643
	.quad	.Lfunc_begin7
	.quad	.Ltmp651
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_995e48e1443ce375
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
    .sgpr_count:     30
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
