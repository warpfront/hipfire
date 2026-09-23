	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	attention_fp8_e4m3_fa2_gqa_gfx1201 ; -- Begin function attention_fp8_e4m3_fa2_gqa_gfx1201
	.globl	attention_fp8_e4m3_fa2_gqa_gfx1201
	.p2align	8
	.type	attention_fp8_e4m3_fa2_gqa_gfx1201,@function
attention_fp8_e4m3_fa2_gqa_gfx1201:     ; @attention_fp8_e4m3_fa2_gqa_gfx1201
.Lfunc_begin0:
	.file	0 "/home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt2" "/home/kaden/ClaudeCode/warpfront/wt-fapkt/kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0xc352af1830a9071c756ca835d9a8773b
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.file	1 "/home/kaden/ClaudeCode/warpfront/wt-fapkt" "kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip" md5 0xc352af1830a9071c756ca835d9a8773b
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
	v_lshrrev_b32_e32 v156, 4, v0
	v_lshrrev_b32_e32 v157, 3, v0
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
.Ltmp130:
	.loc	1 956 25 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:956:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v159, 4, v9
	v_lshrrev_b32_e32 v13, 1, v0
	v_dual_mov_b32 v167, 1.0 :: v_dual_lshlrev_b32 v14, 3, v0
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp131:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v1, 16, v10
.Ltmp132:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v12, 8, v10
	v_lshl_add_u32 v161, v9, 3, 0
	v_cndmask_b32_e64 v9, 0, v4, s19
.Ltmp133:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshrrev_b32_e32 v162, 5, v0
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
	v_dual_cndmask_b32 v1, v10, v1 :: v_dual_and_b32 v160, 15, v0
.Ltmp138:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
.Ltmp139:
	.loc	1 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s20, s4
	s_and_b32 s6, s6, 0xffff
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v158, 2, v1
	v_and_b32_e32 v165, 8, v13
	v_and_b32_e32 v13, 0xf8, v14
	v_lshlrev_b32_e32 v163, 3, v159
.Ltmp140:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v6, v158, v7
.Ltmp141:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v11, v158, v8
.Ltmp142:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b32_e32 v166, 4, v162
	v_lshlrev_b32_e32 v133, 4, v160
.Ltmp143:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
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
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v10, v12, vcc_lo
.Ltmp145:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:999:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max_i32_e32 v4, v7, v6
.Ltmp146:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v7, 4, v10
	v_cndmask_b32_e64 v12, 0, v3, s19
	v_ashrrev_i32_e32 v3, 31, v2
.Ltmp147:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v0, 2, v5
.Ltmp148:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1004:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v8, v11
.Ltmp149:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_lshlrev_b32_e32 v164, 7, v160
.Ltmp150:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_xor_b32_e32 v11, 2, v10
.Ltmp151:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v6, v0, v4
.Ltmp152:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v0, v0, v5
.Ltmp153:
	.loc	2 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v10, v7, vcc_lo
	v_lshlrev_b64_e32 v[129:130], 2, v[2:3]
	v_mov_b32_e32 v8, v1
.Ltmp154:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v7, 2, v7
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v146, v1
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, v10, v11, vcc_lo
.Ltmp155:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_lshlrev_b64_e32 v[131:132], 2, v[145:146]
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v134, v10, v15, vcc_lo
.Ltmp156:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1000:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v41, v4, v6
.Ltmp157:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1005:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v5, v0
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v4, v1
	v_mov_b32_e32 v6, v1
.Ltmp158:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v42, v7, v41
.Ltmp159:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v43, v7, v0
	v_mov_b32_e32 v7, v1
.Ltmp160:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v65, 2, v44
.Ltmp161:
	.loc	1 1014 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v136, vcc_lo, v12, v163
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v9, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v146, vcc_lo, s0, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, s1, v130, vcc_lo
.Ltmp162:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_lshlrev_b32_e32 v129, 2, v134
.Ltmp163:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_xor_b32 s0, s20, 0x80000000
.Ltmp164:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_or_b32 v135, v162, 8, v13
.Ltmp165:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s6, s0, s4
	s_cvt_u32_f32 s0, s20
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s6, 31
.Ltmp166:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1001:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v138, v41, v42
.Ltmp167:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1006:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v43
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v41, v1
	v_mov_b32_e32 v56, v8
.Ltmp168:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v139, v65, v138
.Ltmp169:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v140, v65, v0
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v72, v8
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v88, v8
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v96, v8
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v104, v8
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v112, v8
.Ltmp170:
	.loc	1 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s6, s4
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v120, v8
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v128, v8
.Ltmp171:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1002:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v138, v139
.Ltmp172:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1007:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v140
	v_dual_mov_b32 v121, v1 :: v_dual_add_nc_u32 v168, 0, v135
.Ltmp173:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v135, vcc_lo, s8, v136
.Ltmp174:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v134, v129, v130
.Ltmp175:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	ds_bpermute_b32 v129, v129, v0
.Ltmp176:
	.loc	1 1582 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_add_co_ci_u32 s6, s0, 0
	s_lshl_b32 s4, s3, 8
.Ltmp177:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, s9, v137, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[4:5]
	s_add_nc_u64 s[20:21], s[12:13], s[4:5]
	s_mul_i32 s4, s7, 0x1800
	v_add_co_u32 v148, vcc_lo, v135, 48
.Ltmp178:
	.loc	1 1582 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1582:42
	s_and_b32 s22, s6, 0xffff
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[8:9], s[4:5]
.Ltmp179:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, 0, v136, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v152, vcc_lo, s6, v131
	v_add_co_u32 v150, s0, s0, v133
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
.Ltmp180:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1003:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v130, v134
.Ltmp181:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1008:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v129
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s7, v132, vcc_lo
.Ltmp182:
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
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v134, v6
	v_dual_mov_b32 v133, v5 :: v_dual_mov_b32 v132, v4
	v_dual_mov_b32 v131, v3 :: v_dual_mov_b32 v130, v2
	v_mov_b32_e32 v129, v1
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mul_i32 s24, s18, s22
	s_lshl_b32 s4, s3, 1
.Ltmp183:
	.loc	1 1584 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1584:23
	s_add_co_i32 s25, s24, s22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
.Ltmp184:
	.loc	1 1014 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1014:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_lshl_b32 s11, s24, 6
	s_branch .LBB3_14
.LBB3_11:                               ;   in Loop: Header=BB3_14 Depth=1
	.loc	1 0 5 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:5
	v_mov_b32_e32 v154, v2
.LBB3_12:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s30
.Ltmp185:
	.loc	4 701 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1263:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp186:
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
	v_mov_b32_e32 v0, v156
	v_mov_b32_e32 v6, v157
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
	v_add_nc_u32_e32 v7, v7, v164
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
	v_or_b32_e32 v6, s29, v165
	v_dual_mov_b32 v7, v166 :: v_dual_mov_b32 v8, v162
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
	v_and_or_b32 v0, 0xf0, v7, v160
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
.Ltmp187:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_wait_dscnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1090:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp188:
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
	v_mov_b32_e32 v4, v161
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
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v172, null, v3, v3, v140
	.loc	1 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v174, null, v3, v3, v139
	.loc	1 1245 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v176, null, v3, v3, v8
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v177, null, v3, v3, v137
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v173, v172
	.loc	1 1251 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1251:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshl_add_u32 v189, s31, 12, v161
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v179, v176
	v_exp_f32_e32 v143, v143
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v180, v177
	.loc	1 1209 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v175, -v172, v173, 1.0
	.loc	1 1189 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1189:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v173, v175, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1226 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v144, v167, v143
	.loc	1 1226 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1226:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1240 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, null, v3, v3, v141
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v134, v134, v6 :: v_dual_fmac_f32 v5, v155, v143
	v_mul_f32_e32 v126, v126, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v143, null, v3, v3, v142
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v167, v154
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v124, v124, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v144, v143
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v122, v122, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v154, v167, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v130, v130, v6 :: v_dual_mul_f32 v129, v129, v6
	v_mul_f32_e32 v118, v118, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v143, v144, 1.0
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v167, v170, v167
	v_div_scale_f32 v170, s0, v141, v3, v141
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v128, v128, v6 :: v_dual_mul_f32 v127, v127, v6
	v_mul_f32_e32 v116, v116, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v144, v155, v144
	v_div_scale_f32 v155, vcc_lo, v142, v3, v142
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v125, v125, v6 :: v_dual_mul_f32 v114, v114, v6
	v_dual_mul_f32 v123, v123, v6 :: v_dual_mul_f32 v112, v112, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v169, v155, v144
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v110, v110, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v108, v108, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v171, -v143, v169, v155
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v12, v12, v6 :: v_dual_mul_f32 v117, v117, v6
	v_dual_mul_f32 v106, v106, v6 :: v_dual_mul_f32 v115, v115, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v104, v104, v6 :: v_dual_fmac_f32 v169, v171, v144
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v170, v167
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v102, v102, v6
	v_mul_f32_e32 v10, v10, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v143, -v143, v169, v155
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v154, v171, v170
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v98, v98, v6
	.loc	1 1240 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	.loc	1 1240 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v155, v167
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v155, s1, v140, v3, v140
	.loc	1 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v144, v174
	.loc	1 1240 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1240 29 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v143, v143, v3, v142
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v142, -v154, v171, v170
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v154, v155, v173
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v170, null, v3, v3, v7
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v96, v96, v6
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v167, v142, v167, v171
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v171, -v172, v154, v155
	.loc	1 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v174, v144, 1.0
	.loc	1 1229 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v21, v21, v6
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v175, v170
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v23, v23, v6
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v154, v171, v173
	.loc	1 1242 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	1 1241 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v142.h, 0
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v155, -v172, v154, v155
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v142.l, v1.l
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v169, v144
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v178, -v170, v175, 1.0
	.loc	1 1240 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1240:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v167, v167, v3, v141
	.loc	1 1242 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v155, v173, v154
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v155, null, v3, v3, v138
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v172, -v174, v171, v169
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v175, v178, v175
	.loc	1 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v173, -v176, v179, 1.0
	.loc	1 1242 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v140, v154, v3, v140
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_rcp_f32_e32 v178, v155
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v172, v144
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v172, s1, v7, v3, v7
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v154, -v177, v180, 1.0
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v179, v173, v179
	v_div_scale_f32 v173, s3, v8, v3, v8
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v174, v171, v169
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v19, v19, v6 :: v_dual_mul_f32 v174, v172, v175
	s_delay_alu instid0(TRANS32_DEP_1)
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v181, -v155, v178, 1.0
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v17, v17, v6 :: v_dual_fmac_f32 v180, v154, v180
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v154, s4, v137, v3, v137
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s0
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v178, v181, v178
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v171, v173, v179
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_scale_f32 v181, s0, v138, v3, v138
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v170, v174, v172
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v15, v15, v6 :: v_dual_mul_f32 v182, v154, v180
	.loc	1 1242 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1242:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v139, v144, v3, v139
	.loc	1 1245 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v176, v171, v173
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v183, v181, v178
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v174, v169, v175
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v169, -v177, v182, v154
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_mov_b32 vcc_lo, s1
	.loc	1 1245 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v171, v144, v179
	.loc	1 1247 46 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v144, -v155, v183, v181
	.loc	1 1245 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1245:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fma_f32 v170, -v170, v174, v172
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_fmac_f32_e32 v182, v169, v180
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v141.h, v142.h
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mov_b16_e64 v141.l, v142.l
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
	.loc	1 1241 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1241:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v141.h, v140, v139
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
	.loc	1 1239 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1239:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v141.l, v143, v167
	.loc	1 1247 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v137, v144, v3, v137
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v94, v94, v6
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 1247 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1247:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_div_fixup_f32 v138, v155, v3, v138
	.loc	1 1244 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1244:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	.loc	1 1252 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1252:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	ds_load_b64 v[7:8], v189 offset:16384
	.loc	1 1256 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1256:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1229 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1229:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v92, v92, v6
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
	v_mov_b32_e32 v167, v3
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
	v_or_b32_e32 v2, s0, v163
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp189:
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
.Ltmp190:
.LBB3_55:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v5, 1, v2
.Ltmp191:
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
.Ltmp192:
.LBB3_57:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp193:
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
.Ltmp194:
.LBB3_59:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp195:
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
.Ltmp196:
.LBB3_61:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp197:
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
.Ltmp198:
.LBB3_63:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp199:
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
.Ltmp200:
.LBB3_65:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp201:
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
.Ltmp202:
.LBB3_67:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	.loc	1 1173 50 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1173:50 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
.Ltmp203:
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
.Ltmp204:
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
.Ltmp205:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s34, s9
	s_or_b32 s8, s34, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
.Ltmp206:
	.loc	1 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s19, s9
.Ltmp207:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v143, v170, v142, v140
.Ltmp208:
	.loc	1 1178 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1178:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
.Ltmp209:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1185:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v175, v143, v173, v144
.Ltmp210:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1186:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v176, v158, v175
.Ltmp211:
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
.Ltmp212:
.LBB3_83:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp213:
	.loc	3 454 44 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1187:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v154, v175, v176
.Ltmp214:
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
.Ltmp215:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v142, 0, v141
.Ltmp216:
	.loc	1 1203 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1203:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v170, v3
.Ltmp217:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
.Ltmp218:
	.loc	1 1207 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1207:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v137, v5, v174
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp219:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v6, v6, v7, v8
.Ltmp220:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp221:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1219:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max3_num_f32 v144, v6, v137, v138
.Ltmp222:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v3, v5, v3
.Ltmp223:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v169, v158, v144
.Ltmp224:
	.loc	1 1206 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1206:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_f32_e32 v5, v143, v3
.Ltmp225:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:552:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1209:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ] ]
	ds_bpermute_b32 v6, v158, v5
.Ltmp226:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1220:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v167
.Ltmp227:
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
.Ltmp228:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1223:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ] ]
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB3_43
.Ltmp229:
.LBB3_85:                               ;   in Loop: Header=BB3_45 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v2, v154
	s_branch .LBB3_44
.LBB3_86:                               ;   in Loop: Header=BB3_45 Depth=2
.Ltmp230:
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
.Ltmp231:
.LBB3_88:
	.loc	1 1267 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1267:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB3_93
; %bb.89:
	.loc	1 1282 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1282:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_cmp_eq_u32_e32 vcc_lo, 0, v159
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
	v_mad_co_u64_u32 v[1:2], null, v145, s17, s[18:19]
	.loc	1 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v163
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v0, v167, v129 :: v_dual_mul_f32 v105, v105, v167
	v_mul_f32_e32 v4, v121, v167
	v_mul_f32_e32 v8, v113, v167
	v_mul_f32_e32 v81, v81, v167
	.loc	1 1297 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1297:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_lo_u32 v1, 0x102, v1
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v7, v124, v167 :: v_dual_mul_f32 v66, v66, v167
	v_dual_mul_f32 v57, v57, v167 :: v_dual_mul_f32 v106, v106, v167
	v_dual_mul_f32 v113, v9, v167 :: v_dual_mul_f32 v34, v34, v167
	v_mul_f32_e32 v9, v114, v167
	.loc	1 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_lshlrev_b64_e32 v[2:3], 2, v[1:2]
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v1, v167, v130 :: v_dual_mul_f32 v26, v26, v167
	v_dual_mul_f32 v107, v107, v167 :: v_dual_mul_f32 v114, v10, v167
	v_mul_f32_e32 v91, v91, v167
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	1 1298 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1298:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v6, vcc_lo, s14, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v2, v167, v131 :: v_dual_mul_f32 v129, v167, v133
	.loc	1 1304 25 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_add_co_u32 v137, vcc_lo, v6, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v138, null, 0, v3, vcc_lo
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v3, v167, v132
	v_dual_mul_f32 v5, v122, v167 :: v_dual_mul_f32 v10, v115, v167
	v_mul_f32_e32 v6, v123, v167
	v_dual_mul_f32 v83, v83, v167 :: v_dual_mul_f32 v108, v108, v167
	v_mul_f32_e32 v75, v75, v167
	v_dual_mul_f32 v115, v11, v167 :: v_dual_mul_f32 v44, v44, v167
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b128 v[137:138], v[0:3], off offset:8
	global_store_b128 v[137:138], v[4:7], off offset:72
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b128 v[137:138], v[8:11], off offset:136
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v20, v20, v167 :: v_dual_mul_f32 v77, v77, v167
	v_dual_mul_f32 v116, v12, v167 :: v_dual_mul_f32 v69, v69, v167
	v_mul_f32_e32 v8, v101, v167
	v_mul_f32_e32 v12, v93, v167
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x1
	global_store_b128 v[137:138], v[105:108], off offset:200
	global_store_b128 v[137:138], v[4:7], off offset:216
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	global_store_b128 v[137:138], v[0:3], off offset:152
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_dual_mul_f32 v0, v21, v167 :: v_dual_mul_f32 v1, v22, v167
	v_dual_mul_f32 v2, v23, v167 :: v_dual_mul_f32 v3, v24, v167
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
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
	.loc	1 1304 64                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:64 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	v_mul_f32_e32 v7, v16, v167
	.loc	1 1304 55                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1304:55 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1585:5 ]
	s_clause 0x5
	global_store_b128 v[137:138], v[25:28], off offset:840
	global_store_b128 v[137:138], v[29:32], off offset:856
	global_store_b128 v[137:138], v[17:20], off offset:904
	global_store_b128 v[137:138], v[0:3], off offset:920
	global_store_b128 v[137:138], v[113:116], off offset:968
	global_store_b128 v[137:138], v[4:7], off offset:984
.Ltmp232:
.LBB3_93:
	.loc	1 1588 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1588:1
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
.Ltmp234:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp235:
	.loc	1 1659 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1659:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp236:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1662:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp237:
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
.Ltmp238:
	.loc	3 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	3 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1671:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB4_7
.Ltmp239:
.LBB4_10:
	.loc	1 1678 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1678:1
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
	.loc	1 2120 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2120:17
	s_load_b128 s[12:15], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	.loc	1 2120 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2120:23
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s14, 0x100
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_58
; %bb.1:
	.loc	1 2123 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2123:14
	s_cmp_gt_i32 ttmp7, 3
	s_cbranch_scc1 .LBB5_58
; %bb.2:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0
	s_load_b32 s18, s[0:1], 0x38
	.loc	1 2125 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2125:30
	s_lshl_b32 s12, ttmp9, 7
	.loc	1 2126 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2126:23
	s_mul_i32 s3, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2126 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2126:18
	s_cmp_ge_i32 s12, s3
	s_cbranch_scc1 .LBB5_58
; %bb.3:
.Ltmp241:
	.loc	1 1765 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1765:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshrrev_b32_e32 v6, 5, v0
	.loc	1 1767 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_and_b32_e32 v8, 15, v0
.Ltmp242:
	.loc	1 2120 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2120:17
	s_load_b256 s[4:11], s[0:1], 0x0
.Ltmp243:
	.loc	1 1768 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1768:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_bfe_u32 v5, v0, 4, 1
.Ltmp244:
	.loc	1 2120 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2120:17
	s_load_b64 s[16:17], s[0:1], 0x20
.Ltmp245:
	.loc	1 1776 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshlrev_b32_e32 v9, 4, v6
	.loc	1 1766 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1766:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_and_b32_e32 v10, 31, v0
	v_bfrev_b32_e32 v7, -2
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshlrev_b32_e32 v183, 3, v5
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s1, exec_lo
	.loc	1 1776 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v1, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1776 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_nc_u32_e32 v4, s12, v1
	.loc	1 1777 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1777:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_hi_i32 v1, 0x2aaaaaab, v4
	.loc	1 1779 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmp_le_i32_e64 s2, s3, v4
	v_cmp_gt_i32_e64 s0, s3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1777 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1777:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshrrev_b32_e32 v2, 31, v1
	v_add_nc_u32_e32 v32, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1778 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_lo_u32 v1, v32, 6
	v_sub_nc_u32_e32 v1, v4, v1
	v_mov_b32_e32 v4, -1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1778 31 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[2:3], null, ttmp7, 6, v[1:2]
	v_mov_b32_e32 v1, v2
	scratch_store_b64 off, v[1:2], off offset:8 ; 8-byte Folded Spill
	v_mad_co_u64_u32 v[1:2], null, v32, 24, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 8, v1
	.loc	1 1792 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, v2, 0, s2
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, s13, s4, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s5, 0, s13
	s_mov_b32 s13, 0
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_co_u32 v2, vcc_lo, v2, v183
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	.loc	1 1797 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1797:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
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
	.loc	1 1792 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b32_e32 v2, 0
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_gt_u32_e32 22, v10
	s_cbranch_execz .LBB5_7
; %bb.4:
	.loc	1 1805 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mul_hi_i32 s14, s12, 0x2aaaaaab
	v_bfrev_b32_e32 v7, -2
	s_lshr_b32 s19, s14, 31
	v_mov_b32_e32 v4, -1
	.loc	1 1805 37 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add3_u32 v3, s14, s19, v10
	.loc	1 1806 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1806:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s14, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v3
	s_cbranch_execz .LBB5_6
; %bb.5:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v4, 31, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	.loc	1 1807 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1807:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	global_load_b32 v4, v[3:4], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v7, v4
.LBB5_6:
	.loc	1 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
.LBB5_7:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	s_or_b32 exec_lo, exec_lo, s1
.Ltmp246:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_and_b32_e32 v15, 1, v0
.Ltmp247:
	.loc	1 1772 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1772:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshl_add_u32 v12, v6, 11, 0
	v_dual_mov_b32 v16, 0x6020400 :: v_dual_lshlrev_b32 v185, 3, v6
.Ltmp248:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_xor_b32_e32 v3, 16, v11
.Ltmp249:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_xor_b32_e32 v20, 8, v11
	v_and_or_b32 v21, v6, 4, v5
.Ltmp250:
	.loc	1 1827 28 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1827:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cndmask_b32_e64 v1, v1, 0, s2
	v_mov_b32_e32 v17, 0x5040100
.Ltmp251:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v3
	v_lshl_add_u32 v186, v10, 4, 0
.Ltmp252:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_xor_b32_e32 v29, 2, v11
	v_and_or_b32 v184, v9, 48, v8
	v_bfe_u32 v19, v0, 5, 1
.Ltmp253:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v3, v11, v3 :: v_dual_and_b32 v14, 3, v0
.Ltmp254:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v20
	v_lshlrev_b32_e32 v13, 3, v10
	v_and_b32_e32 v10, 16, v0
	v_or_b32_e32 v23, 0x100, v0
	v_or_b32_e32 v24, 0x200, v0
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v11, v20, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v15
.Ltmp255:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_lshlrev_b32_e32 v3, 2, v3
	v_or_b32_e32 v20, 0x300, v0
.Ltmp256:
	.loc	1 1825 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_or_b32 s19, s12, 0x7f
.Ltmp257:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v188, 0x3070105, v16, vcc_lo
.Ltmp258:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v18, v3, v4
.Ltmp259:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v3, v7
	v_cmp_gt_u32_e32 vcc_lo, 2, v14
	v_lshlrev_b32_e32 v22, 4, v0
.Ltmp260:
	.loc	1 1825 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s19, s3
	v_cmp_eq_u32_e64 s3, v5, v19
	v_lshrrev_b32_e32 v19, 5, v24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v189, 0x3020706, v17, vcc_lo
.Ltmp261:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_xor_b32_e32 v30, 1, v11
	v_lshrrev_b32_e32 v8, 2, v8
	v_ashrrev_i32_e32 v33, 31, v32
.Ltmp262:
	.loc	1 1826 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mul_i32 s12, s15, 0x1800
	v_lshlrev_b32_e32 v31, 3, v14
	.loc	1 1826 48 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[20:21], s[4:5], s[12:13]
	v_lshl_or_b32 v8, v14, 6, v8
	v_lshlrev_b32_e32 v25, 6, v0
	s_mov_b32 s14, ttmp7
	.loc	1 1825 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cselect_b32 s12, -1, 0
	s_ashr_i32 s15, ttmp7, 31
	v_xad_u32 v190, 0x120, v13, v12
.Ltmp263:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v18, v4, v18
.Ltmp264:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v26, v7, v3
	v_lshlrev_b64_e32 v[3:4], 2, v[1:2]
	v_mov_b32_e32 v1, v32
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[22:23], s[14:15], 8
.Ltmp265:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v27, v6, v18
.Ltmp266:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v28, v6, v26
.Ltmp267:
	.loc	2 523 20 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_xor_b32_e32 v6, 4, v11
	scratch_store_b64 off, v[1:2], off      ; 8-byte Folded Spill
	v_dual_mov_b32 v240, 0xff800000 :: v_dual_add_nc_u32 v1, 0, v10
	v_lshrrev_b32_e32 v10, 5, v23
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
.Ltmp268:
	.loc	1 1828 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_add_nc_u64 s[14:15], s[6:7], s[22:23]
	s_add_nc_u64 s[22:23], s[8:9], s[22:23]
	v_xad_u32 v191, 0x124, v13, v12
	v_xad_u32 v192, 0x240, v13, v12
.Ltmp269:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v11, v6, vcc_lo
.Ltmp270:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v29
	v_and_b32_e32 v9, 0x7f, v0
	v_lshlrev_b64_e32 v[6:7], 2, v[32:33]
	v_or_b32_e32 v33, 24, v8
	v_xad_u32 v193, 0x244, v13, v12
	v_xad_u32 v194, 0x360, v13, v12
	v_and_or_b32 v16, 0x180, v23, v9
	v_and_or_b32 v23, 0x280, v24, v9
	v_lshrrev_b32_e32 v24, 5, v20
	v_and_or_b32 v9, 0x380, v20, v9
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v20, v11, v29, vcc_lo
.Ltmp271:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v17, v18, v27
.Ltmp272:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v18, v26, v28
.Ltmp273:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v30
	v_xad_u32 v195, 0x364, v13, v12
.Ltmp274:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_lshlrev_b32_e32 v202, 2, v20
.Ltmp275:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_lshlrev_b32_e32 v15, 2, v15
	v_xad_u32 v196, 0x520, v13, v12
.Ltmp276:
	.loc	2 524 11 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v11, v30, vcc_lo
	v_add_co_u32 v179, vcc_lo, s20, v3
.Ltmp277:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v26, v15, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v180, null, s21, v4, vcc_lo
.Ltmp278:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_lshlrev_b32_e32 v203, 2, v11
	v_add_co_u32 v181, vcc_lo, s16, v6
.Ltmp279:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v15, v15, v18
	v_xad_u32 v197, 0x524, v13, v12
	v_xad_u32 v198, 0x640, v13, v12
	v_xad_u32 v199, 0x644, v13, v12
	v_xad_u32 v200, 0x760, v13, v12
	v_xad_u32 v201, 0x764, v13, v12
	v_add_co_u32 v177, s5, s22, v13
	v_or_b32_e32 v30, 20, v8
	v_lshlrev_b32_e32 v14, 5, v14
	v_and_or_b32 v10, v10, 12, v5
	v_or_b32_e32 v27, 12, v8
	v_or_b32_e32 v32, 0x114, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v182, null, s17, v7, vcc_lo
.Ltmp280:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v17, v17, v26
	v_or_b32_e32 v26, 0x108, v8
	s_lshl_b32 s4, ttmp7, 1
	v_cmp_gt_u32_e64 s1, 64, v0
	v_add_co_ci_u32_e64 v178, null, s23, 0, s5
.Ltmp281:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v20, v202, v17
.Ltmp282:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	v_min_i32_e32 v15, v18, v15
	v_xor_b32_e32 v4, v26, v31
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	v_lshlrev_b32_e32 v230, 3, v21
.Ltmp283:
	.loc	1 1828 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[6:7], s[4:5]
	s_add_nc_u64 s[8:9], s[8:9], s[4:5]
	v_lshl_add_u32 v144, v4, 2, v12
	v_xor_b32_e32 v4, v30, v31
	s_mov_b32 s5, 0x76543210
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v211, v4, 2, v12
	v_or_b32_e32 v4, 28, v8
	v_xor_b32_e32 v4, v4, v31
.Ltmp284:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v11, v17, v20
	v_or_b32_e32 v17, 0x10c, v8
	v_or_b32_e32 v20, 16, v8
	v_lshl_add_u32 v215, v4, 2, v12
.Ltmp285:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v28, v203, v11
	v_xor_b32_e32 v6, v17, v31
	v_xor_b32_e32 v7, v20, v31
	v_or_b32_e32 v4, 44, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v187, v6, 2, v12
	v_xor_b32_e32 v6, v33, v31
	v_lshl_add_u32 v209, v7, 2, v12
	v_or_b32_e32 v7, 0x128, v8
	v_xor_b32_e32 v4, v4, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v213, v6, 2, v12
	v_or_b32_e32 v6, 40, v8
	v_lshl_add_u32 v219, v4, 2, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v6, v6, v31
	v_lshl_add_u32 v217, v6, 2, v12
	v_or_b32_e32 v6, 0x130, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v6, v6, v31
	v_lshl_add_u32 v222, v6, 2, v12
	v_or_b32_e32 v6, 60, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v6, v6, v31
	v_lshl_add_u32 v227, v6, 2, v12
	v_mov_b32_e32 v6, v2
	v_and_b32_e32 v18, 0x3000, v25
.Ltmp286:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v25, v202, v15
	v_add_nc_u32_e32 v143, v12, v13
	v_lshlrev_b32_e32 v13, 4, v16
	v_and_or_b32 v16, v19, 20, v5
	v_lshlrev_b32_e32 v19, 4, v23
	v_and_or_b32 v23, v24, 28, v5
	v_lshlrev_b32_e32 v24, 4, v9
	v_or_b32_e32 v9, 8, v8
	v_lshlrev_b32_e32 v5, 2, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v239, 1.0 :: v_dual_lshlrev_b32 v234, 3, v23
	v_xor_b32_e32 v3, v9, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add3_u32 v204, v12, v5, v14
	v_xor_b32_e32 v5, v27, v31
	v_lshl_add_u32 v255, v3, 2, v12
.Ltmp287:
	.loc	3 1337 10 is_stmt 1             ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v3, v11, v28
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v142, v5, 2, v12
.Ltmp288:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v15, v15, v25
	v_or_b32_e32 v25, 0x110, v8
	v_xor_b32_e32 v5, v32, v31
.Ltmp289:
	.loc	1 1822 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1822:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_readfirstlane_b32 s16, v3
.Ltmp290:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v29, v203, v15
	v_xor_b32_e32 v3, v25, v31
	v_lshl_add_u32 v212, v5, 2, v12
	v_or_b32_e32 v5, 0x11c, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v210, v3, 2, v12
	v_or_b32_e32 v3, 0x118, v8
	v_xor_b32_e32 v5, v5, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v3, v3, v31
	v_lshl_add_u32 v216, v5, 2, v12
	v_or_b32_e32 v5, 48, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v214, v3, 2, v12
	v_xor_b32_e32 v3, v7, v31
	v_or_b32_e32 v7, 52, v8
	v_xor_b32_e32 v5, v5, v31
.Ltmp291:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v9, v15, v29
	v_lshl_add_u32 v218, v3, 2, v12
	v_or_b32_e32 v3, 0x12c, v8
	v_xor_b32_e32 v4, v7, v31
	v_lshl_add_u32 v221, v5, 2, v12
	v_or_b32_e32 v5, 0x138, v8
	v_or_b32_e32 v7, 0x13c, v8
	v_xor_b32_e32 v3, v3, v31
	v_lshl_add_u32 v223, v4, 2, v12
	v_or_b32_e32 v4, 56, v8
.Ltmp292:
	.loc	1 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_readfirstlane_b32 s17, v9
	v_mov_b32_e32 v9, v2
	v_lshl_add_u32 v220, v3, 2, v12
	v_or_b32_e32 v3, 0x134, v8
	v_mov_b32_e32 v8, v2
	v_xor_b32_e32 v4, v4, v31
	v_xor_b32_e32 v5, v5, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v31
	v_lshl_add_u32 v225, v4, 2, v12
	v_mov_b32_e32 v4, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v226, v5, 2, v12
	v_lshl_add_u32 v224, v3, 2, v12
	v_xor_b32_e32 v3, v7, v31
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v7, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v228, v3, 2, v12
	v_mov_b32_e32 v3, v2
	v_dual_mov_b32 v133, v9 :: v_dual_add_nc_u32 v236, v186, v18
	v_mov_b32_e32 v128, v4
	v_add_nc_u32_e32 v232, 0, v13
	v_lshlrev_b32_e32 v233, 3, v16
	v_add_nc_u32_e32 v235, 0, v24
	v_dual_mov_b32 v126, v2 :: v_dual_add_nc_u32 v237, 0, v22
	v_dual_mov_b32 v131, v7 :: v_dual_add_nc_u32 v238, 0, v19
	v_mov_b32_e32 v125, v9
	v_mov_b32_e32 v117, v9
	v_mov_b32_e32 v109, v9
	v_mov_b32_e32 v101, v9
	v_mov_b32_e32 v93, v9
	v_mov_b32_e32 v85, v9
	v_mov_b32_e32 v77, v9
	v_mov_b32_e32 v69, v9
	v_mov_b32_e32 v61, v9
	v_mov_b32_e32 v53, v9
	v_mov_b32_e32 v45, v9
	v_mov_b32_e32 v37, v9
	v_mov_b32_e32 v29, v9
	v_mov_b32_e32 v21, v9
	v_dual_mov_b32 v122, v6 :: v_dual_lshlrev_b32 v231, 3, v10
	v_mov_b32_e32 v132, v8
	v_dual_mov_b32 v130, v6 :: v_dual_mov_b32 v129, v5
	v_dual_mov_b32 v114, v6 :: v_dual_mov_b32 v127, v3
	v_mov_b32_e32 v112, v4
	v_dual_mov_b32 v124, v8 :: v_dual_mov_b32 v123, v7
	v_dual_mov_b32 v108, v8 :: v_dual_mov_b32 v121, v5
	v_mov_b32_e32 v106, v6
	v_dual_mov_b32 v120, v4 :: v_dual_mov_b32 v119, v3
	v_mov_b32_e32 v104, v4
	v_dual_mov_b32 v118, v2 :: v_dual_mov_b32 v229, 0
	v_mov_b32_e32 v102, v2
	v_dual_mov_b32 v116, v8 :: v_dual_mov_b32 v115, v7
	v_dual_mov_b32 v98, v6 :: v_dual_mov_b32 v113, v5
	v_dual_mov_b32 v96, v4 :: v_dual_mov_b32 v111, v3
	v_mov_b32_e32 v94, v2
	v_dual_mov_b32 v110, v2 :: v_dual_mov_b32 v107, v7
	v_dual_mov_b32 v90, v6 :: v_dual_mov_b32 v105, v5
	v_dual_mov_b32 v88, v4 :: v_dual_mov_b32 v103, v3
	v_mov_b32_e32 v86, v2
	v_dual_mov_b32 v100, v8 :: v_dual_mov_b32 v99, v7
	v_dual_mov_b32 v82, v6 :: v_dual_mov_b32 v97, v5
	v_dual_mov_b32 v80, v4 :: v_dual_mov_b32 v95, v3
	v_mov_b32_e32 v78, v2
	v_dual_mov_b32 v92, v8 :: v_dual_mov_b32 v91, v7
	v_dual_mov_b32 v74, v6 :: v_dual_mov_b32 v89, v5
	v_dual_mov_b32 v72, v4 :: v_dual_mov_b32 v87, v3
	v_mov_b32_e32 v70, v2
	v_dual_mov_b32 v84, v8 :: v_dual_mov_b32 v83, v7
	v_dual_mov_b32 v66, v6 :: v_dual_mov_b32 v81, v5
	v_dual_mov_b32 v64, v4 :: v_dual_mov_b32 v79, v3
	v_mov_b32_e32 v62, v2
	v_dual_mov_b32 v76, v8 :: v_dual_mov_b32 v75, v7
	v_dual_mov_b32 v58, v6 :: v_dual_mov_b32 v73, v5
	v_dual_mov_b32 v56, v4 :: v_dual_mov_b32 v71, v3
	v_mov_b32_e32 v54, v2
	v_dual_mov_b32 v68, v8 :: v_dual_mov_b32 v67, v7
	v_dual_mov_b32 v50, v6 :: v_dual_mov_b32 v65, v5
	v_dual_mov_b32 v48, v4 :: v_dual_mov_b32 v63, v3
	v_mov_b32_e32 v46, v2
	v_dual_mov_b32 v60, v8 :: v_dual_mov_b32 v59, v7
	v_dual_mov_b32 v42, v6 :: v_dual_mov_b32 v57, v5
	v_dual_mov_b32 v40, v4 :: v_dual_mov_b32 v55, v3
	v_mov_b32_e32 v38, v2
	v_dual_mov_b32 v52, v8 :: v_dual_mov_b32 v51, v7
	v_dual_mov_b32 v34, v6 :: v_dual_mov_b32 v49, v5
	v_dual_mov_b32 v32, v4 :: v_dual_mov_b32 v47, v3
	v_mov_b32_e32 v30, v2
	v_dual_mov_b32 v44, v8 :: v_dual_mov_b32 v43, v7
	v_dual_mov_b32 v26, v6 :: v_dual_mov_b32 v41, v5
	v_dual_mov_b32 v24, v4 :: v_dual_mov_b32 v39, v3
	v_mov_b32_e32 v22, v2
	v_dual_mov_b32 v36, v8 :: v_dual_mov_b32 v35, v7
	v_dual_mov_b32 v18, v6 :: v_dual_mov_b32 v33, v5
	v_dual_mov_b32 v16, v4 :: v_dual_mov_b32 v31, v3
	v_mov_b32_e32 v14, v2
	v_dual_mov_b32 v28, v8 :: v_dual_mov_b32 v27, v7
	v_mov_b32_e32 v25, v5
	v_dual_mov_b32 v23, v3 :: v_dual_mov_b32 v20, v8
	v_mov_b32_e32 v19, v7
	v_mov_b32_e32 v17, v5
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v13, v9
	v_dual_mov_b32 v12, v8 :: v_dual_mov_b32 v11, v7
	v_dual_mov_b32 v10, v6 :: v_dual_mov_b32 v9, v5
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v2
	s_branch .LBB5_10
.LBB5_8:                                ;   in Loop: Header=BB5_10 Depth=1
.Ltmp293:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v240, v3
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp294:
.LBB5_9:                                ;   in Loop: Header=BB5_10 Depth=1
	.loc	4 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s13, s13, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s13, 0x3fffffff
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s19, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_56
.LBB5_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_47 Depth 2
	.loc	1 1829 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_lshl_b32 s20, s13, 6
	.loc	1 1830 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1830:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s20, s16
	s_cselect_b32 s19, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_9
; %bb.11:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v3, s20, v184
                                        ; implicit-def: $vgpr134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[138:139], null, 0x408, v3, s[14:15]
	v_cmp_ge_i32_e32 vcc_lo, s16, v3
	.loc	1 1843 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s21, exec_lo, s4
	s_cbranch_execz .LBB5_13
; %bb.12:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1850 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_co_u32 v3, s4, v138, v230
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s4
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
	v_add_co_u32 v3, s4, v138, v231
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s4
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v237, v[134:137]
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_clause 0x1
	global_load_b64 v[134:135], v[3:4], off
	global_load_b64 v[136:137], v[3:4], off offset:16
.LBB5_13:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s21
	s_cbranch_execz .LBB5_15
; %bb.14:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v4, v2
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v134, v137
	v_dual_mov_b32 v136, v137 :: v_dual_mov_b32 v135, v137
	ds_store_b128 v237, v[2:5]
.LBB5_15:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v232, v[134:137]
                                        ; implicit-def: $vgpr134
	.loc	1 1843 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB5_17
; %bb.16:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_co_u32 v3, vcc_lo, v138, v233
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	v_add_co_u32 v136, vcc_lo, v138, v234
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v139, vcc_lo
	s_clause 0x3
	global_load_b64 v[138:139], v[3:4], off
	global_load_b64 v[140:141], v[3:4], off offset:16
	global_load_b64 v[134:135], v[136:137], off
	global_load_b64 v[136:137], v[136:137], off offset:16
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v238, v[138:141]
.LBB5_17:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB5_19
; %bb.18:                               ;   in Loop: Header=BB5_10 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v4, v2
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v134, v137
	v_dual_mov_b32 v136, v137 :: v_dual_mov_b32 v135, v137
	ds_store_b128 v238, v[2:5]
.LBB5_19:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v138, 0 :: v_dual_add_nc_u32 v5, s20, v185
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v3, 0
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v235, v[134:137]
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_21
; %bb.20:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v5, v[177:178]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_21:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v134, 0x8000, v143
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v134, v3, v4 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_gt_i32_e64 s16, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_23
; %bb.22:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1862 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v3, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[177:178]
	global_load_b64 v[138:139], v[3:4], off
.LBB5_23:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v136, 2, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v190, v138 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b32 v191, v139 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v136
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_25
; %bb.24:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v136, v[177:178]
	global_load_b64 v[134:135], v[134:135], off
.LBB5_25:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v136, 3, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v192, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b32 v193, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v136
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_27
; %bb.26:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v136, v[177:178]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_27:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v138, 4, v5
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v194, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b32 v195, v4 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_29
; %bb.28:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v138, v[177:178]
	global_load_b64 v[136:137], v[3:4], off
.LBB5_29:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v3, 5, v5
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_nc_u32_e32 v4, 0x8400, v143
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v4, v136, v137 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v3
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_31
; %bb.30:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v3, v[177:178]
	global_load_b64 v[134:135], v[3:4], off
.LBB5_31:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v138, 6, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v196, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b32 v197, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_33
; %bb.32:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v138, v[177:178]
	global_load_b64 v[136:137], v[134:135], off
.LBB5_33:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v5, 7, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v198, v136 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b32 v199, v137 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_ge_i32_e64 s16, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_execz .LBB5_35
; %bb.34:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[3:4], null, 0x408, v5, v[177:178]
	global_load_b64 v[3:4], v[3:4], off
.LBB5_35:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1871 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v200, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b32 v201, v4 offset:32768
.Ltmp295:
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp296:
	.loc	1 1888 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1888:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_37
; %bb.36:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_nc_u32_e32 v5, 0x8000, v204
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_nc_u32_e32 v138, 0x8400, v204
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_2addr_b32 v[3:4], v5 offset1:4
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_2addr_b32 v[135:136], v138 offset1:4
.Ltmp297:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v203, v3
.Ltmp298:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v134, v3, v188
.Ltmp299:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v203, v135
.Ltmp300:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v134, v135, v188
.Ltmp301:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp302:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v134, v3, v189
.Ltmp303:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v135
.Ltmp304:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v135, v189
.Ltmp305:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v203, v4
.Ltmp306:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v3, v4, v188
.Ltmp307:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v4, v203, v136
.Ltmp308:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v4, v136, v188
.Ltmp309:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v202, v3
.Ltmp310:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v136, v3, v189
.Ltmp311:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp312:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:16384
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v255 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v144 offset:32768
.Ltmp313:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v203, v3
.Ltmp314:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v134, v3, v188
.Ltmp315:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v203, v4
.Ltmp316:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v134, v4, v188
.Ltmp317:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp318:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v134, v3, v189
.Ltmp319:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp320:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v4, v189
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v142 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v187 offset:32768
.Ltmp321:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v203, v3
.Ltmp322:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v136, v3, v188
.Ltmp323:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v203, v4
.Ltmp324:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v136, v4, v188
.Ltmp325:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v202, v3
.Ltmp326:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v136, v3, v189
.Ltmp327:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp328:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:16896
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v209 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v210 offset:32768
.Ltmp329:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v203, v3
.Ltmp330:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v134, v3, v188
.Ltmp331:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v203, v4
.Ltmp332:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v134, v4, v188
.Ltmp333:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp334:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v134, v3, v189
.Ltmp335:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp336:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v4, v189
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v211 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v212 offset:32768
.Ltmp337:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v203, v3
.Ltmp338:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v136, v3, v188
.Ltmp339:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v203, v4
.Ltmp340:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v136, v4, v188
.Ltmp341:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v202, v3
.Ltmp342:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v136, v3, v189
.Ltmp343:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp344:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:17408
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v213 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v214 offset:32768
.Ltmp345:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v134, v203, v3
.Ltmp346:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v134, v3, v188
.Ltmp347:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v203, v4
.Ltmp348:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v134, v4, v188
.Ltmp349:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp350:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v134, v3, v189
.Ltmp351:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp352:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v4, v189
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v215 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v216 offset:32768
.Ltmp353:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v203, v3
.Ltmp354:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v136, v3, v188
.Ltmp355:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v203, v4
.Ltmp356:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v136, v4, v188
.Ltmp357:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v136, v202, v3
.Ltmp358:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v136, v3, v189
.Ltmp359:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp360:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:17920
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_2addr_b32 v[3:4], v5 offset0:32 offset1:36
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_2addr_b32 v[135:136], v138 offset0:32 offset1:36
.Ltmp361:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp362:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp363:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v135
.Ltmp364:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp365:
	.loc	1 1901 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v5, v5, v135, v188
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v134, v3, v189
.Ltmp366:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v5
.Ltmp367:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v5, v189
.Ltmp368:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v203, v4
.Ltmp369:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v3, v4, v188
.Ltmp370:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v4, v203, v136
.Ltmp371:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp372:
	.loc	1 1901 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v4, v136, v188
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v5, v3, v189
.Ltmp373:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp374:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:18432
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v217 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v218 offset:32768
.Ltmp375:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp376:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp377:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v4
.Ltmp378:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v5, v4, v188
.Ltmp379:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp380:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v5, v3, v189
.Ltmp381:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp382:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v4, v189
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v219 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v220 offset:32768
.Ltmp383:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp384:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp385:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v4
.Ltmp386:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v5, v4, v188
.Ltmp387:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp388:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v5, v3, v189
.Ltmp389:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp390:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:18944
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v221 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v222 offset:32768
.Ltmp391:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp392:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp393:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v4
.Ltmp394:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v5, v4, v188
.Ltmp395:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp396:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v5, v3, v189
.Ltmp397:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp398:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v4, v189
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v223 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v224 offset:32768
.Ltmp399:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp400:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp401:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v4
.Ltmp402:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v5, v4, v188
.Ltmp403:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp404:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v5, v3, v189
.Ltmp405:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp406:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:19456
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v225 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v226 offset:32768
.Ltmp407:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp408:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp409:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v4
.Ltmp410:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v5, v4, v188
.Ltmp411:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp412:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v134, v5, v3, v189
.Ltmp413:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp414:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v135, v3, v4, v189
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v3, v227 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b32 v4, v228 offset:32768
.Ltmp415:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v203, v3
.Ltmp416:
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v3, v5, v3, v188
.Ltmp417:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v203, v4
.Ltmp418:
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v4, v5, v4, v188
.Ltmp419:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v5, v202, v3
.Ltmp420:
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v136, v5, v3, v189
.Ltmp421:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	ds_bpermute_b32 v3, v202, v4
.Ltmp422:
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v3, v4, v189
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b128 v236, v[134:137] offset:19968
.LBB5_37:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1915 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1915:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB5_41
; %bb.38:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1916 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v4, s20, v0
	v_mov_b32_e32 v3, 0
	.loc	1 1918 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s21, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s16, v4
	s_cbranch_execz .LBB5_40
; %bb.39:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1923 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[134:135], null, 0x408, v4, s[6:7]
	.loc	1 1925 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mad_co_u64_u32 v[4:5], null, 0x408, v4, s[8:9]
	.loc	1 1923 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	global_load_d16_b16 v3, v[134:135], off offset:1024
	.loc	1 1925 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	global_load_d16_hi_b16 v3, v[4:5], off offset:1024
	.loc	1 1923 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v4.h, 8, v3.l
	.loc	1 1925 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshrrev_b16 v4.l, 8, v3.h
	.loc	1 1925 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_and_b16 v5.h, 0xff, v3.l
	v_and_b16 v5.l, 0xff, v3.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1926 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_pk_lshlrev_b16 v3, 8, v4 op_sel_hi:[0,1]
	.loc	1 1925 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v3, v3, v5
.LBB5_40:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s21
	v_lshl_add_u32 v4, v0, 1, 0
	.loc	1 1928 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1928:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b16_d16_hi v4, v3 offset:49152
	.loc	1 1929 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1929:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_store_b16 v4, v3 offset:49280
.LBB5_41:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.Ltmp423:
	.loc	4 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1931:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp424:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b32_e32 v241, s18
.Ltmp425:
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1931:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	s_barrier_wait -1
	.loc	4 703 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1931:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp426:
	.loc	1 1941 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB5_43
; %bb.42:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1942 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	global_load_b32 v3, v[179:180], off
	.loc	1 1943 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v241, s18, v3
.LBB5_43:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1834 35 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1834:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_or_b32 s4, s20, 63
	v_mov_b32_e32 v242, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s17
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s21, s12, s4
	s_wait_alu depctr_sa_sdst(0)
	s_nor_b32 s22, s2, s21
	.loc	1 1945 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s22
	s_cbranch_execz .LBB5_45
; %bb.44:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 1945 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	global_load_b32 v242, v[181:182], off
.LBB5_45:                               ;   in Loop: Header=BB5_10 Depth=1
	.loc	1 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	.loc	1 1935 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1935:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_or_b32 s4, s20, 16
	s_mov_b32 s22, 0
	.loc	1 1935 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1935:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s4, s16
	s_cselect_b32 s23, -1, 0
	s_branch .LBB5_47
.LBB5_46:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_sub_f32_e32 v135, v240, v3
	.loc	1 1998 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v240
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2012 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_add_f32 v4, v4, v5 :: v_dual_mul_f32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v135, v135
	.loc	1 1998 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v135, 0, v135, vcc_lo
	.loc	1 2022 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_f32_e32 v136, v239, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2022 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v137, null, v134, v134, v136
	v_rcp_f32_e32 v205, v137
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v206, -v137, v205, 1.0
	v_fmac_f32_e32 v205, v206, v205
	v_div_scale_f32 v206, vcc_lo, v136, v134, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v207, v206, v205
	v_fma_f32 v208, -v137, v207, v206
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v207, v208, v205
	v_fma_f32 v137, -v137, v207, v206
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v137, v137, v205, v207
	.loc	1 2028 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v205, null, v134, v134, v245
	v_rcp_f32_e32 v206, v205
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v205, v206, 1.0
	v_fmac_f32_e32 v206, v207, v206
	v_div_scale_f32 v207, vcc_lo, v245, v134, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v208, v207, v206
	v_fma_f32 v239, -v205, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v208, v239, v206
	.loc	1 2013 39                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fmac_f32_e32 v4, v229, v135
	.loc	1 2028 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v205, -v205, v208, v207
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v229, v4
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v205, v205, v206, v208
	.loc	1 2028 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v206, null, v134, v134, v246
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2028 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v245, v205, v134, v245
	.loc	1 2028 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v207, v206
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v206, v207, 1.0
	v_fmac_f32_e32 v207, v208, v207
	v_div_scale_f32 v208, vcc_lo, v246, v134, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v239, v208, v207
	v_fma_f32 v240, -v206, v239, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v239, v240, v207
	v_fma_f32 v206, -v206, v239, v208
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v206, v206, v207, v239
	.loc	1 2030 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v207, null, v134, v134, v243
	.loc	1 2028 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v246, v206, v134, v246
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v208, v207
	.loc	1 2027 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b16_e64 v206.l, v2.l
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b16_e64 v206.h, 0
	.loc	1 2027 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b16_e64 v205.l, v206.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b16_e64 v205.h, v206.h
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v239, -v207, v208, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	.loc	1 2027 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cvt_pk_fp8_f32 v205.l, v245, v246
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v245, null, v134, v134, v140
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fmac_f32_e32 v208, v239, v208
	v_div_scale_f32 v239, vcc_lo, v243, v134, v243
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v246, v245
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_f32_e32 v240, v239, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v247, -v207, v240, v239
	v_fmac_f32_e32 v240, v247, v208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v207, v240, v239
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v207, v207, v208, v240
	.loc	1 2030 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v208, null, v134, v134, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v207, v207, v134, v243
	.loc	1 2030 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v239, v208
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v240, -v208, v239, 1.0
	v_fmac_f32_e32 v239, v240, v239
	v_div_scale_f32 v240, vcc_lo, v244, v134, v244
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v247, v240, v239
	v_fma_f32 v248, -v208, v247, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v247, v248, v239
	v_fma_f32 v208, -v208, v247, v240
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v208, v208, v239, v247
	.loc	1 2033 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v239, null, v134, v134, v138
	.loc	1 2030 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v208, v208, v134, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2033 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v240, v239
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cvt_pk_fp8_f32 v205.h, v207, v208
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2033 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v247, -v239, v240, 1.0
	v_fmac_f32_e32 v240, v247, v240
	v_div_scale_f32 v247, vcc_lo, v138, v134, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v248, v247, v240
	v_fma_f32 v249, -v239, v248, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v248, v249, v240
	v_fma_f32 v239, -v239, v248, v247
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v239, v239, v240, v248
	.loc	1 2033 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v240, null, v134, v134, v139
	.loc	1 2033 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v138, v239, v134, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	.loc	1 2033 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v247, v240
	v_mov_b32_e32 v239, v134
	v_fma_f32 v248, -v240, v247, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v247, v248, v247
	v_div_scale_f32 v248, vcc_lo, v139, v134, v139
	v_mul_f32_e32 v249, v248, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v250, -v240, v249, v248
	v_fmac_f32_e32 v249, v250, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v240, -v240, v249, v248
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v240, v240, v247, v249
	.loc	1 2035 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v247, -v245, v246, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2033 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v139, v240, v134, v139
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fmac_f32_e32 v246, v247, v246
	v_div_scale_f32 v247, vcc_lo, v140, v134, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2032 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2032:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cvt_pk_fp8_f32 v206.l, v138, v139
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_f32_e32 v248, v247, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v249, -v245, v248, v247
	v_fmac_f32_e32 v248, v249, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v245, -v245, v248, v247
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v245, v245, v246, v248
	.loc	1 2035 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v246, null, v134, v134, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v138, v245, v134, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_rcp_f32_e32 v247, v246
	.loc	1 2022 52 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v140, v137, v134, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_f32_e32 v122, v122, v140
	v_dual_mul_f32 v118, v118, v140 :: v_dual_mul_f32 v133, v133, v140
	v_mul_f32_e32 v128, v128, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v248, -v246, v247, 1.0
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v131, v131, v140 :: v_dual_mul_f32 v124, v124, v140
	v_mul_f32_e32 v132, v132, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v130, v130, v140 :: v_dual_fmac_f32 v247, v248, v247
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v248, vcc_lo, v141, v134, v141
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v129, v129, v140 :: v_dual_mul_f32 v120, v120, v140
	v_dual_mul_f32 v127, v127, v140 :: v_dual_mul_f32 v116, v116, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v249, v248, v247 :: v_dual_mul_f32 v126, v126, v140
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v125, v125, v140 :: v_dual_mul_f32 v114, v114, v140
	v_dual_mul_f32 v123, v123, v140 :: v_dual_mul_f32 v112, v112, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v250, -v246, v249, v248
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v121, v121, v140 :: v_dual_mul_f32 v110, v110, v140
	v_dual_mul_f32 v119, v119, v140 :: v_dual_mul_f32 v108, v108, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fmac_f32_e32 v249, v250, v247
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v117, v117, v140 :: v_dual_mul_f32 v106, v106, v140
	v_dual_mul_f32 v115, v115, v140 :: v_dual_mul_f32 v104, v104, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v246, -v246, v249, v248
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v113, v113, v140 :: v_dual_mul_f32 v102, v102, v140
	v_dual_mul_f32 v111, v111, v140 :: v_dual_mul_f32 v100, v100, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v246, v246, v247, v249
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v109, v109, v140 :: v_dual_mul_f32 v98, v98, v140
	v_dual_mul_f32 v107, v107, v140 :: v_dual_mul_f32 v96, v96, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v139, v246, v134, v141
	.loc	1 2040 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2040:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshl_add_u32 v141, s22, 12, v186
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v105, v105, v140 :: v_dual_mul_f32 v94, v94, v140
	v_dual_mul_f32 v103, v103, v140 :: v_dual_mul_f32 v92, v92, v140
	.loc	1 2034 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cvt_pk_fp8_f32 v206.h, v138, v139
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:16384
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v101, v101, v140 :: v_dual_mul_f32 v90, v90, v140
	v_dual_mul_f32 v99, v99, v140 :: v_dual_mul_f32 v88, v88, v140
	v_dual_mul_f32 v97, v97, v140 :: v_dual_mul_f32 v86, v86, v140
	v_dual_mul_f32 v95, v95, v140 :: v_dual_mul_f32 v84, v84, v140
	v_dual_mul_f32 v93, v93, v140 :: v_dual_mul_f32 v82, v82, v140
	v_dual_mul_f32 v91, v91, v140 :: v_dual_mul_f32 v80, v80, v140
	v_dual_mul_f32 v89, v89, v140 :: v_dual_mul_f32 v78, v78, v140
	v_dual_mul_f32 v87, v87, v140 :: v_dual_mul_f32 v76, v76, v140
	v_dual_mul_f32 v85, v85, v140 :: v_dual_mul_f32 v74, v74, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[126:133], v[136:137], v[205:206], v[126:133]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[138:139], v[205:206], v[118:125]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:16896
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v83, v83, v140 :: v_dual_mul_f32 v72, v72, v140
	v_dual_mul_f32 v81, v81, v140 :: v_dual_mul_f32 v70, v70, v140
	v_dual_mul_f32 v79, v79, v140 :: v_dual_mul_f32 v68, v68, v140
	v_dual_mul_f32 v77, v77, v140 :: v_dual_mul_f32 v66, v66, v140
	v_dual_mul_f32 v75, v75, v140 :: v_dual_mul_f32 v64, v64, v140
	v_dual_mul_f32 v73, v73, v140 :: v_dual_mul_f32 v62, v62, v140
	v_dual_mul_f32 v71, v71, v140 :: v_dual_mul_f32 v60, v60, v140
	v_dual_mul_f32 v69, v69, v140 :: v_dual_mul_f32 v58, v58, v140
	v_dual_mul_f32 v67, v67, v140 :: v_dual_mul_f32 v56, v56, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[110:117], v[136:137], v[205:206], v[110:117]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[102:109], v[138:139], v[205:206], v[102:109]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:17408
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v65, v65, v140 :: v_dual_mul_f32 v54, v54, v140
	v_dual_mul_f32 v63, v63, v140 :: v_dual_mul_f32 v52, v52, v140
	v_dual_mul_f32 v61, v61, v140 :: v_dual_mul_f32 v50, v50, v140
	v_dual_mul_f32 v59, v59, v140 :: v_dual_mul_f32 v48, v48, v140
	v_dual_mul_f32 v57, v57, v140 :: v_dual_mul_f32 v46, v46, v140
	v_dual_mul_f32 v55, v55, v140 :: v_dual_mul_f32 v44, v44, v140
	v_dual_mul_f32 v53, v53, v140 :: v_dual_mul_f32 v42, v42, v140
	v_dual_mul_f32 v51, v51, v140 :: v_dual_mul_f32 v40, v40, v140
	v_dual_mul_f32 v49, v49, v140 :: v_dual_mul_f32 v38, v38, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[94:101], v[136:137], v[205:206], v[94:101]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[86:93], v[138:139], v[205:206], v[86:93]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:17920
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v47, v47, v140 :: v_dual_mul_f32 v36, v36, v140
	v_dual_mul_f32 v45, v45, v140 :: v_dual_mul_f32 v34, v34, v140
	v_dual_mul_f32 v43, v43, v140 :: v_dual_mul_f32 v32, v32, v140
	v_dual_mul_f32 v41, v41, v140 :: v_dual_mul_f32 v30, v30, v140
	v_dual_mul_f32 v39, v39, v140 :: v_dual_mul_f32 v28, v28, v140
	v_dual_mul_f32 v37, v37, v140 :: v_dual_mul_f32 v26, v26, v140
	v_dual_mul_f32 v35, v35, v140 :: v_dual_mul_f32 v24, v24, v140
	v_dual_mul_f32 v33, v33, v140 :: v_dual_mul_f32 v22, v22, v140
	v_dual_mul_f32 v31, v31, v140 :: v_dual_mul_f32 v20, v20, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[78:85], v[136:137], v[205:206], v[78:85]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[70:77], v[138:139], v[205:206], v[70:77]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:18432
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v29, v29, v140 :: v_dual_mul_f32 v18, v18, v140
	v_dual_mul_f32 v27, v27, v140 :: v_dual_mul_f32 v16, v16, v140
	v_dual_mul_f32 v25, v25, v140 :: v_dual_mul_f32 v14, v14, v140
	v_dual_mul_f32 v23, v23, v140 :: v_dual_mul_f32 v12, v12, v140
	v_mul_f32_e32 v15, v15, v140
	v_dual_mul_f32 v21, v21, v140 :: v_dual_mul_f32 v10, v10, v140
	v_dual_mul_f32 v19, v19, v140 :: v_dual_mul_f32 v8, v8, v140
	v_dual_mul_f32 v17, v17, v140 :: v_dual_mul_f32 v6, v6, v140
	v_mul_f32_e32 v13, v13, v140
	v_mul_f32_e32 v11, v11, v140
	v_mul_f32_e32 v9, v9, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[62:69], v[136:137], v[205:206], v[62:69]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[54:61], v[138:139], v[205:206], v[54:61]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:18944
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_f32_e32 v7, v7, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[136:137], v[205:206], v[46:53]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[38:45], v[138:139], v[205:206], v[38:45]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:19456
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[30:37], v[136:137], v[205:206], v[30:37]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[22:29], v[138:139], v[205:206], v[22:29]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[136:139], v141 offset:19968
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[14:21], v[136:137], v[205:206], v[14:21]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[6:13], v[138:139], v[205:206], v[6:13]
	v_mov_b32_e32 v240, v3
	.loc	1 1951 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_add_co_i32 s22, s22, 1
	.loc	1 1951 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s22, 4
	.loc	1 1951 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_scc1 .LBB5_8
.LBB5_47:                               ;   Parent Loop BB5_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s22, 1
	s_mov_b32 s4, -1
	s_cbranch_scc1 .LBB5_50
; %bb.48:                               ;   in Loop: Header=BB5_47 Depth=2
	s_cmp_eq_u32 s22, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s4, s23
	s_cbranch_scc1 .LBB5_50
; %bb.49:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 1953 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cmp_eq_u32 s22, 2
	.loc	1 1953 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cselect_b32 s4, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s16, s4
	s_cselect_b32 s4, -1, 0
.LBB5_50:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_55
; %bb.51:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 1964 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshl_add_u32 v3, s22, 9, v186
	.loc	1 1976 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_lshl_b32 s4, s22, 4
	.loc	1 1976 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s4, s20
	.loc	1 1977 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s24, s4, 15
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[243:246], v3
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[247:250], v3 offset:2048
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[251:254], v3 offset:4096
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1977 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s24, s17
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[243:244], v[145:146], 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[245:246], v[147:148], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[243:246], v3 offset:6144
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[247:248], v[149:150], v[134:141]
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[249:250], v[151:152], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[247:250], v3 offset:8192
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[251:252], v[153:154], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[253:254], v[155:156], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[251:254], v3 offset:10240
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[205:208], v3 offset:12288
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[243:244], v[157:158], v[134:141]
	v_lshl_add_u32 v243, s22, 5, v1
	v_mov_b32_e32 v244, 0xff800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[245:246], v[159:160], v[134:141]
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[247:248], v[161:162], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[249:250], v[163:164], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b128 v[246:249], v3 offset:14336
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1981 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_b96 v[3:5], v243 offset:49154
	ds_load_u16_d16 v245, v243 offset:49166
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[251:252], v[165:166], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[253:254], v[167:168], v[134:141]
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[205:206], v[169:170], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[207:208], v[171:172], v[134:141]
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[246:247], v[173:174], v[134:141]
	v_or_b32_e32 v246, s4, v183
	.loc	1 1977 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s21, s4
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[248:249], v[175:176], v[134:141]
	.loc	1 1986 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s24, s0
	s_cbranch_execz .LBB5_53
; %bb.52:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 1981 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_u16_d16 v205, v243 offset:49152
	v_mul_f32_e32 v134, v241, v134
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e32 vcc_lo, v246, v242
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 vcc_lo, s4, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v134, v205, v134, neg(0) op_sel_hi:[1,0,0]
	.loc	1 1988 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v244, 0xff800000, v134, vcc_lo
.LBB5_53:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s24
	v_mul_f32_e32 v134, v241, v135
	v_or_b32_e32 v135, 2, v246
	s_wait_loadcnt 0x0
	v_cmp_ge_i32_e32 vcc_lo, v246, v242
	s_xor_b32 s24, s4, -1
	v_mul_f32_e32 v136, v241, v136
	v_cmp_gt_i32_e64 s4, v135, v242
	v_or_b32_e32 v135, 3, v246
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s25, s24, vcc_lo
	.loc	1 1986 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s25, s2, s25
	s_and_b32 s4, s24, s4
	v_cmp_gt_i32_e32 vcc_lo, v135, v242
	v_mul_f32_e32 v135, v241, v137
	s_wait_dscnt 0x1
	v_fma_mix_f32 v134, v3, v134, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v3, v136, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_or_b32_e32 v136, 4, v246
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	v_cndmask_b32_e64 v134, v134, 0xff800000, s25
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v135, v241, v138
	s_and_b32 s4, s24, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136, v242
	v_or_b32_e32 v136, 5, v246
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v241, v139
	s_and_b32 s4, s24, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136, v242
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	v_or_b32_e32 v136, 7, v246
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, v3, 0xff800000, s4
	v_fma_mix_f32 v3, v5, v4, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v4, 6, v246
	s_and_b32 s4, s24, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, v3, 0xff800000, s4
	v_cmp_gt_i32_e32 vcc_lo, v4, v242
	v_mul_f32_e32 v3, v241, v140
	v_cmp_gt_i32_e64 s4, v136, v242
	v_mul_f32_e32 v4, v241, v141
.Ltmp427:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v136, v244, 0xff800000, v134
	s_and_b32 s25, s24, vcc_lo
	v_fma_mix_f32 v3, v5, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_and_b32 s4, s24, s4
	s_wait_dscnt 0x0
	v_fma_mix_f32 v4, v245, v4, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v136, v137, v138
.Ltmp428:
	.loc	1 1986 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s24, s2, s25
	s_or_b32 s4, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v136, v3, 0xff800000, s24
	v_cndmask_b32_e64 v4, v4, 0xff800000, s4
.Ltmp429:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v3, v5, v135, v139
	v_add_nc_u32_e32 v140, 0xc080, v243
.Ltmp430:
	.loc	1 2020 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2020:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp431:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v3, v3, v136, v4
.Ltmp432:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1995:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_mov_b32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v5, v5, s5, 0xfedcba98
.Ltmp433:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1996:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v3, v240, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v5, v244, v3 :: v_dual_sub_f32 v134, v134, v3
	v_sub_f32_e32 v4, v4, v3
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v3
	v_dual_sub_f32 v136, v136, v3 :: v_dual_mul_f32 v5, 0x3fb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, 0x3fb8aa3b, v4
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v141, v4
	v_dual_sub_f32 v4, v137, v3 :: v_dual_sub_f32 v137, v138, v3
	v_exp_f32_e32 v138, v5
	v_exp_f32_e32 v136, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v134, 0x3fb8aa3b, v134
	v_exp_f32_e32 v137, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v205, v134
	v_dual_sub_f32 v134, v139, v3 :: v_dual_mul_f32 v139, 0x3fb8aa3b, v4
.Ltmp434:
	.loc	1 2002 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2002:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_2addr_b32 v[4:5], v140 offset1:1
	v_sub_f32_e32 v135, v135, v3
	v_add_nc_u32_e32 v140, 0xc088, v243
	v_mul_f32_e32 v134, 0x3fb8aa3b, v134
	v_exp_f32_e32 v139, v139
	.loc	1 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v206, 0x3fb8aa3b, v135
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	v_exp_f32_e32 v207, v134
	.loc	1 2002 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2002:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	ds_load_2addr_b32 v[134:135], v140 offset1:1
	.loc	1 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cndmask_b32_e64 v136, v136, 0, vcc_lo
	v_exp_f32_e32 v140, v206
	v_cndmask_b32_e64 v206, v141, 0, vcc_lo
	v_cndmask_b32_e64 v141, v205, 0, vcc_lo
	v_cndmask_b32_e64 v139, v139, 0, vcc_lo
	v_cndmask_b32_e64 v205, v207, 0, vcc_lo
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v245, v4, v138, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v246, v4, v141, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_f32_e32 v138, v138, v141
	.loc	1 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cndmask_b32_e64 v4, v140, 0, vcc_lo
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_mix_f32 v243, v5, v139, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v244, v5, v137, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp435:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v5, v245, 0, v246
.Ltmp436:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_f32_e32 v140, v139, v138
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v138, v134, v4, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v139, v134, v205, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp437:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v5, v5, v243, v244
.Ltmp438:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_f32_e32 v134, v137, v140
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_mix_f32 v140, v135, v136, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v135, v206, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp439:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v5, v5, v138, v139
.Ltmp440:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_f32_e32 v4, v4, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp441:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max3_num_f32 v134, v5, v140, v141
.Ltmp442:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_add_f32 v4, v205, v4 :: v_dual_mov_b32 v5, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v136, v4
.Ltmp443:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2018:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_permlanex16_b32 v5, v5, s5, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp444:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_add_f32 v4, v206, v4 :: v_dual_max_num_f32 v135, v5, v5
.Ltmp445:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_mov_b32_e32 v5, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp446:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2018:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max_num_f32_e32 v135, v134, v135
.Ltmp447:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_permlanex16_b32 v5, v5, s5, 0xfedcba98
	v_mov_b32_e32 v134, v239
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp448:
	.loc	1 2020 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2020:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmpx_lt_f32_e32 0, v135
	s_cbranch_execz .LBB5_46
; %bb.54:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	1 2021 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v134, null, 0x43e00000, 0x43e00000, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v136, v134
	v_fma_f32 v137, -v134, v136, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v136, v137, v136
	v_div_scale_f32 v137, vcc_lo, v135, 0x43e00000, v135
	v_mul_f32_e32 v205, v137, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v206, -v134, v205, v137
	v_fmac_f32_e32 v205, v206, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v134, -v134, v205, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v134, v134, v136, v205
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v134, v134, 0x43e00000, v135
.Ltmp449:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ] ]
	v_max_num_f32_e32 v134, 0x1f800000, v134
	s_branch .LBB5_46
.Ltmp450:
.LBB5_55:                               ;   in Loop: Header=BB5_47 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v3, v240
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v240, v3
	.loc	1 1951 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_add_co_i32 s22, s22, 1
	.loc	1 1951 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s22, 4
	.loc	1 1951 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_cbranch_scc0 .LBB5_47
	s_branch .LBB5_8
.LBB5_56:
	.loc	1 2063 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2063:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_58
; %bb.57:
	.loc	1 2064 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2064:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_scale_f32 v0, null, v229, v229, 1.0
	v_div_scale_f32 v3, vcc_lo, 1.0, v229, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v1, v0
	v_fma_f32 v2, -v0, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v2, v1
	v_mul_f32_e32 v2, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v0, v2, v3
	v_fmac_f32_e32 v2, v4, v1
	.loc	1 2066 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	scratch_load_b64 v[4:5], off, off th:TH_LOAD_LU ; 8-byte Folded Reload
	.loc	1 2064 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2064:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_fma_f32 v0, -v0, v2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v0, v0, v1, v2
	.loc	1 2066 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	scratch_load_b64 v[1:2], off, off offset:8 th:TH_LOAD_LU ; 8-byte Folded Reload
	.loc	1 2064 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2064:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_cmp_lt_f32_e32 vcc_lo, 0, v229
	.loc	1 2064 31 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2064:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_div_fixup_f32 v3, v0, v229, 1.0
	.loc	1 2066 33 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x1
	v_mul_lo_u32 v4, 0x1800, v4
	.loc	1 2066 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2066:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v2, v1, 8, v4
	.loc	1 2072 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mov_b32_e32 v1, 0
	.loc	1 2068 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2068:18 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_or_b32_e32 v0, v2, v183
	.loc	1 2064 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2064:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v136, v239, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 2072 56 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_mul_f32_e32 v2, v128, v136
	.loc	1 2072 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_add_co_u32 v134, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v135, null, s11, v1, vcc_lo
	.loc	1 2072 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v0, v126, v136 :: v_dual_mul_f32 v3, v129, v136
	v_dual_mul_f32 v1, v127, v136 :: v_dual_mul_f32 v126, v130, v136
	v_mul_f32_e32 v129, v133, v136
	v_dual_mul_f32 v127, v131, v136 :: v_dual_mul_f32 v128, v132, v136
	v_dual_mul_f32 v119, v119, v136 :: v_dual_mul_f32 v118, v118, v136
	v_dual_mul_f32 v121, v121, v136 :: v_dual_mul_f32 v120, v120, v136
	v_dual_mul_f32 v123, v123, v136 :: v_dual_mul_f32 v122, v122, v136
	v_dual_mul_f32 v125, v125, v136 :: v_dual_mul_f32 v124, v124, v136
	v_dual_mul_f32 v111, v111, v136 :: v_dual_mul_f32 v108, v108, v136
	v_dual_mul_f32 v95, v95, v136 :: v_dual_mul_f32 v94, v94, v136
	v_dual_mul_f32 v97, v97, v136 :: v_dual_mul_f32 v96, v96, v136
	v_dual_mul_f32 v99, v99, v136 :: v_dual_mul_f32 v110, v110, v136
	v_dual_mul_f32 v113, v113, v136 :: v_dual_mul_f32 v112, v112, v136
	v_dual_mul_f32 v115, v115, v136 :: v_dual_mul_f32 v98, v98, v136
	v_dual_mul_f32 v101, v101, v136 :: v_dual_mul_f32 v100, v100, v136
	v_dual_mul_f32 v114, v114, v136 :: v_dual_mul_f32 v117, v117, v136
	v_dual_mul_f32 v116, v116, v136 :: v_dual_mul_f32 v103, v103, v136
	v_dual_mul_f32 v102, v102, v136 :: v_dual_mul_f32 v105, v105, v136
	v_dual_mul_f32 v104, v104, v136 :: v_dual_mul_f32 v107, v107, v136
	v_dual_mul_f32 v106, v106, v136 :: v_dual_mul_f32 v109, v109, v136
	.loc	1 2072 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_clause 0x7
	global_store_b128 v[134:135], v[0:3], off
	global_store_b128 v[134:135], v[126:129], off offset:16
	global_store_b128 v[134:135], v[118:121], off offset:64
	global_store_b128 v[134:135], v[122:125], off offset:80
	global_store_b128 v[134:135], v[110:113], off offset:128
	global_store_b128 v[134:135], v[114:117], off offset:144
	global_store_b128 v[134:135], v[102:105], off offset:192
	global_store_b128 v[134:135], v[106:109], off offset:208
	.loc	1 2072 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	v_dual_mul_f32 v0, v86, v136 :: v_dual_mul_f32 v3, v89, v136
	v_dual_mul_f32 v1, v87, v136 :: v_dual_mul_f32 v2, v88, v136
	v_dual_mul_f32 v87, v91, v136 :: v_dual_mul_f32 v86, v90, v136
	v_dual_mul_f32 v89, v93, v136 :: v_dual_mul_f32 v88, v92, v136
	v_dual_mul_f32 v79, v79, v136 :: v_dual_mul_f32 v78, v78, v136
	v_dual_mul_f32 v81, v81, v136 :: v_dual_mul_f32 v80, v80, v136
	v_dual_mul_f32 v83, v83, v136 :: v_dual_mul_f32 v82, v82, v136
	v_dual_mul_f32 v85, v85, v136 :: v_dual_mul_f32 v84, v84, v136
	.loc	1 2072 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[94:97], off offset:256
	global_store_b128 v[134:135], v[98:101], off offset:272
	global_store_b128 v[134:135], v[0:3], off offset:320
	global_store_b128 v[134:135], v[86:89], off offset:336
	global_store_b128 v[134:135], v[78:81], off offset:384
	global_store_b128 v[134:135], v[82:85], off offset:400
	.loc	1 2072 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
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
	.loc	1 2072 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[0:3], off offset:448
	global_store_b128 v[134:135], v[70:73], off offset:464
	global_store_b128 v[134:135], v[62:65], off offset:512
	global_store_b128 v[134:135], v[66:69], off offset:528
	global_store_b128 v[134:135], v[54:57], off offset:576
	global_store_b128 v[134:135], v[58:61], off offset:592
	.loc	1 2072 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
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
	.loc	1 2072 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[0:3], off offset:640
	global_store_b128 v[134:135], v[46:49], off offset:656
	global_store_b128 v[134:135], v[38:41], off offset:704
	global_store_b128 v[134:135], v[42:45], off offset:720
	global_store_b128 v[134:135], v[30:33], off offset:768
	global_store_b128 v[134:135], v[34:37], off offset:784
	.loc	1 2072 56                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
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
	.loc	1 2072 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2072:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2128:5 ]
	s_clause 0x5
	global_store_b128 v[134:135], v[0:3], off offset:832
	global_store_b128 v[134:135], v[22:25], off offset:848
	global_store_b128 v[134:135], v[14:17], off offset:896
	global_store_b128 v[134:135], v[18:21], off offset:912
	global_store_b128 v[134:135], v[4:7], off offset:960
	global_store_b128 v[134:135], v[8:11], off offset:976
.Ltmp451:
.LBB5_58:
	.loc	1 2131 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2131:1
	s_endpgm
.Ltmp452:
.Lfunc_end5:
	.size	attention_fp8_e4m3_fa2_gqa_packet_gfx1201, .Lfunc_end5-attention_fp8_e4m3_fa2_gqa_packet_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 20
		.amdhsa_kernarg_size 60
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_vgpr, 256
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.numbered_sgpr, 26
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.private_seg_size, 20
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.uses_flat_scratch, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10236
; TotalNumSgprs: 28
; NumVgprs: 256
; ScratchSize: 20
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.loc	1 2146 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2146:17
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s12, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s13, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2146 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2146:23
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
	.loc	1 2151 14 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2151:14
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB6_61
; %bb.2:
	.loc	1 2153 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2153:30
	s_lshl_b32 s14, ttmp9, 7
	.loc	1 2154 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:23
	s_mul_i32 s22, s15, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2154 18 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2154:18
	s_cmp_ge_i32 s14, s22
	s_cbranch_scc1 .LBB6_61
; %bb.3:
	.loc	1 0 18                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:18
	s_lshr_b32 s12, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	.loc	1 2157 15 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2157:15
	s_cmp_ge_i32 s12, s17
	s_cbranch_scc1 .LBB6_61
; %bb.4:
.Ltmp453:
	.loc	1 1765 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1765:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshrrev_b32_e32 v6, 5, v0
	.loc	1 1767 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1767:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_and_b32_e32 v7, 15, v0
.Ltmp454:
	.loc	1 2146 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2146:17
	s_load_b256 s[4:11], s[0:1], 0x0
.Ltmp455:
	.loc	1 1778 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mul_i32 s2, s3, 6
	.loc	1 1768 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1768:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_bfe_u32 v31, v0, 4, 1
	v_dual_mov_b32 v5, -1 :: v_dual_lshlrev_b32 v8, 4, v6
.Ltmp456:
	.loc	1 2146 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2146:17
	s_load_b64 s[20:21], s[0:1], 0x20
.Ltmp457:
	.loc	1 1766 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1766:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_and_b32_e32 v10, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshlrev_b32_e32 v183, 3, v31
	.loc	1 1776 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v1, v8, v7
	v_bfrev_b32_e32 v9, -2
	s_mov_b32 s19, 0
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1776 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1776:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_nc_u32_e32 v1, s14, v1
	.loc	1 1777 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1777:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mul_hi_i32 v2, 0x2aaaaaab, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 31, v2
	v_add_nc_u32_e32 v3, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	.loc	1 1778 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1778:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mul_lo_u32 v2, v3, 6
	v_mul_lo_u32 v4, v3, 24
	v_sub_nc_u32_e32 v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v13, v2, s2, v4
	.loc	1 1779 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1779:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmp_gt_i32_e64 s2, s22, v1
	v_lshlrev_b32_e32 v2, 8, v13
	.loc	1 1792 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, 0, v2, s2
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, s13, s4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s5, 0, s13
	.loc	1 1794 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1794:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_co_u32 v1, vcc_lo, v1, v183
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	.loc	1 1797 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1797:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
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
	.loc	1 1792 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1792:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b32_e32 v2, 0
	.loc	1 1804 14                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1804:14 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_gt_u32_e32 22, v10
	s_cbranch_execz .LBB6_8
; %bb.5:
	.loc	1 1805 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mul_hi_i32 s1, s14, 0x2aaaaaab
	v_bfrev_b32_e32 v9, -2
	s_lshr_b32 s13, s1, 31
	v_mov_b32_e32 v5, -1
	.loc	1 1805 37 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1805:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v4, s1, s13, v10
	.loc	1 1806 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1806:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v4
	s_cbranch_execz .LBB6_7
; %bb.6:
	.loc	1 0 16 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:16
	v_ashrrev_i32_e32 v5, 31, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	.loc	1 1807 27 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1807:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	global_load_b32 v5, v[4:5], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v5
.LBB6_7:
	.loc	1 0 27 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:27
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
.LBB6_8:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s0
.Ltmp458:
	v_mbcnt_lo_u32_b32 v11, -1, 0
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v13, off
	scratch_store_b32 off, v31, off offset:4
	v_lshl_add_u32 v187, v10, 4, 0
.Ltmp459:
	.loc	1 1772 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1772:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshl_add_u32 v12, v6, 11, 0
.Ltmp460:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_xor_b32_e32 v1, 16, v11
	v_dual_mov_b32 v18, 0x6020400 :: v_dual_and_b32 v17, 1, v0
	v_and_or_b32 v22, v6, 4, v31
.Ltmp461:
	.loc	2 523 20 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_xor_b32_e32 v28, 2, v11
	s_delay_alu instid0(VALU_DEP_4)
.Ltmp462:
	.loc	2 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	v_lshlrev_b32_e32 v186, 3, v6
.Ltmp463:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_xor_b32_e32 v29, 1, v11
	v_and_or_b32 v185, v8, 48, v7
	v_bfe_u32 v20, v0, 5, 1
.Ltmp464:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v11, v1, vcc_lo
.Ltmp465:
	.loc	1 1827 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1827:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cndmask_b32_e64 v1, 0, v13, s2
	v_lshlrev_b32_e32 v13, 3, v10
.Ltmp466:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_xor_b32_e32 v10, 8, v11
.Ltmp467:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_add_co_i32 s1, s17, 0x1ff
	v_dual_mov_b32 v19, 0x5040100 :: v_dual_lshlrev_b32 v4, 2, v4
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
.Ltmp468:
	.loc	2 524 17                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	v_and_b32_e32 v21, 16, v0
.Ltmp469:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v15, v4, v5
.Ltmp470:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v16, v4, v9
.Ltmp471:
	.loc	1 1826 77 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mul_i32 s18, s15, 0x1800
.Ltmp472:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v11, v10, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	v_and_b32_e32 v14, 3, v0
.Ltmp473:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s15, s1
	v_cmp_eq_u32_e64 s1, v31, v20
.Ltmp474:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_lshlrev_b32_e32 v6, 2, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v189, 0x3070105, v18, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 2, v14
	v_lshlrev_b32_e32 v23, 4, v0
	v_or_b32_e32 v24, 0x100, v0
	v_or_b32_e32 v25, 0x200, v0
.Ltmp475:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_cvt_f32_u32 s13, s17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v190, 0x3020706, v19, vcc_lo
.Ltmp476:
	.loc	1 1826 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1826:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_add_nc_u64 s[4:5], s[4:5], s[18:19]
	v_lshrrev_b32_e32 v18, 5, v24
.Ltmp477:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s18, s13
	v_ashrrev_i32_e32 v4, 31, v3
.Ltmp478:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1812:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v10, v5, v15
.Ltmp479:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1817:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v9, v9, v16
.Ltmp480:
	.loc	2 523 20                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:523:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_xor_b32_e32 v5, 4, v11
	v_or_b32_e32 v15, 0x300, v0
	v_lshrrev_b32_e32 v7, 2, v7
.Ltmp481:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v26, v6, v10
.Ltmp482:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v27, v6, v9
.Ltmp483:
	.loc	2 524 17 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v5
	v_and_b32_e32 v8, 0x7f, v0
.Ltmp484:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_mul_f32 s18, s15, s18
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_lshl_or_b32 v7, v14, 6, v7
.Ltmp485:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, v11, v5, vcc_lo
.Ltmp486:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v28
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	v_add_nc_u32_e32 v1, 0, v21
	v_and_or_b32 v19, 0x180, v24, v8
.Ltmp487:
	.loc	2 527 12 is_stmt 1              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_lshlrev_b32_e32 v17, 2, v17
	v_lshrrev_b32_e32 v21, 5, v25
	v_and_or_b32 v24, 0x280, v25, v8
	v_lshrrev_b32_e32 v25, 5, v15
	v_and_or_b32 v8, 0x380, v15, v8
.Ltmp488:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v11, v28, vcc_lo
.Ltmp489:
	.loc	2 524 17 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_cmp_gt_u32_e32 vcc_lo, 32, v29
.Ltmp490:
	.loc	1 2159 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_trunc_f32 s18, s18
.Ltmp491:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1813:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	v_max_i32_e32 v10, v10, v26
.Ltmp492:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1818:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v9, v9, v27
.Ltmp493:
	.loc	2 524 11                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:524:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v11, v11, v29 :: v_dual_lshlrev_b32 v202, 2, v15
.Ltmp494:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_xor_b32 s23, s18, 0x80000000
.Ltmp495:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v20, v17, v10
.Ltmp496:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v17, v17, v9
.Ltmp497:
	.loc	1 2159 42 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s15, s23, s13
.Ltmp498:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_lshlrev_b32_e32 v205, 2, v11
.Ltmp499:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_cvt_u32_f32 s18, s18
	v_add_co_u32 v174, vcc_lo, s4, v5
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s15, 31
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v175, null, s5, v6, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s15, s13
	v_add_co_u32 v176, vcc_lo, s20, v3
	v_lshlrev_b32_e32 v30, 3, v14
	s_add_co_ci_u32 s13, s18, 0
.Ltmp500:
	.loc	1 1825 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_or_b32 s5, s14, 0x7f
.Ltmp501:
	.loc	1 2159 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2159:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s13, 0xffff
	v_or_b32_e32 v11, 12, v7
	.loc	1 2160 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2160:26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s12, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v177, null, s21, v4, vcc_lo
.Ltmp502:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1819:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x0
	v_min_i32_e32 v9, v9, v17
.Ltmp503:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1814:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max_i32_e32 v10, v10, v20
.Ltmp504:
	.loc	1 2161 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2161:23
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s14, s13, s4
.Ltmp505:
	.loc	1 1825 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cmp_lt_i32 s5, s22
	v_or_b32_e32 v26, 8, v7
.Ltmp506:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v17, v202, v9
.Ltmp507:
	.loc	1 1825 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1825:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cselect_b32 s15, -1, 0
	s_lshl_b32 s18, s3, 8
.Ltmp508:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v15, v202, v10
	s_add_nc_u64 s[20:21], s[8:9], s[18:19]
.Ltmp509:
	.loc	1 1828 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_add_nc_u64 s[4:5], s[6:7], s[18:19]
	s_lshl_b32 s18, s3, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v178, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v179, null, s21, 0, s3
	v_or_b32_e32 v27, 0x108, v7
	v_lshlrev_b32_e32 v16, 6, v0
	v_lshlrev_b32_e32 v14, 5, v14
	v_and_or_b32 v20, v21, 20, v31
	v_dual_mov_b32 v180, 0xff800000 :: v_dual_lshlrev_b32 v21, 4, v24
	v_and_or_b32 v24, v25, 28, v31
	v_add_nc_u32_e32 v191, v12, v13
	v_and_or_b32 v18, v18, 12, v31
.Ltmp510:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1820:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	v_min_i32_e32 v9, v9, v17
	v_dual_mov_b32 v240, 1.0 :: v_dual_lshlrev_b32 v19, 4, v19
.Ltmp511:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1815:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v10, v10, v15
	v_or_b32_e32 v15, 0x10c, v7
.Ltmp512:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v28, v205, v9
	v_xad_u32 v192, 0x120, v13, v12
	v_xad_u32 v193, 0x124, v13, v12
.Ltmp513:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v17, v205, v10
	v_xad_u32 v194, 0x240, v13, v12
	v_xad_u32 v195, 0x244, v13, v12
	v_xad_u32 v196, 0x360, v13, v12
	v_xad_u32 v197, 0x364, v13, v12
	v_xad_u32 v198, 0x520, v13, v12
	v_xad_u32 v199, 0x524, v13, v12
	v_xad_u32 v200, 0x640, v13, v12
	v_xad_u32 v201, 0x644, v13, v12
	v_xad_u32 v203, 0x760, v13, v12
	v_xad_u32 v204, 0x764, v13, v12
	v_cmp_gt_u32_e64 s0, 64, v0
.Ltmp514:
	.loc	1 1828 5 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1828:5 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_add_nc_u64 s[6:7], s[6:7], s[18:19]
	s_add_nc_u64 s[8:9], s[8:9], s[18:19]
	s_mov_b32 s18, 0x76543210
	v_mov_b32_e32 v181, 0
.Ltmp515:
	.loc	3 1334 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1334:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1821:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	v_min_i32_e32 v5, v9, v28
	v_mov_b32_e32 v9, v2
.Ltmp516:
	.loc	3 1337 10                       ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:1337:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1816:11 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x0
	v_max_i32_e32 v4, v10, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
.Ltmp517:
	.loc	1 1823 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1823:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_readfirstlane_b32 s21, v5
	v_xor_b32_e32 v5, v11, v30
	.loc	1 1822 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1822:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_readfirstlane_b32 s20, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v184, v5, 2, v12
	v_or_b32_e32 v5, 20, v7
	v_xor_b32_e32 v5, v5, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v213, v5, 2, v12
	v_or_b32_e32 v5, 0x11c, v7
	v_xor_b32_e32 v5, v5, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v218, v5, 2, v12
	v_or_b32_e32 v5, 48, v7
	v_xor_b32_e32 v5, v5, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v223, v5, 2, v12
	v_or_b32_e32 v5, 0x138, v7
	v_xor_b32_e32 v5, v5, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v228, v5, 2, v12
	v_mov_b32_e32 v5, v2
	v_xor_b32_e32 v3, v26, v30
	v_xor_b32_e32 v4, v27, v30
	v_xor_b32_e32 v6, v15, v30
	v_lshl_add_u32 v255, v3, 2, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v182, v4, 2, v12
	v_lshl_add_u32 v188, v6, 2, v12
	v_or_b32_e32 v3, 16, v7
	v_or_b32_e32 v4, 0x110, v7
	v_or_b32_e32 v6, 0x114, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v30
	v_xor_b32_e32 v4, v4, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v6, v6, v30
	v_lshl_add_u32 v211, v3, 2, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v212, v4, 2, v12
	v_lshl_add_u32 v214, v6, 2, v12
	v_or_b32_e32 v3, 0x118, v7
	v_or_b32_e32 v4, 28, v7
	v_or_b32_e32 v6, 40, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v30
	v_xor_b32_e32 v4, v4, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v6, v6, v30
	v_lshl_add_u32 v216, v3, 2, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v217, v4, 2, v12
	v_lshl_add_u32 v219, v6, 2, v12
	v_or_b32_e32 v3, 44, v7
	v_or_b32_e32 v4, 0x12c, v7
	v_or_b32_e32 v6, 0x130, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v30
	v_xor_b32_e32 v4, v4, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v6, v6, v30
	v_lshl_add_u32 v221, v3, 2, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v222, v4, 2, v12
	v_lshl_add_u32 v224, v6, 2, v12
	v_or_b32_e32 v3, 0x134, v7
	v_or_b32_e32 v4, 56, v7
	v_or_b32_e32 v6, 60, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v3, v3, v30
	v_xor_b32_e32 v4, v4, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v6, v6, v30
	v_lshl_add_u32 v226, v3, 2, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v227, v4, 2, v12
	v_lshl_add_u32 v229, v6, 2, v12
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	v_dual_mov_b32 v6, v2 :: v_dual_lshlrev_b32 v25, 4, v8
	v_lshlrev_b32_e32 v8, 2, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add3_u32 v206, v12, v8, v14
	v_or_b32_e32 v8, 24, v7
	v_xor_b32_e32 v8, v8, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v215, v8, 2, v12
	v_or_b32_e32 v8, 0x128, v7
	v_xor_b32_e32 v8, v8, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v220, v8, 2, v12
	v_or_b32_e32 v8, 52, v7
	v_or_b32_e32 v7, 0x13c, v7
	v_xor_b32_e32 v8, v8, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v7, v7, v30
	v_lshl_add_u32 v225, v8, 2, v12
	v_mov_b32_e32 v8, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v230, v7, 2, v12
	v_dual_mov_b32 v7, v2 :: v_dual_and_b32 v10, 0x3000, v16
	v_mov_b32_e32 v133, v9
	v_dual_mov_b32 v128, v4 :: v_dual_lshlrev_b32 v231, 3, v22
	v_dual_mov_b32 v131, v7 :: v_dual_lshlrev_b32 v232, 3, v18
	v_dual_mov_b32 v126, v2 :: v_dual_add_nc_u32 v233, 0, v19
	v_dual_mov_b32 v129, v5 :: v_dual_lshlrev_b32 v234, 3, v20
	v_lshlrev_b32_e32 v235, 3, v24
	v_dual_mov_b32 v125, v9 :: v_dual_add_nc_u32 v238, 0, v23
	v_dual_mov_b32 v120, v4 :: v_dual_add_nc_u32 v239, 0, v21
	v_mov_b32_e32 v132, v8
	v_dual_mov_b32 v127, v3 :: v_dual_add_nc_u32 v236, 0, v25
	v_dual_mov_b32 v117, v9 :: v_dual_mov_b32 v112, v4
	v_dual_mov_b32 v109, v9 :: v_dual_mov_b32 v104, v4
	v_dual_mov_b32 v101, v9 :: v_dual_mov_b32 v96, v4
	v_dual_mov_b32 v93, v9 :: v_dual_mov_b32 v88, v4
	v_dual_mov_b32 v85, v9 :: v_dual_mov_b32 v80, v4
	v_dual_mov_b32 v77, v9 :: v_dual_mov_b32 v72, v4
	v_dual_mov_b32 v69, v9 :: v_dual_mov_b32 v64, v4
	v_dual_mov_b32 v61, v9 :: v_dual_mov_b32 v56, v4
	v_dual_mov_b32 v53, v9 :: v_dual_mov_b32 v48, v4
	v_dual_mov_b32 v45, v9 :: v_dual_mov_b32 v40, v4
	v_dual_mov_b32 v37, v9 :: v_dual_mov_b32 v32, v4
	v_dual_mov_b32 v29, v9 :: v_dual_mov_b32 v24, v4
	v_dual_mov_b32 v21, v9 :: v_dual_mov_b32 v16, v4
	v_dual_mov_b32 v122, v6 :: v_dual_add_nc_u32 v237, v187, v10
	v_mov_b32_e32 v130, v6
	v_dual_mov_b32 v124, v8 :: v_dual_mov_b32 v123, v7
	v_dual_mov_b32 v118, v2 :: v_dual_mov_b32 v121, v5
	v_dual_mov_b32 v116, v8 :: v_dual_mov_b32 v119, v3
	v_dual_mov_b32 v114, v6 :: v_dual_mov_b32 v115, v7
	v_dual_mov_b32 v110, v2 :: v_dual_mov_b32 v113, v5
	v_dual_mov_b32 v108, v8 :: v_dual_mov_b32 v111, v3
	v_dual_mov_b32 v106, v6 :: v_dual_mov_b32 v107, v7
	v_dual_mov_b32 v102, v2 :: v_dual_mov_b32 v105, v5
	v_dual_mov_b32 v100, v8 :: v_dual_mov_b32 v103, v3
	v_dual_mov_b32 v98, v6 :: v_dual_mov_b32 v99, v7
	v_dual_mov_b32 v94, v2 :: v_dual_mov_b32 v97, v5
	v_dual_mov_b32 v92, v8 :: v_dual_mov_b32 v95, v3
	v_dual_mov_b32 v90, v6 :: v_dual_mov_b32 v91, v7
	v_dual_mov_b32 v86, v2 :: v_dual_mov_b32 v89, v5
	v_dual_mov_b32 v84, v8 :: v_dual_mov_b32 v87, v3
	v_dual_mov_b32 v82, v6 :: v_dual_mov_b32 v83, v7
	v_dual_mov_b32 v78, v2 :: v_dual_mov_b32 v81, v5
	v_dual_mov_b32 v76, v8 :: v_dual_mov_b32 v79, v3
	v_dual_mov_b32 v74, v6 :: v_dual_mov_b32 v75, v7
	v_dual_mov_b32 v70, v2 :: v_dual_mov_b32 v73, v5
	v_dual_mov_b32 v68, v8 :: v_dual_mov_b32 v71, v3
	v_dual_mov_b32 v66, v6 :: v_dual_mov_b32 v67, v7
	v_dual_mov_b32 v62, v2 :: v_dual_mov_b32 v65, v5
	v_dual_mov_b32 v60, v8 :: v_dual_mov_b32 v63, v3
	v_dual_mov_b32 v58, v6 :: v_dual_mov_b32 v59, v7
	v_dual_mov_b32 v54, v2 :: v_dual_mov_b32 v57, v5
	v_dual_mov_b32 v52, v8 :: v_dual_mov_b32 v55, v3
	v_dual_mov_b32 v50, v6 :: v_dual_mov_b32 v51, v7
	v_dual_mov_b32 v46, v2 :: v_dual_mov_b32 v49, v5
	v_dual_mov_b32 v44, v8 :: v_dual_mov_b32 v47, v3
	v_dual_mov_b32 v42, v6 :: v_dual_mov_b32 v43, v7
	v_dual_mov_b32 v38, v2 :: v_dual_mov_b32 v41, v5
	v_dual_mov_b32 v36, v8 :: v_dual_mov_b32 v39, v3
	v_dual_mov_b32 v34, v6 :: v_dual_mov_b32 v35, v7
	v_dual_mov_b32 v30, v2 :: v_dual_mov_b32 v33, v5
	v_dual_mov_b32 v28, v8 :: v_dual_mov_b32 v31, v3
	v_dual_mov_b32 v26, v6 :: v_dual_mov_b32 v27, v7
	v_dual_mov_b32 v22, v2 :: v_dual_mov_b32 v25, v5
	v_dual_mov_b32 v20, v8 :: v_dual_mov_b32 v23, v3
	v_dual_mov_b32 v18, v6 :: v_dual_mov_b32 v19, v7
	v_dual_mov_b32 v14, v2 :: v_dual_mov_b32 v17, v5
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v13, v9
	v_dual_mov_b32 v12, v8 :: v_dual_mov_b32 v11, v7
	v_dual_mov_b32 v10, v6 :: v_dual_mov_b32 v9, v5
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v2
	s_branch .LBB6_11
.LBB6_9:                                ;   in Loop: Header=BB6_11 Depth=1
.Ltmp518:
	.loc	4 701 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:701:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	s_wait_loadcnt 0x0
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	s_barrier_signal -1
	v_mov_b32_e32 v180, v3
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2059:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp519:
.LBB6_10:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	4 0 7 is_stmt 0                 ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:0:7
	s_add_co_i32 s13, s13, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s13, s14
	s_cselect_b32 s3, -1, 0
	s_xor_b32 s19, s19, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s19, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_57
.LBB6_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_48 Depth 2
	.loc	1 1829 32 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1829:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_lshl_b32 s22, s13, 6
	.loc	1 1830 19                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1830:19 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s22, s20
	s_cselect_b32 s19, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_10
; %bb.12:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 19 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:19
	v_or_b32_e32 v3, s22, v185
                                        ; implicit-def: $vgpr134
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[138:139], null, 0x408, v3, s[4:5]
	v_cmp_ge_i32_e32 vcc_lo, s20, v3
	.loc	1 1843 20 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s23, exec_lo, s3
	s_cbranch_execz .LBB6_14
; %bb.13:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 20 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:20
	s_delay_alu instid0(VALU_DEP_2)
	.loc	1 1850 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_co_u32 v3, s3, v138, v231
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, s3
	v_add_co_u32 v136, s3, v138, v232
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v137, null, 0, v139, s3
	s_clause 0x3
	global_load_b64 v[241:242], v[3:4], off
	global_load_b64 v[243:244], v[3:4], off offset:16
	global_load_b64 v[134:135], v[136:137], off
	global_load_b64 v[136:137], v[136:137], off offset:16
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v238, v[241:244]
.LBB6_14:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s23
	s_cbranch_execz .LBB6_16
; %bb.15:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v4, v2
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v134, v137
	v_dual_mov_b32 v136, v137 :: v_dual_mov_b32 v135, v137
	ds_store_b128 v238, v[2:5]
.LBB6_16:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v233, v[134:137]
                                        ; implicit-def: $vgpr134
	.loc	1 1843 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1843:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1850 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1850:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_co_u32 v3, vcc_lo, v138, v234
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v139, vcc_lo
	v_add_co_u32 v136, vcc_lo, v138, v235
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v139, vcc_lo
	s_clause 0x3
	global_load_b64 v[138:139], v[3:4], off
	global_load_b64 v[140:141], v[3:4], off offset:16
	global_load_b64 v[134:135], v[136:137], off
	global_load_b64 v[136:137], v[136:137], off offset:16
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x2
	ds_store_b128 v239, v[138:141]
.LBB6_18:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_11 Depth=1
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v4, v2
	.loc	1 1858 72 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v134, v137
	v_dual_mov_b32 v136, v137 :: v_dual_mov_b32 v135, v137
	ds_store_b128 v239, v[2:5]
.LBB6_20:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 72 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:72
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v138, 0 :: v_dual_add_nc_u32 v5, s22, v186
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v3, 0
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1858 72                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1858:72 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b128 v236, v[134:137]
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[178:179]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_22:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v134, 0x8000, v191
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v134, v3, v4 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_gt_i32_e64 s20, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_24
; %bb.23:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1862 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v3, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[178:179]
	global_load_b64 v[138:139], v[3:4], off
.LBB6_24:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v136, 2, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v192, v138 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b32 v193, v139 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v136
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_26
; %bb.25:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v136, v[178:179]
	global_load_b64 v[134:135], v[134:135], off
.LBB6_26:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v136, 3, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v194, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b32 v195, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v136
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_28
; %bb.27:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v136, v[178:179]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_28:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v138, 4, v5
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v196, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b32 v197, v4 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v138, v[178:179]
	global_load_b64 v[136:137], v[3:4], off
.LBB6_30:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v3, 5, v5
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_nc_u32_e32 v4, 0x8400, v191
	.loc	1 1865 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v4, v136, v137 offset1:1
	.loc	1 1863 34 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v3
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_32
; %bb.31:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v3, v[178:179]
	global_load_b64 v[134:135], v[3:4], off
.LBB6_32:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v138, 6, v5
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v198, v134 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b32 v199, v135 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v138
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_34
; %bb.33:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v138, v[178:179]
	global_load_b64 v[136:137], v[134:135], off
.LBB6_34:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1862 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1862:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v5, 7, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	.loc	1 1871 58                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v200, v136 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b32 v201, v137 offset:32768
	.loc	1 1863 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1863:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_ge_i32_e64 s20, v5
	.loc	1 1865 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1865:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_execz .LBB6_36
; %bb.35:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1869 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1869:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[3:4], null, 0x408, v5, v[178:179]
	global_load_b64 v[3:4], v[3:4], off
.LBB6_36:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 21 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1871 58 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1871:58 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	ds_store_b32 v203, v3 offset:32768
	.loc	1 1873 62                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1873:62 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b32 v204, v4 offset:32768
.Ltmp520:
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	.loc	4 703 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1876:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp521:
	.loc	1 1888 35                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1888:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_nc_u32_e32 v5, 0x8000, v206
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_nc_u32_e32 v138, 0x8400, v206
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_2addr_b32 v[3:4], v5 offset1:4
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_2addr_b32 v[134:135], v138 offset1:4
.Ltmp522:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v205, v3
.Ltmp523:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v137, v205, v134
.Ltmp524:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v139, v205, v4
.Ltmp525:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v140, v205, v135
.Ltmp526:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v136, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v136, v137, v134, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v139, v4, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v135, v189
.Ltmp527:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp528:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v136
.Ltmp529:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v139, v202, v4
.Ltmp530:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v140, v202, v137
.Ltmp531:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v136, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v4, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:16384
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v3, v255 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v4, v182 offset:32768
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v134, v184 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v135, v188 offset:32768
.Ltmp532:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v205, v3
.Ltmp533:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v205, v4
.Ltmp534:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v205, v134
.Ltmp535:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v205, v135
.Ltmp536:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v136, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v137, v4, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v134, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v135, v189
.Ltmp537:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp538:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v4
.Ltmp539:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v139, v202, v136
.Ltmp540:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v140, v202, v137
.Ltmp541:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v4, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v136, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:16896
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v3, v211 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v4, v212 offset:32768
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v134, v213 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v135, v214 offset:32768
.Ltmp542:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v205, v3
.Ltmp543:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v205, v4
.Ltmp544:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v205, v134
.Ltmp545:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v205, v135
.Ltmp546:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v136, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v137, v4, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v134, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v135, v189
.Ltmp547:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp548:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v4
.Ltmp549:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v139, v202, v136
.Ltmp550:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v140, v202, v137
.Ltmp551:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v4, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v136, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:17408
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v3, v215 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v4, v216 offset:32768
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v134, v217 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v135, v218 offset:32768
.Ltmp552:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v205, v3
.Ltmp553:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v205, v4
.Ltmp554:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v139, v205, v134
.Ltmp555:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v140, v205, v135
.Ltmp556:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v136, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v137, v4, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v134, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v135, v189
.Ltmp557:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp558:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v4
.Ltmp559:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v139, v202, v136
.Ltmp560:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v140, v202, v137
.Ltmp561:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v4, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v139, v136, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v140, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:17920
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_2addr_b32 v[3:4], v5 offset0:32 offset1:36
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_2addr_b32 v[134:135], v138 offset0:32 offset1:36
.Ltmp562:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v205, v3
.Ltmp563:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x1
	ds_bpermute_b32 v136, v205, v134
.Ltmp564:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v137, v205, v4
.Ltmp565:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v138, v205, v135
.Ltmp566:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v5, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v5, v136, v134, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v4, v137, v4, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v135, v189
.Ltmp567:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp568:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v5
.Ltmp569:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v136, v202, v4
.Ltmp570:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v138, v202, v137
.Ltmp571:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v5, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v136, v4, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:18432
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v3, v219 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v4, v220 offset:32768
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v5, v221 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v134, v222 offset:32768
.Ltmp572:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v205, v3
.Ltmp573:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v205, v4
.Ltmp574:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v205, v5
.Ltmp575:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v205, v134
.Ltmp576:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v135, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v136, v4, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v5, v137, v5, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v134, v189
.Ltmp577:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp578:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v4
.Ltmp579:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v136, v202, v5
.Ltmp580:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v138, v202, v137
.Ltmp581:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v4, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v136, v5, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:18944
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v3, v223 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v4, v224 offset:32768
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v5, v225 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v134, v226 offset:32768
.Ltmp582:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v205, v3
.Ltmp583:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v205, v4
.Ltmp584:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v205, v5
.Ltmp585:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v205, v134
.Ltmp586:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v135, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v136, v4, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v5, v137, v5, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v134, v189
.Ltmp587:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp588:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v4
.Ltmp589:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v136, v202, v5
.Ltmp590:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v138, v202, v137
.Ltmp591:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v4, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v136, v5, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:19456
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v3, v227 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v4, v228 offset:32768
	.loc	1 1896 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1896:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v5, v229 offset:32768
	.loc	1 1897 45                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1897:45 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b32 v134, v230 offset:32768
.Ltmp592:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v135, v205, v3
.Ltmp593:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v136, v205, v4
.Ltmp594:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1898:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v137, v205, v5
.Ltmp595:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1900:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	s_wait_dscnt 0x3
	ds_bpermute_b32 v138, v205, v134
.Ltmp596:
	.loc	1 1899 30 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v3, v135, v3, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v4, v136, v4, v189
	.loc	1 1899 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1899:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v5, v137, v5, v189
	.loc	1 1901 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1901:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v134, v189
.Ltmp597:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v134, v202, v3
.Ltmp598:
	.loc	2 527 12 is_stmt 0              ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v135, v202, v4
.Ltmp599:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1902:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v136, v202, v5
.Ltmp600:
	.loc	2 527 12                        ; /opt/rocm/core/include/hip/amd_detail/amd_warp_functions.h:527:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1904:49 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	ds_bpermute_b32 v138, v202, v137
.Ltmp601:
	.loc	1 1903 38 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_perm_b32 v134, v134, v3, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_perm_b32 v135, v135, v4, v190
	.loc	1 1903 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1903:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_perm_b32 v136, v136, v5, v190
	.loc	1 1906 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1906:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_perm_b32 v137, v138, v137, v190
	.loc	1 1909 77                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1909:77 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b128 v237, v[134:137] offset:19968
.LBB6_38:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 77 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:77
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1915 17 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1915:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB6_42
; %bb.39:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1916 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1916:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v4, s22, v0
	v_mov_b32_e32 v3, 0
	.loc	1 1918 20                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1918:20 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s23, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ge_i32_e64 s20, v4
	s_cbranch_execz .LBB6_41
; %bb.40:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1923 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[134:135], null, 0x408, v4, s[6:7]
	.loc	1 1925 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mad_co_i64_i32 v[4:5], null, 0x408, v4, s[8:9]
	.loc	1 1923 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	global_load_d16_b16 v3, v[134:135], off offset:1024
	.loc	1 1925 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	global_load_d16_hi_b16 v3, v[4:5], off offset:1024
	.loc	1 1923 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1923:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	v_lshrrev_b16 v4.h, 8, v3.l
	.loc	1 1925 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshrrev_b16 v4.l, 8, v3.h
	.loc	1 1925 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:23 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_and_b16 v5.h, 0xff, v3.l
	v_and_b16 v5.l, 0xff, v3.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 1926 56 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1926:56 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_pk_lshlrev_b16 v3, 8, v4 op_sel_hi:[0,1]
	.loc	1 1925 53                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1925:53 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_or_b32_e32 v3, v3, v5
.LBB6_41:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 53 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:53
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s23
	v_lshl_add_u32 v4, v0, 1, 0
	.loc	1 1928 61 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1928:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b16_d16_hi v4, v3 offset:49152
	.loc	1 1929 61                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1929:61 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_store_b16 v4, v3 offset:49280
.LBB6_42:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 61 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:61
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.Ltmp602:
	.loc	4 702 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1931:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
.Ltmp603:
	.loc	1 0 0 is_stmt 0                 ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b32_e32 v241, s16
.Ltmp604:
	.loc	4 702 7                         ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:702:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1931:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	s_barrier_wait -1
	.loc	4 703 7 is_stmt 1               ; /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:703:7 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:718:50 @[ /opt/rocm/core/include/hip/amd_detail/amd_device_functions.h:721:3 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1931:9 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ] ] ]
	global_inv scope:SCOPE_SE
.Ltmp605:
	.loc	1 1941 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1941:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_44
; %bb.43:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1942 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1942:27 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	global_load_b32 v3, v[174:175], off
	.loc	1 1943 31                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1943:31 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v241, s16, v3
.LBB6_44:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 31 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:31
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1834 35 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1834:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_or_b32 s3, s22, 63
	v_mov_b32_e32 v242, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s21
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s23, s15, s3
	.loc	1 1945 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s23, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s24, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s24
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 1945 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1945:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	global_load_b32 v242, v[176:177], off
.LBB6_46:                               ;   in Loop: Header=BB6_11 Depth=1
	.loc	1 0 46                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:46
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	.loc	1 1935 36 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1935:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_or_b32 s3, s22, 16
	s_mov_b32 s24, 0
	.loc	1 1935 41 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1935:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s3, s20
	s_cselect_b32 s25, -1, 0
	s_branch .LBB6_48
.LBB6_47:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 0 41                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:41
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_sub_f32_e32 v135, v180, v3
	.loc	1 1998 21 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v180
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2012 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_add_f32 v4, v4, v5 :: v_dual_mul_f32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v135, v135
	.loc	1 1998 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1998:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v135, 0, v135, vcc_lo
	.loc	1 2022 42                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mul_f32_e32 v136, v240, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2022 52 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v137, null, v134, v134, v136
	v_rcp_f32_e32 v180, v137
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v207, -v137, v180, 1.0
	v_fmac_f32_e32 v180, v207, v180
	v_div_scale_f32 v207, vcc_lo, v136, v134, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v208, v207, v180
	v_fma_f32 v209, -v137, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v208, v209, v180
	.loc	1 2013 39 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2013:39 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fmac_f32_e32 v4, v181, v135
	.loc	1 2022 52                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fma_f32 v137, -v137, v208, v207
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v181, v4
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v137, v137, v180, v208
	.loc	1 2028 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v180, null, v134, v134, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v207, v180
	v_fma_f32 v208, -v180, v207, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v207, v208, v207
	v_div_scale_f32 v208, vcc_lo, v245, v134, v245
	v_mul_f32_e32 v209, v208, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v210, -v180, v209, v208
	v_fmac_f32_e32 v209, v210, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v180, v209, v208
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v180, v180, v207, v209
	.loc	1 2028 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v207, null, v134, v134, v246
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2028 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v180, v180, v134, v245
	.loc	1 2028 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v208, v207
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v209, -v207, v208, 1.0
	v_fmac_f32_e32 v208, v209, v208
	v_div_scale_f32 v209, vcc_lo, v246, v134, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v210, v209, v208
	v_fma_f32 v240, -v207, v210, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v210, v240, v208
	v_fma_f32 v207, -v207, v210, v209
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v207, v207, v208, v210
	.loc	1 2030 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v208, null, v134, v134, v243
	.loc	1 2028 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2028:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v245, v207, v134, v246
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v209, v208
	v_fma_f32 v210, -v208, v209, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v209, v210, v209
	v_div_scale_f32 v210, vcc_lo, v243, v134, v243
	v_mul_f32_e32 v240, v210, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v247, -v208, v240, v210
	v_fmac_f32_e32 v240, v247, v209
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v208, v240, v210
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v209, v208, v209, v240
	.loc	1 2030 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v208, null, v134, v134, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2030 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v209, v209, v134, v243
	.loc	1 2030 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v210, v208
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v240, -v208, v210, 1.0
	v_fmac_f32_e32 v210, v240, v210
	v_div_scale_f32 v240, vcc_lo, v244, v134, v244
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v247, v240, v210
	v_fma_f32 v248, -v208, v247, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v247, v248, v210
	v_fma_f32 v208, -v208, v247, v240
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v210, v208, v210, v247
	.loc	1 2033 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v208, null, v134, v134, v138
	.loc	1 2030 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2030:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v210, v210, v134, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	.loc	1 2033 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v240, v208
	v_fma_f32 v247, -v208, v240, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v240, v247, v240
	v_div_scale_f32 v247, vcc_lo, v138, v134, v138
	v_mul_f32_e32 v248, v247, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v249, -v208, v248, v247
	v_fmac_f32_e32 v248, v249, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v208, v248, v247
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v240, v208, v240, v248
	.loc	1 2033 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v208, null, v134, v134, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2033 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v138, v240, v134, v138
	.loc	1 2033 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v247, v208
	v_mov_b32_e32 v240, v134
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v248, -v208, v247, 1.0
	v_fmac_f32_e32 v247, v248, v247
	v_div_scale_f32 v248, vcc_lo, v139, v134, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v249, v248, v247
	v_fma_f32 v250, -v208, v249, v248
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v249, v250, v247
	v_fma_f32 v208, -v208, v249, v248
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v247, v208, v247, v249
	.loc	1 2027 22 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b16_e64 v208.l, v2.l
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b16_e64 v208.h, 0
	.loc	1 2033 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2033:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v139, v247, v134, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 2027 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b16_e64 v207.l, v208.l
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b16_e64 v207.h, v208.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	.loc	1 2032 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2032:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cvt_pk_fp8_f32 v208.l, v138, v139
	.loc	1 2027 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2027:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cvt_pk_fp8_f32 v207.l, v180, v245
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v180, null, v134, v134, v140
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	.loc	1 2029 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2029:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cvt_pk_fp8_f32 v207.h, v209, v210
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v245, v180
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v246, -v180, v245, 1.0
	v_fmac_f32_e32 v245, v246, v245
	v_div_scale_f32 v246, vcc_lo, v140, v134, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v248, v246, v245
	v_fma_f32 v249, -v180, v248, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v248, v249, v245
	v_fma_f32 v180, -v180, v248, v246
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v180, v180, v245, v248
	.loc	1 2035 46 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v245, null, v134, v134, v141
	.loc	1 2035 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v138, v180, v134, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_rcp_f32_e32 v246, v245
	.loc	1 2022 52 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2022:52 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v140, v137, v134, v136
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v131, v131, v140 :: v_dual_mul_f32 v130, v130, v140
	v_mul_f32_e32 v133, v133, v140
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fma_f32 v248, -v245, v246, 1.0
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v132, v132, v140 :: v_dual_mul_f32 v129, v129, v140
	v_dual_mul_f32 v128, v128, v140 :: v_dual_mul_f32 v127, v127, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fmac_f32_e32 v246, v248, v246
	v_div_scale_f32 v248, vcc_lo, v141, v134, v141
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v126, v126, v140 :: v_dual_mul_f32 v125, v125, v140
	v_dual_mul_f32 v124, v124, v140 :: v_dual_mul_f32 v123, v123, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v249, v248, v246 :: v_dual_mul_f32 v122, v122, v140
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v121, v121, v140 :: v_dual_mul_f32 v120, v120, v140
	v_mul_f32_e32 v119, v119, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fma_f32 v250, -v245, v249, v248
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v118, v118, v140 :: v_dual_mul_f32 v117, v117, v140
	v_dual_mul_f32 v116, v116, v140 :: v_dual_mul_f32 v115, v115, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fmac_f32_e32 v249, v250, v246
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v114, v114, v140 :: v_dual_mul_f32 v113, v113, v140
	v_dual_mul_f32 v112, v112, v140 :: v_dual_mul_f32 v111, v111, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fma_f32 v245, -v245, v249, v248
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v110, v110, v140 :: v_dual_mul_f32 v109, v109, v140
	v_dual_mul_f32 v108, v108, v140 :: v_dual_mul_f32 v107, v107, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v245, v245, v246, v249
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v106, v106, v140 :: v_dual_mul_f32 v105, v105, v140
	v_dual_mul_f32 v104, v104, v140 :: v_dual_mul_f32 v103, v103, v140
	.loc	1 2035 46                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2035:46 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_fixup_f32 v139, v245, v134, v141
	.loc	1 2040 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2040:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshl_add_u32 v141, s24, 12, v187
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v102, v102, v140 :: v_dual_mul_f32 v101, v101, v140
	v_dual_mul_f32 v100, v100, v140 :: v_dual_mul_f32 v99, v99, v140
	s_delay_alu instid0(VALU_DEP_4)
	.loc	1 2034 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2034:22 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cvt_pk_fp8_f32 v208.h, v138, v139
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:16384
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v98, v98, v140 :: v_dual_mul_f32 v97, v97, v140
	v_dual_mul_f32 v96, v96, v140 :: v_dual_mul_f32 v95, v95, v140
	v_dual_mul_f32 v94, v94, v140 :: v_dual_mul_f32 v93, v93, v140
	v_dual_mul_f32 v92, v92, v140 :: v_dual_mul_f32 v91, v91, v140
	v_dual_mul_f32 v90, v90, v140 :: v_dual_mul_f32 v89, v89, v140
	v_dual_mul_f32 v88, v88, v140 :: v_dual_mul_f32 v87, v87, v140
	v_dual_mul_f32 v86, v86, v140 :: v_dual_mul_f32 v85, v85, v140
	v_dual_mul_f32 v84, v84, v140 :: v_dual_mul_f32 v83, v83, v140
	v_dual_mul_f32 v82, v82, v140 :: v_dual_mul_f32 v81, v81, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[126:133], v[136:137], v[207:208], v[126:133]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[138:139], v[207:208], v[118:125]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:16896
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v80, v80, v140 :: v_dual_mul_f32 v79, v79, v140
	v_dual_mul_f32 v78, v78, v140 :: v_dual_mul_f32 v77, v77, v140
	v_dual_mul_f32 v76, v76, v140 :: v_dual_mul_f32 v75, v75, v140
	v_dual_mul_f32 v74, v74, v140 :: v_dual_mul_f32 v73, v73, v140
	v_dual_mul_f32 v72, v72, v140 :: v_dual_mul_f32 v71, v71, v140
	v_dual_mul_f32 v70, v70, v140 :: v_dual_mul_f32 v69, v69, v140
	v_dual_mul_f32 v68, v68, v140 :: v_dual_mul_f32 v67, v67, v140
	v_dual_mul_f32 v66, v66, v140 :: v_dual_mul_f32 v65, v65, v140
	v_dual_mul_f32 v64, v64, v140 :: v_dual_mul_f32 v63, v63, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[110:117], v[136:137], v[207:208], v[110:117]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[102:109], v[138:139], v[207:208], v[102:109]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:17408
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v62, v62, v140 :: v_dual_mul_f32 v61, v61, v140
	v_dual_mul_f32 v60, v60, v140 :: v_dual_mul_f32 v59, v59, v140
	v_dual_mul_f32 v58, v58, v140 :: v_dual_mul_f32 v57, v57, v140
	v_dual_mul_f32 v56, v56, v140 :: v_dual_mul_f32 v55, v55, v140
	v_dual_mul_f32 v54, v54, v140 :: v_dual_mul_f32 v53, v53, v140
	v_dual_mul_f32 v52, v52, v140 :: v_dual_mul_f32 v51, v51, v140
	v_dual_mul_f32 v50, v50, v140 :: v_dual_mul_f32 v49, v49, v140
	v_dual_mul_f32 v48, v48, v140 :: v_dual_mul_f32 v47, v47, v140
	v_dual_mul_f32 v46, v46, v140 :: v_dual_mul_f32 v45, v45, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[94:101], v[136:137], v[207:208], v[94:101]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[86:93], v[138:139], v[207:208], v[86:93]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:17920
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v44, v44, v140 :: v_dual_mul_f32 v43, v43, v140
	v_dual_mul_f32 v42, v42, v140 :: v_dual_mul_f32 v41, v41, v140
	v_dual_mul_f32 v40, v40, v140 :: v_dual_mul_f32 v39, v39, v140
	v_dual_mul_f32 v38, v38, v140 :: v_dual_mul_f32 v37, v37, v140
	v_dual_mul_f32 v36, v36, v140 :: v_dual_mul_f32 v35, v35, v140
	v_dual_mul_f32 v34, v34, v140 :: v_dual_mul_f32 v33, v33, v140
	v_dual_mul_f32 v32, v32, v140 :: v_dual_mul_f32 v31, v31, v140
	v_dual_mul_f32 v30, v30, v140 :: v_dual_mul_f32 v29, v29, v140
	v_dual_mul_f32 v28, v28, v140 :: v_dual_mul_f32 v27, v27, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[78:85], v[136:137], v[207:208], v[78:85]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[70:77], v[138:139], v[207:208], v[70:77]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:18432
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v26, v26, v140 :: v_dual_mul_f32 v25, v25, v140
	v_dual_mul_f32 v24, v24, v140 :: v_dual_mul_f32 v23, v23, v140
	v_dual_mul_f32 v22, v22, v140 :: v_dual_mul_f32 v21, v21, v140
	v_dual_mul_f32 v20, v20, v140 :: v_dual_mul_f32 v19, v19, v140
	v_dual_mul_f32 v18, v18, v140 :: v_dual_mul_f32 v17, v17, v140
	v_dual_mul_f32 v16, v16, v140 :: v_dual_mul_f32 v15, v15, v140
	v_dual_mul_f32 v14, v14, v140 :: v_dual_mul_f32 v13, v13, v140
	v_dual_mul_f32 v12, v12, v140 :: v_dual_mul_f32 v11, v11, v140
	v_dual_mul_f32 v10, v10, v140 :: v_dual_mul_f32 v9, v9, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[62:69], v[136:137], v[207:208], v[62:69]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[54:61], v[138:139], v[207:208], v[54:61]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:18944
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2025 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2025:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v8, v8, v140 :: v_dual_mul_f32 v7, v7, v140
	v_mul_f32_e32 v6, v6, v140
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[136:137], v[207:208], v[46:53]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[38:45], v[138:139], v[207:208], v[38:45]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:19456
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[30:37], v[136:137], v[207:208], v[30:37]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[22:29], v[138:139], v[207:208], v[22:29]
	.loc	1 2042 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2042:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[136:139], v141 offset:19968
	.loc	1 2051 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2051:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 2046 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2046:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[14:21], v[136:137], v[207:208], v[14:21]
	.loc	1 2049 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2049:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[6:13], v[138:139], v[207:208], v[6:13]
	v_mov_b32_e32 v180, v3
	.loc	1 1951 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_add_co_i32 s24, s24, 1
	.loc	1 1951 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s24, 4
	.loc	1 1951 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_scc1 .LBB6_9
.LBB6_48:                               ;   Parent Loop BB6_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 0 13                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:13
	s_cmp_lt_i32 s24, 1
	s_mov_b32 s3, -1
	s_cbranch_scc1 .LBB6_51
; %bb.49:                               ;   in Loop: Header=BB6_48 Depth=2
	s_cmp_eq_u32 s24, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s3, s25
	s_cbranch_scc1 .LBB6_51
; %bb.50:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 1953 47 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cmp_eq_u32 s24, 2
	.loc	1 1953 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1953:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cselect_b32 s3, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s22
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s20, s3
	s_cselect_b32 s3, -1, 0
.LBB6_51:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 0 42                          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:42
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_56
; %bb.52:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 1964 43 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1964:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshl_add_u32 v3, s24, 9, v187
	.loc	1 1976 48                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:48 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_lshl_b32 s3, s24, 4
	.loc	1 1976 42 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1976:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s3, s22
	.loc	1 1977 44 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s3, 15
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[243:246], v3
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[247:250], v3 offset:2048
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[251:254], v3 offset:4096
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1977 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cmp_le_i32 s26, s21
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[243:244], v[142:143], 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[245:246], v[144:145], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[243:246], v3 offset:6144
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[247:248], v[146:147], v[134:141]
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[249:250], v[148:149], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[247:250], v3 offset:8192
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[251:252], v[150:151], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[253:254], v[152:153], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[251:254], v3 offset:10240
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[207:210], v3 offset:12288
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[243:244], v[154:155], v[134:141]
	v_lshl_add_u32 v243, s24, 5, v1
	v_mov_b32_e32 v244, 0xff800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[245:246], v[156:157], v[134:141]
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[247:248], v[158:159], v[134:141]
	s_delay_alu instid0(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[249:250], v[160:161], v[134:141]
	.loc	1 1966 38                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1966:38 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b128 v[246:249], v3 offset:14336
	.loc	1 1973 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1973:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	;;#ASMSTART
	;;#ASMEND
	.loc	1 1981 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_b96 v[3:5], v243 offset:49154
	ds_load_u16_d16 v245, v243 offset:49166
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[251:252], v[162:163], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[253:254], v[164:165], v[134:141]
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[207:208], v[166:167], v[134:141]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[209:210], v[168:169], v[134:141]
	.loc	1 1969 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1969:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[246:247], v[170:171], v[134:141]
	v_or_b32_e32 v246, s3, v183
	.loc	1 1977 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1977:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cselect_b32 s3, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s23, s3
	.loc	1 1971 28                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1971:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_wmma_f32_16x16x16_fp8_fp8 v[134:141], v[248:249], v[172:173], v[134:141]
	.loc	1 1986 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_54
; %bb.53:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 1981 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1981:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_u16_d16 v207, v243 offset:49152
	v_mul_f32_e32 v134, v241, v134
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e32 vcc_lo, v246, v242
	s_or_b32 vcc_lo, s26, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v134, v207, v134, neg(0) op_sel_hi:[1,0,0]
	.loc	1 1988 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1988:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v244, 0xff800000, v134, vcc_lo
.LBB6_54:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 0 25 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:25
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v134, v241, v135
	v_or_b32_e32 v135, 2, v246
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, v246, v242
	v_mul_f32_e32 v136, v241, v136
	s_wait_dscnt 0x1
	v_fma_mix_f32 v134, v3, v134, neg(0) op_sel_hi:[1,0,0]
	v_cmp_le_i32_e64 s3, v135, v242
	v_or_b32_e32 v135, 3, v246
	s_or_b32 s27, s26, vcc_lo
	v_fma_mix_f32 v3, v3, v136, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 1986 29 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_b32 vcc_lo, s2, s27
	s_or_b32 s3, s26, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v134, 0xff800000, v134, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135, v242
	v_mul_f32_e32 v135, v241, v137
	v_or_b32_e32 v136, 4, v246
	s_and_b32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v135, v241, v138
	s_or_b32 s3, s26, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v242
	v_or_b32_e32 v136, 5, v246
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v4, v135, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v241, v139
	s_or_b32 s3, s26, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v242
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s2, s3
	v_or_b32_e32 v136, 7, v246
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v135, 0xff800000, v3, s3
	v_fma_mix_f32 v3, v5, v4, neg(0) op_sel_hi:[1,0,0]
	v_or_b32_e32 v4, 6, v246
	s_or_b32 s3, s26, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, s2, s3
	v_cmp_le_i32_e64 s3, v136, v242
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v139, 0xff800000, v3, vcc_lo
	v_mul_f32_e32 v3, v241, v140
	v_cmp_le_i32_e32 vcc_lo, v4, v242
	v_mul_f32_e32 v4, v241, v141
.Ltmp606:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v136, v244, 0xff800000, v134
	s_or_b32 s3, s26, s3
	v_fma_mix_f32 v3, v5, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_or_b32 s27, s26, vcc_lo
	s_wait_dscnt 0x0
	v_fma_mix_f32 v4, v245, v4, neg(0) op_sel_hi:[1,0,0]
	v_max3_num_f32 v5, v136, v137, v138
.Ltmp607:
	.loc	1 1986 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_b32 vcc_lo, s2, s27
	v_add_nc_u32_e32 v140, 0xc080, v243
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v136, 0xff800000, v3, vcc_lo
	s_and_b32 vcc_lo, s2, s3
.Ltmp608:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v3, v5, v135, v139
.Ltmp609:
	.loc	1 1986 29                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1986:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e32 v4, 0xff800000, v4, vcc_lo
	.loc	1 2020 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2020:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_mov_b32 s3, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp610:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1994:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v3, v3, v136, v4
.Ltmp611:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1995:32 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_mov_b32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v5, v5, s18, 0xfedcba98
.Ltmp612:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1996:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v3, v180, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v5, v244, v3
	v_dual_sub_f32 v4, v4, v3 :: v_dual_sub_f32 v135, v135, v3
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v3
	v_dual_sub_f32 v136, v136, v3 :: v_dual_mul_f32 v5, 0x3fb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v4, 0x3fb8aa3b, v4
	v_mul_f32_e32 v208, 0x3fb8aa3b, v135
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	v_exp_f32_e32 v141, v4
	v_sub_f32_e32 v4, v137, v3
	v_sub_f32_e32 v134, v134, v3
	v_sub_f32_e32 v137, v138, v3
	v_exp_f32_e32 v138, v5
	v_exp_f32_e32 v136, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v134, 0x3fb8aa3b, v134 :: v_dual_mul_f32 v137, 0x3fb8aa3b, v137
	v_exp_f32_e32 v207, v134
	v_dual_sub_f32 v134, v139, v3 :: v_dual_mul_f32 v139, 0x3fb8aa3b, v4
.Ltmp613:
	.loc	1 2002 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2002:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_2addr_b32 v[4:5], v140 offset1:1
	v_add_nc_u32_e32 v140, 0xc088, v243
	v_exp_f32_e32 v137, v137
	v_mul_f32_e32 v134, 0x3fb8aa3b, v134
	v_exp_f32_e32 v139, v139
	.loc	1 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_cndmask_b32_e64 v136, v136, 0, vcc_lo
	v_exp_f32_e32 v209, v134
	.loc	1 2002 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2002:41 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	ds_load_2addr_b32 v[134:135], v140 offset1:1
	v_exp_f32_e32 v140, v208
	.loc	1 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cndmask_b32_e64 v208, v141, 0, vcc_lo
	v_cndmask_b32_e64 v141, v207, 0, vcc_lo
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	v_cndmask_b32_e64 v139, v139, 0, vcc_lo
	v_cndmask_b32_e64 v207, v209, 0, vcc_lo
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x1
	v_fma_mix_f32 v245, v4, v138, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v246, v4, v141, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_f32_e32 v138, v138, v141
	.loc	1 2006 37                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2006:37 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cndmask_b32_e64 v4, v140, 0, vcc_lo
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fma_mix_f32 v243, v5, v139, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v244, v5, v137, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp614:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v5, v245, 0, v246
.Ltmp615:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_f32_e32 v140, v139, v138
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_dscnt 0x0
	v_fma_mix_f32 v138, v134, v4, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v139, v134, v207, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
.Ltmp616:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v5, v5, v243, v244
.Ltmp617:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_f32_e32 v134, v137, v140
	.loc	1 2010 33                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2010:33 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_fma_mix_f32 v140, v135, v136, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v135, v208, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
.Ltmp618:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v5, v5, v138, v139
.Ltmp619:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_f32_e32 v4, v4, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp620:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2017:28 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max3_num_f32 v134, v5, v140, v141
.Ltmp621:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_add_f32 v4, v207, v4 :: v_dual_mov_b32 v5, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v136, v4
.Ltmp622:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2018:36 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_permlanex16_b32 v5, v5, s18, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp623:
	.loc	1 2009 25                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2009:25 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_add_f32 v4, v208, v4 :: v_dual_max_num_f32 v135, v5, v5
.Ltmp624:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_mov_b32_e32 v5, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
.Ltmp625:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2018:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max_num_f32_e32 v135, v134, v135
.Ltmp626:
	.loc	1 1740 12                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1740:12 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2012:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_permlanex16_b32 v5, v5, s18, 0xfedcba98
	v_mov_b32_e32 v134, v240
	s_delay_alu instid0(VALU_DEP_3)
.Ltmp627:
	.loc	1 2020 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2020:26 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_cmpx_lt_f32_e32 0, v135
	s_cbranch_execz .LBB6_47
; %bb.55:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	1 2021 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_div_scale_f32 v134, null, 0x43e00000, 0x43e00000, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v136, v134
	v_fma_f32 v137, -v134, v136, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v136, v137, v136
	v_div_scale_f32 v137, vcc_lo, v135, 0x43e00000, v135
	v_mul_f32_e32 v207, v137, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v208, -v134, v207, v137
	v_fmac_f32_e32 v207, v208, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v134, -v134, v207, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v134, v134, v136, v207
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v134, v134, 0x43e00000, v135
.Ltmp628:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2021:29 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ] ]
	v_max_num_f32_e32 v134, 0x1f800000, v134
	s_branch .LBB6_47
.Ltmp629:
.LBB6_56:                               ;   in Loop: Header=BB6_48 Depth=2
	.loc	3 0 44 is_stmt 0                ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:0:44
	v_mov_b32_e32 v3, v180
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v180, v3
	.loc	1 1951 40 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:40 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_add_co_i32 s24, s24, 1
	.loc	1 1951 35 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:35 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s24, 4
	.loc	1 1951 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:1951:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_cbranch_scc0 .LBB6_48
	s_branch .LBB6_9
.LBB6_57:
	.loc	1 2076 16 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2076:16 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	scratch_load_b32 v0, off, off offset:4 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_and_b32 s1, vcc_lo, s2
	s_mov_b32 s0, exec_lo
	scratch_load_b32 v2, off, off           ; 4-byte Folded Reload
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 exec_lo, s1
	s_cbranch_execz .LBB6_59
; %bb.58:
	.loc	1 2080 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2080:47 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, s17, v2, s[12:13]
	.loc	1 2082 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2081 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2081:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mul_lo_u32 v0, 0x102, v0
	.loc	1 2082 34                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2082:34 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s10, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	.loc	1 2084 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2084:24 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	global_store_b64 v[0:1], v[180:181], off
.LBB6_59:
	.loc	1 0 24 is_stmt 0                ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:0:24
	s_or_b32 exec_lo, exec_lo, s0
	.loc	1 2087 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2087:13 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB6_61
; %bb.60:
	.loc	1 2090 43                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2090:43 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[1:2], null, s17, v2, s[12:13]
	.loc	1 2092 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v3, 2, v183
	.loc	1 2098 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mul_f32_e32 v0, v126, v240
	v_dual_mul_f32 v4, v118, v240 :: v_dual_mul_f32 v5, v119, v240
	v_dual_mul_f32 v110, v110, v240 :: v_dual_mul_f32 v111, v111, v240
	.loc	1 2091 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2091:17 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_mul_lo_u32 v1, 0x102, v1
	.loc	1 2098 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v102, v102, v240 :: v_dual_mul_f32 v103, v103, v240
	v_dual_mul_f32 v94, v94, v240 :: v_dual_mul_f32 v95, v95, v240
	v_dual_mul_f32 v86, v86, v240 :: v_dual_mul_f32 v87, v87, v240
	v_dual_mul_f32 v78, v78, v240 :: v_dual_mul_f32 v79, v79, v240
	.loc	1 2092 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	.loc	1 2098 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v70, v70, v240 :: v_dual_mul_f32 v71, v71, v240
	v_dual_mul_f32 v62, v62, v240 :: v_dual_mul_f32 v63, v63, v240
	v_dual_mul_f32 v54, v54, v240 :: v_dual_mul_f32 v55, v55, v240
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	.loc	1 2092 30                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2092:30 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	.loc	1 2098 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v46, v46, v240 :: v_dual_mul_f32 v47, v47, v240
	.loc	1 2098 21 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_add_co_u32 v134, vcc_lo, v1, v3
	.loc	1 2098 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v38, v38, v240 :: v_dual_mul_f32 v39, v39, v240
	v_dual_mul_f32 v30, v30, v240 :: v_dual_mul_f32 v31, v31, v240
	v_dual_mul_f32 v22, v22, v240 :: v_dual_mul_f32 v23, v23, v240
	.loc	1 2098 21                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:21 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, 0, v2, vcc_lo
	.loc	1 2098 60                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:60 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	v_dual_mul_f32 v118, v240, v6 :: v_dual_mul_f32 v119, v240, v7
	v_dual_mul_f32 v1, v127, v240 :: v_dual_mul_f32 v2, v128, v240
	v_dual_mul_f32 v3, v129, v240 :: v_dual_mul_f32 v6, v120, v240
	v_dual_mul_f32 v7, v121, v240 :: v_dual_mul_f32 v112, v112, v240
	v_dual_mul_f32 v113, v113, v240 :: v_dual_mul_f32 v104, v104, v240
	v_dual_mul_f32 v105, v105, v240 :: v_dual_mul_f32 v96, v96, v240
	v_dual_mul_f32 v97, v97, v240 :: v_dual_mul_f32 v88, v88, v240
	v_dual_mul_f32 v89, v89, v240 :: v_dual_mul_f32 v80, v80, v240
	v_dual_mul_f32 v81, v81, v240 :: v_dual_mul_f32 v72, v72, v240
	v_dual_mul_f32 v73, v73, v240 :: v_dual_mul_f32 v64, v64, v240
	v_dual_mul_f32 v65, v65, v240 :: v_dual_mul_f32 v56, v56, v240
	v_dual_mul_f32 v57, v57, v240 :: v_dual_mul_f32 v48, v48, v240
	v_dual_mul_f32 v49, v49, v240 :: v_dual_mul_f32 v40, v40, v240
	v_dual_mul_f32 v41, v41, v240 :: v_dual_mul_f32 v32, v32, v240
	v_dual_mul_f32 v33, v33, v240 :: v_dual_mul_f32 v24, v24, v240
	v_mul_f32_e32 v25, v25, v240
	v_dual_mul_f32 v120, v240, v8 :: v_dual_mul_f32 v121, v240, v9
	v_dual_mul_f32 v126, v130, v240 :: v_dual_mul_f32 v127, v131, v240
	v_dual_mul_f32 v8, v122, v240 :: v_dual_mul_f32 v129, v133, v240
	v_dual_mul_f32 v114, v114, v240 :: v_dual_mul_f32 v9, v123, v240
	v_dual_mul_f32 v106, v106, v240 :: v_dual_mul_f32 v115, v115, v240
	v_dual_mul_f32 v98, v98, v240 :: v_dual_mul_f32 v107, v107, v240
	v_dual_mul_f32 v90, v90, v240 :: v_dual_mul_f32 v99, v99, v240
	v_dual_mul_f32 v82, v82, v240 :: v_dual_mul_f32 v91, v91, v240
	v_dual_mul_f32 v74, v74, v240 :: v_dual_mul_f32 v83, v83, v240
	v_dual_mul_f32 v66, v66, v240 :: v_dual_mul_f32 v75, v75, v240
	v_dual_mul_f32 v58, v58, v240 :: v_dual_mul_f32 v67, v67, v240
	v_dual_mul_f32 v50, v50, v240 :: v_dual_mul_f32 v59, v59, v240
	v_dual_mul_f32 v42, v42, v240 :: v_dual_mul_f32 v51, v51, v240
	v_dual_mul_f32 v34, v34, v240 :: v_dual_mul_f32 v43, v43, v240
	v_dual_mul_f32 v26, v26, v240 :: v_dual_mul_f32 v35, v35, v240
	v_dual_mul_f32 v128, v132, v240 :: v_dual_mul_f32 v27, v27, v240
	v_dual_mul_f32 v122, v240, v10 :: v_dual_mul_f32 v123, v240, v11
	v_dual_mul_f32 v10, v124, v240 :: v_dual_mul_f32 v11, v125, v240
	v_dual_mul_f32 v116, v116, v240 :: v_dual_mul_f32 v117, v117, v240
	v_dual_mul_f32 v108, v108, v240 :: v_dual_mul_f32 v109, v109, v240
	v_dual_mul_f32 v100, v100, v240 :: v_dual_mul_f32 v101, v101, v240
	v_dual_mul_f32 v92, v92, v240 :: v_dual_mul_f32 v93, v93, v240
	v_dual_mul_f32 v84, v84, v240 :: v_dual_mul_f32 v85, v85, v240
	v_dual_mul_f32 v76, v76, v240 :: v_dual_mul_f32 v77, v77, v240
	v_dual_mul_f32 v68, v68, v240 :: v_dual_mul_f32 v69, v69, v240
	v_dual_mul_f32 v60, v60, v240 :: v_dual_mul_f32 v61, v61, v240
	v_dual_mul_f32 v52, v52, v240 :: v_dual_mul_f32 v53, v53, v240
	v_dual_mul_f32 v44, v44, v240 :: v_dual_mul_f32 v45, v45, v240
	v_dual_mul_f32 v36, v36, v240 :: v_dual_mul_f32 v37, v37, v240
	v_dual_mul_f32 v28, v28, v240 :: v_dual_mul_f32 v29, v29, v240
	v_dual_mul_f32 v14, v14, v240 :: v_dual_mul_f32 v15, v15, v240
	v_dual_mul_f32 v16, v16, v240 :: v_dual_mul_f32 v17, v17, v240
	v_dual_mul_f32 v18, v18, v240 :: v_dual_mul_f32 v19, v19, v240
	v_dual_mul_f32 v20, v20, v240 :: v_dual_mul_f32 v21, v21, v240
	v_dual_mul_f32 v124, v240, v12 :: v_dual_mul_f32 v125, v240, v13
	.loc	1 2098 51                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2098:51 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2162:5 ]
	s_clause 0x1f
	global_store_b128 v[134:135], v[0:3], off offset:8
	global_store_b128 v[134:135], v[126:129], off offset:24
	global_store_b128 v[134:135], v[4:7], off offset:72
	global_store_b128 v[134:135], v[8:11], off offset:88
	global_store_b128 v[134:135], v[110:113], off offset:136
	global_store_b128 v[134:135], v[114:117], off offset:152
	global_store_b128 v[134:135], v[102:105], off offset:200
	global_store_b128 v[134:135], v[106:109], off offset:216
	global_store_b128 v[134:135], v[94:97], off offset:264
	global_store_b128 v[134:135], v[98:101], off offset:280
	global_store_b128 v[134:135], v[86:89], off offset:328
	global_store_b128 v[134:135], v[90:93], off offset:344
	global_store_b128 v[134:135], v[78:81], off offset:392
	global_store_b128 v[134:135], v[82:85], off offset:408
	global_store_b128 v[134:135], v[70:73], off offset:456
	global_store_b128 v[134:135], v[74:77], off offset:472
	global_store_b128 v[134:135], v[62:65], off offset:520
	global_store_b128 v[134:135], v[66:69], off offset:536
	global_store_b128 v[134:135], v[54:57], off offset:584
	global_store_b128 v[134:135], v[58:61], off offset:600
	global_store_b128 v[134:135], v[46:49], off offset:648
	global_store_b128 v[134:135], v[50:53], off offset:664
	global_store_b128 v[134:135], v[38:41], off offset:712
	global_store_b128 v[134:135], v[42:45], off offset:728
	global_store_b128 v[134:135], v[30:33], off offset:776
	global_store_b128 v[134:135], v[34:37], off offset:792
	global_store_b128 v[134:135], v[22:25], off offset:840
	global_store_b128 v[134:135], v[26:29], off offset:856
	global_store_b128 v[134:135], v[14:17], off offset:904
	global_store_b128 v[134:135], v[18:21], off offset:920
	global_store_b128 v[134:135], v[118:121], off offset:968
	global_store_b128 v[134:135], v[122:125], off offset:984
.Ltmp630:
.LBB6_61:
	.loc	1 2165 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2165:1
	s_endpgm
.Ltmp631:
.Lfunc_end6:
	.size	attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201, .Lfunc_end6-attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 12
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
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
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_vgpr, 256
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_agpr, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.numbered_sgpr, 28
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.num_named_barrier, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.private_seg_size, 12
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_vcc, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.uses_flat_scratch, 1
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_dyn_sized_stack, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_recursion, 0
	.set .Lattention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10432
; TotalNumSgprs: 30
; NumVgprs: 256
; ScratchSize: 12
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.loc	1 2176 17 prologue_end          ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2176:17
	s_load_b128 s[4:7], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s5, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s5, s7, -9
	.loc	1 2176 23 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2176:23
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s5, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB7_10
; %bb.1:
	.loc	1 2181 26 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2181:26
	v_lshrrev_b32_e32 v1, 5, v0
	.loc	1 2184 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2184:27
	s_mul_i32 s2, s4, 24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	.loc	1 2183 36                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2183:36
	v_lshl_or_b32 v4, ttmp9, 3, v1
	.loc	1 2184 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2184:13
	v_cmp_gt_i32_e32 vcc_lo, s2, v4
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB7_10
; %bb.2:
	.loc	1 2176 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2176:17
	s_load_b128 s[0:3], s[0:1], 0x0
	.loc	1 2187 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5
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
	.loc	1 2190 22                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:22
	global_load_b32 v3, v[6:7], off
.Ltmp632:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:13 ]
	v_max_num_f32_e32 v8, v8, v8
.Ltmp633:
	.loc	1 2187 23                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:23
	v_add_co_u32 v6, vcc_lo, 0x408, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
.Ltmp634:
	.loc	3 454 44                        ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:454:44 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2190:13 ]
	s_wait_loadcnt 0x0
	v_max_num_f32_e32 v3, v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v8, v8, v3
.Ltmp635:
	.loc	1 2187 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2187:5
	s_cbranch_scc0 .LBB7_3
; %bb.4:
	.loc	1 2182 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2182:26
	v_and_b32_e32 v3, 31, v0
	v_lshlrev_b64_e32 v[5:6], 10, v[4:5]
	.loc	1 2192 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2192:5
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
	.loc	1 2204 13 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2204:13
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
	.loc	1 2192 40                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2192:40
	v_add_nc_u32_e32 v14, 32, v3
	.loc	1 2204 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2204:13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v13, v5, v6, v13
	.loc	1 2203 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2203:9
	v_lshlrev_b64_e32 v[5:6], 2, v[3:4]
	.loc	1 2192 26                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2192:26
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v3
	.loc	1 2204 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2204:13
	v_div_fixup_f32 v3, v13, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	.loc	1 2203 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2203:9
	v_add_co_u32 v5, s0, v0, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v7, v6, s0
	.loc	1 2204 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2204:18
	v_cmp_lt_f32_e64 s0, 0, v11
	.loc	1 2192 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2192:5
	s_or_b32 s1, vcc_lo, s1
	.loc	1 2204 13                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2204:13
	v_cndmask_b32_e64 v11, 0, v3, s0
	.loc	1 2192 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2192:5
	v_add_co_u32 v9, s0, 0x80, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_mov_b32_e32 v3, v14
	.loc	1 2203 44                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2203:44
	global_store_b32 v[5:6], v11, off
	.loc	1 2192 5                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2192:5
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
	.loc	1 2201 24 is_stmt 1             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:24
	v_add_co_u32 v15, vcc_lo, v5, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v6, v10, vcc_lo
	.loc	1 2195 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2195:27
	v_add_co_u32 v5, vcc_lo, 0x408, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	.loc	1 2201 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:24
	global_load_b32 v15, v[15:16], off
	.loc	1 2200 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2200:17
	v_fmac_f32_e32 v11, v13, v14
	.loc	1 2195 27                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2195:27
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	.loc	1 2201 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2201:17
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v12, v14, v15
	.loc	1 2195 9                        ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2195:9
	s_cbranch_scc1 .LBB7_5
.LBB7_8:                                ;   Parent Loop BB7_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	1 2199 18                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:18
	global_load_b32 v13, v[5:6], off offset:4
	v_mov_b32_e32 v14, 0
	.loc	1 2199 17 is_stmt 0             ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:17
	s_mov_b32 s2, exec_lo
	.loc	1 2199 24                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:24
	s_wait_loadcnt 0x0
	v_cmpx_lt_f32_e32 0, v13
	.loc	1 2199 17                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:17
	s_cbranch_execz .LBB7_7
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=2
	.loc	1 2199 41                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:41
	global_load_b32 v14, v[5:6], off
	.loc	1 2199 47                       ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:47
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v14, v14, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
.Ltmp636:
	.loc	3 202 42 is_stmt 1              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:42 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:34 ]
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	.loc	3 202 10 is_stmt 0              ; /opt/rocm/core-10.0/lib/llvm/lib/clang/23/include/__clang_hip_math.h:202:10 @[ kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2199:34 ]
	v_exp_f32_e32 v14, v14
	s_branch .LBB7_7
.Ltmp637:
.LBB7_10:
	.loc	1 2206 1 is_stmt 1              ; kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip:2206:1
	s_endpgm
.Ltmp638:
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
	.type	__hip_cuid_d74eb5f92d393693,@object ; @__hip_cuid_d74eb5f92d393693
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d74eb5f92d393693
__hip_cuid_d74eb5f92d393693:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d74eb5f92d393693, 1

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
	.byte	1                               ; Abbrev [1] 0xc:0xa52 DW_TAG_compile_unit
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
	.long	.Ltmp146-.Ltmp145               ; DW_AT_high_pc
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
	.long	.Ltmp149-.Ltmp148               ; DW_AT_high_pc
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
	.long	.Ltmp153-.Ltmp152               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x424:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_low_pc
	.long	.Ltmp157-.Ltmp156               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1000                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x432:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_low_pc
	.long	.Ltmp158-.Ltmp157               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1005                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x440:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_low_pc
	.long	.Ltmp160-.Ltmp159               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x44e:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_low_pc
	.long	.Ltmp167-.Ltmp166               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1001                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x45c:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_low_pc
	.long	.Ltmp168-.Ltmp167               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1006                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x46a:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_low_pc
	.long	.Ltmp170-.Ltmp169               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x478:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_low_pc
	.long	.Ltmp172-.Ltmp171               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1002                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x486:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_low_pc
	.long	.Ltmp173-.Ltmp172               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1007                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x494:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_low_pc
	.long	.Ltmp176-.Ltmp175               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4a2:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_low_pc
	.long	.Ltmp181-.Ltmp180               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1003                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4b0:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_low_pc
	.long	.Ltmp182-.Ltmp181               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4be:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp186-.Ltmp185               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1263                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4cc:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp186-.Ltmp185               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4da:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_low_pc
	.long	.Ltmp186-.Ltmp185               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x4ea:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1090                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x4f8:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x506:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
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
	.long	.Ltmp211-.Ltmp210               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1186                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x538:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_low_pc
	.long	.Ltmp211-.Ltmp210               ; DW_AT_high_pc
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
	.long	.Ltmp214-.Ltmp213               ; DW_AT_high_pc
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
	.long	.Ltmp224-.Ltmp223               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x577:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	58                              ; DW_AT_low_pc
	.long	.Ltmp224-.Ltmp223               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x586:0x1d DW_TAG_inlined_subroutine
	.long	59                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_low_pc
	.long	.Ltmp226-.Ltmp225               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1209                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x594:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	59                              ; DW_AT_low_pc
	.long	.Ltmp226-.Ltmp225               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	552                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x5a3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	60                              ; DW_AT_low_pc
	.long	.Ltmp227-.Ltmp226               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1220                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x5b1:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	61                              ; DW_AT_low_pc
	.long	.Ltmp229-.Ltmp228               ; DW_AT_high_pc
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
	.long	.Ltmp239-.Ltmp238               ; DW_AT_high_pc
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
	.byte	3                               ; Abbrev [3] 0x5e7:0x22a DW_TAG_subprogram
	.byte	64                              ; DW_AT_low_pc
	.long	.Lfunc_end5-.Lfunc_begin5       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	25                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x5ee:0x222 DW_TAG_inlined_subroutine
	.long	1507                            ; DW_AT_abstract_origin
	.byte	28                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2128                            ; DW_AT_call_line
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
	.long	.Ltmp247-.Ltmp246               ; DW_AT_high_pc
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
	.byte	5                               ; Abbrev [5] 0x625:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	66                              ; DW_AT_low_pc
	.long	.Ltmp260-.Ltmp259               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x633:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	32                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x63d:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	67                              ; DW_AT_low_pc
	.long	.Ltmp264-.Ltmp263               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x64b:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	68                              ; DW_AT_low_pc
	.long	.Ltmp265-.Ltmp264               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x659:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	69                              ; DW_AT_low_pc
	.long	.Ltmp267-.Ltmp266               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x667:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	33                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x671:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	70                              ; DW_AT_low_pc
	.long	.Ltmp272-.Ltmp271               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x67f:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	71                              ; DW_AT_low_pc
	.long	.Ltmp273-.Ltmp272               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x68d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	72                              ; DW_AT_low_pc
	.long	.Ltmp280-.Ltmp279               ; DW_AT_high_pc
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
	.long	.Ltmp283-.Ltmp282               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6b7:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	75                              ; DW_AT_low_pc
	.long	.Ltmp285-.Ltmp284               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6c5:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	76                              ; DW_AT_low_pc
	.long	.Ltmp287-.Ltmp286               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6d3:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	77                              ; DW_AT_low_pc
	.long	.Ltmp288-.Ltmp287               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6e1:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	78                              ; DW_AT_low_pc
	.long	.Ltmp289-.Ltmp288               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6ef:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	79                              ; DW_AT_low_pc
	.long	.Ltmp291-.Ltmp290               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6fd:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	80                              ; DW_AT_low_pc
	.long	.Ltmp292-.Ltmp291               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x70b:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2059                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x719:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x727:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	81                              ; DW_AT_low_pc
	.long	.Ltmp294-.Ltmp293               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x737:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp296-.Ltmp295               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1876                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x745:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp296-.Ltmp295               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x753:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	82                              ; DW_AT_low_pc
	.long	.Ltmp296-.Ltmp295               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x763:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	34                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1898                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x76d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	35                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1900                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x777:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	36                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1902                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x781:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	37                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1904                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x78b:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1931                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x795:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x79f:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	38                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x7ab:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	39                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1994                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7b5:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	83                              ; DW_AT_low_pc
	.long	.Ltmp433-.Ltmp432               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1995                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7c3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	84                              ; DW_AT_low_pc
	.long	.Ltmp434-.Ltmp433               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1996                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7d1:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	40                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2017                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7db:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	85                              ; DW_AT_low_pc
	.long	.Ltmp444-.Ltmp443               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2018                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7e9:0xa DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	41                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2012                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7f3:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	86                              ; DW_AT_low_pc
	.long	.Ltmp447-.Ltmp446               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2018                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x801:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	87                              ; DW_AT_low_pc
	.long	.Ltmp450-.Ltmp449               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2021                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	2                               ; Abbrev [2] 0x811:0x2 DW_TAG_subprogram
	.byte	19                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x813:0x22a DW_TAG_subprogram
	.byte	88                              ; DW_AT_low_pc
	.long	.Lfunc_end6-.Lfunc_begin6       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	26                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x81a:0x222 DW_TAG_inlined_subroutine
	.long	2065                            ; DW_AT_abstract_origin
	.byte	42                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2162                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x824:0x19 DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	43                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x82e:0xe DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	89                              ; DW_AT_low_pc
	.long	.Ltmp459-.Ltmp458               ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.short	522                             ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x83d:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	44                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x847:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	45                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x851:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	46                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x85b:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	90                              ; DW_AT_low_pc
	.long	.Ltmp471-.Ltmp470               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x869:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	91                              ; DW_AT_low_pc
	.long	.Ltmp479-.Ltmp478               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1812                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x877:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	92                              ; DW_AT_low_pc
	.long	.Ltmp480-.Ltmp479               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1817                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x885:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	47                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x88f:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	93                              ; DW_AT_low_pc
	.long	.Ltmp483-.Ltmp482               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x89d:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	94                              ; DW_AT_low_pc
	.long	.Ltmp492-.Ltmp491               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1813                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8ab:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	95                              ; DW_AT_low_pc
	.long	.Ltmp493-.Ltmp492               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1818                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8b9:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	96                              ; DW_AT_low_pc
	.long	.Ltmp497-.Ltmp496               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8c7:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	97                              ; DW_AT_low_pc
	.long	.Ltmp503-.Ltmp502               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1819                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8d5:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	98                              ; DW_AT_low_pc
	.long	.Ltmp504-.Ltmp503               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1814                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8e3:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	99                              ; DW_AT_low_pc
	.long	.Ltmp507-.Ltmp506               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8f1:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	100                             ; DW_AT_low_pc
	.long	.Ltmp511-.Ltmp510               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1820                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8ff:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	101                             ; DW_AT_low_pc
	.long	.Ltmp512-.Ltmp511               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1815                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x90d:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	102                             ; DW_AT_low_pc
	.long	.Ltmp513-.Ltmp512               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x91b:0xe DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	103                             ; DW_AT_low_pc
	.long	.Ltmp516-.Ltmp515               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1821                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x929:0xe DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	104                             ; DW_AT_low_pc
	.long	.Ltmp517-.Ltmp516               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1816                            ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x937:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp519-.Ltmp518               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2059                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x945:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp519-.Ltmp518               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x953:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	105                             ; DW_AT_low_pc
	.long	.Ltmp519-.Ltmp518               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x963:0x2c DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp521-.Ltmp520               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1876                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x971:0x1d DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp521-.Ltmp520               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x97f:0xe DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	106                             ; DW_AT_low_pc
	.long	.Ltmp521-.Ltmp520               ; DW_AT_high_pc
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x98f:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	48                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1898                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x999:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	49                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1900                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9a3:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	50                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1902                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9ad:0xa DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	51                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1904                            ; DW_AT_call_line
	.byte	49                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9b7:0x20 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1931                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9c1:0x15 DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	721                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9cb:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	52                              ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	718                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x9d7:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	53                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1994                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9e1:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	107                             ; DW_AT_low_pc
	.long	.Ltmp612-.Ltmp611               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1995                            ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x9ef:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	108                             ; DW_AT_low_pc
	.long	.Ltmp613-.Ltmp612               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1996                            ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x9fd:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	54                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2017                            ; DW_AT_call_line
	.byte	28                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa07:0xe DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	109                             ; DW_AT_low_pc
	.long	.Ltmp623-.Ltmp622               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2018                            ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa15:0xa DW_TAG_inlined_subroutine
	.long	1509                            ; DW_AT_abstract_origin
	.byte	55                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2012                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa1f:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	110                             ; DW_AT_low_pc
	.long	.Ltmp626-.Ltmp625               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2018                            ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa2d:0xe DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	111                             ; DW_AT_low_pc
	.long	.Ltmp629-.Ltmp628               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2021                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	3                               ; Abbrev [3] 0xa3d:0x20 DW_TAG_subprogram
	.byte	112                             ; DW_AT_low_pc
	.long	.Lfunc_end7-.Lfunc_begin7       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	27                              ; DW_AT_name
	.byte	6                               ; Abbrev [6] 0xa44:0xa DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	56                              ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2190                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa4e:0xe DW_TAG_inlined_subroutine
	.long	1473                            ; DW_AT_abstract_origin
	.byte	113                             ; DW_AT_low_pc
	.long	.Ltmp637-.Ltmp636               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	2199                            ; DW_AT_call_line
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
	.long	57                              ; Offset entry count
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
	.uleb128 .Ltmp241-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp242-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp243-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp244-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp245-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp451-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges29:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp246-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp247-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp248-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp249-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp251-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp255-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp256-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp258-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp259-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges30:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp249-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp250-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp254-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp255-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp257-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp258-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp265-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp266-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges31:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp252-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp253-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp271-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp281-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp282-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges32:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp261-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp262-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp273-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp274-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp276-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp278-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp279-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp285-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp286-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges33:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp267-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp268-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp269-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp270-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp275-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp276-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp277-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp278-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges34:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp297-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp298-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp305-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp306-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp313-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp314-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp321-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp322-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp329-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp330-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp337-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp338-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp345-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp346-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp353-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp354-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp361-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp362-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp368-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp369-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp375-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp376-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp383-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp384-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp391-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp392-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp399-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp400-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp407-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp408-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp415-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp416-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges35:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp299-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp300-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp307-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp308-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp315-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp316-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp323-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp324-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp331-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp332-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp339-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp340-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp347-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp348-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp355-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp356-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp363-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp364-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp370-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp377-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp378-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp385-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp386-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp393-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp394-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp401-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp402-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp409-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp410-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp417-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp418-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges36:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp301-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp302-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp309-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp310-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp317-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp318-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp325-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp326-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp333-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp334-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp341-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp342-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp349-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp350-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp357-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp358-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp364-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp365-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp371-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp372-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp379-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp380-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp387-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp388-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp395-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp396-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp403-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp404-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp411-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp412-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp419-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp420-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges37:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp303-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp304-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp311-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp312-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp319-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp320-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp327-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp328-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp335-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp336-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp343-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp344-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp351-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp352-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp359-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp360-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp366-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp367-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp373-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp374-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp381-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp382-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp389-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp390-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp397-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp398-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp405-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp406-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp413-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp414-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp421-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp422-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges38:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp423-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp424-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp425-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp426-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges39:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp427-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp428-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp429-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp430-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp431-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp432-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges40:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp435-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp436-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp437-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp438-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp439-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp440-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp441-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp442-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges41:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp445-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp446-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp447-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp448-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges42:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp453-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp454-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp455-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp456-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp457-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp467-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp468-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp473-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp474-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp475-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp476-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp477-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp478-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp484-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp485-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp490-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp491-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp494-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp495-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp497-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp498-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp499-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp500-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp501-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp502-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp504-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp505-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp630-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges43:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp458-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp459-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp460-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp461-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp462-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp463-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp464-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp465-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp469-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp470-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges44:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp461-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp462-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp486-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp487-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp488-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp489-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp508-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp509-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges45:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp463-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp464-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp489-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp490-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp493-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp494-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp498-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp499-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp513-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp514-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges46:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp466-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp467-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp468-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp469-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp472-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp473-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp474-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp475-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp482-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges47:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp480-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp481-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp483-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp484-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp485-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp486-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp487-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp488-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp495-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp496-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges48:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp522-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp523-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp524-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp525-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp532-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp533-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp534-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp535-.Lfunc_begin0         ;   ending offset
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
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges49:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp523-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp524-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp525-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp526-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp533-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp534-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp535-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp536-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp543-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp544-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp545-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp546-.Lfunc_begin0         ;   ending offset
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
.Ldebug_ranges50:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp527-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp528-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp529-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp530-.Lfunc_begin0         ;   ending offset
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
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges51:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp528-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp529-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp530-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp531-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp538-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp539-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp540-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp541-.Lfunc_begin0         ;   ending offset
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
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp598-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp599-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp600-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp601-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges52:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp602-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp603-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp604-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp605-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges53:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp606-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp607-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp608-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp609-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp610-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp611-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges54:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp614-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp615-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp616-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp617-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp618-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp619-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp620-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp621-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges55:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp624-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp625-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp626-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp627-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges56:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp632-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp633-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp634-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp635-.Lfunc_begin0         ;   ending offset
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
	.asciz	"/home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt2" ; string offset=201 ; /home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt2
.Linfo_string3:
	.asciz	"fa2_stageb_nbody<false>"       ; string offset=269 ; fa2_stageb_nbody<false>
.Linfo_string4:
	.asciz	"__lane_id"                     ; string offset=293 ; __lane_id
.Linfo_string5:
	.asciz	"__shfl_xor"                    ; string offset=303 ; __shfl_xor
.Linfo_string6:
	.asciz	"max"                           ; string offset=314 ; max
.Linfo_string7:
	.asciz	"min"                           ; string offset=318 ; min
.Linfo_string8:
	.asciz	"__work_group_barrier"          ; string offset=322 ; __work_group_barrier
.Linfo_string9:
	.asciz	"__barrier"                     ; string offset=343 ; __barrier
.Linfo_string10:
	.asciz	"__syncthreads"                 ; string offset=353 ; __syncthreads
.Linfo_string11:
	.asciz	"fa2_scale_n"                   ; string offset=367 ; fa2_scale_n
.Linfo_string12:
	.asciz	"fmaxf"                         ; string offset=379 ; fmaxf
.Linfo_string13:
	.asciz	"__hip_get_thread_idx_x"        ; string offset=385 ; __hip_get_thread_idx_x
.Linfo_string14:
	.asciz	"__get_x"                       ; string offset=408 ; __get_x
.Linfo_string15:
	.asciz	"fa2_stageb_nbody<true>"        ; string offset=416 ; fa2_stageb_nbody<true>
.Linfo_string16:
	.asciz	"__expf"                        ; string offset=439 ; __expf
.Linfo_string17:
	.asciz	"fa2_stageb_packet_body<false>" ; string offset=446 ; fa2_stageb_packet_body<false>
.Linfo_string18:
	.asciz	"fa2_pkt_xor16"                 ; string offset=476 ; fa2_pkt_xor16
.Linfo_string19:
	.asciz	"fa2_stageb_packet_body<true>"  ; string offset=490 ; fa2_stageb_packet_body<true>
.Linfo_string20:
	.asciz	"attention_fp8_e4m3_fa2_gqa_gfx1201" ; string offset=519 ; attention_fp8_e4m3_fa2_gqa_gfx1201
.Linfo_string21:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201" ; string offset=554 ; attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
.Linfo_string22:
	.asciz	"attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201" ; string offset=602 ; attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201
.Linfo_string23:
	.asciz	"attention_fp8_e4m3_fa2_gqa_partial_gfx1201" ; string offset=650 ; attention_fp8_e4m3_fa2_gqa_partial_gfx1201
.Linfo_string24:
	.asciz	"attention_fp8_e4m3_fa2_gqa_merge_gfx1201" ; string offset=693 ; attention_fp8_e4m3_fa2_gqa_merge_gfx1201
.Linfo_string25:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_gfx1201" ; string offset=734 ; attention_fp8_e4m3_fa2_gqa_packet_gfx1201
.Linfo_string26:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201" ; string offset=776 ; attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201
.Linfo_string27:
	.asciz	"attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201" ; string offset=826 ; attention_fp8_e4m3_fa2_gqa_packet_merge_gfx1201
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
	.quad	.Ltmp246
	.quad	.Ltmp259
	.quad	.Ltmp263
	.quad	.Ltmp264
	.quad	.Ltmp266
	.quad	.Ltmp271
	.quad	.Ltmp272
	.quad	.Ltmp279
	.quad	.Ltmp280
	.quad	.Ltmp282
	.quad	.Ltmp284
	.quad	.Ltmp286
	.quad	.Ltmp287
	.quad	.Ltmp288
	.quad	.Ltmp290
	.quad	.Ltmp291
	.quad	.Ltmp293
	.quad	.Ltmp295
	.quad	.Ltmp432
	.quad	.Ltmp433
	.quad	.Ltmp443
	.quad	.Ltmp446
	.quad	.Ltmp449
	.quad	.Lfunc_begin6
	.quad	.Ltmp458
	.quad	.Ltmp470
	.quad	.Ltmp478
	.quad	.Ltmp479
	.quad	.Ltmp482
	.quad	.Ltmp491
	.quad	.Ltmp492
	.quad	.Ltmp496
	.quad	.Ltmp502
	.quad	.Ltmp503
	.quad	.Ltmp506
	.quad	.Ltmp510
	.quad	.Ltmp511
	.quad	.Ltmp512
	.quad	.Ltmp515
	.quad	.Ltmp516
	.quad	.Ltmp518
	.quad	.Ltmp520
	.quad	.Ltmp611
	.quad	.Ltmp612
	.quad	.Ltmp622
	.quad	.Ltmp625
	.quad	.Ltmp628
	.quad	.Lfunc_begin7
	.quad	.Ltmp636
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d74eb5f92d393693
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
    .private_segment_fixed_size: 20
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 4
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
    .private_segment_fixed_size: 12
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         attention_fp8_e4m3_fa2_gqa_packet_partial_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 2
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
