	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._Z3runILi0ELi1EEvPfPji,"axG",@progbits,_Z3runILi0ELi1EEvPfPji,comdat
	.protected	_Z3runILi0ELi1EEvPfPji  ; -- Begin function _Z3runILi0ELi1EEvPfPji
	.globl	_Z3runILi0ELi1EEvPfPji
	.p2align	8
	.type	_Z3runILi0ELi1EEvPfPji,@function
_Z3runILi0ELi1EEvPfPji:                 ; @_Z3runILi0ELi1EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v13, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v13
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v1, ttmp9, 3, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s6, v1
	v_add_co_ci_u32_e64 v2, null, s7, v2, vcc_lo
	global_store_b32 v[1:2], v3, off
.LBB0_2:                                ; %.preheader75
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	v_lshl_add_u32 v14, v0, 2, 0
	ds_store_b32 v14, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_7
; %bb.3:                                ; %.preheader.i.preheader.preheader
	v_mov_b32_e32 v9, 0x10101010
	v_mov_b32_e32 v1, 0
	s_mov_b32 s1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v11, 0x18181818
	v_mov_b32_e32 v8, v1
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_mov_b32_e32 v12, v11
.LBB0_4:                                ; %.preheader.i.preheader
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
	s_mov_b32 s2, 32
.LBB0_5:                                ; %.preheader.i
                                        ;   Parent Loop BB0_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[11:12], v[9:10], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %_Z6matrixILi1EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB0_4 Depth=1
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s1, s0
	s_cbranch_scc1 .LBB0_4
	s_branch .LBB0_8
.LBB0_7:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
.LBB0_8:                                ; %.preheader70
	s_mov_b32 s0, 0x3a83126f
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, 0, v1
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v0, ttmp9, 8, v0
	v_add_f32_e32 v1, v1, v2
	v_cvt_f32_ubyte0_e32 v2, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v3
	v_fmaak_f32 v3, s0, v2, 0x3dcccccd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v1, v1, v4 :: v_dual_fmaak_f32 v4, s0, v2, 0x3e4ccccd
	v_add_f32_e32 v1, v1, v5
	v_fmaak_f32 v5, s0, v2, 0x3ecccccd
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v6
	v_add_f32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v8
	v_add_f32_e32 v1, 1.0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8020c5, v1
	v_add_f32_e32 v1, 0x3f804189, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f80624e, v1
	v_add_f32_e32 v1, 0x3f808312, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f80a3d7, v1
	v_add_f32_e32 v1, 0x3f80c49c, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f80e560, v1
	v_add_f32_e32 v1, 0x3f8147ae, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f816873, v1
	v_add_f32_e32 v1, 0x3f818937, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f81a9fc, v1
	v_add_f32_e32 v1, 0x3f81cac0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f81eb85, v1
	v_add_f32_e32 v1, 0x3f820c4a, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f822d0e, v1
	v_add_f32_e32 v1, 0x3f828f5c, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f82b021, v1
	v_add_f32_e32 v1, 0x3f82d0e5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f82f1aa, v1
	v_add_f32_e32 v1, 0x3f83126e, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f833333, v1
	v_add_f32_e32 v1, 0x3f8353f8, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8374bc, v1
	v_add_f32_e32 v1, 0x3f83d70a, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f83f7cf, v1
	v_add_f32_e32 v1, 0x3f841893, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f843958, v1
	v_add_f32_e32 v1, 0x3f845a1c, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f847ae1, v1
	v_add_f32_e32 v1, 0x3f849ba6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f84bc6a, v1
	v_add_f32_e32 v1, 0x3f851eb8, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f853f7d, v1
	v_add_f32_e32 v1, 0x3f856041, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f858106, v1
	v_add_f32_e32 v1, 0x3f85a1ca, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f85c28f, v1
	v_add_f32_e32 v1, 0x3f85e354, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f860418, v1
	v_add_f32_e32 v1, 0x3f866666, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f86872b, v1
	v_add_f32_e32 v1, 0x3f86a7ef, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f86c8b4, v1
	v_add_f32_e32 v1, 0x3f86e978, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f870a3d, v1
	v_add_f32_e32 v1, 0x3f872b02, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f874bc6, v1
	v_add_f32_e32 v1, 0x3f87ae14, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f87ced9, v1
	v_add_f32_e32 v1, 0x3f87ef9d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f881062, v1
	v_add_f32_e32 v1, 0x3f883126, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8851eb, v1
	v_add_f32_e32 v1, 0x3f8872b0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f889374, v1
	v_add_f32_e32 v1, 0x3f88f5c3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f891688, v1
	v_add_f32_e32 v1, 0x3f89374c, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f895811, v1
	v_add_f32_e32 v1, 0x3f8978d5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f89999a, v1
	v_add_f32_e32 v1, 0x3f89ba5f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f89db23, v1
	v_add_f32_e32 v1, 0x3f8a3d71, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8a5e36, v1
	v_add_f32_e32 v1, 0x3f8a7efa, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8a9fbf, v1
	v_add_f32_e32 v1, 0x3f8ac083, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ae148, v1
	v_add_f32_e32 v1, 0x3f8b020d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8b22d1, v1
	v_add_f32_e32 v1, 0x3f8b851f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ba5e4, v1
	v_add_f32_e32 v1, 0x3f8bc6a8, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8be76d, v1
	v_add_f32_e32 v1, 0x3f8c0831, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8c28f6, v1
	v_add_f32_e32 v1, 0x3f8c49bb, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8c6a7f, v1
	v_add_f32_e32 v1, 0x3f8ccccd, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ced92, v1
	v_add_f32_e32 v1, 0x3f8d0e56, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8d2f1b, v1
	v_add_f32_e32 v1, 0x3f8d4fdf, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8d70a4, v1
	v_add_f32_e32 v1, 0x3f8d9169, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8db22d, v1
	v_add_f32_e32 v1, 0x3f8e147b, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8e3540, v1
	v_add_f32_e32 v1, 0x3f8e5604, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8e76c9, v1
	v_add_f32_e32 v1, 0x3f8e978d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8eb852, v1
	v_add_f32_e32 v1, 0x3f8ed917, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ef9db, v1
	v_add_f32_e32 v1, 0x3f8f5c29, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8f7cee, v1
	v_add_f32_e32 v1, 0x3f8f9db2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8fbe77, v1
	v_add_f32_e32 v1, 0x3f8fdf3b, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f900000, v1
	v_add_f32_e32 v1, 0x3f9020c5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f904189, v1
	v_add_f32_e32 v1, 0x3f90a3d7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f90c49c, v1
	v_add_f32_e32 v1, 0x3f90e560, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f910625, v1
	v_add_f32_e32 v1, 0x3f9126e9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f9147ae, v1
	v_add_f32_e32 v1, 0x3f916873, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f918937, v1
	v_add_f32_e32 v1, 0x3f91eb85, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f920c4a, v1
	v_add_f32_e32 v1, 0x3f922d0e, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f924dd3, v1
	v_add_f32_e32 v1, 0x3f926e97, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f928f5c, v1
	v_add_f32_e32 v1, 0x3f92b021, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f92d0e5, v1
	v_add_f32_e32 v1, 0x3f933333, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f9353f8, v1
	v_add_f32_e32 v1, 0x3f9374bc, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f939581, v1
	v_add_f32_e32 v1, 0x3f93b645, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f93d70a, v1
	v_add_f32_e32 v1, 0x3f93f7cf, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f941893, v1
	v_fmac_f32_e32 v1, 0x3a83126f, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v3
	v_fmaak_f32 v3, s0, v2, 0x3e99999a
	v_add_f32_e32 v1, v1, v4
	ds_load_b32 v4, v14
	v_add_f32_e32 v1, v1, v3
	v_fma_f32 v3, 0x3a83126f, v2, 0.5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v5
	v_fmaak_f32 v5, s0, v2, 0x3f19999a
	v_dual_fmaak_f32 v2, s0, v2, 0x3f333333 :: v_dual_add_f32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v3, v1, v5
	v_mov_b32_e32 v1, 0
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v4, v4
	v_add_f32_e32 v2, v3, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v4, 0x33800000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_endpgm
.Lfunc_end0:
	.size	_Z3runILi0ELi1EEvPfPji, .Lfunc_end0-_Z3runILi0ELi1EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi0ELi1EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 15
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z3runILi0ELi1EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi0ELi1EEvPfPji,"axG",@progbits,_Z3runILi0ELi1EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi0ELi1EEvPfPji.num_vgpr, 15
	.set .L_Z3runILi0ELi1EEvPfPji.num_agpr, 0
	.set .L_Z3runILi0ELi1EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi0ELi1EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi0ELi1EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi0ELi1EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi0ELi1EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi0ELi1EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi0ELi1EEvPfPji.has_recursion, 0
	.set .L_Z3runILi0ELi1EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1864
; TotalNumSgprs: 10
; NumVgprs: 15
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 15
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi1ELi1EEvPfPji,"axG",@progbits,_Z3runILi1ELi1EEvPfPji,comdat
	.protected	_Z3runILi1ELi1EEvPfPji  ; -- Begin function _Z3runILi1ELi1EEvPfPji
	.globl	_Z3runILi1ELi1EEvPfPji
	.p2align	8
	.type	_Z3runILi1ELi1EEvPfPji,@function
_Z3runILi1ELi1EEvPfPji:                 ; @_Z3runILi1ELi1EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v2, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v2
	s_cbranch_execz .LBB1_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v3, ttmp9, 3, v1
	v_mov_b32_e32 v1, s3
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, s6, v3
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	global_store_b32 v[3:4], v1, off
.LBB1_2:                                ; %.preheader72
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	v_lshl_add_u32 v1, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v2, v2
	s_mov_b32 s1, 0x3a83126f
	ds_store_b32 v1, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v132, 0x3a83126f, v2
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v134, s1, v2, 0x3dcccccd
	v_fmaak_f32 v138, s1, v2, 0x3e4ccccd
	v_fmaak_f32 v133, s1, v2, 0x3e99999a
	v_fmaak_f32 v137, s1, v2, 0x3ecccccd
	v_fma_f32 v139, 0x3a83126f, v2, 0.5
	v_fmaak_f32 v135, s1, v2, 0x3f19999a
	v_fmaak_f32 v136, s1, v2, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_5
; %bb.3:                                ; %.lr.ph
	v_mbcnt_lo_u32_b32 v9, -1, 0
	v_dual_mov_b32 v2, 0x3f941893 :: v_dual_mov_b32 v129, 1.0
	v_dual_mov_b32 v4, 0x3f93d70a :: v_dual_mov_b32 v131, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_xor_b32_e32 v10, 16, v9
	v_mov_b32_e32 v6, 0x3f939581
	v_mov_b32_e32 v7, 0x3f9374bc
	v_mov_b32_e32 v8, 0x3f9353f8
	v_mov_b32_e32 v11, 0x3f92b021
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	v_mov_b32_e32 v3, 0x3f93f7cf
	v_mov_b32_e32 v12, 0x3f928f5c
	v_mov_b32_e32 v13, 0x3f926e97
	v_mov_b32_e32 v14, 0x3f924dd3
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v130, v9, v10 :: v_dual_mov_b32 v5, 0x3f93b645
	v_mov_b32_e32 v9, 0x3f933333
	v_mov_b32_e32 v10, 0x3f92d0e5
	v_mov_b32_e32 v15, 0x3f922d0e
	v_mov_b32_e32 v16, 0x3f920c4a
	v_mov_b32_e32 v17, 0x3f91eb85
	v_mov_b32_e32 v18, 0x3f918937
	v_mov_b32_e32 v19, 0x3f916873
	v_mov_b32_e32 v20, 0x3f9147ae
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v22, 0x3f910625
	v_mov_b32_e32 v23, 0x3f90e560
	v_mov_b32_e32 v24, 0x3f90c49c
	v_mov_b32_e32 v25, 0x3f90a3d7
	v_mov_b32_e32 v26, 0x3f904189
	v_mov_b32_e32 v27, 0x3f9020c5
	v_mov_b32_e32 v28, 0x3f900000
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v30, 0x3f8fbe77
	v_mov_b32_e32 v31, 0x3f8f9db2
	v_mov_b32_e32 v32, 0x3f8f7cee
	v_mov_b32_e32 v33, 0x3f8f5c29
	v_mov_b32_e32 v34, 0x3f8ef9db
	v_mov_b32_e32 v35, 0x3f8ed917
	v_mov_b32_e32 v36, 0x3f8eb852
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v38, 0x3f8e76c9
	v_mov_b32_e32 v39, 0x3f8e5604
	v_mov_b32_e32 v40, 0x3f8e3540
	v_mov_b32_e32 v41, 0x3f8e147b
	v_mov_b32_e32 v42, 0x3f8db22d
	v_mov_b32_e32 v43, 0x3f8d9169
	v_mov_b32_e32 v44, 0x3f8d70a4
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v46, 0x3f8d2f1b
	v_mov_b32_e32 v47, 0x3f8d0e56
	v_mov_b32_e32 v48, 0x3f8ced92
	v_mov_b32_e32 v49, 0x3f8ccccd
	v_mov_b32_e32 v50, 0x3f8c6a7f
	v_mov_b32_e32 v51, 0x3f8c49bb
	v_mov_b32_e32 v52, 0x3f8c28f6
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v54, 0x3f8be76d
	v_mov_b32_e32 v55, 0x3f8bc6a8
	v_mov_b32_e32 v56, 0x3f8ba5e4
	v_mov_b32_e32 v57, 0x3f8b851f
	v_mov_b32_e32 v58, 0x3f8b22d1
	v_mov_b32_e32 v59, 0x3f8b020d
	v_mov_b32_e32 v60, 0x3f8ae148
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v62, 0x3f8a9fbf
	v_mov_b32_e32 v63, 0x3f8a7efa
	v_mov_b32_e32 v64, 0x3f8a5e36
	v_mov_b32_e32 v65, 0x3f8a3d71
	v_mov_b32_e32 v66, 0x3f89db23
	v_mov_b32_e32 v67, 0x3f89ba5f
	v_mov_b32_e32 v68, 0x3f89999a
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v70, 0x3f895811
	v_mov_b32_e32 v71, 0x3f89374c
	v_mov_b32_e32 v72, 0x3f891688
	v_mov_b32_e32 v73, 0x3f88f5c3
	v_mov_b32_e32 v74, 0x3f889374
	v_mov_b32_e32 v75, 0x3f8872b0
	v_mov_b32_e32 v76, 0x3f8851eb
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v78, 0x3f881062
	v_mov_b32_e32 v79, 0x3f87ef9d
	v_mov_b32_e32 v80, 0x3f87ced9
	v_mov_b32_e32 v81, 0x3f87ae14
	v_mov_b32_e32 v82, 0x3f874bc6
	v_mov_b32_e32 v83, 0x3f872b02
	v_mov_b32_e32 v84, 0x3f870a3d
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v86, 0x3f86c8b4
	v_mov_b32_e32 v87, 0x3f86a7ef
	v_mov_b32_e32 v88, 0x3f86872b
	v_mov_b32_e32 v89, 0x3f866666
	v_mov_b32_e32 v90, 0x3f860418
	v_mov_b32_e32 v91, 0x3f85e354
	v_mov_b32_e32 v92, 0x3f85c28f
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v94, 0x3f858106
	v_mov_b32_e32 v95, 0x3f856041
	v_mov_b32_e32 v96, 0x3f853f7d
	v_mov_b32_e32 v97, 0x3f851eb8
	v_mov_b32_e32 v98, 0x3f84bc6a
	v_mov_b32_e32 v99, 0x3f849ba6
	v_mov_b32_e32 v100, 0x3f847ae1
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v102, 0x3f843958
	v_mov_b32_e32 v103, 0x3f841893
	v_mov_b32_e32 v104, 0x3f83f7cf
	v_mov_b32_e32 v105, 0x3f83d70a
	v_mov_b32_e32 v106, 0x3f8374bc
	v_mov_b32_e32 v107, 0x3f8353f8
	v_mov_b32_e32 v108, 0x3f833333
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v110, 0x3f82f1aa
	v_mov_b32_e32 v111, 0x3f82d0e5
	v_mov_b32_e32 v112, 0x3f82b021
	v_mov_b32_e32 v113, 0x3f828f5c
	v_mov_b32_e32 v114, 0x3f822d0e
	v_mov_b32_e32 v115, 0x3f820c4a
	v_mov_b32_e32 v116, 0x3f81eb85
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v118, 0x3f81a9fc
	v_mov_b32_e32 v119, 0x3f818937
	v_mov_b32_e32 v120, 0x3f816873
	v_mov_b32_e32 v121, 0x3f8147ae
	v_mov_b32_e32 v122, 0x3f80e560
	v_mov_b32_e32 v123, 0x3f80c49c
	v_mov_b32_e32 v124, 0x3f80a3d7
	v_mov_b32_e32 v125, 0x3f808312
	v_mov_b32_e32 v126, 0x3f80624e
	v_mov_b32_e32 v127, 0x3f804189
	v_mov_b32_e32 v128, 0x3f8020c5
	v_lshlrev_b32_e32 v130, 2, v130
	s_mov_b32 s1, 0x35800000
.LBB1_4:                                ; =>This Inner Loop Header: Depth=1
	v_max3_num_f32 v140, v132, 0xff800000, v134
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v140, v140, v138, v133
	v_max3_num_f32 v140, v140, v137, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v140, v140, v135, v136
	ds_bpermute_b32 v141, v130, v140
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v141, v141, v141
	v_max_num_f32_e32 v140, v140, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v139, v139, v140
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_sub_f32 v132, v132, v140
	v_dual_sub_f32 v134, v134, v140 :: v_dual_sub_f32 v133, v133, v140
	v_dual_sub_f32 v138, v138, v140 :: v_dual_sub_f32 v137, v137, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v135, v135, v140 :: v_dual_mul_f32 v134, 0x3fb8aa3b, v134
	v_dual_mul_f32 v132, 0x3fb8aa3b, v132 :: v_dual_mul_f32 v133, 0x3fb8aa3b, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v138, 0x3fb8aa3b, v138 :: v_dual_mul_f32 v137, 0x3fb8aa3b, v137
	v_exp_f32_e32 v144, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v143, v132
	v_sub_f32_e32 v136, v136, v140
	v_exp_f32_e32 v145, v138
	v_exp_f32_e32 v146, v133
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v147, v137
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	v_exp_f32_e32 v148, v139
	v_mul_f32_e32 v133, 0x3b03126f, v144
	v_mul_f32_e32 v132, 0x3a83126f, v143
	v_exp_f32_e32 v149, v135
	v_exp_f32_e32 v150, v136
	v_mul_f32_e32 v134, 0x3b449ba6, v145
	v_mul_f32_e32 v135, 0x3b83126f, v146
	v_max3_num_f32 v136, v132, 0, v133
	v_add_f32_e32 v143, v143, v144
	v_mul_f32_e32 v137, 0x3bc49ba6, v148
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_max3_num_f32 v138, v136, v134, v135
	v_mul_f32_e32 v136, 0x3ba3d70b, v147
	v_mul_f32_e32 v139, 0x3c03126f, v150
	v_add_f32_e32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v140, v138, v136, v137
	v_dual_mul_f32 v138, 0x3be56042, v149 :: v_dual_add_f32 v143, v146, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v140, v140, v138, v139
	v_add_f32_e32 v143, v147, v143
	ds_bpermute_b32 v141, v130, v140
	v_add_f32_e32 v143, v148, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v143, v149, v143
	v_add_f32_e32 v143, v150, v143
	ds_bpermute_b32 v144, v130, v143
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v141, v141, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v140, v140, v141
	v_div_scale_f32 v141, null, 0x43e00000, 0x43e00000, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v142, v141
	s_wait_dscnt 0x0
	v_add_f32_e32 v143, v143, v144
	v_div_scale_f32 v144, null, v143, v143, v132
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v151, -v141, v142, 1.0
	v_rcp_f32_e32 v145, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v142, v151, v142
	v_div_scale_f32 v151, vcc_lo, v140, 0x43e00000, v140
	v_mul_f32_e32 v152, v151, v142
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v146, -v144, v145, 1.0
	v_fma_f32 v153, -v141, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v145, v146, v145 :: v_dual_fmac_f32 v152, v153, v142
	v_fma_f32 v141, -v141, v152, v151
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v141, v141, v142, v152
	v_div_fixup_f32 v140, v141, 0x43e00000, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v140, 0x1f800000, v140
	v_div_scale_f32 v141, null, v140, v140, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v142, v141
	v_fma_f32 v151, -v141, v142, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v142, v151, v142
	v_div_scale_f32 v151, vcc_lo, v132, v140, v132
	v_mul_f32_e32 v152, v151, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v153, -v141, v152, v151
	v_fmac_f32_e32 v152, v153, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v141, -v141, v152, v151
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v141, v141, v142, v152
	v_div_scale_f32 v142, null, v140, v140, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v151, v142
	v_fma_f32 v152, -v142, v151, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v133, v140, v133
	v_mul_f32_e32 v153, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v142, v153, v152
	v_fmac_f32_e32 v153, v154, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v142, -v142, v153, v152
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v142, v142, v151, v153
	v_div_scale_f32 v151, null, v140, v140, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v152, v151
	v_fma_f32 v153, -v151, v152, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v152, v153, v152
	v_div_scale_f32 v153, vcc_lo, v134, v140, v134
	v_mul_f32_e32 v154, v153, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v155, -v151, v154, v153
	v_fmac_f32_e32 v154, v155, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v151, v154, v153
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v151, v151, v152, v154
	v_div_scale_f32 v152, null, v140, v140, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v153, v152
	v_fma_f32 v154, -v152, v153, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v154, v153
	v_div_scale_f32 v154, vcc_lo, v135, v140, v135
	v_mul_f32_e32 v155, v154, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v152, v155, v154
	v_fmac_f32_e32 v155, v156, v153
	v_div_fixup_f32 v151, v151, v140, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v155, v154
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v153, v155
	v_div_scale_f32 v153, null, v140, v140, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v152, v152, v140, v135
	v_rcp_f32_e32 v154, v153
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v155, -v153, v154, 1.0
	v_fmac_f32_e32 v154, v155, v154
	v_div_scale_f32 v155, vcc_lo, v136, v140, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v155, v154
	v_fma_f32 v157, -v153, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v157, v154
	v_fma_f32 v153, -v153, v156, v155
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v153, v153, v154, v156
	v_div_scale_f32 v154, null, v140, v140, v137
	v_rcp_f32_e32 v155, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v154, v155, 1.0
	v_fmac_f32_e32 v155, v156, v155
	v_div_scale_f32 v156, vcc_lo, v137, v140, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v157, v156, v155
	v_fma_f32 v158, -v154, v157, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v157, v158, v155
	v_fma_f32 v154, -v154, v157, v156
	v_div_fixup_f32 v156, v142, v140, v133
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v154, v154, v155, v157
	v_div_fixup_f32 v155, v141, v140, v132
	v_mov_b16_e64 v141.l, v131.l
	v_mov_b16_e64 v141.h, 0
	v_mov_b16_e64 v142.l, v141.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v142.h, v141.h
	v_cvt_pk_fp8_f32 v142.l, v155, v156
	v_div_scale_f32 v155, null, v140, v140, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_pk_fp8_f32 v142.h, v151, v152
	v_rcp_f32_e32 v156, v155
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v155, v156, 1.0
	v_fmac_f32_e32 v156, v157, v156
	v_div_scale_f32 v157, vcc_lo, v138, v140, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v158, v157, v156
	v_fma_f32 v159, -v155, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v158, v159, v156
	v_fma_f32 v155, -v155, v158, v157
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v155, v155, v156, v158
	v_div_scale_f32 v156, null, v140, v140, v139
	v_rcp_f32_e32 v157, v156
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v156, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v139, v140, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v156, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v156, -v156, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v156, v156, v157, v159
	v_div_scale_f32 v146, vcc_lo, v132, v143, v132
	v_mul_f32_e32 v147, v146, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v148, -v144, v147, v146
	v_fmac_f32_e32 v147, v148, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v144, -v144, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v144, v144, v145, v147
	v_div_scale_f32 v145, null, v143, v143, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v146, v145
	v_fma_f32 v147, -v145, v146, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v146, v147, v146
	v_div_scale_f32 v147, vcc_lo, v133, v143, v133
	v_mul_f32_e32 v148, v147, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v145, v148, v147
	v_fmac_f32_e32 v148, v149, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v145, -v145, v148, v147
	v_div_fixup_f32 v147, v154, v140, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v145, v145, v146, v148
	v_div_fixup_f32 v146, v153, v140, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v141.l, v146, v147
	v_div_scale_f32 v146, null, v143, v143, v134
	v_rcp_f32_e32 v147, v146
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v148, -v146, v147, 1.0
	v_fmac_f32_e32 v147, v148, v147
	v_div_scale_f32 v148, vcc_lo, v134, v143, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v149, v148, v147
	v_fma_f32 v150, -v146, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v147
	v_fma_f32 v146, -v146, v149, v148
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v146, v146, v147, v149
	v_div_scale_f32 v147, null, v143, v143, v135
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v135, v143, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v148, v155, v140, v138
	v_div_fixup_f32 v140, v156, v140, v139
	v_cvt_pk_fp8_f32 v141.h, v148, v140
	v_div_scale_f32 v140, null, v143, v143, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v148, v140
	v_fma_f32 v149, -v140, v148, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v136, v143, v136
	v_mul_f32_e32 v150, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v140, v150, v149
	v_fmac_f32_e32 v150, v151, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v140, -v140, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v140, v140, v148, v150
	v_div_scale_f32 v148, null, v143, v143, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v149, v148
	v_fma_f32 v150, -v148, v149, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v149
	v_div_scale_f32 v150, vcc_lo, v137, v143, v137
	v_mul_f32_e32 v151, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v148, v151, v150
	v_fmac_f32_e32 v151, v152, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v148, -v148, v151, v150
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v148, v148, v149, v151
	v_div_scale_f32 v149, null, v143, v143, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v151, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v150
	v_div_scale_f32 v151, vcc_lo, v138, v143, v138
	v_mul_f32_e32 v152, v151, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v153, -v149, v152, v151
	v_fmac_f32_e32 v152, v153, v150
	v_div_fixup_f32 v132, v144, v143, v132
	v_div_fixup_f32 v144, v146, v143, v134
	v_xor_b32_e32 v134, v142, v141
	v_div_fixup_f32 v133, v145, v143, v133
	v_fma_f32 v149, -v149, v152, v151
	v_div_fixup_f32 v135, v147, v143, v135
	v_div_fixup_f32 v136, v140, v143, v136
	v_cvt_f32_ubyte0_e32 v134, v134
	v_div_fixup_f32 v140, v148, v143, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v152
	v_div_scale_f32 v150, null, v143, v143, v139
	v_add_f32_e32 v132, 0, v132
	v_fmaak_f32 v134, s1, v134, 0x3f7fbe77
	v_add_f32_e32 v137, 0x3d23d70a, v136
	v_div_fixup_f32 v145, v149, v143, v138
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v138, 0x3ca3d70a, v144 :: v_dual_mul_f32 v5, v134, v5
	v_mul_f32_e32 v126, v126, v134
	v_mul_f32_e32 v122, v122, v134
	v_dual_mul_f32 v128, v128, v134 :: v_dual_mul_f32 v129, v129, v134
	v_dual_mul_f32 v124, v124, v134 :: v_dual_mul_f32 v127, v127, v134
	v_mul_f32_e32 v120, v120, v134
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v152, -v150, v151, 1.0
	v_dual_mul_f32 v125, v125, v134 :: v_dual_mul_f32 v118, v118, v134
	v_dual_mul_f32 v123, v123, v134 :: v_dual_mul_f32 v116, v116, v134
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v139, v143, v139
	v_dual_mul_f32 v121, v121, v134 :: v_dual_mul_f32 v114, v114, v134
	v_dual_mul_f32 v119, v119, v134 :: v_dual_mul_f32 v112, v112, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v153, v152, v151
	v_dual_mul_f32 v117, v117, v134 :: v_dual_mul_f32 v110, v110, v134
	v_dual_mul_f32 v115, v115, v134 :: v_dual_mul_f32 v108, v108, v134
	v_fma_f32 v154, -v150, v153, v152
	v_dual_mul_f32 v113, v113, v134 :: v_dual_mul_f32 v106, v106, v134
	v_dual_mul_f32 v111, v111, v134 :: v_dual_mul_f32 v104, v104, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v153, v154, v151
	v_dual_mul_f32 v109, v109, v134 :: v_dual_mul_f32 v102, v102, v134
	v_dual_mul_f32 v107, v107, v134 :: v_dual_mul_f32 v100, v100, v134
	v_fma_f32 v150, -v150, v153, v152
	v_dual_mul_f32 v105, v105, v134 :: v_dual_mul_f32 v98, v98, v134
	v_dual_mul_f32 v103, v103, v134 :: v_dual_mul_f32 v96, v96, v134
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v150, v150, v151, v153
	v_dual_mul_f32 v101, v101, v134 :: v_dual_mul_f32 v94, v94, v134
	v_dual_mul_f32 v99, v99, v134 :: v_dual_mul_f32 v92, v92, v134
	v_div_fixup_f32 v143, v150, v143, v139
	v_dual_mul_f32 v97, v97, v134 :: v_dual_mul_f32 v90, v90, v134
	v_dual_mul_f32 v95, v95, v134 :: v_dual_mul_f32 v88, v88, v134
	v_dual_mul_f32 v93, v93, v134 :: v_dual_mul_f32 v86, v86, v134
	v_dual_mul_f32 v91, v91, v134 :: v_dual_mul_f32 v84, v84, v134
	v_dual_mul_f32 v89, v89, v134 :: v_dual_mul_f32 v82, v82, v134
	v_dual_mul_f32 v87, v87, v134 :: v_dual_mul_f32 v80, v80, v134
	v_dual_mul_f32 v85, v85, v134 :: v_dual_mul_f32 v78, v78, v134
	v_dual_mul_f32 v83, v83, v134 :: v_dual_mul_f32 v76, v76, v134
	v_dual_mul_f32 v81, v81, v134 :: v_dual_mul_f32 v74, v74, v134
	v_dual_mul_f32 v79, v79, v134 :: v_dual_mul_f32 v72, v72, v134
	v_dual_mul_f32 v77, v77, v134 :: v_dual_mul_f32 v70, v70, v134
	v_dual_mul_f32 v75, v75, v134 :: v_dual_mul_f32 v68, v68, v134
	v_dual_mul_f32 v73, v73, v134 :: v_dual_mul_f32 v66, v66, v134
	v_dual_mul_f32 v71, v71, v134 :: v_dual_mul_f32 v64, v64, v134
	v_dual_mul_f32 v69, v69, v134 :: v_dual_mul_f32 v62, v62, v134
	v_dual_mul_f32 v67, v67, v134 :: v_dual_mul_f32 v60, v60, v134
	v_dual_mul_f32 v65, v65, v134 :: v_dual_mul_f32 v58, v58, v134
	v_dual_mul_f32 v63, v63, v134 :: v_dual_mul_f32 v56, v56, v134
	v_dual_mul_f32 v61, v61, v134 :: v_dual_mul_f32 v54, v54, v134
	v_dual_mul_f32 v59, v59, v134 :: v_dual_mul_f32 v52, v52, v134
	v_dual_mul_f32 v57, v57, v134 :: v_dual_mul_f32 v50, v50, v134
	v_dual_mul_f32 v55, v55, v134 :: v_dual_mul_f32 v48, v48, v134
	v_dual_mul_f32 v53, v53, v134 :: v_dual_mul_f32 v46, v46, v134
	v_dual_mul_f32 v51, v51, v134 :: v_dual_mul_f32 v44, v44, v134
	v_dual_mul_f32 v49, v49, v134 :: v_dual_mul_f32 v42, v42, v134
	v_dual_mul_f32 v47, v47, v134 :: v_dual_mul_f32 v40, v40, v134
	v_dual_mul_f32 v45, v45, v134 :: v_dual_mul_f32 v38, v38, v134
	v_dual_mul_f32 v43, v43, v134 :: v_dual_mul_f32 v36, v36, v134
	v_dual_mul_f32 v41, v41, v134 :: v_dual_mul_f32 v34, v34, v134
	v_dual_mul_f32 v39, v39, v134 :: v_dual_mul_f32 v32, v134, v32
	v_dual_mul_f32 v37, v37, v134 :: v_dual_mul_f32 v28, v134, v28
	v_dual_mul_f32 v35, v35, v134 :: v_dual_mul_f32 v24, v134, v24
	v_dual_mul_f32 v33, v134, v33 :: v_dual_mul_f32 v30, v134, v30
	v_dual_mul_f32 v31, v134, v31 :: v_dual_mul_f32 v26, v134, v26
	v_dual_mul_f32 v29, v134, v29 :: v_dual_mul_f32 v22, v134, v22
	v_dual_mul_f32 v27, v134, v27 :: v_dual_mul_f32 v20, v134, v20
	v_dual_mul_f32 v25, v134, v25 :: v_dual_mul_f32 v18, v134, v18
	v_dual_mul_f32 v23, v134, v23 :: v_dual_mul_f32 v16, v134, v16
	v_dual_mul_f32 v21, v134, v21 :: v_dual_mul_f32 v14, v134, v14
	v_dual_mul_f32 v19, v134, v19 :: v_dual_mul_f32 v12, v134, v12
	v_dual_mul_f32 v17, v134, v17 :: v_dual_mul_f32 v10, v134, v10
	v_dual_mul_f32 v15, v134, v15 :: v_dual_mul_f32 v8, v134, v8
	v_dual_mul_f32 v13, v134, v13 :: v_dual_mul_f32 v6, v134, v6
	v_dual_mul_f32 v11, v134, v11 :: v_dual_mul_f32 v4, v134, v4
	v_dual_mul_f32 v9, v134, v9 :: v_dual_mul_f32 v2, v134, v2
	v_mul_f32_e32 v7, v134, v7
	v_dual_mul_f32 v3, v134, v3 :: v_dual_add_f32 v134, 0x3c23d70a, v133
	v_add_f32_e32 v133, 0x3cf5c28f, v135
	v_add_f32_e32 v139, 0x3d4ccccc, v140
	v_add_f32_e32 v135, 0x3d75c28f, v145
	v_add_f32_e32 v136, 0x3d8f5c29, v143
	s_cbranch_scc0 .LBB1_4
	s_branch .LBB1_6
.LBB1_5:
	v_dual_mov_b32 v129, 1.0 :: v_dual_mov_b32 v128, 0x3f8020c5
	v_mov_b32_e32 v127, 0x3f804189
	v_mov_b32_e32 v126, 0x3f80624e
	v_mov_b32_e32 v125, 0x3f808312
	v_mov_b32_e32 v124, 0x3f80a3d7
	v_mov_b32_e32 v123, 0x3f80c49c
	v_mov_b32_e32 v122, 0x3f80e560
	v_mov_b32_e32 v121, 0x3f8147ae
	v_mov_b32_e32 v120, 0x3f816873
	v_mov_b32_e32 v119, 0x3f818937
	v_mov_b32_e32 v118, 0x3f81a9fc
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v116, 0x3f81eb85
	v_mov_b32_e32 v115, 0x3f820c4a
	v_mov_b32_e32 v114, 0x3f822d0e
	v_mov_b32_e32 v113, 0x3f828f5c
	v_mov_b32_e32 v112, 0x3f82b021
	v_mov_b32_e32 v111, 0x3f82d0e5
	v_mov_b32_e32 v110, 0x3f82f1aa
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v108, 0x3f833333
	v_mov_b32_e32 v107, 0x3f8353f8
	v_mov_b32_e32 v106, 0x3f8374bc
	v_mov_b32_e32 v105, 0x3f83d70a
	v_mov_b32_e32 v104, 0x3f83f7cf
	v_mov_b32_e32 v103, 0x3f841893
	v_mov_b32_e32 v102, 0x3f843958
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v100, 0x3f847ae1
	v_mov_b32_e32 v99, 0x3f849ba6
	v_mov_b32_e32 v98, 0x3f84bc6a
	v_mov_b32_e32 v97, 0x3f851eb8
	v_mov_b32_e32 v96, 0x3f853f7d
	v_mov_b32_e32 v95, 0x3f856041
	v_mov_b32_e32 v94, 0x3f858106
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v92, 0x3f85c28f
	v_mov_b32_e32 v91, 0x3f85e354
	v_mov_b32_e32 v90, 0x3f860418
	v_mov_b32_e32 v89, 0x3f866666
	v_mov_b32_e32 v88, 0x3f86872b
	v_mov_b32_e32 v87, 0x3f86a7ef
	v_mov_b32_e32 v86, 0x3f86c8b4
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v84, 0x3f870a3d
	v_mov_b32_e32 v83, 0x3f872b02
	v_mov_b32_e32 v82, 0x3f874bc6
	v_mov_b32_e32 v81, 0x3f87ae14
	v_mov_b32_e32 v80, 0x3f87ced9
	v_mov_b32_e32 v79, 0x3f87ef9d
	v_mov_b32_e32 v78, 0x3f881062
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v76, 0x3f8851eb
	v_mov_b32_e32 v75, 0x3f8872b0
	v_mov_b32_e32 v74, 0x3f889374
	v_mov_b32_e32 v73, 0x3f88f5c3
	v_mov_b32_e32 v72, 0x3f891688
	v_mov_b32_e32 v71, 0x3f89374c
	v_mov_b32_e32 v70, 0x3f895811
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v68, 0x3f89999a
	v_mov_b32_e32 v67, 0x3f89ba5f
	v_mov_b32_e32 v66, 0x3f89db23
	v_mov_b32_e32 v65, 0x3f8a3d71
	v_mov_b32_e32 v64, 0x3f8a5e36
	v_mov_b32_e32 v63, 0x3f8a7efa
	v_mov_b32_e32 v62, 0x3f8a9fbf
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v60, 0x3f8ae148
	v_mov_b32_e32 v59, 0x3f8b020d
	v_mov_b32_e32 v58, 0x3f8b22d1
	v_mov_b32_e32 v57, 0x3f8b851f
	v_mov_b32_e32 v56, 0x3f8ba5e4
	v_mov_b32_e32 v55, 0x3f8bc6a8
	v_mov_b32_e32 v54, 0x3f8be76d
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v52, 0x3f8c28f6
	v_mov_b32_e32 v51, 0x3f8c49bb
	v_mov_b32_e32 v50, 0x3f8c6a7f
	v_mov_b32_e32 v49, 0x3f8ccccd
	v_mov_b32_e32 v48, 0x3f8ced92
	v_mov_b32_e32 v47, 0x3f8d0e56
	v_mov_b32_e32 v46, 0x3f8d2f1b
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v44, 0x3f8d70a4
	v_mov_b32_e32 v43, 0x3f8d9169
	v_mov_b32_e32 v42, 0x3f8db22d
	v_mov_b32_e32 v41, 0x3f8e147b
	v_mov_b32_e32 v40, 0x3f8e3540
	v_mov_b32_e32 v39, 0x3f8e5604
	v_mov_b32_e32 v38, 0x3f8e76c9
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v36, 0x3f8eb852
	v_mov_b32_e32 v35, 0x3f8ed917
	v_mov_b32_e32 v34, 0x3f8ef9db
	v_mov_b32_e32 v33, 0x3f8f5c29
	v_mov_b32_e32 v32, 0x3f8f7cee
	v_mov_b32_e32 v31, 0x3f8f9db2
	v_mov_b32_e32 v30, 0x3f8fbe77
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v28, 0x3f900000
	v_mov_b32_e32 v27, 0x3f9020c5
	v_mov_b32_e32 v26, 0x3f904189
	v_mov_b32_e32 v25, 0x3f90a3d7
	v_mov_b32_e32 v24, 0x3f90c49c
	v_mov_b32_e32 v23, 0x3f90e560
	v_mov_b32_e32 v22, 0x3f910625
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v20, 0x3f9147ae
	v_mov_b32_e32 v19, 0x3f916873
	v_mov_b32_e32 v18, 0x3f918937
	v_mov_b32_e32 v17, 0x3f91eb85
	v_mov_b32_e32 v16, 0x3f920c4a
	v_mov_b32_e32 v15, 0x3f922d0e
	v_mov_b32_e32 v14, 0x3f924dd3
	v_mov_b32_e32 v13, 0x3f926e97
	v_mov_b32_e32 v12, 0x3f928f5c
	v_mov_b32_e32 v11, 0x3f92b021
	v_mov_b32_e32 v10, 0x3f92d0e5
	v_mov_b32_e32 v9, 0x3f933333
	v_mov_b32_e32 v8, 0x3f9353f8
	v_mov_b32_e32 v7, 0x3f9374bc
	v_mov_b32_e32 v6, 0x3f939581
	v_mov_b32_e32 v5, 0x3f93b645
	v_mov_b32_e32 v4, 0x3f93d70a
	v_mov_b32_e32 v3, 0x3f93f7cf
	v_mov_b32_e32 v2, 0x3f941893
.LBB1_6:                                ; %Flow915
	v_add_f32_e32 v129, 0, v129
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v128, v129, v128
	v_add_f32_e32 v127, v128, v127
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v126, v127, v126
	v_add_f32_e32 v125, v126, v125
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v124, v125, v124
	v_add_f32_e32 v123, v124, v123
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v122, v123, v122
	v_add_f32_e32 v121, v122, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v120, v121, v120
	v_add_f32_e32 v119, v120, v119
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v118, v119, v118
	v_add_f32_e32 v117, v118, v117
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v116, v117, v116
	v_add_f32_e32 v115, v116, v115
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v114, v115, v114
	v_add_f32_e32 v113, v114, v113
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v112, v113, v112
	v_add_f32_e32 v111, v112, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v110, v111, v110
	v_add_f32_e32 v109, v110, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v108, v109, v108
	v_add_f32_e32 v107, v108, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v106, v107, v106
	v_add_f32_e32 v105, v106, v105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v104, v105, v104
	v_add_f32_e32 v103, v104, v103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v102, v103, v102
	v_add_f32_e32 v101, v102, v101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v100, v101, v100
	v_add_f32_e32 v99, v100, v99
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v98, v99, v98
	v_add_f32_e32 v97, v98, v97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v96, v97, v96
	v_add_f32_e32 v95, v96, v95
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v94, v95, v94
	v_add_f32_e32 v93, v94, v93
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v92, v93, v92
	v_add_f32_e32 v91, v92, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v90, v91, v90
	v_add_f32_e32 v89, v90, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v88, v89, v88
	v_add_f32_e32 v87, v88, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v86, v87, v86
	v_add_f32_e32 v85, v86, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v84, v85, v84
	v_add_f32_e32 v83, v84, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v82, v83, v82
	v_add_f32_e32 v81, v82, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v80, v81, v80
	v_add_f32_e32 v79, v80, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v78, v79, v78
	v_add_f32_e32 v77, v78, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v76, v77, v76
	v_add_f32_e32 v75, v76, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v74, v75, v74
	v_add_f32_e32 v73, v74, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v72, v73, v72
	v_add_f32_e32 v71, v72, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v70, v71, v70
	v_add_f32_e32 v69, v70, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v68, v69, v68
	v_add_f32_e32 v67, v68, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v66, v67, v66
	v_add_f32_e32 v65, v66, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v64, v65, v64
	v_add_f32_e32 v63, v64, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v62, v63, v62
	v_add_f32_e32 v61, v62, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v61, v60
	v_add_f32_e32 v59, v60, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v58, v59, v58
	v_add_f32_e32 v57, v58, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v56, v57, v56
	v_add_f32_e32 v55, v56, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v54, v55, v54
	v_add_f32_e32 v53, v54, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v52, v53, v52
	v_add_f32_e32 v51, v52, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v50, v51, v50
	v_add_f32_e32 v49, v50, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v48, v49, v48
	v_add_f32_e32 v47, v48, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v46, v47, v46
	v_add_f32_e32 v45, v46, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v44, v45, v44
	v_add_f32_e32 v43, v44, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v42, v43, v42
	v_add_f32_e32 v41, v42, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v40, v41, v40
	v_add_f32_e32 v39, v40, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v38, v39, v38
	v_add_f32_e32 v37, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v36, v37, v36
	v_add_f32_e32 v35, v36, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v34, v35, v34
	v_add_f32_e32 v33, v34, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v32, v33, v32
	v_add_f32_e32 v31, v32, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v30, v31, v30
	v_add_f32_e32 v29, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v28, v29, v28
	v_add_f32_e32 v27, v28, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v26, v27, v26
	v_add_f32_e32 v25, v26, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v24, v25, v24
	v_add_f32_e32 v23, v24, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v22, v23, v22
	v_add_f32_e32 v21, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v20, v21, v20
	v_add_f32_e32 v19, v20, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v19, v18
	v_add_f32_e32 v17, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v17, v16
	v_add_f32_e32 v15, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v15, v14
	v_add_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v13, v12
	v_add_f32_e32 v11, v12, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v11, v10
	v_add_f32_e32 v9, v10, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v8, v9, v8
	v_add_f32_e32 v7, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v7, v6
	v_add_f32_e32 v5, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v5, v4
	v_add_f32_e32 v3, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v3, v2
	ds_load_b32 v3, v1
	v_add_f32_e32 v2, v2, v132
	v_add_f32_e32 v2, v2, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v138
	v_add_f32_e32 v1, v2, v133
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v3, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v135 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v3, 0x33800000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end1:
	.size	_Z3runILi1ELi1EEvPfPji, .Lfunc_end1-_Z3runILi1ELi1EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi1ELi1EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 161
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_Z3runILi1ELi1EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi1ELi1EEvPfPji,"axG",@progbits,_Z3runILi1ELi1EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi1ELi1EEvPfPji.num_vgpr, 161
	.set .L_Z3runILi1ELi1EEvPfPji.num_agpr, 0
	.set .L_Z3runILi1ELi1EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi1ELi1EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi1ELi1EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi1ELi1EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi1ELi1EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi1ELi1EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi1ELi1EEvPfPji.has_recursion, 0
	.set .L_Z3runILi1ELi1EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5904
; TotalNumSgprs: 10
; NumVgprs: 161
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 161
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi2ELi1EEvPfPji,"axG",@progbits,_Z3runILi2ELi1EEvPfPji,comdat
	.protected	_Z3runILi2ELi1EEvPfPji  ; -- Begin function _Z3runILi2ELi1EEvPfPji
	.globl	_Z3runILi2ELi1EEvPfPji
	.p2align	8
	.type	_Z3runILi2ELi1EEvPfPji,@function
_Z3runILi2ELi1EEvPfPji:                 ; @_Z3runILi2ELi1EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB2_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB2_2:                                ; %.preheader77
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	v_lshl_add_u32 v157, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s1, 0x3a83126f
	ds_store_b32 v157, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v137, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v138, s1, v1, 0x3dcccccd
	v_fmaak_f32 v139, s1, v1, 0x3e4ccccd
	v_fmaak_f32 v140, s1, v1, 0x3e99999a
	v_fmaak_f32 v141, s1, v1, 0x3ecccccd
	v_fma_f32 v142, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v143, s1, v1, 0x3f19999a
	v_fmaak_f32 v144, s1, v1, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_7
; %bb.3:                                ; %.preheader.i.preheader.preheader
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v8, 0x3f941893
	v_dual_mov_b32 v6, 0x3f93d70a :: v_dual_mov_b32 v121, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v13, 0x3f926e97 :: v_dual_mov_b32 v146, v145
	v_dual_mov_b32 v147, v145 :: v_dual_mov_b32 v148, v145
	v_dual_mov_b32 v149, v145 :: v_dual_mov_b32 v150, v145
	v_dual_mov_b32 v151, v145 :: v_dual_mov_b32 v152, v145
	v_mov_b32_e32 v7, 0x3f93f7cf
	v_dual_mov_b32 v4, 0x3f939581 :: v_dual_mov_b32 v129, v145
	v_dual_mov_b32 v5, 0x3f93b645 :: v_dual_mov_b32 v130, v146
	v_dual_mov_b32 v3, 0x3f9374bc :: v_dual_mov_b32 v132, v148
	v_dual_mov_b32 v2, 0x3f9353f8 :: v_dual_mov_b32 v131, v147
	v_dual_mov_b32 v1, 0x3f933333 :: v_dual_mov_b32 v134, v150
	v_dual_mov_b32 v16, 0x3f92d0e5 :: v_dual_mov_b32 v133, v149
	v_dual_mov_b32 v15, 0x3f92b021 :: v_dual_mov_b32 v136, v152
	v_dual_mov_b32 v14, 0x3f928f5c :: v_dual_mov_b32 v135, v151
	v_mov_b32_e32 v12, 0x3f924dd3
	v_mov_b32_e32 v11, 0x3f922d0e
	v_mov_b32_e32 v10, 0x3f920c4a
	v_mov_b32_e32 v9, 0x3f91eb85
	v_mov_b32_e32 v24, 0x3f918937
	v_mov_b32_e32 v23, 0x3f916873
	v_mov_b32_e32 v22, 0x3f9147ae
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v20, 0x3f910625
	v_mov_b32_e32 v19, 0x3f90e560
	v_mov_b32_e32 v18, 0x3f90c49c
	v_mov_b32_e32 v17, 0x3f90a3d7
	v_mov_b32_e32 v32, 0x3f904189
	v_mov_b32_e32 v31, 0x3f9020c5
	v_mov_b32_e32 v30, 0x3f900000
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v28, 0x3f8fbe77
	v_mov_b32_e32 v27, 0x3f8f9db2
	v_mov_b32_e32 v26, 0x3f8f7cee
	v_mov_b32_e32 v25, 0x3f8f5c29
	v_mov_b32_e32 v40, 0x3f8ef9db
	v_mov_b32_e32 v39, 0x3f8ed917
	v_mov_b32_e32 v38, 0x3f8eb852
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v36, 0x3f8e76c9
	v_mov_b32_e32 v35, 0x3f8e5604
	v_mov_b32_e32 v34, 0x3f8e3540
	v_mov_b32_e32 v33, 0x3f8e147b
	v_mov_b32_e32 v48, 0x3f8db22d
	v_mov_b32_e32 v47, 0x3f8d9169
	v_mov_b32_e32 v46, 0x3f8d70a4
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v44, 0x3f8d2f1b
	v_mov_b32_e32 v43, 0x3f8d0e56
	v_mov_b32_e32 v42, 0x3f8ced92
	v_mov_b32_e32 v41, 0x3f8ccccd
	v_mov_b32_e32 v56, 0x3f8c6a7f
	v_mov_b32_e32 v55, 0x3f8c49bb
	v_mov_b32_e32 v54, 0x3f8c28f6
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v52, 0x3f8be76d
	v_mov_b32_e32 v51, 0x3f8bc6a8
	v_mov_b32_e32 v50, 0x3f8ba5e4
	v_mov_b32_e32 v49, 0x3f8b851f
	v_mov_b32_e32 v64, 0x3f8b22d1
	v_mov_b32_e32 v63, 0x3f8b020d
	v_mov_b32_e32 v62, 0x3f8ae148
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v60, 0x3f8a9fbf
	v_mov_b32_e32 v59, 0x3f8a7efa
	v_mov_b32_e32 v58, 0x3f8a5e36
	v_mov_b32_e32 v57, 0x3f8a3d71
	v_mov_b32_e32 v72, 0x3f89db23
	v_mov_b32_e32 v71, 0x3f89ba5f
	v_mov_b32_e32 v70, 0x3f89999a
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v68, 0x3f895811
	v_mov_b32_e32 v67, 0x3f89374c
	v_mov_b32_e32 v66, 0x3f891688
	v_mov_b32_e32 v65, 0x3f88f5c3
	v_mov_b32_e32 v80, 0x3f889374
	v_mov_b32_e32 v79, 0x3f8872b0
	v_mov_b32_e32 v78, 0x3f8851eb
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v76, 0x3f881062
	v_mov_b32_e32 v75, 0x3f87ef9d
	v_mov_b32_e32 v74, 0x3f87ced9
	v_mov_b32_e32 v73, 0x3f87ae14
	v_mov_b32_e32 v88, 0x3f874bc6
	v_mov_b32_e32 v87, 0x3f872b02
	v_mov_b32_e32 v86, 0x3f870a3d
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v84, 0x3f86c8b4
	v_mov_b32_e32 v83, 0x3f86a7ef
	v_mov_b32_e32 v82, 0x3f86872b
	v_mov_b32_e32 v81, 0x3f866666
	v_mov_b32_e32 v96, 0x3f860418
	v_mov_b32_e32 v95, 0x3f85e354
	v_mov_b32_e32 v94, 0x3f85c28f
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v92, 0x3f858106
	v_mov_b32_e32 v91, 0x3f856041
	v_mov_b32_e32 v90, 0x3f853f7d
	v_mov_b32_e32 v89, 0x3f851eb8
	v_mov_b32_e32 v104, 0x3f84bc6a
	v_mov_b32_e32 v103, 0x3f849ba6
	v_mov_b32_e32 v102, 0x3f847ae1
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v100, 0x3f843958
	v_mov_b32_e32 v99, 0x3f841893
	v_mov_b32_e32 v98, 0x3f83f7cf
	v_mov_b32_e32 v97, 0x3f83d70a
	v_mov_b32_e32 v112, 0x3f8374bc
	v_mov_b32_e32 v111, 0x3f8353f8
	v_mov_b32_e32 v110, 0x3f833333
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v108, 0x3f82f1aa
	v_mov_b32_e32 v107, 0x3f82d0e5
	v_mov_b32_e32 v106, 0x3f82b021
	v_mov_b32_e32 v105, 0x3f828f5c
	v_mov_b32_e32 v120, 0x3f822d0e
	v_mov_b32_e32 v119, 0x3f820c4a
	v_mov_b32_e32 v118, 0x3f81eb85
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v116, 0x3f81a9fc
	v_mov_b32_e32 v115, 0x3f818937
	v_mov_b32_e32 v114, 0x3f816873
	v_mov_b32_e32 v113, 0x3f8147ae
	v_mov_b32_e32 v128, 0x3f80e560
	v_mov_b32_e32 v127, 0x3f80c49c
	v_mov_b32_e32 v126, 0x3f80a3d7
	v_mov_b32_e32 v125, 0x3f808312
	v_mov_b32_e32 v124, 0x3f80624e
	v_mov_b32_e32 v123, 0x3f804189
	v_mov_b32_e32 v122, 0x3f8020c5
	v_mov_b32_e32 v153, 0x10101010
	v_mov_b32_e32 v155, 0x18181818
	v_mbcnt_lo_u32_b32 v158, -1, 0
	s_mov_b32 s1, 0
	s_mov_b32 s2, 0x35800000
.LBB2_4:                                ; %.preheader.i.preheader
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_5 Depth 2
	s_mov_b32 s3, 32
.LBB2_5:                                ; %.preheader.i
                                        ;   Parent Loop BB2_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v154, v153
	v_mov_b32_e32 v156, v155
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s3, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[155:156], v[153:154], v[129:136]
	s_cbranch_scc1 .LBB2_5
; %bb.6:                                ; %_Z6matrixILi1EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB2_4 Depth=1
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_xor_b32_e32 v147, 16, v158
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s1, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v147
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v147, v158, v147, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v146, v137, 0xff800000, v138
	v_lshlrev_b32_e32 v148, 2, v147
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v146, v146, v139, v140
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v146, v146, v141, v142
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v146, v146, v143, v144
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	ds_bpermute_b32 v147, v148, v146
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v147, v147, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v146, v146, v147
	v_sub_f32_e32 v137, v137, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_sub_f32 v138, v138, v146
	v_dual_sub_f32 v139, v139, v146 :: v_dual_sub_f32 v140, v140, v146
	v_sub_f32_e32 v142, v142, v146
	v_exp_f32_e32 v137, v137
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v138, 0x3fb8aa3b, v138 :: v_dual_mul_f32 v139, 0x3fb8aa3b, v139
	v_dual_sub_f32 v141, v141, v146 :: v_dual_sub_f32 v144, v144, v146
	v_mul_f32_e32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v138, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v139, v139
	v_mul_f32_e32 v142, 0x3fb8aa3b, v142
	v_mul_f32_e32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v140, v140
	v_exp_f32_e32 v142, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_exp_f32_e32 v144, v144
	v_add_f32_e32 v147, v137, v138
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_sub_f32_e32 v143, v143, v146
	v_mul_f32_e32 v137, 0x3a83126f, v137
	v_dual_mul_f32 v138, 0x3b03126f, v138 :: v_dual_add_f32 v147, v139, v147
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v141, v141
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	v_mul_f32_e32 v139, 0x3b449ba6, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_max3_num_f32 v149, v137, 0, v138
	v_dual_add_f32 v147, v140, v147 :: v_dual_mul_f32 v140, 0x3b83126f, v140
	v_exp_f32_e32 v143, v143
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v147, v141, v147
	v_mul_f32_e32 v141, 0x3ba3d70b, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v149, v149, v139, v140
	v_dual_add_f32 v147, v142, v147 :: v_dual_mul_f32 v142, 0x3bc49ba6, v142
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v147, v143, v147
	v_mul_f32_e32 v143, 0x3be56042, v143
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v149, v149, v141, v142
	v_add_f32_e32 v146, v144, v147
	v_mul_f32_e32 v144, 0x3c03126f, v144
	ds_bpermute_b32 v147, v148, v146
	v_max3_num_f32 v149, v149, v143, v144
	ds_bpermute_b32 v148, v148, v149
	s_wait_dscnt 0x1
	v_add_f32_e32 v146, v146, v147
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v148, v148, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_num_f32_e32 v148, v149, v148
	v_div_scale_f32 v147, null, v146, v146, v137
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v149, null, 0x43e00000, 0x43e00000, v148
	v_rcp_f32_e32 v150, v149
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v149, v150, 1.0
	v_fmac_f32_e32 v150, v151, v150
	v_div_scale_f32 v151, vcc_lo, v148, 0x43e00000, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v152, v151, v150
	v_fma_f32 v154, -v149, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v152, v154, v150
	v_fma_f32 v149, -v149, v152, v151
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v149, v149, v150, v152
	v_div_fixup_f32 v148, v149, 0x43e00000, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v148, 0x1f800000, v148
	v_div_scale_f32 v149, null, v148, v148, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v151, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v150
	v_div_scale_f32 v151, vcc_lo, v137, v148, v137
	v_mul_f32_e32 v152, v151, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v149, v152, v151
	v_fmac_f32_e32 v152, v154, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v149, v152, v151
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v152
	v_div_scale_f32 v150, null, v148, v148, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v151, v150
	v_fma_f32 v152, -v150, v151, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v138, v148, v138
	v_div_fixup_f32 v149, v149, v148, v137
	v_mul_f32_e32 v154, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v154, v152
	v_fmac_f32_e32 v154, v156, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v154, v152
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v151, v154
	v_mov_b16_e64 v151.l, v145.l
	v_mov_b16_e64 v151.h, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v150, v150, v148, v138
	v_mov_b16_e64 v152.l, v151.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v152.h, v151.h
	v_cvt_pk_fp8_f32 v152.l, v149, v150
	v_div_scale_f32 v149, null, v148, v148, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v154, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v154, v150
	v_div_scale_f32 v154, vcc_lo, v139, v148, v139
	v_mul_f32_e32 v156, v154, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v149, v156, v154
	v_fmac_f32_e32 v156, v159, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v149, v156, v154
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v156
	v_div_scale_f32 v150, null, v148, v148, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v149, v149, v148, v139
	v_rcp_f32_e32 v154, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v154, 1.0
	v_fmac_f32_e32 v154, v156, v154
	v_div_scale_f32 v156, vcc_lo, v140, v148, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v156, v154
	v_fma_f32 v160, -v150, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v154
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v150, v150, v154, v159
	v_div_fixup_f32 v150, v150, v148, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v152.h, v149, v150
	v_div_scale_f32 v149, null, v148, v148, v141
	v_rcp_f32_e32 v150, v149
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v149, v150, 1.0
	v_fmac_f32_e32 v150, v154, v150
	v_div_scale_f32 v154, vcc_lo, v141, v148, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v154, v150
	v_fma_f32 v159, -v149, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v159, v150
	v_fma_f32 v149, -v149, v156, v154
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v149, v149, v150, v156
	v_div_scale_f32 v150, null, v148, v148, v142
	v_div_fixup_f32 v149, v149, v148, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v154, v150
	v_fma_f32 v156, -v150, v154, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v156, v154
	v_div_scale_f32 v156, vcc_lo, v142, v148, v142
	v_mul_f32_e32 v159, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v150, v159, v156
	v_fmac_f32_e32 v159, v160, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v154, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v150, v150, v148, v142
	v_cvt_pk_fp8_f32 v151.l, v149, v150
	v_div_scale_f32 v149, null, v148, v148, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v154, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v154, v150
	v_div_scale_f32 v154, vcc_lo, v143, v148, v143
	v_mul_f32_e32 v156, v154, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v149, v156, v154
	v_fmac_f32_e32 v156, v159, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v149, v156, v154
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v156
	v_div_scale_f32 v150, null, v148, v148, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v149, v149, v148, v143
	v_rcp_f32_e32 v154, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v154, 1.0
	v_fmac_f32_e32 v154, v156, v154
	v_div_scale_f32 v156, vcc_lo, v144, v148, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v156, v154
	v_fma_f32 v160, -v150, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v154
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v150, v150, v154, v159
	v_div_fixup_f32 v148, v150, v148, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v151.h, v149, v148
	v_xor_b32_e32 v148, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v148, v148
	v_fmaak_f32 v148, s2, v148, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v128, v128, v148 :: v_dual_mul_f32 v99, v99, v148
	v_dual_mul_f32 v127, v127, v148 :: v_dual_mul_f32 v126, v126, v148
	v_dual_mul_f32 v125, v125, v148 :: v_dual_mul_f32 v124, v124, v148
	v_mul_f32_e32 v95, v95, v148
	v_dual_mul_f32 v123, v123, v148 :: v_dual_mul_f32 v122, v122, v148
	v_dual_mul_f32 v121, v121, v148 :: v_dual_mul_f32 v120, v120, v148
	v_mul_f32_e32 v93, v93, v148
	v_dual_mul_f32 v119, v119, v148 :: v_dual_mul_f32 v118, v118, v148
	v_dual_mul_f32 v117, v117, v148 :: v_dual_mul_f32 v116, v116, v148
	v_mul_f32_e32 v91, v91, v148
	v_dual_mul_f32 v115, v115, v148 :: v_dual_mul_f32 v114, v114, v148
	v_mul_f32_e32 v89, v89, v148
	v_dual_mul_f32 v113, v113, v148 :: v_dual_mul_f32 v112, v112, v148
	v_mul_f32_e32 v87, v87, v148
	v_dual_mul_f32 v111, v111, v148 :: v_dual_mul_f32 v110, v110, v148
	v_mul_f32_e32 v85, v85, v148
	v_dual_mul_f32 v109, v109, v148 :: v_dual_mul_f32 v108, v108, v148
	v_mul_f32_e32 v83, v83, v148
	v_dual_mul_f32 v107, v107, v148 :: v_dual_mul_f32 v106, v106, v148
	v_mul_f32_e32 v81, v81, v148
	v_dual_mul_f32 v105, v105, v148 :: v_dual_mul_f32 v104, v104, v148
	v_mul_f32_e32 v79, v79, v148
	v_dual_mul_f32 v103, v103, v148 :: v_dual_mul_f32 v102, v102, v148
	v_mul_f32_e32 v77, v77, v148
	v_dual_mul_f32 v101, v101, v148 :: v_dual_mul_f32 v100, v100, v148
	v_dual_mul_f32 v75, v75, v148 :: v_dual_mul_f32 v98, v98, v148
	v_mul_f32_e32 v73, v73, v148
	v_dual_mul_f32 v97, v97, v148 :: v_dual_mul_f32 v96, v96, v148
	v_dual_mul_f32 v71, v71, v148 :: v_dual_mul_f32 v94, v94, v148
	v_dual_mul_f32 v69, v69, v148 :: v_dual_mul_f32 v92, v92, v148
	v_dual_mul_f32 v67, v67, v148 :: v_dual_mul_f32 v90, v90, v148
	v_dual_mul_f32 v65, v65, v148 :: v_dual_mul_f32 v88, v88, v148
	v_dual_mul_f32 v63, v63, v148 :: v_dual_mul_f32 v86, v86, v148
	v_dual_mul_f32 v61, v61, v148 :: v_dual_mul_f32 v84, v84, v148
	v_dual_mul_f32 v59, v59, v148 :: v_dual_mul_f32 v82, v82, v148
	v_dual_mul_f32 v57, v57, v148 :: v_dual_mul_f32 v80, v80, v148
	v_dual_mul_f32 v55, v55, v148 :: v_dual_mul_f32 v78, v78, v148
	v_dual_mul_f32 v53, v53, v148 :: v_dual_mul_f32 v76, v76, v148
	v_dual_mul_f32 v51, v51, v148 :: v_dual_mul_f32 v74, v74, v148
	v_dual_mul_f32 v49, v49, v148 :: v_dual_mul_f32 v72, v72, v148
	v_dual_mul_f32 v47, v47, v148 :: v_dual_mul_f32 v70, v70, v148
	v_dual_mul_f32 v45, v45, v148 :: v_dual_mul_f32 v68, v68, v148
	v_dual_mul_f32 v43, v43, v148 :: v_dual_mul_f32 v66, v66, v148
	v_dual_mul_f32 v41, v41, v148 :: v_dual_mul_f32 v64, v64, v148
	v_dual_mul_f32 v39, v39, v148 :: v_dual_mul_f32 v62, v62, v148
	v_dual_mul_f32 v37, v37, v148 :: v_dual_mul_f32 v60, v60, v148
	v_dual_mul_f32 v35, v35, v148 :: v_dual_mul_f32 v58, v58, v148
	v_dual_mul_f32 v33, v33, v148 :: v_dual_mul_f32 v56, v56, v148
	v_dual_mul_f32 v7, v7, v148 :: v_dual_mul_f32 v54, v54, v148
	v_mul_f32_e32 v31, v148, v31
	v_dual_mul_f32 v52, v52, v148 :: v_dual_mul_f32 v5, v5, v148
	v_dual_mul_f32 v50, v50, v148 :: v_dual_mul_f32 v29, v148, v29
	v_dual_mul_f32 v48, v48, v148 :: v_dual_mul_f32 v3, v3, v148
	v_dual_mul_f32 v46, v46, v148 :: v_dual_mul_f32 v27, v148, v27
	v_dual_mul_f32 v44, v44, v148 :: v_dual_mul_f32 v1, v1, v148
	v_dual_mul_f32 v42, v42, v148 :: v_dual_mul_f32 v25, v148, v25
	v_mul_f32_e32 v40, v40, v148
	v_dual_mul_f32 v38, v38, v148 :: v_dual_mul_f32 v23, v148, v23
	v_mul_f32_e32 v36, v36, v148
	v_dual_mul_f32 v34, v34, v148 :: v_dual_mul_f32 v21, v148, v21
	v_dual_mul_f32 v32, v148, v32 :: v_dual_mul_f32 v19, v148, v19
	v_dual_mul_f32 v30, v148, v30 :: v_dual_mul_f32 v17, v148, v17
	v_dual_mul_f32 v28, v148, v28 :: v_dual_mul_f32 v15, v148, v15
	v_dual_mul_f32 v26, v148, v26 :: v_dual_mul_f32 v13, v148, v13
	v_dual_mul_f32 v24, v148, v24 :: v_dual_mul_f32 v11, v148, v11
	v_dual_mul_f32 v22, v148, v22 :: v_dual_mul_f32 v9, v148, v9
	v_mul_f32_e32 v20, v148, v20
	v_mul_f32_e32 v18, v148, v18
	v_mul_f32_e32 v16, v148, v16
	v_mul_f32_e32 v14, v148, v14
	v_mul_f32_e32 v12, v148, v12
	v_mul_f32_e32 v10, v148, v10
	v_mul_f32_e32 v8, v8, v148
	v_mul_f32_e32 v6, v6, v148
	v_mul_f32_e32 v4, v4, v148
	v_mul_f32_e32 v2, v2, v148
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v137, v146, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v137, v147, v146, v137
	v_div_scale_f32 v147, null, v146, v146, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v137, 0, v137
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v138, v146, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v138, v147, v146, v138
	v_div_scale_f32 v147, null, v146, v146, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v138, 0x3c23d70a, v138
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v139, v146, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v139, v147, v146, v139
	v_div_scale_f32 v147, null, v146, v146, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v139, 0x3ca3d70a, v139
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v140, v146, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v140, v147, v146, v140
	v_div_scale_f32 v147, null, v146, v146, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v140, 0x3cf5c28f, v140
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v141, v146, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v141, v147, v146, v141
	v_div_scale_f32 v147, null, v146, v146, v142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v141, 0x3d23d70a, v141
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v142, v146, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v142, v147, v146, v142
	v_div_scale_f32 v147, null, v146, v146, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v142, 0x3d4ccccc, v142
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v143, v146, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v143, v147, v146, v143
	v_div_scale_f32 v147, null, v146, v146, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v143, 0x3d75c28f, v143
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v144, v146, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v144, v147, v146, v144
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v144, 0x3d8f5c29, v144
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB2_4
	s_branch .LBB2_8
.LBB2_7:
	v_dual_mov_b32 v121, 1.0 :: v_dual_mov_b32 v122, 0x3f8020c5
	v_dual_mov_b32 v123, 0x3f804189 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v124, 0x3f80624e :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v125, 0x3f808312 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v126, 0x3f80a3d7 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v127, 0x3f80c49c :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v128, 0x3f80e560 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v113, 0x3f8147ae :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v114, 0x3f816873 :: v_dual_mov_b32 v129, 0
	v_mov_b32_e32 v115, 0x3f818937
	v_mov_b32_e32 v116, 0x3f81a9fc
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v118, 0x3f81eb85
	v_mov_b32_e32 v119, 0x3f820c4a
	v_mov_b32_e32 v120, 0x3f822d0e
	v_mov_b32_e32 v105, 0x3f828f5c
	v_mov_b32_e32 v106, 0x3f82b021
	v_mov_b32_e32 v107, 0x3f82d0e5
	v_mov_b32_e32 v108, 0x3f82f1aa
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v110, 0x3f833333
	v_mov_b32_e32 v111, 0x3f8353f8
	v_mov_b32_e32 v112, 0x3f8374bc
	v_mov_b32_e32 v97, 0x3f83d70a
	v_mov_b32_e32 v98, 0x3f83f7cf
	v_mov_b32_e32 v99, 0x3f841893
	v_mov_b32_e32 v100, 0x3f843958
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v102, 0x3f847ae1
	v_mov_b32_e32 v103, 0x3f849ba6
	v_mov_b32_e32 v104, 0x3f84bc6a
	v_mov_b32_e32 v89, 0x3f851eb8
	v_mov_b32_e32 v90, 0x3f853f7d
	v_mov_b32_e32 v91, 0x3f856041
	v_mov_b32_e32 v92, 0x3f858106
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v94, 0x3f85c28f
	v_mov_b32_e32 v95, 0x3f85e354
	v_mov_b32_e32 v96, 0x3f860418
	v_mov_b32_e32 v81, 0x3f866666
	v_mov_b32_e32 v82, 0x3f86872b
	v_mov_b32_e32 v83, 0x3f86a7ef
	v_mov_b32_e32 v84, 0x3f86c8b4
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v86, 0x3f870a3d
	v_mov_b32_e32 v87, 0x3f872b02
	v_mov_b32_e32 v88, 0x3f874bc6
	v_mov_b32_e32 v73, 0x3f87ae14
	v_mov_b32_e32 v74, 0x3f87ced9
	v_mov_b32_e32 v75, 0x3f87ef9d
	v_mov_b32_e32 v76, 0x3f881062
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v78, 0x3f8851eb
	v_mov_b32_e32 v79, 0x3f8872b0
	v_mov_b32_e32 v80, 0x3f889374
	v_mov_b32_e32 v65, 0x3f88f5c3
	v_mov_b32_e32 v66, 0x3f891688
	v_mov_b32_e32 v67, 0x3f89374c
	v_mov_b32_e32 v68, 0x3f895811
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v70, 0x3f89999a
	v_mov_b32_e32 v71, 0x3f89ba5f
	v_mov_b32_e32 v72, 0x3f89db23
	v_mov_b32_e32 v57, 0x3f8a3d71
	v_mov_b32_e32 v58, 0x3f8a5e36
	v_mov_b32_e32 v59, 0x3f8a7efa
	v_mov_b32_e32 v60, 0x3f8a9fbf
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v62, 0x3f8ae148
	v_mov_b32_e32 v63, 0x3f8b020d
	v_mov_b32_e32 v64, 0x3f8b22d1
	v_mov_b32_e32 v49, 0x3f8b851f
	v_mov_b32_e32 v50, 0x3f8ba5e4
	v_mov_b32_e32 v51, 0x3f8bc6a8
	v_mov_b32_e32 v52, 0x3f8be76d
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v54, 0x3f8c28f6
	v_mov_b32_e32 v55, 0x3f8c49bb
	v_mov_b32_e32 v56, 0x3f8c6a7f
	v_mov_b32_e32 v41, 0x3f8ccccd
	v_mov_b32_e32 v42, 0x3f8ced92
	v_mov_b32_e32 v43, 0x3f8d0e56
	v_mov_b32_e32 v44, 0x3f8d2f1b
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v46, 0x3f8d70a4
	v_mov_b32_e32 v47, 0x3f8d9169
	v_mov_b32_e32 v48, 0x3f8db22d
	v_mov_b32_e32 v33, 0x3f8e147b
	v_mov_b32_e32 v34, 0x3f8e3540
	v_mov_b32_e32 v35, 0x3f8e5604
	v_mov_b32_e32 v36, 0x3f8e76c9
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v38, 0x3f8eb852
	v_mov_b32_e32 v39, 0x3f8ed917
	v_mov_b32_e32 v40, 0x3f8ef9db
	v_mov_b32_e32 v25, 0x3f8f5c29
	v_mov_b32_e32 v26, 0x3f8f7cee
	v_mov_b32_e32 v27, 0x3f8f9db2
	v_mov_b32_e32 v28, 0x3f8fbe77
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v30, 0x3f900000
	v_mov_b32_e32 v31, 0x3f9020c5
	v_mov_b32_e32 v32, 0x3f904189
	v_mov_b32_e32 v17, 0x3f90a3d7
	v_mov_b32_e32 v18, 0x3f90c49c
	v_mov_b32_e32 v19, 0x3f90e560
	v_mov_b32_e32 v20, 0x3f910625
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v22, 0x3f9147ae
	v_mov_b32_e32 v23, 0x3f916873
	v_mov_b32_e32 v24, 0x3f918937
	v_mov_b32_e32 v9, 0x3f91eb85
	v_mov_b32_e32 v10, 0x3f920c4a
	v_mov_b32_e32 v11, 0x3f922d0e
	v_mov_b32_e32 v12, 0x3f924dd3
	v_mov_b32_e32 v13, 0x3f926e97
	v_mov_b32_e32 v14, 0x3f928f5c
	v_mov_b32_e32 v15, 0x3f92b021
	v_mov_b32_e32 v16, 0x3f92d0e5
	v_mov_b32_e32 v1, 0x3f933333
	v_mov_b32_e32 v2, 0x3f9353f8
	v_mov_b32_e32 v3, 0x3f9374bc
	v_mov_b32_e32 v4, 0x3f939581
	v_mov_b32_e32 v5, 0x3f93b645
	v_mov_b32_e32 v6, 0x3f93d70a
	v_mov_b32_e32 v7, 0x3f93f7cf
	v_mov_b32_e32 v8, 0x3f941893
.LBB2_8:                                ; %.preheader72
	v_add_f32_e32 v129, 0, v129
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v130
	v_add_f32_e32 v129, v129, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v132
	v_add_f32_e32 v129, v129, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v134
	v_add_f32_e32 v129, v129, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v136
	v_add_f32_e32 v121, v129, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v122
	v_add_f32_e32 v121, v121, v123
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v124
	v_add_f32_e32 v121, v121, v125
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v126
	v_add_f32_e32 v121, v121, v127
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v128
	v_add_f32_e32 v113, v121, v113
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v114
	v_add_f32_e32 v113, v113, v115
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v116
	v_add_f32_e32 v113, v113, v117
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v118
	v_add_f32_e32 v113, v113, v119
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v120
	v_add_f32_e32 v105, v113, v105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v106
	v_add_f32_e32 v105, v105, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v108
	v_add_f32_e32 v105, v105, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v110
	v_add_f32_e32 v105, v105, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v112
	v_add_f32_e32 v97, v105, v97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v98
	v_add_f32_e32 v97, v97, v99
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v100
	v_add_f32_e32 v97, v97, v101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v102
	v_add_f32_e32 v97, v97, v103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v104
	v_add_f32_e32 v89, v97, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v90
	v_add_f32_e32 v89, v89, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v92
	v_add_f32_e32 v89, v89, v93
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v94
	v_add_f32_e32 v89, v89, v95
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v96
	v_add_f32_e32 v81, v89, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v82
	v_add_f32_e32 v81, v81, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v84
	v_add_f32_e32 v81, v81, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v86
	v_add_f32_e32 v81, v81, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v88
	v_add_f32_e32 v73, v81, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v74
	v_add_f32_e32 v73, v73, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v76
	v_add_f32_e32 v73, v73, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v78
	v_add_f32_e32 v73, v73, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v80
	v_add_f32_e32 v65, v73, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v66
	v_add_f32_e32 v65, v65, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v68
	v_add_f32_e32 v65, v65, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v70
	v_add_f32_e32 v65, v65, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v72
	v_add_f32_e32 v57, v65, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v58
	v_add_f32_e32 v57, v57, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v60
	v_add_f32_e32 v57, v57, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v62
	v_add_f32_e32 v57, v57, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v64
	v_add_f32_e32 v49, v57, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v50
	v_add_f32_e32 v49, v49, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v52
	v_add_f32_e32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v54
	v_add_f32_e32 v49, v49, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v56
	v_add_f32_e32 v41, v49, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v42
	v_add_f32_e32 v41, v41, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v44
	v_add_f32_e32 v41, v41, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v46
	v_add_f32_e32 v41, v41, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v48
	v_add_f32_e32 v33, v41, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v34
	v_add_f32_e32 v33, v33, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v36
	v_add_f32_e32 v33, v33, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v38
	v_add_f32_e32 v33, v33, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v40
	v_add_f32_e32 v25, v33, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v157
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v139
	v_add_f32_e32 v1, v1, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v141
	v_add_f32_e32 v1, v1, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v143
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end2:
	.size	_Z3runILi2ELi1EEvPfPji, .Lfunc_end2-_Z3runILi2ELi1EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi2ELi1EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 161
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_Z3runILi2ELi1EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi2ELi1EEvPfPji,"axG",@progbits,_Z3runILi2ELi1EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi2ELi1EEvPfPji.num_vgpr, 161
	.set .L_Z3runILi2ELi1EEvPfPji.num_agpr, 0
	.set .L_Z3runILi2ELi1EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi2ELi1EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi2ELi1EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi2ELi1EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi2ELi1EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi2ELi1EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi2ELi1EEvPfPji.has_recursion, 0
	.set .L_Z3runILi2ELi1EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6220
; TotalNumSgprs: 10
; NumVgprs: 161
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 161
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi3ELi1EEvPfPji,"axG",@progbits,_Z3runILi3ELi1EEvPfPji,comdat
	.protected	_Z3runILi3ELi1EEvPfPji  ; -- Begin function _Z3runILi3ELi1EEvPfPji
	.globl	_Z3runILi3ELi1EEvPfPji
	.p2align	8
	.type	_Z3runILi3ELi1EEvPfPji,@function
_Z3runILi3ELi1EEvPfPji:                 ; @_Z3runILi3ELi1EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB3_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB3_2:                                ; %.preheader100
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s1, s[0:1], 0x10
	v_lshl_add_u32 v157, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s0, 0x3a83126f
	v_mov_b32_e32 v2, 0x3f8020c5
	v_mov_b32_e32 v3, 0x3f804189
	ds_store_b32 v157, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v145, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v146, s0, v1, 0x3dcccccd
	v_fmaak_f32 v147, s0, v1, 0x3e4ccccd
	v_fmaak_f32 v148, s0, v1, 0x3e99999a
	v_fmaak_f32 v149, s0, v1, 0x3ecccccd
	v_fma_f32 v150, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v151, s0, v1, 0x3f19999a
	v_dual_fmaak_f32 v152, s0, v1, 0x3f333333 :: v_dual_mov_b32 v1, 1.0
	v_mov_b32_e32 v4, 0x3f80624e
	v_mov_b32_e32 v5, 0x3f808312
	v_mov_b32_e32 v6, 0x3f80a3d7
	v_mov_b32_e32 v7, 0x3f80c49c
	v_mov_b32_e32 v8, 0x3f80e560
	v_mov_b32_e32 v9, 0x3f8147ae
	v_mov_b32_e32 v10, 0x3f816873
	v_mov_b32_e32 v11, 0x3f818937
	v_mov_b32_e32 v12, 0x3f81a9fc
	v_mov_b32_e32 v13, 0x3f81cac0
	v_mov_b32_e32 v14, 0x3f81eb85
	v_mov_b32_e32 v15, 0x3f820c4a
	v_mov_b32_e32 v16, 0x3f822d0e
	v_mov_b32_e32 v17, 0x3f828f5c
	v_mov_b32_e32 v18, 0x3f82b021
	v_mov_b32_e32 v19, 0x3f82d0e5
	v_mov_b32_e32 v20, 0x3f82f1aa
	v_mov_b32_e32 v21, 0x3f83126e
	v_mov_b32_e32 v22, 0x3f833333
	v_mov_b32_e32 v23, 0x3f8353f8
	v_mov_b32_e32 v24, 0x3f8374bc
	v_mov_b32_e32 v25, 0x3f83d70a
	v_mov_b32_e32 v26, 0x3f83f7cf
	v_mov_b32_e32 v27, 0x3f841893
	v_mov_b32_e32 v28, 0x3f843958
	v_mov_b32_e32 v29, 0x3f845a1c
	v_mov_b32_e32 v30, 0x3f847ae1
	v_mov_b32_e32 v31, 0x3f849ba6
	v_mov_b32_e32 v32, 0x3f84bc6a
	v_mov_b32_e32 v33, 0x3f851eb8
	v_mov_b32_e32 v34, 0x3f853f7d
	v_mov_b32_e32 v35, 0x3f856041
	v_mov_b32_e32 v36, 0x3f858106
	v_mov_b32_e32 v37, 0x3f85a1ca
	v_mov_b32_e32 v38, 0x3f85c28f
	v_mov_b32_e32 v39, 0x3f85e354
	v_mov_b32_e32 v40, 0x3f860418
	v_mov_b32_e32 v41, 0x3f866666
	v_mov_b32_e32 v42, 0x3f86872b
	v_mov_b32_e32 v43, 0x3f86a7ef
	v_mov_b32_e32 v44, 0x3f86c8b4
	v_mov_b32_e32 v45, 0x3f86e978
	v_mov_b32_e32 v46, 0x3f870a3d
	v_mov_b32_e32 v47, 0x3f872b02
	v_mov_b32_e32 v48, 0x3f874bc6
	v_mov_b32_e32 v49, 0x3f87ae14
	v_mov_b32_e32 v50, 0x3f87ced9
	v_mov_b32_e32 v51, 0x3f87ef9d
	v_mov_b32_e32 v52, 0x3f881062
	v_mov_b32_e32 v53, 0x3f883126
	v_mov_b32_e32 v54, 0x3f8851eb
	v_mov_b32_e32 v55, 0x3f8872b0
	v_mov_b32_e32 v56, 0x3f889374
	v_mov_b32_e32 v57, 0x3f88f5c3
	v_mov_b32_e32 v58, 0x3f891688
	v_mov_b32_e32 v59, 0x3f89374c
	v_mov_b32_e32 v60, 0x3f895811
	v_mov_b32_e32 v61, 0x3f8978d5
	v_mov_b32_e32 v62, 0x3f89999a
	v_mov_b32_e32 v63, 0x3f89ba5f
	v_mov_b32_e32 v64, 0x3f89db23
	v_mov_b32_e32 v65, 0x3f8a3d71
	v_mov_b32_e32 v66, 0x3f8a5e36
	v_mov_b32_e32 v67, 0x3f8a7efa
	v_mov_b32_e32 v68, 0x3f8a9fbf
	v_mov_b32_e32 v69, 0x3f8ac083
	v_mov_b32_e32 v70, 0x3f8ae148
	v_mov_b32_e32 v71, 0x3f8b020d
	v_mov_b32_e32 v72, 0x3f8b22d1
	v_mov_b32_e32 v73, 0x3f8b851f
	v_mov_b32_e32 v74, 0x3f8ba5e4
	v_mov_b32_e32 v75, 0x3f8bc6a8
	v_mov_b32_e32 v76, 0x3f8be76d
	v_mov_b32_e32 v77, 0x3f8c0831
	v_mov_b32_e32 v78, 0x3f8c28f6
	v_mov_b32_e32 v79, 0x3f8c49bb
	v_mov_b32_e32 v80, 0x3f8c6a7f
	v_mov_b32_e32 v81, 0x3f8ccccd
	v_mov_b32_e32 v82, 0x3f8ced92
	v_mov_b32_e32 v83, 0x3f8d0e56
	v_mov_b32_e32 v84, 0x3f8d2f1b
	v_mov_b32_e32 v85, 0x3f8d4fdf
	v_mov_b32_e32 v86, 0x3f8d70a4
	v_mov_b32_e32 v87, 0x3f8d9169
	v_mov_b32_e32 v88, 0x3f8db22d
	v_mov_b32_e32 v89, 0x3f8e147b
	v_mov_b32_e32 v90, 0x3f8e3540
	v_mov_b32_e32 v91, 0x3f8e5604
	v_mov_b32_e32 v92, 0x3f8e76c9
	v_mov_b32_e32 v93, 0x3f8e978d
	v_mov_b32_e32 v94, 0x3f8eb852
	v_mov_b32_e32 v95, 0x3f8ed917
	v_mov_b32_e32 v96, 0x3f8ef9db
	v_mov_b32_e32 v97, 0x3f8f5c29
	v_mov_b32_e32 v98, 0x3f8f7cee
	v_mov_b32_e32 v99, 0x3f8f9db2
	v_mov_b32_e32 v100, 0x3f8fbe77
	v_mov_b32_e32 v101, 0x3f8fdf3b
	v_mov_b32_e32 v102, 0x3f900000
	v_mov_b32_e32 v103, 0x3f9020c5
	v_mov_b32_e32 v104, 0x3f904189
	v_mov_b32_e32 v105, 0x3f90a3d7
	v_mov_b32_e32 v106, 0x3f90c49c
	v_mov_b32_e32 v107, 0x3f90e560
	v_mov_b32_e32 v108, 0x3f910625
	v_mov_b32_e32 v109, 0x3f9126e9
	v_mov_b32_e32 v110, 0x3f9147ae
	v_mov_b32_e32 v111, 0x3f916873
	v_mov_b32_e32 v112, 0x3f918937
	v_mov_b32_e32 v113, 0x3f91eb85
	v_mov_b32_e32 v114, 0x3f920c4a
	v_mov_b32_e32 v115, 0x3f922d0e
	v_mov_b32_e32 v116, 0x3f924dd3
	v_mov_b32_e32 v117, 0x3f926e97
	v_mov_b32_e32 v118, 0x3f928f5c
	v_mov_b32_e32 v119, 0x3f92b021
	v_mov_b32_e32 v120, 0x3f92d0e5
	v_mov_b32_e32 v121, 0x3f933333
	v_mov_b32_e32 v122, 0x3f9353f8
	v_mov_b32_e32 v123, 0x3f9374bc
	v_mov_b32_e32 v124, 0x3f939581
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s1, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_16
; %bb.3:                                ; %.lr.ph
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v126, 0x3f93d70a
	v_mov_b32_e32 v125, 0x3f93b645
	v_mov_b32_e32 v127, 0x3f93f7cf
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v128, 0x3f941893 :: v_dual_mov_b32 v139, v137
	v_mov_b32_e32 v138, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
	v_mov_b32_e32 v129, v137
	v_cmp_lt_u32_e64 s0, 0x7f, v0
	v_mbcnt_lo_u32_b32 v158, -1, 0
	v_dual_mov_b32 v153, 0x10101010 :: v_dual_mov_b32 v130, v138
	v_dual_mov_b32 v155, 0x18181818 :: v_dual_mov_b32 v132, v140
	v_dual_mov_b32 v131, v139 :: v_dual_mov_b32 v136, v144
	v_dual_mov_b32 v133, v141 :: v_dual_mov_b32 v134, v142
	v_mov_b32_e32 v135, v143
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0x35800000
	s_branch .LBB3_5
.LBB3_4:                                ; %_Z6matrixILi1EEvRAT__Dv8_fDv2_iS3_.exit76
                                        ;   in Loop: Header=BB3_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s2, s2, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s2, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB3_17
.LBB3_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_9 Depth 2
                                        ;     Child Loop BB3_12 Depth 2
	v_xor_b32_e32 v138, 16, v158
	s_and_saveexec_b32 s6, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB3_7
; %bb.6:                                ;   in Loop: Header=BB3_5 Depth=1
	v_max3_num_f32 v139, v145, 0xff800000, v146
	v_cmp_gt_u32_e32 vcc_lo, 32, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_num_f32 v139, v139, v147, v148
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v140, v158, v138, vcc_lo
	v_max3_num_f32 v139, v139, v149, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v139, v139, v151, v152
	v_lshlrev_b32_e32 v154, 2, v140
	ds_bpermute_b32 v140, v154, v139
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v140, v140, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v139, v139, v140
	v_sub_f32_e32 v144, v148, v139
	v_dual_sub_f32 v140, v145, v139 :: v_dual_sub_f32 v143, v147, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v141, v146, v139 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v144
	v_dual_mul_f32 v140, 0x3fb8aa3b, v140 :: v_dual_mul_f32 v143, 0x3fb8aa3b, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_exp_f32_e32 v156, v144
	v_sub_f32_e32 v144, v149, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v140, v140
	v_exp_f32_e32 v141, v141
	v_exp_f32_e32 v143, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v149, v144
	v_sub_f32_e32 v144, v150, v139
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v146, 0x3b03126f, v141
	v_mul_f32_e32 v145, 0x3a83126f, v140
	v_mul_f32_e32 v144, 0x3fb8aa3b, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v150, v144
	v_dual_sub_f32 v144, v151, v139 :: v_dual_sub_f32 v139, v152, v139
	v_add_f32_e32 v142, v140, v141
	v_max3_num_f32 v140, v145, 0, v146
	v_dual_mul_f32 v144, 0x3fb8aa3b, v144 :: v_dual_mul_f32 v139, 0x3fb8aa3b, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v142, v143, v142
	v_mul_f32_e32 v141, 0x3bc49ba6, v150
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v151, v144
	v_exp_f32_e32 v139, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v142, v156, v142
	v_mul_f32_e32 v144, 0x3b449ba6, v143
	v_dual_mul_f32 v143, 0x3b83126f, v156 :: v_dual_add_f32 v142, v149, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v140, v140, v144, v143
	v_add_f32_e32 v142, v150, v142
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v142, v151, v142
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v147, v139, v142 :: v_dual_mul_f32 v142, 0x3ba3d70b, v149
	v_mul_f32_e32 v139, 0x3c03126f, v139
	ds_bpermute_b32 v148, v154, v147
	v_max3_num_f32 v149, v140, v142, v141
	v_mul_f32_e32 v140, 0x3be56042, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v149, v149, v140, v139
	ds_bpermute_b32 v150, v154, v149
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v150, v150, v150
	v_max_num_f32_e32 v149, v149, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v150, null, 0x43e00000, 0x43e00000, v149
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v150, v151, 1.0
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v149, 0x43e00000, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v154, v152, v151
	v_fma_f32 v156, -v150, v154, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v156, v151
	v_fma_f32 v150, -v150, v154, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v150, v150, v151, v154
	v_div_fixup_f32 v149, v150, 0x43e00000, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v149, 0x1f800000, v149
	v_div_scale_f32 v150, null, v149, v149, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v151, v150
	v_fma_f32 v152, -v150, v151, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v145, v149, v145
	v_mul_f32_e32 v154, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v154, v152
	v_fmac_f32_e32 v154, v156, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v154, v152
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v151, v154
	v_div_scale_f32 v151, null, v149, v149, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v150, v150, v149, v145
	v_rcp_f32_e32 v152, v151
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v151, v152, 1.0
	v_fmac_f32_e32 v152, v154, v152
	v_div_scale_f32 v154, vcc_lo, v146, v149, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v154, v152
	v_fma_f32 v159, -v151, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v159, v152
	v_fma_f32 v151, -v151, v156, v154
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v151, v151, v152, v156
	v_mov_b16_e64 v152.l, v137.l
	v_mov_b16_e64 v152.h, 0
	v_div_fixup_f32 v151, v151, v149, v146
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v154.l, v152.l
	v_mov_b16_e64 v154.h, v152.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v154.l, v150, v151
	v_div_scale_f32 v150, null, v149, v149, v144
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v151, 1.0
	v_fmac_f32_e32 v151, v156, v151
	v_div_scale_f32 v156, vcc_lo, v144, v149, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v156, v151
	v_fma_f32 v160, -v150, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v151
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v150, v150, v151, v159
	v_div_scale_f32 v151, null, v149, v149, v143
	v_div_fixup_f32 v150, v150, v149, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v156, v151
	v_fma_f32 v159, -v151, v156, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v159, v156
	v_div_scale_f32 v159, vcc_lo, v143, v149, v143
	v_mul_f32_e32 v160, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v161, -v151, v160, v159
	v_fmac_f32_e32 v160, v161, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v151, v160, v159
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v151, v151, v156, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v151, v151, v149, v143
	v_cvt_pk_fp8_f32 v154.h, v150, v151
	v_div_scale_f32 v150, null, v149, v149, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v151, v150
	v_fma_f32 v156, -v150, v151, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v151, v156, v151
	v_div_scale_f32 v156, vcc_lo, v142, v149, v142
	v_mul_f32_e32 v159, v156, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v150, v159, v156
	v_fmac_f32_e32 v159, v160, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v151, v159
	v_div_scale_f32 v151, null, v149, v149, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v150, v150, v149, v142
	v_rcp_f32_e32 v156, v151
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v151, v156, 1.0
	v_fmac_f32_e32 v156, v159, v156
	v_div_scale_f32 v159, vcc_lo, v141, v149, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v160, v159, v156
	v_fma_f32 v161, -v151, v160, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v160, v161, v156
	v_fma_f32 v151, -v151, v160, v159
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v151, v151, v156, v160
	v_div_fixup_f32 v151, v151, v149, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v152.l, v150, v151
	v_div_scale_f32 v150, null, v149, v149, v140
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v151, 1.0
	v_fmac_f32_e32 v151, v156, v151
	v_div_scale_f32 v156, vcc_lo, v140, v149, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v156, v151
	v_fma_f32 v160, -v150, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v151
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v150, v150, v151, v159
	v_div_scale_f32 v151, null, v149, v149, v139
	v_div_fixup_f32 v150, v150, v149, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v156, v151
	v_fma_f32 v159, -v151, v156, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v159, v156
	v_div_scale_f32 v159, vcc_lo, v139, v149, v139
	v_mul_f32_e32 v160, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v161, -v151, v160, v159
	v_fmac_f32_e32 v160, v161, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v151, v160, v159
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v151, v151, v156, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v149, v151, v149, v139
	v_cvt_pk_fp8_f32 v152.h, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v149, v154, v152
	v_add_f32_e32 v152, v147, v148
	v_cvt_f32_ubyte0_e32 v149, v149
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v147, null, v152, v152, v145
	v_fmaak_f32 v149, s3, v149, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v148, v147
	v_dual_mul_f32 v8, v8, v149 :: v_dual_mul_f32 v19, v19, v149
	v_dual_mul_f32 v7, v7, v149 :: v_dual_mul_f32 v6, v6, v149
	v_mul_f32_e32 v31, v31, v149
	v_dual_mul_f32 v5, v5, v149 :: v_dual_mul_f32 v4, v4, v149
	v_mul_f32_e32 v27, v27, v149
	v_dual_mul_f32 v3, v3, v149 :: v_dual_mul_f32 v2, v2, v149
	v_mul_f32_e32 v39, v39, v149
	v_dual_mul_f32 v1, v1, v149 :: v_dual_mul_f32 v16, v16, v149
	v_mul_f32_e32 v35, v35, v149
	v_dual_mul_f32 v15, v15, v149 :: v_dual_mul_f32 v14, v14, v149
	v_mul_f32_e32 v33, v33, v149
	v_dual_mul_f32 v13, v13, v149 :: v_dual_mul_f32 v12, v12, v149
	v_mul_f32_e32 v47, v47, v149
	v_dual_mul_f32 v11, v11, v149 :: v_dual_mul_f32 v10, v10, v149
	v_mul_f32_e32 v45, v45, v149
	v_dual_mul_f32 v9, v9, v149 :: v_dual_mul_f32 v24, v24, v149
	v_mul_f32_e32 v43, v43, v149
	v_dual_mul_f32 v23, v23, v149 :: v_dual_mul_f32 v22, v22, v149
	v_mul_f32_e32 v41, v41, v149
	v_dual_mul_f32 v21, v21, v149 :: v_dual_mul_f32 v20, v20, v149
	v_dual_mul_f32 v55, v55, v149 :: v_dual_mul_f32 v18, v18, v149
	v_mul_f32_e32 v53, v53, v149
	v_dual_mul_f32 v17, v17, v149 :: v_dual_mul_f32 v32, v32, v149
	v_dual_mul_f32 v51, v51, v149 :: v_dual_mul_f32 v30, v30, v149
	v_mul_f32_e32 v49, v49, v149
	v_dual_mul_f32 v29, v29, v149 :: v_dual_mul_f32 v28, v28, v149
	v_dual_mul_f32 v63, v63, v149 :: v_dual_mul_f32 v26, v26, v149
	v_mul_f32_e32 v61, v61, v149
	v_dual_mul_f32 v25, v25, v149 :: v_dual_mul_f32 v40, v40, v149
	v_dual_mul_f32 v59, v59, v149 :: v_dual_mul_f32 v38, v38, v149
	v_mul_f32_e32 v57, v57, v149
	v_dual_mul_f32 v37, v37, v149 :: v_dual_mul_f32 v36, v36, v149
	v_dual_mul_f32 v71, v71, v149 :: v_dual_mul_f32 v34, v34, v149
	v_dual_mul_f32 v69, v69, v149 :: v_dual_mul_f32 v48, v48, v149
	v_dual_mul_f32 v67, v67, v149 :: v_dual_mul_f32 v46, v46, v149
	v_dual_mul_f32 v65, v65, v149 :: v_dual_mul_f32 v44, v44, v149
	v_dual_mul_f32 v79, v79, v149 :: v_dual_mul_f32 v42, v42, v149
	v_dual_mul_f32 v77, v77, v149 :: v_dual_mul_f32 v56, v56, v149
	v_dual_mul_f32 v75, v75, v149 :: v_dual_mul_f32 v54, v54, v149
	v_dual_mul_f32 v73, v73, v149 :: v_dual_mul_f32 v52, v52, v149
	v_dual_mul_f32 v87, v87, v149 :: v_dual_mul_f32 v50, v50, v149
	v_dual_mul_f32 v85, v85, v149 :: v_dual_mul_f32 v64, v64, v149
	v_dual_mul_f32 v83, v83, v149 :: v_dual_mul_f32 v62, v62, v149
	v_dual_mul_f32 v81, v81, v149 :: v_dual_mul_f32 v60, v60, v149
	v_dual_mul_f32 v95, v95, v149 :: v_dual_mul_f32 v58, v58, v149
	v_dual_mul_f32 v93, v93, v149 :: v_dual_mul_f32 v72, v72, v149
	v_dual_mul_f32 v91, v91, v149 :: v_dual_mul_f32 v70, v70, v149
	v_dual_mul_f32 v89, v89, v149 :: v_dual_mul_f32 v68, v68, v149
	v_dual_mul_f32 v103, v149, v103 :: v_dual_mul_f32 v66, v66, v149
	v_dual_mul_f32 v99, v149, v99 :: v_dual_mul_f32 v80, v80, v149
	v_dual_mul_f32 v111, v149, v111 :: v_dual_mul_f32 v78, v78, v149
	v_dual_mul_f32 v107, v149, v107 :: v_dual_mul_f32 v76, v76, v149
	v_dual_mul_f32 v119, v149, v119 :: v_dual_mul_f32 v74, v74, v149
	v_dual_mul_f32 v115, v149, v115 :: v_dual_mul_f32 v88, v88, v149
	v_dual_mul_f32 v127, v149, v127 :: v_dual_mul_f32 v86, v86, v149
	v_dual_mul_f32 v123, v149, v123 :: v_dual_mul_f32 v84, v84, v149
	v_mul_f32_e32 v82, v82, v149
	v_mul_f32_e32 v96, v96, v149
	v_mul_f32_e32 v94, v94, v149
	v_mul_f32_e32 v92, v92, v149
	v_mul_f32_e32 v90, v90, v149
	v_dual_mul_f32 v104, v149, v104 :: v_dual_mul_f32 v101, v149, v101
	v_dual_mul_f32 v102, v149, v102 :: v_dual_mul_f32 v97, v149, v97
	v_dual_mul_f32 v100, v149, v100 :: v_dual_mul_f32 v109, v149, v109
	v_dual_mul_f32 v98, v149, v98 :: v_dual_mul_f32 v105, v149, v105
	v_dual_mul_f32 v112, v149, v112 :: v_dual_mul_f32 v117, v149, v117
	v_dual_mul_f32 v110, v149, v110 :: v_dual_mul_f32 v113, v149, v113
	v_dual_mul_f32 v108, v149, v108 :: v_dual_mul_f32 v125, v149, v125
	v_dual_mul_f32 v106, v149, v106 :: v_dual_mul_f32 v121, v149, v121
	v_mul_f32_e32 v120, v149, v120
	v_mul_f32_e32 v118, v149, v118
	v_mul_f32_e32 v116, v149, v116
	v_mul_f32_e32 v114, v149, v114
	v_mul_f32_e32 v128, v149, v128
	v_mul_f32_e32 v126, v149, v126
	v_mul_f32_e32 v124, v149, v124
	v_mul_f32_e32 v122, v149, v122
	v_fma_f32 v149, -v147, v148, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v145, v152, v145
	v_mul_f32_e32 v150, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v147, v150, v149
	v_fmac_f32_e32 v150, v151, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v147, v147, v148, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v145, v147, v152, v145
	v_div_scale_f32 v147, null, v152, v152, v146
	v_add_f32_e32 v145, 0, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v148, v147
	v_fma_f32 v149, -v147, v148, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v146, v152, v146
	v_mul_f32_e32 v150, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v147, v150, v149
	v_fmac_f32_e32 v150, v151, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v147, v147, v148, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v146, v147, v152, v146
	v_div_scale_f32 v147, null, v152, v152, v144
	v_add_f32_e32 v146, 0x3c23d70a, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v148, v147
	v_fma_f32 v149, -v147, v148, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v144, v152, v144
	v_mul_f32_e32 v150, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v147, v150, v149
	v_fmac_f32_e32 v150, v151, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v147, v147, v148, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v144, v147, v152, v144
	v_add_f32_e32 v147, 0x3ca3d70a, v144
	v_div_scale_f32 v144, null, v152, v152, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v148, v144
	v_fma_f32 v149, -v144, v148, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v143, v152, v143
	v_mul_f32_e32 v150, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v144, v150, v149
	v_fmac_f32_e32 v150, v151, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v144, -v144, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v144, v144, v148, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v143, v144, v152, v143
	v_add_f32_e32 v148, 0x3cf5c28f, v143
	v_div_scale_f32 v143, null, v152, v152, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v143
	v_fma_f32 v149, -v143, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v149, v144
	v_div_scale_f32 v149, vcc_lo, v142, v152, v142
	v_mul_f32_e32 v150, v149, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v143, v150, v149
	v_fmac_f32_e32 v150, v151, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v143, -v143, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v142, v143, v152, v142
	v_add_f32_e32 v149, 0x3d23d70a, v142
	v_div_scale_f32 v142, null, v152, v152, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v143, v142
	v_fma_f32 v144, -v142, v143, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v143, v144, v143
	v_div_scale_f32 v144, vcc_lo, v141, v152, v141
	v_mul_f32_e32 v150, v144, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v142, v150, v144
	v_fmac_f32_e32 v150, v151, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v142, -v142, v150, v144
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v142, v142, v143, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v141, v142, v152, v141
	v_add_f32_e32 v150, 0x3d4ccccc, v141
	v_div_scale_f32 v141, null, v152, v152, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v142, v141
	v_fma_f32 v143, -v141, v142, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v142, v143, v142
	v_div_scale_f32 v143, vcc_lo, v140, v152, v140
	v_mul_f32_e32 v144, v143, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v141, v144, v143
	v_fmac_f32_e32 v144, v151, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v141, -v141, v144, v143
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v141, v141, v142, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v140, v141, v152, v140
	v_add_f32_e32 v151, 0x3d75c28f, v140
	v_div_scale_f32 v140, null, v152, v152, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v141, v140
	v_fma_f32 v142, -v140, v141, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v141, v142, v141
	v_div_scale_f32 v142, vcc_lo, v139, v152, v139
	v_mul_f32_e32 v143, v142, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v144, -v140, v143, v142
	v_fmac_f32_e32 v143, v144, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v140, -v140, v143, v142
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v140, v140, v141, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v139, v140, v152, v139
	v_add_f32_e32 v152, 0x3d8f5c29, v139
.LBB3_7:                                ; %Flow834
                                        ;   in Loop: Header=BB3_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB3_10
; %bb.8:                                ; %.preheader.i.preheader
                                        ;   in Loop: Header=BB3_5 Depth=1
	s_mov_b32 s7, 32
.LBB3_9:                                ; %.preheader.i
                                        ;   Parent Loop BB3_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v154, v153
	v_mov_b32_e32 v156, v155
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s7, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[155:156], v[153:154], v[129:136]
	s_cbranch_scc0 .LBB3_9
.LBB3_10:                               ; %Flow835
                                        ;   in Loop: Header=BB3_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s6, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB3_14
; %bb.11:                               ; %.preheader.i72.preheader
                                        ;   in Loop: Header=BB3_5 Depth=1
	s_mov_b32 s7, 32
.LBB3_12:                               ; %.preheader.i72
                                        ;   Parent Loop BB3_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v154, v153
	v_mov_b32_e32 v156, v155
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s7, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s7, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[155:156], v[153:154], v[129:136]
	s_cbranch_scc1 .LBB3_12
; %bb.13:                               ; %_Z6matrixILi1EEvRAT__Dv8_fDv2_iS3_.exit76.loopexit
                                        ;   in Loop: Header=BB3_5 Depth=1
                                        ; implicit-def: $vgpr138
.LBB3_14:                               ; %Flow
                                        ;   in Loop: Header=BB3_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB3_4
; %bb.15:                               ;   in Loop: Header=BB3_5 Depth=1
	v_max3_num_f32 v139, v145, 0xff800000, v146
	v_cmp_gt_u32_e32 vcc_lo, 32, v138
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_num_f32 v139, v139, v147, v148
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v138, v158, v138, vcc_lo
	v_max3_num_f32 v139, v139, v149, v150
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v154, 2, v138
	v_max3_num_f32 v139, v139, v151, v152
	ds_bpermute_b32 v138, v154, v139
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v138, v138, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v138, v139, v138
	v_dual_sub_f32 v139, v145, v138 :: v_dual_sub_f32 v140, v146, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v142, v147, v138 :: v_dual_mul_f32 v139, 0x3fb8aa3b, v139
	v_mul_f32_e32 v140, 0x3fb8aa3b, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v142, 0x3fb8aa3b, v142
	v_exp_f32_e32 v139, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v140, v140
	v_exp_f32_e32 v142, v142
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_mul_f32_e32 v145, 0x3a83126f, v139
	v_dual_add_f32 v141, v139, v140 :: v_dual_mul_f32 v144, 0x3b03126f, v140
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v141, v142, v141
	v_sub_f32_e32 v143, v148, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v139, v145, 0, v144
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v148, v143
	v_sub_f32_e32 v143, v149, v138
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v141, v148, v141
	v_exp_f32_e32 v149, v143
	v_sub_f32_e32 v143, v150, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	v_exp_f32_e32 v150, v143
	v_dual_sub_f32 v143, v151, v138 :: v_dual_sub_f32 v138, v152, v138
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v141, v149, v141
	v_dual_mul_f32 v143, 0x3fb8aa3b, v143 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v141, v150, v141 :: v_dual_mul_f32 v140, 0x3bc49ba6, v150
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v151, v143
	v_exp_f32_e32 v138, v138
	v_mul_f32_e32 v143, 0x3b449ba6, v142
	v_mul_f32_e32 v142, 0x3b83126f, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_max3_num_f32 v139, v139, v143, v142
	v_add_f32_e32 v141, v151, v141
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v146, v138, v141
	v_mul_f32_e32 v141, 0x3ba3d70b, v149
	v_mul_f32_e32 v138, 0x3c03126f, v138
	ds_bpermute_b32 v147, v154, v146
	v_max3_num_f32 v148, v139, v141, v140
	v_mul_f32_e32 v139, 0x3be56042, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v148, v148, v139, v138
	ds_bpermute_b32 v149, v154, v148
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v149, v149, v149
	v_max_num_f32_e32 v148, v148, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v149, null, 0x43e00000, 0x43e00000, v148
	v_rcp_f32_e32 v150, v149
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v149, v150, 1.0
	v_fmac_f32_e32 v150, v151, v150
	v_div_scale_f32 v151, vcc_lo, v148, 0x43e00000, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v152, v151, v150
	v_fma_f32 v154, -v149, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v152, v154, v150
	v_fma_f32 v149, -v149, v152, v151
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v149, v149, v150, v152
	v_div_fixup_f32 v148, v149, 0x43e00000, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v148, 0x1f800000, v148
	v_div_scale_f32 v149, null, v148, v148, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v151, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v150
	v_div_scale_f32 v151, vcc_lo, v145, v148, v145
	v_mul_f32_e32 v152, v151, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v149, v152, v151
	v_fmac_f32_e32 v152, v154, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v149, v152, v151
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v152
	v_div_scale_f32 v150, null, v148, v148, v144
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v149, v149, v148, v145
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v150, v151, 1.0
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v144, v148, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v154, v152, v151
	v_fma_f32 v156, -v150, v154, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v156, v151
	v_fma_f32 v150, -v150, v154, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v150, v150, v151, v154
	v_mov_b16_e64 v151.l, v137.l
	v_mov_b16_e64 v151.h, 0
	v_div_fixup_f32 v150, v150, v148, v144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v152.l, v151.l
	v_mov_b16_e64 v152.h, v151.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v152.l, v149, v150
	v_div_scale_f32 v149, null, v148, v148, v143
	v_rcp_f32_e32 v150, v149
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v149, v150, 1.0
	v_fmac_f32_e32 v150, v154, v150
	v_div_scale_f32 v154, vcc_lo, v143, v148, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v154, v150
	v_fma_f32 v159, -v149, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v159, v150
	v_fma_f32 v149, -v149, v156, v154
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v149, v149, v150, v156
	v_div_scale_f32 v150, null, v148, v148, v142
	v_div_fixup_f32 v149, v149, v148, v143
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v154, v150
	v_fma_f32 v156, -v150, v154, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v156, v154
	v_div_scale_f32 v156, vcc_lo, v142, v148, v142
	v_mul_f32_e32 v159, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v150, v159, v156
	v_fmac_f32_e32 v159, v160, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v154, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v150, v150, v148, v142
	v_cvt_pk_fp8_f32 v152.h, v149, v150
	v_div_scale_f32 v149, null, v148, v148, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v154, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v154, v150
	v_div_scale_f32 v154, vcc_lo, v141, v148, v141
	v_mul_f32_e32 v156, v154, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v149, v156, v154
	v_fmac_f32_e32 v156, v159, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v149, v156, v154
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v156
	v_div_scale_f32 v150, null, v148, v148, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v149, v149, v148, v141
	v_rcp_f32_e32 v154, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v150, v154, 1.0
	v_fmac_f32_e32 v154, v156, v154
	v_div_scale_f32 v156, vcc_lo, v140, v148, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v156, v154
	v_fma_f32 v160, -v150, v159, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v154
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v150, v150, v154, v159
	v_div_fixup_f32 v150, v150, v148, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v151.l, v149, v150
	v_div_scale_f32 v149, null, v148, v148, v139
	v_rcp_f32_e32 v150, v149
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v149, v150, 1.0
	v_fmac_f32_e32 v150, v154, v150
	v_div_scale_f32 v154, vcc_lo, v139, v148, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v154, v150
	v_fma_f32 v159, -v149, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v159, v150
	v_fma_f32 v149, -v149, v156, v154
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v149, v149, v150, v156
	v_div_scale_f32 v150, null, v148, v148, v138
	v_div_fixup_f32 v149, v149, v148, v139
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v154, v150
	v_fma_f32 v156, -v150, v154, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v156, v154
	v_div_scale_f32 v156, vcc_lo, v138, v148, v138
	v_mul_f32_e32 v159, v156, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v150, v159, v156
	v_fmac_f32_e32 v159, v160, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v150, v159, v156
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v150, v150, v154, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v148, v150, v148, v138
	v_cvt_pk_fp8_f32 v151.h, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v148, v152, v151
	v_add_f32_e32 v152, v146, v147
	v_cvt_f32_ubyte0_e32 v148, v148
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v146, null, v152, v152, v145
	v_fmaak_f32 v148, s3, v148, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v147, v146
	v_dual_mul_f32 v8, v8, v148 :: v_dual_mul_f32 v31, v31, v148
	v_dual_mul_f32 v7, v7, v148 :: v_dual_mul_f32 v6, v6, v148
	v_dual_mul_f32 v5, v5, v148 :: v_dual_mul_f32 v4, v4, v148
	v_mul_f32_e32 v27, v27, v148
	v_dual_mul_f32 v3, v3, v148 :: v_dual_mul_f32 v2, v2, v148
	v_dual_mul_f32 v1, v1, v148 :: v_dual_mul_f32 v16, v16, v148
	v_mul_f32_e32 v25, v25, v148
	v_dual_mul_f32 v15, v15, v148 :: v_dual_mul_f32 v14, v14, v148
	v_dual_mul_f32 v13, v13, v148 :: v_dual_mul_f32 v12, v12, v148
	v_mul_f32_e32 v39, v39, v148
	v_dual_mul_f32 v11, v11, v148 :: v_dual_mul_f32 v10, v10, v148
	v_mul_f32_e32 v37, v37, v148
	v_dual_mul_f32 v9, v9, v148 :: v_dual_mul_f32 v24, v24, v148
	v_mul_f32_e32 v35, v35, v148
	v_dual_mul_f32 v23, v23, v148 :: v_dual_mul_f32 v22, v22, v148
	v_mul_f32_e32 v33, v33, v148
	v_dual_mul_f32 v21, v21, v148 :: v_dual_mul_f32 v20, v20, v148
	v_mul_f32_e32 v47, v47, v148
	v_dual_mul_f32 v19, v19, v148 :: v_dual_mul_f32 v18, v18, v148
	v_mul_f32_e32 v45, v45, v148
	v_dual_mul_f32 v17, v17, v148 :: v_dual_mul_f32 v32, v32, v148
	v_dual_mul_f32 v43, v43, v148 :: v_dual_mul_f32 v30, v30, v148
	v_mul_f32_e32 v41, v41, v148
	v_dual_mul_f32 v29, v29, v148 :: v_dual_mul_f32 v28, v28, v148
	v_dual_mul_f32 v55, v55, v148 :: v_dual_mul_f32 v26, v26, v148
	v_dual_mul_f32 v53, v53, v148 :: v_dual_mul_f32 v40, v40, v148
	v_dual_mul_f32 v51, v51, v148 :: v_dual_mul_f32 v38, v38, v148
	v_dual_mul_f32 v49, v49, v148 :: v_dual_mul_f32 v36, v36, v148
	v_dual_mul_f32 v63, v63, v148 :: v_dual_mul_f32 v34, v34, v148
	v_dual_mul_f32 v61, v61, v148 :: v_dual_mul_f32 v48, v48, v148
	v_dual_mul_f32 v59, v59, v148 :: v_dual_mul_f32 v46, v46, v148
	v_dual_mul_f32 v57, v57, v148 :: v_dual_mul_f32 v44, v44, v148
	v_dual_mul_f32 v71, v71, v148 :: v_dual_mul_f32 v42, v42, v148
	v_dual_mul_f32 v69, v69, v148 :: v_dual_mul_f32 v56, v56, v148
	v_dual_mul_f32 v67, v67, v148 :: v_dual_mul_f32 v54, v54, v148
	v_dual_mul_f32 v65, v65, v148 :: v_dual_mul_f32 v52, v52, v148
	v_dual_mul_f32 v79, v79, v148 :: v_dual_mul_f32 v50, v50, v148
	v_dual_mul_f32 v77, v77, v148 :: v_dual_mul_f32 v64, v64, v148
	v_dual_mul_f32 v75, v75, v148 :: v_dual_mul_f32 v62, v62, v148
	v_dual_mul_f32 v73, v73, v148 :: v_dual_mul_f32 v60, v60, v148
	v_dual_mul_f32 v87, v87, v148 :: v_dual_mul_f32 v58, v58, v148
	v_dual_mul_f32 v85, v85, v148 :: v_dual_mul_f32 v72, v72, v148
	v_dual_mul_f32 v83, v83, v148 :: v_dual_mul_f32 v70, v70, v148
	v_dual_mul_f32 v81, v81, v148 :: v_dual_mul_f32 v68, v68, v148
	v_dual_mul_f32 v95, v95, v148 :: v_dual_mul_f32 v66, v66, v148
	v_dual_mul_f32 v93, v93, v148 :: v_dual_mul_f32 v80, v80, v148
	v_dual_mul_f32 v91, v91, v148 :: v_dual_mul_f32 v78, v78, v148
	v_dual_mul_f32 v89, v89, v148 :: v_dual_mul_f32 v76, v76, v148
	v_dual_mul_f32 v127, v127, v148 :: v_dual_mul_f32 v74, v74, v148
	v_mul_f32_e32 v103, v148, v103
	v_dual_mul_f32 v88, v88, v148 :: v_dual_mul_f32 v125, v125, v148
	v_dual_mul_f32 v86, v86, v148 :: v_dual_mul_f32 v101, v148, v101
	v_dual_mul_f32 v84, v84, v148 :: v_dual_mul_f32 v123, v123, v148
	v_dual_mul_f32 v82, v82, v148 :: v_dual_mul_f32 v99, v148, v99
	v_dual_mul_f32 v96, v96, v148 :: v_dual_mul_f32 v121, v121, v148
	v_dual_mul_f32 v94, v94, v148 :: v_dual_mul_f32 v97, v148, v97
	v_mul_f32_e32 v92, v92, v148
	v_dual_mul_f32 v90, v90, v148 :: v_dual_mul_f32 v111, v148, v111
	v_dual_mul_f32 v104, v148, v104 :: v_dual_mul_f32 v109, v148, v109
	v_dual_mul_f32 v102, v148, v102 :: v_dual_mul_f32 v107, v148, v107
	v_dual_mul_f32 v100, v148, v100 :: v_dual_mul_f32 v105, v148, v105
	v_dual_mul_f32 v98, v148, v98 :: v_dual_mul_f32 v119, v148, v119
	v_dual_mul_f32 v112, v148, v112 :: v_dual_mul_f32 v117, v148, v117
	v_dual_mul_f32 v110, v148, v110 :: v_dual_mul_f32 v115, v148, v115
	v_dual_mul_f32 v108, v148, v108 :: v_dual_mul_f32 v113, v148, v113
	v_mul_f32_e32 v106, v148, v106
	v_mul_f32_e32 v120, v148, v120
	v_mul_f32_e32 v118, v148, v118
	v_mul_f32_e32 v116, v148, v116
	v_mul_f32_e32 v114, v148, v114
	v_mul_f32_e32 v128, v128, v148
	v_mul_f32_e32 v126, v126, v148
	v_mul_f32_e32 v124, v124, v148
	v_mul_f32_e32 v122, v122, v148
	v_fma_f32 v148, -v146, v147, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v147
	v_div_scale_f32 v148, vcc_lo, v145, v152, v145
	v_mul_f32_e32 v149, v148, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v146, v149, v148
	v_fmac_f32_e32 v149, v150, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v146, -v146, v149, v148
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v146, v146, v147, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v145, v146, v152, v145
	v_div_scale_f32 v146, null, v152, v152, v144
	v_add_f32_e32 v145, 0, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v147, v146
	v_fma_f32 v148, -v146, v147, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v147
	v_div_scale_f32 v148, vcc_lo, v144, v152, v144
	v_mul_f32_e32 v149, v148, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v146, v149, v148
	v_fmac_f32_e32 v149, v150, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v146, -v146, v149, v148
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v146, v146, v147, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v144, v146, v152, v144
	v_add_f32_e32 v146, 0x3c23d70a, v144
	v_div_scale_f32 v144, null, v152, v152, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v147, v144
	v_fma_f32 v148, -v144, v147, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v147, v148, v147
	v_div_scale_f32 v148, vcc_lo, v143, v152, v143
	v_mul_f32_e32 v149, v148, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v144, v149, v148
	v_fmac_f32_e32 v149, v150, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v144, -v144, v149, v148
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v144, v144, v147, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v143, v144, v152, v143
	v_add_f32_e32 v147, 0x3ca3d70a, v143
	v_div_scale_f32 v143, null, v152, v152, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v143
	v_fma_f32 v148, -v143, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v148, v144
	v_div_scale_f32 v148, vcc_lo, v142, v152, v142
	v_mul_f32_e32 v149, v148, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v143, v149, v148
	v_fmac_f32_e32 v149, v150, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v143, -v143, v149, v148
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v142, v143, v152, v142
	v_add_f32_e32 v148, 0x3cf5c28f, v142
	v_div_scale_f32 v142, null, v152, v152, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v143, v142
	v_fma_f32 v144, -v142, v143, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v143, v144, v143
	v_div_scale_f32 v144, vcc_lo, v141, v152, v141
	v_mul_f32_e32 v149, v144, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v142, v149, v144
	v_fmac_f32_e32 v149, v150, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v142, -v142, v149, v144
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v142, v142, v143, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v141, v142, v152, v141
	v_add_f32_e32 v149, 0x3d23d70a, v141
	v_div_scale_f32 v141, null, v152, v152, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v142, v141
	v_fma_f32 v143, -v141, v142, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v142, v143, v142
	v_div_scale_f32 v143, vcc_lo, v140, v152, v140
	v_mul_f32_e32 v144, v143, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v150, -v141, v144, v143
	v_fmac_f32_e32 v144, v150, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v141, -v141, v144, v143
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v141, v141, v142, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v140, v141, v152, v140
	v_add_f32_e32 v150, 0x3d4ccccc, v140
	v_div_scale_f32 v140, null, v152, v152, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v141, v140
	v_fma_f32 v142, -v140, v141, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v141, v142, v141
	v_div_scale_f32 v142, vcc_lo, v139, v152, v139
	v_mul_f32_e32 v143, v142, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v144, -v140, v143, v142
	v_fmac_f32_e32 v143, v144, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v140, -v140, v143, v142
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v140, v140, v141, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v139, v140, v152, v139
	v_add_f32_e32 v151, 0x3d75c28f, v139
	v_div_scale_f32 v139, null, v152, v152, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v140, v139
	v_fma_f32 v141, -v139, v140, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v140, v141, v140
	v_div_scale_f32 v141, vcc_lo, v138, v152, v138
	v_mul_f32_e32 v142, v141, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v143, -v139, v142, v141
	v_fmac_f32_e32 v142, v143, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v139, -v139, v142, v141
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v139, v139, v140, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v138, v139, v152, v138
	v_add_f32_e32 v152, 0x3d8f5c29, v138
	s_branch .LBB3_4
.LBB3_16:
	v_dual_mov_b32 v125, 0x3f93b645 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v126, 0x3f93d70a :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v127, 0x3f93f7cf :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v128, 0x3f941893 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
.LBB3_17:                               ; %.preheader95
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v129, 0, v129
	v_lshl_or_b32 v0, ttmp9, 8, v0
	v_add_f32_e32 v129, v129, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v131
	v_add_f32_e32 v129, v129, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v133
	v_add_f32_e32 v129, v129, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v135
	v_add_f32_e32 v129, v129, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v129, v1
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v157
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	v_add_f32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v8
	v_add_f32_e32 v1, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	v_add_f32_e32 v1, v1, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v12
	v_add_f32_e32 v1, v1, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v14
	v_add_f32_e32 v1, v1, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v16
	v_add_f32_e32 v1, v1, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v18
	v_add_f32_e32 v1, v1, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v20
	v_add_f32_e32 v1, v1, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v22
	v_add_f32_e32 v1, v1, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v24
	v_add_f32_e32 v1, v1, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v26
	v_add_f32_e32 v1, v1, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v28
	v_add_f32_e32 v1, v1, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v30
	v_add_f32_e32 v1, v1, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v32
	v_add_f32_e32 v1, v1, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v34
	v_add_f32_e32 v1, v1, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v36
	v_add_f32_e32 v1, v1, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v38
	v_add_f32_e32 v1, v1, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v40
	v_add_f32_e32 v1, v1, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v42
	v_add_f32_e32 v1, v1, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v44
	v_add_f32_e32 v1, v1, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v46
	v_add_f32_e32 v1, v1, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v48
	v_add_f32_e32 v1, v1, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v50
	v_add_f32_e32 v1, v1, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v52
	v_add_f32_e32 v1, v1, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v54
	v_add_f32_e32 v1, v1, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v56
	v_add_f32_e32 v1, v1, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v58
	v_add_f32_e32 v1, v1, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v60
	v_add_f32_e32 v1, v1, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v62
	v_add_f32_e32 v1, v1, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v64
	v_add_f32_e32 v1, v1, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v66
	v_add_f32_e32 v1, v1, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v68
	v_add_f32_e32 v1, v1, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v70
	v_add_f32_e32 v1, v1, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v72
	v_add_f32_e32 v1, v1, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v74
	v_add_f32_e32 v1, v1, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v76
	v_add_f32_e32 v1, v1, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v78
	v_add_f32_e32 v1, v1, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v80
	v_add_f32_e32 v1, v1, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v82
	v_add_f32_e32 v1, v1, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v84
	v_add_f32_e32 v1, v1, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v86
	v_add_f32_e32 v1, v1, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v88
	v_add_f32_e32 v1, v1, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v90
	v_add_f32_e32 v1, v1, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v92
	v_add_f32_e32 v1, v1, v93
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v94
	v_add_f32_e32 v1, v1, v95
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v96
	v_add_f32_e32 v1, v1, v97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v98
	v_add_f32_e32 v1, v1, v99
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v100
	v_add_f32_e32 v1, v1, v101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v102
	v_add_f32_e32 v1, v1, v103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v104
	v_add_f32_e32 v1, v1, v105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v106
	v_add_f32_e32 v1, v1, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v108
	v_add_f32_e32 v1, v1, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v110
	v_add_f32_e32 v1, v1, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v112
	v_add_f32_e32 v1, v1, v113
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v114
	v_add_f32_e32 v1, v1, v115
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v116
	v_add_f32_e32 v1, v1, v117
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v118
	v_add_f32_e32 v1, v1, v119
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v120
	v_add_f32_e32 v1, v1, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v122
	v_add_f32_e32 v1, v1, v123
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v124
	v_add_f32_e32 v1, v1, v125
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v126
	v_add_f32_e32 v1, v1, v127
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v128
	v_add_f32_e32 v1, v1, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v146
	v_add_f32_e32 v1, v1, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v148
	v_add_f32_e32 v1, v1, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v150
	v_add_f32_e32 v3, v1, v151
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v152
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmamk_f32 v2, v2, 0x33800000, v3
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
	.size	_Z3runILi3ELi1EEvPfPji, .Lfunc_end3-_Z3runILi3ELi1EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi3ELi1EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 162
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_Z3runILi3ELi1EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi3ELi1EEvPfPji,"axG",@progbits,_Z3runILi3ELi1EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi3ELi1EEvPfPji.num_vgpr, 162
	.set .L_Z3runILi3ELi1EEvPfPji.num_agpr, 0
	.set .L_Z3runILi3ELi1EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi3ELi1EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi3ELi1EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi3ELi1EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi3ELi1EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi3ELi1EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi3ELi1EEvPfPji.has_recursion, 0
	.set .L_Z3runILi3ELi1EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8104
; TotalNumSgprs: 10
; NumVgprs: 162
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 162
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi4ELi1EEvPfPji,"axG",@progbits,_Z3runILi4ELi1EEvPfPji,comdat
	.protected	_Z3runILi4ELi1EEvPfPji  ; -- Begin function _Z3runILi4ELi1EEvPfPji
	.globl	_Z3runILi4ELi1EEvPfPji
	.p2align	8
	.type	_Z3runILi4ELi1EEvPfPji,@function
_Z3runILi4ELi1EEvPfPji:                 ; @_Z3runILi4ELi1EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB4_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB4_2:                                ; %.preheader80
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s1, s[0:1], 0x10
	v_lshl_add_u32 v21, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s0, 0x3a83126f
	s_mov_b32 s2, 0
	ds_store_b32 v21, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v150, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v151, s0, v1, 0x3dcccccd
	v_fmaak_f32 v152, s0, v1, 0x3e4ccccd
	v_fmaak_f32 v154, s0, v1, 0x3e99999a
	v_fmaak_f32 v155, s0, v1, 0x3ecccccd
	v_fma_f32 v156, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v157, s0, v1, 0x3f19999a
	v_fmaak_f32 v158, s0, v1, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s1, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_10
; %bb.3:                                ; %.lr.ph
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v22, 0x3f941893
	v_cmp_lt_u32_e64 s0, 0x7f, v0
	v_dual_mov_b32 v24, 0x3f93d70a :: v_dual_mov_b32 v149, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v23, 0x3f93f7cf
	v_dual_mov_b32 v26, 0x3f939581 :: v_dual_mov_b32 v1, v9
	v_dual_mov_b32 v25, 0x3f93b645 :: v_dual_mov_b32 v2, v10
	v_dual_mov_b32 v27, 0x3f9374bc :: v_dual_mov_b32 v4, v12
	v_dual_mov_b32 v28, 0x3f9353f8 :: v_dual_mov_b32 v3, v11
	v_dual_mov_b32 v29, 0x3f933333 :: v_dual_mov_b32 v6, v14
	v_dual_mov_b32 v30, 0x3f92d0e5 :: v_dual_mov_b32 v5, v13
	v_dual_mov_b32 v31, 0x3f92b021 :: v_dual_mov_b32 v8, v16
	v_dual_mov_b32 v32, 0x3f928f5c :: v_dual_mov_b32 v7, v15
	v_mov_b32_e32 v33, 0x3f926e97
	v_mov_b32_e32 v34, 0x3f924dd3
	v_mov_b32_e32 v35, 0x3f922d0e
	v_mov_b32_e32 v36, 0x3f920c4a
	v_mov_b32_e32 v37, 0x3f91eb85
	v_mov_b32_e32 v38, 0x3f918937
	v_mov_b32_e32 v39, 0x3f916873
	v_mov_b32_e32 v40, 0x3f9147ae
	v_mov_b32_e32 v41, 0x3f9126e9
	v_mov_b32_e32 v42, 0x3f910625
	v_mov_b32_e32 v43, 0x3f90e560
	v_mov_b32_e32 v44, 0x3f90c49c
	v_mov_b32_e32 v45, 0x3f90a3d7
	v_mov_b32_e32 v46, 0x3f904189
	v_mov_b32_e32 v47, 0x3f9020c5
	v_mov_b32_e32 v48, 0x3f900000
	v_mov_b32_e32 v49, 0x3f8fdf3b
	v_mov_b32_e32 v50, 0x3f8fbe77
	v_mov_b32_e32 v51, 0x3f8f9db2
	v_mov_b32_e32 v52, 0x3f8f7cee
	v_mov_b32_e32 v53, 0x3f8f5c29
	v_mov_b32_e32 v54, 0x3f8ef9db
	v_mov_b32_e32 v55, 0x3f8ed917
	v_mov_b32_e32 v56, 0x3f8eb852
	v_mov_b32_e32 v57, 0x3f8e978d
	v_mov_b32_e32 v58, 0x3f8e76c9
	v_mov_b32_e32 v59, 0x3f8e5604
	v_mov_b32_e32 v60, 0x3f8e3540
	v_mov_b32_e32 v61, 0x3f8e147b
	v_mov_b32_e32 v62, 0x3f8db22d
	v_mov_b32_e32 v63, 0x3f8d9169
	v_mov_b32_e32 v64, 0x3f8d70a4
	v_mov_b32_e32 v65, 0x3f8d4fdf
	v_mov_b32_e32 v66, 0x3f8d2f1b
	v_mov_b32_e32 v67, 0x3f8d0e56
	v_mov_b32_e32 v68, 0x3f8ced92
	v_mov_b32_e32 v69, 0x3f8ccccd
	v_mov_b32_e32 v70, 0x3f8c6a7f
	v_mov_b32_e32 v71, 0x3f8c49bb
	v_mov_b32_e32 v72, 0x3f8c28f6
	v_mov_b32_e32 v73, 0x3f8c0831
	v_mov_b32_e32 v74, 0x3f8be76d
	v_mov_b32_e32 v75, 0x3f8bc6a8
	v_mov_b32_e32 v76, 0x3f8ba5e4
	v_mov_b32_e32 v77, 0x3f8b851f
	v_mov_b32_e32 v78, 0x3f8b22d1
	v_mov_b32_e32 v79, 0x3f8b020d
	v_mov_b32_e32 v80, 0x3f8ae148
	v_mov_b32_e32 v81, 0x3f8ac083
	v_mov_b32_e32 v82, 0x3f8a9fbf
	v_mov_b32_e32 v83, 0x3f8a7efa
	v_mov_b32_e32 v84, 0x3f8a5e36
	v_mov_b32_e32 v85, 0x3f8a3d71
	v_mov_b32_e32 v86, 0x3f89db23
	v_mov_b32_e32 v87, 0x3f89ba5f
	v_mov_b32_e32 v88, 0x3f89999a
	v_mov_b32_e32 v89, 0x3f8978d5
	v_mov_b32_e32 v90, 0x3f895811
	v_mov_b32_e32 v91, 0x3f89374c
	v_mov_b32_e32 v92, 0x3f891688
	v_mov_b32_e32 v93, 0x3f88f5c3
	v_mov_b32_e32 v94, 0x3f889374
	v_mov_b32_e32 v95, 0x3f8872b0
	v_mov_b32_e32 v96, 0x3f8851eb
	v_mov_b32_e32 v97, 0x3f883126
	v_mov_b32_e32 v98, 0x3f881062
	v_mov_b32_e32 v99, 0x3f87ef9d
	v_mov_b32_e32 v100, 0x3f87ced9
	v_mov_b32_e32 v101, 0x3f87ae14
	v_mov_b32_e32 v102, 0x3f874bc6
	v_mov_b32_e32 v103, 0x3f872b02
	v_mov_b32_e32 v104, 0x3f870a3d
	v_mov_b32_e32 v105, 0x3f86e978
	v_mov_b32_e32 v106, 0x3f86c8b4
	v_mov_b32_e32 v107, 0x3f86a7ef
	v_mov_b32_e32 v108, 0x3f86872b
	v_mov_b32_e32 v109, 0x3f866666
	v_mov_b32_e32 v110, 0x3f860418
	v_mov_b32_e32 v111, 0x3f85e354
	v_mov_b32_e32 v112, 0x3f85c28f
	v_mov_b32_e32 v113, 0x3f85a1ca
	v_mov_b32_e32 v114, 0x3f858106
	v_mov_b32_e32 v115, 0x3f856041
	v_mov_b32_e32 v116, 0x3f853f7d
	v_mov_b32_e32 v117, 0x3f851eb8
	v_mov_b32_e32 v118, 0x3f84bc6a
	v_mov_b32_e32 v119, 0x3f849ba6
	v_mov_b32_e32 v120, 0x3f847ae1
	v_mov_b32_e32 v121, 0x3f845a1c
	v_mov_b32_e32 v122, 0x3f843958
	v_mov_b32_e32 v123, 0x3f841893
	v_mov_b32_e32 v124, 0x3f83f7cf
	v_mov_b32_e32 v125, 0x3f83d70a
	v_mov_b32_e32 v126, 0x3f8374bc
	v_mov_b32_e32 v127, 0x3f8353f8
	v_mov_b32_e32 v128, 0x3f833333
	v_mov_b32_e32 v129, 0x3f83126e
	v_mov_b32_e32 v130, 0x3f82f1aa
	v_mov_b32_e32 v131, 0x3f82d0e5
	v_mov_b32_e32 v132, 0x3f82b021
	v_mov_b32_e32 v133, 0x3f828f5c
	v_mov_b32_e32 v134, 0x3f822d0e
	v_mov_b32_e32 v135, 0x3f820c4a
	v_mov_b32_e32 v136, 0x3f81eb85
	v_mov_b32_e32 v137, 0x3f81cac0
	v_mov_b32_e32 v138, 0x3f81a9fc
	v_mov_b32_e32 v139, 0x3f818937
	v_mov_b32_e32 v140, 0x3f816873
	v_mov_b32_e32 v141, 0x3f8147ae
	v_mov_b32_e32 v142, 0x3f80e560
	v_mov_b32_e32 v143, 0x3f80c49c
	v_mov_b32_e32 v144, 0x3f80a3d7
	v_mov_b32_e32 v145, 0x3f808312
	v_mov_b32_e32 v146, 0x3f80624e
	v_mov_b32_e32 v147, 0x3f804189
	v_mov_b32_e32 v148, 0x3f8020c5
	v_mbcnt_lo_u32_b32 v153, -1, 0
	v_mov_b32_e32 v17, 0x10101010
	v_mov_b32_e32 v19, 0x18181818
	s_mov_b32 s3, 0x35800000
	s_branch .LBB4_5
.LBB4_4:                                ; %Flow1452
                                        ;   in Loop: Header=BB4_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_add_co_i32 s2, s2, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, s1
	s_cbranch_scc1 .LBB4_11
.LBB4_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_9 Depth 2
	s_and_saveexec_b32 s6, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB4_7
; %bb.6:                                ;   in Loop: Header=BB4_5 Depth=1
	v_max3_num_f32 v10, v150, 0xff800000, v151
	v_xor_b32_e32 v11, 16, v153
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v10, v10, v152, v154
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_num_f32 v10, v10, v155, v156
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v153, v11, vcc_lo
	v_max3_num_f32 v10, v10, v157, v158
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v159, 2, v11
	ds_bpermute_b32 v11, v159, v10
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v11, v11, v11
	v_max_num_f32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v11, v150, v10 :: v_dual_sub_f32 v14, v152, v10
	v_dual_sub_f32 v12, v151, v10 :: v_dual_mul_f32 v11, 0x3fb8aa3b, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	v_mul_f32_e32 v12, 0x3fb8aa3b, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v11, v11
	v_sub_f32_e32 v15, v154, v10
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v14, v14
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v18, 0x3a83126f, v11
	v_mul_f32_e32 v15, 0x3fb8aa3b, v15
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v13, v11, v12 :: v_dual_mul_f32 v16, 0x3b03126f, v12
	v_exp_f32_e32 v151, v15
	v_sub_f32_e32 v15, v155, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v11, v18, 0, v16
	v_mul_f32_e32 v15, 0x3fb8aa3b, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v152, v15
	v_sub_f32_e32 v15, v156, v10
	v_mul_f32_e32 v15, 0x3fb8aa3b, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v154, v15
	v_dual_sub_f32 v15, v157, v10 :: v_dual_sub_f32 v10, v158, v10
	v_add_f32_e32 v13, v14, v13
	v_dual_mul_f32 v15, 0x3fb8aa3b, v15 :: v_dual_mul_f32 v10, 0x3fb8aa3b, v10
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v13, v151, v13 :: v_dual_mul_f32 v12, 0x3bc49ba6, v154
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v155, v15
	v_exp_f32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v13, v152, v13
	v_mul_f32_e32 v15, 0x3b449ba6, v14
	v_dual_mul_f32 v14, 0x3b83126f, v151 :: v_dual_add_f32 v13, v154, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v11, v11, v15, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_2)
	v_add_f32_e32 v13, v155, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_add_f32 v20, v10, v13 :: v_dual_mul_f32 v13, 0x3ba3d70b, v152
	v_mul_f32_e32 v10, 0x3c03126f, v10
	ds_bpermute_b32 v150, v159, v20
	v_max3_num_f32 v151, v11, v13, v12
	v_mul_f32_e32 v11, 0x3be56042, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v151, v151, v11, v10
	ds_bpermute_b32 v152, v159, v151
	s_wait_dscnt 0x1
	v_add_f32_e32 v20, v20, v150
	v_div_scale_f32 v150, null, v20, v20, v18
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v152, v152, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v151, v151, v152
	v_div_scale_f32 v152, null, 0x43e00000, 0x43e00000, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v154, v152
	v_fma_f32 v155, -v152, v154, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v154
	v_div_scale_f32 v155, vcc_lo, v151, 0x43e00000, v151
	v_mul_f32_e32 v156, v155, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v152, v156, v155
	v_fmac_f32_e32 v156, v157, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v156, v155
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v154, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v151, v152, 0x43e00000, v151
	v_max_num_f32_e32 v151, 0x1f800000, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v152, null, v151, v151, v18
	v_rcp_f32_e32 v154, v152
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v155, -v152, v154, 1.0
	v_fmac_f32_e32 v154, v155, v154
	v_div_scale_f32 v155, vcc_lo, v18, v151, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v155, v154
	v_fma_f32 v157, -v152, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v157, v154
	v_fma_f32 v152, -v152, v156, v155
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v152, v152, v154, v156
	v_div_scale_f32 v154, null, v151, v151, v16
	v_div_fixup_f32 v152, v152, v151, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v155, v154
	v_fma_f32 v156, -v154, v155, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v155, v156, v155
	v_div_scale_f32 v156, vcc_lo, v16, v151, v16
	v_mul_f32_e32 v157, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v154, v157, v156
	v_fmac_f32_e32 v157, v158, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v154, v157, v156
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v154, v154, v155, v157
	v_mov_b16_e64 v155.l, v9.l
	v_mov_b16_e64 v155.h, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v154, v154, v151, v16
	v_mov_b16_e64 v156.l, v155.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v156.h, v155.h
	v_cvt_pk_fp8_f32 v156.l, v152, v154
	v_div_scale_f32 v152, null, v151, v151, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v154, v152
	v_fma_f32 v157, -v152, v154, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v157, v154
	v_div_scale_f32 v157, vcc_lo, v15, v151, v15
	v_mul_f32_e32 v158, v157, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v152, v158, v157
	v_fmac_f32_e32 v158, v159, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v158, v157
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v154, v158
	v_div_scale_f32 v154, null, v151, v151, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v152, v152, v151, v15
	v_rcp_f32_e32 v157, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v154, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v14, v151, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v154, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v154, -v154, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v154, v154, v157, v159
	v_div_fixup_f32 v154, v154, v151, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v156.h, v152, v154
	v_div_scale_f32 v152, null, v151, v151, v13
	v_rcp_f32_e32 v154, v152
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v152, v154, 1.0
	v_fmac_f32_e32 v154, v157, v154
	v_div_scale_f32 v157, vcc_lo, v13, v151, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v158, v157, v154
	v_fma_f32 v159, -v152, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v158, v159, v154
	v_fma_f32 v152, -v152, v158, v157
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v152, v152, v154, v158
	v_div_scale_f32 v154, null, v151, v151, v12
	v_div_fixup_f32 v152, v152, v151, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v157, v154
	v_fma_f32 v158, -v154, v157, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v12, v151, v12
	v_mul_f32_e32 v159, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v154, v159, v158
	v_fmac_f32_e32 v159, v160, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v154, v159, v158
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v154, v154, v157, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v154, v154, v151, v12
	v_cvt_pk_fp8_f32 v155.l, v152, v154
	v_div_scale_f32 v152, null, v151, v151, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v154, v152
	v_fma_f32 v157, -v152, v154, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v157, v154
	v_div_scale_f32 v157, vcc_lo, v11, v151, v11
	v_mul_f32_e32 v158, v157, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v152, v158, v157
	v_fmac_f32_e32 v158, v159, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v158, v157
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v154, v158
	v_div_scale_f32 v154, null, v151, v151, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v152, v152, v151, v11
	v_rcp_f32_e32 v157, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v154, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v10, v151, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v154, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v154, -v154, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v154, v154, v157, v159
	v_div_fixup_f32 v151, v154, v151, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v155.h, v152, v151
	v_xor_b32_e32 v151, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v151, v151
	v_fmaak_f32 v151, s3, v151, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v149, v149, v151 :: v_dual_mul_f32 v138, v138, v151
	v_dual_mul_f32 v148, v148, v151 :: v_dual_mul_f32 v147, v147, v151
	v_mul_f32_e32 v134, v134, v151
	v_dual_mul_f32 v146, v146, v151 :: v_dual_mul_f32 v145, v145, v151
	v_mul_f32_e32 v130, v130, v151
	v_dual_mul_f32 v144, v144, v151 :: v_dual_mul_f32 v143, v143, v151
	v_mul_f32_e32 v128, v128, v151
	v_dual_mul_f32 v142, v142, v151 :: v_dual_mul_f32 v141, v141, v151
	v_mul_f32_e32 v126, v126, v151
	v_dual_mul_f32 v140, v140, v151 :: v_dual_mul_f32 v139, v139, v151
	v_dual_mul_f32 v124, v124, v151 :: v_dual_mul_f32 v137, v137, v151
	v_mul_f32_e32 v122, v122, v151
	v_dual_mul_f32 v136, v136, v151 :: v_dual_mul_f32 v135, v135, v151
	v_dual_mul_f32 v120, v120, v151 :: v_dual_mul_f32 v133, v133, v151
	v_mul_f32_e32 v118, v118, v151
	v_dual_mul_f32 v132, v132, v151 :: v_dual_mul_f32 v131, v131, v151
	v_dual_mul_f32 v116, v116, v151 :: v_dual_mul_f32 v129, v129, v151
	v_dual_mul_f32 v114, v114, v151 :: v_dual_mul_f32 v127, v127, v151
	v_dual_mul_f32 v112, v112, v151 :: v_dual_mul_f32 v125, v125, v151
	v_dual_mul_f32 v110, v110, v151 :: v_dual_mul_f32 v123, v123, v151
	v_dual_mul_f32 v108, v108, v151 :: v_dual_mul_f32 v121, v121, v151
	v_dual_mul_f32 v106, v106, v151 :: v_dual_mul_f32 v119, v119, v151
	v_dual_mul_f32 v104, v104, v151 :: v_dual_mul_f32 v117, v117, v151
	v_dual_mul_f32 v102, v102, v151 :: v_dual_mul_f32 v115, v115, v151
	v_dual_mul_f32 v100, v100, v151 :: v_dual_mul_f32 v113, v113, v151
	v_dual_mul_f32 v98, v98, v151 :: v_dual_mul_f32 v111, v111, v151
	v_dual_mul_f32 v96, v96, v151 :: v_dual_mul_f32 v109, v109, v151
	v_dual_mul_f32 v94, v94, v151 :: v_dual_mul_f32 v107, v107, v151
	v_dual_mul_f32 v92, v92, v151 :: v_dual_mul_f32 v105, v105, v151
	v_dual_mul_f32 v90, v90, v151 :: v_dual_mul_f32 v103, v103, v151
	v_dual_mul_f32 v88, v88, v151 :: v_dual_mul_f32 v101, v101, v151
	v_dual_mul_f32 v86, v86, v151 :: v_dual_mul_f32 v99, v99, v151
	v_dual_mul_f32 v84, v84, v151 :: v_dual_mul_f32 v97, v97, v151
	v_dual_mul_f32 v82, v82, v151 :: v_dual_mul_f32 v95, v95, v151
	v_dual_mul_f32 v80, v80, v151 :: v_dual_mul_f32 v93, v93, v151
	v_dual_mul_f32 v78, v78, v151 :: v_dual_mul_f32 v91, v91, v151
	v_dual_mul_f32 v76, v76, v151 :: v_dual_mul_f32 v89, v89, v151
	v_dual_mul_f32 v74, v74, v151 :: v_dual_mul_f32 v87, v87, v151
	v_dual_mul_f32 v72, v72, v151 :: v_dual_mul_f32 v85, v85, v151
	v_dual_mul_f32 v70, v70, v151 :: v_dual_mul_f32 v83, v83, v151
	v_dual_mul_f32 v68, v68, v151 :: v_dual_mul_f32 v81, v81, v151
	v_dual_mul_f32 v66, v66, v151 :: v_dual_mul_f32 v79, v79, v151
	v_dual_mul_f32 v64, v64, v151 :: v_dual_mul_f32 v77, v77, v151
	v_dual_mul_f32 v62, v62, v151 :: v_dual_mul_f32 v75, v75, v151
	v_dual_mul_f32 v60, v60, v151 :: v_dual_mul_f32 v73, v73, v151
	v_dual_mul_f32 v58, v58, v151 :: v_dual_mul_f32 v71, v71, v151
	v_dual_mul_f32 v56, v56, v151 :: v_dual_mul_f32 v69, v69, v151
	v_dual_mul_f32 v54, v54, v151 :: v_dual_mul_f32 v67, v67, v151
	v_dual_mul_f32 v65, v65, v151 :: v_dual_mul_f32 v52, v151, v52
	v_mul_f32_e32 v63, v63, v151
	v_dual_mul_f32 v61, v61, v151 :: v_dual_mul_f32 v50, v151, v50
	v_mul_f32_e32 v59, v59, v151
	v_dual_mul_f32 v57, v57, v151 :: v_dual_mul_f32 v48, v151, v48
	v_mul_f32_e32 v55, v55, v151
	v_dual_mul_f32 v53, v151, v53 :: v_dual_mul_f32 v46, v151, v46
	v_dual_mul_f32 v51, v151, v51 :: v_dual_mul_f32 v44, v151, v44
	v_dual_mul_f32 v49, v151, v49 :: v_dual_mul_f32 v42, v151, v42
	v_dual_mul_f32 v47, v151, v47 :: v_dual_mul_f32 v40, v151, v40
	v_dual_mul_f32 v45, v151, v45 :: v_dual_mul_f32 v38, v151, v38
	v_dual_mul_f32 v43, v151, v43 :: v_dual_mul_f32 v36, v151, v36
	v_dual_mul_f32 v41, v151, v41 :: v_dual_mul_f32 v34, v151, v34
	v_dual_mul_f32 v39, v151, v39 :: v_dual_mul_f32 v32, v151, v32
	v_dual_mul_f32 v37, v151, v37 :: v_dual_mul_f32 v30, v151, v30
	v_dual_mul_f32 v35, v151, v35 :: v_dual_mul_f32 v28, v151, v28
	v_dual_mul_f32 v33, v151, v33 :: v_dual_mul_f32 v26, v151, v26
	v_dual_mul_f32 v31, v151, v31 :: v_dual_mul_f32 v24, v151, v24
	v_dual_mul_f32 v29, v151, v29 :: v_dual_mul_f32 v22, v151, v22
	v_mul_f32_e32 v27, v151, v27
	v_mul_f32_e32 v25, v151, v25
	v_mul_f32_e32 v23, v151, v23
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v150, v151, 1.0
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v18, v20, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v154, v152, v151
	v_fma_f32 v155, -v150, v154, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	v_fma_f32 v150, -v150, v154, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v150, v150, v151, v154
	v_div_fixup_f32 v18, v150, v20, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v150, 0, v18
	v_div_scale_f32 v18, null, v20, v20, v16
	v_rcp_f32_e32 v151, v18
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v18, v151, 1.0
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v16, v20, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v154, v152, v151
	v_fma_f32 v155, -v18, v154, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v151
	v_fma_f32 v18, -v18, v154, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v18, v18, v151, v154
	v_div_fixup_f32 v16, v18, v20, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v151, 0x3c23d70a, v16
	v_div_scale_f32 v16, null, v20, v20, v15
	v_rcp_f32_e32 v18, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v16, v18, 1.0
	v_fmac_f32_e32 v18, v152, v18
	v_div_scale_f32 v152, vcc_lo, v15, v20, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v154, v152, v18
	v_fma_f32 v155, -v16, v154, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v18
	v_fma_f32 v16, -v16, v154, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v18, v154
	v_div_fixup_f32 v15, v16, v20, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v152, 0x3ca3d70a, v15
	v_div_scale_f32 v15, null, v20, v20, v14
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v15, v16, 1.0
	v_fmac_f32_e32 v16, v18, v16
	v_div_scale_f32 v18, vcc_lo, v14, v20, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v154, v18, v16
	v_fma_f32 v155, -v15, v154, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v154, v155, v16
	v_fma_f32 v15, -v15, v154, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v15, v15, v16, v154
	v_div_fixup_f32 v14, v15, v20, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v154, 0x3cf5c28f, v14
	v_div_scale_f32 v14, null, v20, v20, v13
	v_rcp_f32_e32 v15, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v14, v15, 1.0
	v_fmac_f32_e32 v15, v16, v15
	v_div_scale_f32 v16, vcc_lo, v13, v20, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v18, v16, v15
	v_fma_f32 v155, -v14, v18, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v155, v15
	v_fma_f32 v14, -v14, v18, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v14, v14, v15, v18
	v_div_fixup_f32 v13, v14, v20, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v155, 0x3d23d70a, v13
	v_div_scale_f32 v13, null, v20, v20, v12
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v12, v20, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v18, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v18, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v12, v13, v20, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v156, 0x3d4ccccc, v12
	v_div_scale_f32 v12, null, v20, v20, v11
	v_rcp_f32_e32 v13, v12
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v12, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	v_div_scale_f32 v14, vcc_lo, v11, v20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v13
	v_fma_f32 v16, -v12, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v13
	v_fma_f32 v12, -v12, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v12, v12, v13, v15
	v_div_fixup_f32 v11, v12, v20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v157, 0x3d75c28f, v11
	v_div_scale_f32 v11, null, v20, v20, v10
	v_rcp_f32_e32 v12, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v11, v12, 1.0
	v_fmac_f32_e32 v12, v13, v12
	v_div_scale_f32 v13, vcc_lo, v10, v20, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v13, v12
	v_fma_f32 v15, -v11, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v15, v12
	v_fma_f32 v11, -v11, v14, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v12, v14
	v_div_fixup_f32 v10, v11, v20, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v158, 0x3d8f5c29, v10
.LBB4_7:                                ; %Flow1451
                                        ;   in Loop: Header=BB4_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s6, s6
	s_cbranch_execz .LBB4_4
; %bb.8:                                ; %.preheader.i.preheader
                                        ;   in Loop: Header=BB4_5 Depth=1
	s_mov_b32 s7, 32
.LBB4_9:                                ; %.preheader.i
                                        ;   Parent Loop BB4_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v20, v19
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s7, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[19:20], v[17:18], v[1:8]
	s_cbranch_scc0 .LBB4_9
	s_branch .LBB4_4
.LBB4_10:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v148, 0x3f8020c5
	v_mov_b32_e32 v147, 0x3f804189
	v_mov_b32_e32 v146, 0x3f80624e
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v145, 0x3f808312 :: v_dual_mov_b32 v2, v1
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v6, v1
	v_dual_mov_b32 v7, v1 :: v_dual_mov_b32 v8, v1
	v_dual_mov_b32 v149, 1.0 :: v_dual_mov_b32 v144, 0x3f80a3d7
	v_mov_b32_e32 v143, 0x3f80c49c
	v_mov_b32_e32 v142, 0x3f80e560
	v_mov_b32_e32 v141, 0x3f8147ae
	v_mov_b32_e32 v140, 0x3f816873
	v_mov_b32_e32 v139, 0x3f818937
	v_mov_b32_e32 v138, 0x3f81a9fc
	v_mov_b32_e32 v137, 0x3f81cac0
	v_mov_b32_e32 v136, 0x3f81eb85
	v_mov_b32_e32 v135, 0x3f820c4a
	v_mov_b32_e32 v134, 0x3f822d0e
	v_mov_b32_e32 v133, 0x3f828f5c
	v_mov_b32_e32 v132, 0x3f82b021
	v_mov_b32_e32 v131, 0x3f82d0e5
	v_mov_b32_e32 v130, 0x3f82f1aa
	v_mov_b32_e32 v129, 0x3f83126e
	v_mov_b32_e32 v128, 0x3f833333
	v_mov_b32_e32 v127, 0x3f8353f8
	v_mov_b32_e32 v126, 0x3f8374bc
	v_mov_b32_e32 v125, 0x3f83d70a
	v_mov_b32_e32 v124, 0x3f83f7cf
	v_mov_b32_e32 v123, 0x3f841893
	v_mov_b32_e32 v122, 0x3f843958
	v_mov_b32_e32 v121, 0x3f845a1c
	v_mov_b32_e32 v120, 0x3f847ae1
	v_mov_b32_e32 v119, 0x3f849ba6
	v_mov_b32_e32 v118, 0x3f84bc6a
	v_mov_b32_e32 v117, 0x3f851eb8
	v_mov_b32_e32 v116, 0x3f853f7d
	v_mov_b32_e32 v115, 0x3f856041
	v_mov_b32_e32 v114, 0x3f858106
	v_mov_b32_e32 v113, 0x3f85a1ca
	v_mov_b32_e32 v112, 0x3f85c28f
	v_mov_b32_e32 v111, 0x3f85e354
	v_mov_b32_e32 v110, 0x3f860418
	v_mov_b32_e32 v109, 0x3f866666
	v_mov_b32_e32 v108, 0x3f86872b
	v_mov_b32_e32 v107, 0x3f86a7ef
	v_mov_b32_e32 v106, 0x3f86c8b4
	v_mov_b32_e32 v105, 0x3f86e978
	v_mov_b32_e32 v104, 0x3f870a3d
	v_mov_b32_e32 v103, 0x3f872b02
	v_mov_b32_e32 v102, 0x3f874bc6
	v_mov_b32_e32 v101, 0x3f87ae14
	v_mov_b32_e32 v100, 0x3f87ced9
	v_mov_b32_e32 v99, 0x3f87ef9d
	v_mov_b32_e32 v98, 0x3f881062
	v_mov_b32_e32 v97, 0x3f883126
	v_mov_b32_e32 v96, 0x3f8851eb
	v_mov_b32_e32 v95, 0x3f8872b0
	v_mov_b32_e32 v94, 0x3f889374
	v_mov_b32_e32 v93, 0x3f88f5c3
	v_mov_b32_e32 v92, 0x3f891688
	v_mov_b32_e32 v91, 0x3f89374c
	v_mov_b32_e32 v90, 0x3f895811
	v_mov_b32_e32 v89, 0x3f8978d5
	v_mov_b32_e32 v88, 0x3f89999a
	v_mov_b32_e32 v87, 0x3f89ba5f
	v_mov_b32_e32 v86, 0x3f89db23
	v_mov_b32_e32 v85, 0x3f8a3d71
	v_mov_b32_e32 v84, 0x3f8a5e36
	v_mov_b32_e32 v83, 0x3f8a7efa
	v_mov_b32_e32 v82, 0x3f8a9fbf
	v_mov_b32_e32 v81, 0x3f8ac083
	v_mov_b32_e32 v80, 0x3f8ae148
	v_mov_b32_e32 v79, 0x3f8b020d
	v_mov_b32_e32 v78, 0x3f8b22d1
	v_mov_b32_e32 v77, 0x3f8b851f
	v_mov_b32_e32 v76, 0x3f8ba5e4
	v_mov_b32_e32 v75, 0x3f8bc6a8
	v_mov_b32_e32 v74, 0x3f8be76d
	v_mov_b32_e32 v73, 0x3f8c0831
	v_mov_b32_e32 v72, 0x3f8c28f6
	v_mov_b32_e32 v71, 0x3f8c49bb
	v_mov_b32_e32 v70, 0x3f8c6a7f
	v_mov_b32_e32 v69, 0x3f8ccccd
	v_mov_b32_e32 v68, 0x3f8ced92
	v_mov_b32_e32 v67, 0x3f8d0e56
	v_mov_b32_e32 v66, 0x3f8d2f1b
	v_mov_b32_e32 v65, 0x3f8d4fdf
	v_mov_b32_e32 v64, 0x3f8d70a4
	v_mov_b32_e32 v63, 0x3f8d9169
	v_mov_b32_e32 v62, 0x3f8db22d
	v_mov_b32_e32 v61, 0x3f8e147b
	v_mov_b32_e32 v60, 0x3f8e3540
	v_mov_b32_e32 v59, 0x3f8e5604
	v_mov_b32_e32 v58, 0x3f8e76c9
	v_mov_b32_e32 v57, 0x3f8e978d
	v_mov_b32_e32 v56, 0x3f8eb852
	v_mov_b32_e32 v55, 0x3f8ed917
	v_mov_b32_e32 v54, 0x3f8ef9db
	v_mov_b32_e32 v53, 0x3f8f5c29
	v_mov_b32_e32 v52, 0x3f8f7cee
	v_mov_b32_e32 v51, 0x3f8f9db2
	v_mov_b32_e32 v50, 0x3f8fbe77
	v_mov_b32_e32 v49, 0x3f8fdf3b
	v_mov_b32_e32 v48, 0x3f900000
	v_mov_b32_e32 v47, 0x3f9020c5
	v_mov_b32_e32 v46, 0x3f904189
	v_mov_b32_e32 v45, 0x3f90a3d7
	v_mov_b32_e32 v44, 0x3f90c49c
	v_mov_b32_e32 v43, 0x3f90e560
	v_mov_b32_e32 v42, 0x3f910625
	v_mov_b32_e32 v41, 0x3f9126e9
	v_mov_b32_e32 v40, 0x3f9147ae
	v_mov_b32_e32 v39, 0x3f916873
	v_mov_b32_e32 v38, 0x3f918937
	v_mov_b32_e32 v37, 0x3f91eb85
	v_mov_b32_e32 v36, 0x3f920c4a
	v_mov_b32_e32 v35, 0x3f922d0e
	v_mov_b32_e32 v34, 0x3f924dd3
	v_mov_b32_e32 v33, 0x3f926e97
	v_mov_b32_e32 v32, 0x3f928f5c
	v_mov_b32_e32 v31, 0x3f92b021
	v_mov_b32_e32 v30, 0x3f92d0e5
	v_mov_b32_e32 v29, 0x3f933333
	v_mov_b32_e32 v28, 0x3f9353f8
	v_mov_b32_e32 v27, 0x3f9374bc
	v_mov_b32_e32 v26, 0x3f939581
	v_mov_b32_e32 v25, 0x3f93b645
	v_mov_b32_e32 v24, 0x3f93d70a
	v_mov_b32_e32 v23, 0x3f93f7cf
	v_mov_b32_e32 v22, 0x3f941893
.LBB4_11:                               ; %Flow1454
	v_add_f32_e32 v1, 0, v1
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v21
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v149
	v_add_f32_e32 v1, v1, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v147
	v_add_f32_e32 v1, v1, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v145
	v_add_f32_e32 v1, v1, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v143
	v_add_f32_e32 v1, v1, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v141
	v_add_f32_e32 v1, v1, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v139
	v_add_f32_e32 v1, v1, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v135
	v_add_f32_e32 v1, v1, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v133
	v_add_f32_e32 v1, v1, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v131
	v_add_f32_e32 v1, v1, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v129
	v_add_f32_e32 v1, v1, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v127
	v_add_f32_e32 v1, v1, v126
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v125
	v_add_f32_e32 v1, v1, v124
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v123
	v_add_f32_e32 v1, v1, v122
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v121
	v_add_f32_e32 v1, v1, v120
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v119
	v_add_f32_e32 v1, v1, v118
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v117
	v_add_f32_e32 v1, v1, v116
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v115
	v_add_f32_e32 v1, v1, v114
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v113
	v_add_f32_e32 v1, v1, v112
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v111
	v_add_f32_e32 v1, v1, v110
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v109
	v_add_f32_e32 v1, v1, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v107
	v_add_f32_e32 v1, v1, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v105
	v_add_f32_e32 v1, v1, v104
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v103
	v_add_f32_e32 v1, v1, v102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v101
	v_add_f32_e32 v1, v1, v100
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v99
	v_add_f32_e32 v1, v1, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v97
	v_add_f32_e32 v1, v1, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v95
	v_add_f32_e32 v1, v1, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v93
	v_add_f32_e32 v1, v1, v92
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v91
	v_add_f32_e32 v1, v1, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v89
	v_add_f32_e32 v1, v1, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v87
	v_add_f32_e32 v1, v1, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v85
	v_add_f32_e32 v1, v1, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v83
	v_add_f32_e32 v1, v1, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v81
	v_add_f32_e32 v1, v1, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v79
	v_add_f32_e32 v1, v1, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v77
	v_add_f32_e32 v1, v1, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v75
	v_add_f32_e32 v1, v1, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v73
	v_add_f32_e32 v1, v1, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v71
	v_add_f32_e32 v1, v1, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v69
	v_add_f32_e32 v1, v1, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v67
	v_add_f32_e32 v1, v1, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v65
	v_add_f32_e32 v1, v1, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v63
	v_add_f32_e32 v1, v1, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v61
	v_add_f32_e32 v1, v1, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v59
	v_add_f32_e32 v1, v1, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v57
	v_add_f32_e32 v1, v1, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v55
	v_add_f32_e32 v1, v1, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v53
	v_add_f32_e32 v1, v1, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v51
	v_add_f32_e32 v1, v1, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v49
	v_add_f32_e32 v1, v1, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v47
	v_add_f32_e32 v1, v1, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v45
	v_add_f32_e32 v1, v1, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v43
	v_add_f32_e32 v1, v1, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v41
	v_add_f32_e32 v1, v1, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v39
	v_add_f32_e32 v1, v1, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v37
	v_add_f32_e32 v1, v1, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v35
	v_add_f32_e32 v1, v1, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v33
	v_add_f32_e32 v1, v1, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v31
	v_add_f32_e32 v1, v1, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v29
	v_add_f32_e32 v1, v1, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v27
	v_add_f32_e32 v1, v1, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v25
	v_add_f32_e32 v1, v1, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v23
	v_add_f32_e32 v1, v1, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v150
	v_add_f32_e32 v1, v1, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v152
	v_add_f32_e32 v1, v1, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v155
	v_add_f32_e32 v1, v1, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v157
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v158
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end4:
	.size	_Z3runILi4ELi1EEvPfPji, .Lfunc_end4-_Z3runILi4ELi1EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi4ELi1EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 161
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_Z3runILi4ELi1EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi4ELi1EEvPfPji,"axG",@progbits,_Z3runILi4ELi1EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi4ELi1EEvPfPji.num_vgpr, 161
	.set .L_Z3runILi4ELi1EEvPfPji.num_agpr, 0
	.set .L_Z3runILi4ELi1EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi4ELi1EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi4ELi1EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi4ELi1EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi4ELi1EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi4ELi1EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi4ELi1EEvPfPji.has_recursion, 0
	.set .L_Z3runILi4ELi1EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6252
; TotalNumSgprs: 10
; NumVgprs: 161
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 161
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi5ELi1EEvPfPji,"axG",@progbits,_Z3runILi5ELi1EEvPfPji,comdat
	.protected	_Z3runILi5ELi1EEvPfPji  ; -- Begin function _Z3runILi5ELi1EEvPfPji
	.globl	_Z3runILi5ELi1EEvPfPji
	.p2align	8
	.type	_Z3runILi5ELi1EEvPfPji,@function
_Z3runILi5ELi1EEvPfPji:                 ; @_Z3runILi5ELi1EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB5_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB5_2:                                ; %.preheader77
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	v_lshl_add_u32 v21, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s1, 0x3a83126f
	ds_store_b32 v21, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v150, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v151, s1, v1, 0x3dcccccd
	v_fmaak_f32 v152, s1, v1, 0x3e4ccccd
	v_fmaak_f32 v153, s1, v1, 0x3e99999a
	v_fmaak_f32 v155, s1, v1, 0x3ecccccd
	v_fma_f32 v156, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v157, s1, v1, 0x3f19999a
	v_fmaak_f32 v158, s1, v1, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_7
; %bb.3:                                ; %.preheader.i.preheader.lr.ph
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v106, 0x3f941893
	v_dual_mov_b32 v110, 0x3f939581 :: v_dual_mov_b32 v105, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v2, 16, v1
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v13, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v107, 0x3f93f7cf
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v1, v2 :: v_dual_mov_b32 v108, 0x3f93d70a
	v_mov_b32_e32 v109, 0x3f93b645
	v_mov_b32_e32 v112, 0x3f9353f8
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v111, 0x3f9374bc :: v_dual_lshlrev_b32 v154, 2, v1
	v_mov_b32_e32 v1, v9
	v_dual_mov_b32 v113, 0x3f933333 :: v_dual_mov_b32 v4, v12
	v_dual_mov_b32 v114, 0x3f92d0e5 :: v_dual_mov_b32 v3, v11
	v_dual_mov_b32 v115, 0x3f92b021 :: v_dual_mov_b32 v6, v14
	v_dual_mov_b32 v116, 0x3f928f5c :: v_dual_mov_b32 v5, v13
	v_dual_mov_b32 v117, 0x3f926e97 :: v_dual_mov_b32 v8, v16
	v_dual_mov_b32 v118, 0x3f924dd3 :: v_dual_mov_b32 v7, v15
	v_mov_b32_e32 v119, 0x3f922d0e
	v_mov_b32_e32 v120, 0x3f920c4a
	v_mov_b32_e32 v121, 0x3f91eb85
	v_mov_b32_e32 v122, 0x3f918937
	v_mov_b32_e32 v123, 0x3f916873
	v_mov_b32_e32 v124, 0x3f9147ae
	v_mov_b32_e32 v125, 0x3f9126e9
	v_mov_b32_e32 v126, 0x3f910625
	v_mov_b32_e32 v127, 0x3f90e560
	v_mov_b32_e32 v128, 0x3f90c49c
	v_mov_b32_e32 v129, 0x3f90a3d7
	v_mov_b32_e32 v130, 0x3f904189
	v_mov_b32_e32 v131, 0x3f9020c5
	v_mov_b32_e32 v132, 0x3f900000
	v_mov_b32_e32 v133, 0x3f8fdf3b
	v_mov_b32_e32 v134, 0x3f8fbe77
	v_mov_b32_e32 v135, 0x3f8f9db2
	v_mov_b32_e32 v136, 0x3f8f7cee
	v_mov_b32_e32 v137, 0x3f8f5c29
	v_mov_b32_e32 v138, 0x3f8ef9db
	v_mov_b32_e32 v139, 0x3f8ed917
	v_mov_b32_e32 v140, 0x3f8eb852
	v_mov_b32_e32 v141, 0x3f8e978d
	v_mov_b32_e32 v142, 0x3f8e76c9
	v_mov_b32_e32 v143, 0x3f8e5604
	v_mov_b32_e32 v144, 0x3f8e3540
	v_mov_b32_e32 v145, 0x3f8e147b
	v_mov_b32_e32 v146, 0x3f8db22d
	v_mov_b32_e32 v147, 0x3f8d9169
	v_mov_b32_e32 v148, 0x3f8d70a4
	v_mov_b32_e32 v149, 0x3f8d4fdf
	v_mov_b32_e32 v22, 0x3f8d2f1b
	v_mov_b32_e32 v23, 0x3f8d0e56
	v_mov_b32_e32 v24, 0x3f8ced92
	v_mov_b32_e32 v25, 0x3f8ccccd
	v_mov_b32_e32 v26, 0x3f8c6a7f
	v_mov_b32_e32 v27, 0x3f8c49bb
	v_mov_b32_e32 v28, 0x3f8c28f6
	v_mov_b32_e32 v29, 0x3f8c0831
	v_mov_b32_e32 v30, 0x3f8be76d
	v_mov_b32_e32 v31, 0x3f8bc6a8
	v_mov_b32_e32 v32, 0x3f8ba5e4
	v_mov_b32_e32 v33, 0x3f8b851f
	v_mov_b32_e32 v34, 0x3f8b22d1
	v_mov_b32_e32 v35, 0x3f8b020d
	v_mov_b32_e32 v36, 0x3f8ae148
	v_mov_b32_e32 v37, 0x3f8ac083
	v_mov_b32_e32 v38, 0x3f8a9fbf
	v_mov_b32_e32 v39, 0x3f8a7efa
	v_mov_b32_e32 v40, 0x3f8a5e36
	v_mov_b32_e32 v41, 0x3f8a3d71
	v_mov_b32_e32 v42, 0x3f89db23
	v_mov_b32_e32 v43, 0x3f89ba5f
	v_mov_b32_e32 v44, 0x3f89999a
	v_mov_b32_e32 v45, 0x3f8978d5
	v_mov_b32_e32 v46, 0x3f895811
	v_mov_b32_e32 v47, 0x3f89374c
	v_mov_b32_e32 v48, 0x3f891688
	v_mov_b32_e32 v49, 0x3f88f5c3
	v_mov_b32_e32 v50, 0x3f889374
	v_mov_b32_e32 v51, 0x3f8872b0
	v_mov_b32_e32 v52, 0x3f8851eb
	v_mov_b32_e32 v53, 0x3f883126
	v_mov_b32_e32 v54, 0x3f881062
	v_mov_b32_e32 v55, 0x3f87ef9d
	v_mov_b32_e32 v56, 0x3f87ced9
	v_mov_b32_e32 v57, 0x3f87ae14
	v_mov_b32_e32 v58, 0x3f874bc6
	v_mov_b32_e32 v59, 0x3f872b02
	v_mov_b32_e32 v60, 0x3f870a3d
	v_mov_b32_e32 v61, 0x3f86e978
	v_mov_b32_e32 v62, 0x3f86c8b4
	v_mov_b32_e32 v63, 0x3f86a7ef
	v_mov_b32_e32 v64, 0x3f86872b
	v_mov_b32_e32 v65, 0x3f866666
	v_mov_b32_e32 v66, 0x3f860418
	v_mov_b32_e32 v67, 0x3f85e354
	v_mov_b32_e32 v68, 0x3f85c28f
	v_mov_b32_e32 v69, 0x3f85a1ca
	v_mov_b32_e32 v70, 0x3f858106
	v_mov_b32_e32 v71, 0x3f856041
	v_mov_b32_e32 v72, 0x3f853f7d
	v_mov_b32_e32 v73, 0x3f851eb8
	v_mov_b32_e32 v74, 0x3f84bc6a
	v_mov_b32_e32 v75, 0x3f849ba6
	v_mov_b32_e32 v76, 0x3f847ae1
	v_mov_b32_e32 v77, 0x3f845a1c
	v_mov_b32_e32 v78, 0x3f843958
	v_mov_b32_e32 v79, 0x3f841893
	v_mov_b32_e32 v80, 0x3f83f7cf
	v_mov_b32_e32 v81, 0x3f83d70a
	v_mov_b32_e32 v82, 0x3f8374bc
	v_mov_b32_e32 v83, 0x3f8353f8
	v_mov_b32_e32 v84, 0x3f833333
	v_mov_b32_e32 v85, 0x3f83126e
	v_mov_b32_e32 v86, 0x3f82f1aa
	v_mov_b32_e32 v87, 0x3f82d0e5
	v_mov_b32_e32 v88, 0x3f82b021
	v_mov_b32_e32 v89, 0x3f828f5c
	v_mov_b32_e32 v90, 0x3f822d0e
	v_mov_b32_e32 v91, 0x3f820c4a
	v_mov_b32_e32 v92, 0x3f81eb85
	v_mov_b32_e32 v93, 0x3f81cac0
	v_mov_b32_e32 v94, 0x3f81a9fc
	v_mov_b32_e32 v95, 0x3f818937
	v_mov_b32_e32 v96, 0x3f816873
	v_mov_b32_e32 v97, 0x3f8147ae
	v_mov_b32_e32 v98, 0x3f80e560
	v_mov_b32_e32 v99, 0x3f80c49c
	v_mov_b32_e32 v100, 0x3f80a3d7
	v_mov_b32_e32 v101, 0x3f808312
	v_mov_b32_e32 v102, 0x3f80624e
	v_mov_b32_e32 v103, 0x3f804189
	v_mov_b32_e32 v104, 0x3f8020c5
	v_mov_b32_e32 v17, 0x10101010
	v_dual_mov_b32 v19, 0x18181818 :: v_dual_mov_b32 v2, v10
	s_mov_b32 s1, 0
	s_mov_b32 s2, 0x35800000
.LBB5_4:                                ; %.preheader.i.preheader
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_5 Depth 2
	s_mov_b32 s3, 32
.LBB5_5:                                ; %.preheader.i
                                        ;   Parent Loop BB5_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v20, v19
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s3, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[19:20], v[17:18], v[1:8]
	s_cbranch_scc1 .LBB5_5
; %bb.6:                                ; %_Z6matrixILi1EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB5_4 Depth=1
	v_max3_num_f32 v10, v150, 0xff800000, v151
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v10, v10, v152, v153
	v_max3_num_f32 v10, v10, v155, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v10, v10, v157, v158
	ds_bpermute_b32 v11, v154, v10
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v11, v11, v11
	v_max_num_f32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v11, v150, v10
	v_dual_mul_f32 v11, 0x3fb8aa3b, v11 :: v_dual_sub_f32 v12, v151, v10
	v_dual_sub_f32 v15, v153, v10 :: v_dual_sub_f32 v14, v152, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v11, v11
	v_dual_mul_f32 v12, 0x3fb8aa3b, v12 :: v_dual_mul_f32 v15, 0x3fb8aa3b, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v14
	v_exp_f32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v151, v15
	v_sub_f32_e32 v15, v155, v10
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v18, 0x3a83126f, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_mul_f32_e32 v15, 0x3fb8aa3b, v15
	v_dual_add_f32 v13, v11, v12 :: v_dual_mul_f32 v16, 0x3b03126f, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v152, v15
	v_sub_f32_e32 v15, v156, v10
	v_max3_num_f32 v11, v18, 0, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0x3fb8aa3b, v15
	v_exp_f32_e32 v153, v15
	v_dual_sub_f32 v15, v157, v10 :: v_dual_sub_f32 v10, v158, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v13, v14, v13 :: v_dual_mul_f32 v10, 0x3fb8aa3b, v10
	v_add_f32_e32 v13, v151, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v12, 0x3bc49ba6, v153
	v_mul_f32_e32 v15, 0x3fb8aa3b, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v10, v10
	v_add_f32_e32 v13, v152, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v155, v15
	v_mul_f32_e32 v15, 0x3b449ba6, v14
	v_dual_mul_f32 v14, 0x3b83126f, v151 :: v_dual_add_f32 v13, v153, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v11, v11, v15, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v13, v155, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_dual_add_f32 v20, v10, v13 :: v_dual_mul_f32 v13, 0x3ba3d70b, v152
	v_mul_f32_e32 v10, 0x3c03126f, v10
	ds_bpermute_b32 v150, v154, v20
	v_max3_num_f32 v151, v11, v13, v12
	v_mul_f32_e32 v11, 0x3be56042, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v151, v151, v11, v10
	ds_bpermute_b32 v152, v154, v151
	s_wait_dscnt 0x1
	v_add_f32_e32 v20, v20, v150
	v_div_scale_f32 v150, null, v20, v20, v18
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v152, v152, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v151, v151, v152
	v_div_scale_f32 v152, null, 0x43e00000, 0x43e00000, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v153, v152
	v_fma_f32 v155, -v152, v153, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v155, v153
	v_div_scale_f32 v155, vcc_lo, v151, 0x43e00000, v151
	v_mul_f32_e32 v156, v155, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v152, v156, v155
	v_fmac_f32_e32 v156, v157, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v156, v155
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v153, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v151, v152, 0x43e00000, v151
	v_max_num_f32_e32 v151, 0x1f800000, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v152, null, v151, v151, v18
	v_rcp_f32_e32 v153, v152
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v155, -v152, v153, 1.0
	v_fmac_f32_e32 v153, v155, v153
	v_div_scale_f32 v155, vcc_lo, v18, v151, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v155, v153
	v_fma_f32 v157, -v152, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v157, v153
	v_fma_f32 v152, -v152, v156, v155
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v152, v152, v153, v156
	v_div_scale_f32 v153, null, v151, v151, v16
	v_div_fixup_f32 v152, v152, v151, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v155, v153
	v_fma_f32 v156, -v153, v155, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v155, v156, v155
	v_div_scale_f32 v156, vcc_lo, v16, v151, v16
	v_mul_f32_e32 v157, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v153, v157, v156
	v_fmac_f32_e32 v157, v158, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v153, -v153, v157, v156
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v153, v153, v155, v157
	v_mov_b16_e64 v155.l, v9.l
	v_mov_b16_e64 v155.h, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v153, v153, v151, v16
	v_mov_b16_e64 v156.l, v155.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v156.h, v155.h
	v_cvt_pk_fp8_f32 v156.l, v152, v153
	v_div_scale_f32 v152, null, v151, v151, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v153, v152
	v_fma_f32 v157, -v152, v153, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v157, v153
	v_div_scale_f32 v157, vcc_lo, v15, v151, v15
	v_mul_f32_e32 v158, v157, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v152, v158, v157
	v_fmac_f32_e32 v158, v159, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v158, v157
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v153, v158
	v_div_scale_f32 v153, null, v151, v151, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v152, v152, v151, v15
	v_rcp_f32_e32 v157, v153
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v153, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v14, v151, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v153, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v153, -v153, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v153, v153, v157, v159
	v_div_fixup_f32 v153, v153, v151, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v156.h, v152, v153
	v_div_scale_f32 v152, null, v151, v151, v13
	v_rcp_f32_e32 v153, v152
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v152, v153, 1.0
	v_fmac_f32_e32 v153, v157, v153
	v_div_scale_f32 v157, vcc_lo, v13, v151, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v158, v157, v153
	v_fma_f32 v159, -v152, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v158, v159, v153
	v_fma_f32 v152, -v152, v158, v157
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v152, v152, v153, v158
	v_div_scale_f32 v153, null, v151, v151, v12
	v_div_fixup_f32 v152, v152, v151, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v157, v153
	v_fma_f32 v158, -v153, v157, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v12, v151, v12
	v_mul_f32_e32 v159, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v153, v159, v158
	v_fmac_f32_e32 v159, v160, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v153, -v153, v159, v158
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v153, v153, v157, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v153, v153, v151, v12
	v_cvt_pk_fp8_f32 v155.l, v152, v153
	v_div_scale_f32 v152, null, v151, v151, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v153, v152
	v_fma_f32 v157, -v152, v153, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v157, v153
	v_div_scale_f32 v157, vcc_lo, v11, v151, v11
	v_mul_f32_e32 v158, v157, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v152, v158, v157
	v_fmac_f32_e32 v158, v159, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v158, v157
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v153, v158
	v_div_scale_f32 v153, null, v151, v151, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v152, v152, v151, v11
	v_rcp_f32_e32 v157, v153
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v153, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v10, v151, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v153, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v153, -v153, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v153, v153, v157, v159
	v_div_fixup_f32 v151, v153, v151, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v155.h, v152, v151
	v_xor_b32_e32 v151, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v151, v151
	v_fmaak_f32 v151, s2, v151, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v39, v39, v151
	v_dual_mul_f32 v105, v105, v151 :: v_dual_mul_f32 v74, v74, v151
	v_dual_mul_f32 v104, v104, v151 :: v_dual_mul_f32 v103, v103, v151
	v_mul_f32_e32 v72, v72, v151
	v_dual_mul_f32 v102, v102, v151 :: v_dual_mul_f32 v101, v101, v151
	v_mul_f32_e32 v70, v70, v151
	v_dual_mul_f32 v100, v100, v151 :: v_dual_mul_f32 v99, v99, v151
	v_mul_f32_e32 v68, v68, v151
	v_dual_mul_f32 v98, v98, v151 :: v_dual_mul_f32 v97, v97, v151
	v_mul_f32_e32 v66, v66, v151
	v_dual_mul_f32 v96, v96, v151 :: v_dual_mul_f32 v95, v95, v151
	v_mul_f32_e32 v64, v64, v151
	v_dual_mul_f32 v94, v94, v151 :: v_dual_mul_f32 v93, v93, v151
	v_mul_f32_e32 v62, v62, v151
	v_dual_mul_f32 v92, v92, v151 :: v_dual_mul_f32 v91, v91, v151
	v_mul_f32_e32 v60, v60, v151
	v_dual_mul_f32 v90, v90, v151 :: v_dual_mul_f32 v89, v89, v151
	v_mul_f32_e32 v58, v58, v151
	v_dual_mul_f32 v88, v88, v151 :: v_dual_mul_f32 v87, v87, v151
	v_mul_f32_e32 v56, v56, v151
	v_dual_mul_f32 v86, v86, v151 :: v_dual_mul_f32 v85, v85, v151
	v_mul_f32_e32 v54, v54, v151
	v_dual_mul_f32 v84, v84, v151 :: v_dual_mul_f32 v83, v83, v151
	v_mul_f32_e32 v52, v52, v151
	v_dual_mul_f32 v82, v82, v151 :: v_dual_mul_f32 v81, v81, v151
	v_mul_f32_e32 v50, v50, v151
	v_dual_mul_f32 v80, v80, v151 :: v_dual_mul_f32 v79, v79, v151
	v_mul_f32_e32 v48, v48, v151
	v_dual_mul_f32 v78, v78, v151 :: v_dual_mul_f32 v77, v77, v151
	v_mul_f32_e32 v46, v46, v151
	v_dual_mul_f32 v76, v76, v151 :: v_dual_mul_f32 v75, v75, v151
	v_dual_mul_f32 v44, v44, v151 :: v_dual_mul_f32 v73, v73, v151
	v_dual_mul_f32 v42, v42, v151 :: v_dual_mul_f32 v71, v71, v151
	v_dual_mul_f32 v40, v40, v151 :: v_dual_mul_f32 v69, v69, v151
	v_dual_mul_f32 v38, v38, v151 :: v_dual_mul_f32 v67, v67, v151
	v_dual_mul_f32 v36, v36, v151 :: v_dual_mul_f32 v65, v65, v151
	v_dual_mul_f32 v34, v34, v151 :: v_dual_mul_f32 v63, v63, v151
	v_dual_mul_f32 v32, v32, v151 :: v_dual_mul_f32 v61, v61, v151
	v_dual_mul_f32 v30, v30, v151 :: v_dual_mul_f32 v59, v59, v151
	v_dual_mul_f32 v28, v28, v151 :: v_dual_mul_f32 v57, v57, v151
	v_dual_mul_f32 v26, v26, v151 :: v_dual_mul_f32 v55, v55, v151
	v_dual_mul_f32 v24, v24, v151 :: v_dual_mul_f32 v53, v53, v151
	v_dual_mul_f32 v22, v22, v151 :: v_dual_mul_f32 v51, v51, v151
	v_dual_mul_f32 v148, v148, v151 :: v_dual_mul_f32 v49, v49, v151
	v_dual_mul_f32 v146, v146, v151 :: v_dual_mul_f32 v47, v47, v151
	v_dual_mul_f32 v144, v144, v151 :: v_dual_mul_f32 v45, v45, v151
	v_dual_mul_f32 v142, v142, v151 :: v_dual_mul_f32 v43, v43, v151
	v_dual_mul_f32 v140, v140, v151 :: v_dual_mul_f32 v41, v41, v151
	v_dual_mul_f32 v138, v138, v151 :: v_dual_mul_f32 v37, v37, v151
	v_mul_f32_e32 v134, v151, v134
	v_mul_f32_e32 v35, v35, v151
	v_dual_mul_f32 v33, v33, v151 :: v_dual_mul_f32 v132, v151, v132
	v_mul_f32_e32 v31, v31, v151
	v_dual_mul_f32 v29, v29, v151 :: v_dual_mul_f32 v130, v151, v130
	v_mul_f32_e32 v27, v27, v151
	v_dual_mul_f32 v25, v25, v151 :: v_dual_mul_f32 v128, v151, v128
	v_mul_f32_e32 v23, v23, v151
	v_dual_mul_f32 v149, v149, v151 :: v_dual_mul_f32 v126, v151, v126
	v_mul_f32_e32 v147, v147, v151
	v_dual_mul_f32 v145, v145, v151 :: v_dual_mul_f32 v124, v151, v124
	v_mul_f32_e32 v143, v143, v151
	v_dual_mul_f32 v141, v141, v151 :: v_dual_mul_f32 v122, v151, v122
	v_mul_f32_e32 v139, v139, v151
	v_dual_mul_f32 v137, v151, v137 :: v_dual_mul_f32 v120, v151, v120
	v_dual_mul_f32 v136, v151, v136 :: v_dual_mul_f32 v135, v151, v135
	v_dual_mul_f32 v118, v151, v118 :: v_dual_mul_f32 v133, v151, v133
	v_dual_mul_f32 v116, v151, v116 :: v_dual_mul_f32 v131, v151, v131
	v_dual_mul_f32 v114, v151, v114 :: v_dual_mul_f32 v129, v151, v129
	v_dual_mul_f32 v112, v151, v112 :: v_dual_mul_f32 v127, v151, v127
	v_dual_mul_f32 v110, v151, v110 :: v_dual_mul_f32 v125, v151, v125
	v_dual_mul_f32 v108, v151, v108 :: v_dual_mul_f32 v123, v151, v123
	v_dual_mul_f32 v106, v151, v106 :: v_dual_mul_f32 v121, v151, v121
	v_mul_f32_e32 v119, v151, v119
	v_mul_f32_e32 v117, v151, v117
	v_mul_f32_e32 v115, v151, v115
	v_mul_f32_e32 v113, v151, v113
	v_mul_f32_e32 v111, v151, v111
	v_mul_f32_e32 v109, v151, v109
	v_mul_f32_e32 v107, v151, v107
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v150, v151, 1.0
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v18, v20, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v153, v152, v151
	v_fma_f32 v155, -v150, v153, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v155, v151
	v_fma_f32 v150, -v150, v153, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v150, v150, v151, v153
	v_div_fixup_f32 v18, v150, v20, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v150, 0, v18
	v_div_scale_f32 v18, null, v20, v20, v16
	v_rcp_f32_e32 v151, v18
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v18, v151, 1.0
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v16, v20, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v153, v152, v151
	v_fma_f32 v155, -v18, v153, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v155, v151
	v_fma_f32 v18, -v18, v153, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v18, v18, v151, v153
	v_div_fixup_f32 v16, v18, v20, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v151, 0x3c23d70a, v16
	v_div_scale_f32 v16, null, v20, v20, v15
	v_rcp_f32_e32 v18, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v16, v18, 1.0
	v_fmac_f32_e32 v18, v152, v18
	v_div_scale_f32 v152, vcc_lo, v15, v20, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v153, v152, v18
	v_fma_f32 v155, -v16, v153, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v155, v18
	v_fma_f32 v16, -v16, v153, v152
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v18, v153
	v_div_fixup_f32 v15, v16, v20, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v152, 0x3ca3d70a, v15
	v_div_scale_f32 v15, null, v20, v20, v14
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v15, v16, 1.0
	v_fmac_f32_e32 v16, v18, v16
	v_div_scale_f32 v18, vcc_lo, v14, v20, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v153, v18, v16
	v_fma_f32 v155, -v15, v153, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v155, v16
	v_fma_f32 v15, -v15, v153, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v15, v15, v16, v153
	v_div_fixup_f32 v14, v15, v20, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v153, 0x3cf5c28f, v14
	v_div_scale_f32 v14, null, v20, v20, v13
	v_rcp_f32_e32 v15, v14
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v14, v15, 1.0
	v_fmac_f32_e32 v15, v16, v15
	v_div_scale_f32 v16, vcc_lo, v13, v20, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v18, v16, v15
	v_fma_f32 v155, -v14, v18, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v155, v15
	v_fma_f32 v14, -v14, v18, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v14, v14, v15, v18
	v_div_fixup_f32 v13, v14, v20, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v155, 0x3d23d70a, v13
	v_div_scale_f32 v13, null, v20, v20, v12
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v12, v20, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v18, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v18, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v12, v13, v20, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v156, 0x3d4ccccc, v12
	v_div_scale_f32 v12, null, v20, v20, v11
	v_rcp_f32_e32 v13, v12
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v12, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	v_div_scale_f32 v14, vcc_lo, v11, v20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v13
	v_fma_f32 v16, -v12, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v13
	v_fma_f32 v12, -v12, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v12, v12, v13, v15
	v_div_fixup_f32 v11, v12, v20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v157, 0x3d75c28f, v11
	v_div_scale_f32 v11, null, v20, v20, v10
	v_rcp_f32_e32 v12, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v11, v12, 1.0
	v_fmac_f32_e32 v12, v13, v12
	v_div_scale_f32 v13, vcc_lo, v10, v20, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v13, v12
	v_fma_f32 v15, -v11, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v15, v12
	v_fma_f32 v11, -v11, v14, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v12, v14
	v_div_fixup_f32 v10, v11, v20, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v158, 0x3d8f5c29, v10
	s_cbranch_scc0 .LBB5_4
	s_branch .LBB5_8
.LBB5_7:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v105, 1.0
	v_dual_mov_b32 v104, 0x3f8020c5 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v103, 0x3f804189 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v102, 0x3f80624e :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v101, 0x3f808312 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v100, 0x3f80a3d7 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v99, 0x3f80c49c :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v98, 0x3f80e560 :: v_dual_mov_b32 v1, 0
	v_mov_b32_e32 v97, 0x3f8147ae
	v_mov_b32_e32 v96, 0x3f816873
	v_mov_b32_e32 v95, 0x3f818937
	v_mov_b32_e32 v94, 0x3f81a9fc
	v_mov_b32_e32 v93, 0x3f81cac0
	v_mov_b32_e32 v92, 0x3f81eb85
	v_mov_b32_e32 v91, 0x3f820c4a
	v_mov_b32_e32 v90, 0x3f822d0e
	v_mov_b32_e32 v89, 0x3f828f5c
	v_mov_b32_e32 v88, 0x3f82b021
	v_mov_b32_e32 v87, 0x3f82d0e5
	v_mov_b32_e32 v86, 0x3f82f1aa
	v_mov_b32_e32 v85, 0x3f83126e
	v_mov_b32_e32 v84, 0x3f833333
	v_mov_b32_e32 v83, 0x3f8353f8
	v_mov_b32_e32 v82, 0x3f8374bc
	v_mov_b32_e32 v81, 0x3f83d70a
	v_mov_b32_e32 v80, 0x3f83f7cf
	v_mov_b32_e32 v79, 0x3f841893
	v_mov_b32_e32 v78, 0x3f843958
	v_mov_b32_e32 v77, 0x3f845a1c
	v_mov_b32_e32 v76, 0x3f847ae1
	v_mov_b32_e32 v75, 0x3f849ba6
	v_mov_b32_e32 v74, 0x3f84bc6a
	v_mov_b32_e32 v73, 0x3f851eb8
	v_mov_b32_e32 v72, 0x3f853f7d
	v_mov_b32_e32 v71, 0x3f856041
	v_mov_b32_e32 v70, 0x3f858106
	v_mov_b32_e32 v69, 0x3f85a1ca
	v_mov_b32_e32 v68, 0x3f85c28f
	v_mov_b32_e32 v67, 0x3f85e354
	v_mov_b32_e32 v66, 0x3f860418
	v_mov_b32_e32 v65, 0x3f866666
	v_mov_b32_e32 v64, 0x3f86872b
	v_mov_b32_e32 v63, 0x3f86a7ef
	v_mov_b32_e32 v62, 0x3f86c8b4
	v_mov_b32_e32 v61, 0x3f86e978
	v_mov_b32_e32 v60, 0x3f870a3d
	v_mov_b32_e32 v59, 0x3f872b02
	v_mov_b32_e32 v58, 0x3f874bc6
	v_mov_b32_e32 v57, 0x3f87ae14
	v_mov_b32_e32 v56, 0x3f87ced9
	v_mov_b32_e32 v55, 0x3f87ef9d
	v_mov_b32_e32 v54, 0x3f881062
	v_mov_b32_e32 v53, 0x3f883126
	v_mov_b32_e32 v52, 0x3f8851eb
	v_mov_b32_e32 v51, 0x3f8872b0
	v_mov_b32_e32 v50, 0x3f889374
	v_mov_b32_e32 v49, 0x3f88f5c3
	v_mov_b32_e32 v48, 0x3f891688
	v_mov_b32_e32 v47, 0x3f89374c
	v_mov_b32_e32 v46, 0x3f895811
	v_mov_b32_e32 v45, 0x3f8978d5
	v_mov_b32_e32 v44, 0x3f89999a
	v_mov_b32_e32 v43, 0x3f89ba5f
	v_mov_b32_e32 v42, 0x3f89db23
	v_mov_b32_e32 v41, 0x3f8a3d71
	v_mov_b32_e32 v40, 0x3f8a5e36
	v_mov_b32_e32 v39, 0x3f8a7efa
	v_mov_b32_e32 v38, 0x3f8a9fbf
	v_mov_b32_e32 v37, 0x3f8ac083
	v_mov_b32_e32 v36, 0x3f8ae148
	v_mov_b32_e32 v35, 0x3f8b020d
	v_mov_b32_e32 v34, 0x3f8b22d1
	v_mov_b32_e32 v33, 0x3f8b851f
	v_mov_b32_e32 v32, 0x3f8ba5e4
	v_mov_b32_e32 v31, 0x3f8bc6a8
	v_mov_b32_e32 v30, 0x3f8be76d
	v_mov_b32_e32 v29, 0x3f8c0831
	v_mov_b32_e32 v28, 0x3f8c28f6
	v_mov_b32_e32 v27, 0x3f8c49bb
	v_mov_b32_e32 v26, 0x3f8c6a7f
	v_mov_b32_e32 v25, 0x3f8ccccd
	v_mov_b32_e32 v24, 0x3f8ced92
	v_mov_b32_e32 v23, 0x3f8d0e56
	v_mov_b32_e32 v22, 0x3f8d2f1b
	v_mov_b32_e32 v149, 0x3f8d4fdf
	v_mov_b32_e32 v148, 0x3f8d70a4
	v_mov_b32_e32 v147, 0x3f8d9169
	v_mov_b32_e32 v146, 0x3f8db22d
	v_mov_b32_e32 v145, 0x3f8e147b
	v_mov_b32_e32 v144, 0x3f8e3540
	v_mov_b32_e32 v143, 0x3f8e5604
	v_mov_b32_e32 v142, 0x3f8e76c9
	v_mov_b32_e32 v141, 0x3f8e978d
	v_mov_b32_e32 v140, 0x3f8eb852
	v_mov_b32_e32 v139, 0x3f8ed917
	v_mov_b32_e32 v138, 0x3f8ef9db
	v_mov_b32_e32 v137, 0x3f8f5c29
	v_mov_b32_e32 v136, 0x3f8f7cee
	v_mov_b32_e32 v135, 0x3f8f9db2
	v_mov_b32_e32 v134, 0x3f8fbe77
	v_mov_b32_e32 v133, 0x3f8fdf3b
	v_mov_b32_e32 v132, 0x3f900000
	v_mov_b32_e32 v131, 0x3f9020c5
	v_mov_b32_e32 v130, 0x3f904189
	v_mov_b32_e32 v129, 0x3f90a3d7
	v_mov_b32_e32 v128, 0x3f90c49c
	v_mov_b32_e32 v127, 0x3f90e560
	v_mov_b32_e32 v126, 0x3f910625
	v_mov_b32_e32 v125, 0x3f9126e9
	v_mov_b32_e32 v124, 0x3f9147ae
	v_mov_b32_e32 v123, 0x3f916873
	v_mov_b32_e32 v122, 0x3f918937
	v_mov_b32_e32 v121, 0x3f91eb85
	v_mov_b32_e32 v120, 0x3f920c4a
	v_mov_b32_e32 v119, 0x3f922d0e
	v_mov_b32_e32 v118, 0x3f924dd3
	v_mov_b32_e32 v117, 0x3f926e97
	v_mov_b32_e32 v116, 0x3f928f5c
	v_mov_b32_e32 v115, 0x3f92b021
	v_mov_b32_e32 v114, 0x3f92d0e5
	v_mov_b32_e32 v113, 0x3f933333
	v_mov_b32_e32 v112, 0x3f9353f8
	v_mov_b32_e32 v111, 0x3f9374bc
	v_mov_b32_e32 v110, 0x3f939581
	v_mov_b32_e32 v109, 0x3f93b645
	v_mov_b32_e32 v108, 0x3f93d70a
	v_mov_b32_e32 v107, 0x3f93f7cf
	v_mov_b32_e32 v106, 0x3f941893
.LBB5_8:                                ; %.preheader73
	v_add_f32_e32 v1, 0, v1
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v21
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v105
	v_add_f32_e32 v1, v1, v104
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v103
	v_add_f32_e32 v1, v1, v102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v101
	v_add_f32_e32 v1, v1, v100
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v99
	v_add_f32_e32 v1, v1, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v97
	v_add_f32_e32 v1, v1, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v95
	v_add_f32_e32 v1, v1, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v93
	v_add_f32_e32 v1, v1, v92
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v91
	v_add_f32_e32 v1, v1, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v89
	v_add_f32_e32 v1, v1, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v87
	v_add_f32_e32 v1, v1, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v85
	v_add_f32_e32 v1, v1, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v83
	v_add_f32_e32 v1, v1, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v81
	v_add_f32_e32 v1, v1, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v79
	v_add_f32_e32 v1, v1, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v77
	v_add_f32_e32 v1, v1, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v75
	v_add_f32_e32 v1, v1, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v73
	v_add_f32_e32 v1, v1, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v71
	v_add_f32_e32 v1, v1, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v69
	v_add_f32_e32 v1, v1, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v67
	v_add_f32_e32 v1, v1, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v65
	v_add_f32_e32 v1, v1, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v63
	v_add_f32_e32 v1, v1, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v61
	v_add_f32_e32 v1, v1, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v59
	v_add_f32_e32 v1, v1, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v57
	v_add_f32_e32 v1, v1, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v55
	v_add_f32_e32 v1, v1, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v53
	v_add_f32_e32 v1, v1, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v51
	v_add_f32_e32 v1, v1, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v49
	v_add_f32_e32 v1, v1, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v47
	v_add_f32_e32 v1, v1, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v45
	v_add_f32_e32 v1, v1, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v43
	v_add_f32_e32 v1, v1, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v41
	v_add_f32_e32 v1, v1, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v39
	v_add_f32_e32 v1, v1, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v37
	v_add_f32_e32 v1, v1, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v35
	v_add_f32_e32 v1, v1, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v33
	v_add_f32_e32 v1, v1, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v31
	v_add_f32_e32 v1, v1, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v29
	v_add_f32_e32 v1, v1, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v27
	v_add_f32_e32 v1, v1, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v25
	v_add_f32_e32 v1, v1, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v23
	v_add_f32_e32 v1, v1, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v149
	v_add_f32_e32 v1, v1, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v147
	v_add_f32_e32 v1, v1, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v145
	v_add_f32_e32 v1, v1, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v143
	v_add_f32_e32 v1, v1, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v141
	v_add_f32_e32 v1, v1, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v139
	v_add_f32_e32 v1, v1, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v135
	v_add_f32_e32 v1, v1, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v133
	v_add_f32_e32 v1, v1, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v131
	v_add_f32_e32 v1, v1, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v129
	v_add_f32_e32 v1, v1, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v127
	v_add_f32_e32 v1, v1, v126
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v125
	v_add_f32_e32 v1, v1, v124
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v123
	v_add_f32_e32 v1, v1, v122
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v121
	v_add_f32_e32 v1, v1, v120
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v119
	v_add_f32_e32 v1, v1, v118
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v117
	v_add_f32_e32 v1, v1, v116
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v115
	v_add_f32_e32 v1, v1, v114
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v113
	v_add_f32_e32 v1, v1, v112
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v111
	v_add_f32_e32 v1, v1, v110
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v109
	v_add_f32_e32 v1, v1, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v107
	v_add_f32_e32 v1, v1, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v150
	v_add_f32_e32 v1, v1, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v152
	v_add_f32_e32 v1, v1, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v155
	v_add_f32_e32 v1, v1, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v157
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v158
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end5:
	.size	_Z3runILi5ELi1EEvPfPji, .Lfunc_end5-_Z3runILi5ELi1EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi5ELi1EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 161
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-_Z3runILi5ELi1EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi5ELi1EEvPfPji,"axG",@progbits,_Z3runILi5ELi1EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi5ELi1EEvPfPji.num_vgpr, 161
	.set .L_Z3runILi5ELi1EEvPfPji.num_agpr, 0
	.set .L_Z3runILi5ELi1EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi5ELi1EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi5ELi1EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi5ELi1EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi5ELi1EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi5ELi1EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi5ELi1EEvPfPji.has_recursion, 0
	.set .L_Z3runILi5ELi1EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6200
; TotalNumSgprs: 10
; NumVgprs: 161
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 161
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi0ELi4EEvPfPji,"axG",@progbits,_Z3runILi0ELi4EEvPfPji,comdat
	.protected	_Z3runILi0ELi4EEvPfPji  ; -- Begin function _Z3runILi0ELi4EEvPfPji
	.globl	_Z3runILi0ELi4EEvPfPji
	.p2align	8
	.type	_Z3runILi0ELi4EEvPfPji,@function
_Z3runILi0ELi4EEvPfPji:                 ; @_Z3runILi0ELi4EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v37, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v37
	s_cbranch_execz .LBB6_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v1, ttmp9, 3, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s6, v1
	v_add_co_ci_u32_e64 v2, null, s7, v2, vcc_lo
	global_store_b32 v[1:2], v3, off
.LBB6_2:                                ; %.preheader77
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	v_lshl_add_u32 v38, v0, 2, 0
	ds_store_b32 v38, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB6_7
; %bb.3:                                ; %.lr.ph.preheader
	s_mov_b32 s2, 0x18181818
	s_mov_b32 s6, 0x10101010
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s3, s2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v34, s3
	s_mov_b32 s7, s6
	s_mov_b32 s1, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v33, s2 :: v_dual_mov_b32 v8, v1
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v36, s7 :: v_dual_mov_b32 v35, s6
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v26, v1
	v_dual_mov_b32 v27, v1 :: v_dual_mov_b32 v28, v1
	v_dual_mov_b32 v29, v1 :: v_dual_mov_b32 v30, v1
	v_dual_mov_b32 v31, v1 :: v_dual_mov_b32 v32, v1
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v1
	v_dual_mov_b32 v19, v1 :: v_dual_mov_b32 v20, v1
	v_dual_mov_b32 v21, v1 :: v_dual_mov_b32 v22, v1
	v_dual_mov_b32 v23, v1 :: v_dual_mov_b32 v24, v1
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v10, v1
	v_dual_mov_b32 v11, v1 :: v_dual_mov_b32 v12, v1
	v_dual_mov_b32 v13, v1 :: v_dual_mov_b32 v14, v1
	v_dual_mov_b32 v15, v1 :: v_dual_mov_b32 v16, v1
.LBB6_4:                                ; %.lr.ph
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_5 Depth 2
	s_mov_b32 s2, 32
.LBB6_5:                                ; %.preheader.i
                                        ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[33:34], v[35:36], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[33:34], v[35:36], v[17:24]
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[33:34], v[35:36], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[33:34], v[35:36], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB6_5
; %bb.6:                                ; %_Z6matrixILi4EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB6_4 Depth=1
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	s_cbranch_scc0 .LBB6_4
	s_branch .LBB6_8
.LBB6_7:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v32, 0
.LBB6_8:                                ; %Flow
	s_mov_b32 s0, 0x3a83126f
	v_add_f32_e32 v25, 0, v25
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v2
	v_cvt_f32_ubyte0_e32 v2, v37
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmaak_f32 v3, s0, v2, 0x3dcccccd
	v_dual_add_f32 v1, v1, v4 :: v_dual_fmaak_f32 v4, s0, v2, 0x3e4ccccd
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v5
	v_fmaak_f32 v5, s0, v2, 0x3ecccccd
	v_add_f32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 1.0, v1
	v_add_f32_e32 v1, 0x3f8020c5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f804189, v1
	v_add_f32_e32 v1, 0x3f80624e, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f808312, v1
	v_add_f32_e32 v1, 0x3f80a3d7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f80c49c, v1
	v_add_f32_e32 v1, 0x3f80e560, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8147ae, v1
	v_add_f32_e32 v1, 0x3f816873, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f818937, v1
	v_add_f32_e32 v1, 0x3f81a9fc, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f81cac0, v1
	v_add_f32_e32 v1, 0x3f81eb85, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f820c4a, v1
	v_add_f32_e32 v1, 0x3f822d0e, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f828f5c, v1
	v_add_f32_e32 v1, 0x3f82b021, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f82d0e5, v1
	v_add_f32_e32 v1, 0x3f82f1aa, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f83126e, v1
	v_add_f32_e32 v1, 0x3f833333, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8353f8, v1
	v_add_f32_e32 v1, 0x3f8374bc, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f83d70a, v1
	v_add_f32_e32 v1, 0x3f83f7cf, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f841893, v1
	v_add_f32_e32 v1, 0x3f843958, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f845a1c, v1
	v_add_f32_e32 v1, 0x3f847ae1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f849ba6, v1
	v_add_f32_e32 v1, 0x3f84bc6a, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f851eb8, v1
	v_add_f32_e32 v1, 0x3f853f7d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f856041, v1
	v_add_f32_e32 v1, 0x3f858106, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f85a1ca, v1
	v_add_f32_e32 v1, 0x3f85c28f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f85e354, v1
	v_add_f32_e32 v1, 0x3f860418, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f866666, v1
	v_add_f32_e32 v1, 0x3f86872b, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f86a7ef, v1
	v_add_f32_e32 v1, 0x3f86c8b4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f86e978, v1
	v_add_f32_e32 v1, 0x3f870a3d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f872b02, v1
	v_add_f32_e32 v1, 0x3f874bc6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f87ae14, v1
	v_add_f32_e32 v1, 0x3f87ced9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f87ef9d, v1
	v_add_f32_e32 v1, 0x3f881062, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f883126, v1
	v_add_f32_e32 v1, 0x3f8851eb, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8872b0, v1
	v_add_f32_e32 v1, 0x3f889374, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f88f5c3, v1
	v_add_f32_e32 v1, 0x3f891688, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f89374c, v1
	v_add_f32_e32 v1, 0x3f895811, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8978d5, v1
	v_add_f32_e32 v1, 0x3f89999a, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f89ba5f, v1
	v_add_f32_e32 v1, 0x3f89db23, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8a3d71, v1
	v_add_f32_e32 v1, 0x3f8a5e36, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8a7efa, v1
	v_add_f32_e32 v1, 0x3f8a9fbf, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ac083, v1
	v_add_f32_e32 v1, 0x3f8ae148, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8b020d, v1
	v_add_f32_e32 v1, 0x3f8b22d1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8b851f, v1
	v_add_f32_e32 v1, 0x3f8ba5e4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8bc6a8, v1
	v_add_f32_e32 v1, 0x3f8be76d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8c0831, v1
	v_add_f32_e32 v1, 0x3f8c28f6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8c49bb, v1
	v_add_f32_e32 v1, 0x3f8c6a7f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ccccd, v1
	v_add_f32_e32 v1, 0x3f8ced92, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8d0e56, v1
	v_add_f32_e32 v1, 0x3f8d2f1b, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8d4fdf, v1
	v_add_f32_e32 v1, 0x3f8d70a4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8d9169, v1
	v_add_f32_e32 v1, 0x3f8db22d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8e147b, v1
	v_add_f32_e32 v1, 0x3f8e3540, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8e5604, v1
	v_add_f32_e32 v1, 0x3f8e76c9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8e978d, v1
	v_add_f32_e32 v1, 0x3f8eb852, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8ed917, v1
	v_add_f32_e32 v1, 0x3f8ef9db, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8f5c29, v1
	v_add_f32_e32 v1, 0x3f8f7cee, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8f9db2, v1
	v_add_f32_e32 v1, 0x3f8fbe77, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f8fdf3b, v1
	v_add_f32_e32 v1, 0x3f900000, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f9020c5, v1
	v_add_f32_e32 v1, 0x3f904189, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f90a3d7, v1
	v_add_f32_e32 v1, 0x3f90c49c, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f90e560, v1
	v_add_f32_e32 v1, 0x3f910625, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f9126e9, v1
	v_add_f32_e32 v1, 0x3f9147ae, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f916873, v1
	v_add_f32_e32 v1, 0x3f918937, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f91eb85, v1
	v_add_f32_e32 v1, 0x3f920c4a, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f922d0e, v1
	v_add_f32_e32 v1, 0x3f924dd3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f926e97, v1
	v_add_f32_e32 v1, 0x3f928f5c, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f92b021, v1
	v_add_f32_e32 v1, 0x3f92d0e5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f933333, v1
	v_add_f32_e32 v1, 0x3f9353f8, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f9374bc, v1
	v_add_f32_e32 v1, 0x3f939581, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f93b645, v1
	v_add_f32_e32 v1, 0x3f93d70a, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, 0x3f93f7cf, v1
	v_add_f32_e32 v1, 0x3f941893, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, 0x3a83126f, v2
	v_add_f32_e32 v1, v1, v3
	v_fmaak_f32 v3, s0, v2, 0x3e99999a
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v4
	ds_load_b32 v4, v38
	v_add_f32_e32 v1, v1, v3
	v_fma_f32 v3, 0x3a83126f, v2, 0.5
	v_add_f32_e32 v1, v1, v5
	v_fmaak_f32 v5, s0, v2, 0x3f19999a
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmaak_f32 v2, s0, v2, 0x3f333333 :: v_dual_add_f32 v1, v1, v3
	v_add_f32_e32 v3, v1, v5
	v_mov_b32_e32 v1, 0
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v3, v2
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmamk_f32 v2, v4, 0x33800000, v2
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_endpgm
.Lfunc_end6:
	.size	_Z3runILi0ELi4EEvPfPji, .Lfunc_end6-_Z3runILi0ELi4EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi0ELi4EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 39
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-_Z3runILi0ELi4EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi0ELi4EEvPfPji,"axG",@progbits,_Z3runILi0ELi4EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi0ELi4EEvPfPji.num_vgpr, 39
	.set .L_Z3runILi0ELi4EEvPfPji.num_agpr, 0
	.set .L_Z3runILi0ELi4EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi0ELi4EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi0ELi4EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi0ELi4EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi0ELi4EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi0ELi4EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi0ELi4EEvPfPji.has_recursion, 0
	.set .L_Z3runILi0ELi4EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2252
; TotalNumSgprs: 10
; NumVgprs: 39
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 39
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi1ELi4EEvPfPji,"axG",@progbits,_Z3runILi1ELi4EEvPfPji,comdat
	.protected	_Z3runILi1ELi4EEvPfPji  ; -- Begin function _Z3runILi1ELi4EEvPfPji
	.globl	_Z3runILi1ELi4EEvPfPji
	.p2align	8
	.type	_Z3runILi1ELi4EEvPfPji,@function
_Z3runILi1ELi4EEvPfPji:                 ; @_Z3runILi1ELi4EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v2, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v2
	s_cbranch_execz .LBB7_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 5, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v3, ttmp9, 3, v1
	v_mov_b32_e32 v1, s3
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, s6, v3
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	global_store_b32 v[3:4], v1, off
.LBB7_2:                                ; %.preheader73
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	v_lshl_add_u32 v1, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v2, v2
	s_mov_b32 s1, 0x3a83126f
	ds_store_b32 v1, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v132, 0x3a83126f, v2
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v134, s1, v2, 0x3dcccccd
	v_fmaak_f32 v138, s1, v2, 0x3e4ccccd
	v_fmaak_f32 v133, s1, v2, 0x3e99999a
	v_fmaak_f32 v137, s1, v2, 0x3ecccccd
	v_fma_f32 v139, 0x3a83126f, v2, 0.5
	v_fmaak_f32 v135, s1, v2, 0x3f19999a
	v_fmaak_f32 v136, s1, v2, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB7_5
; %bb.3:                                ; %.lr.ph
	v_mbcnt_lo_u32_b32 v9, -1, 0
	v_dual_mov_b32 v2, 0x3f941893 :: v_dual_mov_b32 v129, 1.0
	v_dual_mov_b32 v4, 0x3f93d70a :: v_dual_mov_b32 v131, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_xor_b32_e32 v10, 16, v9
	v_mov_b32_e32 v6, 0x3f939581
	v_mov_b32_e32 v7, 0x3f9374bc
	v_mov_b32_e32 v8, 0x3f9353f8
	v_mov_b32_e32 v11, 0x3f92b021
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	v_mov_b32_e32 v3, 0x3f93f7cf
	v_mov_b32_e32 v12, 0x3f928f5c
	v_mov_b32_e32 v13, 0x3f926e97
	v_mov_b32_e32 v14, 0x3f924dd3
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v130, v9, v10 :: v_dual_mov_b32 v5, 0x3f93b645
	v_mov_b32_e32 v9, 0x3f933333
	v_mov_b32_e32 v10, 0x3f92d0e5
	v_mov_b32_e32 v15, 0x3f922d0e
	v_mov_b32_e32 v16, 0x3f920c4a
	v_mov_b32_e32 v17, 0x3f91eb85
	v_mov_b32_e32 v18, 0x3f918937
	v_mov_b32_e32 v19, 0x3f916873
	v_mov_b32_e32 v20, 0x3f9147ae
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v22, 0x3f910625
	v_mov_b32_e32 v23, 0x3f90e560
	v_mov_b32_e32 v24, 0x3f90c49c
	v_mov_b32_e32 v25, 0x3f90a3d7
	v_mov_b32_e32 v26, 0x3f904189
	v_mov_b32_e32 v27, 0x3f9020c5
	v_mov_b32_e32 v28, 0x3f900000
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v30, 0x3f8fbe77
	v_mov_b32_e32 v31, 0x3f8f9db2
	v_mov_b32_e32 v32, 0x3f8f7cee
	v_mov_b32_e32 v33, 0x3f8f5c29
	v_mov_b32_e32 v34, 0x3f8ef9db
	v_mov_b32_e32 v35, 0x3f8ed917
	v_mov_b32_e32 v36, 0x3f8eb852
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v38, 0x3f8e76c9
	v_mov_b32_e32 v39, 0x3f8e5604
	v_mov_b32_e32 v40, 0x3f8e3540
	v_mov_b32_e32 v41, 0x3f8e147b
	v_mov_b32_e32 v42, 0x3f8db22d
	v_mov_b32_e32 v43, 0x3f8d9169
	v_mov_b32_e32 v44, 0x3f8d70a4
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v46, 0x3f8d2f1b
	v_mov_b32_e32 v47, 0x3f8d0e56
	v_mov_b32_e32 v48, 0x3f8ced92
	v_mov_b32_e32 v49, 0x3f8ccccd
	v_mov_b32_e32 v50, 0x3f8c6a7f
	v_mov_b32_e32 v51, 0x3f8c49bb
	v_mov_b32_e32 v52, 0x3f8c28f6
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v54, 0x3f8be76d
	v_mov_b32_e32 v55, 0x3f8bc6a8
	v_mov_b32_e32 v56, 0x3f8ba5e4
	v_mov_b32_e32 v57, 0x3f8b851f
	v_mov_b32_e32 v58, 0x3f8b22d1
	v_mov_b32_e32 v59, 0x3f8b020d
	v_mov_b32_e32 v60, 0x3f8ae148
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v62, 0x3f8a9fbf
	v_mov_b32_e32 v63, 0x3f8a7efa
	v_mov_b32_e32 v64, 0x3f8a5e36
	v_mov_b32_e32 v65, 0x3f8a3d71
	v_mov_b32_e32 v66, 0x3f89db23
	v_mov_b32_e32 v67, 0x3f89ba5f
	v_mov_b32_e32 v68, 0x3f89999a
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v70, 0x3f895811
	v_mov_b32_e32 v71, 0x3f89374c
	v_mov_b32_e32 v72, 0x3f891688
	v_mov_b32_e32 v73, 0x3f88f5c3
	v_mov_b32_e32 v74, 0x3f889374
	v_mov_b32_e32 v75, 0x3f8872b0
	v_mov_b32_e32 v76, 0x3f8851eb
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v78, 0x3f881062
	v_mov_b32_e32 v79, 0x3f87ef9d
	v_mov_b32_e32 v80, 0x3f87ced9
	v_mov_b32_e32 v81, 0x3f87ae14
	v_mov_b32_e32 v82, 0x3f874bc6
	v_mov_b32_e32 v83, 0x3f872b02
	v_mov_b32_e32 v84, 0x3f870a3d
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v86, 0x3f86c8b4
	v_mov_b32_e32 v87, 0x3f86a7ef
	v_mov_b32_e32 v88, 0x3f86872b
	v_mov_b32_e32 v89, 0x3f866666
	v_mov_b32_e32 v90, 0x3f860418
	v_mov_b32_e32 v91, 0x3f85e354
	v_mov_b32_e32 v92, 0x3f85c28f
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v94, 0x3f858106
	v_mov_b32_e32 v95, 0x3f856041
	v_mov_b32_e32 v96, 0x3f853f7d
	v_mov_b32_e32 v97, 0x3f851eb8
	v_mov_b32_e32 v98, 0x3f84bc6a
	v_mov_b32_e32 v99, 0x3f849ba6
	v_mov_b32_e32 v100, 0x3f847ae1
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v102, 0x3f843958
	v_mov_b32_e32 v103, 0x3f841893
	v_mov_b32_e32 v104, 0x3f83f7cf
	v_mov_b32_e32 v105, 0x3f83d70a
	v_mov_b32_e32 v106, 0x3f8374bc
	v_mov_b32_e32 v107, 0x3f8353f8
	v_mov_b32_e32 v108, 0x3f833333
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v110, 0x3f82f1aa
	v_mov_b32_e32 v111, 0x3f82d0e5
	v_mov_b32_e32 v112, 0x3f82b021
	v_mov_b32_e32 v113, 0x3f828f5c
	v_mov_b32_e32 v114, 0x3f822d0e
	v_mov_b32_e32 v115, 0x3f820c4a
	v_mov_b32_e32 v116, 0x3f81eb85
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v118, 0x3f81a9fc
	v_mov_b32_e32 v119, 0x3f818937
	v_mov_b32_e32 v120, 0x3f816873
	v_mov_b32_e32 v121, 0x3f8147ae
	v_mov_b32_e32 v122, 0x3f80e560
	v_mov_b32_e32 v123, 0x3f80c49c
	v_mov_b32_e32 v124, 0x3f80a3d7
	v_mov_b32_e32 v125, 0x3f808312
	v_mov_b32_e32 v126, 0x3f80624e
	v_mov_b32_e32 v127, 0x3f804189
	v_mov_b32_e32 v128, 0x3f8020c5
	v_lshlrev_b32_e32 v130, 2, v130
	s_mov_b32 s1, 0x35800000
.LBB7_4:                                ; =>This Inner Loop Header: Depth=1
	v_max3_num_f32 v140, v132, 0xff800000, v134
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v140, v140, v138, v133
	v_max3_num_f32 v140, v140, v137, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v140, v140, v135, v136
	ds_bpermute_b32 v141, v130, v140
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v141, v141, v141
	v_max_num_f32_e32 v140, v140, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v139, v139, v140
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_sub_f32 v132, v132, v140
	v_dual_sub_f32 v134, v134, v140 :: v_dual_sub_f32 v133, v133, v140
	v_dual_sub_f32 v138, v138, v140 :: v_dual_sub_f32 v137, v137, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v135, v135, v140 :: v_dual_mul_f32 v134, 0x3fb8aa3b, v134
	v_dual_mul_f32 v132, 0x3fb8aa3b, v132 :: v_dual_mul_f32 v133, 0x3fb8aa3b, v133
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v138, 0x3fb8aa3b, v138 :: v_dual_mul_f32 v137, 0x3fb8aa3b, v137
	v_exp_f32_e32 v144, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v143, v132
	v_sub_f32_e32 v136, v136, v140
	v_exp_f32_e32 v145, v138
	v_exp_f32_e32 v146, v133
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v147, v137
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	v_exp_f32_e32 v148, v139
	v_mul_f32_e32 v133, 0x3b03126f, v144
	v_mul_f32_e32 v132, 0x3a83126f, v143
	v_exp_f32_e32 v149, v135
	v_exp_f32_e32 v150, v136
	v_mul_f32_e32 v134, 0x3b449ba6, v145
	v_mul_f32_e32 v135, 0x3b83126f, v146
	v_max3_num_f32 v136, v132, 0, v133
	v_add_f32_e32 v143, v143, v144
	v_mul_f32_e32 v137, 0x3bc49ba6, v148
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_max3_num_f32 v138, v136, v134, v135
	v_mul_f32_e32 v136, 0x3ba3d70b, v147
	v_mul_f32_e32 v139, 0x3c03126f, v150
	v_add_f32_e32 v143, v145, v143
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v140, v138, v136, v137
	v_dual_mul_f32 v138, 0x3be56042, v149 :: v_dual_add_f32 v143, v146, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v140, v140, v138, v139
	v_add_f32_e32 v143, v147, v143
	ds_bpermute_b32 v141, v130, v140
	v_add_f32_e32 v143, v148, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v143, v149, v143
	v_add_f32_e32 v143, v150, v143
	ds_bpermute_b32 v144, v130, v143
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v141, v141, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v140, v140, v141
	v_div_scale_f32 v141, null, 0x43e00000, 0x43e00000, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v142, v141
	s_wait_dscnt 0x0
	v_add_f32_e32 v143, v143, v144
	v_div_scale_f32 v144, null, v143, v143, v132
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v151, -v141, v142, 1.0
	v_rcp_f32_e32 v145, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v142, v151, v142
	v_div_scale_f32 v151, vcc_lo, v140, 0x43e00000, v140
	v_mul_f32_e32 v152, v151, v142
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v146, -v144, v145, 1.0
	v_fma_f32 v153, -v141, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v145, v146, v145 :: v_dual_fmac_f32 v152, v153, v142
	v_fma_f32 v141, -v141, v152, v151
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v141, v141, v142, v152
	v_div_fixup_f32 v140, v141, 0x43e00000, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v140, 0x1f800000, v140
	v_div_scale_f32 v141, null, v140, v140, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v142, v141
	v_fma_f32 v151, -v141, v142, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v142, v151, v142
	v_div_scale_f32 v151, vcc_lo, v132, v140, v132
	v_mul_f32_e32 v152, v151, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v153, -v141, v152, v151
	v_fmac_f32_e32 v152, v153, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v141, -v141, v152, v151
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v141, v141, v142, v152
	v_div_scale_f32 v142, null, v140, v140, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v151, v142
	v_fma_f32 v152, -v142, v151, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v133, v140, v133
	v_mul_f32_e32 v153, v152, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v154, -v142, v153, v152
	v_fmac_f32_e32 v153, v154, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v142, -v142, v153, v152
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v142, v142, v151, v153
	v_div_scale_f32 v151, null, v140, v140, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v152, v151
	v_fma_f32 v153, -v151, v152, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v152, v153, v152
	v_div_scale_f32 v153, vcc_lo, v134, v140, v134
	v_mul_f32_e32 v154, v153, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v155, -v151, v154, v153
	v_fmac_f32_e32 v154, v155, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v151, v154, v153
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v151, v151, v152, v154
	v_div_scale_f32 v152, null, v140, v140, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v153, v152
	v_fma_f32 v154, -v152, v153, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v153, v154, v153
	v_div_scale_f32 v154, vcc_lo, v135, v140, v135
	v_mul_f32_e32 v155, v154, v153
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v152, v155, v154
	v_fmac_f32_e32 v155, v156, v153
	v_div_fixup_f32 v151, v151, v140, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v152, v155, v154
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v152, v152, v153, v155
	v_div_scale_f32 v153, null, v140, v140, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v152, v152, v140, v135
	v_rcp_f32_e32 v154, v153
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v155, -v153, v154, 1.0
	v_fmac_f32_e32 v154, v155, v154
	v_div_scale_f32 v155, vcc_lo, v136, v140, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v156, v155, v154
	v_fma_f32 v157, -v153, v156, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v156, v157, v154
	v_fma_f32 v153, -v153, v156, v155
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v153, v153, v154, v156
	v_div_scale_f32 v154, null, v140, v140, v137
	v_rcp_f32_e32 v155, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v154, v155, 1.0
	v_fmac_f32_e32 v155, v156, v155
	v_div_scale_f32 v156, vcc_lo, v137, v140, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v157, v156, v155
	v_fma_f32 v158, -v154, v157, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v157, v158, v155
	v_fma_f32 v154, -v154, v157, v156
	v_div_fixup_f32 v156, v142, v140, v133
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v154, v154, v155, v157
	v_div_fixup_f32 v155, v141, v140, v132
	v_mov_b16_e64 v141.l, v131.l
	v_mov_b16_e64 v141.h, 0
	v_mov_b16_e64 v142.l, v141.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v142.h, v141.h
	v_cvt_pk_fp8_f32 v142.l, v155, v156
	v_div_scale_f32 v155, null, v140, v140, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_pk_fp8_f32 v142.h, v151, v152
	v_rcp_f32_e32 v156, v155
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v155, v156, 1.0
	v_fmac_f32_e32 v156, v157, v156
	v_div_scale_f32 v157, vcc_lo, v138, v140, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v158, v157, v156
	v_fma_f32 v159, -v155, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v158, v159, v156
	v_fma_f32 v155, -v155, v158, v157
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v155, v155, v156, v158
	v_div_scale_f32 v156, null, v140, v140, v139
	v_rcp_f32_e32 v157, v156
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v156, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v139, v140, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v156, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v156, -v156, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v156, v156, v157, v159
	v_div_scale_f32 v146, vcc_lo, v132, v143, v132
	v_mul_f32_e32 v147, v146, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v148, -v144, v147, v146
	v_fmac_f32_e32 v147, v148, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v144, -v144, v147, v146
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v144, v144, v145, v147
	v_div_scale_f32 v145, null, v143, v143, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v146, v145
	v_fma_f32 v147, -v145, v146, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v146, v147, v146
	v_div_scale_f32 v147, vcc_lo, v133, v143, v133
	v_mul_f32_e32 v148, v147, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v145, v148, v147
	v_fmac_f32_e32 v148, v149, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v145, -v145, v148, v147
	v_div_fixup_f32 v147, v154, v140, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v145, v145, v146, v148
	v_div_fixup_f32 v146, v153, v140, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v141.l, v146, v147
	v_div_scale_f32 v146, null, v143, v143, v134
	v_rcp_f32_e32 v147, v146
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v148, -v146, v147, 1.0
	v_fmac_f32_e32 v147, v148, v147
	v_div_scale_f32 v148, vcc_lo, v134, v143, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v149, v148, v147
	v_fma_f32 v150, -v146, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v147
	v_fma_f32 v146, -v146, v149, v148
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v146, v146, v147, v149
	v_div_scale_f32 v147, null, v143, v143, v135
	v_rcp_f32_e32 v148, v147
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v149, -v147, v148, 1.0
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v135, v143, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v150, v149, v148
	v_fma_f32 v151, -v147, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v148
	v_fma_f32 v147, -v147, v150, v149
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v147, v147, v148, v150
	v_div_fixup_f32 v148, v155, v140, v138
	v_div_fixup_f32 v140, v156, v140, v139
	v_cvt_pk_fp8_f32 v141.h, v148, v140
	v_div_scale_f32 v140, null, v143, v143, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v148, v140
	v_fma_f32 v149, -v140, v148, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v148, v149, v148
	v_div_scale_f32 v149, vcc_lo, v136, v143, v136
	v_mul_f32_e32 v150, v149, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v151, -v140, v150, v149
	v_fmac_f32_e32 v150, v151, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v140, -v140, v150, v149
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v140, v140, v148, v150
	v_div_scale_f32 v148, null, v143, v143, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v149, v148
	v_fma_f32 v150, -v148, v149, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v149, v150, v149
	v_div_scale_f32 v150, vcc_lo, v137, v143, v137
	v_mul_f32_e32 v151, v150, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v152, -v148, v151, v150
	v_fmac_f32_e32 v151, v152, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v148, -v148, v151, v150
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v148, v148, v149, v151
	v_div_scale_f32 v149, null, v143, v143, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v150, v149
	v_fma_f32 v151, -v149, v150, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v150, v151, v150
	v_div_scale_f32 v151, vcc_lo, v138, v143, v138
	v_mul_f32_e32 v152, v151, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v153, -v149, v152, v151
	v_fmac_f32_e32 v152, v153, v150
	v_div_fixup_f32 v132, v144, v143, v132
	v_div_fixup_f32 v144, v146, v143, v134
	v_xor_b32_e32 v134, v142, v141
	v_div_fixup_f32 v133, v145, v143, v133
	v_fma_f32 v149, -v149, v152, v151
	v_div_fixup_f32 v135, v147, v143, v135
	v_div_fixup_f32 v136, v140, v143, v136
	v_cvt_f32_ubyte0_e32 v134, v134
	v_div_fixup_f32 v140, v148, v143, v137
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v149, v149, v150, v152
	v_div_scale_f32 v150, null, v143, v143, v139
	v_add_f32_e32 v132, 0, v132
	v_fmaak_f32 v134, s1, v134, 0x3f7fbe77
	v_add_f32_e32 v137, 0x3d23d70a, v136
	v_div_fixup_f32 v145, v149, v143, v138
	v_rcp_f32_e32 v151, v150
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v138, 0x3ca3d70a, v144 :: v_dual_mul_f32 v5, v134, v5
	v_mul_f32_e32 v126, v126, v134
	v_mul_f32_e32 v122, v122, v134
	v_dual_mul_f32 v128, v128, v134 :: v_dual_mul_f32 v129, v129, v134
	v_dual_mul_f32 v124, v124, v134 :: v_dual_mul_f32 v127, v127, v134
	v_mul_f32_e32 v120, v120, v134
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v152, -v150, v151, 1.0
	v_dual_mul_f32 v125, v125, v134 :: v_dual_mul_f32 v118, v118, v134
	v_dual_mul_f32 v123, v123, v134 :: v_dual_mul_f32 v116, v116, v134
	v_fmac_f32_e32 v151, v152, v151
	v_div_scale_f32 v152, vcc_lo, v139, v143, v139
	v_dual_mul_f32 v121, v121, v134 :: v_dual_mul_f32 v114, v114, v134
	v_dual_mul_f32 v119, v119, v134 :: v_dual_mul_f32 v112, v112, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v153, v152, v151
	v_dual_mul_f32 v117, v117, v134 :: v_dual_mul_f32 v110, v110, v134
	v_dual_mul_f32 v115, v115, v134 :: v_dual_mul_f32 v108, v108, v134
	v_fma_f32 v154, -v150, v153, v152
	v_dual_mul_f32 v113, v113, v134 :: v_dual_mul_f32 v106, v106, v134
	v_dual_mul_f32 v111, v111, v134 :: v_dual_mul_f32 v104, v104, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v153, v154, v151
	v_dual_mul_f32 v109, v109, v134 :: v_dual_mul_f32 v102, v102, v134
	v_dual_mul_f32 v107, v107, v134 :: v_dual_mul_f32 v100, v100, v134
	v_fma_f32 v150, -v150, v153, v152
	v_dual_mul_f32 v105, v105, v134 :: v_dual_mul_f32 v98, v98, v134
	v_dual_mul_f32 v103, v103, v134 :: v_dual_mul_f32 v96, v96, v134
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v150, v150, v151, v153
	v_dual_mul_f32 v101, v101, v134 :: v_dual_mul_f32 v94, v94, v134
	v_dual_mul_f32 v99, v99, v134 :: v_dual_mul_f32 v92, v92, v134
	v_div_fixup_f32 v143, v150, v143, v139
	v_dual_mul_f32 v97, v97, v134 :: v_dual_mul_f32 v90, v90, v134
	v_dual_mul_f32 v95, v95, v134 :: v_dual_mul_f32 v88, v88, v134
	v_dual_mul_f32 v93, v93, v134 :: v_dual_mul_f32 v86, v86, v134
	v_dual_mul_f32 v91, v91, v134 :: v_dual_mul_f32 v84, v84, v134
	v_dual_mul_f32 v89, v89, v134 :: v_dual_mul_f32 v82, v82, v134
	v_dual_mul_f32 v87, v87, v134 :: v_dual_mul_f32 v80, v80, v134
	v_dual_mul_f32 v85, v85, v134 :: v_dual_mul_f32 v78, v78, v134
	v_dual_mul_f32 v83, v83, v134 :: v_dual_mul_f32 v76, v76, v134
	v_dual_mul_f32 v81, v81, v134 :: v_dual_mul_f32 v74, v74, v134
	v_dual_mul_f32 v79, v79, v134 :: v_dual_mul_f32 v72, v72, v134
	v_dual_mul_f32 v77, v77, v134 :: v_dual_mul_f32 v70, v70, v134
	v_dual_mul_f32 v75, v75, v134 :: v_dual_mul_f32 v68, v68, v134
	v_dual_mul_f32 v73, v73, v134 :: v_dual_mul_f32 v66, v66, v134
	v_dual_mul_f32 v71, v71, v134 :: v_dual_mul_f32 v64, v64, v134
	v_dual_mul_f32 v69, v69, v134 :: v_dual_mul_f32 v62, v62, v134
	v_dual_mul_f32 v67, v67, v134 :: v_dual_mul_f32 v60, v60, v134
	v_dual_mul_f32 v65, v65, v134 :: v_dual_mul_f32 v58, v58, v134
	v_dual_mul_f32 v63, v63, v134 :: v_dual_mul_f32 v56, v56, v134
	v_dual_mul_f32 v61, v61, v134 :: v_dual_mul_f32 v54, v54, v134
	v_dual_mul_f32 v59, v59, v134 :: v_dual_mul_f32 v52, v52, v134
	v_dual_mul_f32 v57, v57, v134 :: v_dual_mul_f32 v50, v50, v134
	v_dual_mul_f32 v55, v55, v134 :: v_dual_mul_f32 v48, v48, v134
	v_dual_mul_f32 v53, v53, v134 :: v_dual_mul_f32 v46, v46, v134
	v_dual_mul_f32 v51, v51, v134 :: v_dual_mul_f32 v44, v44, v134
	v_dual_mul_f32 v49, v49, v134 :: v_dual_mul_f32 v42, v42, v134
	v_dual_mul_f32 v47, v47, v134 :: v_dual_mul_f32 v40, v40, v134
	v_dual_mul_f32 v45, v45, v134 :: v_dual_mul_f32 v38, v38, v134
	v_dual_mul_f32 v43, v43, v134 :: v_dual_mul_f32 v36, v36, v134
	v_dual_mul_f32 v41, v41, v134 :: v_dual_mul_f32 v34, v34, v134
	v_dual_mul_f32 v39, v39, v134 :: v_dual_mul_f32 v32, v134, v32
	v_dual_mul_f32 v37, v37, v134 :: v_dual_mul_f32 v28, v134, v28
	v_dual_mul_f32 v35, v35, v134 :: v_dual_mul_f32 v24, v134, v24
	v_dual_mul_f32 v33, v134, v33 :: v_dual_mul_f32 v30, v134, v30
	v_dual_mul_f32 v31, v134, v31 :: v_dual_mul_f32 v26, v134, v26
	v_dual_mul_f32 v29, v134, v29 :: v_dual_mul_f32 v22, v134, v22
	v_dual_mul_f32 v27, v134, v27 :: v_dual_mul_f32 v20, v134, v20
	v_dual_mul_f32 v25, v134, v25 :: v_dual_mul_f32 v18, v134, v18
	v_dual_mul_f32 v23, v134, v23 :: v_dual_mul_f32 v16, v134, v16
	v_dual_mul_f32 v21, v134, v21 :: v_dual_mul_f32 v14, v134, v14
	v_dual_mul_f32 v19, v134, v19 :: v_dual_mul_f32 v12, v134, v12
	v_dual_mul_f32 v17, v134, v17 :: v_dual_mul_f32 v10, v134, v10
	v_dual_mul_f32 v15, v134, v15 :: v_dual_mul_f32 v8, v134, v8
	v_dual_mul_f32 v13, v134, v13 :: v_dual_mul_f32 v6, v134, v6
	v_dual_mul_f32 v11, v134, v11 :: v_dual_mul_f32 v4, v134, v4
	v_dual_mul_f32 v9, v134, v9 :: v_dual_mul_f32 v2, v134, v2
	v_mul_f32_e32 v7, v134, v7
	v_dual_mul_f32 v3, v134, v3 :: v_dual_add_f32 v134, 0x3c23d70a, v133
	v_add_f32_e32 v133, 0x3cf5c28f, v135
	v_add_f32_e32 v139, 0x3d4ccccc, v140
	v_add_f32_e32 v135, 0x3d75c28f, v145
	v_add_f32_e32 v136, 0x3d8f5c29, v143
	s_cbranch_scc0 .LBB7_4
	s_branch .LBB7_6
.LBB7_5:
	v_dual_mov_b32 v129, 1.0 :: v_dual_mov_b32 v128, 0x3f8020c5
	v_mov_b32_e32 v127, 0x3f804189
	v_mov_b32_e32 v126, 0x3f80624e
	v_mov_b32_e32 v125, 0x3f808312
	v_mov_b32_e32 v124, 0x3f80a3d7
	v_mov_b32_e32 v123, 0x3f80c49c
	v_mov_b32_e32 v122, 0x3f80e560
	v_mov_b32_e32 v121, 0x3f8147ae
	v_mov_b32_e32 v120, 0x3f816873
	v_mov_b32_e32 v119, 0x3f818937
	v_mov_b32_e32 v118, 0x3f81a9fc
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v116, 0x3f81eb85
	v_mov_b32_e32 v115, 0x3f820c4a
	v_mov_b32_e32 v114, 0x3f822d0e
	v_mov_b32_e32 v113, 0x3f828f5c
	v_mov_b32_e32 v112, 0x3f82b021
	v_mov_b32_e32 v111, 0x3f82d0e5
	v_mov_b32_e32 v110, 0x3f82f1aa
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v108, 0x3f833333
	v_mov_b32_e32 v107, 0x3f8353f8
	v_mov_b32_e32 v106, 0x3f8374bc
	v_mov_b32_e32 v105, 0x3f83d70a
	v_mov_b32_e32 v104, 0x3f83f7cf
	v_mov_b32_e32 v103, 0x3f841893
	v_mov_b32_e32 v102, 0x3f843958
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v100, 0x3f847ae1
	v_mov_b32_e32 v99, 0x3f849ba6
	v_mov_b32_e32 v98, 0x3f84bc6a
	v_mov_b32_e32 v97, 0x3f851eb8
	v_mov_b32_e32 v96, 0x3f853f7d
	v_mov_b32_e32 v95, 0x3f856041
	v_mov_b32_e32 v94, 0x3f858106
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v92, 0x3f85c28f
	v_mov_b32_e32 v91, 0x3f85e354
	v_mov_b32_e32 v90, 0x3f860418
	v_mov_b32_e32 v89, 0x3f866666
	v_mov_b32_e32 v88, 0x3f86872b
	v_mov_b32_e32 v87, 0x3f86a7ef
	v_mov_b32_e32 v86, 0x3f86c8b4
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v84, 0x3f870a3d
	v_mov_b32_e32 v83, 0x3f872b02
	v_mov_b32_e32 v82, 0x3f874bc6
	v_mov_b32_e32 v81, 0x3f87ae14
	v_mov_b32_e32 v80, 0x3f87ced9
	v_mov_b32_e32 v79, 0x3f87ef9d
	v_mov_b32_e32 v78, 0x3f881062
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v76, 0x3f8851eb
	v_mov_b32_e32 v75, 0x3f8872b0
	v_mov_b32_e32 v74, 0x3f889374
	v_mov_b32_e32 v73, 0x3f88f5c3
	v_mov_b32_e32 v72, 0x3f891688
	v_mov_b32_e32 v71, 0x3f89374c
	v_mov_b32_e32 v70, 0x3f895811
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v68, 0x3f89999a
	v_mov_b32_e32 v67, 0x3f89ba5f
	v_mov_b32_e32 v66, 0x3f89db23
	v_mov_b32_e32 v65, 0x3f8a3d71
	v_mov_b32_e32 v64, 0x3f8a5e36
	v_mov_b32_e32 v63, 0x3f8a7efa
	v_mov_b32_e32 v62, 0x3f8a9fbf
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v60, 0x3f8ae148
	v_mov_b32_e32 v59, 0x3f8b020d
	v_mov_b32_e32 v58, 0x3f8b22d1
	v_mov_b32_e32 v57, 0x3f8b851f
	v_mov_b32_e32 v56, 0x3f8ba5e4
	v_mov_b32_e32 v55, 0x3f8bc6a8
	v_mov_b32_e32 v54, 0x3f8be76d
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v52, 0x3f8c28f6
	v_mov_b32_e32 v51, 0x3f8c49bb
	v_mov_b32_e32 v50, 0x3f8c6a7f
	v_mov_b32_e32 v49, 0x3f8ccccd
	v_mov_b32_e32 v48, 0x3f8ced92
	v_mov_b32_e32 v47, 0x3f8d0e56
	v_mov_b32_e32 v46, 0x3f8d2f1b
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v44, 0x3f8d70a4
	v_mov_b32_e32 v43, 0x3f8d9169
	v_mov_b32_e32 v42, 0x3f8db22d
	v_mov_b32_e32 v41, 0x3f8e147b
	v_mov_b32_e32 v40, 0x3f8e3540
	v_mov_b32_e32 v39, 0x3f8e5604
	v_mov_b32_e32 v38, 0x3f8e76c9
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v36, 0x3f8eb852
	v_mov_b32_e32 v35, 0x3f8ed917
	v_mov_b32_e32 v34, 0x3f8ef9db
	v_mov_b32_e32 v33, 0x3f8f5c29
	v_mov_b32_e32 v32, 0x3f8f7cee
	v_mov_b32_e32 v31, 0x3f8f9db2
	v_mov_b32_e32 v30, 0x3f8fbe77
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v28, 0x3f900000
	v_mov_b32_e32 v27, 0x3f9020c5
	v_mov_b32_e32 v26, 0x3f904189
	v_mov_b32_e32 v25, 0x3f90a3d7
	v_mov_b32_e32 v24, 0x3f90c49c
	v_mov_b32_e32 v23, 0x3f90e560
	v_mov_b32_e32 v22, 0x3f910625
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v20, 0x3f9147ae
	v_mov_b32_e32 v19, 0x3f916873
	v_mov_b32_e32 v18, 0x3f918937
	v_mov_b32_e32 v17, 0x3f91eb85
	v_mov_b32_e32 v16, 0x3f920c4a
	v_mov_b32_e32 v15, 0x3f922d0e
	v_mov_b32_e32 v14, 0x3f924dd3
	v_mov_b32_e32 v13, 0x3f926e97
	v_mov_b32_e32 v12, 0x3f928f5c
	v_mov_b32_e32 v11, 0x3f92b021
	v_mov_b32_e32 v10, 0x3f92d0e5
	v_mov_b32_e32 v9, 0x3f933333
	v_mov_b32_e32 v8, 0x3f9353f8
	v_mov_b32_e32 v7, 0x3f9374bc
	v_mov_b32_e32 v6, 0x3f939581
	v_mov_b32_e32 v5, 0x3f93b645
	v_mov_b32_e32 v4, 0x3f93d70a
	v_mov_b32_e32 v3, 0x3f93f7cf
	v_mov_b32_e32 v2, 0x3f941893
.LBB7_6:                                ; %Flow926
	v_add_f32_e32 v129, 0, v129
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v128, v129, v128
	v_add_f32_e32 v127, v128, v127
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v126, v127, v126
	v_add_f32_e32 v125, v126, v125
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v124, v125, v124
	v_add_f32_e32 v123, v124, v123
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v122, v123, v122
	v_add_f32_e32 v121, v122, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v120, v121, v120
	v_add_f32_e32 v119, v120, v119
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v118, v119, v118
	v_add_f32_e32 v117, v118, v117
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v116, v117, v116
	v_add_f32_e32 v115, v116, v115
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v114, v115, v114
	v_add_f32_e32 v113, v114, v113
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v112, v113, v112
	v_add_f32_e32 v111, v112, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v110, v111, v110
	v_add_f32_e32 v109, v110, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v108, v109, v108
	v_add_f32_e32 v107, v108, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v106, v107, v106
	v_add_f32_e32 v105, v106, v105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v104, v105, v104
	v_add_f32_e32 v103, v104, v103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v102, v103, v102
	v_add_f32_e32 v101, v102, v101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v100, v101, v100
	v_add_f32_e32 v99, v100, v99
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v98, v99, v98
	v_add_f32_e32 v97, v98, v97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v96, v97, v96
	v_add_f32_e32 v95, v96, v95
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v94, v95, v94
	v_add_f32_e32 v93, v94, v93
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v92, v93, v92
	v_add_f32_e32 v91, v92, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v90, v91, v90
	v_add_f32_e32 v89, v90, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v88, v89, v88
	v_add_f32_e32 v87, v88, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v86, v87, v86
	v_add_f32_e32 v85, v86, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v84, v85, v84
	v_add_f32_e32 v83, v84, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v82, v83, v82
	v_add_f32_e32 v81, v82, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v80, v81, v80
	v_add_f32_e32 v79, v80, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v78, v79, v78
	v_add_f32_e32 v77, v78, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v76, v77, v76
	v_add_f32_e32 v75, v76, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v74, v75, v74
	v_add_f32_e32 v73, v74, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v72, v73, v72
	v_add_f32_e32 v71, v72, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v70, v71, v70
	v_add_f32_e32 v69, v70, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v68, v69, v68
	v_add_f32_e32 v67, v68, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v66, v67, v66
	v_add_f32_e32 v65, v66, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v64, v65, v64
	v_add_f32_e32 v63, v64, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v62, v63, v62
	v_add_f32_e32 v61, v62, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v61, v60
	v_add_f32_e32 v59, v60, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v58, v59, v58
	v_add_f32_e32 v57, v58, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v56, v57, v56
	v_add_f32_e32 v55, v56, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v54, v55, v54
	v_add_f32_e32 v53, v54, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v52, v53, v52
	v_add_f32_e32 v51, v52, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v50, v51, v50
	v_add_f32_e32 v49, v50, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v48, v49, v48
	v_add_f32_e32 v47, v48, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v46, v47, v46
	v_add_f32_e32 v45, v46, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v44, v45, v44
	v_add_f32_e32 v43, v44, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v42, v43, v42
	v_add_f32_e32 v41, v42, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v40, v41, v40
	v_add_f32_e32 v39, v40, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v38, v39, v38
	v_add_f32_e32 v37, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v36, v37, v36
	v_add_f32_e32 v35, v36, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v34, v35, v34
	v_add_f32_e32 v33, v34, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v32, v33, v32
	v_add_f32_e32 v31, v32, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v30, v31, v30
	v_add_f32_e32 v29, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v28, v29, v28
	v_add_f32_e32 v27, v28, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v26, v27, v26
	v_add_f32_e32 v25, v26, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v24, v25, v24
	v_add_f32_e32 v23, v24, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v22, v23, v22
	v_add_f32_e32 v21, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v20, v21, v20
	v_add_f32_e32 v19, v20, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v19, v18
	v_add_f32_e32 v17, v18, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v17, v16
	v_add_f32_e32 v15, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v15, v14
	v_add_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v13, v12
	v_add_f32_e32 v11, v12, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v11, v10
	v_add_f32_e32 v9, v10, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v8, v9, v8
	v_add_f32_e32 v7, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v7, v6
	v_add_f32_e32 v5, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v5, v4
	v_add_f32_e32 v3, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v3, v2
	ds_load_b32 v3, v1
	v_add_f32_e32 v2, v2, v132
	v_add_f32_e32 v2, v2, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v138
	v_add_f32_e32 v1, v2, v133
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v3, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v135 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v136
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v3, 0x33800000, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end7:
	.size	_Z3runILi1ELi4EEvPfPji, .Lfunc_end7-_Z3runILi1ELi4EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi1ELi4EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 161
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-_Z3runILi1ELi4EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi1ELi4EEvPfPji,"axG",@progbits,_Z3runILi1ELi4EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi1ELi4EEvPfPji.num_vgpr, 161
	.set .L_Z3runILi1ELi4EEvPfPji.num_agpr, 0
	.set .L_Z3runILi1ELi4EEvPfPji.numbered_sgpr, 8
	.set .L_Z3runILi1ELi4EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi1ELi4EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi1ELi4EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi1ELi4EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi1ELi4EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi1ELi4EEvPfPji.has_recursion, 0
	.set .L_Z3runILi1ELi4EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5904
; TotalNumSgprs: 10
; NumVgprs: 161
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 161
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi2ELi4EEvPfPji,"axG",@progbits,_Z3runILi2ELi4EEvPfPji,comdat
	.protected	_Z3runILi2ELi4EEvPfPji  ; -- Begin function _Z3runILi2ELi4EEvPfPji
	.globl	_Z3runILi2ELi4EEvPfPji
	.p2align	8
	.type	_Z3runILi2ELi4EEvPfPji,@function
_Z3runILi2ELi4EEvPfPji:                 ; @_Z3runILi2ELi4EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB8_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB8_2:                                ; %.preheader79
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_kmcnt 0x0
	s_load_b32 s6, s[0:1], 0x10
	v_lshl_add_u32 v177, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s0, 0x3a83126f
	ds_store_b32 v177, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v161, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v162, s0, v1, 0x3dcccccd
	v_fmaak_f32 v163, s0, v1, 0x3e4ccccd
	v_fmaak_f32 v164, s0, v1, 0x3e99999a
	v_fmaak_f32 v165, s0, v1, 0x3ecccccd
	v_fma_f32 v166, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v167, s0, v1, 0x3f19999a
	v_fmaak_f32 v168, s0, v1, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s6, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB8_7
; %bb.3:                                ; %.lr.ph.preheader
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v8, 0x3f941893
	v_dual_mov_b32 v6, 0x3f93d70a :: v_dual_mov_b32 v121, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v13, 0x3f926e97 :: v_dual_mov_b32 v154, v169
	v_dual_mov_b32 v170, v169 :: v_dual_mov_b32 v171, v169
	v_dual_mov_b32 v172, v169 :: v_dual_mov_b32 v173, v169
	v_dual_mov_b32 v174, v169 :: v_dual_mov_b32 v175, v169
	v_dual_mov_b32 v176, v169 :: v_dual_mov_b32 v7, 0x3f93f7cf
	v_dual_mov_b32 v4, 0x3f939581 :: v_dual_mov_b32 v129, v169
	v_dual_mov_b32 v5, 0x3f93b645 :: v_dual_mov_b32 v130, v170
	v_dual_mov_b32 v3, 0x3f9374bc :: v_dual_mov_b32 v132, v172
	v_dual_mov_b32 v2, 0x3f9353f8 :: v_dual_mov_b32 v131, v171
	v_dual_mov_b32 v1, 0x3f933333 :: v_dual_mov_b32 v134, v174
	v_dual_mov_b32 v16, 0x3f92d0e5 :: v_dual_mov_b32 v133, v173
	v_dual_mov_b32 v15, 0x3f92b021 :: v_dual_mov_b32 v136, v176
	v_dual_mov_b32 v14, 0x3f928f5c :: v_dual_mov_b32 v135, v175
	v_dual_mov_b32 v12, 0x3f924dd3 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v11, 0x3f922d0e :: v_dual_mov_b32 v156, v169
	v_dual_mov_b32 v10, 0x3f920c4a :: v_dual_mov_b32 v155, v169
	v_dual_mov_b32 v9, 0x3f91eb85 :: v_dual_mov_b32 v158, v169
	v_dual_mov_b32 v24, 0x3f918937 :: v_dual_mov_b32 v157, v169
	v_dual_mov_b32 v23, 0x3f916873 :: v_dual_mov_b32 v160, v169
	v_dual_mov_b32 v22, 0x3f9147ae :: v_dual_mov_b32 v159, v169
	v_dual_mov_b32 v21, 0x3f9126e9 :: v_dual_mov_b32 v146, v169
	v_dual_mov_b32 v20, 0x3f910625 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v19, 0x3f90e560 :: v_dual_mov_b32 v148, v169
	v_dual_mov_b32 v18, 0x3f90c49c :: v_dual_mov_b32 v147, v169
	v_dual_mov_b32 v17, 0x3f90a3d7 :: v_dual_mov_b32 v150, v169
	v_dual_mov_b32 v32, 0x3f904189 :: v_dual_mov_b32 v149, v169
	v_dual_mov_b32 v31, 0x3f9020c5 :: v_dual_mov_b32 v152, v169
	v_dual_mov_b32 v30, 0x3f900000 :: v_dual_mov_b32 v151, v169
	v_dual_mov_b32 v29, 0x3f8fdf3b :: v_dual_mov_b32 v138, v169
	v_dual_mov_b32 v28, 0x3f8fbe77 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v27, 0x3f8f9db2 :: v_dual_mov_b32 v140, v169
	v_dual_mov_b32 v26, 0x3f8f7cee :: v_dual_mov_b32 v139, v169
	v_dual_mov_b32 v25, 0x3f8f5c29 :: v_dual_mov_b32 v142, v169
	v_dual_mov_b32 v40, 0x3f8ef9db :: v_dual_mov_b32 v141, v169
	v_dual_mov_b32 v39, 0x3f8ed917 :: v_dual_mov_b32 v144, v169
	v_dual_mov_b32 v38, 0x3f8eb852 :: v_dual_mov_b32 v143, v169
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v36, 0x3f8e76c9
	v_mov_b32_e32 v35, 0x3f8e5604
	v_mov_b32_e32 v34, 0x3f8e3540
	v_mov_b32_e32 v33, 0x3f8e147b
	v_mov_b32_e32 v48, 0x3f8db22d
	v_mov_b32_e32 v47, 0x3f8d9169
	v_mov_b32_e32 v46, 0x3f8d70a4
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v44, 0x3f8d2f1b
	v_mov_b32_e32 v43, 0x3f8d0e56
	v_mov_b32_e32 v42, 0x3f8ced92
	v_mov_b32_e32 v41, 0x3f8ccccd
	v_mov_b32_e32 v56, 0x3f8c6a7f
	v_mov_b32_e32 v55, 0x3f8c49bb
	v_mov_b32_e32 v54, 0x3f8c28f6
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v52, 0x3f8be76d
	v_mov_b32_e32 v51, 0x3f8bc6a8
	v_mov_b32_e32 v50, 0x3f8ba5e4
	v_mov_b32_e32 v49, 0x3f8b851f
	v_mov_b32_e32 v64, 0x3f8b22d1
	v_mov_b32_e32 v63, 0x3f8b020d
	v_mov_b32_e32 v62, 0x3f8ae148
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v60, 0x3f8a9fbf
	v_mov_b32_e32 v59, 0x3f8a7efa
	v_mov_b32_e32 v58, 0x3f8a5e36
	v_mov_b32_e32 v57, 0x3f8a3d71
	v_mov_b32_e32 v72, 0x3f89db23
	v_mov_b32_e32 v71, 0x3f89ba5f
	v_mov_b32_e32 v70, 0x3f89999a
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v68, 0x3f895811
	v_mov_b32_e32 v67, 0x3f89374c
	v_mov_b32_e32 v66, 0x3f891688
	v_mov_b32_e32 v65, 0x3f88f5c3
	v_mov_b32_e32 v80, 0x3f889374
	v_mov_b32_e32 v79, 0x3f8872b0
	v_mov_b32_e32 v78, 0x3f8851eb
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v76, 0x3f881062
	v_mov_b32_e32 v75, 0x3f87ef9d
	v_mov_b32_e32 v74, 0x3f87ced9
	v_mov_b32_e32 v73, 0x3f87ae14
	v_mov_b32_e32 v88, 0x3f874bc6
	v_mov_b32_e32 v87, 0x3f872b02
	v_mov_b32_e32 v86, 0x3f870a3d
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v84, 0x3f86c8b4
	v_mov_b32_e32 v83, 0x3f86a7ef
	v_mov_b32_e32 v82, 0x3f86872b
	v_mov_b32_e32 v81, 0x3f866666
	v_mov_b32_e32 v96, 0x3f860418
	v_mov_b32_e32 v95, 0x3f85e354
	v_mov_b32_e32 v94, 0x3f85c28f
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v92, 0x3f858106
	v_mov_b32_e32 v91, 0x3f856041
	v_mov_b32_e32 v90, 0x3f853f7d
	v_mov_b32_e32 v89, 0x3f851eb8
	v_mov_b32_e32 v104, 0x3f84bc6a
	v_mov_b32_e32 v103, 0x3f849ba6
	v_mov_b32_e32 v102, 0x3f847ae1
	v_mov_b32_e32 v101, 0x3f845a1c
	v_mov_b32_e32 v100, 0x3f843958
	v_mov_b32_e32 v99, 0x3f841893
	v_mov_b32_e32 v98, 0x3f83f7cf
	v_mov_b32_e32 v97, 0x3f83d70a
	v_mov_b32_e32 v112, 0x3f8374bc
	v_mov_b32_e32 v111, 0x3f8353f8
	v_mov_b32_e32 v110, 0x3f833333
	v_mov_b32_e32 v109, 0x3f83126e
	v_mov_b32_e32 v108, 0x3f82f1aa
	v_mov_b32_e32 v107, 0x3f82d0e5
	v_mov_b32_e32 v106, 0x3f82b021
	v_mov_b32_e32 v105, 0x3f828f5c
	v_mov_b32_e32 v120, 0x3f822d0e
	v_mov_b32_e32 v119, 0x3f820c4a
	v_mov_b32_e32 v118, 0x3f81eb85
	v_mov_b32_e32 v117, 0x3f81cac0
	v_mov_b32_e32 v116, 0x3f81a9fc
	v_mov_b32_e32 v115, 0x3f818937
	v_mov_b32_e32 v114, 0x3f816873
	v_mov_b32_e32 v113, 0x3f8147ae
	v_mov_b32_e32 v128, 0x3f80e560
	v_mov_b32_e32 v127, 0x3f80c49c
	v_mov_b32_e32 v126, 0x3f80a3d7
	v_mov_b32_e32 v125, 0x3f808312
	v_mov_b32_e32 v124, 0x3f80624e
	v_mov_b32_e32 v123, 0x3f804189
	v_mov_b32_e32 v122, 0x3f8020c5
	v_mbcnt_lo_u32_b32 v178, -1, 0
	s_mov_b32 s0, 0x10101010
	s_mov_b32 s2, 0x18181818
	s_mov_b32 s7, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s1, s0
	s_mov_b32 s3, s2
	s_mov_b32 s8, 0x35800000
.LBB8_4:                                ; %.lr.ph
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB8_5 Depth 2
	s_mov_b32 s9, 32
.LBB8_5:                                ; %.preheader.i
                                        ;   Parent Loop BB8_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v171, s3 :: v_dual_mov_b32 v170, s2
	v_dual_mov_b32 v173, s1 :: v_dual_mov_b32 v172, s0
	s_add_co_i32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s9, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[153:160], v[170:171], v[172:173], v[153:160]
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[170:171], v[172:173], v[145:152]
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[170:171], v[172:173], v[137:144]
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[170:171], v[172:173], v[129:136]
	s_cbranch_scc1 .LBB8_5
; %bb.6:                                ; %_Z6matrixILi4EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB8_4 Depth=1
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_xor_b32_e32 v171, 16, v178
	s_add_co_i32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, s6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v171
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v171, v178, v171, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v170, v161, 0xff800000, v162
	v_lshlrev_b32_e32 v172, 2, v171
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v170, v170, v163, v164
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v170, v170, v165, v166
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_max3_num_f32 v170, v170, v167, v168
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	ds_bpermute_b32 v171, v172, v170
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v171, v171, v171
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v170, v170, v171
	v_sub_f32_e32 v161, v161, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v161, 0x3fb8aa3b, v161 :: v_dual_sub_f32 v162, v162, v170
	v_dual_sub_f32 v163, v163, v170 :: v_dual_sub_f32 v164, v164, v170
	v_sub_f32_e32 v166, v166, v170
	v_exp_f32_e32 v161, v161
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v162, 0x3fb8aa3b, v162 :: v_dual_mul_f32 v163, 0x3fb8aa3b, v163
	v_dual_sub_f32 v165, v165, v170 :: v_dual_sub_f32 v168, v168, v170
	v_mul_f32_e32 v164, 0x3fb8aa3b, v164
	v_exp_f32_e32 v162, v162
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v163, v163
	v_mul_f32_e32 v166, 0x3fb8aa3b, v166
	v_mul_f32_e32 v168, 0x3fb8aa3b, v168
	v_exp_f32_e32 v164, v164
	v_exp_f32_e32 v166, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_exp_f32_e32 v168, v168
	v_add_f32_e32 v171, v161, v162
	v_mul_f32_e32 v165, 0x3fb8aa3b, v165
	v_sub_f32_e32 v167, v167, v170
	v_mul_f32_e32 v161, 0x3a83126f, v161
	v_dual_mul_f32 v162, 0x3b03126f, v162 :: v_dual_add_f32 v171, v163, v171
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v165, v165
	v_mul_f32_e32 v167, 0x3fb8aa3b, v167
	v_mul_f32_e32 v163, 0x3b449ba6, v163
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_max3_num_f32 v173, v161, 0, v162
	v_dual_add_f32 v171, v164, v171 :: v_dual_mul_f32 v164, 0x3b83126f, v164
	v_exp_f32_e32 v167, v167
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v171, v165, v171
	v_mul_f32_e32 v165, 0x3ba3d70b, v165
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v173, v173, v163, v164
	v_dual_add_f32 v171, v166, v171 :: v_dual_mul_f32 v166, 0x3bc49ba6, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v171, v167, v171
	v_mul_f32_e32 v167, 0x3be56042, v167
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v173, v173, v165, v166
	v_add_f32_e32 v170, v168, v171
	v_mul_f32_e32 v168, 0x3c03126f, v168
	ds_bpermute_b32 v171, v172, v170
	v_max3_num_f32 v173, v173, v167, v168
	ds_bpermute_b32 v172, v172, v173
	s_wait_dscnt 0x1
	v_add_f32_e32 v170, v170, v171
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v172, v172, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max_num_f32_e32 v172, v173, v172
	v_div_scale_f32 v171, null, v170, v170, v161
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v173, null, 0x43e00000, 0x43e00000, v172
	v_rcp_f32_e32 v174, v173
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v175, -v173, v174, 1.0
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v172, 0x43e00000, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v176, v175, v174
	v_fma_f32 v179, -v173, v176, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v179, v174
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v173, v173, v174, v176
	v_div_fixup_f32 v172, v173, 0x43e00000, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v172, 0x1f800000, v172
	v_div_scale_f32 v173, null, v172, v172, v161
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v161, v172, v161
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v173, v176, v175
	v_fmac_f32_e32 v176, v179, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	v_div_scale_f32 v174, null, v172, v172, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v175, v174
	v_fma_f32 v176, -v174, v175, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v175, v176, v175
	v_div_scale_f32 v176, vcc_lo, v162, v172, v162
	v_div_fixup_f32 v173, v173, v172, v161
	v_mul_f32_e32 v179, v176, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v174, v179, v176
	v_fmac_f32_e32 v179, v180, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v179, v176
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v175, v179
	v_mov_b16_e64 v175.l, v169.l
	v_mov_b16_e64 v175.h, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v174, v174, v172, v162
	v_mov_b16_e64 v176.l, v175.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b16_e64 v176.h, v175.h
	v_cvt_pk_fp8_f32 v176.l, v173, v174
	v_div_scale_f32 v173, null, v172, v172, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v179, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v179, v174
	v_div_scale_f32 v179, vcc_lo, v163, v172, v163
	v_mul_f32_e32 v180, v179, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v173, v180, v179
	v_fmac_f32_e32 v180, v181, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v180, v179
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v180
	v_div_scale_f32 v174, null, v172, v172, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v173, v173, v172, v163
	v_rcp_f32_e32 v179, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v174, v179, 1.0
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v164, v172, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v181, v180, v179
	v_fma_f32 v182, -v174, v181, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v182, v179
	v_fma_f32 v174, -v174, v181, v180
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v174, v174, v179, v181
	v_div_fixup_f32 v174, v174, v172, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v176.h, v173, v174
	v_div_scale_f32 v173, null, v172, v172, v165
	v_rcp_f32_e32 v174, v173
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v173, v174, 1.0
	v_fmac_f32_e32 v174, v179, v174
	v_div_scale_f32 v179, vcc_lo, v165, v172, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v174
	v_fma_f32 v181, -v173, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v174
	v_fma_f32 v173, -v173, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v173, v173, v174, v180
	v_div_scale_f32 v174, null, v172, v172, v166
	v_div_fixup_f32 v173, v173, v172, v165
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v174
	v_fma_f32 v180, -v174, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v166, v172, v166
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v174, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v174, v174, v172, v166
	v_cvt_pk_fp8_f32 v175.l, v173, v174
	v_div_scale_f32 v173, null, v172, v172, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v179, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v179, v174
	v_div_scale_f32 v179, vcc_lo, v167, v172, v167
	v_mul_f32_e32 v180, v179, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v173, v180, v179
	v_fmac_f32_e32 v180, v181, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v180, v179
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v180
	v_div_scale_f32 v174, null, v172, v172, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v173, v173, v172, v167
	v_rcp_f32_e32 v179, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v174, v179, 1.0
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v168, v172, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v181, v180, v179
	v_fma_f32 v182, -v174, v181, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v182, v179
	v_fma_f32 v174, -v174, v181, v180
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v174, v174, v179, v181
	v_div_fixup_f32 v172, v174, v172, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v175.h, v173, v172
	v_xor_b32_e32 v172, v176, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v172, v172
	v_fmaak_f32 v172, s8, v172, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v128, v128, v172 :: v_dual_mul_f32 v121, v121, v172
	v_dual_mul_f32 v127, v127, v172 :: v_dual_mul_f32 v126, v126, v172
	v_dual_mul_f32 v125, v125, v172 :: v_dual_mul_f32 v124, v124, v172
	v_mul_f32_e32 v119, v119, v172
	v_dual_mul_f32 v123, v123, v172 :: v_dual_mul_f32 v122, v122, v172
	v_dual_mul_f32 v117, v117, v172 :: v_dual_mul_f32 v120, v120, v172
	v_dual_mul_f32 v115, v115, v172 :: v_dual_mul_f32 v118, v118, v172
	v_dual_mul_f32 v113, v113, v172 :: v_dual_mul_f32 v116, v116, v172
	v_dual_mul_f32 v111, v111, v172 :: v_dual_mul_f32 v114, v114, v172
	v_dual_mul_f32 v109, v109, v172 :: v_dual_mul_f32 v112, v112, v172
	v_dual_mul_f32 v107, v107, v172 :: v_dual_mul_f32 v110, v110, v172
	v_dual_mul_f32 v105, v105, v172 :: v_dual_mul_f32 v108, v108, v172
	v_dual_mul_f32 v103, v103, v172 :: v_dual_mul_f32 v106, v106, v172
	v_dual_mul_f32 v101, v101, v172 :: v_dual_mul_f32 v104, v104, v172
	v_dual_mul_f32 v99, v99, v172 :: v_dual_mul_f32 v102, v102, v172
	v_dual_mul_f32 v97, v97, v172 :: v_dual_mul_f32 v100, v100, v172
	v_dual_mul_f32 v95, v95, v172 :: v_dual_mul_f32 v98, v98, v172
	v_dual_mul_f32 v93, v93, v172 :: v_dual_mul_f32 v96, v96, v172
	v_dual_mul_f32 v91, v91, v172 :: v_dual_mul_f32 v94, v94, v172
	v_dual_mul_f32 v89, v89, v172 :: v_dual_mul_f32 v92, v92, v172
	v_dual_mul_f32 v87, v87, v172 :: v_dual_mul_f32 v90, v90, v172
	v_dual_mul_f32 v85, v85, v172 :: v_dual_mul_f32 v88, v88, v172
	v_dual_mul_f32 v83, v83, v172 :: v_dual_mul_f32 v86, v86, v172
	v_dual_mul_f32 v81, v81, v172 :: v_dual_mul_f32 v84, v84, v172
	v_dual_mul_f32 v79, v79, v172 :: v_dual_mul_f32 v82, v82, v172
	v_dual_mul_f32 v77, v77, v172 :: v_dual_mul_f32 v80, v80, v172
	v_dual_mul_f32 v75, v75, v172 :: v_dual_mul_f32 v78, v78, v172
	v_dual_mul_f32 v73, v73, v172 :: v_dual_mul_f32 v76, v76, v172
	v_dual_mul_f32 v71, v71, v172 :: v_dual_mul_f32 v74, v74, v172
	v_dual_mul_f32 v69, v69, v172 :: v_dual_mul_f32 v72, v72, v172
	v_dual_mul_f32 v67, v67, v172 :: v_dual_mul_f32 v70, v70, v172
	v_dual_mul_f32 v65, v65, v172 :: v_dual_mul_f32 v68, v68, v172
	v_dual_mul_f32 v63, v63, v172 :: v_dual_mul_f32 v66, v66, v172
	v_dual_mul_f32 v61, v61, v172 :: v_dual_mul_f32 v64, v64, v172
	v_dual_mul_f32 v59, v59, v172 :: v_dual_mul_f32 v62, v62, v172
	v_dual_mul_f32 v57, v57, v172 :: v_dual_mul_f32 v60, v60, v172
	v_dual_mul_f32 v55, v55, v172 :: v_dual_mul_f32 v58, v58, v172
	v_dual_mul_f32 v53, v53, v172 :: v_dual_mul_f32 v56, v56, v172
	v_dual_mul_f32 v51, v51, v172 :: v_dual_mul_f32 v54, v54, v172
	v_dual_mul_f32 v49, v49, v172 :: v_dual_mul_f32 v52, v52, v172
	v_dual_mul_f32 v47, v47, v172 :: v_dual_mul_f32 v50, v50, v172
	v_dual_mul_f32 v45, v45, v172 :: v_dual_mul_f32 v48, v48, v172
	v_dual_mul_f32 v43, v43, v172 :: v_dual_mul_f32 v46, v46, v172
	v_dual_mul_f32 v41, v41, v172 :: v_dual_mul_f32 v44, v44, v172
	v_dual_mul_f32 v39, v39, v172 :: v_dual_mul_f32 v42, v42, v172
	v_dual_mul_f32 v37, v37, v172 :: v_dual_mul_f32 v40, v40, v172
	v_dual_mul_f32 v35, v35, v172 :: v_dual_mul_f32 v38, v38, v172
	v_dual_mul_f32 v33, v33, v172 :: v_dual_mul_f32 v36, v36, v172
	v_dual_mul_f32 v7, v7, v172 :: v_dual_mul_f32 v34, v34, v172
	v_dual_mul_f32 v31, v172, v31 :: v_dual_mul_f32 v32, v172, v32
	v_dual_mul_f32 v29, v172, v29 :: v_dual_mul_f32 v30, v172, v30
	v_dual_mul_f32 v27, v172, v27 :: v_dual_mul_f32 v28, v172, v28
	v_dual_mul_f32 v25, v172, v25 :: v_dual_mul_f32 v26, v172, v26
	v_dual_mul_f32 v23, v172, v23 :: v_dual_mul_f32 v24, v172, v24
	v_dual_mul_f32 v21, v172, v21 :: v_dual_mul_f32 v22, v172, v22
	v_dual_mul_f32 v19, v172, v19 :: v_dual_mul_f32 v20, v172, v20
	v_dual_mul_f32 v17, v172, v17 :: v_dual_mul_f32 v18, v172, v18
	v_dual_mul_f32 v15, v172, v15 :: v_dual_mul_f32 v16, v172, v16
	v_dual_mul_f32 v13, v172, v13 :: v_dual_mul_f32 v14, v172, v14
	v_dual_mul_f32 v11, v172, v11 :: v_dual_mul_f32 v12, v172, v12
	v_dual_mul_f32 v9, v172, v9 :: v_dual_mul_f32 v10, v172, v10
	v_dual_mul_f32 v5, v5, v172 :: v_dual_mul_f32 v8, v8, v172
	v_dual_mul_f32 v3, v3, v172 :: v_dual_mul_f32 v6, v6, v172
	v_dual_mul_f32 v1, v1, v172 :: v_dual_mul_f32 v4, v4, v172
	v_mul_f32_e32 v2, v2, v172
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v161, v170, v161
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v161, v171, v170, v161
	v_div_scale_f32 v171, null, v170, v170, v162
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v161, 0, v161
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v162, v170, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v162, v171, v170, v162
	v_div_scale_f32 v171, null, v170, v170, v163
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v162, 0x3c23d70a, v162
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v163, v170, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v163, v171, v170, v163
	v_div_scale_f32 v171, null, v170, v170, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v163, 0x3ca3d70a, v163
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v164, v170, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v164, v171, v170, v164
	v_div_scale_f32 v171, null, v170, v170, v165
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v164, 0x3cf5c28f, v164
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v165, v170, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v165, v171, v170, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v165, 0x3d23d70a, v165
	v_div_scale_f32 v171, null, v170, v170, v166
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v166, v170, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v166, v171, v170, v166
	v_div_scale_f32 v171, null, v170, v170, v167
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v166, 0x3d4ccccc, v166
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v167, v170, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v167, v171, v170, v167
	v_div_scale_f32 v171, null, v170, v170, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v167, 0x3d75c28f, v167
	v_rcp_f32_e32 v172, v171
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v171, v172, 1.0
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v168, v170, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v174, v173, v172
	v_fma_f32 v175, -v171, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v172
	v_fma_f32 v171, -v171, v174, v173
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v171, v171, v172, v174
	v_div_fixup_f32 v168, v171, v170, v168
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v168, 0x3d8f5c29, v168
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB8_4
	s_branch .LBB8_8
.LBB8_7:
	v_dual_mov_b32 v121, 1.0 :: v_dual_mov_b32 v122, 0x3f8020c5
	v_dual_mov_b32 v123, 0x3f804189 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v124, 0x3f80624e :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v125, 0x3f808312 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v126, 0x3f80a3d7 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v127, 0x3f80c49c :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v128, 0x3f80e560 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v113, 0x3f8147ae :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v114, 0x3f816873 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v115, 0x3f818937 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v116, 0x3f81a9fc :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v117, 0x3f81cac0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v118, 0x3f81eb85 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v119, 0x3f820c4a :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v120, 0x3f822d0e :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v105, 0x3f828f5c :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v106, 0x3f82b021 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v107, 0x3f82d0e5 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v108, 0x3f82f1aa :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v109, 0x3f83126e :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v110, 0x3f833333 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v111, 0x3f8353f8 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v112, 0x3f8374bc :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v97, 0x3f83d70a :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v98, 0x3f83f7cf :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v99, 0x3f841893 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v100, 0x3f843958 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v101, 0x3f845a1c :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v102, 0x3f847ae1 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v103, 0x3f849ba6 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v104, 0x3f84bc6a :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v89, 0x3f851eb8 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v90, 0x3f853f7d :: v_dual_mov_b32 v159, 0
	v_mov_b32_e32 v91, 0x3f856041
	v_mov_b32_e32 v92, 0x3f858106
	v_mov_b32_e32 v93, 0x3f85a1ca
	v_mov_b32_e32 v94, 0x3f85c28f
	v_mov_b32_e32 v95, 0x3f85e354
	v_mov_b32_e32 v96, 0x3f860418
	v_mov_b32_e32 v81, 0x3f866666
	v_mov_b32_e32 v82, 0x3f86872b
	v_mov_b32_e32 v83, 0x3f86a7ef
	v_mov_b32_e32 v84, 0x3f86c8b4
	v_mov_b32_e32 v85, 0x3f86e978
	v_mov_b32_e32 v86, 0x3f870a3d
	v_mov_b32_e32 v87, 0x3f872b02
	v_mov_b32_e32 v88, 0x3f874bc6
	v_mov_b32_e32 v73, 0x3f87ae14
	v_mov_b32_e32 v74, 0x3f87ced9
	v_mov_b32_e32 v75, 0x3f87ef9d
	v_mov_b32_e32 v76, 0x3f881062
	v_mov_b32_e32 v77, 0x3f883126
	v_mov_b32_e32 v78, 0x3f8851eb
	v_mov_b32_e32 v79, 0x3f8872b0
	v_mov_b32_e32 v80, 0x3f889374
	v_mov_b32_e32 v65, 0x3f88f5c3
	v_mov_b32_e32 v66, 0x3f891688
	v_mov_b32_e32 v67, 0x3f89374c
	v_mov_b32_e32 v68, 0x3f895811
	v_mov_b32_e32 v69, 0x3f8978d5
	v_mov_b32_e32 v70, 0x3f89999a
	v_mov_b32_e32 v71, 0x3f89ba5f
	v_mov_b32_e32 v72, 0x3f89db23
	v_mov_b32_e32 v57, 0x3f8a3d71
	v_mov_b32_e32 v58, 0x3f8a5e36
	v_mov_b32_e32 v59, 0x3f8a7efa
	v_mov_b32_e32 v60, 0x3f8a9fbf
	v_mov_b32_e32 v61, 0x3f8ac083
	v_mov_b32_e32 v62, 0x3f8ae148
	v_mov_b32_e32 v63, 0x3f8b020d
	v_mov_b32_e32 v64, 0x3f8b22d1
	v_mov_b32_e32 v49, 0x3f8b851f
	v_mov_b32_e32 v50, 0x3f8ba5e4
	v_mov_b32_e32 v51, 0x3f8bc6a8
	v_mov_b32_e32 v52, 0x3f8be76d
	v_mov_b32_e32 v53, 0x3f8c0831
	v_mov_b32_e32 v54, 0x3f8c28f6
	v_mov_b32_e32 v55, 0x3f8c49bb
	v_mov_b32_e32 v56, 0x3f8c6a7f
	v_mov_b32_e32 v41, 0x3f8ccccd
	v_mov_b32_e32 v42, 0x3f8ced92
	v_mov_b32_e32 v43, 0x3f8d0e56
	v_mov_b32_e32 v44, 0x3f8d2f1b
	v_mov_b32_e32 v45, 0x3f8d4fdf
	v_mov_b32_e32 v46, 0x3f8d70a4
	v_mov_b32_e32 v47, 0x3f8d9169
	v_mov_b32_e32 v48, 0x3f8db22d
	v_mov_b32_e32 v33, 0x3f8e147b
	v_mov_b32_e32 v34, 0x3f8e3540
	v_mov_b32_e32 v35, 0x3f8e5604
	v_mov_b32_e32 v36, 0x3f8e76c9
	v_mov_b32_e32 v37, 0x3f8e978d
	v_mov_b32_e32 v38, 0x3f8eb852
	v_mov_b32_e32 v39, 0x3f8ed917
	v_mov_b32_e32 v40, 0x3f8ef9db
	v_mov_b32_e32 v25, 0x3f8f5c29
	v_mov_b32_e32 v26, 0x3f8f7cee
	v_mov_b32_e32 v27, 0x3f8f9db2
	v_mov_b32_e32 v28, 0x3f8fbe77
	v_mov_b32_e32 v29, 0x3f8fdf3b
	v_mov_b32_e32 v30, 0x3f900000
	v_mov_b32_e32 v31, 0x3f9020c5
	v_mov_b32_e32 v32, 0x3f904189
	v_mov_b32_e32 v17, 0x3f90a3d7
	v_mov_b32_e32 v18, 0x3f90c49c
	v_mov_b32_e32 v19, 0x3f90e560
	v_mov_b32_e32 v20, 0x3f910625
	v_mov_b32_e32 v21, 0x3f9126e9
	v_mov_b32_e32 v22, 0x3f9147ae
	v_mov_b32_e32 v23, 0x3f916873
	v_mov_b32_e32 v24, 0x3f918937
	v_mov_b32_e32 v9, 0x3f91eb85
	v_mov_b32_e32 v10, 0x3f920c4a
	v_mov_b32_e32 v11, 0x3f922d0e
	v_mov_b32_e32 v12, 0x3f924dd3
	v_mov_b32_e32 v13, 0x3f926e97
	v_mov_b32_e32 v14, 0x3f928f5c
	v_mov_b32_e32 v15, 0x3f92b021
	v_mov_b32_e32 v16, 0x3f92d0e5
	v_mov_b32_e32 v1, 0x3f933333
	v_mov_b32_e32 v2, 0x3f9353f8
	v_mov_b32_e32 v3, 0x3f9374bc
	v_mov_b32_e32 v4, 0x3f939581
	v_mov_b32_e32 v5, 0x3f93b645
	v_mov_b32_e32 v6, 0x3f93d70a
	v_mov_b32_e32 v7, 0x3f93f7cf
	v_mov_b32_e32 v8, 0x3f941893
.LBB8_8:                                ; %Flow
	v_add_f32_e32 v153, 0, v153
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v153, v153, v154
	v_add_f32_e32 v153, v153, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v153, v153, v156
	v_add_f32_e32 v153, v153, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v153, v153, v158
	v_add_f32_e32 v153, v153, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v153, v153, v160
	v_add_f32_e32 v145, v153, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v146
	v_add_f32_e32 v145, v145, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v148
	v_add_f32_e32 v145, v145, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v150
	v_add_f32_e32 v145, v145, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v152
	v_add_f32_e32 v137, v145, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v138
	v_add_f32_e32 v137, v137, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v140
	v_add_f32_e32 v137, v137, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v142
	v_add_f32_e32 v137, v137, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v144
	v_add_f32_e32 v129, v137, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v130
	v_add_f32_e32 v129, v129, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v132
	v_add_f32_e32 v129, v129, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v134
	v_add_f32_e32 v129, v129, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v136
	v_add_f32_e32 v121, v129, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v122
	v_add_f32_e32 v121, v121, v123
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v124
	v_add_f32_e32 v121, v121, v125
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v126
	v_add_f32_e32 v121, v121, v127
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v128
	v_add_f32_e32 v113, v121, v113
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v114
	v_add_f32_e32 v113, v113, v115
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v116
	v_add_f32_e32 v113, v113, v117
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v118
	v_add_f32_e32 v113, v113, v119
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v120
	v_add_f32_e32 v105, v113, v105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v106
	v_add_f32_e32 v105, v105, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v108
	v_add_f32_e32 v105, v105, v109
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v110
	v_add_f32_e32 v105, v105, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v112
	v_add_f32_e32 v97, v105, v97
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v98
	v_add_f32_e32 v97, v97, v99
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v100
	v_add_f32_e32 v97, v97, v101
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v102
	v_add_f32_e32 v97, v97, v103
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v104
	v_add_f32_e32 v89, v97, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v90
	v_add_f32_e32 v89, v89, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v92
	v_add_f32_e32 v89, v89, v93
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v94
	v_add_f32_e32 v89, v89, v95
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v96
	v_add_f32_e32 v81, v89, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v82
	v_add_f32_e32 v81, v81, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v84
	v_add_f32_e32 v81, v81, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v86
	v_add_f32_e32 v81, v81, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v88
	v_add_f32_e32 v73, v81, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v74
	v_add_f32_e32 v73, v73, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v76
	v_add_f32_e32 v73, v73, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v78
	v_add_f32_e32 v73, v73, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v80
	v_add_f32_e32 v65, v73, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v66
	v_add_f32_e32 v65, v65, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v68
	v_add_f32_e32 v65, v65, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v70
	v_add_f32_e32 v65, v65, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v72
	v_add_f32_e32 v57, v65, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v58
	v_add_f32_e32 v57, v57, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v60
	v_add_f32_e32 v57, v57, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v62
	v_add_f32_e32 v57, v57, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v64
	v_add_f32_e32 v49, v57, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v50
	v_add_f32_e32 v49, v49, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v52
	v_add_f32_e32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v54
	v_add_f32_e32 v49, v49, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v56
	v_add_f32_e32 v41, v49, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v42
	v_add_f32_e32 v41, v41, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v44
	v_add_f32_e32 v41, v41, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v46
	v_add_f32_e32 v41, v41, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v48
	v_add_f32_e32 v33, v41, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v34
	v_add_f32_e32 v33, v33, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v36
	v_add_f32_e32 v33, v33, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v38
	v_add_f32_e32 v33, v33, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v40
	v_add_f32_e32 v25, v33, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v177
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v161
	v_add_f32_e32 v1, v1, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v163
	v_add_f32_e32 v1, v1, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v165
	v_add_f32_e32 v1, v1, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v167
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end8:
	.size	_Z3runILi2ELi4EEvPfPji, .Lfunc_end8-_Z3runILi2ELi4EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi2ELi4EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 183
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end8-_Z3runILi2ELi4EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi2ELi4EEvPfPji,"axG",@progbits,_Z3runILi2ELi4EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi2ELi4EEvPfPji.num_vgpr, 183
	.set .L_Z3runILi2ELi4EEvPfPji.num_agpr, 0
	.set .L_Z3runILi2ELi4EEvPfPji.numbered_sgpr, 10
	.set .L_Z3runILi2ELi4EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi2ELi4EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi2ELi4EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi2ELi4EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi2ELi4EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi2ELi4EEvPfPji.has_recursion, 0
	.set .L_Z3runILi2ELi4EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6608
; TotalNumSgprs: 12
; NumVgprs: 183
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 183
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi3ELi4EEvPfPji,"axG",@progbits,_Z3runILi3ELi4EEvPfPji,comdat
	.protected	_Z3runILi3ELi4EEvPfPji  ; -- Begin function _Z3runILi3ELi4EEvPfPji
	.globl	_Z3runILi3ELi4EEvPfPji
	.p2align	8
	.type	_Z3runILi3ELi4EEvPfPji,@function
_Z3runILi3ELi4EEvPfPji:                 ; @_Z3runILi3ELi4EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB9_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB9_2:                                ; %.preheader108
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s1, s[0:1], 0x10
	v_lshl_add_u32 v185, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s0, 0x3a83126f
	v_mov_b32_e32 v2, 0x3f8020c5
	v_mov_b32_e32 v3, 0x3f804189
	ds_store_b32 v185, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v161, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v162, s0, v1, 0x3dcccccd
	v_fmaak_f32 v163, s0, v1, 0x3e4ccccd
	v_fmaak_f32 v164, s0, v1, 0x3e99999a
	v_fmaak_f32 v165, s0, v1, 0x3ecccccd
	v_fma_f32 v166, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v167, s0, v1, 0x3f19999a
	v_dual_fmaak_f32 v168, s0, v1, 0x3f333333 :: v_dual_mov_b32 v1, 1.0
	v_mov_b32_e32 v4, 0x3f80624e
	v_mov_b32_e32 v5, 0x3f808312
	v_mov_b32_e32 v6, 0x3f80a3d7
	v_mov_b32_e32 v7, 0x3f80c49c
	v_mov_b32_e32 v8, 0x3f80e560
	v_mov_b32_e32 v9, 0x3f8147ae
	v_mov_b32_e32 v10, 0x3f816873
	v_mov_b32_e32 v11, 0x3f818937
	v_mov_b32_e32 v12, 0x3f81a9fc
	v_mov_b32_e32 v13, 0x3f81cac0
	v_mov_b32_e32 v14, 0x3f81eb85
	v_mov_b32_e32 v15, 0x3f820c4a
	v_mov_b32_e32 v16, 0x3f822d0e
	v_mov_b32_e32 v17, 0x3f828f5c
	v_mov_b32_e32 v18, 0x3f82b021
	v_mov_b32_e32 v19, 0x3f82d0e5
	v_mov_b32_e32 v20, 0x3f82f1aa
	v_mov_b32_e32 v21, 0x3f83126e
	v_mov_b32_e32 v22, 0x3f833333
	v_mov_b32_e32 v23, 0x3f8353f8
	v_mov_b32_e32 v24, 0x3f8374bc
	v_mov_b32_e32 v25, 0x3f83d70a
	v_mov_b32_e32 v26, 0x3f83f7cf
	v_mov_b32_e32 v27, 0x3f841893
	v_mov_b32_e32 v28, 0x3f843958
	v_mov_b32_e32 v29, 0x3f845a1c
	v_mov_b32_e32 v30, 0x3f847ae1
	v_mov_b32_e32 v31, 0x3f849ba6
	v_mov_b32_e32 v32, 0x3f84bc6a
	v_mov_b32_e32 v33, 0x3f851eb8
	v_mov_b32_e32 v34, 0x3f853f7d
	v_mov_b32_e32 v35, 0x3f856041
	v_mov_b32_e32 v36, 0x3f858106
	v_mov_b32_e32 v37, 0x3f85a1ca
	v_mov_b32_e32 v38, 0x3f85c28f
	v_mov_b32_e32 v39, 0x3f85e354
	v_mov_b32_e32 v40, 0x3f860418
	v_mov_b32_e32 v41, 0x3f866666
	v_mov_b32_e32 v42, 0x3f86872b
	v_mov_b32_e32 v43, 0x3f86a7ef
	v_mov_b32_e32 v44, 0x3f86c8b4
	v_mov_b32_e32 v45, 0x3f86e978
	v_mov_b32_e32 v46, 0x3f870a3d
	v_mov_b32_e32 v47, 0x3f872b02
	v_mov_b32_e32 v48, 0x3f874bc6
	v_mov_b32_e32 v49, 0x3f87ae14
	v_mov_b32_e32 v50, 0x3f87ced9
	v_mov_b32_e32 v51, 0x3f87ef9d
	v_mov_b32_e32 v52, 0x3f881062
	v_mov_b32_e32 v53, 0x3f883126
	v_mov_b32_e32 v54, 0x3f8851eb
	v_mov_b32_e32 v55, 0x3f8872b0
	v_mov_b32_e32 v56, 0x3f889374
	v_mov_b32_e32 v57, 0x3f88f5c3
	v_mov_b32_e32 v58, 0x3f891688
	v_mov_b32_e32 v59, 0x3f89374c
	v_mov_b32_e32 v60, 0x3f895811
	v_mov_b32_e32 v61, 0x3f8978d5
	v_mov_b32_e32 v62, 0x3f89999a
	v_mov_b32_e32 v63, 0x3f89ba5f
	v_mov_b32_e32 v64, 0x3f89db23
	v_mov_b32_e32 v65, 0x3f8a3d71
	v_mov_b32_e32 v66, 0x3f8a5e36
	v_mov_b32_e32 v67, 0x3f8a7efa
	v_mov_b32_e32 v68, 0x3f8a9fbf
	v_mov_b32_e32 v69, 0x3f8ac083
	v_mov_b32_e32 v70, 0x3f8ae148
	v_mov_b32_e32 v71, 0x3f8b020d
	v_mov_b32_e32 v72, 0x3f8b22d1
	v_mov_b32_e32 v73, 0x3f8b851f
	v_mov_b32_e32 v74, 0x3f8ba5e4
	v_mov_b32_e32 v75, 0x3f8bc6a8
	v_mov_b32_e32 v76, 0x3f8be76d
	v_mov_b32_e32 v77, 0x3f8c0831
	v_mov_b32_e32 v78, 0x3f8c28f6
	v_mov_b32_e32 v79, 0x3f8c49bb
	v_mov_b32_e32 v80, 0x3f8c6a7f
	v_mov_b32_e32 v81, 0x3f8ccccd
	v_mov_b32_e32 v82, 0x3f8ced92
	v_mov_b32_e32 v83, 0x3f8d0e56
	v_mov_b32_e32 v84, 0x3f8d2f1b
	v_mov_b32_e32 v85, 0x3f8d4fdf
	v_mov_b32_e32 v86, 0x3f8d70a4
	v_mov_b32_e32 v87, 0x3f8d9169
	v_mov_b32_e32 v88, 0x3f8db22d
	v_mov_b32_e32 v89, 0x3f8e147b
	v_mov_b32_e32 v90, 0x3f8e3540
	v_mov_b32_e32 v91, 0x3f8e5604
	v_mov_b32_e32 v92, 0x3f8e76c9
	v_mov_b32_e32 v93, 0x3f8e978d
	v_mov_b32_e32 v94, 0x3f8eb852
	v_mov_b32_e32 v95, 0x3f8ed917
	v_mov_b32_e32 v96, 0x3f8ef9db
	v_mov_b32_e32 v97, 0x3f8f5c29
	v_mov_b32_e32 v98, 0x3f8f7cee
	v_mov_b32_e32 v99, 0x3f8f9db2
	v_mov_b32_e32 v100, 0x3f8fbe77
	v_mov_b32_e32 v101, 0x3f8fdf3b
	v_mov_b32_e32 v102, 0x3f900000
	v_mov_b32_e32 v103, 0x3f9020c5
	v_mov_b32_e32 v104, 0x3f904189
	v_mov_b32_e32 v105, 0x3f90a3d7
	v_mov_b32_e32 v106, 0x3f90c49c
	v_mov_b32_e32 v107, 0x3f90e560
	v_mov_b32_e32 v108, 0x3f910625
	v_mov_b32_e32 v109, 0x3f9126e9
	v_mov_b32_e32 v110, 0x3f9147ae
	v_mov_b32_e32 v111, 0x3f916873
	v_mov_b32_e32 v112, 0x3f918937
	v_mov_b32_e32 v113, 0x3f91eb85
	v_mov_b32_e32 v114, 0x3f920c4a
	v_mov_b32_e32 v115, 0x3f922d0e
	v_mov_b32_e32 v116, 0x3f924dd3
	v_mov_b32_e32 v117, 0x3f926e97
	v_mov_b32_e32 v118, 0x3f928f5c
	v_mov_b32_e32 v119, 0x3f92b021
	v_mov_b32_e32 v120, 0x3f92d0e5
	v_mov_b32_e32 v121, 0x3f933333
	v_mov_b32_e32 v122, 0x3f9353f8
	v_mov_b32_e32 v123, 0x3f9374bc
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s1, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB9_17
; %bb.3:                                ; %.lr.ph
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v124, 0x3f939581
	v_mov_b32_e32 v125, 0x3f93b645
	v_mov_b32_e32 v126, 0x3f93d70a
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v127, 0x3f93f7cf :: v_dual_mov_b32 v156, v153
	v_dual_mov_b32 v159, v153 :: v_dual_mov_b32 v160, v153
	v_dual_mov_b32 v154, v153 :: v_dual_mov_b32 v155, v153
	v_dual_mov_b32 v157, v153 :: v_dual_mov_b32 v158, v153
	v_mov_b32_e32 v176, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v128, 0x3f941893 :: v_dual_mov_b32 v175, v159
	v_cmp_lt_u32_e64 s0, 0x7f, v0
	v_mbcnt_lo_u32_b32 v186, -1, 0
	v_dual_mov_b32 v174, v158 :: v_dual_mov_b32 v171, v155
	v_dual_mov_b32 v173, v157 :: v_dual_mov_b32 v172, v156
	v_dual_mov_b32 v169, v153 :: v_dual_mov_b32 v170, v154
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v146, v153
	v_dual_mov_b32 v147, v153 :: v_dual_mov_b32 v148, v153
	v_dual_mov_b32 v149, v153 :: v_dual_mov_b32 v150, v153
	v_dual_mov_b32 v151, v153 :: v_dual_mov_b32 v152, v153
	v_dual_mov_b32 v137, v153 :: v_dual_mov_b32 v138, v153
	v_dual_mov_b32 v139, v153 :: v_dual_mov_b32 v140, v153
	v_dual_mov_b32 v141, v153 :: v_dual_mov_b32 v142, v153
	v_dual_mov_b32 v143, v153 :: v_dual_mov_b32 v144, v153
	v_dual_mov_b32 v129, v153 :: v_dual_mov_b32 v130, v153
	v_dual_mov_b32 v131, v153 :: v_dual_mov_b32 v132, v153
	v_dual_mov_b32 v133, v153 :: v_dual_mov_b32 v134, v153
	v_dual_mov_b32 v135, v153 :: v_dual_mov_b32 v136, v153
	v_dual_mov_b32 v177, v153 :: v_dual_mov_b32 v178, v153
	v_dual_mov_b32 v179, v153 :: v_dual_mov_b32 v180, v153
	v_dual_mov_b32 v181, v153 :: v_dual_mov_b32 v182, v153
	v_dual_mov_b32 v183, v153 :: v_dual_mov_b32 v184, v153
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0x35800000
	s_mov_b32 s2, 0x10101010
	s_mov_b32 s6, 0x18181818
	s_branch .LBB9_5
.LBB9_4:                                ;   in Loop: Header=BB9_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s8, s8, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_dual_mov_b32 v177, v169 :: v_dual_mov_b32 v178, v170
	v_dual_mov_b32 v179, v171 :: v_dual_mov_b32 v180, v172
	v_dual_mov_b32 v181, v173 :: v_dual_mov_b32 v182, v174
	v_dual_mov_b32 v183, v175 :: v_dual_mov_b32 v184, v176
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB9_18
.LBB9_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB9_9 Depth 2
                                        ;     Child Loop BB9_13 Depth 2
	v_xor_b32_e32 v154, 16, v186
	s_and_saveexec_b32 s3, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB9_7
; %bb.6:                                ;   in Loop: Header=BB9_5 Depth=1
	v_max3_num_f32 v155, v161, 0xff800000, v162
	v_cmp_gt_u32_e32 vcc_lo, 32, v154
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_num_f32 v155, v155, v163, v164
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v156, v186, v154, vcc_lo
	v_max3_num_f32 v155, v155, v165, v166
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v169, 2, v156
	v_max3_num_f32 v155, v155, v167, v168
	ds_bpermute_b32 v156, v169, v155
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v156, v156, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v155, v155, v156
	v_sub_f32_e32 v156, v161, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v156, 0x3fb8aa3b, v156 :: v_dual_sub_f32 v157, v162, v155
	v_dual_sub_f32 v160, v164, v155 :: v_dual_sub_f32 v159, v163, v155
	v_exp_f32_e32 v156, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v157, 0x3fb8aa3b, v157 :: v_dual_mul_f32 v160, 0x3fb8aa3b, v160
	v_mul_f32_e32 v159, 0x3fb8aa3b, v159
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v157, v157
	v_exp_f32_e32 v170, v160
	v_sub_f32_e32 v160, v165, v155
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v159, v159
	v_mul_f32_e32 v161, 0x3a83126f, v156
	v_mul_f32_e32 v160, 0x3fb8aa3b, v160
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v158, v156, v157
	v_mul_f32_e32 v162, 0x3b03126f, v157
	v_exp_f32_e32 v165, v160
	v_sub_f32_e32 v160, v166, v155
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v158, v159, v158
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v156, v161, 0, v162
	v_mul_f32_e32 v160, 0x3fb8aa3b, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v166, v160
	v_dual_sub_f32 v160, v167, v155 :: v_dual_sub_f32 v155, v168, v155
	v_dual_add_f32 v158, v170, v158 :: v_dual_mul_f32 v155, 0x3fb8aa3b, v155
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v157, 0x3bc49ba6, v166
	v_mul_f32_e32 v160, 0x3fb8aa3b, v160
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v155, v155
	v_add_f32_e32 v158, v165, v158
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v167, v160
	v_mul_f32_e32 v160, 0x3b449ba6, v159
	v_mul_f32_e32 v159, 0x3b83126f, v170
	v_add_f32_e32 v158, v166, v158
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v156, v156, v160, v159
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v158, v167, v158
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_dual_add_f32 v163, v155, v158 :: v_dual_mul_f32 v158, 0x3ba3d70b, v165
	v_mul_f32_e32 v155, 0x3c03126f, v155
	ds_bpermute_b32 v164, v169, v163
	v_max3_num_f32 v165, v156, v158, v157
	v_mul_f32_e32 v156, 0x3be56042, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v165, v165, v156, v155
	ds_bpermute_b32 v166, v169, v165
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v166, v166, v166
	v_max_num_f32_e32 v165, v165, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v166, null, 0x43e00000, 0x43e00000, v165
	v_rcp_f32_e32 v167, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v168, -v166, v167, 1.0
	v_fmac_f32_e32 v167, v168, v167
	v_div_scale_f32 v168, vcc_lo, v165, 0x43e00000, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v169, v168, v167
	v_fma_f32 v170, -v166, v169, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v169, v170, v167
	v_fma_f32 v166, -v166, v169, v168
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v166, v166, v167, v169
	v_div_fixup_f32 v165, v166, 0x43e00000, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v165, 0x1f800000, v165
	v_div_scale_f32 v166, null, v165, v165, v161
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v167, v166
	v_fma_f32 v168, -v166, v167, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v167, v168, v167
	v_div_scale_f32 v168, vcc_lo, v161, v165, v161
	v_mul_f32_e32 v169, v168, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v170, -v166, v169, v168
	v_fmac_f32_e32 v169, v170, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v166, -v166, v169, v168
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v166, v166, v167, v169
	v_div_scale_f32 v167, null, v165, v165, v162
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v166, v166, v165, v161
	v_rcp_f32_e32 v168, v167
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v169, -v167, v168, 1.0
	v_fmac_f32_e32 v168, v169, v168
	v_div_scale_f32 v169, vcc_lo, v162, v165, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v170, v169, v168
	v_fma_f32 v171, -v167, v170, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v170, v171, v168
	v_fma_f32 v167, -v167, v170, v169
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v167, v167, v168, v170
	v_mov_b16_e64 v168.l, v153.l
	v_mov_b16_e64 v168.h, 0
	v_div_fixup_f32 v167, v167, v165, v162
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v169.l, v168.l
	v_mov_b16_e64 v169.h, v168.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v169.l, v166, v167
	v_div_scale_f32 v166, null, v165, v165, v160
	v_rcp_f32_e32 v167, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v170, -v166, v167, 1.0
	v_fmac_f32_e32 v167, v170, v167
	v_div_scale_f32 v170, vcc_lo, v160, v165, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v171, v170, v167
	v_fma_f32 v172, -v166, v171, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v171, v172, v167
	v_fma_f32 v166, -v166, v171, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v166, v166, v167, v171
	v_div_scale_f32 v167, null, v165, v165, v159
	v_div_fixup_f32 v166, v166, v165, v160
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v170, v167
	v_fma_f32 v171, -v167, v170, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v170, v171, v170
	v_div_scale_f32 v171, vcc_lo, v159, v165, v159
	v_mul_f32_e32 v172, v171, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v167, v172, v171
	v_fmac_f32_e32 v172, v173, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v167, v172, v171
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v167, v167, v170, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v167, v167, v165, v159
	v_cvt_pk_fp8_f32 v169.h, v166, v167
	v_div_scale_f32 v166, null, v165, v165, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v167, v166
	v_fma_f32 v170, -v166, v167, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v167, v170, v167
	v_div_scale_f32 v170, vcc_lo, v158, v165, v158
	v_mul_f32_e32 v171, v170, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v172, -v166, v171, v170
	v_fmac_f32_e32 v171, v172, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v166, -v166, v171, v170
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v166, v166, v167, v171
	v_div_scale_f32 v167, null, v165, v165, v157
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v166, v166, v165, v158
	v_rcp_f32_e32 v170, v167
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v167, v170, 1.0
	v_fmac_f32_e32 v170, v171, v170
	v_div_scale_f32 v171, vcc_lo, v157, v165, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v172, v171, v170
	v_fma_f32 v173, -v167, v172, v171
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v172, v173, v170
	v_fma_f32 v167, -v167, v172, v171
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v167, v167, v170, v172
	v_div_fixup_f32 v167, v167, v165, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v168.l, v166, v167
	v_div_scale_f32 v166, null, v165, v165, v156
	v_rcp_f32_e32 v167, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v170, -v166, v167, 1.0
	v_fmac_f32_e32 v167, v170, v167
	v_div_scale_f32 v170, vcc_lo, v156, v165, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v171, v170, v167
	v_fma_f32 v172, -v166, v171, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v171, v172, v167
	v_fma_f32 v166, -v166, v171, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v166, v166, v167, v171
	v_div_scale_f32 v167, null, v165, v165, v155
	v_div_fixup_f32 v166, v166, v165, v156
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v170, v167
	v_fma_f32 v171, -v167, v170, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v170, v171, v170
	v_div_scale_f32 v171, vcc_lo, v155, v165, v155
	v_mul_f32_e32 v172, v171, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v167, v172, v171
	v_fmac_f32_e32 v172, v173, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v167, v172, v171
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v167, v167, v170, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v165, v167, v165, v155
	v_cvt_pk_fp8_f32 v168.h, v166, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v165, v169, v168
	v_add_f32_e32 v168, v163, v164
                                        ; implicit-def: $vgpr169_vgpr170_vgpr171_vgpr172_vgpr173_vgpr174_vgpr175_vgpr176
	v_cvt_f32_ubyte0_e32 v165, v165
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v163, null, v168, v168, v161
	v_fmaak_f32 v165, s9, v165, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v164, v163
	v_dual_mul_f32 v8, v8, v165 :: v_dual_mul_f32 v3, v3, v165
	v_dual_mul_f32 v7, v7, v165 :: v_dual_mul_f32 v6, v6, v165
	v_mul_f32_e32 v15, v15, v165
	v_dual_mul_f32 v5, v5, v165 :: v_dual_mul_f32 v4, v4, v165
	v_dual_mul_f32 v11, v11, v165 :: v_dual_mul_f32 v2, v2, v165
	v_mul_f32_e32 v9, v9, v165
	v_dual_mul_f32 v1, v1, v165 :: v_dual_mul_f32 v16, v16, v165
	v_dual_mul_f32 v23, v23, v165 :: v_dual_mul_f32 v14, v14, v165
	v_mul_f32_e32 v21, v21, v165
	v_dual_mul_f32 v13, v13, v165 :: v_dual_mul_f32 v12, v12, v165
	v_dual_mul_f32 v19, v19, v165 :: v_dual_mul_f32 v10, v10, v165
	v_dual_mul_f32 v17, v17, v165 :: v_dual_mul_f32 v24, v24, v165
	v_dual_mul_f32 v31, v31, v165 :: v_dual_mul_f32 v22, v22, v165
	v_dual_mul_f32 v29, v29, v165 :: v_dual_mul_f32 v20, v20, v165
	v_dual_mul_f32 v27, v27, v165 :: v_dual_mul_f32 v18, v18, v165
	v_dual_mul_f32 v25, v25, v165 :: v_dual_mul_f32 v32, v32, v165
	v_dual_mul_f32 v39, v39, v165 :: v_dual_mul_f32 v30, v30, v165
	v_dual_mul_f32 v37, v37, v165 :: v_dual_mul_f32 v28, v28, v165
	v_dual_mul_f32 v35, v35, v165 :: v_dual_mul_f32 v26, v26, v165
	v_dual_mul_f32 v33, v33, v165 :: v_dual_mul_f32 v40, v40, v165
	v_dual_mul_f32 v47, v47, v165 :: v_dual_mul_f32 v38, v38, v165
	v_dual_mul_f32 v45, v45, v165 :: v_dual_mul_f32 v36, v36, v165
	v_dual_mul_f32 v43, v43, v165 :: v_dual_mul_f32 v34, v34, v165
	v_dual_mul_f32 v41, v41, v165 :: v_dual_mul_f32 v48, v48, v165
	v_dual_mul_f32 v55, v55, v165 :: v_dual_mul_f32 v46, v46, v165
	v_dual_mul_f32 v53, v53, v165 :: v_dual_mul_f32 v44, v44, v165
	v_dual_mul_f32 v51, v51, v165 :: v_dual_mul_f32 v42, v42, v165
	v_dual_mul_f32 v49, v49, v165 :: v_dual_mul_f32 v56, v56, v165
	v_dual_mul_f32 v63, v63, v165 :: v_dual_mul_f32 v54, v54, v165
	v_dual_mul_f32 v61, v61, v165 :: v_dual_mul_f32 v52, v52, v165
	v_dual_mul_f32 v59, v59, v165 :: v_dual_mul_f32 v50, v50, v165
	v_dual_mul_f32 v57, v57, v165 :: v_dual_mul_f32 v64, v64, v165
	v_dual_mul_f32 v71, v71, v165 :: v_dual_mul_f32 v62, v62, v165
	v_dual_mul_f32 v69, v69, v165 :: v_dual_mul_f32 v60, v60, v165
	v_dual_mul_f32 v67, v67, v165 :: v_dual_mul_f32 v58, v58, v165
	v_dual_mul_f32 v65, v65, v165 :: v_dual_mul_f32 v72, v72, v165
	v_dual_mul_f32 v79, v79, v165 :: v_dual_mul_f32 v70, v70, v165
	v_dual_mul_f32 v77, v77, v165 :: v_dual_mul_f32 v68, v68, v165
	v_dual_mul_f32 v75, v75, v165 :: v_dual_mul_f32 v66, v66, v165
	v_dual_mul_f32 v73, v73, v165 :: v_dual_mul_f32 v80, v80, v165
	v_dual_mul_f32 v87, v87, v165 :: v_dual_mul_f32 v78, v78, v165
	v_dual_mul_f32 v85, v85, v165 :: v_dual_mul_f32 v76, v76, v165
	v_dual_mul_f32 v83, v83, v165 :: v_dual_mul_f32 v74, v74, v165
	v_dual_mul_f32 v81, v81, v165 :: v_dual_mul_f32 v88, v88, v165
	v_dual_mul_f32 v95, v95, v165 :: v_dual_mul_f32 v86, v86, v165
	v_dual_mul_f32 v93, v93, v165 :: v_dual_mul_f32 v84, v84, v165
	v_dual_mul_f32 v91, v91, v165 :: v_dual_mul_f32 v82, v82, v165
	v_dual_mul_f32 v89, v89, v165 :: v_dual_mul_f32 v96, v96, v165
	v_dual_mul_f32 v103, v165, v103 :: v_dual_mul_f32 v94, v94, v165
	v_dual_mul_f32 v99, v165, v99 :: v_dual_mul_f32 v92, v92, v165
	v_dual_mul_f32 v111, v165, v111 :: v_dual_mul_f32 v90, v90, v165
	v_dual_mul_f32 v107, v165, v107 :: v_dual_mul_f32 v104, v165, v104
	v_dual_mul_f32 v101, v165, v101 :: v_dual_mul_f32 v102, v165, v102
	v_dual_mul_f32 v97, v165, v97 :: v_dual_mul_f32 v100, v165, v100
	v_dual_mul_f32 v109, v165, v109 :: v_dual_mul_f32 v98, v165, v98
	v_dual_mul_f32 v105, v165, v105 :: v_dual_mul_f32 v112, v165, v112
	v_dual_mul_f32 v119, v165, v119 :: v_dual_mul_f32 v110, v165, v110
	v_dual_mul_f32 v117, v165, v117 :: v_dual_mul_f32 v108, v165, v108
	v_dual_mul_f32 v115, v165, v115 :: v_dual_mul_f32 v106, v165, v106
	v_dual_mul_f32 v113, v165, v113 :: v_dual_mul_f32 v120, v165, v120
	v_dual_mul_f32 v127, v165, v127 :: v_dual_mul_f32 v118, v165, v118
	v_dual_mul_f32 v125, v165, v125 :: v_dual_mul_f32 v116, v165, v116
	v_dual_mul_f32 v123, v165, v123 :: v_dual_mul_f32 v114, v165, v114
	v_dual_mul_f32 v121, v165, v121 :: v_dual_mul_f32 v128, v165, v128
	v_mul_f32_e32 v126, v165, v126
	v_mul_f32_e32 v124, v165, v124
	v_mul_f32_e32 v122, v165, v122
	v_fma_f32 v165, -v163, v164, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v164, v165, v164
	v_div_scale_f32 v165, vcc_lo, v161, v168, v161
	v_mul_f32_e32 v166, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v163, v166, v165
	v_fmac_f32_e32 v166, v167, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v163, -v163, v166, v165
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v163, v163, v164, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v161, v163, v168, v161
	v_div_scale_f32 v163, null, v168, v168, v162
	v_add_f32_e32 v161, 0, v161
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v164, v163
	v_fma_f32 v165, -v163, v164, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v164, v165, v164
	v_div_scale_f32 v165, vcc_lo, v162, v168, v162
	v_mul_f32_e32 v166, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v163, v166, v165
	v_fmac_f32_e32 v166, v167, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v163, -v163, v166, v165
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v163, v163, v164, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v162, v163, v168, v162
	v_div_scale_f32 v163, null, v168, v168, v160
	v_add_f32_e32 v162, 0x3c23d70a, v162
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v164, v163
	v_fma_f32 v165, -v163, v164, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v164, v165, v164
	v_div_scale_f32 v165, vcc_lo, v160, v168, v160
	v_mul_f32_e32 v166, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v163, v166, v165
	v_fmac_f32_e32 v166, v167, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v163, -v163, v166, v165
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v163, v163, v164, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v160, v163, v168, v160
	v_add_f32_e32 v163, 0x3ca3d70a, v160
	v_div_scale_f32 v160, null, v168, v168, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v164, v160
	v_fma_f32 v165, -v160, v164, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v164, v165, v164
	v_div_scale_f32 v165, vcc_lo, v159, v168, v159
	v_mul_f32_e32 v166, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v160, v166, v165
	v_fmac_f32_e32 v166, v167, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v160, v166, v165
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v160, v160, v164, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v159, v160, v168, v159
	v_add_f32_e32 v164, 0x3cf5c28f, v159
	v_div_scale_f32 v159, null, v168, v168, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v160, v159
	v_fma_f32 v165, -v159, v160, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v160, v165, v160
	v_div_scale_f32 v165, vcc_lo, v158, v168, v158
	v_mul_f32_e32 v166, v165, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v159, v166, v165
	v_fmac_f32_e32 v166, v167, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v159, v166, v165
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v159, v159, v160, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v158, v159, v168, v158
	v_add_f32_e32 v165, 0x3d23d70a, v158
	v_div_scale_f32 v158, null, v168, v168, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v159, v158
	v_fma_f32 v160, -v158, v159, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v159
	v_div_scale_f32 v160, vcc_lo, v157, v168, v157
	v_mul_f32_e32 v166, v160, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v158, v166, v160
	v_fmac_f32_e32 v166, v167, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v158, v166, v160
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v158, v158, v159, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v157, v158, v168, v157
	v_add_f32_e32 v166, 0x3d4ccccc, v157
	v_div_scale_f32 v157, null, v168, v168, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v158, v157
	v_fma_f32 v159, -v157, v158, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v158, v159, v158
	v_div_scale_f32 v159, vcc_lo, v156, v168, v156
	v_mul_f32_e32 v160, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v157, v160, v159
	v_fmac_f32_e32 v160, v167, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v157, v160, v159
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v157, v157, v158, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v156, v157, v168, v156
	v_add_f32_e32 v167, 0x3d75c28f, v156
	v_div_scale_f32 v156, null, v168, v168, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v157, v156
	v_fma_f32 v158, -v156, v157, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v155, v168, v155
	v_mul_f32_e32 v159, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v156, v159, v158
	v_fmac_f32_e32 v159, v160, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v156, -v156, v159, v158
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v156, v156, v157, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v155, v156, v168, v155
	v_add_f32_e32 v168, 0x3d8f5c29, v155
.LBB9_7:                                ; %Flow1042
                                        ;   in Loop: Header=BB9_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s3
	s_cbranch_execz .LBB9_11
; %bb.8:                                ;   in Loop: Header=BB9_5 Depth=1
	s_mov_b32 s11, 32
.LBB9_9:                                ; %.preheader.i
                                        ;   Parent Loop BB9_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_mov_b32 s3, s2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v156, s7 :: v_dual_mov_b32 v155, s6
	v_dual_mov_b32 v158, s3 :: v_dual_mov_b32 v157, s2
	s_add_co_i32 s11, s11, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_cmp_lg_u32 s11, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[155:156], v[157:158], v[145:152]
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[155:156], v[157:158], v[137:144]
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[155:156], v[157:158], v[129:136]
	v_wmma_f32_16x16x16_fp8_fp8 v[169:176], v[155:156], v[157:158], v[169:176]
	s_cbranch_scc1 .LBB9_9
; %bb.10:                               ; %_Z6matrixILi4EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB9_5 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v177, v169 :: v_dual_mov_b32 v178, v170
	v_dual_mov_b32 v179, v171 :: v_dual_mov_b32 v180, v172
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v181, v173 :: v_dual_mov_b32 v182, v174
	v_dual_mov_b32 v183, v175 :: v_dual_mov_b32 v184, v176
.LBB9_11:                               ; %Flow1043
                                        ;   in Loop: Header=BB9_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v169, v177 :: v_dual_mov_b32 v170, v178
	v_dual_mov_b32 v171, v179 :: v_dual_mov_b32 v172, v180
	v_dual_mov_b32 v173, v181 :: v_dual_mov_b32 v174, v182
	v_dual_mov_b32 v175, v183 :: v_dual_mov_b32 v176, v184
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s3, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s3
	s_cbranch_execz .LBB9_15
; %bb.12:                               ; %.preheader.i78.preheader
                                        ;   in Loop: Header=BB9_5 Depth=1
	s_mov_b32 s11, 32
.LBB9_13:                               ; %.preheader.i78
                                        ;   Parent Loop BB9_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_mov_b32 s3, s2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v155, s7 :: v_dual_mov_b32 v154, s6
	v_dual_mov_b32 v157, s3 :: v_dual_mov_b32 v156, s2
	s_add_co_i32 s11, s11, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_cmp_lg_u32 s11, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[154:155], v[156:157], v[145:152]
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[154:155], v[156:157], v[137:144]
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[154:155], v[156:157], v[129:136]
	v_wmma_f32_16x16x16_fp8_fp8 v[169:176], v[154:155], v[156:157], v[169:176]
	s_cbranch_scc1 .LBB9_13
; %bb.14:                               ; %_Z6matrixILi4EEvRAT__Dv8_fDv2_iS3_.exit81
                                        ;   in Loop: Header=BB9_5 Depth=1
                                        ; implicit-def: $vgpr154
.LBB9_15:                               ; %Flow
                                        ;   in Loop: Header=BB9_5 Depth=1
	s_and_not1_saveexec_b32 s3, s10
	s_cbranch_execz .LBB9_4
; %bb.16:                               ;   in Loop: Header=BB9_5 Depth=1
	v_max3_num_f32 v155, v161, 0xff800000, v162
	v_cmp_gt_u32_e32 vcc_lo, 32, v154
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_num_f32 v155, v155, v163, v164
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v154, v186, v154, vcc_lo
	v_max3_num_f32 v155, v155, v165, v166
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v177, 2, v154
	v_max3_num_f32 v155, v155, v167, v168
	ds_bpermute_b32 v154, v177, v155
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v154, v154, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v154, v155, v154
	v_dual_sub_f32 v156, v162, v154 :: v_dual_sub_f32 v155, v161, v154
	v_sub_f32_e32 v158, v163, v154
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v156, 0x3fb8aa3b, v156 :: v_dual_mul_f32 v155, 0x3fb8aa3b, v155
	v_mul_f32_e32 v158, 0x3fb8aa3b, v158
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v156, v156
	v_exp_f32_e32 v155, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v158, v158
	v_dual_mul_f32 v160, 0x3b03126f, v156 :: v_dual_add_f32 v157, v155, v156
	v_mul_f32_e32 v161, 0x3a83126f, v155
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v157, v158, v157
	v_sub_f32_e32 v159, v164, v154
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v155, v161, 0, v160
	v_mul_f32_e32 v159, 0x3fb8aa3b, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v164, v159
	v_sub_f32_e32 v159, v165, v154
	v_mul_f32_e32 v159, 0x3fb8aa3b, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v165, v159
	v_sub_f32_e32 v159, v166, v154
	v_mul_f32_e32 v159, 0x3fb8aa3b, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v166, v159
	v_dual_sub_f32 v159, v167, v154 :: v_dual_sub_f32 v154, v168, v154
	v_add_f32_e32 v157, v164, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v159, 0x3fb8aa3b, v159 :: v_dual_mul_f32 v154, 0x3fb8aa3b, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v157, v165, v157 :: v_dual_mul_f32 v156, 0x3bc49ba6, v166
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v167, v159
	v_exp_f32_e32 v154, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v157, v166, v157
	v_mul_f32_e32 v159, 0x3b449ba6, v158
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v158, 0x3b83126f, v164 :: v_dual_add_f32 v157, v167, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v155, v155, v159, v158
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v162, v154, v157
	v_mul_f32_e32 v157, 0x3ba3d70b, v165
	v_mul_f32_e32 v154, 0x3c03126f, v154
	ds_bpermute_b32 v163, v177, v162
	v_max3_num_f32 v164, v155, v157, v156
	v_mul_f32_e32 v155, 0x3be56042, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v164, v164, v155, v154
	ds_bpermute_b32 v165, v177, v164
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v165, v165, v165
	v_max_num_f32_e32 v164, v164, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v165, null, 0x43e00000, 0x43e00000, v164
	v_rcp_f32_e32 v166, v165
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v167, -v165, v166, 1.0
	v_fmac_f32_e32 v166, v167, v166
	v_div_scale_f32 v167, vcc_lo, v164, 0x43e00000, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v168, v167, v166
	v_fma_f32 v177, -v165, v168, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v168, v177, v166
	v_fma_f32 v165, -v165, v168, v167
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v165, v165, v166, v168
	v_div_fixup_f32 v164, v165, 0x43e00000, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v164, 0x1f800000, v164
	v_div_scale_f32 v165, null, v164, v164, v161
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v166, v165
	v_fma_f32 v167, -v165, v166, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v166, v167, v166
	v_div_scale_f32 v167, vcc_lo, v161, v164, v161
	v_mul_f32_e32 v168, v167, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v165, v168, v167
	v_fmac_f32_e32 v168, v177, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v165, -v165, v168, v167
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v165, v165, v166, v168
	v_div_scale_f32 v166, null, v164, v164, v160
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v165, v165, v164, v161
	v_rcp_f32_e32 v167, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v168, -v166, v167, 1.0
	v_fmac_f32_e32 v167, v168, v167
	v_div_scale_f32 v168, vcc_lo, v160, v164, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v177, v168, v167
	v_fma_f32 v178, -v166, v177, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v177, v178, v167
	v_fma_f32 v166, -v166, v177, v168
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v166, v166, v167, v177
	v_mov_b16_e64 v167.l, v153.l
	v_mov_b16_e64 v167.h, 0
	v_div_fixup_f32 v166, v166, v164, v160
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v168.l, v167.l
	v_mov_b16_e64 v168.h, v167.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v168.l, v165, v166
	v_div_scale_f32 v165, null, v164, v164, v159
	v_rcp_f32_e32 v166, v165
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v165, v166, 1.0
	v_fmac_f32_e32 v166, v177, v166
	v_div_scale_f32 v177, vcc_lo, v159, v164, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v178, v177, v166
	v_fma_f32 v179, -v165, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v166
	v_fma_f32 v165, -v165, v178, v177
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v165, v165, v166, v178
	v_div_scale_f32 v166, null, v164, v164, v158
	v_div_fixup_f32 v165, v165, v164, v159
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v177, v166
	v_fma_f32 v178, -v166, v177, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v158, v164, v158
	v_mul_f32_e32 v179, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v166, v179, v178
	v_fmac_f32_e32 v179, v180, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v166, -v166, v179, v178
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v166, v166, v177, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v166, v166, v164, v158
	v_cvt_pk_fp8_f32 v168.h, v165, v166
	v_div_scale_f32 v165, null, v164, v164, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v166, v165
	v_fma_f32 v177, -v165, v166, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v166, v177, v166
	v_div_scale_f32 v177, vcc_lo, v157, v164, v157
	v_mul_f32_e32 v178, v177, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v165, v178, v177
	v_fmac_f32_e32 v178, v179, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v165, -v165, v178, v177
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v165, v165, v166, v178
	v_div_scale_f32 v166, null, v164, v164, v156
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v165, v165, v164, v157
	v_rcp_f32_e32 v177, v166
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v178, -v166, v177, 1.0
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v156, v164, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v179, v178, v177
	v_fma_f32 v180, -v166, v179, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v177
	v_fma_f32 v166, -v166, v179, v178
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v166, v166, v177, v179
	v_div_fixup_f32 v166, v166, v164, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v167.l, v165, v166
	v_div_scale_f32 v165, null, v164, v164, v155
	v_rcp_f32_e32 v166, v165
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v165, v166, 1.0
	v_fmac_f32_e32 v166, v177, v166
	v_div_scale_f32 v177, vcc_lo, v155, v164, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v178, v177, v166
	v_fma_f32 v179, -v165, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v166
	v_fma_f32 v165, -v165, v178, v177
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v165, v165, v166, v178
	v_div_scale_f32 v166, null, v164, v164, v154
	v_div_fixup_f32 v165, v165, v164, v155
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v177, v166
	v_fma_f32 v178, -v166, v177, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v154, v164, v154
	v_mul_f32_e32 v179, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v166, v179, v178
	v_fmac_f32_e32 v179, v180, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v166, -v166, v179, v178
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v166, v166, v177, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v164, v166, v164, v154
	v_cvt_pk_fp8_f32 v167.h, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v164, v168, v167
	v_add_f32_e32 v168, v162, v163
	v_cvt_f32_ubyte0_e32 v164, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v162, null, v168, v168, v161
	v_fmaak_f32 v164, s9, v164, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v163, v162
	v_dual_mul_f32 v123, v123, v164 :: v_dual_mul_f32 v8, v8, v164
	v_dual_mul_f32 v7, v7, v164 :: v_dual_mul_f32 v6, v6, v164
	v_dual_mul_f32 v5, v5, v164 :: v_dual_mul_f32 v2, v2, v164
	v_dual_mul_f32 v4, v4, v164 :: v_dual_mul_f32 v3, v3, v164
	v_dual_mul_f32 v16, v16, v164 :: v_dual_mul_f32 v1, v1, v164
	v_dual_mul_f32 v14, v14, v164 :: v_dual_mul_f32 v15, v15, v164
	v_dual_mul_f32 v12, v12, v164 :: v_dual_mul_f32 v13, v13, v164
	v_dual_mul_f32 v10, v10, v164 :: v_dual_mul_f32 v11, v11, v164
	v_dual_mul_f32 v24, v24, v164 :: v_dual_mul_f32 v9, v9, v164
	v_dual_mul_f32 v22, v22, v164 :: v_dual_mul_f32 v23, v23, v164
	v_dual_mul_f32 v20, v20, v164 :: v_dual_mul_f32 v21, v21, v164
	v_dual_mul_f32 v18, v18, v164 :: v_dual_mul_f32 v19, v19, v164
	v_dual_mul_f32 v32, v32, v164 :: v_dual_mul_f32 v17, v17, v164
	v_dual_mul_f32 v30, v30, v164 :: v_dual_mul_f32 v31, v31, v164
	v_dual_mul_f32 v28, v28, v164 :: v_dual_mul_f32 v29, v29, v164
	v_dual_mul_f32 v26, v26, v164 :: v_dual_mul_f32 v27, v27, v164
	v_dual_mul_f32 v40, v40, v164 :: v_dual_mul_f32 v25, v25, v164
	v_dual_mul_f32 v38, v38, v164 :: v_dual_mul_f32 v39, v39, v164
	v_dual_mul_f32 v36, v36, v164 :: v_dual_mul_f32 v37, v37, v164
	v_dual_mul_f32 v34, v34, v164 :: v_dual_mul_f32 v35, v35, v164
	v_dual_mul_f32 v48, v48, v164 :: v_dual_mul_f32 v33, v33, v164
	v_dual_mul_f32 v46, v46, v164 :: v_dual_mul_f32 v47, v47, v164
	v_dual_mul_f32 v44, v44, v164 :: v_dual_mul_f32 v45, v45, v164
	v_dual_mul_f32 v42, v42, v164 :: v_dual_mul_f32 v43, v43, v164
	v_dual_mul_f32 v56, v56, v164 :: v_dual_mul_f32 v41, v41, v164
	v_dual_mul_f32 v54, v54, v164 :: v_dual_mul_f32 v55, v55, v164
	v_dual_mul_f32 v52, v52, v164 :: v_dual_mul_f32 v53, v53, v164
	v_dual_mul_f32 v50, v50, v164 :: v_dual_mul_f32 v51, v51, v164
	v_dual_mul_f32 v64, v64, v164 :: v_dual_mul_f32 v49, v49, v164
	v_dual_mul_f32 v62, v62, v164 :: v_dual_mul_f32 v63, v63, v164
	v_dual_mul_f32 v60, v60, v164 :: v_dual_mul_f32 v61, v61, v164
	v_dual_mul_f32 v58, v58, v164 :: v_dual_mul_f32 v59, v59, v164
	v_dual_mul_f32 v72, v72, v164 :: v_dual_mul_f32 v57, v57, v164
	v_dual_mul_f32 v70, v70, v164 :: v_dual_mul_f32 v71, v71, v164
	v_dual_mul_f32 v68, v68, v164 :: v_dual_mul_f32 v69, v69, v164
	v_dual_mul_f32 v66, v66, v164 :: v_dual_mul_f32 v67, v67, v164
	v_dual_mul_f32 v80, v80, v164 :: v_dual_mul_f32 v65, v65, v164
	v_dual_mul_f32 v78, v78, v164 :: v_dual_mul_f32 v79, v79, v164
	v_dual_mul_f32 v76, v76, v164 :: v_dual_mul_f32 v77, v77, v164
	v_dual_mul_f32 v74, v74, v164 :: v_dual_mul_f32 v75, v75, v164
	v_dual_mul_f32 v88, v88, v164 :: v_dual_mul_f32 v73, v73, v164
	v_dual_mul_f32 v86, v86, v164 :: v_dual_mul_f32 v87, v87, v164
	v_dual_mul_f32 v84, v84, v164 :: v_dual_mul_f32 v85, v85, v164
	v_dual_mul_f32 v82, v82, v164 :: v_dual_mul_f32 v83, v83, v164
	v_dual_mul_f32 v96, v96, v164 :: v_dual_mul_f32 v81, v81, v164
	v_dual_mul_f32 v94, v94, v164 :: v_dual_mul_f32 v95, v95, v164
	v_dual_mul_f32 v92, v92, v164 :: v_dual_mul_f32 v93, v93, v164
	v_dual_mul_f32 v90, v90, v164 :: v_dual_mul_f32 v91, v91, v164
	v_dual_mul_f32 v102, v164, v102 :: v_dual_mul_f32 v89, v89, v164
	v_mul_f32_e32 v98, v164, v98
	v_dual_mul_f32 v104, v164, v104 :: v_dual_mul_f32 v103, v164, v103
	v_dual_mul_f32 v101, v164, v101 :: v_dual_mul_f32 v100, v164, v100
	v_dual_mul_f32 v99, v164, v99 :: v_dual_mul_f32 v112, v164, v112
	v_dual_mul_f32 v97, v164, v97 :: v_dual_mul_f32 v110, v164, v110
	v_dual_mul_f32 v111, v164, v111 :: v_dual_mul_f32 v108, v164, v108
	v_dual_mul_f32 v109, v164, v109 :: v_dual_mul_f32 v106, v164, v106
	v_dual_mul_f32 v107, v164, v107 :: v_dual_mul_f32 v120, v164, v120
	v_dual_mul_f32 v105, v164, v105 :: v_dual_mul_f32 v118, v164, v118
	v_dual_mul_f32 v119, v164, v119 :: v_dual_mul_f32 v116, v164, v116
	v_dual_mul_f32 v117, v164, v117 :: v_dual_mul_f32 v114, v164, v114
	v_dual_mul_f32 v115, v164, v115 :: v_dual_mul_f32 v126, v126, v164
	v_dual_mul_f32 v113, v164, v113 :: v_dual_mul_f32 v122, v122, v164
	v_dual_mul_f32 v128, v128, v164 :: v_dual_mul_f32 v127, v127, v164
	v_dual_mul_f32 v125, v125, v164 :: v_dual_mul_f32 v124, v124, v164
	v_mul_f32_e32 v121, v121, v164
	v_fma_f32 v164, -v162, v163, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v163, v164, v163
	v_div_scale_f32 v164, vcc_lo, v161, v168, v161
	v_mul_f32_e32 v165, v164, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v166, -v162, v165, v164
	v_fmac_f32_e32 v165, v166, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v162, -v162, v165, v164
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v162, v162, v163, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v161, v162, v168, v161
	v_div_scale_f32 v162, null, v168, v168, v160
	v_rcp_f32_e32 v163, v162
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v164, -v162, v163, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_div_scale_f32 v164, vcc_lo, v160, v168, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v165, v164, v163
	v_fma_f32 v166, -v162, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v165, v166, v163
	v_fma_f32 v162, -v162, v165, v164
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v162, v162, v163, v165
	v_div_fixup_f32 v160, v162, v168, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v162, 0x3c23d70a, v160
	v_div_scale_f32 v160, null, v168, v168, v159
	v_rcp_f32_e32 v163, v160
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v164, -v160, v163, 1.0
	v_fmac_f32_e32 v163, v164, v163
	v_div_scale_f32 v164, vcc_lo, v159, v168, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v165, v164, v163
	v_fma_f32 v166, -v160, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v165, v166, v163
	v_fma_f32 v160, -v160, v165, v164
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v160, v160, v163, v165
	v_div_fixup_f32 v159, v160, v168, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v163, 0x3ca3d70a, v159
	v_div_scale_f32 v159, null, v168, v168, v158
	v_rcp_f32_e32 v160, v159
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v164, -v159, v160, 1.0
	v_fmac_f32_e32 v160, v164, v160
	v_div_scale_f32 v164, vcc_lo, v158, v168, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v165, v164, v160
	v_fma_f32 v166, -v159, v165, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v165, v166, v160
	v_fma_f32 v159, -v159, v165, v164
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v159, v159, v160, v165
	v_div_fixup_f32 v158, v159, v168, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v164, 0x3cf5c28f, v158
	v_div_scale_f32 v158, null, v168, v168, v157
	v_rcp_f32_e32 v159, v158
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v160, -v158, v159, 1.0
	v_fmac_f32_e32 v159, v160, v159
	v_div_scale_f32 v160, vcc_lo, v157, v168, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v165, v160, v159
	v_fma_f32 v166, -v158, v165, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v165, v166, v159
	v_fma_f32 v158, -v158, v165, v160
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v158, v158, v159, v165
	v_div_fixup_f32 v157, v158, v168, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v165, 0x3d23d70a, v157
	v_div_scale_f32 v157, null, v168, v168, v156
	v_rcp_f32_e32 v158, v157
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v159, -v157, v158, 1.0
	v_fmac_f32_e32 v158, v159, v158
	v_div_scale_f32 v159, vcc_lo, v156, v168, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v161, 0, v161 :: v_dual_mul_f32 v160, v159, v158
	v_fma_f32 v166, -v157, v160, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v160, v166, v158
	v_fma_f32 v157, -v157, v160, v159
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v157, v157, v158, v160
	v_div_fixup_f32 v156, v157, v168, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v166, 0x3d4ccccc, v156
	v_div_scale_f32 v156, null, v168, v168, v155
	v_rcp_f32_e32 v157, v156
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v158, -v156, v157, 1.0
	v_fmac_f32_e32 v157, v158, v157
	v_div_scale_f32 v158, vcc_lo, v155, v168, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v159, v158, v157
	v_fma_f32 v160, -v156, v159, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v159, v160, v157
	v_fma_f32 v156, -v156, v159, v158
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v156, v156, v157, v159
	v_div_fixup_f32 v155, v156, v168, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v167, 0x3d75c28f, v155
	v_div_scale_f32 v155, null, v168, v168, v154
	v_rcp_f32_e32 v156, v155
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v157, -v155, v156, 1.0
	v_fmac_f32_e32 v156, v157, v156
	v_div_scale_f32 v157, vcc_lo, v154, v168, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v158, v157, v156
	v_fma_f32 v159, -v155, v158, v157
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v158, v159, v156
	v_fma_f32 v155, -v155, v158, v157
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v155, v155, v156, v158
	v_div_fixup_f32 v154, v155, v168, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v168, 0x3d8f5c29, v154
	s_branch .LBB9_4
.LBB9_17:
	v_dual_mov_b32 v124, 0x3f939581 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v125, 0x3f93b645 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v126, 0x3f93d70a :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v127, 0x3f93f7cf :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v152, 0
	v_mov_b32_e32 v128, 0x3f941893
.LBB9_18:                               ; %Flow1044
	v_add_f32_e32 v145, 0, v145
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v146
	v_add_f32_e32 v145, v145, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v148
	v_add_f32_e32 v145, v145, v149
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v150
	v_add_f32_e32 v145, v145, v151
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v145, v145, v152
	v_add_f32_e32 v137, v145, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v138
	v_add_f32_e32 v137, v137, v139
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v140
	v_add_f32_e32 v137, v137, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v142
	v_add_f32_e32 v137, v137, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v137, v137, v144
	v_add_f32_e32 v129, v137, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v130
	v_add_f32_e32 v129, v129, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v132
	v_add_f32_e32 v129, v129, v133
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v134
	v_add_f32_e32 v129, v129, v135
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v136
	v_add_f32_e32 v129, v129, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v170
	v_add_f32_e32 v129, v129, v171
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v172
	v_add_f32_e32 v129, v129, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v174
	v_add_f32_e32 v129, v129, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v176
	v_add_f32_e32 v1, v129, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v185
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v9
	v_add_f32_e32 v1, v1, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v11
	v_add_f32_e32 v1, v1, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	v_add_f32_e32 v1, v1, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v15
	v_add_f32_e32 v1, v1, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v17
	v_add_f32_e32 v1, v1, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v19
	v_add_f32_e32 v1, v1, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v21
	v_add_f32_e32 v1, v1, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v23
	v_add_f32_e32 v1, v1, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v25
	v_add_f32_e32 v1, v1, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v27
	v_add_f32_e32 v1, v1, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v29
	v_add_f32_e32 v1, v1, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v31
	v_add_f32_e32 v1, v1, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v33
	v_add_f32_e32 v1, v1, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v35
	v_add_f32_e32 v1, v1, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v37
	v_add_f32_e32 v1, v1, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v39
	v_add_f32_e32 v1, v1, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v41
	v_add_f32_e32 v1, v1, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v43
	v_add_f32_e32 v1, v1, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v45
	v_add_f32_e32 v1, v1, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v47
	v_add_f32_e32 v1, v1, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v49
	v_add_f32_e32 v1, v1, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v51
	v_add_f32_e32 v1, v1, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v53
	v_add_f32_e32 v1, v1, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v55
	v_add_f32_e32 v1, v1, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v57
	v_add_f32_e32 v1, v1, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v59
	v_add_f32_e32 v1, v1, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v61
	v_add_f32_e32 v1, v1, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v63
	v_add_f32_e32 v1, v1, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v65
	v_add_f32_e32 v1, v1, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v67
	v_add_f32_e32 v1, v1, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v69
	v_add_f32_e32 v1, v1, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v71
	v_add_f32_e32 v1, v1, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v73
	v_add_f32_e32 v1, v1, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v75
	v_add_f32_e32 v1, v1, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v77
	v_add_f32_e32 v1, v1, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v79
	v_add_f32_e32 v1, v1, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v81
	v_add_f32_e32 v1, v1, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v83
	v_add_f32_e32 v1, v1, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v85
	v_add_f32_e32 v1, v1, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v87
	v_add_f32_e32 v1, v1, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v89
	v_add_f32_e32 v1, v1, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v91
	v_add_f32_e32 v1, v1, v92
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v93
	v_add_f32_e32 v1, v1, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v95
	v_add_f32_e32 v1, v1, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v97
	v_add_f32_e32 v1, v1, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v99
	v_add_f32_e32 v1, v1, v100
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v101
	v_add_f32_e32 v1, v1, v102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v103
	v_add_f32_e32 v1, v1, v104
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v105
	v_add_f32_e32 v1, v1, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v107
	v_add_f32_e32 v1, v1, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v109
	v_add_f32_e32 v1, v1, v110
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v111
	v_add_f32_e32 v1, v1, v112
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v113
	v_add_f32_e32 v1, v1, v114
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v115
	v_add_f32_e32 v1, v1, v116
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v117
	v_add_f32_e32 v1, v1, v118
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v119
	v_add_f32_e32 v1, v1, v120
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v121
	v_add_f32_e32 v1, v1, v122
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v123
	v_add_f32_e32 v1, v1, v124
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v125
	v_add_f32_e32 v1, v1, v126
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v127
	v_add_f32_e32 v1, v1, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v161
	v_add_f32_e32 v1, v1, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v163
	v_add_f32_e32 v1, v1, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v165
	v_add_f32_e32 v1, v1, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v167
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end9:
	.size	_Z3runILi3ELi4EEvPfPji, .Lfunc_end9-_Z3runILi3ELi4EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi3ELi4EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 187
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end9-_Z3runILi3ELi4EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi3ELi4EEvPfPji,"axG",@progbits,_Z3runILi3ELi4EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi3ELi4EEvPfPji.num_vgpr, 187
	.set .L_Z3runILi3ELi4EEvPfPji.num_agpr, 0
	.set .L_Z3runILi3ELi4EEvPfPji.numbered_sgpr, 12
	.set .L_Z3runILi3ELi4EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi3ELi4EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi3ELi4EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi3ELi4EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi3ELi4EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi3ELi4EEvPfPji.has_recursion, 0
	.set .L_Z3runILi3ELi4EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8668
; TotalNumSgprs: 14
; NumVgprs: 187
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 187
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi4ELi4EEvPfPji,"axG",@progbits,_Z3runILi4ELi4EEvPfPji,comdat
	.protected	_Z3runILi4ELi4EEvPfPji  ; -- Begin function _Z3runILi4ELi4EEvPfPji
	.globl	_Z3runILi4ELi4EEvPfPji
	.p2align	8
	.type	_Z3runILi4ELi4EEvPfPji,@function
_Z3runILi4ELi4EEvPfPji:                 ; @_Z3runILi4ELi4EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB10_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB10_2:                               ; %.preheader82
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s1, s[0:1], 0x10
	v_lshl_add_u32 v33, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s0, 0x3a83126f
	ds_store_b32 v33, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v162, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v163, s0, v1, 0x3dcccccd
	v_fmaak_f32 v164, s0, v1, 0x3e4ccccd
	v_fmaak_f32 v165, s0, v1, 0x3e99999a
	v_fmaak_f32 v166, s0, v1, 0x3ecccccd
	v_fma_f32 v167, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v168, s0, v1, 0x3f19999a
	v_fmaak_f32 v169, s0, v1, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s1, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB10_10
; %bb.3:                                ; %.lr.ph
	v_dual_mov_b32 v95, 0x3f93f7cf :: v_dual_mov_b32 v170, 0
	v_cmp_lt_u32_e64 s0, 0x7f, v0
	v_dual_mov_b32 v94, 0x3f941893 :: v_dual_mov_b32 v93, 1.0
	v_dual_mov_b32 v96, 0x3f93d70a :: v_dual_mov_b32 v25, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v97, 0x3f93b645 :: v_dual_mov_b32 v26, v170
	v_dual_mov_b32 v98, 0x3f939581 :: v_dual_mov_b32 v27, v170
	v_dual_mov_b32 v99, 0x3f9374bc :: v_dual_mov_b32 v28, v170
	v_dual_mov_b32 v100, 0x3f9353f8 :: v_dual_mov_b32 v29, v170
	v_dual_mov_b32 v101, 0x3f933333 :: v_dual_mov_b32 v30, v170
	v_dual_mov_b32 v102, 0x3f92d0e5 :: v_dual_mov_b32 v31, v170
	v_dual_mov_b32 v103, 0x3f92b021 :: v_dual_mov_b32 v32, v170
	v_dual_mov_b32 v104, 0x3f928f5c :: v_dual_mov_b32 v17, v170
	v_dual_mov_b32 v105, 0x3f926e97 :: v_dual_mov_b32 v18, v170
	v_dual_mov_b32 v106, 0x3f924dd3 :: v_dual_mov_b32 v19, v170
	v_dual_mov_b32 v107, 0x3f922d0e :: v_dual_mov_b32 v20, v170
	v_dual_mov_b32 v108, 0x3f920c4a :: v_dual_mov_b32 v21, v170
	v_dual_mov_b32 v109, 0x3f91eb85 :: v_dual_mov_b32 v22, v170
	v_dual_mov_b32 v110, 0x3f918937 :: v_dual_mov_b32 v23, v170
	v_dual_mov_b32 v111, 0x3f916873 :: v_dual_mov_b32 v24, v170
	v_dual_mov_b32 v112, 0x3f9147ae :: v_dual_mov_b32 v9, v170
	v_dual_mov_b32 v113, 0x3f9126e9 :: v_dual_mov_b32 v10, v170
	v_dual_mov_b32 v114, 0x3f910625 :: v_dual_mov_b32 v11, v170
	v_dual_mov_b32 v115, 0x3f90e560 :: v_dual_mov_b32 v12, v170
	v_dual_mov_b32 v116, 0x3f90c49c :: v_dual_mov_b32 v13, v170
	v_dual_mov_b32 v117, 0x3f90a3d7 :: v_dual_mov_b32 v14, v170
	v_dual_mov_b32 v118, 0x3f904189 :: v_dual_mov_b32 v15, v170
	v_dual_mov_b32 v119, 0x3f9020c5 :: v_dual_mov_b32 v16, v170
	v_dual_mov_b32 v120, 0x3f900000 :: v_dual_mov_b32 v1, v170
	v_dual_mov_b32 v121, 0x3f8fdf3b :: v_dual_mov_b32 v2, v170
	v_dual_mov_b32 v122, 0x3f8fbe77 :: v_dual_mov_b32 v3, v170
	v_dual_mov_b32 v123, 0x3f8f9db2 :: v_dual_mov_b32 v4, v170
	v_dual_mov_b32 v124, 0x3f8f7cee :: v_dual_mov_b32 v5, v170
	v_dual_mov_b32 v125, 0x3f8f5c29 :: v_dual_mov_b32 v6, v170
	v_dual_mov_b32 v126, 0x3f8ef9db :: v_dual_mov_b32 v7, v170
	v_dual_mov_b32 v127, 0x3f8ed917 :: v_dual_mov_b32 v8, v170
	v_mov_b32_e32 v128, 0x3f8eb852
	v_mov_b32_e32 v129, 0x3f8e978d
	v_mov_b32_e32 v130, 0x3f8e76c9
	v_mov_b32_e32 v131, 0x3f8e5604
	v_mov_b32_e32 v132, 0x3f8e3540
	v_mov_b32_e32 v133, 0x3f8e147b
	v_mov_b32_e32 v134, 0x3f8db22d
	v_mov_b32_e32 v135, 0x3f8d9169
	v_mov_b32_e32 v136, 0x3f8d70a4
	v_mov_b32_e32 v137, 0x3f8d4fdf
	v_mov_b32_e32 v138, 0x3f8d2f1b
	v_mov_b32_e32 v139, 0x3f8d0e56
	v_mov_b32_e32 v140, 0x3f8ced92
	v_mov_b32_e32 v141, 0x3f8ccccd
	v_mov_b32_e32 v142, 0x3f8c6a7f
	v_mov_b32_e32 v143, 0x3f8c49bb
	v_mov_b32_e32 v144, 0x3f8c28f6
	v_mov_b32_e32 v145, 0x3f8c0831
	v_mov_b32_e32 v146, 0x3f8be76d
	v_mov_b32_e32 v147, 0x3f8bc6a8
	v_mov_b32_e32 v148, 0x3f8ba5e4
	v_mov_b32_e32 v149, 0x3f8b851f
	v_mov_b32_e32 v150, 0x3f8b22d1
	v_mov_b32_e32 v151, 0x3f8b020d
	v_mov_b32_e32 v152, 0x3f8ae148
	v_mov_b32_e32 v153, 0x3f8ac083
	v_mov_b32_e32 v154, 0x3f8a9fbf
	v_mov_b32_e32 v155, 0x3f8a7efa
	v_mov_b32_e32 v156, 0x3f8a5e36
	v_mov_b32_e32 v157, 0x3f8a3d71
	v_mov_b32_e32 v158, 0x3f89db23
	v_mov_b32_e32 v159, 0x3f89ba5f
	v_mov_b32_e32 v160, 0x3f89999a
	v_mov_b32_e32 v161, 0x3f8978d5
	v_mov_b32_e32 v34, 0x3f895811
	v_mov_b32_e32 v35, 0x3f89374c
	v_mov_b32_e32 v36, 0x3f891688
	v_mov_b32_e32 v37, 0x3f88f5c3
	v_mov_b32_e32 v38, 0x3f889374
	v_mov_b32_e32 v39, 0x3f8872b0
	v_mov_b32_e32 v40, 0x3f8851eb
	v_mov_b32_e32 v41, 0x3f883126
	v_mov_b32_e32 v42, 0x3f881062
	v_mov_b32_e32 v43, 0x3f87ef9d
	v_mov_b32_e32 v44, 0x3f87ced9
	v_mov_b32_e32 v45, 0x3f87ae14
	v_mov_b32_e32 v46, 0x3f874bc6
	v_mov_b32_e32 v47, 0x3f872b02
	v_mov_b32_e32 v48, 0x3f870a3d
	v_mov_b32_e32 v49, 0x3f86e978
	v_mov_b32_e32 v50, 0x3f86c8b4
	v_mov_b32_e32 v51, 0x3f86a7ef
	v_mov_b32_e32 v52, 0x3f86872b
	v_mov_b32_e32 v53, 0x3f866666
	v_mov_b32_e32 v54, 0x3f860418
	v_mov_b32_e32 v55, 0x3f85e354
	v_mov_b32_e32 v56, 0x3f85c28f
	v_mov_b32_e32 v57, 0x3f85a1ca
	v_mov_b32_e32 v58, 0x3f858106
	v_mov_b32_e32 v59, 0x3f856041
	v_mov_b32_e32 v60, 0x3f853f7d
	v_mov_b32_e32 v61, 0x3f851eb8
	v_mov_b32_e32 v62, 0x3f84bc6a
	v_mov_b32_e32 v63, 0x3f849ba6
	v_mov_b32_e32 v64, 0x3f847ae1
	v_mov_b32_e32 v65, 0x3f845a1c
	v_mov_b32_e32 v66, 0x3f843958
	v_mov_b32_e32 v67, 0x3f841893
	v_mov_b32_e32 v68, 0x3f83f7cf
	v_mov_b32_e32 v69, 0x3f83d70a
	v_mov_b32_e32 v70, 0x3f8374bc
	v_mov_b32_e32 v71, 0x3f8353f8
	v_mov_b32_e32 v72, 0x3f833333
	v_mov_b32_e32 v73, 0x3f83126e
	v_mov_b32_e32 v74, 0x3f82f1aa
	v_mov_b32_e32 v75, 0x3f82d0e5
	v_mov_b32_e32 v76, 0x3f82b021
	v_mov_b32_e32 v77, 0x3f828f5c
	v_mov_b32_e32 v78, 0x3f822d0e
	v_mov_b32_e32 v79, 0x3f820c4a
	v_mov_b32_e32 v80, 0x3f81eb85
	v_mov_b32_e32 v81, 0x3f81cac0
	v_mov_b32_e32 v82, 0x3f81a9fc
	v_mov_b32_e32 v83, 0x3f818937
	v_mov_b32_e32 v84, 0x3f816873
	v_mov_b32_e32 v85, 0x3f8147ae
	v_mov_b32_e32 v86, 0x3f80e560
	v_mov_b32_e32 v87, 0x3f80c49c
	v_mov_b32_e32 v88, 0x3f80a3d7
	v_mov_b32_e32 v89, 0x3f808312
	v_mov_b32_e32 v90, 0x3f80624e
	v_mov_b32_e32 v91, 0x3f804189
	v_mov_b32_e32 v92, 0x3f8020c5
	v_mbcnt_lo_u32_b32 v171, -1, 0
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0x35800000
	s_mov_b32 s2, 0x10101010
	s_mov_b32 s6, 0x18181818
	s_branch .LBB10_5
.LBB10_4:                               ; %Flow1584
                                        ;   in Loop: Header=BB10_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s10
	s_add_co_i32 s8, s8, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, s1
	s_cbranch_scc1 .LBB10_11
.LBB10_5:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB10_9 Depth 2
	s_and_saveexec_b32 s3, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB10_7
; %bb.6:                                ;   in Loop: Header=BB10_5 Depth=1
	v_max3_num_f32 v172, v162, 0xff800000, v163
	v_xor_b32_e32 v173, 16, v171
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v172, v172, v164, v165
	v_cmp_gt_u32_e32 vcc_lo, 32, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_num_f32 v172, v172, v166, v167
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v173, v171, v173, vcc_lo
	v_max3_num_f32 v172, v172, v168, v169
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v173, 2, v173
	ds_bpermute_b32 v174, v173, v172
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v174, v174, v174
	v_max_num_f32_e32 v172, v172, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v162, v162, v172 :: v_dual_sub_f32 v163, v163, v172
	v_dual_sub_f32 v166, v166, v172 :: v_dual_sub_f32 v167, v167, v172
	v_dual_sub_f32 v165, v165, v172 :: v_dual_mul_f32 v162, 0x3fb8aa3b, v162
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v163, 0x3fb8aa3b, v163 :: v_dual_sub_f32 v164, v164, v172
	v_dual_mul_f32 v166, 0x3fb8aa3b, v166 :: v_dual_mul_f32 v165, 0x3fb8aa3b, v165
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v162, v162
	v_exp_f32_e32 v163, v163
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v164, 0x3fb8aa3b, v164 :: v_dual_mul_f32 v167, 0x3fb8aa3b, v167
	v_exp_f32_e32 v166, v166
	v_exp_f32_e32 v165, v165
	v_sub_f32_e32 v169, v169, v172
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v164, v164
	v_exp_f32_e32 v167, v167
	v_sub_f32_e32 v168, v168, v172
	v_add_f32_e32 v174, v162, v163
	v_mul_f32_e32 v162, 0x3a83126f, v162
	v_mul_f32_e32 v169, 0x3fb8aa3b, v169
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v163, 0x3b03126f, v163 :: v_dual_add_f32 v174, v164, v174
	v_mul_f32_e32 v164, 0x3b449ba6, v164
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v169, v169
	v_max3_num_f32 v175, v162, 0, v163
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v174, v165, v174 :: v_dual_mul_f32 v165, 0x3b83126f, v165
	v_add_f32_e32 v174, v166, v174
	v_mul_f32_e32 v166, 0x3ba3d70b, v166
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v175, v175, v164, v165
	v_dual_add_f32 v174, v167, v174 :: v_dual_mul_f32 v167, 0x3bc49ba6, v167
	v_mul_f32_e32 v168, 0x3fb8aa3b, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v175, v175, v166, v167
	v_exp_f32_e32 v168, v168
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v174, v168, v174
	v_mul_f32_e32 v168, 0x3be56042, v168
	v_dual_add_f32 v172, v169, v174 :: v_dual_mul_f32 v169, 0x3c03126f, v169
	ds_bpermute_b32 v174, v173, v172
	v_max3_num_f32 v175, v175, v168, v169
	ds_bpermute_b32 v173, v173, v175
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v173, v173, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v173, v175, v173
	v_div_scale_f32 v175, null, 0x43e00000, 0x43e00000, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v176, v175
	v_fma_f32 v177, -v175, v176, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v173, 0x43e00000, v173
	v_mul_f32_e32 v178, v177, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v175, v178, v177
	v_fmac_f32_e32 v178, v179, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v175, -v175, v178, v177
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v175, v175, v176, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v173, v175, 0x43e00000, v173
	v_dual_max_num_f32 v173, 0x1f800000, v173 :: v_dual_add_f32 v172, v172, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v175, null, v173, v173, v162
	v_rcp_f32_e32 v176, v175
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v175, v176, 1.0
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v162, v173, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v178, v177, v176
	v_fma_f32 v179, -v175, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v176
	v_fma_f32 v175, -v175, v178, v177
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v175, v175, v176, v178
	v_div_scale_f32 v176, null, v173, v173, v163
	v_rcp_f32_e32 v177, v176
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v178, -v176, v177, 1.0
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v163, v173, v163
	v_div_fixup_f32 v175, v175, v173, v162
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v179, v178, v177
	v_fma_f32 v180, -v176, v179, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v177
	v_fma_f32 v176, -v176, v179, v178
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v176, v176, v177, v179
	v_mov_b16_e64 v177.l, v170.l
	v_mov_b16_e64 v177.h, 0
	v_div_fixup_f32 v176, v176, v173, v163
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v178.l, v177.l
	v_mov_b16_e64 v178.h, v177.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v178.l, v175, v176
	v_div_scale_f32 v175, null, v173, v173, v164
	v_rcp_f32_e32 v176, v175
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v175, v176, 1.0
	v_fmac_f32_e32 v176, v179, v176
	v_div_scale_f32 v179, vcc_lo, v164, v173, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v176
	v_fma_f32 v181, -v175, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v176
	v_fma_f32 v175, -v175, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v175, v175, v176, v180
	v_div_scale_f32 v176, null, v173, v173, v165
	v_div_fixup_f32 v175, v175, v173, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v176
	v_fma_f32 v180, -v176, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v165, v173, v165
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v176, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v176, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v176, v176, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v176, v176, v173, v165
	v_cvt_pk_fp8_f32 v178.h, v175, v176
	v_div_scale_f32 v175, null, v173, v173, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v176, v175
	v_fma_f32 v179, -v175, v176, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v179, v176
	v_div_scale_f32 v179, vcc_lo, v166, v173, v166
	v_mul_f32_e32 v180, v179, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v175, v180, v179
	v_fmac_f32_e32 v180, v181, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v175, -v175, v180, v179
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v175, v175, v176, v180
	v_div_scale_f32 v176, null, v173, v173, v167
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v175, v175, v173, v166
	v_rcp_f32_e32 v179, v176
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v176, v179, 1.0
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v167, v173, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v181, v180, v179
	v_fma_f32 v182, -v176, v181, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v182, v179
	v_fma_f32 v176, -v176, v181, v180
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v176, v176, v179, v181
	v_div_fixup_f32 v176, v176, v173, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v177.l, v175, v176
	v_div_scale_f32 v175, null, v173, v173, v168
	v_rcp_f32_e32 v176, v175
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v175, v176, 1.0
	v_fmac_f32_e32 v176, v179, v176
	v_div_scale_f32 v179, vcc_lo, v168, v173, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v176
	v_fma_f32 v181, -v175, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v176
	v_fma_f32 v175, -v175, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v175, v175, v176, v180
	v_div_scale_f32 v176, null, v173, v173, v169
	v_div_fixup_f32 v175, v175, v173, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v176
	v_fma_f32 v180, -v176, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v169, v173, v169
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v176, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v176, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v176, v176, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v173, v176, v173, v169
	v_cvt_pk_fp8_f32 v177.h, v175, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v173, v178, v177
	v_cvt_f32_ubyte0_e32 v173, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmaak_f32 v173, s9, v173, 0x3f7fbe77
	v_dual_mul_f32 v93, v93, v173 :: v_dual_mul_f32 v92, v92, v173
	v_mul_f32_e32 v83, v83, v173
	v_dual_mul_f32 v91, v91, v173 :: v_dual_mul_f32 v90, v90, v173
	v_mul_f32_e32 v79, v79, v173
	v_dual_mul_f32 v89, v89, v173 :: v_dual_mul_f32 v88, v88, v173
	v_mul_f32_e32 v77, v77, v173
	v_dual_mul_f32 v87, v87, v173 :: v_dual_mul_f32 v86, v86, v173
	v_mul_f32_e32 v75, v75, v173
	v_dual_mul_f32 v85, v85, v173 :: v_dual_mul_f32 v84, v84, v173
	v_dual_mul_f32 v73, v73, v173 :: v_dual_mul_f32 v82, v82, v173
	v_mul_f32_e32 v71, v71, v173
	v_dual_mul_f32 v81, v81, v173 :: v_dual_mul_f32 v80, v80, v173
	v_dual_mul_f32 v69, v69, v173 :: v_dual_mul_f32 v78, v78, v173
	v_dual_mul_f32 v67, v67, v173 :: v_dual_mul_f32 v76, v76, v173
	v_dual_mul_f32 v65, v65, v173 :: v_dual_mul_f32 v74, v74, v173
	v_dual_mul_f32 v63, v63, v173 :: v_dual_mul_f32 v72, v72, v173
	v_dual_mul_f32 v61, v61, v173 :: v_dual_mul_f32 v70, v70, v173
	v_dual_mul_f32 v59, v59, v173 :: v_dual_mul_f32 v68, v68, v173
	v_dual_mul_f32 v57, v57, v173 :: v_dual_mul_f32 v66, v66, v173
	v_dual_mul_f32 v55, v55, v173 :: v_dual_mul_f32 v64, v64, v173
	v_dual_mul_f32 v53, v53, v173 :: v_dual_mul_f32 v62, v62, v173
	v_dual_mul_f32 v51, v51, v173 :: v_dual_mul_f32 v60, v60, v173
	v_dual_mul_f32 v49, v49, v173 :: v_dual_mul_f32 v58, v58, v173
	v_dual_mul_f32 v47, v47, v173 :: v_dual_mul_f32 v56, v56, v173
	v_dual_mul_f32 v45, v45, v173 :: v_dual_mul_f32 v54, v54, v173
	v_dual_mul_f32 v43, v43, v173 :: v_dual_mul_f32 v52, v52, v173
	v_dual_mul_f32 v41, v41, v173 :: v_dual_mul_f32 v50, v50, v173
	v_dual_mul_f32 v39, v39, v173 :: v_dual_mul_f32 v48, v48, v173
	v_dual_mul_f32 v37, v37, v173 :: v_dual_mul_f32 v46, v46, v173
	v_dual_mul_f32 v35, v35, v173 :: v_dual_mul_f32 v44, v44, v173
	v_dual_mul_f32 v161, v161, v173 :: v_dual_mul_f32 v42, v42, v173
	v_dual_mul_f32 v159, v159, v173 :: v_dual_mul_f32 v40, v40, v173
	v_dual_mul_f32 v157, v157, v173 :: v_dual_mul_f32 v38, v38, v173
	v_dual_mul_f32 v155, v155, v173 :: v_dual_mul_f32 v36, v36, v173
	v_dual_mul_f32 v153, v153, v173 :: v_dual_mul_f32 v34, v34, v173
	v_dual_mul_f32 v151, v151, v173 :: v_dual_mul_f32 v160, v160, v173
	v_dual_mul_f32 v149, v149, v173 :: v_dual_mul_f32 v158, v158, v173
	v_dual_mul_f32 v147, v147, v173 :: v_dual_mul_f32 v156, v156, v173
	v_dual_mul_f32 v145, v145, v173 :: v_dual_mul_f32 v154, v154, v173
	v_dual_mul_f32 v143, v143, v173 :: v_dual_mul_f32 v152, v152, v173
	v_dual_mul_f32 v141, v141, v173 :: v_dual_mul_f32 v150, v150, v173
	v_dual_mul_f32 v139, v139, v173 :: v_dual_mul_f32 v148, v148, v173
	v_dual_mul_f32 v137, v137, v173 :: v_dual_mul_f32 v146, v146, v173
	v_dual_mul_f32 v135, v135, v173 :: v_dual_mul_f32 v144, v144, v173
	v_dual_mul_f32 v133, v133, v173 :: v_dual_mul_f32 v142, v142, v173
	v_dual_mul_f32 v131, v131, v173 :: v_dual_mul_f32 v140, v140, v173
	v_dual_mul_f32 v129, v129, v173 :: v_dual_mul_f32 v138, v138, v173
	v_dual_mul_f32 v127, v127, v173 :: v_dual_mul_f32 v136, v136, v173
	v_dual_mul_f32 v123, v173, v123 :: v_dual_mul_f32 v134, v134, v173
	v_dual_mul_f32 v119, v173, v119 :: v_dual_mul_f32 v132, v132, v173
	v_dual_mul_f32 v115, v173, v115 :: v_dual_mul_f32 v130, v130, v173
	v_dual_mul_f32 v111, v173, v111 :: v_dual_mul_f32 v128, v128, v173
	v_dual_mul_f32 v107, v173, v107 :: v_dual_mul_f32 v126, v126, v173
	v_mul_f32_e32 v103, v173, v103
	v_dual_mul_f32 v125, v173, v125 :: v_dual_mul_f32 v124, v173, v124
	v_dual_mul_f32 v122, v173, v122 :: v_dual_mul_f32 v121, v173, v121
	v_dual_mul_f32 v120, v173, v120 :: v_dual_mul_f32 v117, v173, v117
	v_dual_mul_f32 v118, v173, v118 :: v_dual_mul_f32 v113, v173, v113
	v_dual_mul_f32 v116, v173, v116 :: v_dual_mul_f32 v109, v173, v109
	v_dual_mul_f32 v114, v173, v114 :: v_dual_mul_f32 v105, v173, v105
	v_dual_mul_f32 v112, v173, v112 :: v_dual_mul_f32 v101, v173, v101
	v_dual_mul_f32 v110, v173, v110 :: v_dual_mul_f32 v99, v173, v99
	v_dual_mul_f32 v108, v173, v108 :: v_dual_mul_f32 v97, v173, v97
	v_dual_mul_f32 v106, v173, v106 :: v_dual_mul_f32 v95, v173, v95
	v_mul_f32_e32 v104, v173, v104
	v_mul_f32_e32 v102, v173, v102
	v_mul_f32_e32 v100, v173, v100
	v_mul_f32_e32 v98, v173, v98
	v_mul_f32_e32 v96, v173, v96
	v_mul_f32_e32 v94, v173, v94
	v_div_scale_f32 v173, null, v172, v172, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v162, v172, v162
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v162, v173, v172, v162
	v_div_scale_f32 v173, null, v172, v172, v163
	v_add_f32_e32 v162, 0, v162
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v163, v172, v163
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v163, v173, v172, v163
	v_div_scale_f32 v173, null, v172, v172, v164
	v_add_f32_e32 v163, 0x3c23d70a, v163
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v164, v172, v164
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v164, v173, v172, v164
	v_div_scale_f32 v173, null, v172, v172, v165
	v_add_f32_e32 v164, 0x3ca3d70a, v164
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v165, v172, v165
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v165, v173, v172, v165
	v_div_scale_f32 v173, null, v172, v172, v166
	v_add_f32_e32 v165, 0x3cf5c28f, v165
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v166, v172, v166
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v166, v173, v172, v166
	v_div_scale_f32 v173, null, v172, v172, v167
	v_add_f32_e32 v166, 0x3d23d70a, v166
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v167, v172, v167
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v167, v173, v172, v167
	v_div_scale_f32 v173, null, v172, v172, v168
	v_add_f32_e32 v167, 0x3d4ccccc, v167
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v168, v172, v168
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v168, v173, v172, v168
	v_div_scale_f32 v173, null, v172, v172, v169
	v_add_f32_e32 v168, 0x3d75c28f, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v174, v173
	v_fma_f32 v175, -v173, v174, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v169, v172, v169
	v_mul_f32_e32 v176, v175, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v173, v176, v175
	v_fmac_f32_e32 v176, v177, v174
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v173, v176, v175
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v173, v173, v174, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v169, v173, v172, v169
	v_add_f32_e32 v169, 0x3d8f5c29, v169
.LBB10_7:                               ; %Flow
                                        ;   in Loop: Header=BB10_5 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s3
	s_cbranch_execz .LBB10_4
; %bb.8:                                ;   in Loop: Header=BB10_5 Depth=1
	s_mov_b32 s11, 32
.LBB10_9:                               ; %.preheader.i
                                        ;   Parent Loop BB10_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, s6
	s_mov_b32 s3, s2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v173, s7 :: v_dual_mov_b32 v172, s6
	v_dual_mov_b32 v175, s3 :: v_dual_mov_b32 v174, s2
	s_add_co_i32 s11, s11, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_cmp_lg_u32 s11, 0
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[172:173], v[174:175], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[172:173], v[174:175], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[172:173], v[174:175], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[172:173], v[174:175], v[1:8]
	s_cbranch_scc1 .LBB10_9
	s_branch .LBB10_4
.LBB10_10:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v93, 1.0 :: v_dual_mov_b32 v92, 0x3f8020c5
	v_dual_mov_b32 v91, 0x3f804189 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v90, 0x3f80624e :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v89, 0x3f808312 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v88, 0x3f80a3d7 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v87, 0x3f80c49c :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v86, 0x3f80e560 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v85, 0x3f8147ae :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v84, 0x3f816873 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v83, 0x3f818937 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v82, 0x3f81a9fc :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v81, 0x3f81cac0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v80, 0x3f81eb85 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v79, 0x3f820c4a :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v78, 0x3f822d0e :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v77, 0x3f828f5c :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v76, 0x3f82b021 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v75, 0x3f82d0e5 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v74, 0x3f82f1aa :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v73, 0x3f83126e :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v72, 0x3f833333 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v71, 0x3f8353f8 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v70, 0x3f8374bc :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v69, 0x3f83d70a :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v68, 0x3f83f7cf :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v67, 0x3f841893 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v66, 0x3f843958 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v65, 0x3f845a1c :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v64, 0x3f847ae1 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v63, 0x3f849ba6 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v62, 0x3f84bc6a :: v_dual_mov_b32 v31, 0
	v_mov_b32_e32 v61, 0x3f851eb8
	v_mov_b32_e32 v60, 0x3f853f7d
	v_mov_b32_e32 v59, 0x3f856041
	v_mov_b32_e32 v58, 0x3f858106
	v_mov_b32_e32 v57, 0x3f85a1ca
	v_mov_b32_e32 v56, 0x3f85c28f
	v_mov_b32_e32 v55, 0x3f85e354
	v_mov_b32_e32 v54, 0x3f860418
	v_mov_b32_e32 v53, 0x3f866666
	v_mov_b32_e32 v52, 0x3f86872b
	v_mov_b32_e32 v51, 0x3f86a7ef
	v_mov_b32_e32 v50, 0x3f86c8b4
	v_mov_b32_e32 v49, 0x3f86e978
	v_mov_b32_e32 v48, 0x3f870a3d
	v_mov_b32_e32 v47, 0x3f872b02
	v_mov_b32_e32 v46, 0x3f874bc6
	v_mov_b32_e32 v45, 0x3f87ae14
	v_mov_b32_e32 v44, 0x3f87ced9
	v_mov_b32_e32 v43, 0x3f87ef9d
	v_mov_b32_e32 v42, 0x3f881062
	v_mov_b32_e32 v41, 0x3f883126
	v_mov_b32_e32 v40, 0x3f8851eb
	v_mov_b32_e32 v39, 0x3f8872b0
	v_mov_b32_e32 v38, 0x3f889374
	v_mov_b32_e32 v37, 0x3f88f5c3
	v_mov_b32_e32 v36, 0x3f891688
	v_mov_b32_e32 v35, 0x3f89374c
	v_mov_b32_e32 v34, 0x3f895811
	v_mov_b32_e32 v161, 0x3f8978d5
	v_mov_b32_e32 v160, 0x3f89999a
	v_mov_b32_e32 v159, 0x3f89ba5f
	v_mov_b32_e32 v158, 0x3f89db23
	v_mov_b32_e32 v157, 0x3f8a3d71
	v_mov_b32_e32 v156, 0x3f8a5e36
	v_mov_b32_e32 v155, 0x3f8a7efa
	v_mov_b32_e32 v154, 0x3f8a9fbf
	v_mov_b32_e32 v153, 0x3f8ac083
	v_mov_b32_e32 v152, 0x3f8ae148
	v_mov_b32_e32 v151, 0x3f8b020d
	v_mov_b32_e32 v150, 0x3f8b22d1
	v_mov_b32_e32 v149, 0x3f8b851f
	v_mov_b32_e32 v148, 0x3f8ba5e4
	v_mov_b32_e32 v147, 0x3f8bc6a8
	v_mov_b32_e32 v146, 0x3f8be76d
	v_mov_b32_e32 v145, 0x3f8c0831
	v_mov_b32_e32 v144, 0x3f8c28f6
	v_mov_b32_e32 v143, 0x3f8c49bb
	v_mov_b32_e32 v142, 0x3f8c6a7f
	v_mov_b32_e32 v141, 0x3f8ccccd
	v_mov_b32_e32 v140, 0x3f8ced92
	v_mov_b32_e32 v139, 0x3f8d0e56
	v_mov_b32_e32 v138, 0x3f8d2f1b
	v_mov_b32_e32 v137, 0x3f8d4fdf
	v_mov_b32_e32 v136, 0x3f8d70a4
	v_mov_b32_e32 v135, 0x3f8d9169
	v_mov_b32_e32 v134, 0x3f8db22d
	v_mov_b32_e32 v133, 0x3f8e147b
	v_mov_b32_e32 v132, 0x3f8e3540
	v_mov_b32_e32 v131, 0x3f8e5604
	v_mov_b32_e32 v130, 0x3f8e76c9
	v_mov_b32_e32 v129, 0x3f8e978d
	v_mov_b32_e32 v128, 0x3f8eb852
	v_mov_b32_e32 v127, 0x3f8ed917
	v_mov_b32_e32 v126, 0x3f8ef9db
	v_mov_b32_e32 v125, 0x3f8f5c29
	v_mov_b32_e32 v124, 0x3f8f7cee
	v_mov_b32_e32 v123, 0x3f8f9db2
	v_mov_b32_e32 v122, 0x3f8fbe77
	v_mov_b32_e32 v121, 0x3f8fdf3b
	v_mov_b32_e32 v120, 0x3f900000
	v_mov_b32_e32 v119, 0x3f9020c5
	v_mov_b32_e32 v118, 0x3f904189
	v_mov_b32_e32 v117, 0x3f90a3d7
	v_mov_b32_e32 v116, 0x3f90c49c
	v_mov_b32_e32 v115, 0x3f90e560
	v_mov_b32_e32 v114, 0x3f910625
	v_mov_b32_e32 v113, 0x3f9126e9
	v_mov_b32_e32 v112, 0x3f9147ae
	v_mov_b32_e32 v111, 0x3f916873
	v_mov_b32_e32 v110, 0x3f918937
	v_mov_b32_e32 v109, 0x3f91eb85
	v_mov_b32_e32 v108, 0x3f920c4a
	v_mov_b32_e32 v107, 0x3f922d0e
	v_mov_b32_e32 v106, 0x3f924dd3
	v_mov_b32_e32 v105, 0x3f926e97
	v_mov_b32_e32 v104, 0x3f928f5c
	v_mov_b32_e32 v103, 0x3f92b021
	v_mov_b32_e32 v102, 0x3f92d0e5
	v_mov_b32_e32 v101, 0x3f933333
	v_mov_b32_e32 v100, 0x3f9353f8
	v_mov_b32_e32 v99, 0x3f9374bc
	v_mov_b32_e32 v98, 0x3f939581
	v_mov_b32_e32 v97, 0x3f93b645
	v_mov_b32_e32 v96, 0x3f93d70a
	v_mov_b32_e32 v95, 0x3f93f7cf
	v_mov_b32_e32 v94, 0x3f941893
.LBB10_11:                              ; %Flow1585
	v_add_f32_e32 v25, 0, v25
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v33
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v93
	v_add_f32_e32 v1, v1, v92
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v91
	v_add_f32_e32 v1, v1, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v89
	v_add_f32_e32 v1, v1, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v87
	v_add_f32_e32 v1, v1, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v85
	v_add_f32_e32 v1, v1, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v83
	v_add_f32_e32 v1, v1, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v81
	v_add_f32_e32 v1, v1, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v79
	v_add_f32_e32 v1, v1, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v77
	v_add_f32_e32 v1, v1, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v75
	v_add_f32_e32 v1, v1, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v73
	v_add_f32_e32 v1, v1, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v71
	v_add_f32_e32 v1, v1, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v69
	v_add_f32_e32 v1, v1, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v67
	v_add_f32_e32 v1, v1, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v65
	v_add_f32_e32 v1, v1, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v63
	v_add_f32_e32 v1, v1, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v61
	v_add_f32_e32 v1, v1, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v59
	v_add_f32_e32 v1, v1, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v57
	v_add_f32_e32 v1, v1, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v55
	v_add_f32_e32 v1, v1, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v53
	v_add_f32_e32 v1, v1, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v51
	v_add_f32_e32 v1, v1, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v49
	v_add_f32_e32 v1, v1, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v47
	v_add_f32_e32 v1, v1, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v45
	v_add_f32_e32 v1, v1, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v43
	v_add_f32_e32 v1, v1, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v41
	v_add_f32_e32 v1, v1, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v39
	v_add_f32_e32 v1, v1, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v37
	v_add_f32_e32 v1, v1, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v35
	v_add_f32_e32 v1, v1, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v161
	v_add_f32_e32 v1, v1, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v159
	v_add_f32_e32 v1, v1, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v157
	v_add_f32_e32 v1, v1, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v155
	v_add_f32_e32 v1, v1, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v153
	v_add_f32_e32 v1, v1, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v151
	v_add_f32_e32 v1, v1, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v149
	v_add_f32_e32 v1, v1, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v147
	v_add_f32_e32 v1, v1, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v145
	v_add_f32_e32 v1, v1, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v143
	v_add_f32_e32 v1, v1, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v141
	v_add_f32_e32 v1, v1, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v139
	v_add_f32_e32 v1, v1, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v135
	v_add_f32_e32 v1, v1, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v133
	v_add_f32_e32 v1, v1, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v131
	v_add_f32_e32 v1, v1, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v129
	v_add_f32_e32 v1, v1, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v127
	v_add_f32_e32 v1, v1, v126
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v125
	v_add_f32_e32 v1, v1, v124
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v123
	v_add_f32_e32 v1, v1, v122
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v121
	v_add_f32_e32 v1, v1, v120
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v119
	v_add_f32_e32 v1, v1, v118
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v117
	v_add_f32_e32 v1, v1, v116
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v115
	v_add_f32_e32 v1, v1, v114
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v113
	v_add_f32_e32 v1, v1, v112
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v111
	v_add_f32_e32 v1, v1, v110
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v109
	v_add_f32_e32 v1, v1, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v107
	v_add_f32_e32 v1, v1, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v105
	v_add_f32_e32 v1, v1, v104
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v103
	v_add_f32_e32 v1, v1, v102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v101
	v_add_f32_e32 v1, v1, v100
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v99
	v_add_f32_e32 v1, v1, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v97
	v_add_f32_e32 v1, v1, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v95
	v_add_f32_e32 v1, v1, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v162
	v_add_f32_e32 v1, v1, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v164
	v_add_f32_e32 v1, v1, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v166
	v_add_f32_e32 v1, v1, v167
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v168
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v169
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end10:
	.size	_Z3runILi4ELi4EEvPfPji, .Lfunc_end10-_Z3runILi4ELi4EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi4ELi4EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 183
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end10-_Z3runILi4ELi4EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi4ELi4EEvPfPji,"axG",@progbits,_Z3runILi4ELi4EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi4ELi4EEvPfPji.num_vgpr, 183
	.set .L_Z3runILi4ELi4EEvPfPji.num_agpr, 0
	.set .L_Z3runILi4ELi4EEvPfPji.numbered_sgpr, 12
	.set .L_Z3runILi4ELi4EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi4ELi4EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi4ELi4EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi4ELi4EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi4ELi4EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi4ELi4EEvPfPji.has_recursion, 0
	.set .L_Z3runILi4ELi4EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6564
; TotalNumSgprs: 14
; NumVgprs: 183
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 183
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z3runILi5ELi4EEvPfPji,"axG",@progbits,_Z3runILi5ELi4EEvPfPji,comdat
	.protected	_Z3runILi5ELi4EEvPfPji  ; -- Begin function _Z3runILi5ELi4EEvPfPji
	.globl	_Z3runILi5ELi4EEvPfPji
	.p2align	8
	.type	_Z3runILi5ELi4EEvPfPji,@function
_Z3runILi5ELi4EEvPfPji:                 ; @_Z3runILi5ELi4EEvPfPji
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 31, v0
	s_mov_b32 s2, exec_lo
	;;#ASMSTART
	s_getreg_b32 s3, hwreg(HW_REG_HW_ID1)
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB11_2
; %bb.1:
	v_lshrrev_b32_e32 v2, 5, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v2, ttmp9, 3, v2
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, s6, v2
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	global_store_b32 v[2:3], v4, off
.LBB11_2:                               ; %.preheader79
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_kmcnt 0x0
	s_load_b32 s6, s[0:1], 0x10
	v_lshl_add_u32 v41, v0, 2, 0
	v_cvt_f32_ubyte0_e32 v1, v1
	s_mov_b32 s0, 0x3a83126f
	ds_store_b32 v41, v0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v170, 0x3a83126f, v1
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v171, s0, v1, 0x3dcccccd
	v_fmaak_f32 v172, s0, v1, 0x3e4ccccd
	v_fmaak_f32 v173, s0, v1, 0x3e99999a
	v_fmaak_f32 v175, s0, v1, 0x3ecccccd
	v_fma_f32 v176, 0x3a83126f, v1, 0.5
	v_fmaak_f32 v177, s0, v1, 0x3f19999a
	v_fmaak_f32 v178, s0, v1, 0x3f333333
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s6, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB11_7
; %bb.3:                                ; %.lr.ph
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v102, 0x3f941893
	v_dual_mov_b32 v106, 0x3f939581 :: v_dual_mov_b32 v101, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v2, 16, v1
	v_dual_mov_b32 v34, v33 :: v_dual_mov_b32 v35, v33
	v_dual_mov_b32 v36, v33 :: v_dual_mov_b32 v37, v33
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_dual_mov_b32 v38, v33 :: v_dual_mov_b32 v39, v33
	v_dual_mov_b32 v40, v33 :: v_dual_mov_b32 v103, 0x3f93f7cf
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v1, v2 :: v_dual_mov_b32 v104, 0x3f93d70a
	v_mov_b32_e32 v105, 0x3f93b645
	v_mov_b32_e32 v108, 0x3f9353f8
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v107, 0x3f9374bc :: v_dual_lshlrev_b32 v174, 2, v1
	v_mov_b32_e32 v1, v33
	v_dual_mov_b32 v109, 0x3f933333 :: v_dual_mov_b32 v4, v36
	v_dual_mov_b32 v110, 0x3f92d0e5 :: v_dual_mov_b32 v3, v35
	v_dual_mov_b32 v111, 0x3f92b021 :: v_dual_mov_b32 v6, v38
	v_dual_mov_b32 v112, 0x3f928f5c :: v_dual_mov_b32 v5, v37
	v_dual_mov_b32 v113, 0x3f926e97 :: v_dual_mov_b32 v8, v40
	v_dual_mov_b32 v114, 0x3f924dd3 :: v_dual_mov_b32 v7, v39
	v_dual_mov_b32 v115, 0x3f922d0e :: v_dual_mov_b32 v26, v33
	v_dual_mov_b32 v116, 0x3f920c4a :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v117, 0x3f91eb85 :: v_dual_mov_b32 v28, v33
	v_dual_mov_b32 v118, 0x3f918937 :: v_dual_mov_b32 v27, v33
	v_dual_mov_b32 v119, 0x3f916873 :: v_dual_mov_b32 v30, v33
	v_dual_mov_b32 v120, 0x3f9147ae :: v_dual_mov_b32 v29, v33
	v_dual_mov_b32 v121, 0x3f9126e9 :: v_dual_mov_b32 v32, v33
	v_dual_mov_b32 v122, 0x3f910625 :: v_dual_mov_b32 v31, v33
	v_dual_mov_b32 v123, 0x3f90e560 :: v_dual_mov_b32 v18, v33
	v_dual_mov_b32 v124, 0x3f90c49c :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v125, 0x3f90a3d7 :: v_dual_mov_b32 v20, v33
	v_dual_mov_b32 v126, 0x3f904189 :: v_dual_mov_b32 v19, v33
	v_dual_mov_b32 v127, 0x3f9020c5 :: v_dual_mov_b32 v22, v33
	v_dual_mov_b32 v128, 0x3f900000 :: v_dual_mov_b32 v21, v33
	v_dual_mov_b32 v129, 0x3f8fdf3b :: v_dual_mov_b32 v24, v33
	v_dual_mov_b32 v130, 0x3f8fbe77 :: v_dual_mov_b32 v23, v33
	v_dual_mov_b32 v131, 0x3f8f9db2 :: v_dual_mov_b32 v10, v33
	v_dual_mov_b32 v132, 0x3f8f7cee :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v133, 0x3f8f5c29 :: v_dual_mov_b32 v12, v33
	v_dual_mov_b32 v134, 0x3f8ef9db :: v_dual_mov_b32 v11, v33
	v_dual_mov_b32 v135, 0x3f8ed917 :: v_dual_mov_b32 v14, v33
	v_dual_mov_b32 v136, 0x3f8eb852 :: v_dual_mov_b32 v13, v33
	v_dual_mov_b32 v137, 0x3f8e978d :: v_dual_mov_b32 v16, v33
	v_dual_mov_b32 v138, 0x3f8e76c9 :: v_dual_mov_b32 v15, v33
	v_mov_b32_e32 v139, 0x3f8e5604
	v_mov_b32_e32 v140, 0x3f8e3540
	v_mov_b32_e32 v141, 0x3f8e147b
	v_mov_b32_e32 v142, 0x3f8db22d
	v_mov_b32_e32 v143, 0x3f8d9169
	v_mov_b32_e32 v144, 0x3f8d70a4
	v_mov_b32_e32 v145, 0x3f8d4fdf
	v_mov_b32_e32 v146, 0x3f8d2f1b
	v_mov_b32_e32 v147, 0x3f8d0e56
	v_mov_b32_e32 v148, 0x3f8ced92
	v_mov_b32_e32 v149, 0x3f8ccccd
	v_mov_b32_e32 v150, 0x3f8c6a7f
	v_mov_b32_e32 v151, 0x3f8c49bb
	v_mov_b32_e32 v152, 0x3f8c28f6
	v_mov_b32_e32 v153, 0x3f8c0831
	v_mov_b32_e32 v154, 0x3f8be76d
	v_mov_b32_e32 v155, 0x3f8bc6a8
	v_mov_b32_e32 v156, 0x3f8ba5e4
	v_mov_b32_e32 v157, 0x3f8b851f
	v_mov_b32_e32 v158, 0x3f8b22d1
	v_mov_b32_e32 v159, 0x3f8b020d
	v_mov_b32_e32 v160, 0x3f8ae148
	v_mov_b32_e32 v161, 0x3f8ac083
	v_mov_b32_e32 v162, 0x3f8a9fbf
	v_mov_b32_e32 v163, 0x3f8a7efa
	v_mov_b32_e32 v164, 0x3f8a5e36
	v_mov_b32_e32 v165, 0x3f8a3d71
	v_mov_b32_e32 v166, 0x3f89db23
	v_mov_b32_e32 v167, 0x3f89ba5f
	v_mov_b32_e32 v168, 0x3f89999a
	v_mov_b32_e32 v169, 0x3f8978d5
	v_mov_b32_e32 v42, 0x3f895811
	v_mov_b32_e32 v43, 0x3f89374c
	v_mov_b32_e32 v44, 0x3f891688
	v_mov_b32_e32 v45, 0x3f88f5c3
	v_mov_b32_e32 v46, 0x3f889374
	v_mov_b32_e32 v47, 0x3f8872b0
	v_mov_b32_e32 v48, 0x3f8851eb
	v_mov_b32_e32 v49, 0x3f883126
	v_mov_b32_e32 v50, 0x3f881062
	v_mov_b32_e32 v51, 0x3f87ef9d
	v_mov_b32_e32 v52, 0x3f87ced9
	v_mov_b32_e32 v53, 0x3f87ae14
	v_mov_b32_e32 v54, 0x3f874bc6
	v_mov_b32_e32 v55, 0x3f872b02
	v_mov_b32_e32 v56, 0x3f870a3d
	v_mov_b32_e32 v57, 0x3f86e978
	v_mov_b32_e32 v58, 0x3f86c8b4
	v_mov_b32_e32 v59, 0x3f86a7ef
	v_mov_b32_e32 v60, 0x3f86872b
	v_mov_b32_e32 v61, 0x3f866666
	v_mov_b32_e32 v62, 0x3f860418
	v_mov_b32_e32 v63, 0x3f85e354
	v_mov_b32_e32 v64, 0x3f85c28f
	v_mov_b32_e32 v65, 0x3f85a1ca
	v_mov_b32_e32 v66, 0x3f858106
	v_mov_b32_e32 v67, 0x3f856041
	v_mov_b32_e32 v68, 0x3f853f7d
	v_mov_b32_e32 v69, 0x3f851eb8
	v_mov_b32_e32 v70, 0x3f84bc6a
	v_mov_b32_e32 v71, 0x3f849ba6
	v_mov_b32_e32 v72, 0x3f847ae1
	v_mov_b32_e32 v73, 0x3f845a1c
	v_mov_b32_e32 v74, 0x3f843958
	v_mov_b32_e32 v75, 0x3f841893
	v_mov_b32_e32 v76, 0x3f83f7cf
	v_mov_b32_e32 v77, 0x3f83d70a
	v_mov_b32_e32 v78, 0x3f8374bc
	v_mov_b32_e32 v79, 0x3f8353f8
	v_mov_b32_e32 v80, 0x3f833333
	v_mov_b32_e32 v81, 0x3f83126e
	v_mov_b32_e32 v82, 0x3f82f1aa
	v_mov_b32_e32 v83, 0x3f82d0e5
	v_mov_b32_e32 v84, 0x3f82b021
	v_mov_b32_e32 v85, 0x3f828f5c
	v_mov_b32_e32 v86, 0x3f822d0e
	v_mov_b32_e32 v87, 0x3f820c4a
	v_mov_b32_e32 v88, 0x3f81eb85
	v_mov_b32_e32 v89, 0x3f81cac0
	v_mov_b32_e32 v90, 0x3f81a9fc
	v_mov_b32_e32 v91, 0x3f818937
	v_mov_b32_e32 v92, 0x3f816873
	v_mov_b32_e32 v93, 0x3f8147ae
	v_mov_b32_e32 v94, 0x3f80e560
	v_mov_b32_e32 v95, 0x3f80c49c
	v_mov_b32_e32 v96, 0x3f80a3d7
	v_mov_b32_e32 v97, 0x3f808312
	v_mov_b32_e32 v98, 0x3f80624e
	v_mov_b32_e32 v99, 0x3f804189
	v_mov_b32_e32 v100, 0x3f8020c5
	v_mov_b32_e32 v2, v34
	s_mov_b32 s0, 0x10101010
	s_mov_b32 s2, 0x18181818
	s_mov_b32 s7, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s1, s0
	s_mov_b32 s3, s2
	s_mov_b32 s8, 0x35800000
.LBB11_4:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB11_5 Depth 2
	s_mov_b32 s9, 32
.LBB11_5:                               ; %.preheader.i
                                        ;   Parent Loop BB11_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v35, s3 :: v_dual_mov_b32 v34, s2
	v_dual_mov_b32 v37, s1 :: v_dual_mov_b32 v36, s0
	s_add_co_i32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s9, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[34:35], v[36:37], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[34:35], v[36:37], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[34:35], v[36:37], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[34:35], v[36:37], v[1:8]
	s_cbranch_scc1 .LBB11_5
; %bb.6:                                ; %_Z6matrixILi4EEvRAT__Dv8_fDv2_iS3_.exit
                                        ;   in Loop: Header=BB11_4 Depth=1
	v_max3_num_f32 v34, v170, 0xff800000, v171
	s_add_co_i32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v34, v34, v172, v173
	v_max3_num_f32 v34, v34, v175, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v34, v34, v177, v178
	ds_bpermute_b32 v35, v174, v34
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v35, v35, v35
	v_max_num_f32_e32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v35, v170, v34
	v_dual_mul_f32 v35, 0x3fb8aa3b, v35 :: v_dual_sub_f32 v36, v171, v34
	v_dual_sub_f32 v39, v173, v34 :: v_dual_sub_f32 v38, v172, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v35, v35
	v_dual_mul_f32 v36, 0x3fb8aa3b, v36 :: v_dual_mul_f32 v39, 0x3fb8aa3b, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v38, 0x3fb8aa3b, v38
	v_exp_f32_e32 v36, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v173, v39
	v_sub_f32_e32 v39, v175, v34
	v_exp_f32_e32 v38, v38
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v170, 0x3a83126f, v35 :: v_dual_add_f32 v37, v35, v36
	v_mul_f32_e32 v40, 0x3b03126f, v36
	v_mul_f32_e32 v39, 0x3fb8aa3b, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v35, v170, 0, v40
	v_exp_f32_e32 v175, v39
	v_sub_f32_e32 v39, v176, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v39, 0x3fb8aa3b, v39
	v_exp_f32_e32 v176, v39
	v_dual_sub_f32 v39, v177, v34 :: v_dual_sub_f32 v34, v178, v34
	v_add_f32_e32 v37, v38, v37
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v39, 0x3fb8aa3b, v39 :: v_dual_mul_f32 v34, 0x3fb8aa3b, v34
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v37, v173, v37 :: v_dual_mul_f32 v36, 0x3bc49ba6, v176
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v177, v39
	v_exp_f32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v37, v175, v37
	v_mul_f32_e32 v39, 0x3b449ba6, v38
	v_mul_f32_e32 v38, 0x3b83126f, v173
	v_add_f32_e32 v37, v176, v37
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v35, v35, v39, v38
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_2)
	v_add_f32_e32 v37, v177, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v171, v34, v37
	v_mul_f32_e32 v37, 0x3ba3d70b, v175
	v_mul_f32_e32 v34, 0x3c03126f, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_num_f32 v173, v35, v37, v36
	v_mul_f32_e32 v35, 0x3be56042, v177
	v_max3_num_f32 v173, v173, v35, v34
	ds_bpermute_b32 v175, v174, v173
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v175, v175, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v173, v173, v175
	v_div_scale_f32 v175, null, 0x43e00000, 0x43e00000, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v176, v175
	v_fma_f32 v177, -v175, v176, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v173, 0x43e00000, v173
	v_mul_f32_e32 v178, v177, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v175, v178, v177
	v_fmac_f32_e32 v178, v179, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v175, -v175, v178, v177
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v175, v175, v176, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v173, v175, 0x43e00000, v173
	v_max_num_f32_e32 v173, 0x1f800000, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v175, null, v173, v173, v170
	v_rcp_f32_e32 v176, v175
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v175, v176, 1.0
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v170, v173, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v178, v177, v176
	v_fma_f32 v179, -v175, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v176
	ds_bpermute_b32 v172, v174, v171
	v_fma_f32 v175, -v175, v178, v177
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v175, v175, v176, v178
	v_div_scale_f32 v176, null, v173, v173, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v175, v175, v173, v170
	v_rcp_f32_e32 v177, v176
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v178, -v176, v177, 1.0
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v40, v173, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v179, v178, v177
	v_fma_f32 v180, -v176, v179, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v177
	v_fma_f32 v176, -v176, v179, v178
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v176, v176, v177, v179
	v_mov_b16_e64 v177.l, v33.l
	v_mov_b16_e64 v177.h, 0
	v_div_fixup_f32 v176, v176, v173, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v178.l, v177.l
	v_mov_b16_e64 v178.h, v177.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v178.l, v175, v176
	v_div_scale_f32 v175, null, v173, v173, v39
	v_rcp_f32_e32 v176, v175
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v175, v176, 1.0
	v_fmac_f32_e32 v176, v179, v176
	v_div_scale_f32 v179, vcc_lo, v39, v173, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v176
	v_fma_f32 v181, -v175, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v176
	v_fma_f32 v175, -v175, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v175, v175, v176, v180
	v_div_scale_f32 v176, null, v173, v173, v38
	v_div_fixup_f32 v175, v175, v173, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v176
	v_fma_f32 v180, -v176, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v38, v173, v38
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v176, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v176, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v176, v176, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v176, v176, v173, v38
	v_cvt_pk_fp8_f32 v178.h, v175, v176
	v_div_scale_f32 v175, null, v173, v173, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v176, v175
	v_fma_f32 v179, -v175, v176, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v179, v176
	v_div_scale_f32 v179, vcc_lo, v37, v173, v37
	v_mul_f32_e32 v180, v179, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v175, v180, v179
	v_fmac_f32_e32 v180, v181, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v175, -v175, v180, v179
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v175, v175, v176, v180
	v_div_scale_f32 v176, null, v173, v173, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v175, v175, v173, v37
	v_rcp_f32_e32 v179, v176
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v176, v179, 1.0
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v36, v173, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v181, v180, v179
	v_fma_f32 v182, -v176, v181, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v182, v179
	v_fma_f32 v176, -v176, v181, v180
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v176, v176, v179, v181
	v_div_fixup_f32 v176, v176, v173, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_fp8_f32 v177.l, v175, v176
	v_div_scale_f32 v175, null, v173, v173, v35
	v_rcp_f32_e32 v176, v175
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v175, v176, 1.0
	v_fmac_f32_e32 v176, v179, v176
	v_div_scale_f32 v179, vcc_lo, v35, v173, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v176
	v_fma_f32 v181, -v175, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v176
	v_fma_f32 v175, -v175, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v175, v175, v176, v180
	v_div_scale_f32 v176, null, v173, v173, v34
	v_div_fixup_f32 v175, v175, v173, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v176
	v_fma_f32 v180, -v176, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v34, v173, v34
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v176, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v176, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v176, v176, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v173, v176, v173, v34
	v_cvt_pk_fp8_f32 v177.h, v175, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v173, v178, v177
	s_wait_dscnt 0x0
	v_add_f32_e32 v178, v171, v172
	v_cvt_f32_ubyte0_e32 v173, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v171, null, v178, v178, v170
	v_fmaak_f32 v173, s8, v173, 0x3f7fbe77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v172, v171
	v_dual_mul_f32 v101, v101, v173 :: v_dual_mul_f32 v88, v88, v173
	v_dual_mul_f32 v100, v100, v173 :: v_dual_mul_f32 v99, v99, v173
	v_mul_f32_e32 v84, v84, v173
	v_dual_mul_f32 v98, v98, v173 :: v_dual_mul_f32 v97, v97, v173
	v_mul_f32_e32 v80, v80, v173
	v_dual_mul_f32 v96, v96, v173 :: v_dual_mul_f32 v95, v95, v173
	v_mul_f32_e32 v78, v78, v173
	v_dual_mul_f32 v94, v94, v173 :: v_dual_mul_f32 v93, v93, v173
	v_mul_f32_e32 v76, v76, v173
	v_dual_mul_f32 v92, v92, v173 :: v_dual_mul_f32 v91, v91, v173
	v_mul_f32_e32 v74, v74, v173
	v_dual_mul_f32 v90, v90, v173 :: v_dual_mul_f32 v89, v89, v173
	v_dual_mul_f32 v72, v72, v173 :: v_dual_mul_f32 v87, v87, v173
	v_mul_f32_e32 v70, v70, v173
	v_dual_mul_f32 v86, v86, v173 :: v_dual_mul_f32 v85, v85, v173
	v_dual_mul_f32 v68, v68, v173 :: v_dual_mul_f32 v83, v83, v173
	v_mul_f32_e32 v66, v66, v173
	v_dual_mul_f32 v82, v82, v173 :: v_dual_mul_f32 v81, v81, v173
	v_dual_mul_f32 v64, v64, v173 :: v_dual_mul_f32 v79, v79, v173
	v_dual_mul_f32 v62, v62, v173 :: v_dual_mul_f32 v77, v77, v173
	v_dual_mul_f32 v60, v60, v173 :: v_dual_mul_f32 v75, v75, v173
	v_dual_mul_f32 v58, v58, v173 :: v_dual_mul_f32 v73, v73, v173
	v_dual_mul_f32 v56, v56, v173 :: v_dual_mul_f32 v71, v71, v173
	v_dual_mul_f32 v54, v54, v173 :: v_dual_mul_f32 v69, v69, v173
	v_dual_mul_f32 v52, v52, v173 :: v_dual_mul_f32 v67, v67, v173
	v_dual_mul_f32 v50, v50, v173 :: v_dual_mul_f32 v65, v65, v173
	v_dual_mul_f32 v48, v48, v173 :: v_dual_mul_f32 v63, v63, v173
	v_dual_mul_f32 v46, v46, v173 :: v_dual_mul_f32 v61, v61, v173
	v_dual_mul_f32 v44, v44, v173 :: v_dual_mul_f32 v59, v59, v173
	v_dual_mul_f32 v42, v42, v173 :: v_dual_mul_f32 v57, v57, v173
	v_dual_mul_f32 v168, v168, v173 :: v_dual_mul_f32 v55, v55, v173
	v_dual_mul_f32 v166, v166, v173 :: v_dual_mul_f32 v53, v53, v173
	v_dual_mul_f32 v164, v164, v173 :: v_dual_mul_f32 v51, v51, v173
	v_dual_mul_f32 v162, v162, v173 :: v_dual_mul_f32 v49, v49, v173
	v_dual_mul_f32 v160, v160, v173 :: v_dual_mul_f32 v47, v47, v173
	v_dual_mul_f32 v158, v158, v173 :: v_dual_mul_f32 v45, v45, v173
	v_dual_mul_f32 v156, v156, v173 :: v_dual_mul_f32 v43, v43, v173
	v_dual_mul_f32 v154, v154, v173 :: v_dual_mul_f32 v169, v169, v173
	v_dual_mul_f32 v152, v152, v173 :: v_dual_mul_f32 v167, v167, v173
	v_dual_mul_f32 v150, v150, v173 :: v_dual_mul_f32 v165, v165, v173
	v_dual_mul_f32 v148, v148, v173 :: v_dual_mul_f32 v163, v163, v173
	v_dual_mul_f32 v146, v146, v173 :: v_dual_mul_f32 v161, v161, v173
	v_dual_mul_f32 v144, v144, v173 :: v_dual_mul_f32 v159, v159, v173
	v_dual_mul_f32 v142, v142, v173 :: v_dual_mul_f32 v157, v157, v173
	v_dual_mul_f32 v140, v140, v173 :: v_dual_mul_f32 v155, v155, v173
	v_dual_mul_f32 v138, v138, v173 :: v_dual_mul_f32 v153, v153, v173
	v_dual_mul_f32 v136, v136, v173 :: v_dual_mul_f32 v151, v151, v173
	v_dual_mul_f32 v134, v134, v173 :: v_dual_mul_f32 v149, v149, v173
	v_dual_mul_f32 v147, v147, v173 :: v_dual_mul_f32 v130, v173, v130
	v_mul_f32_e32 v145, v145, v173
	v_dual_mul_f32 v143, v143, v173 :: v_dual_mul_f32 v128, v173, v128
	v_mul_f32_e32 v141, v141, v173
	v_dual_mul_f32 v139, v139, v173 :: v_dual_mul_f32 v126, v173, v126
	v_mul_f32_e32 v137, v137, v173
	v_dual_mul_f32 v135, v135, v173 :: v_dual_mul_f32 v124, v173, v124
	v_dual_mul_f32 v133, v173, v133 :: v_dual_mul_f32 v122, v173, v122
	v_dual_mul_f32 v132, v173, v132 :: v_dual_mul_f32 v131, v173, v131
	v_dual_mul_f32 v120, v173, v120 :: v_dual_mul_f32 v129, v173, v129
	v_dual_mul_f32 v118, v173, v118 :: v_dual_mul_f32 v127, v173, v127
	v_dual_mul_f32 v116, v173, v116 :: v_dual_mul_f32 v125, v173, v125
	v_dual_mul_f32 v114, v173, v114 :: v_dual_mul_f32 v123, v173, v123
	v_dual_mul_f32 v112, v173, v112 :: v_dual_mul_f32 v121, v173, v121
	v_dual_mul_f32 v110, v173, v110 :: v_dual_mul_f32 v119, v173, v119
	v_dual_mul_f32 v108, v173, v108 :: v_dual_mul_f32 v117, v173, v117
	v_dual_mul_f32 v106, v173, v106 :: v_dual_mul_f32 v115, v173, v115
	v_dual_mul_f32 v104, v173, v104 :: v_dual_mul_f32 v113, v173, v113
	v_dual_mul_f32 v102, v173, v102 :: v_dual_mul_f32 v111, v173, v111
	v_mul_f32_e32 v109, v173, v109
	v_mul_f32_e32 v107, v173, v107
	v_mul_f32_e32 v105, v173, v105
	v_mul_f32_e32 v103, v173, v103
	v_fma_f32 v173, -v171, v172, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v170, v178, v170
	v_mul_f32_e32 v175, v173, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v171, v175, v173
	v_fmac_f32_e32 v175, v176, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v171, v175, v173
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v171, v171, v172, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v170, v171, v178, v170
	v_div_scale_f32 v171, null, v178, v178, v40
	v_add_f32_e32 v170, 0, v170
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v172, v171
	v_fma_f32 v173, -v171, v172, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v40, v178, v40
	v_mul_f32_e32 v175, v173, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v171, v175, v173
	v_fmac_f32_e32 v175, v176, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v171, v175, v173
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v171, v171, v172, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v40, v171, v178, v40
	v_add_f32_e32 v171, 0x3c23d70a, v40
	v_div_scale_f32 v40, null, v178, v178, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v172, v40
	v_fma_f32 v173, -v40, v172, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v172, v173, v172
	v_div_scale_f32 v173, vcc_lo, v39, v178, v39
	v_mul_f32_e32 v175, v173, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v40, v175, v173
	v_fmac_f32_e32 v175, v176, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v40, v175, v173
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v40, v40, v172, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v39, v40, v178, v39
	v_add_f32_e32 v172, 0x3ca3d70a, v39
	v_div_scale_f32 v39, null, v178, v178, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v39
	v_fma_f32 v173, -v39, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v173, v40
	v_div_scale_f32 v173, vcc_lo, v38, v178, v38
	v_mul_f32_e32 v175, v173, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v39, v175, v173
	v_fmac_f32_e32 v175, v176, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v39, -v39, v175, v173
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v39, v40, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v38, v39, v178, v38
	v_add_f32_e32 v173, 0x3cf5c28f, v38
	v_div_scale_f32 v38, null, v178, v178, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v39, v38
	v_fma_f32 v40, -v38, v39, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v39
	v_div_scale_f32 v40, vcc_lo, v37, v178, v37
	v_mul_f32_e32 v175, v40, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v38, v175, v40
	v_fmac_f32_e32 v175, v176, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v38, v175, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v38, v39, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v37, v38, v178, v37
	v_add_f32_e32 v175, 0x3d23d70a, v37
	v_div_scale_f32 v37, null, v178, v178, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v38, v37
	v_fma_f32 v39, -v37, v38, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v38
	v_div_scale_f32 v39, vcc_lo, v36, v178, v36
	v_mul_f32_e32 v40, v39, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v37, v40, v39
	v_fmac_f32_e32 v40, v176, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v37, v40, v39
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v37, v38, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v36, v37, v178, v36
	v_add_f32_e32 v176, 0x3d4ccccc, v36
	v_div_scale_f32 v36, null, v178, v178, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v37, v36
	v_fma_f32 v38, -v36, v37, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v37
	v_div_scale_f32 v38, vcc_lo, v35, v178, v35
	v_mul_f32_e32 v39, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v36, v39, v38
	v_fmac_f32_e32 v39, v40, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v36, v39, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v36, v36, v37, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v35, v36, v178, v35
	v_add_f32_e32 v177, 0x3d75c28f, v35
	v_div_scale_f32 v35, null, v178, v178, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v36, v35
	v_fma_f32 v37, -v35, v36, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v34, v178, v34
	v_mul_f32_e32 v38, v37, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v39, -v35, v38, v37
	v_fmac_f32_e32 v38, v39, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v35, -v35, v38, v37
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v35, v35, v36, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v34, v35, v178, v34
	v_add_f32_e32 v178, 0x3d8f5c29, v34
	s_cbranch_scc0 .LBB11_4
	s_branch .LBB11_8
.LBB11_7:
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v101, 1.0 :: v_dual_mov_b32 v100, 0x3f8020c5
	v_dual_mov_b32 v99, 0x3f804189 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v98, 0x3f80624e :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v97, 0x3f808312 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v96, 0x3f80a3d7 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v95, 0x3f80c49c :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v94, 0x3f80e560 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v93, 0x3f8147ae :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v92, 0x3f816873 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v91, 0x3f818937 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v90, 0x3f81a9fc :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v89, 0x3f81cac0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v88, 0x3f81eb85 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v87, 0x3f820c4a :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v86, 0x3f822d0e :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v85, 0x3f828f5c :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v84, 0x3f82b021 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v83, 0x3f82d0e5 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v82, 0x3f82f1aa :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v81, 0x3f83126e :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v80, 0x3f833333 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v79, 0x3f8353f8 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v78, 0x3f8374bc :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v77, 0x3f83d70a :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v76, 0x3f83f7cf :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v75, 0x3f841893 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v74, 0x3f843958 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v73, 0x3f845a1c :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v72, 0x3f847ae1 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v71, 0x3f849ba6 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v70, 0x3f84bc6a :: v_dual_mov_b32 v31, 0
	v_mov_b32_e32 v69, 0x3f851eb8
	v_mov_b32_e32 v68, 0x3f853f7d
	v_mov_b32_e32 v67, 0x3f856041
	v_mov_b32_e32 v66, 0x3f858106
	v_mov_b32_e32 v65, 0x3f85a1ca
	v_mov_b32_e32 v64, 0x3f85c28f
	v_mov_b32_e32 v63, 0x3f85e354
	v_mov_b32_e32 v62, 0x3f860418
	v_mov_b32_e32 v61, 0x3f866666
	v_mov_b32_e32 v60, 0x3f86872b
	v_mov_b32_e32 v59, 0x3f86a7ef
	v_mov_b32_e32 v58, 0x3f86c8b4
	v_mov_b32_e32 v57, 0x3f86e978
	v_mov_b32_e32 v56, 0x3f870a3d
	v_mov_b32_e32 v55, 0x3f872b02
	v_mov_b32_e32 v54, 0x3f874bc6
	v_mov_b32_e32 v53, 0x3f87ae14
	v_mov_b32_e32 v52, 0x3f87ced9
	v_mov_b32_e32 v51, 0x3f87ef9d
	v_mov_b32_e32 v50, 0x3f881062
	v_mov_b32_e32 v49, 0x3f883126
	v_mov_b32_e32 v48, 0x3f8851eb
	v_mov_b32_e32 v47, 0x3f8872b0
	v_mov_b32_e32 v46, 0x3f889374
	v_mov_b32_e32 v45, 0x3f88f5c3
	v_mov_b32_e32 v44, 0x3f891688
	v_mov_b32_e32 v43, 0x3f89374c
	v_mov_b32_e32 v42, 0x3f895811
	v_mov_b32_e32 v169, 0x3f8978d5
	v_mov_b32_e32 v168, 0x3f89999a
	v_mov_b32_e32 v167, 0x3f89ba5f
	v_mov_b32_e32 v166, 0x3f89db23
	v_mov_b32_e32 v165, 0x3f8a3d71
	v_mov_b32_e32 v164, 0x3f8a5e36
	v_mov_b32_e32 v163, 0x3f8a7efa
	v_mov_b32_e32 v162, 0x3f8a9fbf
	v_mov_b32_e32 v161, 0x3f8ac083
	v_mov_b32_e32 v160, 0x3f8ae148
	v_mov_b32_e32 v159, 0x3f8b020d
	v_mov_b32_e32 v158, 0x3f8b22d1
	v_mov_b32_e32 v157, 0x3f8b851f
	v_mov_b32_e32 v156, 0x3f8ba5e4
	v_mov_b32_e32 v155, 0x3f8bc6a8
	v_mov_b32_e32 v154, 0x3f8be76d
	v_mov_b32_e32 v153, 0x3f8c0831
	v_mov_b32_e32 v152, 0x3f8c28f6
	v_mov_b32_e32 v151, 0x3f8c49bb
	v_mov_b32_e32 v150, 0x3f8c6a7f
	v_mov_b32_e32 v149, 0x3f8ccccd
	v_mov_b32_e32 v148, 0x3f8ced92
	v_mov_b32_e32 v147, 0x3f8d0e56
	v_mov_b32_e32 v146, 0x3f8d2f1b
	v_mov_b32_e32 v145, 0x3f8d4fdf
	v_mov_b32_e32 v144, 0x3f8d70a4
	v_mov_b32_e32 v143, 0x3f8d9169
	v_mov_b32_e32 v142, 0x3f8db22d
	v_mov_b32_e32 v141, 0x3f8e147b
	v_mov_b32_e32 v140, 0x3f8e3540
	v_mov_b32_e32 v139, 0x3f8e5604
	v_mov_b32_e32 v138, 0x3f8e76c9
	v_mov_b32_e32 v137, 0x3f8e978d
	v_mov_b32_e32 v136, 0x3f8eb852
	v_mov_b32_e32 v135, 0x3f8ed917
	v_mov_b32_e32 v134, 0x3f8ef9db
	v_mov_b32_e32 v133, 0x3f8f5c29
	v_mov_b32_e32 v132, 0x3f8f7cee
	v_mov_b32_e32 v131, 0x3f8f9db2
	v_mov_b32_e32 v130, 0x3f8fbe77
	v_mov_b32_e32 v129, 0x3f8fdf3b
	v_mov_b32_e32 v128, 0x3f900000
	v_mov_b32_e32 v127, 0x3f9020c5
	v_mov_b32_e32 v126, 0x3f904189
	v_mov_b32_e32 v125, 0x3f90a3d7
	v_mov_b32_e32 v124, 0x3f90c49c
	v_mov_b32_e32 v123, 0x3f90e560
	v_mov_b32_e32 v122, 0x3f910625
	v_mov_b32_e32 v121, 0x3f9126e9
	v_mov_b32_e32 v120, 0x3f9147ae
	v_mov_b32_e32 v119, 0x3f916873
	v_mov_b32_e32 v118, 0x3f918937
	v_mov_b32_e32 v117, 0x3f91eb85
	v_mov_b32_e32 v116, 0x3f920c4a
	v_mov_b32_e32 v115, 0x3f922d0e
	v_mov_b32_e32 v114, 0x3f924dd3
	v_mov_b32_e32 v113, 0x3f926e97
	v_mov_b32_e32 v112, 0x3f928f5c
	v_mov_b32_e32 v111, 0x3f92b021
	v_mov_b32_e32 v110, 0x3f92d0e5
	v_mov_b32_e32 v109, 0x3f933333
	v_mov_b32_e32 v108, 0x3f9353f8
	v_mov_b32_e32 v107, 0x3f9374bc
	v_mov_b32_e32 v106, 0x3f939581
	v_mov_b32_e32 v105, 0x3f93b645
	v_mov_b32_e32 v104, 0x3f93d70a
	v_mov_b32_e32 v103, 0x3f93f7cf
	v_mov_b32_e32 v102, 0x3f941893
.LBB11_8:                               ; %Flow
	v_add_f32_e32 v25, 0, v25
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	ds_load_b32 v2, v41
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v101
	v_add_f32_e32 v1, v1, v100
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v99
	v_add_f32_e32 v1, v1, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v97
	v_add_f32_e32 v1, v1, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v95
	v_add_f32_e32 v1, v1, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v93
	v_add_f32_e32 v1, v1, v92
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v91
	v_add_f32_e32 v1, v1, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v89
	v_add_f32_e32 v1, v1, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v87
	v_add_f32_e32 v1, v1, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v85
	v_add_f32_e32 v1, v1, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v83
	v_add_f32_e32 v1, v1, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v81
	v_add_f32_e32 v1, v1, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v79
	v_add_f32_e32 v1, v1, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v77
	v_add_f32_e32 v1, v1, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v75
	v_add_f32_e32 v1, v1, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v73
	v_add_f32_e32 v1, v1, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v71
	v_add_f32_e32 v1, v1, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v69
	v_add_f32_e32 v1, v1, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v67
	v_add_f32_e32 v1, v1, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v65
	v_add_f32_e32 v1, v1, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v63
	v_add_f32_e32 v1, v1, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v61
	v_add_f32_e32 v1, v1, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v59
	v_add_f32_e32 v1, v1, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v57
	v_add_f32_e32 v1, v1, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v55
	v_add_f32_e32 v1, v1, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v53
	v_add_f32_e32 v1, v1, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v51
	v_add_f32_e32 v1, v1, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v49
	v_add_f32_e32 v1, v1, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v47
	v_add_f32_e32 v1, v1, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v45
	v_add_f32_e32 v1, v1, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v43
	v_add_f32_e32 v1, v1, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v169
	v_add_f32_e32 v1, v1, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v167
	v_add_f32_e32 v1, v1, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v165
	v_add_f32_e32 v1, v1, v164
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v163
	v_add_f32_e32 v1, v1, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v161
	v_add_f32_e32 v1, v1, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v159
	v_add_f32_e32 v1, v1, v158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v157
	v_add_f32_e32 v1, v1, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v155
	v_add_f32_e32 v1, v1, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v153
	v_add_f32_e32 v1, v1, v152
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v151
	v_add_f32_e32 v1, v1, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v149
	v_add_f32_e32 v1, v1, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v147
	v_add_f32_e32 v1, v1, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v145
	v_add_f32_e32 v1, v1, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v143
	v_add_f32_e32 v1, v1, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v141
	v_add_f32_e32 v1, v1, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v139
	v_add_f32_e32 v1, v1, v138
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v137
	v_add_f32_e32 v1, v1, v136
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v135
	v_add_f32_e32 v1, v1, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v133
	v_add_f32_e32 v1, v1, v132
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v131
	v_add_f32_e32 v1, v1, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v129
	v_add_f32_e32 v1, v1, v128
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v127
	v_add_f32_e32 v1, v1, v126
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v125
	v_add_f32_e32 v1, v1, v124
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v123
	v_add_f32_e32 v1, v1, v122
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v121
	v_add_f32_e32 v1, v1, v120
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v119
	v_add_f32_e32 v1, v1, v118
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v117
	v_add_f32_e32 v1, v1, v116
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v115
	v_add_f32_e32 v1, v1, v114
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v113
	v_add_f32_e32 v1, v1, v112
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v111
	v_add_f32_e32 v1, v1, v110
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v109
	v_add_f32_e32 v1, v1, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v107
	v_add_f32_e32 v1, v1, v106
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v105
	v_add_f32_e32 v1, v1, v104
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v103
	v_add_f32_e32 v1, v1, v102
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v170
	v_add_f32_e32 v1, v1, v171
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v172
	v_add_f32_e32 v1, v1, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v175
	v_add_f32_e32 v1, v1, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v1, v177
	v_mov_b32_e32 v1, 0
	v_add_f32_e32 v3, v3, v178
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fmamk_f32 v2, v2, 0x33800000, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end11:
	.size	_Z3runILi5ELi4EEvPfPji, .Lfunc_end11-_Z3runILi5ELi4EEvPfPji
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z3runILi5ELi4EEvPfPji
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 183
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end11-_Z3runILi5ELi4EEvPfPji)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z3runILi5ELi4EEvPfPji,"axG",@progbits,_Z3runILi5ELi4EEvPfPji,comdat
                                        ; -- End function
	.set .L_Z3runILi5ELi4EEvPfPji.num_vgpr, 183
	.set .L_Z3runILi5ELi4EEvPfPji.num_agpr, 0
	.set .L_Z3runILi5ELi4EEvPfPji.numbered_sgpr, 10
	.set .L_Z3runILi5ELi4EEvPfPji.num_named_barrier, 0
	.set .L_Z3runILi5ELi4EEvPfPji.private_seg_size, 0
	.set .L_Z3runILi5ELi4EEvPfPji.uses_vcc, 1
	.set .L_Z3runILi5ELi4EEvPfPji.uses_flat_scratch, 0
	.set .L_Z3runILi5ELi4EEvPfPji.has_dyn_sized_stack, 0
	.set .L_Z3runILi5ELi4EEvPfPji.has_recursion, 0
	.set .L_Z3runILi5ELi4EEvPfPji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6580
; TotalNumSgprs: 12
; NumVgprs: 183
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 183
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_7997418e9332a5c1,@object ; @__hip_cuid_7997418e9332a5c1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7997418e9332a5c1
__hip_cuid_7997418e9332a5c1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7997418e9332a5c1, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7997418e9332a5c1
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi0ELi1EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi0ELi1EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     15
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi1ELi1EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi1ELi1EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     161
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi2ELi1EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi2ELi1EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     161
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi3ELi1EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi3ELi1EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     162
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi4ELi1EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi4ELi1EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     161
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi5ELi1EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi5ELi1EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     161
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi0ELi4EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi0ELi4EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     39
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi1ELi4EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi1ELi4EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     161
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi2ELi4EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi2ELi4EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     183
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi3ELi4EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi3ELi4EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     187
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi4ELi4EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi4ELi4EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     183
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z3runILi5ELi4EEvPfPji
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z3runILi5ELi4EEvPfPji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     183
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
