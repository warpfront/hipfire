	.att_syntax
	.file	"bench_foldfree.hip"
	.text
	.globl	__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201 # -- Begin function __device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.prefalign	4, .Lfunc_end0, nop
	.type	__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201,@function
__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201: # @__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%r9, 48(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	192(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movq	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end0:
	.size	__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201, .Lfunc_end0-__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function main
.LCPI1_0:
	.long	0x3f800000                      # float 1
.LCPI1_1:
	.long	0x447a0000                      # float 1000
.LCPI1_6:
	.long	0x7f800000                      # float +Inf
.LCPI1_7:
	.long	0x7fc00000                      # float NaN
.LCPI1_8:
	.long	0x3e000000                      # float 0.125
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI1_2:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI1_3:
	.quad	0x412e848000000000              # double 1.0E+6
.LCPI1_4:
	.long	0x3f800000                      # float 1
	.long	0x3fc00000                      # float 1.5
.LCPI1_11:
	.quad	0x01a56e1fc2f8f359              # double 1.0E-300
.LCPI1_12:
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI1_5:
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
.LCPI1_9:
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
.LCPI1_10:
	.quad	0x7fffffffffffffff              # double NaN
	.quad	0x7fffffffffffffff              # double NaN
	.text
	.globl	main
	.prefalign	4, .Lfunc_end1, nop
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$504, %rsp                      # imm = 0x1F8
	.cfi_def_cfa_offset 560
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	cmpl	$2, %edi
	jl	.LBB1_7
# %bb.1:
	movq	%rsi, %rbx
	movq	8(%rsi), %r14
	leaq	160(%rsp), %r12
	movq	%r12, 144(%rsp)
	testq	%r14, %r14
	je	.LBB1_414
# %bb.2:
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	strlen@PLT
	movq	%rax, %r15
	movq	%rax, 112(%rsp)
	movq	%r12, %rax
	cmpq	$16, %r15
	jb	.LBB1_4
# %bb.3:                                # %.noexc.i
	.cfi_escape 0x2e, 0x00
	leaq	144(%rsp), %rdi
	leaq	112(%rsp), %rsi
	xorl	%edx, %edx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@PLT
	movq	%rax, 144(%rsp)
	movq	112(%rsp), %rcx
	movq	%rcx, 160(%rsp)
.LBB1_4:                                # %._crit_edge.i.i
	testq	%r15, %r15
	je	.LBB1_198
# %bb.5:                                # %._crit_edge.i.i
	cmpq	$1, %r15
	jne	.LBB1_197
# %bb.6:
	movzbl	(%r14), %ecx
	movb	%cl, (%rax)
	jmp	.LBB1_198
.LBB1_7:                                # %.critedge362.thread
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rdi
	xorl	%esi, %esi
	xorl	%eax, %eax
	callq	open@PLT
	testl	%eax, %eax
	js	.LBB1_350
# %bb.8:                                # %.thread604
	movl	%eax, %r15d
.LBB1_9:
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4, %edi
	callq	_Znwm@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.10:                               # %.critedge364
	movq	%rax, 472(%rsp)
	movq	%rax, %rcx
	addq	$4, %rcx
	movq	%rcx, 488(%rsp)
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movl	$5120, (%rax)                   # imm = 0x1400
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	movq	%rcx, 480(%rsp)
	movl	$8192, %ebx                     # imm = 0x2000
	movl	$5120, %eax                     # imm = 0x1400
	movl	$0, 224(%rsp)                   # 4-byte Folded Spill
.LBB1_11:
	pxor	%xmm0, %xmm0
	movdqa	%xmm0, 256(%rsp)
	movq	$0, 272(%rsp)
	movdqu	%xmm0, 200(%rsp)
	movdqu	%xmm0, 184(%rsp)
	movdqu	%xmm0, 168(%rsp)
	movdqu	%xmm0, 152(%rsp)
	movl	%eax, 144(%rsp)
	movl	$17408, 148(%rsp)               # imm = 0x4400
	movl	%eax, %r13d
	imulq	$9248, %r13, %r12               # imm = 0x2420
.Ltmp6:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp7:                                 # EH_LABEL
# %bb.12:                               # %.lr.ph.i.i.preheader
	movq	%rax, %r14
	movabsq	$9223372036854775807, %rax      # imm = 0x7FFFFFFFFFFFFFFF
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	movl	%ebx, %eax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movb	$0, (%r14)
	leaq	-1(%r12), %rdx
	leaq	1(%r14), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r14, 152(%rsp)
	movq	%r14, %rax
	addq	%r12, %rax
	movq	%rax, 160(%rsp)
	movq	%rax, 168(%rsp)
	movl	$2102366144, %ebp               # imm = 0x7D4F8FC0
	.p2align	4
.LBB1_13:                               # %.lr.ph.i.i
                                        # =>This Inner Loop Header: Depth=1
.Ltmp8:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%r15d, %edi
	movq	%r14, %rsi
	movq	%r12, %rdx
	movq	%rbp, %rcx
	callq	pread@PLT
.Ltmp9:                                 # EH_LABEL
# %bb.14:                               # %.noexc.i396
                                        #   in Loop: Header=BB1_13 Depth=1
	testq	%rax, %rax
	jle	.LBB1_354
# %bb.15:                               #   in Loop: Header=BB1_13 Depth=1
	addq	%rax, %r14
	addq	%rax, %rbp
	subq	%rax, %r12
	jne	.LBB1_13
# %bb.16:                               # %_ZL9pread_alliPvmm.exit.i
	movq	176(%rsp), %rdx
	movq	184(%rsp), %r14
	movq	%r14, %r12
	subq	%rdx, %r12
	movq	%r12, %rax
	sarq	$2, %rax
	movq	%r13, %rbx
	subq	%rax, %rbx
	jbe	.LBB1_26
# %bb.17:
	movq	192(%rsp), %rsi
	movq	%rsi, %rcx
	subq	%r14, %rcx
	sarq	$2, %rcx
	cmpq	%rbx, %rcx
	leaq	200(%rsp), %rbp
	jae	.LBB1_29
# %bb.18:
	movq	%rsi, 80(%rsp)                  # 8-byte Spill
	movq	%rdx, %rbp
	cmpq	%rbx, %rax
	movq	%rbx, %r14
	cmovaq	%rax, %r14
	addq	%rax, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp11:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp12:                                # EH_LABEL
# %bb.19:                               # %.noexc519
	movl	%r15d, 56(%rsp)                 # 4-byte Spill
	leaq	(%rax,%r12), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rax, %r15
	movl	$0, (%rax,%r12)
	movq	%rbx, %rdx
	decq	%rdx
	je	.LBB1_21
# %bb.20:                               # %_ZSt6fill_nIPfmfET_S1_T0_RKT1_.exit.loopexit.i.i.i29.i
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	4(%rax), %rdi
	shlq	$2, %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
.LBB1_21:                               # %_ZSt27__uninitialized_default_n_aIPfmfET_S1_T0_RSaIT1_E.exit32.i
	testq	%r12, %r12
	jle	.LBB1_23
# %bb.22:
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%rbp, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
.LBB1_23:                               # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i
	testq	%rbp, %rbp
	je	.LBB1_25
# %bb.24:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i
	movq	80(%rsp), %rsi                  # 8-byte Reload
	subq	%rbp, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	callq	_ZdlPvm@PLT
.LBB1_25:                               # %_ZNSt6vectorIfSaIfEE12_Guard_allocD2Ev.exit.i
	movq	%r15, 176(%rsp)
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%rbx,4), %rax
	movq	%rax, 184(%rsp)
	leaq	(%r15,%r14,4), %rax
	movq	%rax, 192(%rsp)
	movl	56(%rsp), %r15d                 # 4-byte Reload
	leaq	200(%rsp), %rbp
	jmp	.LBB1_32
.LBB1_26:
	leaq	200(%rsp), %rbp
	jae	.LBB1_32
# %bb.27:
	leaq	(%rdx,%r13,4), %rax
	cmpq	%rax, %r14
	je	.LBB1_32
# %bb.28:
	movq	%rax, 184(%rsp)
	jmp	.LBB1_32
.LBB1_29:
	movl	$0, (%r14)
	addq	$4, %r14
	decq	%rbx
	je	.LBB1_31
# %bb.30:                               # %_ZSt6fill_nIPfmfET_S1_T0_RKT1_.exit.loopexit.i.i.i.i
	leaq	(,%rbx,4), %rdx
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	(%r14,%rbx,4), %r14
.LBB1_31:                               # %_ZSt27__uninitialized_default_n_aIPfmfET_S1_T0_RSaIT1_E.exit.i
	movq	%r14, 184(%rsp)
.LBB1_32:                               # %.preheader.lr.ph.i
	movq	152(%rsp), %rbx
	movq	176(%rsp), %r14
	addq	$5, %rbx
	xorl	%r12d, %r12d
	.p2align	4
.LBB1_33:                               # %.preheader.i400
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_34 Depth 2
	xorl	%eax, %eax
	xorl	%edi, %edi
	.p2align	4
.LBB1_34:                               #   Parent Loop BB1_33 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movzbl	-5(%rbx,%rax), %r8d
	movzbl	-4(%rbx,%rax), %ecx
	movl	%ecx, %edx
	shll	$8, %edx
	orl	%edx, %r8d
	testb	%cl, %cl
	js	.LBB1_352
# %bb.35:                               #   in Loop: Header=BB1_34 Depth=2
	shrl	$2, %ecx
	andl	$31, %ecx
	leal	-1(%rcx), %esi
	cmpl	$29, %esi
	ja	.LBB1_352
# %bb.36:                               #   in Loop: Header=BB1_34 Depth=2
	movl	%r8d, %esi
	andl	$1023, %esi                     # imm = 0x3FF
	andl	$512, %edx                      # imm = 0x200
	cmpl	%esi, %edx
	jne	.LBB1_352
# %bb.37:                               #   in Loop: Header=BB1_34 Depth=2
	cmpl	%ecx, %edi
	cmoval	%edi, %ecx
	movzbl	-1(%rbx,%rax), %r8d
	movzbl	(%rbx,%rax), %edi
	movl	%edi, %edx
	shll	$8, %edx
	orl	%edx, %r8d
	testb	%dil, %dil
	js	.LBB1_352
# %bb.38:                               #   in Loop: Header=BB1_34 Depth=2
	shrl	$2, %edi
	andl	$31, %edi
	leal	-1(%rdi), %esi
	cmpl	$29, %esi
	ja	.LBB1_352
# %bb.39:                               #   in Loop: Header=BB1_34 Depth=2
	movl	%r8d, %esi
	andl	$1023, %esi                     # imm = 0x3FF
	andl	$512, %edx                      # imm = 0x200
	cmpl	%esi, %edx
	jne	.LBB1_352
# %bb.40:                               #   in Loop: Header=BB1_34 Depth=2
	cmpl	%edi, %ecx
	cmoval	%ecx, %edi
	addq	$136, %rax
	cmpq	$9248, %rax                     # imm = 0x2420
	jne	.LBB1_34
# %bb.41:                               #   in Loop: Header=BB1_33 Depth=1
	addl	$-15, %edi
	.cfi_escape 0x2e, 0x00
	movd	.LCPI1_0(%rip), %xmm0           # xmm0 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
                                        # kill: def $edi killed $edi killed $rdi
	callq	ldexpf@PLT
	movd	%xmm0, (%r14,%r12,4)
	incq	%r12
	addq	$9248, %rbx                     # imm = 0x2420
	cmpq	%r13, %r12
	jne	.LBB1_33
# %bb.42:                               # %._crit_edge.i
.Ltmp14:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	leaq	152(%rsp), %rbx
	movq	%rbx, %rsi
	movl	%r13d, %edx
	movl	$17408, %ecx                    # imm = 0x4400
	callq	_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii
.Ltmp15:                                # EH_LABEL
# %bb.43:
	movq	112(%rsp), %r13
	movq	120(%rsp), %r14
	subq	%r13, %r14
.Ltmp17:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	hipMalloc@PLT
.Ltmp18:                                # EH_LABEL
	movq	64(%rsp), %r12                  # 8-byte Reload
# %bb.44:                               # %_ZL9hipMallocIcE10hipError_tPPT_m.exit.i
	testl	%eax, %eax
	jne	.LBB1_390
# %bb.45:
	movq	184(%rsp), %rsi
	subq	176(%rsp), %rsi
.Ltmp23:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	208(%rsp), %rdi
	callq	hipMalloc@PLT
.Ltmp24:                                # EH_LABEL
# %bb.46:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit.i
	testl	%eax, %eax
	jne	.LBB1_392
# %bb.47:
	movq	200(%rsp), %rdi
.Ltmp29:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rsi
	movq	%r14, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp30:                                # EH_LABEL
# %bb.48:
	testl	%eax, %eax
	jne	.LBB1_394
# %bb.49:
	movq	208(%rsp), %rdi
	movq	176(%rsp), %rsi
	movq	184(%rsp), %rdx
	subq	%rsi, %rdx
.Ltmp35:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp36:                                # EH_LABEL
# %bb.50:
	testl	%eax, %eax
	jne	.LBB1_396
# %bb.51:
	testq	%r13, %r13
	je	.LBB1_53
# %bb.52:
	movq	128(%rsp), %rsi
	subq	%r13, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
.LBB1_53:
	movq	264(%rsp), %rax
	cmpq	272(%rsp), %rax
	je	.LBB1_55
# %bb.54:                               # %_ZNSt6vectorI6WeightSaIS0_EE9push_backEOS0_.exit.thread
	movq	144(%rsp), %rcx
	movq	%rcx, (%rax)
	movups	152(%rsp), %xmm0
	movups	%xmm0, 8(%rax)
	movq	168(%rsp), %rcx
	movq	%rcx, 24(%rax)
	xorps	%xmm0, %xmm0
	movups	%xmm0, (%rbx)
	movq	$0, 16(%rbx)
	movdqu	176(%rsp), %xmm1
	movdqu	%xmm1, 32(%rax)
	movq	192(%rsp), %rcx
	movq	%rcx, 48(%rax)
	movups	%xmm0, 24(%rbx)
	movq	$0, 40(%rbx)
	movdqu	(%rbp), %xmm0
	movdqu	%xmm0, 56(%rax)
	addq	$72, 264(%rsp)
	jmp	.LBB1_58
.LBB1_55:
.Ltmp41:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	256(%rsp), %rdi
	leaq	144(%rsp), %rsi
	callq	_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_
.Ltmp42:                                # EH_LABEL
# %bb.56:                               # %_ZNSt6vectorI6WeightSaIS0_EE9push_backEOS0_.exit
	movq	176(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB1_58
# %bb.57:
	movq	192(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_58:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit.i
	movq	152(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB1_60
# %bb.59:
	movq	168(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_60:                               # %_ZN6WeightD2Ev.exit
.Ltmp44:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%r15d, %edi
	callq	close@PLT
.Ltmp45:                                # EH_LABEL
# %bb.61:
	imull	$17408, %r12d, %r15d            # imm = 0x4400
.Ltmp47:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_Znwm@PLT
.Ltmp48:                                # EH_LABEL
# %bb.62:                               # %.noexc417
	movb	$0, (%rax)
	movq	%rax, 360(%rsp)                 # 8-byte Spill
	leaq	1(%rax), %r14
	leaq	-1(%r15), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movl	$7, %eax
	leaq	.L_ZZL6make_xiiE6values.const(%rip), %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB1_63:                               # %.preheader.i413
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_64 Depth 2
	movl	%ebx, %esi
	movl	%eax, %edi
	xorl	%r8d, %r8d
	.p2align	4
.LBB1_64:                               #   Parent Loop BB1_63 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r8d, %r9d
	shrl	$5, %r9d
	leal	(%rdi,%r9), %r10d
	addl	%esi, %r9d
	andl	$15, %r9d
	movzbl	(%r9,%rcx), %r9d
	movb	%r9b, -1(%r14,%r8)
	andl	$15, %r10d
	movzbl	(%r10,%rcx), %r9d
	movb	%r9b, (%r14,%r8)
	addq	$2, %r8
	addl	$14, %edi
	addl	$14, %esi
	cmpq	$17408, %r8                     # imm = 0x4400
	jne	.LBB1_64
# %bb.65:                               #   in Loop: Header=BB1_63 Depth=1
	incq	%rdx
	addl	$13, %eax
	addq	$17408, %r14                    # imm = 0x4400
	addl	$13, %ebx
	cmpq	%r12, %rdx
	jne	.LBB1_63
# %bb.66:                               # %_ZL6make_xii.exit
.Ltmp50:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_Znwm@PLT
.Ltmp51:                                # EH_LABEL
# %bb.67:                               # %.noexc425
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movq	%rax, 376(%rsp)                 # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	movq	%r15, %rdx
	callq	memset@PLT
	movq	360(%rsp), %rax                 # 8-byte Reload
	incq	%rax
	movq	376(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB1_68:                               # %.preheader.i421
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_69 Depth 2
	movl	%ebx, %ecx
	andl	$15, %ecx
	movl	%ebx, %edx
	shrl	$4, %edx
	imulq	$1088, %rdx, %rdx               # imm = 0x440
	xorl	%esi, %esi
	xorl	%edi, %edi
	.p2align	4
.LBB1_69:                               #   Parent Loop BB1_68 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movzbl	-1(%rax,%rdi), %r8d
	movl	%esi, %r9d
	andl	$16, %r9d
	orl	%ecx, %r9d
	movl	%edi, %r10d
	shrl	$4, %r10d
	addq	%rdx, %r10
	shlq	$8, %r10
	movl	%edi, %r11d
	andl	$6, %r11d
	addq	%r14, %r10
	leaq	(%r10,%r9,8), %r9
	movb	%r8b, (%r11,%r9)
	leal	1(%rdi), %r8d
	movzbl	(%rax,%rdi), %r10d
	andl	$7, %r8d
	movb	%r10b, (%r8,%r9)
	addq	$2, %rdi
	addq	$4, %rsi
	cmpq	$17408, %rdi                    # imm = 0x4400
	jne	.LBB1_69
# %bb.70:                               #   in Loop: Header=BB1_68 Depth=1
	incq	%rbx
	addq	$17408, %rax                    # imm = 0x4400
	cmpq	%r12, %rbx
	jne	.LBB1_68
# %bb.71:                               # %_ZL6tile_xRKSt6vectorIhSaIhEEii.exit
	leal	(,%r12,4), %ebx
.Ltmp53:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Znwm@PLT
.Ltmp54:                                # EH_LABEL
# %bb.72:
	movq	%rax, %r12
	movl	$0, (%rax)
	movq	%rax, %rdi
	addq	$4, %rdi
	movq	%rbx, 352(%rsp)                 # 8-byte Spill
	leaq	-4(%rbx), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	xorl	%esi, %esi
	callq	memset@PLT
	movl	$-2, %ebp
	movl	$3435973837, %r14d              # imm = 0xCCCCCCCD
	movq	64(%rsp), %r13                  # 8-byte Reload
	.p2align	4
.LBB1_73:                               # =>This Inner Loop Header: Depth=1
	movl	%ebx, %eax
	imulq	%r14, %rax
	shrq	$34, %rax
	leal	(%rax,%rax,4), %eax
	movl	%ebp, %edi
	subl	%eax, %edi
	.cfi_escape 0x2e, 0x00
	movd	.LCPI1_0(%rip), %xmm0           # xmm0 = [1.0E+0,0.0E+0,0.0E+0,0.0E+0]
	callq	ldexpf@PLT
	movd	%xmm0, (%r12,%rbx,4)
	incq	%rbx
	incl	%ebp
	cmpq	%rbx, %r13
	jne	.LBB1_73
# %bb.74:
	movq	%r12, 384(%rsp)                 # 8-byte Spill
	movq	$0, 104(%rsp)
	movq	$0, 96(%rsp)
	movq	$0, 88(%rsp)
.Ltmp56:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	104(%rsp), %rdi
	movq	%r15, %rsi
	callq	hipMalloc@PLT
.Ltmp57:                                # EH_LABEL
# %bb.75:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit
	testl	%eax, %eax
	movq	16(%rsp), %rbx                  # 8-byte Reload
	movq	64(%rsp), %rcx                  # 8-byte Reload
	jne	.LBB1_398
# %bb.76:
	imulq	$544, %rcx, %r14                # imm = 0x220
.Ltmp61:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	movq	%r14, %rsi
	callq	hipMalloc@PLT
.Ltmp62:                                # EH_LABEL
# %bb.77:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit
	testl	%eax, %eax
	jne	.LBB1_400
# %bb.78:
.Ltmp66:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	movq	352(%rsp), %rsi                 # 8-byte Reload
	callq	hipMalloc@PLT
.Ltmp67:                                # EH_LABEL
# %bb.79:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit431
	testl	%eax, %eax
	jne	.LBB1_402
# %bb.80:
	movq	104(%rsp), %rdi
.Ltmp71:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	376(%rsp), %rsi                 # 8-byte Reload
	movq	%r15, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp72:                                # EH_LABEL
# %bb.81:
	testl	%eax, %eax
	jne	.LBB1_404
# %bb.82:
	movq	96(%rsp), %rdi
.Ltmp76:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	hipMemset@PLT
.Ltmp77:                                # EH_LABEL
# %bb.83:
	testl	%eax, %eax
	jne	.LBB1_406
# %bb.84:
	movq	88(%rsp), %rdi
.Ltmp81:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	384(%rsp), %rsi                 # 8-byte Reload
	movq	352(%rsp), %rdx                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp82:                                # EH_LABEL
# %bb.85:
	testl	%eax, %eax
	jne	.LBB1_408
# %bb.86:
	movq	%rbx, %rcx
	movq	8(%rsp), %r14                   # 8-byte Reload
	subq	%r14, %rbx
	movq	%rbx, %rax
	sarq	$2, %rax
	movq	%rax, 368(%rsp)                 # 8-byte Spill
	shrq	$60, %rax
	jne	.LBB1_410
# %bb.87:                               # %_ZNSt6vectorIPfSaIS0_EE17_S_check_init_lenEmRKS1_.exit.i
	subq	%r14, %rcx
	movq	%rcx, 456(%rsp)                 # 8-byte Spill
	movq	%r15, 24(%rsp)                  # 8-byte Spill
	je	.LBB1_103
# %bb.88:                               # %.lr.ph.preheader.i.i.i.i.i
	leaq	(%rbx,%rbx), %r12
.Ltmp86:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp87:                                # EH_LABEL
# %bb.89:                               # %.noexc435
	movq	%rax, %r13
	movq	%rax, 144(%rsp)
	movq	368(%rsp), %r14                 # 8-byte Reload
	leaq	(%rax,%r14,8), %rax
	movq	%rax, 160(%rsp)
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
	movq	%r13, 56(%rsp)                  # 8-byte Spill
	leaq	(,%rbx,2), %rax
	addq	%r13, %rax
	movq	%rax, 152(%rsp)
	movabsq	$384307168202282326, %rax       # imm = 0x555555555555556
	cmpq	%rax, %r14
	jae	.LBB1_415
# %bb.90:                               # %.lr.ph.preheader.i.i.i.i.i437
	leaq	(,%r14,8), %rax
	leaq	(%rax,%rax,2), %r12
.Ltmp88:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp89:                                # EH_LABEL
# %bb.91:                               # %.lr.ph
	movq	%rax, %rbp
	movq	%rax, 112(%rsp)
	leaq	(%r14,%r14,2), %rax
	movq	%r14, %rbx
	leaq	(,%rax,8), %r14
	addq	%rbp, %r14
	.cfi_escape 0x2e, 0x00
	movq	$0, 32(%rsp)                    # 8-byte Folded Spill
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
	addq	%r12, %rbp
	movq	%r14, 128(%rsp)
	movq	%rbp, 120(%rsp)
	movq	8(%rsp), %r13                   # 8-byte Reload
	movq	%rbx, %r14
	movq	56(%rsp), %rdi                  # 8-byte Reload
	.p2align	4
.LBB1_92:                               # =>This Inner Loop Header: Depth=1
	movslq	(%r13), %r12
	imulq	64(%rsp), %r12                  # 8-byte Folded Reload
	leaq	(,%r12,4), %rbp
.Ltmp90:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rdi, %rbx
	movq	%rbp, %rsi
	callq	hipMalloc@PLT
.Ltmp91:                                # EH_LABEL
# %bb.93:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit441
                                        #   in Loop: Header=BB1_92 Depth=1
	testl	%eax, %eax
	jne	.LBB1_357
# %bb.94:                               #   in Loop: Header=BB1_92 Depth=1
	movq	112(%rsp), %r15
	addq	32(%rsp), %r15                  # 8-byte Folded Reload
	movl	$1040187392, 320(%rsp)          # imm = 0x3E000000
.Ltmp96:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movq	%r12, %rsi
	leaq	320(%rsp), %rdx
	callq	_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf
.Ltmp97:                                # EH_LABEL
# %bb.95:                               # %_ZNSt6vectorIfSaIfEE6assignEmRKf.exit
                                        #   in Loop: Header=BB1_92 Depth=1
	movq	(%rbx), %rdi
	movq	(%r15), %rsi
.Ltmp99:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp100:                               # EH_LABEL
# %bb.96:                               #   in Loop: Header=BB1_92 Depth=1
	testl	%eax, %eax
	movq	24(%rsp), %r15                  # 8-byte Reload
	jne	.LBB1_355
# %bb.97:                               #   in Loop: Header=BB1_92 Depth=1
	addq	$4, %r13
	addq	$8, %rbx
	addq	$24, 32(%rsp)                   # 8-byte Folded Spill
	decq	%r14
	movq	%rbx, %rdi
	jne	.LBB1_92
# %bb.98:                               # %.preheader676
	movq	456(%rsp), %rax                 # 8-byte Reload
	leaq	-4(%rax), %rcx
	xorl	%esi, %esi
	movq	8(%rsp), %rdi                   # 8-byte Reload
	movq	%rdi, %rax
	cmpq	$28, %rcx
	movq	16(%rsp), %r8                   # 8-byte Reload
	movq	64(%rsp), %r12                  # 8-byte Reload
	jb	.LBB1_102
# %bb.99:                               # %vector.ph
	shrq	$2, %rcx
	incq	%rcx
	movabsq	$9223372036854775807, %rax      # imm = 0x7FFFFFFFFFFFFFFF
	leaq	-7(%rax), %rdx
	andq	%rcx, %rdx
	leaq	(%rdi,%rdx,4), %rax
	pxor	%xmm0, %xmm0
	xorl	%esi, %esi
	pxor	%xmm1, %xmm1
	.p2align	4
.LBB1_100:                              # %vector.body
                                        # =>This Inner Loop Header: Depth=1
	movdqu	(%rdi,%rsi,4), %xmm2
	paddd	%xmm2, %xmm1
	movdqu	16(%rdi,%rsi,4), %xmm2
	paddd	%xmm2, %xmm0
	addq	$8, %rsi
	cmpq	%rsi, %rdx
	jne	.LBB1_100
# %bb.101:                              # %middle.block
	paddd	%xmm1, %xmm0
	pshufd	$238, %xmm0, %xmm1              # xmm1 = xmm0[2,3,2,3]
	paddd	%xmm0, %xmm1
	pshufd	$85, %xmm1, %xmm0               # xmm0 = xmm1[1,1,1,1]
	paddd	%xmm1, %xmm0
	movd	%xmm0, %esi
	cmpq	%rdx, %rcx
	je	.LBB1_104
	.p2align	4
.LBB1_102:                              # %.lr.ph946
                                        # =>This Inner Loop Header: Depth=1
	addl	(%rax), %esi
	addq	$4, %rax
	cmpq	%r8, %rax
	jne	.LBB1_102
	jmp	.LBB1_104
.LBB1_103:                              # %.preheader676.thread
	pxor	%xmm0, %xmm0
	movdqa	%xmm0, 144(%rsp)
	movq	$0, 160(%rsp)
	movdqa	%xmm0, 112(%rsp)
	movq	$0, 128(%rsp)
	xorl	%esi, %esi
	movq	64(%rsp), %r12                  # 8-byte Reload
.LBB1_104:                              # %._crit_edge
	leal	127(%rsi), %eax
	movq	%rsi, 424(%rsp)                 # 8-byte Spill
	leal	254(%rsi), %ebx
	testl	%eax, %eax
	cmovnsl	%eax, %ebx
	movl	$0, 252(%rsp)
.Ltmp108:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201@GOTPCREL(%rip), %rsi
	leaq	252(%rsp), %rdi
	movl	$9728, %ecx                     # imm = 0x2600
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp109:                               # EH_LABEL
# %bb.105:                              # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKcPKfPKhS3_S3_PfiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB1_412
# %bb.106:
	sarl	$7, %ebx
	movl	%r12d, %eax
	shrl	$8, %eax
	shlq	$32, %rax
	orq	%rbx, %rax
	movq	%rax, %rbx
	cmpb	$0, 224(%rsp)                   # 1-byte Folded Reload
	movq	8(%rsp), %r14                   # 8-byte Reload
	je	.LBB1_159
# %bb.107:                              # %_ZNSt6vectorIS_IfSaIfEESaIS1_EE17_S_check_init_lenEmRKS2_.exit.i444
	movq	%rbx, 288(%rsp)                 # 8-byte Spill
	cmpq	%r14, 16(%rsp)                  # 8-byte Folded Reload
	je	.LBB1_222
# %bb.108:                              # %.lr.ph.preheader.i.i.i.i.i446
	movq	368(%rsp), %rbx                 # 8-byte Reload
	leaq	(,%rbx,8), %rax
	leaq	(%rax,%rax,2), %r12
.Ltmp191:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp192:                               # EH_LABEL
# %bb.109:                              # %.lr.ph957.preheader
	movq	%rax, %r13
	movq	%rax, 320(%rsp)
	leaq	(%rbx,%rbx,2), %rax
	leaq	(,%rax,8), %rbx
	addq	%r13, %rbx
	.cfi_escape 0x2e, 0x00
	xorl	%r15d, %r15d
	movq	%r13, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
	addq	%r13, %r12
	movq	%rbx, 336(%rsp)
	movq	%r12, 328(%rsp)
	movq	%r13, 72(%rsp)                  # 8-byte Spill
	jmp	.LBB1_111
	.p2align	4
.LBB1_110:                              # %._crit_edge954
                                        #   in Loop: Header=BB1_111 Depth=1
	incq	%r15
	cmpq	368(%rsp), %r15                 # 8-byte Folded Reload
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	64(%rsp), %r12                  # 8-byte Reload
	movq	72(%rsp), %r13                  # 8-byte Reload
	je	.LBB1_223
.LBB1_111:                              # %.lr.ph957
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_120 Depth 2
                                        #       Child Loop BB1_122 Depth 3
                                        #         Child Loop BB1_132 Depth 4
                                        #         Child Loop BB1_144 Depth 4
	leaq	(%r15,%r15,2), %rbx
	leaq	(,%rbx,8), %rdi
	addq	%r13, %rdi
	movslq	(%r14,%r15,4), %rax
	imulq	$17408, %rax, %rcx              # imm = 0x4400
	movq	(%r13,%rbx,8), %r8
	movq	8(%r13,%rbx,8), %rdx
	movq	%rdx, %r9
	subq	%r8, %r9
	sarq	$2, %r9
	movq	%rcx, %rsi
	subq	%r9, %rsi
	jbe	.LBB1_114
# %bb.112:                              #   in Loop: Header=BB1_111 Depth=1
.Ltmp194:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_ZNSt6vectorIfSaIfEE17_M_default_appendEm
.Ltmp195:                               # EH_LABEL
# %bb.113:                              # %._ZNSt6vectorIfSaIfEE6resizeEm.exit_crit_edge
                                        #   in Loop: Header=BB1_111 Depth=1
	movl	(%r14,%r15,4), %eax
	jmp	.LBB1_117
	.p2align	4
.LBB1_114:                              #   in Loop: Header=BB1_111 Depth=1
	jae	.LBB1_117
# %bb.115:                              #   in Loop: Header=BB1_111 Depth=1
	leaq	(%r8,%rcx,4), %rcx
	cmpq	%rcx, %rdx
	je	.LBB1_117
# %bb.116:                              #   in Loop: Header=BB1_111 Depth=1
	movq	%rcx, 8(%rdi)
	.p2align	4
.LBB1_117:                              # %_ZNSt6vectorIfSaIfEE6resizeEm.exit
                                        #   in Loop: Header=BB1_111 Depth=1
	testl	%eax, %eax
	jle	.LBB1_110
# %bb.118:                              # %.preheader628.lr.ph
                                        #   in Loop: Header=BB1_111 Depth=1
	movq	256(%rsp), %rax
	leaq	(%r15,%r15,8), %rcx
	leaq	(%rax,%rcx,8), %rdx
	movq	%rdx, 136(%rsp)                 # 8-byte Spill
	movq	8(%rax,%rcx,8), %rdx
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
	movq	32(%rax,%rcx,8), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %rax                  # 8-byte Reload
	movq	(%rax,%rbx,8), %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	xorl	%r14d, %r14d
	movq	%r15, 304(%rsp)                 # 8-byte Spill
	jmp	.LBB1_120
	.p2align	4
.LBB1_119:                              #   in Loop: Header=BB1_120 Depth=2
	incq	%r14
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	304(%rsp), %r15                 # 8-byte Reload
	movslq	(%rax,%r15,4), %rax
	cmpq	%rax, %r14
	jge	.LBB1_110
.LBB1_120:                              # %.preheader628
                                        #   Parent Loop BB1_111 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_122 Depth 3
                                        #         Child Loop BB1_132 Depth 4
                                        #         Child Loop BB1_144 Depth 4
	imulq	$69632, %r14, %rbp              # imm = 0x11000
	addq	296(%rsp), %rbp                 # 8-byte Folded Reload
	xorl	%r13d, %r13d
	jmp	.LBB1_122
	.p2align	4
.LBB1_128:                              #   in Loop: Header=BB1_122 Depth=3
	xorps	.LCPI1_9(%rip), %xmm0
.LBB1_121:                              # %_ZL13folded_weightRK6Weightii.exit
                                        #   in Loop: Header=BB1_122 Depth=3
	movss	%xmm0, (%rbp,%r13,4)
	incq	%r13
	cmpq	$17408, %r13                    # imm = 0x4400
	je	.LBB1_119
.LBB1_122:                              #   Parent Loop BB1_111 Depth=1
                                        #     Parent Loop BB1_120 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB1_132 Depth 4
                                        #         Child Loop BB1_144 Depth 4
	movq	136(%rsp), %rax                 # 8-byte Reload
	movl	4(%rax), %eax
	leal	255(%rax), %ecx
	testl	%eax, %eax
	cmovnsl	%eax, %ecx
	sarl	$8, %ecx
	movl	%r13d, %eax
	shrl	$8, %eax
	movslq	%ecx, %rcx
	imulq	%r14, %rcx
	addq	%rax, %rcx
	movq	%rcx, %rax
	shlq	$7, %rax
	leaq	(%rax,%rcx,8), %rax
	addq	56(%rsp), %rax                  # 8-byte Folded Reload
	movl	%r13d, %ecx
	shrl	$5, %ecx
	andl	$4, %ecx
	movzbl	(%rcx,%rax), %edx
	movzbl	1(%rcx,%rax), %r12d
	movl	%r12d, %ecx
	andl	$3, %ecx
	shll	$8, %ecx
	orl	%edx, %ecx
	xorl	%edx, %edx
	cmpl	$512, %ecx                      # imm = 0x200
	sete	%dl
	movl	%r13d, %ecx
	shrl	%ecx
	andl	$127, %ecx
	movzbl	8(%rcx,%rax), %eax
	movl	%eax, %ecx
	shrl	$4, %ecx
	andl	$15, %eax
	testb	$1, %r13b
	cmovnel	%ecx, %eax
	addl	$-8, %eax
	xorps	%xmm1, %xmm1
	cvtsi2ss	%eax, %xmm1
	leaq	.LCPI1_4(%rip), %rax
	mulss	(%rax,%rdx,4), %xmm1
	xorl	%eax, %eax
	pxor	%xmm0, %xmm0
	ucomiss	%xmm0, %xmm1
	movl	$0, %ebx
	jne	.LBB1_129
.LBB1_123:                              # %_ZL7fp8_rnef.exit.thread.i
                                        #   in Loop: Header=BB1_122 Depth=3
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$-9, %edi
	callq	ldexpf@PLT
	testb	%bl, %bl
	jns	.LBB1_125
.LBB1_124:                              #   in Loop: Header=BB1_122 Depth=3
	xorps	.LCPI1_9(%rip), %xmm0
.LBB1_125:                              #   in Loop: Header=BB1_122 Depth=3
	movaps	%xmm0, 32(%rsp)                 # 16-byte Spill
.LBB1_126:                              # %_ZL9fp8_valueh.exit.i
                                        #   in Loop: Header=BB1_122 Depth=3
	shrl	$2, %r12d
	andl	$31, %r12d
	movq	80(%rsp), %rax                  # 8-byte Reload
	movss	(%rax,%r14,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	.cfi_escape 0x2e, 0x00
	leaq	452(%rsp), %rdi
	callq	frexpf@PLT
	subl	452(%rsp), %r12d
	addl	$-14, %r12d
	.cfi_escape 0x2e, 0x00
	movaps	32(%rsp), %xmm0                 # 16-byte Reload
	movl	%r12d, %edi
	callq	ldexpf@PLT
	movaps	%xmm0, %xmm1
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	ucomiss	%xmm0, %xmm1
	movl	$0, %ebx
	jne	.LBB1_141
	jp	.LBB1_141
.LBB1_127:                              # %_ZL7fp8_rnef.exit38.thread.i
                                        #   in Loop: Header=BB1_122 Depth=3
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$-9, %edi
	callq	ldexpf@PLT
	testb	%bl, %bl
	jns	.LBB1_121
	jmp	.LBB1_128
	.p2align	4
.LBB1_129:                              #   in Loop: Header=BB1_122 Depth=3
	movaps	%xmm1, %xmm0
	movaps	%xmm1, 400(%rsp)                # 16-byte Spill
	andps	.LCPI1_5(%rip), %xmm0
	movaps	%xmm0, 224(%rsp)                # 16-byte Spill
	xorl	%r15d, %r15d
	movss	.LCPI1_6(%rip), %xmm1           # xmm1 = [+Inf,0.0E+0,0.0E+0,0.0E+0]
	xorl	%ebx, %ebx
	jmp	.LBB1_132
	.p2align	4
.LBB1_130:                              #   in Loop: Header=BB1_132 Depth=4
	movl	%r15d, %ebx
	movaps	%xmm0, %xmm1
.LBB1_131:                              #   in Loop: Header=BB1_132 Depth=4
	incl	%r15d
	cmpl	$127, %r15d
	je	.LBB1_153
.LBB1_132:                              #   Parent Loop BB1_111 Depth=1
                                        #     Parent Loop BB1_120 Depth=2
                                        #       Parent Loop BB1_122 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movl	%r15d, %edi
	shrl	$3, %edi
	movl	%r15d, %eax
	andl	$7, %eax
	testl	%edi, %edi
	je	.LBB1_135
# %bb.133:                              #   in Loop: Header=BB1_132 Depth=4
	movl	%edi, %ecx
	xorl	$15, %ecx
	movl	%eax, %edx
	xorl	$7, %edx
	orl	%ecx, %edx
	movss	.LCPI1_7(%rip), %xmm0           # xmm0 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	je	.LBB1_137
# %bb.134:                              #   in Loop: Header=BB1_132 Depth=4
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI1_8(%rip), %xmm0
	addss	.LCPI1_0(%rip), %xmm0
	addl	$-7, %edi
	.cfi_escape 0x2e, 0x00
	jmp	.LBB1_136
	.p2align	4
.LBB1_135:                              #   in Loop: Header=BB1_132 Depth=4
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$-9, %edi
.LBB1_136:                              # %_ZL9fp8_valueh.exit.i.i
                                        #   in Loop: Header=BB1_132 Depth=4
	movss	%xmm1, 32(%rsp)                 # 4-byte Spill
	callq	ldexpf@PLT
	movss	32(%rsp), %xmm1                 # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB1_137:                              # %_ZL9fp8_valueh.exit.i.i
                                        #   in Loop: Header=BB1_132 Depth=4
	subss	224(%rsp), %xmm0                # 16-byte Folded Reload
	andps	.LCPI1_5(%rip), %xmm0
	ucomiss	%xmm0, %xmm1
	ja	.LBB1_130
# %bb.138:                              #   in Loop: Header=BB1_132 Depth=4
	ucomiss	%xmm1, %xmm0
	jne	.LBB1_131
	jp	.LBB1_131
# %bb.139:                              #   in Loop: Header=BB1_132 Depth=4
	testb	$1, %r15b
	jne	.LBB1_131
# %bb.140:                              #   in Loop: Header=BB1_132 Depth=4
	movl	%ebx, %eax
	andb	$1, %al
	jne	.LBB1_130
	jmp	.LBB1_131
	.p2align	4
.LBB1_141:                              #   in Loop: Header=BB1_122 Depth=3
	movaps	%xmm1, 400(%rsp)                # 16-byte Spill
	andps	.LCPI1_5(%rip), %xmm1
	movaps	%xmm1, 224(%rsp)                # 16-byte Spill
	xorl	%r15d, %r15d
	movss	.LCPI1_6(%rip), %xmm1           # xmm1 = [+Inf,0.0E+0,0.0E+0,0.0E+0]
	xorl	%ebx, %ebx
	jmp	.LBB1_144
	.p2align	4
.LBB1_142:                              #   in Loop: Header=BB1_144 Depth=4
	movl	%r15d, %ebx
	movaps	%xmm0, %xmm1
.LBB1_143:                              #   in Loop: Header=BB1_144 Depth=4
	incl	%r15d
	cmpl	$127, %r15d
	je	.LBB1_156
.LBB1_144:                              #   Parent Loop BB1_111 Depth=1
                                        #     Parent Loop BB1_120 Depth=2
                                        #       Parent Loop BB1_122 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movl	%r15d, %edi
	shrl	$3, %edi
	movl	%r15d, %eax
	andl	$7, %eax
	testl	%edi, %edi
	je	.LBB1_147
# %bb.145:                              #   in Loop: Header=BB1_144 Depth=4
	movl	%edi, %ecx
	xorl	$15, %ecx
	movl	%eax, %edx
	xorl	$7, %edx
	orl	%ecx, %edx
	movss	.LCPI1_7(%rip), %xmm0           # xmm0 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	je	.LBB1_149
# %bb.146:                              #   in Loop: Header=BB1_144 Depth=4
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI1_8(%rip), %xmm0
	addss	.LCPI1_0(%rip), %xmm0
	addl	$-7, %edi
	.cfi_escape 0x2e, 0x00
	jmp	.LBB1_148
	.p2align	4
.LBB1_147:                              #   in Loop: Header=BB1_144 Depth=4
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$-9, %edi
.LBB1_148:                              # %_ZL9fp8_valueh.exit.i29.i
                                        #   in Loop: Header=BB1_144 Depth=4
	movss	%xmm1, 32(%rsp)                 # 4-byte Spill
	callq	ldexpf@PLT
	movss	32(%rsp), %xmm1                 # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB1_149:                              # %_ZL9fp8_valueh.exit.i29.i
                                        #   in Loop: Header=BB1_144 Depth=4
	subss	224(%rsp), %xmm0                # 16-byte Folded Reload
	andps	.LCPI1_5(%rip), %xmm0
	ucomiss	%xmm0, %xmm1
	ja	.LBB1_142
# %bb.150:                              #   in Loop: Header=BB1_144 Depth=4
	ucomiss	%xmm1, %xmm0
	jne	.LBB1_143
	jp	.LBB1_143
# %bb.151:                              #   in Loop: Header=BB1_144 Depth=4
	testb	$1, %r15b
	jne	.LBB1_143
# %bb.152:                              #   in Loop: Header=BB1_144 Depth=4
	movl	%ebx, %eax
	andb	$1, %al
	jne	.LBB1_142
	jmp	.LBB1_143
	.p2align	4
.LBB1_153:                              # %_ZL7fp8_rnef.exit.i
                                        #   in Loop: Header=BB1_122 Depth=3
	movdqa	400(%rsp), %xmm0                # 16-byte Reload
	movd	%xmm0, %eax
	movzbl	%bl, %ecx
	orb	$-128, %bl
	testl	%eax, %eax
	movzbl	%bl, %ebx
	cmovnsl	%ecx, %ebx
	movzbl	%bl, %edi
	shrl	$3, %edi
	movl	%ebx, %eax
	andl	$7, %eax
	andl	$15, %edi
	je	.LBB1_123
# %bb.154:                              #   in Loop: Header=BB1_122 Depth=3
	movl	%edi, %ecx
	xorl	$15, %ecx
	movl	%eax, %edx
	xorl	$7, %edx
	orl	%ecx, %edx
	movss	.LCPI1_7(%rip), %xmm0           # xmm0 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	movaps	%xmm0, 32(%rsp)                 # 16-byte Spill
	je	.LBB1_126
# %bb.155:                              #   in Loop: Header=BB1_122 Depth=3
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI1_8(%rip), %xmm0
	addss	.LCPI1_0(%rip), %xmm0
	addl	$-7, %edi
	.cfi_escape 0x2e, 0x00
	callq	ldexpf@PLT
	testb	%bl, %bl
	js	.LBB1_124
	jmp	.LBB1_125
	.p2align	4
.LBB1_156:                              # %_ZL7fp8_rnef.exit38.i
                                        #   in Loop: Header=BB1_122 Depth=3
	movdqa	400(%rsp), %xmm0                # 16-byte Reload
	movd	%xmm0, %eax
	movzbl	%bl, %ecx
	orb	$-128, %bl
	testl	%eax, %eax
	movzbl	%bl, %ebx
	cmovnsl	%ecx, %ebx
	movzbl	%bl, %edi
	shrl	$3, %edi
	movl	%ebx, %eax
	andl	$7, %eax
	andl	$15, %edi
	je	.LBB1_127
# %bb.157:                              #   in Loop: Header=BB1_122 Depth=3
	movl	%edi, %ecx
	xorl	$15, %ecx
	movl	%eax, %edx
	xorl	$7, %edx
	orl	%ecx, %edx
	movss	.LCPI1_7(%rip), %xmm0           # xmm0 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	je	.LBB1_121
# %bb.158:                              #   in Loop: Header=BB1_122 Depth=3
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI1_8(%rip), %xmm0
	addss	.LCPI1_0(%rip), %xmm0
	addl	$-7, %edi
	.cfi_escape 0x2e, 0x00
	callq	ldexpf@PLT
	testb	%bl, %bl
	jns	.LBB1_121
	jmp	.LBB1_128
.LBB1_159:                              # %.preheader675.preheader
	movq	104(%rsp), %r9
.Ltmp113:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	480(%rsp), %r14
	leaq	152(%rsp), %r13
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	264(%rsp), %r8
	movq	%rbx, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp114:                               # EH_LABEL
# %bb.160:                              # %.preheader675.1
	movq	104(%rsp), %r9
.Ltmp115:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	264(%rsp), %r8
	movq	%rbx, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp116:                               # EH_LABEL
# %bb.161:                              # %.preheader675.2
	movq	104(%rsp), %r9
.Ltmp117:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	264(%rsp), %r8
	movq	%rbx, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp118:                               # EH_LABEL
# %bb.162:                              # %.preheader675.3
	movq	104(%rsp), %r9
.Ltmp119:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	264(%rsp), %r8
	movq	%rbx, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp120:                               # EH_LABEL
# %bb.163:                              # %.preheader675.4
	movq	104(%rsp), %r9
.Ltmp121:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	264(%rsp), %r8
	movq	%rbx, %rdi
	movl	$1, %esi
	movl	$1, %ecx
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp122:                               # EH_LABEL
# %bb.164:
.Ltmp124:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp125:                               # EH_LABEL
# %bb.165:
	testl	%eax, %eax
	jne	.LBB1_417
# %bb.166:                              # %.preheader629.preheader
	movl	$20, %r13d
	movq	$0, 32(%rsp)                    # 8-byte Folded Spill
	addq	$-3, 136(%rsp)                  # 8-byte Folded Spill
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	.p2align	4
.LBB1_167:                              # %.preheader629
                                        # =>This Inner Loop Header: Depth=1
.Ltmp129:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	320(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp130:                               # EH_LABEL
# %bb.168:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_363
# %bb.169:                              #   in Loop: Header=BB1_167 Depth=1
.Ltmp135:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	392(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp136:                               # EH_LABEL
	movq	64(%rsp), %r14                  # 8-byte Reload
# %bb.170:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_365
# %bb.171:                              #   in Loop: Header=BB1_167 Depth=1
	movq	320(%rsp), %rdi
.Ltmp141:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp142:                               # EH_LABEL
# %bb.172:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_368
# %bb.173:                              #   in Loop: Header=BB1_167 Depth=1
	movq	104(%rsp), %r9
.Ltmp147:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movq	%rbx, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	leaq	264(%rsp), %r8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	leaq	488(%rsp), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	leaq	168(%rsp), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp148:                               # EH_LABEL
# %bb.174:                              #   in Loop: Header=BB1_167 Depth=1
	movq	392(%rsp), %rdi
.Ltmp150:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp151:                               # EH_LABEL
# %bb.175:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_370
# %bb.176:                              #   in Loop: Header=BB1_167 Depth=1
	movq	392(%rsp), %rdi
.Ltmp156:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp157:                               # EH_LABEL
# %bb.177:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_359
# %bb.178:                              #   in Loop: Header=BB1_167 Depth=1
	movl	$0, 316(%rsp)
	movq	320(%rsp), %rsi
	movq	392(%rsp), %rdx
.Ltmp162:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	316(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp163:                               # EH_LABEL
# %bb.179:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_361
# %bb.180:                              #   in Loop: Header=BB1_167 Depth=1
	movss	316(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI1_1(%rip), %xmm0
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r15
	je	.LBB1_182
# %bb.181:                              #   in Loop: Header=BB1_167 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r12
	jmp	.LBB1_189
	.p2align	4
.LBB1_182:                              #   in Loop: Header=BB1_167 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%rax, %r15
	subq	%rbp, %r15
	cmpq	136(%rsp), %r15                 # 8-byte Folded Reload
	je	.LBB1_388
# %bb.183:                              # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB1_167 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp168:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp169:                               # EH_LABEL
# %bb.184:                              # %.noexc476
                                        #   in Loop: Header=BB1_167 Depth=1
	movq	%rax, %r12
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB1_186
# %bb.185:                              #   in Loop: Header=BB1_167 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB1_186:                              # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB1_167 Depth=1
	testq	%rbp, %rbp
	je	.LBB1_188
# %bb.187:                              # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB1_167 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB1_188:                              # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB1_167 Depth=1
	addq	%r12, %r15
	leaq	(%r12,%r14,4), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
.LBB1_189:                              # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB1_167 Depth=1
	movq	320(%rsp), %rdi
.Ltmp171:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp172:                               # EH_LABEL
# %bb.190:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_373
# %bb.191:                              #   in Loop: Header=BB1_167 Depth=1
	movq	392(%rsp), %rdi
.Ltmp177:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp178:                               # EH_LABEL
# %bb.192:                              #   in Loop: Header=BB1_167 Depth=1
	testl	%eax, %eax
	jne	.LBB1_375
# %bb.193:                              #   in Loop: Header=BB1_167 Depth=1
	addq	$4, %r15
	movq	%r12, %rbp
	decl	%r13d
	jne	.LBB1_167
# %bb.194:
	cmpq	%r15, %r12
	je	.LBB1_266
# %bb.195:
	movq	%r15, %rax
	subq	%r12, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp183:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	32(%rsp), %rbx                  # 8-byte Reload
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp184:                               # EH_LABEL
# %bb.196:                              # %.noexc471
.Ltmp185:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp186:                               # EH_LABEL
	jmp	.LBB1_267
.LBB1_197:
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movq	%r14, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB1_198:
	movq	112(%rsp), %rax
	movq	%rax, 152(%rsp)
	movq	144(%rsp), %rcx
	movb	$0, (%rcx,%rax)
	movq	144(%rsp), %rdi
	cmpq	$5, 152(%rsp)
	jne	.LBB1_200
# %bb.199:
	movl	$1869574768, %eax               # imm = 0x6F6F7270
	xorl	(%rdi), %eax
	movzbl	4(%rdi), %ecx
	xorl	$102, %ecx
	orl	%eax, %ecx
	sete	%bpl
	cmpq	%r12, %rdi
	jne	.LBB1_201
	jmp	.LBB1_202
.LBB1_200:
	xorl	%ebp, %ebp
	cmpq	%r12, %rdi
	je	.LBB1_202
.LBB1_201:                              # %_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv.exit.i.i
	movq	160(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_202:                              # %.critedge358
	testb	%bpl, %bpl
	je	.LBB1_216
# %bb.203:
	movabsq	$4035295638022274048, %rbx      # imm = 0x38004000BC003C00
	.cfi_escape 0x2e, 0x00
	movl	$8704, %edi                     # imm = 0x2200
	callq	_Znwm@PLT
	movq	%rax, %r12
	movq	%rax, 144(%rsp)
	leaq	8704(%rax), %r15
	movq	%r15, 160(%rsp)
	.cfi_escape 0x2e, 0x00
	xorl	%r14d, %r14d
	movl	$8704, %edx                     # imm = 0x2200
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r15, 152(%rsp)
	leaq	11(%r12), %rax
	movq	%r12, 56(%rsp)                  # 8-byte Spill
	leaq	147(%r12), %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB1_204:                              # %.preheader116.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_205 Depth 2
                                        #     Child Loop BB1_207 Depth 2
	imulq	$272, %rdx, %rsi                # imm = 0x110
	movq	56(%rsp), %rdi                  # 8-byte Reload
	movq	%rbx, (%rdi,%rsi)
	movq	$-896, %rdi                     # imm = 0xFC80
	movq	%rax, %r8
	.p2align	4
.LBB1_205:                              #   Parent Loop BB1_204 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leal	(%r14,%rdi), %r9d
	leal	-128(%r9), %r10d
	movb	%r10b, -3(%r8)
	leal	-121(%r9), %r10d
	movb	%r10b, -2(%r8)
	leal	-114(%r9), %r10d
	movb	%r10b, -1(%r8)
	addb	$-107, %r9b
	movb	%r9b, (%r8)
	addq	$4, %r8
	addq	$28, %rdi
	jne	.LBB1_205
# %bb.206:                              #   in Loop: Header=BB1_204 Depth=1
	movq	56(%rsp), %rdi                  # 8-byte Reload
	movq	%rbx, 136(%rdi,%rsi)
	movq	$-896, %rsi                     # imm = 0xFC80
	movq	%rcx, %rdi
	.p2align	4
.LBB1_207:                              #   Parent Loop BB1_204 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leal	(%r14,%rsi), %r8d
	leal	-111(%r8), %r9d
	movb	%r9b, -3(%rdi)
	leal	-104(%r8), %r9d
	movb	%r9b, -2(%rdi)
	leal	-97(%r8), %r9d
	movb	%r9b, -1(%rdi)
	addb	$-90, %r8b
	movb	%r8b, (%rdi)
	addq	$4, %rdi
	addq	$28, %rsi
	jne	.LBB1_207
# %bb.208:                              #   in Loop: Header=BB1_204 Depth=1
	incq	%rdx
	addq	$29, %r14
	addq	$272, %rax                      # imm = 0x110
	addq	$272, %rcx                      # imm = 0x110
	cmpq	$32, %rdx
	jne	.LBB1_204
# %bb.209:
.Ltmp220:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	leaq	144(%rsp), %rsi
	movl	$32, %edx
	movl	$512, %ecx                      # imm = 0x200
	callq	_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii
.Ltmp221:                               # EH_LABEL
# %bb.210:
	movq	112(%rsp), %rbx
	movq	120(%rsp), %r12
	subq	%rbx, %r12
	je	.LBB1_286
# %bb.211:                              # %.noexc62.i.i
.Ltmp223:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp224:                               # EH_LABEL
# %bb.212:                              # %.noexc.i373
	movq	%rax, %r14
	leaq	(%rax,%r12), %r15
	movb	$0, (%rax)
	movq	%rax, %r13
	incq	%r13
	movq	%r12, %rdx
	decq	%rdx
	je	.LBB1_214
# %bb.213:
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r15, %r13
.LBB1_214:
.Ltmp226:                               # EH_LABEL
	movq	%r15, 280(%rsp)                 # 8-byte Spill
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp227:                               # EH_LABEL
# %bb.215:                              # %.noexc67.i.i
	movq	%rax, %rbp
	movq	%rax, %r15
	addq	%r12, %r15
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
	jmp	.LBB1_287
.LBB1_216:
	movq	8(%rbx), %rbx
	leaq	160(%rsp), %r15
	movq	%r15, 144(%rsp)
	testq	%rbx, %rbx
	je	.LBB1_421
# %bb.217:
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	strlen@PLT
	movq	%rax, %r14
	movq	%rax, 112(%rsp)
	movq	%r15, %rax
	cmpq	$16, %r14
	jb	.LBB1_219
# %bb.218:                              # %.noexc.i378
	.cfi_escape 0x2e, 0x00
	leaq	144(%rsp), %rdi
	leaq	112(%rsp), %rsi
	xorl	%edx, %edx
	callq	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@PLT
	movq	%rax, 144(%rsp)
	movq	112(%rsp), %rcx
	movq	%rcx, 160(%rsp)
.LBB1_219:                              # %._crit_edge.i.i377
	testq	%r14, %r14
	je	.LBB1_342
# %bb.220:                              # %._crit_edge.i.i377
	cmpq	$1, %r14
	jne	.LBB1_341
# %bb.221:
	movzbl	(%rbx), %ecx
	movb	%cl, (%rax)
	jmp	.LBB1_342
.LBB1_222:                              # %.thread1345
	pxor	%xmm0, %xmm0
	movdqa	%xmm0, 320(%rsp)
	movq	$0, 336(%rsp)
.LBB1_223:                              # %._crit_edge958
	movq	104(%rsp), %r9
.Ltmp197:                               # EH_LABEL
	.cfi_escape 0x2e, 0x30
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	480(%rsp), %rax
	leaq	152(%rsp), %r10
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	leaq	264(%rsp), %r8
	movq	296(%rsp), %rdi                 # 8-byte Reload
	movl	$1, %esi
	movl	$1, %ecx
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	120(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	addq	$48, %rsp
	.cfi_adjust_cfa_offset -48
.Ltmp198:                               # EH_LABEL
# %bb.224:
.Ltmp200:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	16(%rsp), %rbx                  # 8-byte Reload
	callq	hipDeviceSynchronize@PLT
.Ltmp201:                               # EH_LABEL
# %bb.225:
	testl	%eax, %eax
	jne	.LBB1_419
# %bb.226:                              # %.preheader622
	cmpq	8(%rsp), %rbx                   # 8-byte Folded Reload
	je	.LBB1_254
# %bb.227:                              # %.lr.ph989
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1
	movq	144(%rsp), %rax
	movq	%rax, 496(%rsp)                 # 8-byte Spill
	xorl	%r14d, %r14d
	xorl	%ebp, %ebp
	jmp	.LBB1_229
	.p2align	4
.LBB1_228:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit468
                                        #   in Loop: Header=BB1_229 Depth=1
	incq	%rbp
	cmpq	368(%rsp), %rbp                 # 8-byte Folded Reload
	je	.LBB1_255
.LBB1_229:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_240 Depth 2
                                        #       Child Loop BB1_243 Depth 3
                                        #         Child Loop BB1_246 Depth 4
	movq	8(%rsp), %rax                   # 8-byte Reload
	movslq	(%rax,%rbp,4), %rax
	movq	%rax, %rbx
	imulq	%r12, %rbx
	movabsq	$2305843009213693951, %rcx      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rcx, %rbx
	ja	.LBB1_383
# %bb.230:                              # %_ZNSt6vectorIfSaIfEE17_S_check_init_lenEmRKS0_.exit.i
                                        #   in Loop: Header=BB1_229 Depth=1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	movdqa	%xmm0, 224(%rsp)                # 16-byte Spill
	testl	%eax, %eax
	je	.LBB1_233
# %bb.231:                              #   in Loop: Header=BB1_229 Depth=1
	leaq	(,%rbx,4), %r12
.Ltmp205:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp206:                               # EH_LABEL
# %bb.232:                              # %.noexc464
                                        #   in Loop: Header=BB1_229 Depth=1
	leaq	(%rax,%rbx,4), %rcx
	movq	%rcx, 432(%rsp)                 # 8-byte Spill
	movl	$0, (%rax)
	movq	%rax, %rbx
	leaq	4(%rax), %rdi
	leaq	-4(%r12), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbx, %rdi
	movq	%rbx, %rdx
	addq	%r12, %rdx
	movq	64(%rsp), %r12                  # 8-byte Reload
	jmp	.LBB1_234
	.p2align	4
.LBB1_233:                              #   in Loop: Header=BB1_229 Depth=1
	xorl	%edi, %edi
	movq	$0, 432(%rsp)                   # 8-byte Folded Spill
	xorl	%edx, %edx
.LBB1_234:                              # %_ZNSt6vectorIfSaIfEEC2EmRKS0_.exit465
                                        #   in Loop: Header=BB1_229 Depth=1
	movq	496(%rsp), %rax                 # 8-byte Reload
	movq	(%rax,%rbp,8), %rsi
	subq	%rdi, %rdx
.Ltmp208:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rdi, 72(%rsp)                  # 8-byte Spill
	movl	$2, %ecx
	movq	360(%rsp), %rbx                 # 8-byte Reload
	callq	hipMemcpy@PLT
.Ltmp209:                               # EH_LABEL
# %bb.235:                              #   in Loop: Header=BB1_229 Depth=1
	testl	%eax, %eax
	movdqa	32(%rsp), %xmm1                 # 16-byte Reload
	jne	.LBB1_385
# %bb.236:                              # %.preheader621
                                        #   in Loop: Header=BB1_229 Depth=1
	movq	320(%rsp), %rdx
	movq	8(%rsp), %rax                   # 8-byte Reload
	movl	(%rax,%rbp,4), %eax
	testl	%eax, %eax
	movapd	224(%rsp), %xmm0                # 16-byte Reload
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movq	%rdx, 464(%rsp)                 # 8-byte Spill
	jle	.LBB1_252
# %bb.237:                              # %.preheader620.preheader
                                        #   in Loop: Header=BB1_229 Depth=1
	leaq	(,%rbp,8), %rcx
	leaq	(%rcx,%rcx,2), %rcx
	addq	%rcx, %rdx
	movq	%rdx, 440(%rsp)                 # 8-byte Spill
	addq	112(%rsp), %rcx
	movq	%rcx, 288(%rsp)                 # 8-byte Spill
	leaq	(,%rbp,8), %rcx
	addq	%rbp, %rcx
	shlq	$3, %rcx
	addq	256(%rsp), %rcx
	movq	%rcx, 280(%rsp)                 # 8-byte Spill
	movq	$0, 80(%rsp)                    # 8-byte Folded Spill
	movdqa	%xmm1, %xmm2
	jmp	.LBB1_240
	.p2align	4
.LBB1_238:                              #   in Loop: Header=BB1_240 Depth=2
	movapd	%xmm1, %xmm2
	movq	64(%rsp), %r12                  # 8-byte Reload
.LBB1_239:                              # %._crit_edge967
                                        #   in Loop: Header=BB1_240 Depth=2
	movq	80(%rsp), %rdx                  # 8-byte Reload
	incq	%rdx
	addq	$17408, %rbx                    # imm = 0x4400
	movq	%rdx, 80(%rsp)                  # 8-byte Spill
	cmpq	%r12, %rdx
	je	.LBB1_252
.LBB1_240:                              # %.preheader620
                                        #   Parent Loop BB1_229 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_243 Depth 3
                                        #         Child Loop BB1_246 Depth 4
	testl	%eax, %eax
	jle	.LBB1_251
# %bb.241:                              # %.preheader.lr.ph
                                        #   in Loop: Header=BB1_240 Depth=2
	movq	440(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %r15
	movq	288(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	movq	%rax, 400(%rsp)                 # 8-byte Spill
	movq	384(%rsp), %rax                 # 8-byte Reload
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movss	(%rax,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, 304(%rsp)                # 4-byte Spill
	movq	280(%rsp), %rax                 # 8-byte Reload
	movq	32(%rax), %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	xorl	%r13d, %r13d
	jmp	.LBB1_243
	.p2align	4
.LBB1_242:                              #   in Loop: Header=BB1_243 Depth=3
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	%r14, %rbp
	movslq	(%rax,%r14,4), %rax
	movq	80(%rsp), %rcx                  # 8-byte Reload
	imulq	%rax, %rcx
	addq	%r13, %rcx
	movq	296(%rsp), %rdx                 # 8-byte Reload
	movss	(%rdx,%r13,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	mulss	304(%rsp), %xmm0                # 4-byte Folded Reload
	mulss	%xmm0, %xmm2
	movq	400(%rsp), %rdx                 # 8-byte Reload
	addss	(%rdx,%rcx,4), %xmm2
	movd	%xmm2, %edx
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movd	(%rdi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movd	%xmm0, %ecx
	addl	%ecx, %ecx
	cmpl	$-16777216, %ecx                # imm = 0xFF000000
	setae	%cl
	addl	%edx, %edx
	cmpl	$-16777216, %edx                # imm = 0xFF000000
	setae	%dl
	orb	%cl, %dl
	movzbl	%dl, %ecx
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtss2sd	%xmm2, %xmm0
	movq	136(%rsp), %r14                 # 8-byte Reload
	addq	%rcx, %r14
	subsd	%xmm0, %xmm1
	movapd	%xmm1, %xmm2
	unpcklpd	%xmm0, %xmm2                    # xmm2 = xmm2[0],xmm0[0]
	mulpd	%xmm2, %xmm2
	movapd	224(%rsp), %xmm0                # 16-byte Reload
	addpd	%xmm2, %xmm0
	andpd	.LCPI1_10(%rip), %xmm1
	maxsd	56(%rsp), %xmm1                 # 8-byte Folded Reload
	incq	%r13
	addq	$69632, %r15                    # imm = 0x11000
	movapd	%xmm1, %xmm2
	cmpq	%rax, %r13
	jge	.LBB1_238
.LBB1_243:                              # %.preheader
                                        #   Parent Loop BB1_229 Depth=1
                                        #     Parent Loop BB1_240 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB1_246 Depth 4
	movq	%xmm2, 56(%rsp)                 # 8-byte Folded Spill
	movq	%r14, 136(%rsp)                 # 8-byte Spill
	movq	%rbp, %r14
	movapd	%xmm0, 224(%rsp)                # 16-byte Spill
	pxor	%xmm2, %xmm2
	xorl	%ebp, %ebp
	jmp	.LBB1_246
	.p2align	4
.LBB1_250:                              #   in Loop: Header=BB1_246 Depth=4
	xorps	.LCPI1_9(%rip), %xmm0
.LBB1_244:                              #   in Loop: Header=BB1_246 Depth=4
	movd	32(%rsp), %xmm2                 # 4-byte Folded Reload
                                        # xmm2 = mem[0],zero,zero,zero
.LBB1_245:                              # %_ZL9fp8_valueh.exit
                                        #   in Loop: Header=BB1_246 Depth=4
	movd	(%r15,%rbp,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	.cfi_escape 0x2e, 0x00
	callq	fmaf@PLT
	movaps	%xmm0, %xmm2
	incq	%rbp
	cmpq	$17408, %rbp                    # imm = 0x4400
	je	.LBB1_242
.LBB1_246:                              #   Parent Loop BB1_229 Depth=1
                                        #     Parent Loop BB1_240 Depth=2
                                        #       Parent Loop BB1_243 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movzbl	(%rbx,%rbp), %r12d
	movl	%r12d, %edi
	shrl	$3, %edi
	movl	%r12d, %eax
	andl	$7, %eax
	andl	$15, %edi
	je	.LBB1_249
# %bb.247:                              #   in Loop: Header=BB1_246 Depth=4
	movl	%edi, %ecx
	xorl	$15, %ecx
	movl	%eax, %edx
	xorl	$7, %edx
	orl	%ecx, %edx
	movss	.LCPI1_7(%rip), %xmm0           # xmm0 = [NaN,0.0E+0,0.0E+0,0.0E+0]
	je	.LBB1_245
# %bb.248:                              #   in Loop: Header=BB1_246 Depth=4
	movd	%xmm2, 32(%rsp)                 # 4-byte Folded Spill
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI1_8(%rip), %xmm0
	addss	.LCPI1_0(%rip), %xmm0
	addl	$-7, %edi
	.cfi_escape 0x2e, 0x00
	callq	ldexpf@PLT
	testb	%r12b, %r12b
	jns	.LBB1_244
	jmp	.LBB1_250
	.p2align	4
.LBB1_249:                              #   in Loop: Header=BB1_246 Depth=4
	movd	%xmm2, 32(%rsp)                 # 4-byte Folded Spill
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	.cfi_escape 0x2e, 0x00
	movl	$-9, %edi
	callq	ldexpf@PLT
	testb	%r12b, %r12b
	jns	.LBB1_244
	jmp	.LBB1_250
	.p2align	4
.LBB1_251:                              #   in Loop: Header=BB1_240 Depth=2
	movdqa	%xmm2, %xmm1
	jmp	.LBB1_239
	.p2align	4
.LBB1_252:                              # %.split.us
                                        #   in Loop: Header=BB1_229 Depth=1
	testq	%rdi, %rdi
	je	.LBB1_228
# %bb.253:                              #   in Loop: Header=BB1_229 Depth=1
	movq	432(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	movapd	%xmm0, 224(%rsp)                # 16-byte Spill
	movapd	%xmm1, 32(%rsp)                 # 16-byte Spill
	callq	_ZdlPvm@PLT
	movapd	32(%rsp), %xmm1                 # 16-byte Reload
	movapd	224(%rsp), %xmm0                # 16-byte Reload
	jmp	.LBB1_228
.LBB1_254:                              # %.preheader622.._crit_edge990_crit_edge
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1
	movq	320(%rsp), %rax
	movq	%rax, 464(%rsp)                 # 8-byte Spill
	xorl	%r14d, %r14d
.LBB1_255:                              # %._crit_edge990
	movdqa	%xmm0, %xmm2
	punpckhqdq	%xmm0, %xmm2            # xmm2 = xmm2[1],xmm0[1]
	movsd	.LCPI1_11(%rip), %xmm3          # xmm3 = [1.0E-300,0.0E+0]
	maxsd	%xmm2, %xmm3
	divsd	%xmm3, %xmm0
	xorpd	%xmm2, %xmm2
	ucomisd	%xmm2, %xmm0
	jb	.LBB1_257
# %bb.256:
	sqrtsd	%xmm0, %xmm0
	jmp	.LBB1_258
.LBB1_257:                              # %call.sqrt
	.cfi_escape 0x2e, 0x00
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	callq	sqrt@PLT
	movdqa	32(%rsp), %xmm1                 # 16-byte Reload
.LBB1_258:                              # %._crit_edge990.split
	movq	424(%rsp), %rdx                 # 8-byte Reload
	movl	252(%rsp), %r10d
	.cfi_escape 0x2e, 0x10
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	.L.str.6(%rip), %rdi
	movl	$1, %esi
                                        # kill: def $edx killed $edx killed $rdx
	movl	%r12d, %ecx
	movl	$17408, %r8d                    # imm = 0x4400
	movsd	%xmm0, 40(%rsp)                 # 8-byte Spill
	movq	%r14, %r9
	movb	$2, %al
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	callq	printf@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	testq	%r14, %r14
	setne	%bl
	movsd	.LCPI1_12(%rip), %xmm0          # xmm0 = [9.9999999999999995E-7,0.0E+0]
	ucomisd	32(%rsp), %xmm0                 # 8-byte Folded Reload
	setb	%bpl
	movq	328(%rsp), %r14
	movq	464(%rsp), %r12                 # 8-byte Reload
	cmpq	%r14, %r12
	je	.LBB1_263
# %bb.259:                              # %.lr.ph.i.i.i.preheader
	movq	%r12, %r15
	jmp	.LBB1_261
	.p2align	4
.LBB1_260:                              # %_ZSt8_DestroyISt6vectorIfSaIfEEEvPT_.exit.i.i.i
                                        #   in Loop: Header=BB1_261 Depth=1
	addq	$24, %r15
	cmpq	%r14, %r15
	je	.LBB1_263
.LBB1_261:                              # %.lr.ph.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movq	(%r15), %rdi
	testq	%rdi, %rdi
	je	.LBB1_260
# %bb.262:                              #   in Loop: Header=BB1_261 Depth=1
	movq	16(%r15), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB1_260
.LBB1_263:                              # %_ZSt8_DestroyIPSt6vectorIfSaIfEES2_EvT_S4_RSaIT0_E.exit.i
	orb	%bpl, %bl
	testq	%r12, %r12
	je	.LBB1_265
# %bb.264:
	movq	336(%rsp), %rsi
	subq	%r12, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
.LBB1_265:                              # %_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev.exit
	movzbl	%bl, %ebp
	movq	112(%rsp), %r14
	movq	120(%rsp), %rbx
	cmpq	%rbx, %r14
	jne	.LBB1_268
	jmp	.LBB1_272
.LBB1_266:                              # %._ZNSt6vectorIfSaIfEED2Ev.exit478_crit_edge
	movq	32(%rsp), %rbx                  # 8-byte Reload
.LBB1_267:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit478
	movss	36(%r12), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addss	40(%r12), %xmm0
	cvtss2sd	%xmm0, %xmm0
	mulsd	.LCPI1_2(%rip), %xmm0
	movq	64(%rsp), %rcx                  # 8-byte Reload
	imull	$34816, %ecx, %eax              # imm = 0x8800
	cvtsi2sd	%eax, %xmm2
	movq	424(%rsp), %rdx                 # 8-byte Reload
	cvtsi2sd	%edx, %xmm1
	mulsd	%xmm2, %xmm1
	divsd	%xmm0, %xmm1
	divsd	.LCPI1_3(%rip), %xmm1
	movl	252(%rsp), %r9d
	.cfi_escape 0x2e, 0x10
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	.L.str.7(%rip), %rdi
	movl	$1, %esi
                                        # kill: def $edx killed $edx killed $rdx
                                        # kill: def $ecx killed $ecx killed $rcx
	movl	$17408, %r8d                    # imm = 0x4400
	movb	$2, %al
	pushq	$9728                           # imm = 0x2600
	.cfi_adjust_cfa_offset 8
	callq	printf@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	subq	%r12, %rbx
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	xorl	%ebp, %ebp
	movq	112(%rsp), %r14
	movq	120(%rsp), %rbx
	cmpq	%rbx, %r14
	je	.LBB1_272
.LBB1_268:                              # %.lr.ph.i.i.i482.preheader
	movq	%r14, %r15
	jmp	.LBB1_270
	.p2align	4
.LBB1_269:                              # %_ZSt8_DestroyISt6vectorIfSaIfEEEvPT_.exit.i.i.i485
                                        #   in Loop: Header=BB1_270 Depth=1
	addq	$24, %r15
	cmpq	%rbx, %r15
	je	.LBB1_272
.LBB1_270:                              # %.lr.ph.i.i.i482
                                        # =>This Inner Loop Header: Depth=1
	movq	(%r15), %rdi
	testq	%rdi, %rdi
	je	.LBB1_269
# %bb.271:                              #   in Loop: Header=BB1_270 Depth=1
	movq	16(%r15), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB1_269
.LBB1_272:                              # %_ZSt8_DestroyIPSt6vectorIfSaIfEES2_EvT_S4_RSaIT0_E.exit.i489
	testq	%r14, %r14
	je	.LBB1_274
# %bb.273:
	movq	128(%rsp), %rsi
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
.LBB1_274:                              # %_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev.exit491
	movq	144(%rsp), %rdi
	testq	%rdi, %rdi
	movq	24(%rsp), %rbx                  # 8-byte Reload
	movq	384(%rsp), %r14                 # 8-byte Reload
	je	.LBB1_276
# %bb.275:
	movq	160(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_276:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit497
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	352(%rsp), %rsi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movq	376(%rsp), %rdi                 # 8-byte Reload
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movq	360(%rsp), %rdi                 # 8-byte Reload
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	movq	256(%rsp), %r14
	movq	264(%rsp), %rbx
	cmpq	%rbx, %r14
	jne	.LBB1_281
# %bb.277:                              # %_ZSt8_DestroyIP6WeightS0_EvT_S2_RSaIT0_E.exit.i
	testq	%r14, %r14
	je	.LBB1_279
.LBB1_278:
	movq	272(%rsp), %rsi
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
.LBB1_279:                              # %_ZNSt6vectorI6WeightSaIS0_EED2Ev.exit
	.cfi_escape 0x2e, 0x00
	movq	8(%rsp), %rdi                   # 8-byte Reload
	movq	456(%rsp), %rsi                 # 8-byte Reload
	jmp	.LBB1_339
	.p2align	4
.LBB1_280:                              # %_ZSt8_DestroyI6WeightEvPT_.exit.i.i.i
                                        #   in Loop: Header=BB1_281 Depth=1
	addq	$72, %r14
	cmpq	%rbx, %r14
	je	.LBB1_285
.LBB1_281:                              # %.lr.ph.i.i.i499
                                        # =>This Inner Loop Header: Depth=1
	movq	32(%r14), %rdi
	testq	%rdi, %rdi
	je	.LBB1_283
# %bb.282:                              #   in Loop: Header=BB1_281 Depth=1
	movq	48(%r14), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_283:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit.i.i.i.i.i
                                        #   in Loop: Header=BB1_281 Depth=1
	movq	8(%r14), %rdi
	testq	%rdi, %rdi
	je	.LBB1_280
# %bb.284:                              #   in Loop: Header=BB1_281 Depth=1
	movq	24(%r14), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	jmp	.LBB1_280
.LBB1_285:                              # %_ZSt8_DestroyIP6WeightS0_EvT_S2_RSaIT0_E.exitthread-pre-split.i
	movq	256(%rsp), %r14
	testq	%r14, %r14
	jne	.LBB1_278
	jmp	.LBB1_279
.LBB1_286:
	xorl	%r14d, %r14d
	xorl	%r13d, %r13d
	movq	$0, 280(%rsp)                   # 8-byte Folded Spill
	xorl	%ebp, %ebp
	xorl	%r15d, %r15d
.LBB1_287:                              # %_ZNSt6vectorIhSaIhEEC2EmRKhRKS0_.exit.i.i
	subq	%r14, %r13
	xorl	%eax, %eax
	.p2align	4
.LBB1_288:                              # %.preheader91.i.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_289 Depth 2
	leal	(%rax,%rax), %ecx
	movl	%ecx, %esi
	andl	$32, %esi
	movl	%eax, %edi
	andl	$15, %edi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	.p2align	4
.LBB1_289:                              #   Parent Loop BB1_288 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r9d, %edx
	shrl	$7, %edx
	addl	%ecx, %edx
	movl	%edx, %r10d
	shll	$7, %edx
	leal	(%rdx,%r10,8), %edx
	movl	%r9d, %r10d
	andl	$127, %r10d
	addq	%r10, %rdx
	addq	$8, %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.290:                              #   in Loop: Header=BB1_289 Depth=2
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.291:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit.i.i
                                        #   in Loop: Header=BB1_289 Depth=2
	movl	%r9d, %r10d
	shrl	$3, %r10d
	addl	%esi, %r10d
	movl	%r8d, %r11d
	andl	$16, %r11d
	orl	%edi, %r11d
	shll	$7, %r10d
	leal	(%r10,%r11,4), %r10d
	movl	%r9d, %r11d
	andl	$3, %r11d
	orl	%r10d, %r11d
	movzbl	(%rbx,%r11), %r10d
	movb	%r10b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	incl	%r9d
	addl	$4, %r8d
	cmpl	$256, %r9d                      # imm = 0x100
	jne	.LBB1_289
# %bb.292:                              # %.preheader.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	imulq	$272, %rax, %rcx                # imm = 0x110
	cmpq	%rcx, %r13
	jbe	.LBB1_353
# %bb.293:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rcx)
	jne	.LBB1_353
# %bb.294:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movl	%eax, %esi
	andl	$15, %esi
	movl	%eax, %edx
	andl	$16, %edx
	leaq	(%rbx,%rdx,4), %rdi
	movzbl	8208(%rsi,%rdi), %edx
	movb	%dl, (%r14,%rcx)
	movb	$1, (%rbp,%rcx)
	leaq	1(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.295:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.296:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.1.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8192(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	2(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.297:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.298:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.2.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8224(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	3(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.299:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.300:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.3.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8240(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	4(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.301:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.302:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.1106.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8336(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	5(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.303:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.304:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.1.1.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8320(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	6(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.305:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.306:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.2.1.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8352(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	7(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.307:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.308:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.3.1.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8368(%rdi,%rsi), %r8d
	movb	%r8b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	leaq	136(%rcx), %r8
	cmpq	%r8, %r13
	jbe	.LBB1_381
# %bb.309:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%r8)
	jne	.LBB1_381
# %bb.310:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.2109.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8464(%rdi,%rsi), %edx
	movb	%dl, 136(%r14,%rcx)
	movb	$1, 136(%rbp,%rcx)
	leaq	137(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.311:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.312:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.1.2.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8448(%rdi,%rsi), %edx
	movb	%dl, 137(%r14,%rcx)
	movb	$1, 137(%rbp,%rcx)
	leaq	138(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.313:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.314:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.2.2.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8480(%rdi,%rsi), %edx
	movb	%dl, 138(%r14,%rcx)
	movb	$1, 138(%rbp,%rcx)
	leaq	139(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.315:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.316:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.3.2.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8496(%rdi,%rsi), %r9d
	movb	%r9b, (%r14,%rdx)
	movb	$1, (%rbp,%rdx)
	orq	$4, %r8
	cmpq	%r8, %r13
	jbe	.LBB1_381
# %bb.317:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%r8)
	jne	.LBB1_381
# %bb.318:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.3112.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8592(%rdi,%rsi), %edx
	movb	%dl, (%r14,%r8)
	movb	$1, (%rbp,%r8)
	leaq	141(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.319:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.320:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.1.3.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8576(%rdi,%rsi), %edx
	movb	%dl, 141(%r14,%rcx)
	movb	$1, 141(%rbp,%rcx)
	leaq	142(%rcx), %rdx
	cmpq	%rdx, %r13
	jbe	.LBB1_382
# %bb.321:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rdx)
	jne	.LBB1_382
# %bb.322:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.2.3.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8608(%rdi,%rsi), %edx
	movb	%dl, 142(%r14,%rcx)
	movb	$1, 142(%rbp,%rcx)
	addq	$143, %rcx
	cmpq	%rcx, %r13
	jbe	.LBB1_353
# %bb.323:                              #   in Loop: Header=BB1_288 Depth=1
	cmpb	$0, (%rbp,%rcx)
	jne	.LBB1_353
# %bb.324:                              # %_ZZL16unshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.3.3.i.i
                                        #   in Loop: Header=BB1_288 Depth=1
	movzbl	8624(%rdi,%rsi), %edx
	movb	%dl, (%r14,%rcx)
	movb	$1, (%rbp,%rcx)
	incq	%rax
	cmpq	$32, %rax
	jne	.LBB1_288
# %bb.325:
	movq	%r15, %r12
	subq	%rbp, %r12
	testq	%r12, %r12
	jle	.LBB1_327
# %bb.326:
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memchr@PLT
	testq	%rax, %rax
	setne	%cl
	cmpq	%r15, %rax
	setne	%al
	testb	%al, %cl
	jne	.LBB1_422
.LBB1_327:
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	cmpq	$8704, %r13                     # imm = 0x2200
	jne	.LBB1_351
# %bb.328:                              # %_ZStneIhSaIhEEbRKSt6vectorIT_T0_ES6_.exit.i
	.cfi_escape 0x2e, 0x00
	movl	$8704, %edx                     # imm = 0x2200
	movq	%r14, %rdi
	movq	56(%rsp), %rsi                  # 8-byte Reload
	callq	bcmp@PLT
	testl	%eax, %eax
	jne	.LBB1_351
# %bb.329:                              # %.preheader115.i
	leaq	8192(%rbx), %rax
	movq	%rax, 440(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
	movq	%rbx, 72(%rsp)                  # 8-byte Spill
	movq	%r14, 288(%rsp)                 # 8-byte Spill
	.p2align	4
.LBB1_330:                              # %.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_331 Depth 2
	leal	(%rcx,%rcx), %eax
	movl	%eax, 80(%rsp)                  # 4-byte Spill
	movl	%ecx, %edx
	shrl	$4, %edx
	movl	%edx, %eax
	shll	$6, %eax
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movl	%ecx, %esi
	andl	$15, %esi
	movq	440(%rsp), %rcx                 # 8-byte Reload
	movq	%rsi, 8(%rsp)                   # 8-byte Spill
	addq	%rsi, %rcx
	movq	%rcx, 304(%rsp)                 # 8-byte Spill
	shll	$10, %edx
	movl	%edx, 400(%rsp)                 # 4-byte Spill
	movl	%eax, %eax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	xorl	%ebp, %ebp
	.p2align	4
.LBB1_331:                              #   Parent Loop BB1_330 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ebp, %eax
	shrl	$8, %eax
	addl	80(%rsp), %eax                  # 4-byte Folded Reload
	movq	%rax, %rcx
	shlq	$7, %rcx
	leaq	(%rcx,%rax,8), %rax
	addq	56(%rsp), %rax                  # 8-byte Folded Reload
	movl	%ebp, %ecx
	andl	$384, %ecx                      # imm = 0x180
	addq	296(%rsp), %rcx                 # 8-byte Folded Reload
	movq	304(%rsp), %rdi                 # 8-byte Reload
	movzbl	16(%rdi,%rcx), %edx
	movzbl	(%rdi,%rcx), %esi
	shll	$8, %esi
	orl	%edx, %esi
	movzbl	32(%rdi,%rcx), %edx
	shll	$16, %edx
	orl	%esi, %edx
	movzbl	48(%rdi,%rcx), %r14d
	shll	$24, %r14d
	orl	%edx, %r14d
	movl	%r8d, %ecx
	andl	$992, %ecx                      # imm = 0x3E0
	addl	400(%rsp), %ecx                 # 4-byte Folded Reload
	movl	%r8d, 224(%rsp)                 # 4-byte Spill
	movl	%r8d, %edx
	andl	$16, %edx
	addl	8(%rsp), %edx                   # 4-byte Folded Reload
	orl	%ecx, %edx
	movl	%ebp, %ecx
	shrl	%ecx
	movl	%ecx, %esi
	andl	$3, %esi
	leal	(%rsi,%rdx,4), %edx
	movl	%ebp, %esi
	shrl	$5, %esi
	andl	$4, %esi
	movl	(%rsi,%rax), %r13d
	movzbl	(%rbx,%rdx), %r15d
	andl	$127, %ecx
	movzbl	8(%rcx,%rax), %r12d
	movl	%r9d, 136(%rsp)                 # 4-byte Spill
	movl	%r9d, %ecx
	andb	$4, %cl
	shrl	%cl, %r15d
	shrl	%cl, %r12d
	movl	%r13d, %ebx
	shrl	$16, %ebx
	pinsrw	$0, %r13d, %xmm0
	.cfi_escape 0x2e, 0x00
	callq	__extendhfsf2@PLT
	movd	%xmm0, 32(%rsp)                 # 4-byte Folded Spill
	pinsrw	$0, %ebx, %xmm0
	.cfi_escape 0x2e, 0x00
	callq	__extendhfsf2@PLT
	cmpl	%r13d, %r14d
	jne	.LBB1_337
# %bb.332:                              #   in Loop: Header=BB1_331 Depth=2
	andl	$15, %r15d
	andl	$15, %r12d
	cmpl	%r12d, %r15d
	jne	.LBB1_337
# %bb.333:                              #   in Loop: Header=BB1_331 Depth=2
	cvtsi2ss	%r15d, %xmm1
	movss	32(%rsp), %xmm3                 # 4-byte Reload
                                        # xmm3 = mem[0],zero,zero,zero
	mulss	%xmm3, %xmm1
	addss	%xmm0, %xmm1
	cvtsi2ss	%r12d, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm0, %xmm2
	ucomiss	%xmm2, %xmm1
	jne	.LBB1_337
	jp	.LBB1_337
# %bb.334:                              #   in Loop: Header=BB1_331 Depth=2
	incl	%ebp
	movl	136(%rsp), %r9d                 # 4-byte Reload
	addl	$4, %r9d
	movl	224(%rsp), %r8d                 # 4-byte Reload
	addl	$2, %r8d
	cmpl	$512, %ebp                      # imm = 0x200
	movq	72(%rsp), %rbx                  # 8-byte Reload
	jne	.LBB1_331
# %bb.335:                              # %.critedge97.i
                                        #   in Loop: Header=BB1_330 Depth=1
	movq	64(%rsp), %rcx                  # 8-byte Reload
	incl	%ecx
	cmpl	$32, %ecx
	movq	288(%rsp), %r14                 # 8-byte Reload
	jne	.LBB1_330
# %bb.336:                              # %.critedge99.i
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.10(%rip), %rdi
	xorl	%ebp, %ebp
	movl	$8704, %r8d                     # imm = 0x2200
	movl	$19, %esi
	movl	$32, %edx
	movl	$13, %ecx
	xorl	%eax, %eax
	callq	printf@PLT
	jmp	.LBB1_338
.LBB1_337:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.9(%rip), %rsi
	movq	64(%rsp), %rdx                  # 8-byte Reload
                                        # kill: def $edx killed $edx killed $rdx
	movl	%ebp, %ecx
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$1, %ebp
	movq	72(%rsp), %rbx                  # 8-byte Reload
	movq	288(%rsp), %r14                 # 8-byte Reload
.LBB1_338:                              # %_ZL21cpu_permutation_proofv.exit
	movq	280(%rsp), %rsi                 # 8-byte Reload
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
	movq	128(%rsp), %rsi
	subq	%rbx, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$8704, %esi                     # imm = 0x2200
	movq	56(%rsp), %rdi                  # 8-byte Reload
.LBB1_339:
	callq	_ZdlPvm@PLT
.LBB1_340:
	movl	%ebp, %eax
	addq	$504, %rsp                      # imm = 0x1F8
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB1_341:
	.cfi_def_cfa_offset 560
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB1_342:
	movq	112(%rsp), %rax
	movq	%rax, 152(%rsp)
	movq	144(%rsp), %rcx
	movb	$0, (%rcx,%rax)
	movq	144(%rsp), %rdi
	cmpq	$5, 152(%rsp)
	jne	.LBB1_344
# %bb.343:
	movl	$1667332197, %eax               # imm = 0x63617865
	xorl	(%rdi), %eax
	movzbl	4(%rdi), %ecx
	xorl	$116, %ecx
	orl	%eax, %ecx
	sete	%bl
	cmpq	%r15, %rdi
	jne	.LBB1_345
	jmp	.LBB1_346
.LBB1_344:
	xorl	%ebx, %ebx
	cmpq	%r15, %rdi
	je	.LBB1_346
.LBB1_345:                              # %_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv.exit.i.i384
	movq	160(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_346:                              # %.critedge362
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rdi
	xorl	%esi, %esi
	xorl	%eax, %eax
	callq	open@PLT
	testl	%eax, %eax
	js	.LBB1_350
# %bb.347:
	movl	%eax, %r15d
	testb	%bl, %bl
	je	.LBB1_9
# %bb.348:
.Ltmp3:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4, %edi
	callq	_Znwm@PLT
.Ltmp4:                                 # EH_LABEL
# %bb.349:                              # %.critedge365
	movq	%rax, 472(%rsp)
	movq	%rax, %rcx
	addq	$4, %rcx
	movq	%rcx, 488(%rsp)
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movl	$128, (%rax)
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	movq	%rcx, 480(%rsp)
	movl	$256, %ebx                      # imm = 0x100
	movb	$1, %al
	movl	%eax, 224(%rsp)                 # 4-byte Spill
	movl	$128, %eax
	jmp	.LBB1_11
.LBB1_350:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.3(%rip), %rdi
	callq	perror@PLT
	movl	$2, %ebp
	jmp	.LBB1_340
.LBB1_351:                              # %.thread.i
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rcx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.8(%rip), %rdi
	movl	$25, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	movl	$1, %ebp
	jmp	.LBB1_338
.LBB1_352:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.17(%rip), %rsi
	leaq	.L.str.20(%rip), %rdx
	movl	%r12d, %ecx
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB1_353:
	movq	%rcx, %rdx
	jmp	.LBB1_382
.LBB1_381:
	movq	%r8, %rdx
.LBB1_382:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.16(%rip), %rsi
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB1_354:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.19(%rip), %rdi
	callq	perror@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB1_355:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp102:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp103:                               # EH_LABEL
# %bb.356:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$427, %ecx                      # imm = 0x1AB
	jmp	.LBB1_387
.LBB1_357:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp93:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp94:                                # EH_LABEL
# %bb.358:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$425, %ecx                      # imm = 0x1A9
	jmp	.LBB1_387
.LBB1_359:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp159:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp160:                               # EH_LABEL
# %bb.360:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB1_372
.LBB1_361:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp165:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp166:                               # EH_LABEL
# %bb.362:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$481, %ecx                      # imm = 0x1E1
	jmp	.LBB1_387
.LBB1_363:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp132:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp133:                               # EH_LABEL
# %bb.364:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB1_367
.LBB1_365:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp138:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp139:                               # EH_LABEL
# %bb.366:
	.cfi_escape 0x2e, 0x00
.LBB1_367:
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$477, %ecx                      # imm = 0x1DD
	jmp	.LBB1_387
.LBB1_368:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp144:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp145:                               # EH_LABEL
# %bb.369:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$478, %ecx                      # imm = 0x1DE
	jmp	.LBB1_387
.LBB1_370:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp153:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp154:                               # EH_LABEL
# %bb.371:
	.cfi_escape 0x2e, 0x00
.LBB1_372:
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$480, %ecx                      # imm = 0x1E0
	jmp	.LBB1_387
.LBB1_373:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp174:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp175:                               # EH_LABEL
# %bb.374:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB1_377
.LBB1_375:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp180:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp181:                               # EH_LABEL
# %bb.376:
	.cfi_escape 0x2e, 0x00
.LBB1_377:
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$482, %ecx                      # imm = 0x1E2
	jmp	.LBB1_387
.LBB1_383:
.Ltmp214:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.11(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp215:                               # EH_LABEL
# %bb.384:                              # %.noexc463
.LBB1_385:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp211:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp212:                               # EH_LABEL
# %bb.386:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$449, %ecx                      # imm = 0x1C1
.LBB1_387:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB1_388:
.Ltmp188:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.22(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp189:                               # EH_LABEL
# %bb.389:                              # %.noexc475
.LBB1_390:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp20:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp21:                                # EH_LABEL
# %bb.391:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$288, %ecx                      # imm = 0x120
	jmp	.LBB1_387
.LBB1_392:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp26:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp27:                                # EH_LABEL
# %bb.393:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$289, %ecx                      # imm = 0x121
	jmp	.LBB1_387
.LBB1_394:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp32:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp33:                                # EH_LABEL
# %bb.395:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$290, %ecx                      # imm = 0x122
	jmp	.LBB1_387
.LBB1_396:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp38:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp39:                                # EH_LABEL
# %bb.397:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$291, %ecx                      # imm = 0x123
	jmp	.LBB1_387
.LBB1_398:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp58:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp59:                                # EH_LABEL
# %bb.399:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$414, %ecx                      # imm = 0x19E
	jmp	.LBB1_387
.LBB1_400:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp63:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp64:                                # EH_LABEL
# %bb.401:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$415, %ecx                      # imm = 0x19F
	jmp	.LBB1_387
.LBB1_402:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp68:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp69:                                # EH_LABEL
# %bb.403:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$416, %ecx                      # imm = 0x1A0
	jmp	.LBB1_387
.LBB1_404:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp73:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp74:                                # EH_LABEL
# %bb.405:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$417, %ecx                      # imm = 0x1A1
	jmp	.LBB1_387
.LBB1_406:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp78:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp79:                                # EH_LABEL
# %bb.407:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$418, %ecx                      # imm = 0x1A2
	jmp	.LBB1_387
.LBB1_408:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp83:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp84:                                # EH_LABEL
# %bb.409:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$419, %ecx                      # imm = 0x1A3
	jmp	.LBB1_387
.LBB1_410:
.Ltmp217:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.11(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp218:                               # EH_LABEL
# %bb.411:                              # %.noexc434
.LBB1_412:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp110:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp111:                               # EH_LABEL
# %bb.413:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$433, %ecx                      # imm = 0x1B1
	jmp	.LBB1_387
.LBB1_414:                              # %.noexc
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.21(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.LBB1_415:
.Ltmp105:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.11(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp106:                               # EH_LABEL
# %bb.416:                              # %.noexc438
.LBB1_417:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp126:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	movq	24(%rsp), %r15                  # 8-byte Reload
	callq	hipGetErrorString@PLT
.Ltmp127:                               # EH_LABEL
# %bb.418:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$474, %ecx                      # imm = 0x1DA
	jmp	.LBB1_387
.LBB1_419:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp202:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp203:                               # EH_LABEL
# %bb.420:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	movq	%r14, %rdi
	movl	$445, %ecx                      # imm = 0x1BD
	jmp	.LBB1_387
.LBB1_421:                              # %.noexc379
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.21(%rip), %rdi
	callq	_ZSt19__throw_logic_errorPKc@PLT
.LBB1_422:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rcx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.15(%rip), %rdi
	movl	$36, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB1_423:                              # %.body.thread
.Ltmp5:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_Unwind_Resume@PLT
.LBB1_424:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit.i.i
.Ltmp228:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB1_426
.LBB1_425:
.Ltmp225:                               # EH_LABEL
	movq	%rax, %r13
.LBB1_426:                              # %.body.i
	testq	%rbx, %rbx
	je	.LBB1_430
# %bb.427:
	movq	128(%rsp), %rsi
	subq	%rbx, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
	jmp	.LBB1_430
.LBB1_428:
.Ltmp193:                               # EH_LABEL
	movq	%rax, %r13
	jmp	.LBB1_501
.LBB1_429:
.Ltmp222:                               # EH_LABEL
	movq	%rax, %r13
.LBB1_430:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit109.i
	.cfi_escape 0x2e, 0x00
	movl	$8704, %esi                     # imm = 0x2200
	movq	56(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_Unwind_Resume@PLT
.LBB1_431:
.Ltmp43:                                # EH_LABEL
	jmp	.LBB1_513
.LBB1_432:
.Ltmp199:                               # EH_LABEL
	jmp	.LBB1_479
.LBB1_433:                              # %.loopexit.split-lp706
.Ltmp40:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_434:                              # %.loopexit.split-lp701
.Ltmp34:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_435:                              # %.loopexit.split-lp696
.Ltmp28:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_436:                              # %.loopexit.split-lp691
.Ltmp22:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_437:
.Ltmp187:                               # EH_LABEL
	jmp	.LBB1_488
.LBB1_438:
.Ltmp204:                               # EH_LABEL
	jmp	.LBB1_479
.LBB1_439:
.Ltmp128:                               # EH_LABEL
	jmp	.LBB1_500
.LBB1_440:                              # %.body.thread609
.Ltmp2:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_Unwind_Resume@PLT
.LBB1_441:
.Ltmp55:                                # EH_LABEL
	movq	%rax, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB1_509
.LBB1_442:
.Ltmp52:                                # EH_LABEL
	movq	%rax, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB1_510
.LBB1_443:
.Ltmp49:                                # EH_LABEL
	jmp	.LBB1_445
.LBB1_444:
.Ltmp46:                                # EH_LABEL
.LBB1_445:                              # %.thread1352
	movq	%rax, %r15
	jmp	.LBB1_515
.LBB1_446:                              # %.loopexit705
.Ltmp37:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_447:                              # %.loopexit700
.Ltmp31:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_448:                              # %.loopexit695
.Ltmp25:                                # EH_LABEL
	jmp	.LBB1_450
.LBB1_449:                              # %.loopexit690
.Ltmp19:                                # EH_LABEL
.LBB1_450:
	movq	%rax, %r15
	testq	%r13, %r13
	je	.LBB1_514
# %bb.451:
	movq	128(%rsp), %rsi
	subq	%r13, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
	jmp	.LBB1_514
.LBB1_452:
.Ltmp16:                                # EH_LABEL
	jmp	.LBB1_513
.LBB1_453:                              # %.thread1348
.Ltmp107:                               # EH_LABEL
	movq	%rax, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	movq	56(%rsp), %rdi                  # 8-byte Reload
	jmp	.LBB1_507
.LBB1_454:                              # %.loopexit.split-lp.i.loopexit
.Ltmp13:                                # EH_LABEL
	jmp	.LBB1_513
.LBB1_455:
.Ltmp112:                               # EH_LABEL
	jmp	.LBB1_500
.LBB1_456:
.Ltmp85:                                # EH_LABEL
	jmp	.LBB1_461
.LBB1_457:
.Ltmp80:                                # EH_LABEL
	jmp	.LBB1_461
.LBB1_458:
.Ltmp75:                                # EH_LABEL
	jmp	.LBB1_461
.LBB1_459:
.Ltmp70:                                # EH_LABEL
	jmp	.LBB1_461
.LBB1_460:
.Ltmp65:                                # EH_LABEL
.LBB1_461:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit509
	movq	%rax, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	jmp	.LBB1_508
.LBB1_462:
.Ltmp60:                                # EH_LABEL
	movq	%rax, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB1_508
.LBB1_463:
.Ltmp219:                               # EH_LABEL
	movq	%rax, %r13
	movq	16(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB1_508
.LBB1_464:
.Ltmp123:                               # EH_LABEL
	jmp	.LBB1_500
.LBB1_465:                              # %.loopexit.split-lp624
.Ltmp213:                               # EH_LABEL
	jmp	.LBB1_481
.LBB1_466:
.Ltmp196:                               # EH_LABEL
	jmp	.LBB1_479
.LBB1_467:                              # %.loopexit.split-lp661
.Ltmp190:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_468:                              # %.loopexit660
.Ltmp170:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_469:                              # %.loopexit
.Ltmp207:                               # EH_LABEL
	jmp	.LBB1_479
.LBB1_470:                              # %.loopexit.split-lp671
.Ltmp182:                               # EH_LABEL
	jmp	.LBB1_488
.LBB1_471:                              # %.loopexit.split-lp666
.Ltmp176:                               # EH_LABEL
	jmp	.LBB1_488
.LBB1_472:                              # %.loopexit.split-lp646
.Ltmp155:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_473:                              # %.loopexit.split-lp641
.Ltmp146:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_474:                              # %.loopexit.split-lp636
.Ltmp140:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_475:                              # %.loopexit.split-lp631
.Ltmp134:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_476:                              # %.loopexit.split-lp656
.Ltmp167:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_477:                              # %.loopexit.split-lp651
.Ltmp161:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_478:                              # %.loopexit.split-lp
.Ltmp216:                               # EH_LABEL
.LBB1_479:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movq	%rax, %r13
	jmp	.LBB1_483
.LBB1_480:                              # %.loopexit623
.Ltmp210:                               # EH_LABEL
.LBB1_481:
	movq	%rax, %r13
	cmpq	$0, 72(%rsp)                    # 8-byte Folded Reload
	je	.LBB1_483
# %bb.482:
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movq	432(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_483:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit
	.cfi_escape 0x2e, 0x00
	leaq	320(%rsp), %rdi
	callq	_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev
	jmp	.LBB1_505
.LBB1_484:                              # %.loopexit.split-lp678
.Ltmp95:                                # EH_LABEL
	jmp	.LBB1_504
.LBB1_485:                              # %.loopexit.split-lp683
.Ltmp104:                               # EH_LABEL
	jmp	.LBB1_504
.LBB1_486:                              # %.loopexit670
.Ltmp179:                               # EH_LABEL
	jmp	.LBB1_488
.LBB1_487:                              # %.loopexit665
.Ltmp173:                               # EH_LABEL
.LBB1_488:
	movq	%rax, %r13
	jmp	.LBB1_497
.LBB1_489:                              # %.loopexit650
.Ltmp158:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_490:                              # %.loopexit645
.Ltmp152:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_491:
.Ltmp149:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_492:                              # %.loopexit640
.Ltmp143:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_493:                              # %.loopexit635
.Ltmp137:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_494:                              # %.loopexit630
.Ltmp131:                               # EH_LABEL
	jmp	.LBB1_496
.LBB1_495:                              # %.loopexit655
.Ltmp164:                               # EH_LABEL
.LBB1_496:
	movq	%rax, %r13
	movq	%rbp, %r12
.LBB1_497:
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	movq	24(%rsp), %r15                  # 8-byte Reload
	testq	%r12, %r12
	je	.LBB1_506
# %bb.498:
	movq	32(%rsp), %rsi                  # 8-byte Reload
	subq	%r12, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
	jmp	.LBB1_506
.LBB1_499:                              # %.loopexit677
.Ltmp92:                                # EH_LABEL
.LBB1_500:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit480
	movq	%rax, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
.LBB1_501:
	movq	16(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB1_506
.LBB1_502:                              # %.loopexit682
.Ltmp101:                               # EH_LABEL
	jmp	.LBB1_504
.LBB1_503:
.Ltmp98:                                # EH_LABEL
.LBB1_504:
	movq	%rax, %r13
.LBB1_505:
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	movq	24(%rsp), %r15                  # 8-byte Reload
.LBB1_506:
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	callq	_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev
	movq	144(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB1_508
.LBB1_507:
	movq	160(%rsp), %rsi
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB1_508:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit509
	.cfi_escape 0x2e, 0x00
	movq	384(%rsp), %rdi                 # 8-byte Reload
	movq	352(%rsp), %rsi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB1_509:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit511
	.cfi_escape 0x2e, 0x00
	movq	376(%rsp), %rdi                 # 8-byte Reload
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB1_510:
	.cfi_escape 0x2e, 0x00
	movq	360(%rsp), %rdi                 # 8-byte Reload
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	leaq	256(%rsp), %rdi
	callq	_ZNSt6vectorI6WeightSaIS0_EED2Ev
.LBB1_511:
	subq	%r14, %rbx
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_Unwind_Resume@PLT
.LBB1_512:                              # %.loopexit.i
.Ltmp10:                                # EH_LABEL
.LBB1_513:                              # %.body405
	movq	%rax, %r15
.LBB1_514:                              # %.body405
	.cfi_escape 0x2e, 0x00
	leaq	144(%rsp), %rdi
	callq	_ZN6WeightD2Ev
.LBB1_515:                              # %.thread1352
	.cfi_escape 0x2e, 0x00
	leaq	256(%rsp), %rdi
	callq	_ZNSt6vectorI6WeightSaIS0_EED2Ev
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	16(%rsp), %rbx                  # 8-byte Reload
	movq	%r15, %r13
	jmp	.LBB1_511
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table1:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp7-.Ltmp6                  #   Call between .Ltmp6 and .Ltmp7
	.uleb128 .Ltmp13-.Lfunc_begin0          #     jumps to .Ltmp13
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp7-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp8-.Ltmp7                  #   Call between .Ltmp7 and .Ltmp8
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp8-.Lfunc_begin0           # >> Call Site 5 <<
	.uleb128 .Ltmp9-.Ltmp8                  #   Call between .Ltmp8 and .Ltmp9
	.uleb128 .Ltmp10-.Lfunc_begin0          #     jumps to .Ltmp10
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp12-.Ltmp11                #   Call between .Ltmp11 and .Ltmp12
	.uleb128 .Ltmp13-.Lfunc_begin0          #     jumps to .Ltmp13
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp14-.Ltmp12                #   Call between .Ltmp12 and .Ltmp14
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp14-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp15-.Ltmp14                #   Call between .Ltmp14 and .Ltmp15
	.uleb128 .Ltmp16-.Lfunc_begin0          #     jumps to .Ltmp16
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp17-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp18-.Ltmp17                #   Call between .Ltmp17 and .Ltmp18
	.uleb128 .Ltmp19-.Lfunc_begin0          #     jumps to .Ltmp19
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp23-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp24-.Ltmp23                #   Call between .Ltmp23 and .Ltmp24
	.uleb128 .Ltmp25-.Lfunc_begin0          #     jumps to .Ltmp25
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp29-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp30-.Ltmp29                #   Call between .Ltmp29 and .Ltmp30
	.uleb128 .Ltmp31-.Lfunc_begin0          #     jumps to .Ltmp31
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp35-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp36-.Ltmp35                #   Call between .Ltmp35 and .Ltmp36
	.uleb128 .Ltmp37-.Lfunc_begin0          #     jumps to .Ltmp37
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp41-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp42-.Ltmp41                #   Call between .Ltmp41 and .Ltmp42
	.uleb128 .Ltmp43-.Lfunc_begin0          #     jumps to .Ltmp43
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp44-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp45-.Ltmp44                #   Call between .Ltmp44 and .Ltmp45
	.uleb128 .Ltmp46-.Lfunc_begin0          #     jumps to .Ltmp46
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp47-.Lfunc_begin0          # >> Call Site 15 <<
	.uleb128 .Ltmp48-.Ltmp47                #   Call between .Ltmp47 and .Ltmp48
	.uleb128 .Ltmp49-.Lfunc_begin0          #     jumps to .Ltmp49
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp48-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Ltmp50-.Ltmp48                #   Call between .Ltmp48 and .Ltmp50
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp50-.Lfunc_begin0          # >> Call Site 17 <<
	.uleb128 .Ltmp51-.Ltmp50                #   Call between .Ltmp50 and .Ltmp51
	.uleb128 .Ltmp52-.Lfunc_begin0          #     jumps to .Ltmp52
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp51-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp53-.Ltmp51                #   Call between .Ltmp51 and .Ltmp53
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp53-.Lfunc_begin0          # >> Call Site 19 <<
	.uleb128 .Ltmp54-.Ltmp53                #   Call between .Ltmp53 and .Ltmp54
	.uleb128 .Ltmp55-.Lfunc_begin0          #     jumps to .Ltmp55
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp54-.Lfunc_begin0          # >> Call Site 20 <<
	.uleb128 .Ltmp56-.Ltmp54                #   Call between .Ltmp54 and .Ltmp56
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp56-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Ltmp57-.Ltmp56                #   Call between .Ltmp56 and .Ltmp57
	.uleb128 .Ltmp60-.Lfunc_begin0          #     jumps to .Ltmp60
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp61-.Lfunc_begin0          # >> Call Site 22 <<
	.uleb128 .Ltmp62-.Ltmp61                #   Call between .Ltmp61 and .Ltmp62
	.uleb128 .Ltmp65-.Lfunc_begin0          #     jumps to .Ltmp65
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp66-.Lfunc_begin0          # >> Call Site 23 <<
	.uleb128 .Ltmp67-.Ltmp66                #   Call between .Ltmp66 and .Ltmp67
	.uleb128 .Ltmp70-.Lfunc_begin0          #     jumps to .Ltmp70
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp71-.Lfunc_begin0          # >> Call Site 24 <<
	.uleb128 .Ltmp72-.Ltmp71                #   Call between .Ltmp71 and .Ltmp72
	.uleb128 .Ltmp75-.Lfunc_begin0          #     jumps to .Ltmp75
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp76-.Lfunc_begin0          # >> Call Site 25 <<
	.uleb128 .Ltmp77-.Ltmp76                #   Call between .Ltmp76 and .Ltmp77
	.uleb128 .Ltmp80-.Lfunc_begin0          #     jumps to .Ltmp80
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp81-.Lfunc_begin0          # >> Call Site 26 <<
	.uleb128 .Ltmp82-.Ltmp81                #   Call between .Ltmp81 and .Ltmp82
	.uleb128 .Ltmp85-.Lfunc_begin0          #     jumps to .Ltmp85
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp86-.Lfunc_begin0          # >> Call Site 27 <<
	.uleb128 .Ltmp87-.Ltmp86                #   Call between .Ltmp86 and .Ltmp87
	.uleb128 .Ltmp219-.Lfunc_begin0         #     jumps to .Ltmp219
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp87-.Lfunc_begin0          # >> Call Site 28 <<
	.uleb128 .Ltmp88-.Ltmp87                #   Call between .Ltmp87 and .Ltmp88
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp88-.Lfunc_begin0          # >> Call Site 29 <<
	.uleb128 .Ltmp89-.Ltmp88                #   Call between .Ltmp88 and .Ltmp89
	.uleb128 .Ltmp107-.Lfunc_begin0         #     jumps to .Ltmp107
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp89-.Lfunc_begin0          # >> Call Site 30 <<
	.uleb128 .Ltmp90-.Ltmp89                #   Call between .Ltmp89 and .Ltmp90
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp90-.Lfunc_begin0          # >> Call Site 31 <<
	.uleb128 .Ltmp91-.Ltmp90                #   Call between .Ltmp90 and .Ltmp91
	.uleb128 .Ltmp92-.Lfunc_begin0          #     jumps to .Ltmp92
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp96-.Lfunc_begin0          # >> Call Site 32 <<
	.uleb128 .Ltmp97-.Ltmp96                #   Call between .Ltmp96 and .Ltmp97
	.uleb128 .Ltmp98-.Lfunc_begin0          #     jumps to .Ltmp98
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp99-.Lfunc_begin0          # >> Call Site 33 <<
	.uleb128 .Ltmp100-.Ltmp99               #   Call between .Ltmp99 and .Ltmp100
	.uleb128 .Ltmp101-.Lfunc_begin0         #     jumps to .Ltmp101
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp108-.Lfunc_begin0         # >> Call Site 34 <<
	.uleb128 .Ltmp109-.Ltmp108              #   Call between .Ltmp108 and .Ltmp109
	.uleb128 .Ltmp112-.Lfunc_begin0         #     jumps to .Ltmp112
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp191-.Lfunc_begin0         # >> Call Site 35 <<
	.uleb128 .Ltmp192-.Ltmp191              #   Call between .Ltmp191 and .Ltmp192
	.uleb128 .Ltmp193-.Lfunc_begin0         #     jumps to .Ltmp193
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp192-.Lfunc_begin0         # >> Call Site 36 <<
	.uleb128 .Ltmp194-.Ltmp192              #   Call between .Ltmp192 and .Ltmp194
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp194-.Lfunc_begin0         # >> Call Site 37 <<
	.uleb128 .Ltmp195-.Ltmp194              #   Call between .Ltmp194 and .Ltmp195
	.uleb128 .Ltmp196-.Lfunc_begin0         #     jumps to .Ltmp196
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp195-.Lfunc_begin0         # >> Call Site 38 <<
	.uleb128 .Ltmp113-.Ltmp195              #   Call between .Ltmp195 and .Ltmp113
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp113-.Lfunc_begin0         # >> Call Site 39 <<
	.uleb128 .Ltmp122-.Ltmp113              #   Call between .Ltmp113 and .Ltmp122
	.uleb128 .Ltmp123-.Lfunc_begin0         #     jumps to .Ltmp123
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp124-.Lfunc_begin0         # >> Call Site 40 <<
	.uleb128 .Ltmp125-.Ltmp124              #   Call between .Ltmp124 and .Ltmp125
	.uleb128 .Ltmp128-.Lfunc_begin0         #     jumps to .Ltmp128
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp129-.Lfunc_begin0         # >> Call Site 41 <<
	.uleb128 .Ltmp130-.Ltmp129              #   Call between .Ltmp129 and .Ltmp130
	.uleb128 .Ltmp131-.Lfunc_begin0         #     jumps to .Ltmp131
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp135-.Lfunc_begin0         # >> Call Site 42 <<
	.uleb128 .Ltmp136-.Ltmp135              #   Call between .Ltmp135 and .Ltmp136
	.uleb128 .Ltmp137-.Lfunc_begin0         #     jumps to .Ltmp137
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp141-.Lfunc_begin0         # >> Call Site 43 <<
	.uleb128 .Ltmp142-.Ltmp141              #   Call between .Ltmp141 and .Ltmp142
	.uleb128 .Ltmp143-.Lfunc_begin0         #     jumps to .Ltmp143
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp147-.Lfunc_begin0         # >> Call Site 44 <<
	.uleb128 .Ltmp148-.Ltmp147              #   Call between .Ltmp147 and .Ltmp148
	.uleb128 .Ltmp149-.Lfunc_begin0         #     jumps to .Ltmp149
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp150-.Lfunc_begin0         # >> Call Site 45 <<
	.uleb128 .Ltmp151-.Ltmp150              #   Call between .Ltmp150 and .Ltmp151
	.uleb128 .Ltmp152-.Lfunc_begin0         #     jumps to .Ltmp152
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp156-.Lfunc_begin0         # >> Call Site 46 <<
	.uleb128 .Ltmp157-.Ltmp156              #   Call between .Ltmp156 and .Ltmp157
	.uleb128 .Ltmp158-.Lfunc_begin0         #     jumps to .Ltmp158
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp162-.Lfunc_begin0         # >> Call Site 47 <<
	.uleb128 .Ltmp163-.Ltmp162              #   Call between .Ltmp162 and .Ltmp163
	.uleb128 .Ltmp164-.Lfunc_begin0         #     jumps to .Ltmp164
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp168-.Lfunc_begin0         # >> Call Site 48 <<
	.uleb128 .Ltmp169-.Ltmp168              #   Call between .Ltmp168 and .Ltmp169
	.uleb128 .Ltmp170-.Lfunc_begin0         #     jumps to .Ltmp170
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp169-.Lfunc_begin0         # >> Call Site 49 <<
	.uleb128 .Ltmp171-.Ltmp169              #   Call between .Ltmp169 and .Ltmp171
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp171-.Lfunc_begin0         # >> Call Site 50 <<
	.uleb128 .Ltmp172-.Ltmp171              #   Call between .Ltmp171 and .Ltmp172
	.uleb128 .Ltmp173-.Lfunc_begin0         #     jumps to .Ltmp173
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp177-.Lfunc_begin0         # >> Call Site 51 <<
	.uleb128 .Ltmp178-.Ltmp177              #   Call between .Ltmp177 and .Ltmp178
	.uleb128 .Ltmp179-.Lfunc_begin0         #     jumps to .Ltmp179
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp183-.Lfunc_begin0         # >> Call Site 52 <<
	.uleb128 .Ltmp186-.Ltmp183              #   Call between .Ltmp183 and .Ltmp186
	.uleb128 .Ltmp187-.Lfunc_begin0         #     jumps to .Ltmp187
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp186-.Lfunc_begin0         # >> Call Site 53 <<
	.uleb128 .Ltmp220-.Ltmp186              #   Call between .Ltmp186 and .Ltmp220
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp220-.Lfunc_begin0         # >> Call Site 54 <<
	.uleb128 .Ltmp221-.Ltmp220              #   Call between .Ltmp220 and .Ltmp221
	.uleb128 .Ltmp222-.Lfunc_begin0         #     jumps to .Ltmp222
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp223-.Lfunc_begin0         # >> Call Site 55 <<
	.uleb128 .Ltmp224-.Ltmp223              #   Call between .Ltmp223 and .Ltmp224
	.uleb128 .Ltmp225-.Lfunc_begin0         #     jumps to .Ltmp225
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp224-.Lfunc_begin0         # >> Call Site 56 <<
	.uleb128 .Ltmp226-.Ltmp224              #   Call between .Ltmp224 and .Ltmp226
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp226-.Lfunc_begin0         # >> Call Site 57 <<
	.uleb128 .Ltmp227-.Ltmp226              #   Call between .Ltmp226 and .Ltmp227
	.uleb128 .Ltmp228-.Lfunc_begin0         #     jumps to .Ltmp228
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp227-.Lfunc_begin0         # >> Call Site 58 <<
	.uleb128 .Ltmp197-.Ltmp227              #   Call between .Ltmp227 and .Ltmp197
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp197-.Lfunc_begin0         # >> Call Site 59 <<
	.uleb128 .Ltmp198-.Ltmp197              #   Call between .Ltmp197 and .Ltmp198
	.uleb128 .Ltmp199-.Lfunc_begin0         #     jumps to .Ltmp199
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp200-.Lfunc_begin0         # >> Call Site 60 <<
	.uleb128 .Ltmp201-.Ltmp200              #   Call between .Ltmp200 and .Ltmp201
	.uleb128 .Ltmp204-.Lfunc_begin0         #     jumps to .Ltmp204
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp205-.Lfunc_begin0         # >> Call Site 61 <<
	.uleb128 .Ltmp206-.Ltmp205              #   Call between .Ltmp205 and .Ltmp206
	.uleb128 .Ltmp207-.Lfunc_begin0         #     jumps to .Ltmp207
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp206-.Lfunc_begin0         # >> Call Site 62 <<
	.uleb128 .Ltmp208-.Ltmp206              #   Call between .Ltmp206 and .Ltmp208
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp208-.Lfunc_begin0         # >> Call Site 63 <<
	.uleb128 .Ltmp209-.Ltmp208              #   Call between .Ltmp208 and .Ltmp209
	.uleb128 .Ltmp210-.Lfunc_begin0         #     jumps to .Ltmp210
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp209-.Lfunc_begin0         # >> Call Site 64 <<
	.uleb128 .Ltmp3-.Ltmp209                #   Call between .Ltmp209 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 65 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp102-.Lfunc_begin0         # >> Call Site 66 <<
	.uleb128 .Ltmp103-.Ltmp102              #   Call between .Ltmp102 and .Ltmp103
	.uleb128 .Ltmp104-.Lfunc_begin0         #     jumps to .Ltmp104
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp93-.Lfunc_begin0          # >> Call Site 67 <<
	.uleb128 .Ltmp94-.Ltmp93                #   Call between .Ltmp93 and .Ltmp94
	.uleb128 .Ltmp95-.Lfunc_begin0          #     jumps to .Ltmp95
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp159-.Lfunc_begin0         # >> Call Site 68 <<
	.uleb128 .Ltmp160-.Ltmp159              #   Call between .Ltmp159 and .Ltmp160
	.uleb128 .Ltmp161-.Lfunc_begin0         #     jumps to .Ltmp161
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp165-.Lfunc_begin0         # >> Call Site 69 <<
	.uleb128 .Ltmp166-.Ltmp165              #   Call between .Ltmp165 and .Ltmp166
	.uleb128 .Ltmp167-.Lfunc_begin0         #     jumps to .Ltmp167
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp132-.Lfunc_begin0         # >> Call Site 70 <<
	.uleb128 .Ltmp133-.Ltmp132              #   Call between .Ltmp132 and .Ltmp133
	.uleb128 .Ltmp134-.Lfunc_begin0         #     jumps to .Ltmp134
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp138-.Lfunc_begin0         # >> Call Site 71 <<
	.uleb128 .Ltmp139-.Ltmp138              #   Call between .Ltmp138 and .Ltmp139
	.uleb128 .Ltmp140-.Lfunc_begin0         #     jumps to .Ltmp140
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp144-.Lfunc_begin0         # >> Call Site 72 <<
	.uleb128 .Ltmp145-.Ltmp144              #   Call between .Ltmp144 and .Ltmp145
	.uleb128 .Ltmp146-.Lfunc_begin0         #     jumps to .Ltmp146
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp153-.Lfunc_begin0         # >> Call Site 73 <<
	.uleb128 .Ltmp154-.Ltmp153              #   Call between .Ltmp153 and .Ltmp154
	.uleb128 .Ltmp155-.Lfunc_begin0         #     jumps to .Ltmp155
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp174-.Lfunc_begin0         # >> Call Site 74 <<
	.uleb128 .Ltmp175-.Ltmp174              #   Call between .Ltmp174 and .Ltmp175
	.uleb128 .Ltmp176-.Lfunc_begin0         #     jumps to .Ltmp176
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp180-.Lfunc_begin0         # >> Call Site 75 <<
	.uleb128 .Ltmp181-.Ltmp180              #   Call between .Ltmp180 and .Ltmp181
	.uleb128 .Ltmp182-.Lfunc_begin0         #     jumps to .Ltmp182
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp214-.Lfunc_begin0         # >> Call Site 76 <<
	.uleb128 .Ltmp215-.Ltmp214              #   Call between .Ltmp214 and .Ltmp215
	.uleb128 .Ltmp216-.Lfunc_begin0         #     jumps to .Ltmp216
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp211-.Lfunc_begin0         # >> Call Site 77 <<
	.uleb128 .Ltmp212-.Ltmp211              #   Call between .Ltmp211 and .Ltmp212
	.uleb128 .Ltmp213-.Lfunc_begin0         #     jumps to .Ltmp213
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp188-.Lfunc_begin0         # >> Call Site 78 <<
	.uleb128 .Ltmp189-.Ltmp188              #   Call between .Ltmp188 and .Ltmp189
	.uleb128 .Ltmp190-.Lfunc_begin0         #     jumps to .Ltmp190
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp20-.Lfunc_begin0          # >> Call Site 79 <<
	.uleb128 .Ltmp21-.Ltmp20                #   Call between .Ltmp20 and .Ltmp21
	.uleb128 .Ltmp22-.Lfunc_begin0          #     jumps to .Ltmp22
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp26-.Lfunc_begin0          # >> Call Site 80 <<
	.uleb128 .Ltmp27-.Ltmp26                #   Call between .Ltmp26 and .Ltmp27
	.uleb128 .Ltmp28-.Lfunc_begin0          #     jumps to .Ltmp28
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp32-.Lfunc_begin0          # >> Call Site 81 <<
	.uleb128 .Ltmp33-.Ltmp32                #   Call between .Ltmp32 and .Ltmp33
	.uleb128 .Ltmp34-.Lfunc_begin0          #     jumps to .Ltmp34
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp38-.Lfunc_begin0          # >> Call Site 82 <<
	.uleb128 .Ltmp39-.Ltmp38                #   Call between .Ltmp38 and .Ltmp39
	.uleb128 .Ltmp40-.Lfunc_begin0          #     jumps to .Ltmp40
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp58-.Lfunc_begin0          # >> Call Site 83 <<
	.uleb128 .Ltmp59-.Ltmp58                #   Call between .Ltmp58 and .Ltmp59
	.uleb128 .Ltmp60-.Lfunc_begin0          #     jumps to .Ltmp60
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp63-.Lfunc_begin0          # >> Call Site 84 <<
	.uleb128 .Ltmp64-.Ltmp63                #   Call between .Ltmp63 and .Ltmp64
	.uleb128 .Ltmp65-.Lfunc_begin0          #     jumps to .Ltmp65
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp68-.Lfunc_begin0          # >> Call Site 85 <<
	.uleb128 .Ltmp69-.Ltmp68                #   Call between .Ltmp68 and .Ltmp69
	.uleb128 .Ltmp70-.Lfunc_begin0          #     jumps to .Ltmp70
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp73-.Lfunc_begin0          # >> Call Site 86 <<
	.uleb128 .Ltmp74-.Ltmp73                #   Call between .Ltmp73 and .Ltmp74
	.uleb128 .Ltmp75-.Lfunc_begin0          #     jumps to .Ltmp75
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp78-.Lfunc_begin0          # >> Call Site 87 <<
	.uleb128 .Ltmp79-.Ltmp78                #   Call between .Ltmp78 and .Ltmp79
	.uleb128 .Ltmp80-.Lfunc_begin0          #     jumps to .Ltmp80
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp83-.Lfunc_begin0          # >> Call Site 88 <<
	.uleb128 .Ltmp84-.Ltmp83                #   Call between .Ltmp83 and .Ltmp84
	.uleb128 .Ltmp85-.Lfunc_begin0          #     jumps to .Ltmp85
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp217-.Lfunc_begin0         # >> Call Site 89 <<
	.uleb128 .Ltmp218-.Ltmp217              #   Call between .Ltmp217 and .Ltmp218
	.uleb128 .Ltmp219-.Lfunc_begin0         #     jumps to .Ltmp219
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp110-.Lfunc_begin0         # >> Call Site 90 <<
	.uleb128 .Ltmp111-.Ltmp110              #   Call between .Ltmp110 and .Ltmp111
	.uleb128 .Ltmp112-.Lfunc_begin0         #     jumps to .Ltmp112
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp111-.Lfunc_begin0         # >> Call Site 91 <<
	.uleb128 .Ltmp105-.Ltmp111              #   Call between .Ltmp111 and .Ltmp105
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp105-.Lfunc_begin0         # >> Call Site 92 <<
	.uleb128 .Ltmp106-.Ltmp105              #   Call between .Ltmp105 and .Ltmp106
	.uleb128 .Ltmp107-.Lfunc_begin0         #     jumps to .Ltmp107
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp126-.Lfunc_begin0         # >> Call Site 93 <<
	.uleb128 .Ltmp127-.Ltmp126              #   Call between .Ltmp126 and .Ltmp127
	.uleb128 .Ltmp128-.Lfunc_begin0         #     jumps to .Ltmp128
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp202-.Lfunc_begin0         # >> Call Site 94 <<
	.uleb128 .Ltmp203-.Ltmp202              #   Call between .Ltmp202 and .Ltmp203
	.uleb128 .Ltmp204-.Lfunc_begin0         #     jumps to .Ltmp204
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp203-.Lfunc_begin0         # >> Call Site 95 <<
	.uleb128 .Lfunc_end1-.Ltmp203           #   Call between .Ltmp203 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN6WeightD2Ev,"axG",@progbits,_ZN6WeightD2Ev,comdat
	.weak	_ZN6WeightD2Ev                  # -- Begin function _ZN6WeightD2Ev
	.p2align	1
	.prefalign	4, .Lfunc_end2, nop
	.type	_ZN6WeightD2Ev,@function
_ZN6WeightD2Ev:                         # @_ZN6WeightD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	movq	32(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB2_2
# %bb.1:
	movq	48(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB2_2:                                # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movq	8(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB2_3
# %bb.4:
	movq	24(%rbx), %rsi
	subq	%rdi, %rsi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB2_3:                                # %_ZNSt6vectorIhSaIhEED2Ev.exit
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	_ZN6WeightD2Ev, .Lfunc_end2-_ZN6WeightD2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end3, nop     # -- Begin function _ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	.type	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii,@function
_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii: # @_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$184, %rsp
	.cfi_def_cfa_offset 208
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%r9, %rbx
	movq	%r8, %r14
	movl	$9728, %r8d                     # imm = 0x2600
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	je	.LBB3_1
# %bb.2:
	addq	$184, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB3_1:
	.cfi_def_cfa_offset 208
	movl	240(%rsp), %eax
	movq	232(%rsp), %rcx
	movq	224(%rsp), %rdx
	movq	216(%rsp), %rsi
	movq	208(%rsp), %rdi
	movq	(%r14), %r8
	movq	56(%r8), %r9
	movq	64(%r8), %r8
	movq	(%rdx), %rdx
	movq	(%rdx), %rdx
	movq	(%rcx), %rcx
	movl	(%rcx), %ecx
	movq	%r9, 104(%rsp)
	movq	%r8, 96(%rsp)
	movq	%rbx, 88(%rsp)
	movq	%rdi, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movl	%ecx, 12(%rsp)
	movl	$17408, 8(%rsp)                 # imm = 0x4400
	movl	%eax, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201@GOTPCREL(%rip), %rdi
	leaq	112(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	addq	$184, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii, .Lfunc_end3-_ZL6launch4dim3S_mP12ihipStream_tRKSt6vectorI6WeightSaIS3_EEPhPfS9_RKS2_IS9_SaIS9_EERKS2_IiSaIiEEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev,"axG",@progbits,_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev,comdat
	.weak	_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev # -- Begin function _ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev
	.p2align	1
	.prefalign	4, .Lfunc_end4, nop
	.type	_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev,@function
_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev:    # @_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %r14
	movq	(%rdi), %rbx
	movq	8(%rdi), %r15
	cmpq	%r15, %rbx
	jne	.LBB4_1
# %bb.5:                                # %_ZSt8_DestroyIPSt6vectorIfSaIfEES2_EvT_S4_RSaIT0_E.exit
	testq	%rbx, %rbx
	je	.LBB4_6
.LBB4_7:
	movq	16(%r14), %rsi
	subq	%rbx, %rsi
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
	.p2align	4
.LBB4_3:                                # %_ZSt8_DestroyISt6vectorIfSaIfEEEvPT_.exit.i.i
                                        #   in Loop: Header=BB4_1 Depth=1
	.cfi_def_cfa_offset 32
	addq	$24, %rbx
	cmpq	%r15, %rbx
	je	.LBB4_4
.LBB4_1:                                # %.lr.ph.i.i
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB4_3
# %bb.2:                                #   in Loop: Header=BB4_1 Depth=1
	movq	16(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB4_3
.LBB4_4:                                # %_ZSt8_DestroyIPSt6vectorIfSaIfEES2_EvT_S4_RSaIT0_E.exitthread-pre-split
	movq	(%r14), %rbx
	testq	%rbx, %rbx
	jne	.LBB4_7
.LBB4_6:                                # %_ZNSt12_Vector_baseISt6vectorIfSaIfEESaIS2_EED2Ev.exit
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev, .Lfunc_end4-_ZNSt6vectorIS_IfSaIfEESaIS1_EED2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt6vectorI6WeightSaIS0_EED2Ev,"axG",@progbits,_ZNSt6vectorI6WeightSaIS0_EED2Ev,comdat
	.weak	_ZNSt6vectorI6WeightSaIS0_EED2Ev # -- Begin function _ZNSt6vectorI6WeightSaIS0_EED2Ev
	.p2align	1
	.prefalign	4, .Lfunc_end5, nop
	.type	_ZNSt6vectorI6WeightSaIS0_EED2Ev,@function
_ZNSt6vectorI6WeightSaIS0_EED2Ev:       # @_ZNSt6vectorI6WeightSaIS0_EED2Ev
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %r14
	movq	(%rdi), %rbx
	movq	8(%rdi), %r15
	cmpq	%r15, %rbx
	jne	.LBB5_1
# %bb.7:                                # %_ZSt8_DestroyIP6WeightS0_EvT_S2_RSaIT0_E.exit
	testq	%rbx, %rbx
	je	.LBB5_8
.LBB5_9:
	movq	16(%r14), %rsi
	subq	%rbx, %rsi
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
	.p2align	4
.LBB5_5:                                # %_ZSt8_DestroyI6WeightEvPT_.exit.i.i
                                        #   in Loop: Header=BB5_1 Depth=1
	.cfi_def_cfa_offset 32
	addq	$72, %rbx
	cmpq	%r15, %rbx
	je	.LBB5_6
.LBB5_1:                                # %.lr.ph.i.i
                                        # =>This Inner Loop Header: Depth=1
	movq	32(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB5_3
# %bb.2:                                #   in Loop: Header=BB5_1 Depth=1
	movq	48(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB5_3:                                # %_ZNSt6vectorIfSaIfEED2Ev.exit.i.i.i.i
                                        #   in Loop: Header=BB5_1 Depth=1
	movq	8(%rbx), %rdi
	testq	%rdi, %rdi
	je	.LBB5_5
# %bb.4:                                #   in Loop: Header=BB5_1 Depth=1
	movq	24(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	jmp	.LBB5_5
.LBB5_6:                                # %_ZSt8_DestroyIP6WeightS0_EvT_S2_RSaIT0_E.exitthread-pre-split
	movq	(%r14), %rbx
	testq	%rbx, %rbx
	jne	.LBB5_9
.LBB5_8:                                # %_ZNSt12_Vector_baseI6WeightSaIS0_EED2Ev.exit
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end5:
	.size	_ZNSt6vectorI6WeightSaIS0_EED2Ev, .Lfunc_end5-_ZNSt6vectorI6WeightSaIS0_EED2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end6, nop     # -- Begin function _ZL17preshuffle_weightRKSt6vectorIhSaIhEEii
	.type	_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii,@function
_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii: # @_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception1
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%ecx, %ebx
	movl	%edx, %eax
	andl	$15, %eax
	movzbl	%bl, %ecx
	orl	%eax, %ecx
	jne	.LBB6_34
# %bb.1:                                # %_ZNSt6vectorIhSaIhEE17_S_check_init_lenEmRKS0_.exit.i
	movq	%rdi, %r15
	movq	(%rsi), %r13
	movq	8(%rsi), %r12
	subq	%r13, %r12
	je	.LBB6_4
# %bb.2:                                # %.noexc62
	movl	%edx, (%rsp)                    # 4-byte Spill
	movq	%r12, %rdi
	callq	_Znwm@PLT
	movq	%rax, %r14
	movq	%rax, (%r15)
	leaq	(%rax,%r12), %rbp
	movq	%rbp, 16(%r15)
	movb	$0, (%rax)
	leaq	1(%rax), %rdi
	movq	%r12, %rdx
	decq	%rdx
	je	.LBB6_5
# %bb.3:
	xorl	%esi, %esi
	callq	memset@PLT
	jmp	.LBB6_6
.LBB6_4:                                # %_ZNSt12_Vector_baseIhSaIhEEC2EmRKS0_.exit.thread.i65
	xorps	%xmm0, %xmm0
	movups	%xmm0, (%r15)
	movq	$0, 16(%r15)
	xorl	%r14d, %r14d
	xorl	%ebp, %ebp
	xorl	%r15d, %r15d
	movq	$0, 8(%rsp)                     # 8-byte Folded Spill
	testl	%edx, %edx
	jg	.LBB6_8
	jmp	.LBB6_24
.LBB6_5:
	movq	%rdi, %rbp
.LBB6_6:
	movq	%rbp, 8(%r15)
.Ltmp229:                               # EH_LABEL
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp230:                               # EH_LABEL
# %bb.7:                                # %.noexc67
	movq	%rax, %r15
	addq	%r12, %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%r15, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
	movl	(%rsp), %edx                    # 4-byte Reload
	testl	%edx, %edx
	jle	.LBB6_24
.LBB6_8:                                # %.preheader92.lr.ph
	movl	%ebx, %ecx
	movl	%ebx, %eax
	shrl	$8, %eax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movslq	%edx, %rax
	imulq	%rax, %rcx
	shrq	%rcx
	movq	%rcx, 24(%rsp)                  # 8-byte Spill
	shrq	$4, %rax
	movl	%ebx, %edi
	shrl	%edi
	movl	%ebx, %ecx
	shrl	$4, %ecx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	subq	%r14, %rbp
	shrl	$7, %ebx
	movl	%edx, %ecx
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	xorl	%ecx, %ecx
	movq	%rbx, 32(%rsp)                  # 8-byte Spill
	.p2align	4
.LBB6_9:                                # %.preheader92
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB6_10 Depth 2
                                        #     Child Loop BB6_14 Depth 2
	movq	48(%rsp), %r11                  # 8-byte Reload
	imulq	%rcx, %r11
	movl	%ecx, %r12d
	shrl	$4, %r12d
	movq	%r12, %rsi
	imulq	40(%rsp), %rsi                  # 8-byte Folded Reload
	movq	%rcx, (%rsp)                    # 8-byte Spill
	movl	%ecx, %r9d
	andl	$15, %r9d
	xorl	%r10d, %r10d
	xorl	%r8d, %r8d
	.p2align	4
.LBB6_10:                               #   Parent Loop BB6_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r8d, %ecx
	shrl	$3, %ecx
	addq	%rsi, %rcx
	movl	%r10d, %edx
	andl	$16, %edx
	orl	%r9d, %edx
	shlq	$7, %rcx
	leaq	(%rcx,%rdx,4), %rcx
	movl	%r8d, %edx
	andl	$3, %edx
	orq	%rcx, %rdx
	cmpq	%rbp, %rdx
	jae	.LBB6_33
# %bb.11:                               #   in Loop: Header=BB6_10 Depth=2
	cmpb	$0, (%r15,%rdx)
	jne	.LBB6_33
# %bb.12:                               # %_ZZL17preshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit
                                        #   in Loop: Header=BB6_10 Depth=2
	movl	%r8d, %ebx
	shrl	$7, %ebx
	addq	%r11, %rbx
	movq	%rbx, %rcx
	shlq	$7, %rcx
	leaq	(%rcx,%rbx,8), %rcx
	movl	%r8d, %ebx
	andl	$127, %ebx
	addq	%r13, %rcx
	movzbl	8(%rbx,%rcx), %ecx
	movb	%cl, (%r14,%rdx)
	movb	$1, (%r15,%rdx)
	incl	%r8d
	addl	$4, %r10d
	cmpl	%r8d, %edi
	jne	.LBB6_10
# %bb.13:                               # %.preheader
                                        #   in Loop: Header=BB6_9 Depth=1
	movq	(%rsp), %r10                    # 8-byte Reload
                                        # kill: def $r10d killed $r10d killed $r10 def $r10
	andl	$15, %r10d
	addq	24(%rsp), %r10                  # 8-byte Folded Reload
	xorl	%r9d, %r9d
	movq	32(%rsp), %rbx                  # 8-byte Reload
	.p2align	4
.LBB6_14:                               #   Parent Loop BB6_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rax, %rsi
	imulq	%r9, %rsi
	addq	%r12, %rsi
	shlq	$6, %rsi
	leaq	(%r10,%rsi), %rdx
	addq	$16, %rdx
	cmpq	%rbp, %rdx
	jae	.LBB6_33
# %bb.15:                               #   in Loop: Header=BB6_14 Depth=2
	cmpb	$0, (%r15,%rdx)
	jne	.LBB6_33
# %bb.16:                               # %_ZZL17preshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71
                                        #   in Loop: Header=BB6_14 Depth=2
	movl	%r9d, %ecx
	shrl	%ecx
	addq	%r11, %rcx
	movq	%rcx, %rdx
	shlq	$7, %rdx
	leaq	(%rdx,%rcx,8), %r8
	leal	(,%r9,4), %ecx
	andl	$4, %ecx
	addq	%r10, %rsi
	addq	%r13, %r8
	movzbl	(%rcx,%r8), %edx
	movb	%dl, 16(%r14,%rsi)
	movb	$1, 16(%r15,%rsi)
	cmpq	%rbp, %rsi
	jae	.LBB6_32
# %bb.17:                               #   in Loop: Header=BB6_14 Depth=2
	cmpb	$0, (%r15,%rsi)
	jne	.LBB6_32
# %bb.18:                               # %_ZZL17preshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.1
                                        #   in Loop: Header=BB6_14 Depth=2
	movzbl	1(%r8,%rcx), %edx
	movb	%dl, (%r14,%rsi)
	movb	$1, (%r15,%rsi)
	leaq	32(%rsi), %rdx
	cmpq	%rbp, %rdx
	jae	.LBB6_33
# %bb.19:                               #   in Loop: Header=BB6_14 Depth=2
	cmpb	$0, (%r15,%rdx)
	jne	.LBB6_33
# %bb.20:                               # %_ZZL17preshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.2
                                        #   in Loop: Header=BB6_14 Depth=2
	movzbl	2(%r8,%rcx), %edx
	movb	%dl, 32(%r14,%rsi)
	movb	$1, 32(%r15,%rsi)
	addq	$48, %rsi
	cmpq	%rbp, %rsi
	jae	.LBB6_32
# %bb.21:                               #   in Loop: Header=BB6_14 Depth=2
	cmpb	$0, (%r15,%rsi)
	jne	.LBB6_32
# %bb.22:                               # %_ZZL17preshuffle_weightRKSt6vectorIhSaIhEEiiENKUlmhE_clEmh.exit71.3
                                        #   in Loop: Header=BB6_14 Depth=2
	movzbl	3(%r8,%rcx), %ecx
	movb	%cl, (%r14,%rsi)
	movb	$1, (%r15,%rsi)
	incq	%r9
	cmpq	%rbx, %r9
	jne	.LBB6_14
# %bb.23:                               #   in Loop: Header=BB6_9 Depth=1
	movq	(%rsp), %rcx                    # 8-byte Reload
	incq	%rcx
	cmpq	16(%rsp), %rcx                  # 8-byte Folded Reload
	jne	.LBB6_9
.LBB6_24:                               # %._crit_edge
	movq	8(%rsp), %r14                   # 8-byte Reload
	movq	%r14, %rbx
	subq	%r15, %rbx
	testq	%rbx, %rbx
	jle	.LBB6_26
# %bb.25:
	movq	%r15, %rdi
	xorl	%esi, %esi
	movq	%rbx, %rdx
	callq	memchr@PLT
	testq	%rax, %rax
	setne	%cl
	cmpq	%r14, %rax
	setne	%al
	testb	%al, %cl
	jne	.LBB6_35
.LBB6_26:                               # %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPhSt6vectorIhSaIhEEEEhET_S7_S7_RKT0_.exit.thread
	testq	%r15, %r15
	je	.LBB6_28
# %bb.27:
	movq	%r15, %rdi
	movq	%rbx, %rsi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB6_28:                               # %_ZNSt6vectorIhSaIhEED2Ev.exit73
	.cfi_def_cfa_offset 112
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB6_32:
	.cfi_def_cfa_offset 112
	movq	%rsi, %rdx
.LBB6_33:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	leaq	.L.str.14(%rip), %rsi
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$2, %edi
	callq	exit@PLT
.LBB6_34:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	leaq	.L.str.12(%rip), %rsi
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$2, %edi
	callq	exit@PLT
.LBB6_35:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rcx
	leaq	.L.str.13(%rip), %rdi
	movl	$44, %esi
	movl	$1, %edx
	callq	fwrite@PLT
	movl	$2, %edi
	callq	exit@PLT
.LBB6_36:                               # %_ZNSt6vectorIhSaIhEED2Ev.exit
.Ltmp231:                               # EH_LABEL
	movq	%rax, %rbx
	movq	%r14, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end6:
	.size	_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii, .Lfunc_end6-_ZL17preshuffle_weightRKSt6vectorIhSaIhEEii
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table6:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp229-.Lfunc_begin1         #   Call between .Lfunc_begin1 and .Ltmp229
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp229-.Lfunc_begin1         # >> Call Site 2 <<
	.uleb128 .Ltmp230-.Ltmp229              #   Call between .Ltmp229 and .Ltmp230
	.uleb128 .Ltmp231-.Lfunc_begin1         #     jumps to .Ltmp231
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp230-.Lfunc_begin1         # >> Call Site 3 <<
	.uleb128 .Lfunc_end6-.Ltmp230           #   Call between .Ltmp230 and .Lfunc_end6
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt6vectorIfSaIfEE17_M_default_appendEm,"axG",@progbits,_ZNSt6vectorIfSaIfEE17_M_default_appendEm,comdat
	.weak	_ZNSt6vectorIfSaIfEE17_M_default_appendEm # -- Begin function _ZNSt6vectorIfSaIfEE17_M_default_appendEm
	.p2align	1
	.prefalign	4, .Lfunc_end7, nop
	.type	_ZNSt6vectorIfSaIfEE17_M_default_appendEm,@function
_ZNSt6vectorIfSaIfEE17_M_default_appendEm: # @_ZNSt6vectorIfSaIfEE17_M_default_appendEm
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	testq	%rsi, %rsi
	je	.LBB7_13
# %bb.1:
	movq	%rsi, %r14
	movq	%rdi, %rbx
	movq	8(%rdi), %r15
	movq	16(%rdi), %rcx
	movq	%rcx, %rax
	subq	%r15, %rax
	sarq	$2, %rax
	cmpq	%rsi, %rax
	jae	.LBB7_2
# %bb.5:
	movq	%rcx, 8(%rsp)                   # 8-byte Spill
	movq	(%rbx), %rax
	movq	%rax, %r12
	subq	%rax, %r15
	movq	%r15, %rcx
	sarq	$2, %rcx
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	movq	%rcx, %rdx
	xorq	%rax, %rdx
	cmpq	%r14, %rdx
	jb	.LBB7_14
# %bb.6:
	cmpq	%r14, %rcx
	movq	%r14, %r13
	cmovaq	%rcx, %r13
	addq	%rcx, %r13
	cmpq	%rax, %r13
	cmovaeq	%rax, %r13
	leaq	(,%r13,4), %rdi
	callq	_Znwm@PLT
	movq	%rax, %rbp
	addq	%r15, %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movl	$0, (%rbp,%r15)
	movq	%r14, %rdx
	decq	%rdx
	je	.LBB7_8
# %bb.7:                                # %_ZSt6fill_nIPfmfET_S1_T0_RKT1_.exit.loopexit.i.i.i29
	leaq	(%r15,%rbp), %rax
	leaq	4(%rax), %rdi
	shlq	$2, %rdx
	xorl	%esi, %esi
	callq	memset@PLT
.LBB7_8:                                # %_ZSt27__uninitialized_default_n_aIPfmfET_S1_T0_RSaIT1_E.exit32
	testq	%r15, %r15
	jle	.LBB7_10
# %bb.9:
	movq	%rbp, %rdi
	movq	%r12, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB7_10:                               # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit
	testq	%r12, %r12
	je	.LBB7_12
# %bb.11:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i
	movq	8(%rsp), %rsi                   # 8-byte Reload
	subq	%r12, %rsi
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
.LBB7_12:                               # %_ZNSt6vectorIfSaIfEE12_Guard_allocD2Ev.exit
	movq	%rbp, (%rbx)
	movq	16(%rsp), %rax                  # 8-byte Reload
	leaq	(%rax,%r14,4), %rax
	movq	%rax, 8(%rbx)
	leaq	(,%r13,4), %rax
	addq	%rbp, %rax
	movq	%rax, 16(%rbx)
	jmp	.LBB7_13
.LBB7_2:
	movl	$0, (%r15)
	addq	$4, %r15
	decq	%r14
	je	.LBB7_4
# %bb.3:                                # %_ZSt6fill_nIPfmfET_S1_T0_RKT1_.exit.loopexit.i.i.i
	leaq	(,%r14,4), %rdx
	movq	%r15, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	(%r15,%r14,4), %r15
.LBB7_4:                                # %_ZSt27__uninitialized_default_n_aIPfmfET_S1_T0_RSaIT1_E.exit
	movq	%r15, 8(%rbx)
.LBB7_13:
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB7_14:
	.cfi_def_cfa_offset 80
	leaq	.L.str.18(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end7:
	.size	_ZNSt6vectorIfSaIfEE17_M_default_appendEm, .Lfunc_end7-_ZNSt6vectorIfSaIfEE17_M_default_appendEm
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_,"axG",@progbits,_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_,comdat
	.weak	_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_ # -- Begin function _ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_
	.p2align	1
	.prefalign	4, .Lfunc_end8, nop
	.type	_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_,@function
_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_: # @_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	(%rdi), %rbx
	movq	%rdi, (%rsp)                    # 8-byte Spill
	movq	8(%rdi), %rbp
	movq	%rbp, %r14
	subq	%rbx, %r14
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r14
	je	.LBB8_7
# %bb.1:                                # %_ZNKSt6vectorI6WeightSaIS0_EE12_M_check_lenEmPKc.exit
	movq	%rsi, %r12
	movq	%r14, %rax
	shrq	$3, %rax
	movabsq	$-8198552921648689607, %r13     # imm = 0x8E38E38E38E38E39
	imulq	%rax, %r13
	cmpq	$1, %r13
	adcq	%r13, %r13
	movabsq	$128102389400760775, %rax       # imm = 0x1C71C71C71C71C7
	cmpq	%rax, %r13
	cmovaeq	%rax, %r13
	leaq	(,%r13,8), %rax
	leaq	(%rax,%rax,8), %rdi
	callq	_Znwm@PLT
	movq	%rax, %r15
	movq	(%r12), %rax
	movq	%rax, (%r15,%r14)
	movups	8(%r12), %xmm0
	movups	%xmm0, 8(%r15,%r14)
	movq	24(%r12), %rax
	movq	%rax, 24(%r15,%r14)
	xorps	%xmm0, %xmm0
	movups	%xmm0, 8(%r12)
	movq	$0, 24(%r12)
	movups	32(%r12), %xmm1
	movups	%xmm1, 32(%r15,%r14)
	movq	48(%r12), %rax
	movq	%rax, 48(%r15,%r14)
	movups	%xmm0, 32(%r12)
	movq	$0, 48(%r12)
	movups	56(%r12), %xmm1
	movups	%xmm1, 56(%r15,%r14)
	movq	%r15, %r12
	cmpq	%rbp, %rbx
	movq	%rbx, %rdi
	je	.LBB8_4
# %bb.2:                                # %.lr.ph.i.i.i.preheader
	movq	%r15, %r12
	movq	%rdi, %rax
	.p2align	4
.LBB8_3:                                # %.lr.ph.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rax), %rcx
	movq	%rcx, (%r12)
	movups	8(%rax), %xmm1
	movups	%xmm1, 8(%r12)
	movq	24(%rax), %rcx
	movq	%rcx, 24(%r12)
	movups	%xmm0, 8(%rax)
	movq	$0, 24(%rax)
	movups	32(%rax), %xmm1
	movups	%xmm1, 32(%r12)
	movq	48(%rax), %rcx
	movq	%rcx, 48(%r12)
	movups	%xmm0, 32(%rax)
	movq	$0, 48(%rax)
	movups	56(%rax), %xmm1
	movups	%xmm1, 56(%r12)
	addq	$72, %rax
	addq	$72, %r12
	cmpq	%rbp, %rax
	jne	.LBB8_3
.LBB8_4:                                # %_ZNSt6vectorI6WeightSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_.exit
	testq	%rdi, %rdi
	movq	(%rsp), %rbx                    # 8-byte Reload
	je	.LBB8_6
# %bb.5:                                # %_ZNSt12_Vector_baseI6WeightSaIS0_EE13_M_deallocateEPS0_m.exit.i
	movq	16(%rbx), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB8_6:                                # %_ZNSt6vectorI6WeightSaIS0_EE12_Guard_allocD2Ev.exit
	addq	$72, %r12
	movq	%r15, (%rbx)
	movq	%r12, 8(%rbx)
	leaq	(,%r13,8), %rax
	addq	%r13, %rax
	leaq	(%r15,%rax,8), %rax
	movq	%rax, 16(%rbx)
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB8_7:
	.cfi_def_cfa_offset 64
	leaq	.L.str.22(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end8:
	.size	_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_, .Lfunc_end8-_ZNSt6vectorI6WeightSaIS0_EE17_M_realloc_appendIJS0_EEEvDpOT_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf,"axG",@progbits,_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf,comdat
	.weak	_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf # -- Begin function _ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf
	.p2align	1
	.prefalign	4, .Lfunc_end9, nop
	.type	_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf,@function
_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf: # @_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r12
	movq	%rsi, %r15
	movq	%rdi, %rbx
	movq	(%rdi), %r14
	movq	16(%rdi), %r13
	subq	%r14, %r13
	movq	%r13, %rax
	sarq	$2, %rax
	cmpq	%rax, %rsi
	jbe	.LBB9_9
# %bb.1:
	movabsq	$2305843009213693944, %rbp      # imm = 0x1FFFFFFFFFFFFFF8
	leaq	7(%rbp), %rax
	cmpq	%rax, %r15
	ja	.LBB9_36
# %bb.2:                                # %.lr.ph.preheader.i.i.i.i.i
	leaq	(,%r15,4), %rdi
	callq	_Znwm@PLT
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movq	%r15, %rcx
	movq	%rax, %rdx
	cmpq	$8, %r15
	jb	.LBB9_6
# %bb.3:                                # %vector.ph64
	andq	%r15, %rbp
	movl	%r15d, %ecx
	andl	$7, %ecx
	leaq	(,%rbp,4), %rsi
	leaq	(%rax,%rbp,4), %rdx
	movaps	%xmm0, %xmm1
	shufps	$0, %xmm0, %xmm1                # xmm1 = xmm1[0,0],xmm0[0,0]
	xorl	%edi, %edi
	.p2align	4
.LBB9_4:                                # %vector.body69
                                        # =>This Inner Loop Header: Depth=1
	movups	%xmm1, (%rax,%rdi)
	movups	%xmm1, 16(%rax,%rdi)
	addq	$32, %rdi
	cmpq	%rdi, %rsi
	jne	.LBB9_4
# %bb.5:                                # %middle.block73
	cmpq	%rbp, %r15
	je	.LBB9_7
	.p2align	4
.LBB9_6:                                # %.lr.ph.i.i.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movss	%xmm0, (%rdx)
	addq	$4, %rdx
	decq	%rcx
	jne	.LBB9_6
.LBB9_7:                                # %_ZNSt6vectorIfSaIfEEC2EmRKfRKS0_.exit
	leaq	(%rax,%r15,4), %rcx
	movq	%rax, (%rbx)
	movq	%rdx, 8(%rbx)
	movq	%rcx, 16(%rbx)
	testq	%r14, %r14
	je	.LBB9_35
# %bb.8:
	movq	%r14, %rdi
	movq	%r13, %rsi
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT                     # TAILCALL
.LBB9_9:
	.cfi_def_cfa_offset 64
	movq	8(%rbx), %rax
	movq	%rax, %rdx
	subq	%r14, %rdx
	movq	%rdx, %rsi
	sarq	$2, %rsi
	movq	%r15, %rcx
	subq	%rsi, %rcx
	jbe	.LBB9_25
# %bb.10:
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cmpq	%rax, %r14
	je	.LBB9_18
# %bb.11:                               # %.lr.ph.i.i.i.i.preheader
	addq	$-4, %rdx
	cmpq	$28, %rdx
	jae	.LBB9_13
# %bb.12:
	movq	%r14, %rsi
	jmp	.LBB9_16
.LBB9_25:
	testq	%r15, %r15
	je	.LBB9_33
# %bb.26:
	leaq	(%r14,%r15,4), %rcx
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	leaq	-4(,%r15,4), %rsi
	cmpq	$28, %rsi
	jae	.LBB9_28
# %bb.27:
	movq	%r14, %rdx
	jmp	.LBB9_31
.LBB9_13:                               # %vector.ph35
	shrq	$2, %rdx
	incq	%rdx
	movq	%rdx, %rdi
	andq	$-8, %rdi
	leaq	(%r14,%rdi,4), %rsi
	movaps	%xmm0, %xmm1
	shufps	$0, %xmm0, %xmm1                # xmm1 = xmm1[0,0],xmm0[0,0]
	xorl	%r8d, %r8d
	.p2align	4
.LBB9_14:                               # %vector.body40
                                        # =>This Inner Loop Header: Depth=1
	movups	%xmm1, (%r14,%r8,4)
	movups	%xmm1, 16(%r14,%r8,4)
	addq	$8, %r8
	cmpq	%r8, %rdi
	jne	.LBB9_14
# %bb.15:                               # %middle.block44
	cmpq	%rdi, %rdx
	je	.LBB9_17
	.p2align	4
.LBB9_16:                               # %.lr.ph.i.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movss	%xmm0, (%rsi)
	addq	$4, %rsi
	cmpq	%rax, %rsi
	jne	.LBB9_16
.LBB9_17:                               # %_ZSt4fillIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEfEvT_S7_RKT0_.exit.loopexit
	movss	(%r12), %xmm0                   # xmm0 = mem[0],zero,zero,zero
.LBB9_18:                               # %_ZSt4fillIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEfEvT_S7_RKT0_.exit
	cmpq	$8, %rcx
	jae	.LBB9_20
# %bb.19:
	movq	%rcx, %rdx
	movq	%rax, %rsi
	jmp	.LBB9_23
.LBB9_20:                               # %vector.ph49
	movq	%rcx, %rdi
	andq	$-8, %rdi
	movl	%ecx, %edx
	andl	$7, %edx
	leaq	(%rax,%rdi,4), %rsi
	movaps	%xmm0, %xmm1
	shufps	$0, %xmm0, %xmm1                # xmm1 = xmm1[0,0],xmm0[0,0]
	xorl	%r8d, %r8d
	.p2align	4
.LBB9_21:                               # %vector.body54
                                        # =>This Inner Loop Header: Depth=1
	movups	%xmm1, (%rax,%r8,4)
	movups	%xmm1, 16(%rax,%r8,4)
	addq	$8, %r8
	cmpq	%r8, %rdi
	jne	.LBB9_21
# %bb.22:                               # %middle.block58
	cmpq	%rdi, %rcx
	je	.LBB9_24
	.p2align	4
.LBB9_23:                               # %.lr.ph.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movss	%xmm0, (%rsi)
	addq	$4, %rsi
	decq	%rdx
	jne	.LBB9_23
.LBB9_24:                               # %_ZSt24__uninitialized_fill_n_aIPfmffET_S1_T0_RKT1_RSaIT2_E.exit
	movq	%rsi, 8(%rbx)
	jmp	.LBB9_35
.LBB9_28:                               # %vector.ph
	shrq	$2, %rsi
	incq	%rsi
	movq	%rsi, %rdi
	andq	$-8, %rdi
	leaq	(%r14,%rdi,4), %rdx
	movaps	%xmm0, %xmm1
	shufps	$0, %xmm0, %xmm1                # xmm1 = xmm1[0,0],xmm0[0,0]
	xorl	%r8d, %r8d
	.p2align	4
.LBB9_29:                               # %vector.body
                                        # =>This Inner Loop Header: Depth=1
	movups	%xmm1, (%r14,%r8,4)
	movups	%xmm1, 16(%r14,%r8,4)
	addq	$8, %r8
	cmpq	%r8, %rdi
	jne	.LBB9_29
# %bb.30:                               # %middle.block
	cmpq	%rdi, %rsi
	je	.LBB9_32
	.p2align	4
.LBB9_31:                               # %.lr.ph.i.i.i.i16
                                        # =>This Inner Loop Header: Depth=1
	movss	%xmm0, (%rdx)
	addq	$4, %rdx
	cmpq	%rcx, %rdx
	jne	.LBB9_31
.LBB9_32:
	movq	%rcx, %r14
.LBB9_33:                               # %_ZSt6fill_nIPfmfET_S1_T0_RKT1_.exit
	cmpq	%r14, %rax
	je	.LBB9_35
# %bb.34:
	movq	%r14, 8(%rbx)
.LBB9_35:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB9_36:
	.cfi_def_cfa_offset 64
	leaq	.L.str.11(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Lfunc_end9:
	.size	_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf, .Lfunc_end9-_ZNSt6vectorIfSaIfEE14_M_fill_assignEmRKf
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.prefalign	4, .Lfunc_end10, nop
	.type	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,@function
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_: # @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbp
	subq	%rdi, %rbp
	sarq	$2, %rbp
	cmpq	$17, %rbp
	jl	.LBB10_40
# %bb.1:                                # %.lr.ph
	movq	%rdx, %r14
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.LBB10_8
# %bb.2:                                # %.lr.ph43.preheader
	movq	$-4, %r13
	subq	%rbx, %r13
	.p2align	4
.LBB10_3:                               # %.lr.ph43
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB10_33 Depth 2
                                        #       Child Loop BB10_34 Depth 3
                                        #       Child Loop BB10_36 Depth 3
	shrq	%rbp
	movss	4(%rbx), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%rbx,%rbp,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm2
	movss	-4(%rsi), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	jbe	.LBB10_27
# %bb.4:                                #   in Loop: Header=BB10_3 Depth=1
	ucomiss	%xmm2, %xmm0
	jbe	.LBB10_24
# %bb.5:                                #   in Loop: Header=BB10_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm2, (%rbx)
	movss	%xmm0, (%rbx,%rbp,4)
	jmp	.LBB10_32
	.p2align	4
.LBB10_27:                              #   in Loop: Header=BB10_3 Depth=1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB10_29
# %bb.28:                               #   in Loop: Header=BB10_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx)
	movss	%xmm0, 4(%rbx)
	jmp	.LBB10_32
	.p2align	4
.LBB10_24:                              #   in Loop: Header=BB10_3 Depth=1
	ucomiss	%xmm1, %xmm0
	movss	(%rbx), %xmm2                   # xmm2 = mem[0],zero,zero,zero
	jbe	.LBB10_26
# %bb.25:                               #   in Loop: Header=BB10_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm2, -4(%rsi)
	jmp	.LBB10_32
	.p2align	4
.LBB10_29:                              #   in Loop: Header=BB10_3 Depth=1
	ucomiss	%xmm2, %xmm0
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	jbe	.LBB10_31
# %bb.30:                               #   in Loop: Header=BB10_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm1, -4(%rsi)
	jmp	.LBB10_32
.LBB10_26:                              #   in Loop: Header=BB10_3 Depth=1
	movss	%xmm1, (%rbx)
	movss	%xmm2, 4(%rbx)
	jmp	.LBB10_32
.LBB10_31:                              #   in Loop: Header=BB10_3 Depth=1
	movss	%xmm2, (%rbx)
	movss	%xmm1, (%rbx,%rbp,4)
	.p2align	4
.LBB10_32:                              # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i.preheader
                                        #   in Loop: Header=BB10_3 Depth=1
	decq	%r14
	leaq	4(%rbx), %r12
	movq	%rsi, %rax
	.p2align	4
.LBB10_33:                              # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i
                                        #   Parent Loop BB10_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB10_34 Depth 3
                                        #       Child Loop BB10_36 Depth 3
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	leaq	(%r12,%r13), %rbp
	.p2align	4
.LBB10_34:                              #   Parent Loop BB10_3 Depth=1
                                        #     Parent Loop BB10_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%r12), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	addq	$4, %r12
	addq	$4, %rbp
	ucomiss	%xmm1, %xmm0
	ja	.LBB10_34
# %bb.35:                               # %.preheader.i.i.preheader
                                        #   in Loop: Header=BB10_33 Depth=2
	leaq	-4(%r12), %r15
	.p2align	4
.LBB10_36:                              # %.preheader.i.i
                                        #   Parent Loop BB10_3 Depth=1
                                        #     Parent Loop BB10_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	-4(%rax), %xmm2                 # xmm2 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm2
	ja	.LBB10_36
# %bb.37:                               #   in Loop: Header=BB10_33 Depth=2
	cmpq	%rax, %r15
	jae	.LBB10_39
# %bb.38:                               #   in Loop: Header=BB10_33 Depth=2
	movss	%xmm2, (%r15)
	movss	%xmm1, (%rax)
	jmp	.LBB10_33
	.p2align	4
.LBB10_39:                              # %_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_.exit
                                        #   in Loop: Header=BB10_3 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	sarq	$2, %rbp
	cmpq	$16, %rbp
	jle	.LBB10_40
# %bb.6:                                #   in Loop: Header=BB10_3 Depth=1
	movq	%r15, %rsi
	testq	%r14, %r14
	jne	.LBB10_3
# %bb.7:                                # %._crit_edge.loopexit
	addq	$-4, %r12
	movq	%r12, %rsi
.LBB10_8:                               # %._crit_edge
	leaq	7(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%rsi, %r14
	callq	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	jmp	.LBB10_9
	.p2align	4
.LBB10_22:                              #   in Loop: Header=BB10_9 Depth=1
	xorl	%ecx, %ecx
.LBB10_23:                              # %_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_.exit.i.i
                                        #   in Loop: Header=BB10_9 Depth=1
	movss	%xmm0, (%rbx,%rcx,4)
	cmpq	$4, %rax
	jle	.LBB10_40
.LBB10_9:                               # %.lr.ph.i.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB10_12 Depth 2
                                        #     Child Loop BB10_20 Depth 2
	movss	-4(%r14), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, -4(%r14)
	addq	$-4, %r14
	movq	%r14, %rax
	subq	%rbx, %rax
	movq	%rax, %rdx
	sarq	$2, %rdx
	cmpq	$3, %rdx
	jl	.LBB10_10
# %bb.11:                               # %.lr.ph.i.i.i.i.preheader
                                        #   in Loop: Header=BB10_9 Depth=1
	leaq	-1(%rdx), %rcx
	shrq	$63, %rcx
	leaq	(%rdx,%rcx), %rsi
	decq	%rsi
	sarq	%rsi
	xorl	%edi, %edi
	jmp	.LBB10_12
	.p2align	4
.LBB10_14:                              # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB10_12 Depth=2
	leaq	2(,%rdi,2), %rcx
.LBB10_15:                              # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB10_12 Depth=2
	movss	(%rbx,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rdi,4)
	movq	%rcx, %rdi
	cmpq	%rsi, %rcx
	jge	.LBB10_16
.LBB10_12:                              # %.lr.ph.i.i.i.i
                                        #   Parent Loop BB10_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rdi,%rdi), %rcx
	movss	4(%rbx,%rcx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rbx,%rcx,4), %xmm1
	jbe	.LBB10_14
# %bb.13:                               #   in Loop: Header=BB10_12 Depth=2
	leaq	1(,%rdi,2), %rcx
	jmp	.LBB10_15
	.p2align	4
.LBB10_10:                              #   in Loop: Header=BB10_9 Depth=1
	xorl	%ecx, %ecx
.LBB10_16:                              # %._crit_edge.i.i.i.i
                                        #   in Loop: Header=BB10_9 Depth=1
	testb	$4, %al
	jne	.LBB10_19
# %bb.17:                               #   in Loop: Header=BB10_9 Depth=1
	addq	$-2, %rdx
	sarq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB10_19
# %bb.18:                               # %.thread.i.i.i
                                        #   in Loop: Header=BB10_9 Depth=1
	leaq	(%rcx,%rcx), %rdx
	movss	4(%rbx,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rcx,4)
	leaq	1(,%rcx,2), %rcx
	jmp	.LBB10_20
	.p2align	4
.LBB10_19:                              #   in Loop: Header=BB10_9 Depth=1
	testq	%rcx, %rcx
	je	.LBB10_22
	.p2align	4
.LBB10_20:                              # %.lr.ph.i.i.i.i.i
                                        #   Parent Loop BB10_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rcx), %rdx
	shrq	%rdx
	movss	(%rbx,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB10_23
# %bb.21:                               #   in Loop: Header=BB10_20 Depth=2
	movss	%xmm1, (%rbx,%rcx,4)
	movq	%rdx, %rcx
	testq	%rdx, %rdx
	jne	.LBB10_20
	jmp	.LBB10_22
.LBB10_40:                              # %_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_.exit
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end10:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end10-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ # -- Begin function _ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.prefalign	4, .Lfunc_end11, nop
	.type	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,@function
_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_: # @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movq	%rsi, %rax
	subq	%rdi, %rax
	cmpq	$65, %rax
	jl	.LBB11_17
# %bb.1:                                # %.lr.ph.i
	leaq	4(%r14), %r15
	movl	$4, %r12d
	movq	%r15, %r13
	movq	%r14, %rbp
	jmp	.LBB11_2
.LBB11_17:
	cmpq	%rbx, %r14
	je	.LBB11_29
# %bb.18:
	leaq	4(%r14), %rax
	cmpq	%rbx, %rax
	je	.LBB11_29
# %bb.19:                               # %.lr.ph.i15.preheader
	movq	%r14, %r15
	jmp	.LBB11_20
	.p2align	4
.LBB11_28:                              # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i18
                                        #   in Loop: Header=BB11_20 Depth=1
	movss	%xmm1, (%rax)
	leaq	4(%r15), %rax
	cmpq	%rbx, %rax
	je	.LBB11_29
.LBB11_20:                              # %.lr.ph.i15
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB11_27 Depth 2
	movq	%r15, %rdi
	movq	%rax, %r15
	movss	4(%rdi), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB11_25
# %bb.21:                               # %_ZSt7advanceIPflEvRT_T0_.exit.i.i.i.i.i28
                                        #   in Loop: Header=BB11_20 Depth=1
	movss	%xmm1, 4(%rsp)                  # 4-byte Spill
	movq	%r15, %rdx
	subq	%r14, %rdx
	subq	%rdx, %rdi
	movq	%rdx, %rax
	sarq	$2, %rax
	addq	$8, %rdi
	cmpq	$2, %rax
	jl	.LBB11_23
# %bb.22:                               #   in Loop: Header=BB11_20 Depth=1
	movq	%r14, %rsi
	callq	memmove@PLT
	movq	%r14, %rax
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jmp	.LBB11_28
	.p2align	4
.LBB11_25:                              #   in Loop: Header=BB11_20 Depth=1
	movss	(%rdi), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%r15, %rax
	jbe	.LBB11_28
# %bb.26:                               # %.lr.ph.i.i22.preheader
                                        #   in Loop: Header=BB11_20 Depth=1
	movq	%r15, %rax
	.p2align	4
.LBB11_27:                              # %.lr.ph.i.i22
                                        #   Parent Loop BB11_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB11_27
	jmp	.LBB11_28
.LBB11_23:                              # %_ZSt7advanceIPflEvRT_T0_.exit.thread.i.i.i.i.i29
                                        #   in Loop: Header=BB11_20 Depth=1
	movq	%r14, %rax
	cmpq	$4, %rdx
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jne	.LBB11_28
# %bb.24:                               #   in Loop: Header=BB11_20 Depth=1
	movss	%xmm0, (%rdi)
	movq	%r14, %rax
	jmp	.LBB11_28
.LBB11_5:                               # %_ZSt7advanceIPflEvRT_T0_.exit.thread.i.i.i.i.i
                                        #   in Loop: Header=BB11_2 Depth=1
	movss	%xmm0, (%r15)
	.p2align	4
.LBB11_6:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB11_2 Depth=1
	movq	%r14, %rax
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB11_10:                              # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB11_2 Depth=1
	movss	%xmm1, (%rax)
	addq	$4, %r12
	addq	$4, %r13
	cmpq	$64, %r12
	je	.LBB11_11
.LBB11_2:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB11_9 Depth 2
	movq	%rbp, %rax
	leaq	(%r14,%r12), %rbp
	movss	(%r14,%r12), %xmm1              # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB11_7
# %bb.3:                                # %_ZSt7advanceIPflEvRT_T0_.exit.i.i.i.i.i
                                        #   in Loop: Header=BB11_2 Depth=1
	movss	%xmm1, 4(%rsp)                  # 4-byte Spill
	cmpq	$5, %r12
	jb	.LBB11_5
# %bb.4:                                #   in Loop: Header=BB11_2 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
	jmp	.LBB11_6
	.p2align	4
.LBB11_7:                               #   in Loop: Header=BB11_2 Depth=1
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%rbp, %rax
	jbe	.LBB11_10
# %bb.8:                                # %.lr.ph.i.i.preheader
                                        #   in Loop: Header=BB11_2 Depth=1
	movq	%r13, %rax
	.p2align	4
.LBB11_9:                               # %.lr.ph.i.i
                                        #   Parent Loop BB11_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB11_9
	jmp	.LBB11_10
.LBB11_11:                              # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
	addq	$64, %r14
	jmp	.LBB11_12
	.p2align	4
.LBB11_16:                              # %_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops14_Val_less_iterEEvT_T0_.exit.i
                                        #   in Loop: Header=BB11_12 Depth=1
	movss	%xmm0, (%rax)
	addq	$4, %r14
.LBB11_12:                              # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB11_15 Depth 2
	cmpq	%rbx, %r14
	je	.LBB11_29
# %bb.13:                               # %.lr.ph.i6
                                        #   in Loop: Header=BB11_12 Depth=1
	movss	-4(%r14), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm0, %xmm1
	movq	%r14, %rax
	jbe	.LBB11_16
# %bb.14:                               # %.lr.ph.i.i8.preheader
                                        #   in Loop: Header=BB11_12 Depth=1
	movq	%r14, %rax
	.p2align	4
.LBB11_15:                              # %.lr.ph.i.i8
                                        #   Parent Loop BB11_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm1, (%rax)
	movss	-8(%rax), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm1
	ja	.LBB11_15
	jmp	.LBB11_16
.LBB11_29:                              # %_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end11:
	.size	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_, .Lfunc_end11-_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,"axG",@progbits,_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,comdat
	.weak	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ # -- Begin function _ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.prefalign	4, .Lfunc_end12, nop
	.type	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,@function
_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_: # @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_startproc
# %bb.0:
	subq	%rdi, %rsi
	movq	%rsi, %rax
	sarq	$2, %rax
	cmpq	$2, %rax
	jge	.LBB12_2
.LBB12_1:                               # %.loopexit
	retq
.LBB12_2:
	leaq	-2(%rax), %rdx
	movq	%rdx, %rcx
	shrq	%rcx
	decq	%rax
	shrq	%rax
	testb	$4, %sil
	jne	.LBB12_20
# %bb.3:                                # %.split.preheader
	incq	%rdx
	movq	%rcx, %rsi
	jmp	.LBB12_6
	.p2align	4
.LBB12_4:                               #   in Loop: Header=BB12_6 Depth=1
	movq	%r8, %r9
.LBB12_5:                               # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElfNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit
                                        #   in Loop: Header=BB12_6 Depth=1
	movss	%xmm0, (%rdi,%r9,4)
	subq	$1, %rsi
	jb	.LBB12_1
.LBB12_6:                               # %.split
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB12_10 Depth 2
                                        #     Child Loop BB12_15 Depth 2
	movss	(%rdi,%rsi,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rsi, %r8
	cmpq	%rax, %rsi
	jge	.LBB12_12
# %bb.7:                                # %.lr.ph.i.preheader
                                        #   in Loop: Header=BB12_6 Depth=1
	movq	%rsi, %r9
	jmp	.LBB12_10
	.p2align	4
.LBB12_8:                               # %.lr.ph.i
                                        #   in Loop: Header=BB12_10 Depth=2
	leaq	2(,%r9,2), %r8
.LBB12_9:                               # %.lr.ph.i
                                        #   in Loop: Header=BB12_10 Depth=2
	movss	(%rdi,%r8,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%r9,4)
	movq	%r8, %r9
	cmpq	%rax, %r8
	jge	.LBB12_12
.LBB12_10:                              # %.lr.ph.i
                                        #   Parent Loop BB12_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r9,%r9), %r8
	movss	4(%rdi,%r8,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rdi,%r8,4), %xmm1
	jbe	.LBB12_8
# %bb.11:                               #   in Loop: Header=BB12_10 Depth=2
	leaq	1(,%r9,2), %r8
	jmp	.LBB12_9
	.p2align	4
.LBB12_12:                              # %._crit_edge.i
                                        #   in Loop: Header=BB12_6 Depth=1
	cmpq	%rcx, %r8
	jne	.LBB12_14
# %bb.13:                               #   in Loop: Header=BB12_6 Depth=1
	movss	(%rdi,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%rcx,4)
	movq	%rdx, %r8
.LBB12_14:                              #   in Loop: Header=BB12_6 Depth=1
	cmpq	%rsi, %r8
	jle	.LBB12_4
	.p2align	4
.LBB12_15:                              # %.lr.ph.i.i
                                        #   Parent Loop BB12_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%r8), %r9
	shrq	$63, %r9
	addq	%r8, %r9
	decq	%r9
	sarq	%r9
	movss	(%rdi,%r9,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB12_4
# %bb.16:                               #   in Loop: Header=BB12_15 Depth=2
	movss	%xmm1, (%rdi,%r8,4)
	movq	%r9, %r8
	cmpq	%rsi, %r9
	jg	.LBB12_15
	jmp	.LBB12_5
	.p2align	4
.LBB12_18:                              #   in Loop: Header=BB12_20 Depth=1
	movq	%rdx, %rsi
.LBB12_19:                              # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElfNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit.us
                                        #   in Loop: Header=BB12_20 Depth=1
	movss	%xmm0, (%rdi,%rsi,4)
	subq	$1, %rcx
	jb	.LBB12_1
.LBB12_20:                              # %.split.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB12_24 Depth 2
                                        #     Child Loop BB12_27 Depth 2
	movss	(%rdi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rcx, %rsi
	cmpq	%rax, %rcx
	jge	.LBB12_19
# %bb.21:                               # %.lr.ph.i.us.preheader
                                        #   in Loop: Header=BB12_20 Depth=1
	movq	%rcx, %rsi
	jmp	.LBB12_24
	.p2align	4
.LBB12_22:                              # %.lr.ph.i.us
                                        #   in Loop: Header=BB12_24 Depth=2
	leaq	2(,%rsi,2), %rdx
.LBB12_23:                              # %.lr.ph.i.us
                                        #   in Loop: Header=BB12_24 Depth=2
	movss	(%rdi,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%rsi,4)
	movq	%rdx, %rsi
	cmpq	%rax, %rdx
	jge	.LBB12_26
.LBB12_24:                              # %.lr.ph.i.us
                                        #   Parent Loop BB12_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rsi,%rsi), %rdx
	movss	4(%rdi,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rdi,%rdx,4), %xmm1
	jbe	.LBB12_22
# %bb.25:                               #   in Loop: Header=BB12_24 Depth=2
	leaq	1(,%rsi,2), %rdx
	jmp	.LBB12_23
	.p2align	4
.LBB12_26:                              # %._crit_edge.i.us
                                        #   in Loop: Header=BB12_20 Depth=1
	cmpq	%rcx, %rdx
	jle	.LBB12_18
	.p2align	4
.LBB12_27:                              # %.lr.ph.i.i.us
                                        #   Parent Loop BB12_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rdx), %rsi
	shrq	$63, %rsi
	addq	%rdx, %rsi
	decq	%rsi
	sarq	%rsi
	movss	(%rdi,%rsi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB12_18
# %bb.28:                               #   in Loop: Header=BB12_27 Depth=2
	movss	%xmm1, (%rdi,%rdx,4)
	movq	%rsi, %rdx
	cmpq	%rcx, %rsi
	jg	.LBB12_27
	jmp	.LBB12_19
.Lfunc_end12:
	.size	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_, .Lfunc_end12-_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end13, nop    # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_4dbe48b018d034c5(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB13_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_4dbe48b018d034c5(%rip)
.LBB13_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	movl	$0, 8(%rsp)
	movl	$1, (%rsp)
	leaq	ff_fold_lut(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movl	$512, %r9d                      # imm = 0x200
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	xorl	%r8d, %r8d
	callq	__hipRegisterVar@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end13:
	.size	__hip_module_ctor, .Lfunc_end13-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end14, nop    # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_4dbe48b018d034c5(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB14_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_4dbe48b018d034c5(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB14_2:
	retq
.Lfunc_end14:
	.size	__hip_module_dtor, .Lfunc_end14-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	ff_fold_lut,@object             # @ff_fold_lut
	.local	ff_fold_lut
	.comm	ff_fold_lut,512,16
	.type	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201,@object # @gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.section	.data.rel.ro,"aw",@progbits
	.globl	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.p2align	3, 0x0
gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201:
	.quad	__device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.size	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201, 8

	.type	.L.str.2,@object                # @.str.2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.2:
	.asciz	"/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2h-a035.hfq"
	.size	.L.str.2, 65

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"open artifact"
	.size	.L.str.3, 14

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"%s:%d: %s\n"
	.size	.L.str.4, 11

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"bench_foldfree.hip"
	.size	.L.str.5, 19

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"family=%d exact rows=%d n=%d k=%d rel_rms=%.9g max_abs=%.9g nonfinite=%zu blocks_per_cu=%d\n"
	.size	.L.str.6, 92

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"family=%d m=%d n=%d k=%d median_us=%.3f tflops=%.3f blocks_per_cu=%d lds=%zu\n"
	.size	.L.str.7, 78

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"round-trip byte mismatch\n"
	.size	.L.str.8, 26

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"dequant mismatch row=%d col=%d\n"
	.size	.L.str.9, 32

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"cpu_proof bijection=ok roundtrip=ok dequant=bit_exact logical_rows=%d storage_rows=%d padded_tail_rows=%d bytes=%zu\n"
	.size	.L.str.10, 117

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.11, 49

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"preshuffle needs rows%%16=0 and k%%256=0\n"
	.size	.L.str.12, 42

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"preshuffle left destination bytes unwritten\n"
	.size	.L.str.13, 45

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"non-bijective destination %zu\n"
	.size	.L.str.14, 31

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"inverse left source bytes unwritten\n"
	.size	.L.str.15, 37

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"non-bijective inverse destination %zu\n"
	.size	.L.str.16, 39

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"%s row %d invalid pow2h d=0x%04x\n"
	.size	.L.str.17, 34

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"vector::_M_default_append"
	.size	.L.str.18, 26

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"pread"
	.size	.L.str.19, 6

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"residual"
	.size	.L.str.20, 9

	.type	.L_ZZL6make_xiiE6values.const,@object # @_ZZL6make_xiiE6values.const
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.L_ZZL6make_xiiE6values.const:
	.ascii	"\00008<@D\260\270\274\300(\250B\3024\264"
	.size	.L_ZZL6make_xiiE6values.const, 16

	.type	.L.str.21,@object               # @.str.21
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.21:
	.asciz	"basic_string: construction from null is not valid"
	.size	.L.str.21, 50

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"vector::_M_realloc_append"
	.size	.L.str.22, 26

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201"
	.size	.L__unnamed_1, 53

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"ff_fold_lut"
	.size	.L__unnamed_2, 12

	.type	.L__unnamed_3,@object           # @2
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_3:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\000\340l\000\000\000\000\000\000 \000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx1201\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000`h\000\000\000\000\000\000N\000\000\000@\0008\000\t\000@\000\022\000\020\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000t\n\000\000\000\000\000\000t\n\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\013\000\000\000\000\000\000\000\033\000\000\000\000\000\000\000\033\000\000\000\000\000\000\200Y\000\000\000\000\000\000\200Y\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\200d\000\000\000\000\000\000\200\204\000\000\000\000\000\000\200\204\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\013\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\360d\000\000\000\000\000\000\360\224\000\000\000\000\000\000\360\224\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\200d\000\000\000\000\000\000\200\204\000\000\000\000\000\000\200\204\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\200d\000\000\000\000\000\000\200\204\000\000\000\000\000\000\200\204\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\013\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000T\004\000\000\000\000\000\000T\004\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000@\004\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\336\000\023\245.args\231\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset(\245.size\b\253.value_kind\255global_buffer\203\247.offset0\245.size\004\253.value_kind\250by_value\203\247.offset4\245.size\004\253.value_kind\250by_value\203\247.offset8\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size<\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\3314gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201\273.private_segment_fixed_size\000\253.sgpr_count\036\261.sgpr_spill_count\000\247.symbol\3317gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\277\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\272amdgcn-amd-amdhsa--gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\b\000\000\033\000\000\000\000\000\000\234W\000\000\000\000\000\0006\000\000\000\021\003\006\000@\b\000\000\000\000\000\000\000\002\000\000\000\000\000\000B\000\000\000\021\003\006\000\000\b\000\000\000\000\000\000@\000\000\000\000\000\000\000z\000\000\000\021\000\013\000\360\224\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\001\000\200\204\000\002\t\000\001\000\000\0002\000o^\350\224\255j0\311+|\201\002\324k\005\000\000\000\005\000\000\000\000\000\000\000\004\000\000\000\002\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201\000ff_fold_lut\000gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.kd\000__hip_cuid_4dbe48b018d034c5\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000<\000\000\000\000\000\000\000\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\013\000\000\027\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\320\316\314\312\310\304\300\270\0008@DHJLN\310\306\304\302\300\274\270\260\00008<@BDF\300\276\274\272\270\264\260\250\000(048:<>\270\266\264\262\260\254\250\240\000 (,0246\260\256\254\252\250\244\240\230\000\030 $(*,.\250\246\244\242\240\234\230\220\000\020\030\034 \"$&\240\236\234\232\230\224\220\210\000\b\020\024\030\032\034\036\230\226\224\222\220\214\210\204\000\004\b\f\020\022\024\026\220\216\214\212\210\206\204\202\000\002\004\006\b\n\f\016\210\207\206\205\204\203\202\201\000\001\002\003\004\005\006\007\204\204\203\202\202\202\201\000\000\000\001\002\002\002\003\004\202\202\202\201\201\201\000\000\000\000\000\001\001\001\002\002\201\201\201\201\000\000\000\000\000\000\000\000\000\001\001\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\324\322\321\317\314\311\304\274\000<DILOQR\314\312\311\307\304\301\274\264\0004<ADGIJ\304\302\301\277\274\271\264\254\000,49<?AB\274\272\271\267\264\261\254\244\000$,1479:\264\262\261\257\254\251\244\234\000\034$),/12\254\252\251\247\244\241\234\224\000\024\034!$')*\244\242\241\237\234\231\224\214\000\f\024\031\034\037!\"\234\232\231\227\224\221\214\206\000\006\f\021\024\027\031\032\224\222\221\217\214\211\206\203\000\003\006\t\f\017\021\022\214\212\211\210\206\204\203\202\000\002\003\004\006\b\t\n\206\205\204\204\203\202\202\201\000\001\002\002\003\004\004\005\203\202\202\202\202\201\201\000\000\000\001\001\002\002\002\002\202\201\201\201\201\201\000\000\000\000\000\001\001\001\001\001\201\201\201\000\000\000\000\000\000\000\000\000\000\000\001\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\001zR\000\004\004\020\001\033\000\000\000\030\000\000\000\030\000\000\000\244\020\000\000\234W\000\000\000\017\00406\351\002\007\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\202\000\0041~\000\202\276\377\000\230}\200\000\000\000\r\000\245\277\000G\204\276\005\017\205\276\004\377\004\200(\355\377\377\005\377\005\202\377\377\377\377\200\004\005J\004\000\005\356\001\000\000\000\202\000\000\000\000\000\300\277\000$4\330\002\001\000\000~\002~\214\001\000\205\277\000\242\000\3640\000\000\370\000A\000\364\000\000\000\370\377\000\0026\300\000\000\000\000\000\306\277\301N\200\276\201\000\0243\200\000\203\276\210\000V\326s\020\005\004\001\000\207\277\204\020\0034\377\377\224\277\000\000\307\277\t\204\030\206\t\200\002\277\001\000,\327\0011\000\002Q\002\207\277\003|\376\326\030\006\005\004\030\002\bJ\005\000F\326\030\002\005\004\201\000\0040\001\005\004~\004\005 ~\004\000\207\277\005\005$~\003\005(~\004\000\242\277\201\000\0022\b\237\r\206\b\000\214\276\001\000\240\277\301\000\203\276\200\002\002\177\300\004\0227~\003j\221u\207\013\204\002\000\207\277\201\001\020\312\201\001~\200\201\001\020\312\201\001|~\201\001\020\312\201\001z|\201\001\020\312\201\001xz\201\001\020\312\201\001vx\201\001\020\312\201\001tv\201\001\020\312\201\001rt\201\001\020\312\201\001pr\201\001\020\312\201\001np\201\001\020\312\201\001ln\201\001\020\312\201\001jl\201\001\020\312\201\001hj\201\001\020\312\201\001fh\201\001\020\312\201\001df\201\001\020\312\201\001bd\201\001\020\312\201\001`b\201\001\020\312\201\001^`\201\001\020\312\201\001\\^\201\001\020\312\201\001Z\\\201\001\020\312\201\001XZ\201\001\020\312\201\001VX\201\001\020\312\201\001TV\201\001\020\312\201\001RT\201\001\020\312\201\001PR\201\001\020\312\201\001NP\201\001\020\312\201\001LN\201\001\020\312\201\001JL\201\001\020\312\201\001HJ\201\001\020\312\201\001FH\201\001\020\312\201\001DF\201\001\020\312\201\001BD\201\001\020\312\201\001@B\201\001\020\312\201\001>@\201\001\020\312\201\001<>\201\001\020\312\201\001:<\201\001\020\312\201\0018:\201\001\020\312\201\00168\201\001\020\312\201\00146\201\001\020\312\201\00124\201\001\020\312\201\00102\201\001\020\312\201\001.0\201\001\020\312\201\001,.\201\001\020\312\201\001*,\201\001\020\312\201\001(*\201\001\020\312\201\001&(\201\001\020\312\201\001$&\201\001\020\312\201\001\"$\201\001\020\312\201\001 \"\201\001\020\312\201\001\036 \201\001\020\312\201\001\034\036\201\001\020\312\201\001\032\034\201\001\020\312\201\001\030\032\201\001\020\312\201\001\026\030\201\001\020\312\201\001\024\026\201\001\020\312\201\001\022\024\201\001\020\312\201\001\020\022\201\001\020\312\201\001\016\020\201\001\020\312\201\001\f\016\201\001\020\312\201\001\n\f\201\001\020\312\201\001\b\n\201\001\020\312\201\001\006\b\201\001\020\312\201\001\004\006\201\001\020\312\201\001\002\004\201\003\004~\342\002\244\277\200&\000\364\020\000\000\370\200\000\"\312\203\000\b\002\214\004\0256\002\210\002\204\200\000\203\276\b\237\r\206\b\000\214\276\001\000W\326\377\024+\004p\000\000\000\203\000\0262\204\000\0300\b\204\026\206\002\001\020\312\002\001\016\016\013\002\fJ\001\000\020\326\000\007\t\002\002\001\020\312\002\001\020\020\002\001\020\312\002\001\022\022\004\000\207\277\204\f\0164\000\000\307\277\032\002\216\251\020\210\002\204\002\001\020\312\002\001\024\024\003|\376\326\030\016\006\004\032\002\220\251\022\210\002\204\237\016\0204\032\002\222\251\024\210\002\204\377\022\0026\370\000\000\000\032\002\224\251\t\000\202\276\004|\376\326\030\020\022\004\002\f\230\252\206\016\020>\236\377\210\277\030\201\230\205\237\f\0164\236\377\210\277\004\030\230\251\002\001\020\312\002\001\026\026\236\377\210\277\bj\000\327\030\020\002\002\202\f\n>\t| \325\031\022\252\001\263\001\207\277\202j\000\327\b\025\002\002\207\006\006>\235\377\210\277\203| \325\200\022\252\001\204j\000\327\006\n\002\002\205\000\n0\007\000\013\326\377\026\002\002 \001\000\000\235\377\210\277\205| \325\007\f\252\001\202\022\r2\377\n\n6\340\000\000\000\002\001\020\312\002\001\n\t\002\003\032~\024\002\207\277\377\f\f\026 \001\000\000\007\013\026K\002\003\016~\003\000W\326\377\030\016\004p\000\000\000\002\003\n~\002\001\020\312\002\001\f\013\002\001\020\312\002\001\030\030\004\000\207\277\206j\000\327\004\006\002\002\235\377\210\277\207| \325\005\b\252\001\002\003\b~\002\001 \312\200\002\b\003\002\001\020\312\002\001\032\032\002\001\020\312\002\001\034\034\003\000\207\277\b\r\030K\002\003\f~\002\003\020~\002\001\020\312\002\001\036\036\002\001\020\312\002\001  \002\001\020\312\002\001\"\"\002\001\020\312\002\001$$\002\001\020\312\002\001&&\002\001\020\312\002\001((\002\001\020\312\002\001**\002\001\020\312\002\001,,\002\001\020\312\002\001..\002\001\020\312\002\00100\002\001\020\312\002\00122\002\001\020\312\002\00144\002\001\020\312\002\00166\002\001\020\312\002\00188\002\001\020\312\002\001::\002\001\020\312\002\001<<\002\001\020\312\002\001>>\002\001\020\312\002\001@@\002\001\020\312\002\001BB\002\001\020\312\002\001DD\002\001\020\312\002\001FF\002\001\020\312\002\001HH\002\001\020\312\002\001JJ\002\001\020\312\002\001LL\002\001\020\312\002\001NN\002\001\020\312\002\001PP\002\001\020\312\002\001RR\002\001\020\312\002\001TT\002\001\020\312\002\001VV\002\001\020\312\002\001XX\002\001\020\312\002\001ZZ\002\001\020\312\002\001\\\\\002\001\020\312\002\001^^\002\001\020\312\002\001``\002\001\020\312\002\001bb\002\001\020\312\002\001dd\002\001\020\312\002\001ff\002\001\020\312\002\001hh\002\001\020\312\002\001jj\002\001\020\312\002\001ll\002\001\020\312\002\001nn\002\001\020\312\002\001pp\002\001\020\312\002\001rr\002\001\020\312\002\001tt\002\001\020\312\002\001vv\002\001\020\312\002\001xx\002\001\020\312\002\001zz\002\001\020\312\002\001||\002\001\020\312\002\001~~\002\001\020\312\002\001\200\200\377\030\033K\000\b\000\000\377\030\035K\000\f\000\000\026\237\027\206\177\000\002\260\026\206\204\204\377\000\226\276\004\004\004\004\003\207\027\205\007\000\205\277\016@\005\356\227\000\000\000\001\000\000\000\016@\005\356\231\000\000\000\001\000\001\000\016@\005\356\233\000\000\000\001\000\002\000\016@\005\356\235\000\000\000\001\000\003\000\020@\005\356\237\000\000\000\001\000\000\000\020@\005\356\241\000\000\000\001\000\001\000\020@\005\356\243\000\000\000\001\000\002\000\020@\005\356\245\000\000\000\001\000\003\000\236\377\210\277\223|\376\326\004.\b\006\007\000\205\277\022@\005\356\247\000\000\000\001\000\000\000\022@\005\356\251\000\000\000\001\000\001\000\022@\005\356\253\000\000\000\001\000\002\000\022@\005\356\255\000\000\000\001\000\003\000\024@\005\356\257\000\000\000\001\000\000\000\024@\005\356\261\000\000\000\001\000\001\000\024@\005\356\263\000\000\000\001\000\002\000\024@\005\356\265\000\000\000\001\000\003\000\224|\376\326\005.P\006|\300\005\356\217\000\000\000\204\000\000\000|\000\005\356\273\000\000\000\223\000\000\000|\300\005\356\223\000\000\000\206\000\000\000\002\000\300\277\217\000\020\326\217/!\002\001\000\300\277\267\000\020\326\273\005\025\002\203vq7\000\000\300\277\204&y3\220\000\020\326\220/!\002\221\000\020\326\221/!\002\217o\037M\200p\225|\262\000\207\277\217\000 \326\377\036\013\000p\000\000\000\235\377\210\277\267\000\001\325\220\000\251\001\217o\037K\221\000\207\277\217\000F\326\217\t\001\002\377\036oK\000\035\000\000\377\036\037K\b\035\000\000\000\001\334\330\267\000\000\267\000\001\334\330\217\000\000\271\377&\0377\017\017\017\017\377x'7\017\017\017\017!\001\207\277\274\000D\326\223\037\377\003\000\004\001\005\217\000D\326\223\037\377\003\002\006\003\007\377x'7\007\007\007\007\262\001\207\277\377\036{7\007\007\007\007\201\036\0373\001\000\306\277\276\000D\326\270oO\006\003\000\207\277\267\000D\326\270o\367\006\000\000\306\277\223\000D\326\272sO\006\270\000D\326\272s\367\006\271\000\020\326\273\025\025\002\201xu3\"\001\207\277\220s!M\377vs7\000\003\000\000\220\000 \326\377 \013\000p\000\000\000\262\000\207\277\200r\225|\235\377\210\277\271\000\001\325\220\000\251\001\220s!K\271\000W\326\272-\374\003\000\001\002\003\272\000W\326\217-\374\003\000\001\002\003\223\001\207\277\274\000F\326\220\t\001\002\217\000D\326\223}\347\006#\002\207\277\220\000D\326\270o\353\006\204(s3\377x'K\000\035\000\000\377xoK\b\035\000\000\000\0004\331\213\217\000\000\000\001\334\330\223\000\000\217\000\001\334\330\267\000\000\267\377('7\017\017\017\017\377r)7\017\017\017\017!\001\207\277\271\000D\326\224'\377\003\000\004\001\005\223\000D\326\224'\377\003\002\006\003\007\377r)7\007\007\007\007\262\001\207\277\377&u7\007\007\007\007\201&'3\001\000\306\277\274\000D\326\220\037S\006\003\000\207\277\220\000D\326\220\037\353\006\000\000\306\277\217\000D\326\270oS\006\224\000D\326\270o\353\006\267\000\020\326\273%\025\002\201rq3\223\000W\326\223-\374\003\000\001\002\003\243\001\207\277\221o#M\377vo7\000\000\003\000\220\000D\326\224!O\006\223\001\207\277\221\000 \326\377\"\013\000p\000\000\000\200n\225|\235\377\210\277\267\000\001\325\220\000\251\001A\002\207\277\221o#K\267\000W\326\270-\374\003\000\001\002\003\204*q3\377*+7\017\017\017\017\221\000F\326\221\t\001\002\024\002\207\277\217\000D\326\217y\337\006\377po7\017\017\017\017\003\000\207\277\377\"'K\000\035\000\000\377\"#K\b\035\000\000\b\0004\331\213\217\000\000\000\001\334\330\223\000\000\217\000\001\334\330\221\000\000\223\221\000\020\326\222/!\002\222\000\020\326\2735\025\002\270\000D\326\267+\377\003\000\004\001\005\225\000D\326\267+\377\003\002\006\003\007\377vo7\000\000\000\003\024\002\207\277\221%#M\201p%3\024\002\207\277\201*s3\200n\225|\377pq7\007\007\007\007\221\000 \326\377\"\013\000p\000\000\000\377*+7\007\007\007\007\222\000W\326\222-\374\003\000\001\002\003\235\377\210\277\267\000\001\325\220\000\251\001\271\000W\326\271-\374\003\000\001\002\003\002\000\207\277\221o#K\001\000\306\277\267\000D\326\220\037\343\006\000\000\306\277\270\000D\326\224'\343\006\220\000D\326\220\037W\006\223\000D\326\224'W\006\221\000F\326\221\t\001\002\377,)7\017\017\017\017\217\000D\326\270oK\006\024\002\207\277\220\000D\326\223!\347\006\377\"%K\000\035\000\000\377\"#K\b\035\000\000\204,'3\020\0004\331\213\217\000\000\000\001\334\330\222\000\000\217\000\001\334\330\221\000\000\221\377&'7\017\017\017\017!\001\207\277\225\000D\326\223)\377\003\000\004\001\005\223\000D\326\223)\377\003\002\006\003\007\201*)32\002\207\277\201&-3\377*+7\007\007\007\007\377&'7\007\007\007\007\224\000W\326\224-\374\003\000\001\002\003\004\000\207\277\226\000W\326\226-\374\003\000\001\002\003\001\000\306\277\267\000D\326\220\037W\006\000\000\306\277\225\000D\326\222#W\006\220\000D\326\220\037O\006\221\000D\326\222#O\006\023\001\207\277\217\000D\326\225oS\006\220\000D\326\221![\006\030\0004\331\213\217\000\000\000\000\306\277\301N\200\276\377\377\224\277\000\220\334\331\214\000\000\217 \260\334\331\215\000\000\223\001\000\306\277z@F\314\217/\353\035r@F\314\221/\313\035\000\000\306\277j@F\314\223/\253\035b@F\314\225/\213\035Z@F\314\217?k\035R@F\314\221?K\035J@F\314\223?+\035B@F\314\225?\013\035:@F\314\217O\353\0342@F\314\221O\313\034*@F\314\223O\253\034\"@F\314\225O\213\034\032@F\314\217_k\034\022@F\314\221_K\034\n@F\314\223_+\034\002@F\314\225_\013\034$\264\334\331\214\000\000\217D\324\334\331\215\000\000\223\001\000\306\277z@F\314\2173\353\035r@F\314\2213\313\035\000\000\306\277j@F\314\2233\253\035b@F\314\2253\213\035Z@F\314\217Ck\035R@F\314\221CK\035J@F\314\223C+\035B@F\314\225C\013\035:@F\314\217S\353\0342@F\314\221S\313\034*@F\314\223S\253\034\"@F\314\225S\213\034\032@F\314\217ck\034\022@F\314\221cK\034\n@F\314\223c+\034\002@F\314\225c\013\034H\330\334\331\214\000\000\217h\370\334\331\215\000\000\223\001\000\306\277z@F\314\2177\353\035r@F\314\2217\313\035\000\000\306\277j@F\314\2237\253\035b@F\314\2257\213\035Z@F\314\217Gk\035R@F\314\221GK\035J@F\314\223G+\035B@F\314\225G\013\035:@F\314\217W\353\0342@F\314\221W\313\034*@F\314\223W\253\034\"@F\314\225W\213\034\032@F\314\217gk\034\022@F\314\221gK\034\n@F\314\223g+\034\002@F\314\225g\013\034l\374\334\331\214\000\000\217\f\234\334\331\216\000\000\223\001\000\306\277z@F\314\217;\353\035r@F\314\221;\313\035Z@F\314\217Kk\035R@F\314\221KK\035:@F\314\217[\353\0342@F\314\221[\313\034\032@F\314\217kk\034\022@F\314\221kK\034\000\000\306\277j@F\314\223;\253\035b@F\314\225;\213\035J@F\314\223K+\035B@F\314\225K\013\035*@F\314\223[\253\034\"@F\314\225[\213\034\n@F\314\223k+\034\002@F\314\225k\013\034\301N\200\276\206j\000\327\377\f\003\002\000\002\000\000\235\377\210\277\207| \325\200\016\253\001\377\002\002J\000\004\000\000\003\300\003\201\236\377\210\277\003\t\003\277\377\377\224\277\373\375\241\277\212\003\002~\000@\000\364 \000\000\3701\001\207\277\210\002\0026\202\000W\326\000\037!\006~\000\204\276\000\000X\326\013\002&\006\002\000\207\277\237\004\0075~\000\304\324\n\004\003\002\226\004\245\277\322\001\207\277\202\004\t?\001\000,\327\r\004\003\002\207\000,\327\f\006\003\002~\000\205\276\000\000\307\277\204j\000\327\000\b\003\002\235\377\210\277\205| \325\001\n\253\001|\000\005\356\204\000\000\000\204\000\000\000\205|\376\326\f\004\003\002\221\000\207\277\206\000U\326\206\017\007\004\202\n\013?!\001\207\277\205j\000\327\002\n\003\002\235\377\210\277\206| \325\003\f\253\001~\000\304\324\b\000\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\016?\211j\000\327\006\016\003\002\235\377\210\277\002\000\207\277\212| \325\007\020\253\001\207j\000\327\205\017\003\002\235\377\210\277\210| \325\206\021\253\001|\000\005\356\001\000\000\000\211\000\000\000|\000\005\356\211\000\000\000\207\000\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\365\022W|\200\006\356\000\000\200D\207\000\000\000\236\377\210\277~\005~\214\201\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\016?\211j\000\327\006\016\003\002\235\377\210\277\002\000\207\277\212| \325\007\020\253\001\207j\000\327\205\017\003\002\235\377\210\277\210| \325\206\021\253\001|\000\005\356\001\000\000\000\211\004\000\000|\000\005\356z\000\000\000\207\004\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\367\364V|\200\006\356\000\000\000=\207\004\000\000\236\377\210\277~\005~\214\202\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>\207j\000\327\006\364\002\002\235\377\210\277\002\000\207\277\210| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000\207\b\000\000|\000\005\356\207\000\000\000z\b\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\371\016W|\200\006\356\000\000\200Cz\b\000\000\236\377\210\277~\005~\214\203\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>\207j\000\327\006\364\002\002\235\377\210\277\002\000\207\277\210| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000\207\f\000\000|\000\005\356|\000\000\000z\f\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\373\370V|\200\006\356\000\000\000>z\f\000\000\236\377\210\277~\005~\214\204\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>|j\000\327\006\364\002\002\235\377\210\277\002\000\207\277}| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000|\020\000\000|\000\005\356|\000\000\000z\020\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\375\370V|\200\006\356\000\000\000>z\020\000\000\236\377\210\277~\005~\214\205\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>|j\000\327\006\364\002\002\235\377\210\277\002\000\207\277}| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000|\024\000\000|\000\005\356|\000\000\000z\024\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\377\370V|\200\006\356\000\000\000>z\024\000\000\236\377\210\277~\005~\214\206\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>|j\000\327\006\364\002\002\235\377\210\277\002\000\207\277}| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000|\030\000\000|\000\005\356|\000\000\000z\030\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\001\371V|\200\006\356\000\000\000>z\030\000\000\236\377\210\277~\005~\214\207\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>|j\000\327\006\364\002\002\235\377\210\277\002\000\207\277}| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000|\034\000\000|\000\005\356|\000\000\000z\034\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\003\371V|\200\006\356\000\000\000>z\034\000\000\236\377\210\277~\005~\214\220\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>|j\000\327\006\364\002\002\235\377\210\277\002\000\207\277}| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000|@\000\000|\000\005\356|\000\000\000z@\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\345\370V|\200\006\356\000\000\000>z@\000\000\236\377\210\277~\005~\214\221\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\364>|j\000\327\006\364\002\002\235\377\210\277\002\000\207\277}| \325\007\366\252\001zj\000\327\205\365\002\002\235\377\210\277{| \325\206\367\252\001|\000\005\356\001\000\000\000|D\000\000|\000\005\356r\000\000\000zD\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\347\344V|\200\006\356\000\000\0009zD\000\000\236\377\210\277~\005~\214\222\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>zj\000\327\006\344\002\002\235\377\210\277\002\000\207\277{| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000zH\000\000|\000\005\356z\000\000\000rH\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\351\364V|\200\006\356\000\000\000=rH\000\000\236\377\210\277~\005~\214\223\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>zj\000\327\006\344\002\002\235\377\210\277\002\000\207\277{| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000zL\000\000|\000\005\356t\000\000\000rL\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\353\350V|\200\006\356\000\000\000:rL\000\000\236\377\210\277~\005~\214\224\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>tj\000\327\006\344\002\002\235\377\210\277\002\000\207\277u| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000tP\000\000|\000\005\356t\000\000\000rP\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\355\350V|\200\006\356\000\000\000:rP\000\000\236\377\210\277~\005~\214\225\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>tj\000\327\006\344\002\002\235\377\210\277\002\000\207\277u| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000tT\000\000|\000\005\356t\000\000\000rT\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\357\350V|\200\006\356\000\000\000:rT\000\000\236\377\210\277~\005~\214\226\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>tj\000\327\006\344\002\002\235\377\210\277\002\000\207\277u| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000tX\000\000|\000\005\356t\000\000\000rX\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\361\350V|\200\006\356\000\000\000:rX\000\000\236\377\210\277~\005~\214\227\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>tj\000\327\006\344\002\002\235\377\210\277\002\000\207\277u| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000t\\\000\000|\000\005\356t\000\000\000r\\\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\363\350V|\200\006\356\000\000\000:r\\\000\000\236\377\210\277~\005~\214\240\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>tj\000\327\006\344\002\002\235\377\210\277\002\000\207\277u| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000t\200\000\000|\000\005\356t\000\000\000r\200\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\325\350V|\200\006\356\000\000\000:r\200\000\000\236\377\210\277~\005~\214\241\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\344>tj\000\327\006\344\002\002\235\377\210\277\002\000\207\277u| \325\007\346\252\001rj\000\327\205\345\002\002\235\377\210\277s| \325\206\347\252\001|\000\005\356\001\000\000\000t\204\000\000|\000\005\356j\000\000\000r\204\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\327\324V|\200\006\356\000\000\0005r\204\000\000\236\377\210\277~\005~\214\242\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>rj\000\327\006\324\002\002\235\377\210\277\002\000\207\277s| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000r\210\000\000|\000\005\356r\000\000\000j\210\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\331\344V|\200\006\356\000\000\0009j\210\000\000\236\377\210\277~\005~\214\243\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>rj\000\327\006\324\002\002\235\377\210\277\002\000\207\277s| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000r\214\000\000|\000\005\356l\000\000\000j\214\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\333\330V|\200\006\356\000\000\0006j\214\000\000\236\377\210\277~\005~\214\244\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>lj\000\327\006\324\002\002\235\377\210\277\002\000\207\277m| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000l\220\000\000|\000\005\356l\000\000\000j\220\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\335\330V|\200\006\356\000\000\0006j\220\000\000\236\377\210\277~\005~\214\245\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>lj\000\327\006\324\002\002\235\377\210\277\002\000\207\277m| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000l\224\000\000|\000\005\356l\000\000\000j\224\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\337\330V|\200\006\356\000\000\0006j\224\000\000\236\377\210\277~\005~\214\246\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>lj\000\327\006\324\002\002\235\377\210\277\002\000\207\277m| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000l\230\000\000|\000\005\356l\000\000\000j\230\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\341\330V|\200\006\356\000\000\0006j\230\000\000\236\377\210\277~\005~\214\247\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>lj\000\327\006\324\002\002\235\377\210\277\002\000\207\277m| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000l\234\000\000|\000\005\356l\000\000\000j\234\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\343\330V|\200\006\356\000\000\0006j\234\000\000\236\377\210\277~\005~\214\260\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>lj\000\327\006\324\002\002\235\377\210\277\002\000\207\277m| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000l\300\000\000|\000\005\356l\000\000\000j\300\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\305\330V|\200\006\356\000\000\0006j\300\000\000\236\377\210\277~\005~\214\261\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\324>lj\000\327\006\324\002\002\235\377\210\277\002\000\207\277m| \325\007\326\252\001jj\000\327\205\325\002\002\235\377\210\277k| \325\206\327\252\001|\000\005\356\001\000\000\000l\304\000\000|\000\005\356b\000\000\000j\304\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\307\304V|\200\006\356\000\000\0001j\304\000\000\236\377\210\277~\005~\214\262\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\304>jj\000\327\006\304\002\002\235\377\210\277\002\000\207\277k| \325\007\306\252\001bj\000\327\205\305\002\002\235\377\210\277c| \325\206\307\252\001|\000\005\356\001\000\000\000j\310\000\000|\000\005\356j\000\000\000b\310\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\311\324V|\200\006\356\000\000\0005b\310\000\000\236\377\210\277~\005~\214\263\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\304>jj\000\327\006\304\002\002\235\377\210\277\002\000\207\277k| \325\007\306\252\001bj\000\327\205\305\002\002\235\377\210\277c| \325\206\307\252\001|\000\005\356\001\000\000\000j\314\000\000|\000\005\356d\000\000\000b\314\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\313\310V|\200\006\356\000\000\0002b\314\000\000\236\377\210\277~\005~\214\264\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\304>dj\000\327\006\304\002\002\235\377\210\277\002\000\207\277e| \325\007\306\252\001bj\000\327\205\305\002\002\235\377\210\277c| \325\206\307\252\001|\000\005\356\001\000\000\000d\320\000\000|\000\005\356d\000\000\000b\320\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\315\310V|\200\006\356\000\000\0002b\320\000\000\236\377\210\277~\005~\214\265\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\304>dj\000\327\006\304\002\002\235\377\210\277\002\000\207\277e| \325\007\306\252\001bj\000\327\205\305\002\002\235\377\210\277c| \325\206\307\252\001|\000\005\356\001\000\000\000d\324\000\000|\000\005\356d\000\000\000b\324\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\317\310V|\200\006\356\000\000\0002b\324\000\000\236\377\210\277~\005~\214\266\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\304>dj\000\327\006\304\002\002\235\377\210\277\002\000\207\277e| \325\007\306\252\001bj\000\327\205\305\002\002\235\377\210\277c| \325\206\307\252\001|\000\005\356\001\000\000\000d\330\000\000|\000\005\356d\000\000\000b\330\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\321\310V|\200\006\356\000\000\0002b\330\000\000\236\377\210\277~\005~\214\267\000\0028\001\000\207\277\b\002\210|~j~\213\034\000\245\277\237\000\0024\221\000\207\277\202\000\304>dj\000\327\006\304\002\002\235\377\210\277\002\000\207\277e| \325\007\306\252\001bj\000\327\205\305\002\002\235\377\210\277c| \325\206\307\252\001|\000\005\356\001\000\000\000d\334\000\000|\000\005\356d\000\000\000b\334\000\000\001\000\300\277\204\003\002\020\000\000\300\277\001\000\207\277\001\323\310V|\200\006\356\000\000\0002b\334\000\000\236\377\210\277~\004~\214\220\004\0038~\000\204\276\001\000\207\277~\000\304\324\n\002\002\002\227\004\245\277\202\004\305>e\000,\327\r\002\002\002~\000\205\276\000\000\307\277\322\000\207\277bj\000\327\000\304\002\002\235\377\210\277c| \325\001\306\252\001|\000\005\356b\000\000\000b@\000\000\237\002\3064f\000,\327\f\306\002\002c|\376\326\f\002\002\002\221\000\207\277d\000U\326d\315\226\005\202\306\306>!\001\207\277cj\000\327\002\306\002\002\235\377\210\277d| \325\003\310\252\001~\000\304\324\b\000\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\312>gj\000\327\006\312\002\002\235\377\210\277\002\000\207\277h| \325\007\314\252\001ej\000\327c\313\002\002\235\377\210\277f| \325d\315\252\001|\000\005\356\001\000\000\000g\000\000\000|\000\005\356g\000\000\000e\000\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\265\316V|\200\006\356\000\000\2003e\000\000\000\236\377\210\277~\005~\214\201\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\312>gj\000\327\006\312\002\002\235\377\210\277\002\000\207\277h| \325\007\314\252\001ej\000\327c\313\002\002\235\377\210\277f| \325d\315\252\001|\000\005\356\001\000\000\000g\004\000\000|\000\005\356Z\000\000\000e\004\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\267\264V|\200\006\356\000\000\000-e\004\000\000\236\377\210\277~\005~\214\202\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>ej\000\327\006\264\002\002\235\377\210\277\002\000\207\277f| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000e\b\000\000|\000\005\356e\000\000\000Z\b\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\271\312V|\200\006\356\000\000\2002Z\b\000\000\236\377\210\277~\005~\214\203\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>ej\000\327\006\264\002\002\235\377\210\277\002\000\207\277f| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000e\f\000\000|\000\005\356\\\000\000\000Z\f\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\273\270V|\200\006\356\000\000\000.Z\f\000\000\236\377\210\277~\005~\214\204\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>\\j\000\327\006\264\002\002\235\377\210\277\002\000\207\277]| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000\\\020\000\000|\000\005\356\\\000\000\000Z\020\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\275\270V|\200\006\356\000\000\000.Z\020\000\000\236\377\210\277~\005~\214\205\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>\\j\000\327\006\264\002\002\235\377\210\277\002\000\207\277]| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000\\\024\000\000|\000\005\356\\\000\000\000Z\024\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\277\270V|\200\006\356\000\000\000.Z\024\000\000\236\377\210\277~\005~\214\206\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>\\j\000\327\006\264\002\002\235\377\210\277\002\000\207\277]| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000\\\030\000\000|\000\005\356\\\000\000\000Z\030\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\301\270V|\200\006\356\000\000\000.Z\030\000\000\236\377\210\277~\005~\214\207\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>\\j\000\327\006\264\002\002\235\377\210\277\002\000\207\277]| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000\\\034\000\000|\000\005\356\\\000\000\000Z\034\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\303\270V|\200\006\356\000\000\000.Z\034\000\000\236\377\210\277~\005~\214\220\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>\\j\000\327\006\264\002\002\235\377\210\277\002\000\207\277]| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000\\@\000\000|\000\005\356\\\000\000\000Z@\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\245\270V|\200\006\356\000\000\000.Z@\000\000\236\377\210\277~\005~\214\221\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\264>\\j\000\327\006\264\002\002\235\377\210\277\002\000\207\277]| \325\007\266\252\001Zj\000\327c\265\002\002\235\377\210\277[| \325d\267\252\001|\000\005\356\001\000\000\000\\D\000\000|\000\005\356R\000\000\000ZD\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\247\244V|\200\006\356\000\000\000)ZD\000\000\236\377\210\277~\005~\214\222\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Zj\000\327\006\244\002\002\235\377\210\277\002\000\207\277[| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000ZH\000\000|\000\005\356Z\000\000\000RH\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\251\264V|\200\006\356\000\000\000-RH\000\000\236\377\210\277~\005~\214\223\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Zj\000\327\006\244\002\002\235\377\210\277\002\000\207\277[| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000ZL\000\000|\000\005\356T\000\000\000RL\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\253\250V|\200\006\356\000\000\000*RL\000\000\236\377\210\277~\005~\214\224\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Tj\000\327\006\244\002\002\235\377\210\277\002\000\207\277U| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000TP\000\000|\000\005\356T\000\000\000RP\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\255\250V|\200\006\356\000\000\000*RP\000\000\236\377\210\277~\005~\214\225\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Tj\000\327\006\244\002\002\235\377\210\277\002\000\207\277U| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000TT\000\000|\000\005\356T\000\000\000RT\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\257\250V|\200\006\356\000\000\000*RT\000\000\236\377\210\277~\005~\214\226\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Tj\000\327\006\244\002\002\235\377\210\277\002\000\207\277U| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000TX\000\000|\000\005\356T\000\000\000RX\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\261\250V|\200\006\356\000\000\000*RX\000\000\236\377\210\277~\005~\214\227\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Tj\000\327\006\244\002\002\235\377\210\277\002\000\207\277U| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000T\\\000\000|\000\005\356T\000\000\000R\\\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\263\250V|\200\006\356\000\000\000*R\\\000\000\236\377\210\277~\005~\214\240\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Tj\000\327\006\244\002\002\235\377\210\277\002\000\207\277U| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000T\200\000\000|\000\005\356T\000\000\000R\200\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\225\250V|\200\006\356\000\000\000*R\200\000\000\236\377\210\277~\005~\214\241\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\244>Tj\000\327\006\244\002\002\235\377\210\277\002\000\207\277U| \325\007\246\252\001Rj\000\327c\245\002\002\235\377\210\277S| \325d\247\252\001|\000\005\356\001\000\000\000T\204\000\000|\000\005\356J\000\000\000R\204\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\227\224V|\200\006\356\000\000\000%R\204\000\000\236\377\210\277~\005~\214\242\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Rj\000\327\006\224\002\002\235\377\210\277\002\000\207\277S| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000R\210\000\000|\000\005\356R\000\000\000J\210\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\231\244V|\200\006\356\000\000\000)J\210\000\000\236\377\210\277~\005~\214\243\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Rj\000\327\006\224\002\002\235\377\210\277\002\000\207\277S| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000R\214\000\000|\000\005\356L\000\000\000J\214\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\233\230V|\200\006\356\000\000\000&J\214\000\000\236\377\210\277~\005~\214\244\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Lj\000\327\006\224\002\002\235\377\210\277\002\000\207\277M| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000L\220\000\000|\000\005\356L\000\000\000J\220\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\235\230V|\200\006\356\000\000\000&J\220\000\000\236\377\210\277~\005~\214\245\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Lj\000\327\006\224\002\002\235\377\210\277\002\000\207\277M| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000L\224\000\000|\000\005\356L\000\000\000J\224\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\237\230V|\200\006\356\000\000\000&J\224\000\000\236\377\210\277~\005~\214\246\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Lj\000\327\006\224\002\002\235\377\210\277\002\000\207\277M| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000L\230\000\000|\000\005\356L\000\000\000J\230\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\241\230V|\200\006\356\000\000\000&J\230\000\000\236\377\210\277~\005~\214\247\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Lj\000\327\006\224\002\002\235\377\210\277\002\000\207\277M| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000L\234\000\000|\000\005\356L\000\000\000J\234\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\243\230V|\200\006\356\000\000\000&J\234\000\000\236\377\210\277~\005~\214\260\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Lj\000\327\006\224\002\002\235\377\210\277\002\000\207\277M| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000L\300\000\000|\000\005\356L\000\000\000J\300\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\205\230V|\200\006\356\000\000\000&J\300\000\000\236\377\210\277~\005~\214\261\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\224>Lj\000\327\006\224\002\002\235\377\210\277\002\000\207\277M| \325\007\226\252\001Jj\000\327c\225\002\002\235\377\210\277K| \325d\227\252\001|\000\005\356\001\000\000\000L\304\000\000|\000\005\356B\000\000\000J\304\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\207\204V|\200\006\356\000\000\000!J\304\000\000\236\377\210\277~\005~\214\262\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\204>Jj\000\327\006\204\002\002\235\377\210\277\002\000\207\277K| \325\007\206\252\001Bj\000\327c\205\002\002\235\377\210\277C| \325d\207\252\001|\000\005\356\001\000\000\000J\310\000\000|\000\005\356J\000\000\000B\310\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\211\224V|\200\006\356\000\000\000%B\310\000\000\236\377\210\277~\005~\214\263\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\204>Jj\000\327\006\204\002\002\235\377\210\277\002\000\207\277K| \325\007\206\252\001Bj\000\327c\205\002\002\235\377\210\277C| \325d\207\252\001|\000\005\356\001\000\000\000J\314\000\000|\000\005\356D\000\000\000B\314\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\213\210V|\200\006\356\000\000\000\"B\314\000\000\236\377\210\277~\005~\214\264\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\204>Dj\000\327\006\204\002\002\235\377\210\277\002\000\207\277E| \325\007\206\252\001Bj\000\327c\205\002\002\235\377\210\277C| \325d\207\252\001|\000\005\356\001\000\000\000D\320\000\000|\000\005\356D\000\000\000B\320\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\215\210V|\200\006\356\000\000\000\"B\320\000\000\236\377\210\277~\005~\214\265\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\204>Dj\000\327\006\204\002\002\235\377\210\277\002\000\207\277E| \325\007\206\252\001Bj\000\327c\205\002\002\235\377\210\277C| \325d\207\252\001|\000\005\356\001\000\000\000D\324\000\000|\000\005\356D\000\000\000B\324\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\217\210V|\200\006\356\000\000\000\"B\324\000\000\236\377\210\277~\005~\214\266\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\204>Dj\000\327\006\204\002\002\235\377\210\277\002\000\207\277E| \325\007\206\252\001Bj\000\327c\205\002\002\235\377\210\277C| \325d\207\252\001|\000\005\356\001\000\000\000D\330\000\000|\000\005\356D\000\000\000B\330\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\221\210V|\200\006\356\000\000\000\"B\330\000\000\236\377\210\277~\005~\214\267\000\0028\001\000\207\277\b\002\210|~j~\213\034\000\245\277\237\000\0024\221\000\207\277\202\000\204>Dj\000\327\006\204\002\002\235\377\210\277\002\000\207\277E| \325\007\206\252\001Bj\000\327c\205\002\002\235\377\210\277C| \325d\207\252\001|\000\005\356\001\000\000\000D\334\000\000|\000\005\356D\000\000\000B\334\000\000\001\000\300\277b\003\002\020\000\000\300\277\001\000\207\277\001\223\210V|\200\006\356\000\000\000\"B\334\000\000\236\377\210\277~\004~\214\240\004\0038~\000\204\276\001\000\207\277~\000\304\324\n\002\002\002\227\004\245\277\202\004\205>E\000,\327\r\002\002\002~\000\205\276\000\000\307\277\322\000\207\277Bj\000\327\000\204\002\002\235\377\210\277C| \325\001\206\252\001|\000\005\356B\000\000\000B\200\000\000\237\002\2064F\000,\327\f\206\002\002C|\376\326\f\002\002\002\221\000\207\277D\000U\326D\215\026\005\202\206\206>!\001\207\277Cj\000\327\002\206\002\002\235\377\210\277D| \325\003\210\252\001~\000\304\324\b\000\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\212>Gj\000\327\006\212\002\002\235\377\210\277\002\000\207\277H| \325\007\214\252\001Ej\000\327C\213\002\002\235\377\210\277F| \325D\215\252\001|\000\005\356\001\000\000\000G\000\000\000|\000\005\356G\000\000\000E\000\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001u\216V|\200\006\356\000\000\200#E\000\000\000\236\377\210\277~\005~\214\201\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000\212>Gj\000\327\006\212\002\002\235\377\210\277\002\000\207\277H| \325\007\214\252\001Ej\000\327C\213\002\002\235\377\210\277F| \325D\215\252\001|\000\005\356\001\000\000\000G\004\000\000|\000\005\356:\000\000\000E\004\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001wtV|\200\006\356\000\000\000\035E\004\000\000\236\377\210\277~\005~\214\202\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t>Ej\000\327\006t\002\002\235\377\210\277\002\000\207\277F| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000E\b\000\000|\000\005\356E\000\000\000:\b\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001y\212V|\200\006\356\000\000\200\":\b\000\000\236\377\210\277~\005~\214\203\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t>Ej\000\327\006t\002\002\235\377\210\277\002\000\207\277F| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000E\f\000\000|\000\005\356<\000\000\000:\f\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001{xV|\200\006\356\000\000\000\036:\f\000\000\236\377\210\277~\005~\214\204\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t><j\000\327\006t\002\002\235\377\210\277\002\000\207\277=| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000<\020\000\000|\000\005\356<\000\000\000:\020\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001}xV|\200\006\356\000\000\000\036:\020\000\000\236\377\210\277~\005~\214\205\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t><j\000\327\006t\002\002\235\377\210\277\002\000\207\277=| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000<\024\000\000|\000\005\356<\000\000\000:\024\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001\177xV|\200\006\356\000\000\000\036:\024\000\000\236\377\210\277~\005~\214\206\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t><j\000\327\006t\002\002\235\377\210\277\002\000\207\277=| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000<\030\000\000|\000\005\356<\000\000\000:\030\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001\201xV|\200\006\356\000\000\000\036:\030\000\000\236\377\210\277~\005~\214\207\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t><j\000\327\006t\002\002\235\377\210\277\002\000\207\277=| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000<\034\000\000|\000\005\356<\000\000\000:\034\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001\203xV|\200\006\356\000\000\000\036:\034\000\000\236\377\210\277~\005~\214\220\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t><j\000\327\006t\002\002\235\377\210\277\002\000\207\277=| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000<@\000\000|\000\005\356<\000\000\000:@\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001exV|\200\006\356\000\000\000\036:@\000\000\236\377\210\277~\005~\214\221\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000t><j\000\327\006t\002\002\235\377\210\277\002\000\207\277=| \325\007v\252\001:j\000\327Cu\002\002\235\377\210\277;| \325Dw\252\001|\000\005\356\001\000\000\000<D\000\000|\000\005\3562\000\000\000:D\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001gdV|\200\006\356\000\000\000\031:D\000\000\236\377\210\277~\005~\214\222\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>:j\000\327\006d\002\002\235\377\210\277\002\000\207\277;| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\000:H\000\000|\000\005\356:\000\000\0002H\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001itV|\200\006\356\000\000\000\0352H\000\000\236\377\210\277~\005~\214\223\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>:j\000\327\006d\002\002\235\377\210\277\002\000\207\277;| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\000:L\000\000|\000\005\3564\000\000\0002L\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001khV|\200\006\356\000\000\000\0322L\000\000\236\377\210\277~\005~\214\224\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>4j\000\327\006d\002\002\235\377\210\277\002\000\207\2775| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\0004P\000\000|\000\005\3564\000\000\0002P\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001mhV|\200\006\356\000\000\000\0322P\000\000\236\377\210\277~\005~\214\225\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>4j\000\327\006d\002\002\235\377\210\277\002\000\207\2775| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\0004T\000\000|\000\005\3564\000\000\0002T\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001ohV|\200\006\356\000\000\000\0322T\000\000\236\377\210\277~\005~\214\226\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>4j\000\327\006d\002\002\235\377\210\277\002\000\207\2775| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\0004X\000\000|\000\005\3564\000\000\0002X\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001qhV|\200\006\356\000\000\000\0322X\000\000\236\377\210\277~\005~\214\227\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>4j\000\327\006d\002\002\235\377\210\277\002\000\207\2775| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\0004\\\000\000|\000\005\3564\000\000\0002\\\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001shV|\200\006\356\000\000\000\0322\\\000\000\236\377\210\277~\005~\214\240\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>4j\000\327\006d\002\002\235\377\210\277\002\000\207\2775| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\0004\200\000\000|\000\005\3564\000\000\0002\200\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001UhV|\200\006\356\000\000\000\0322\200\000\000\236\377\210\277~\005~\214\241\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000d>4j\000\327\006d\002\002\235\377\210\277\002\000\207\2775| \325\007f\252\0012j\000\327Ce\002\002\235\377\210\2773| \325Dg\252\001|\000\005\356\001\000\000\0004\204\000\000|\000\005\356*\000\000\0002\204\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001WTV|\200\006\356\000\000\000\0252\204\000\000\236\377\210\277~\005~\214\242\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>2j\000\327\006T\002\002\235\377\210\277\002\000\207\2773| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\0002\210\000\000|\000\005\3562\000\000\000*\210\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001YdV|\200\006\356\000\000\000\031*\210\000\000\236\377\210\277~\005~\214\243\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>2j\000\327\006T\002\002\235\377\210\277\002\000\207\2773| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\0002\214\000\000|\000\005\356,\000\000\000*\214\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001[XV|\200\006\356\000\000\000\026*\214\000\000\236\377\210\277~\005~\214\244\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>,j\000\327\006T\002\002\235\377\210\277\002\000\207\277-| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\000,\220\000\000|\000\005\356,\000\000\000*\220\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001]XV|\200\006\356\000\000\000\026*\220\000\000\236\377\210\277~\005~\214\245\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>,j\000\327\006T\002\002\235\377\210\277\002\000\207\277-| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\000,\224\000\000|\000\005\356,\000\000\000*\224\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001_XV|\200\006\356\000\000\000\026*\224\000\000\236\377\210\277~\005~\214\246\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>,j\000\327\006T\002\002\235\377\210\277\002\000\207\277-| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\000,\230\000\000|\000\005\356,\000\000\000*\230\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001aXV|\200\006\356\000\000\000\026*\230\000\000\236\377\210\277~\005~\214\247\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>,j\000\327\006T\002\002\235\377\210\277\002\000\207\277-| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\000,\234\000\000|\000\005\356,\000\000\000*\234\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001cXV|\200\006\356\000\000\000\026*\234\000\000\236\377\210\277~\005~\214\260\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>,j\000\327\006T\002\002\235\377\210\277\002\000\207\277-| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\000,\300\000\000|\000\005\356,\000\000\000*\300\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001EXV|\200\006\356\000\000\000\026*\300\000\000\236\377\210\277~\005~\214\261\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000T>,j\000\327\006T\002\002\235\377\210\277\002\000\207\277-| \325\007V\252\001*j\000\327CU\002\002\235\377\210\277+| \325DW\252\001|\000\005\356\001\000\000\000,\304\000\000|\000\005\356\"\000\000\000*\304\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001GDV|\200\006\356\000\000\000\021*\304\000\000\236\377\210\277~\005~\214\262\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000D>*j\000\327\006D\002\002\235\377\210\277\002\000\207\277+| \325\007F\252\001\"j\000\327CE\002\002\235\377\210\277#| \325DG\252\001|\000\005\356\001\000\000\000*\310\000\000|\000\005\356*\000\000\000\"\310\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001ITV|\200\006\356\000\000\000\025\"\310\000\000\236\377\210\277~\005~\214\263\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000D>*j\000\327\006D\002\002\235\377\210\277\002\000\207\277+| \325\007F\252\001\"j\000\327CE\002\002\235\377\210\277#| \325DG\252\001|\000\005\356\001\000\000\000*\314\000\000|\000\005\356$\000\000\000\"\314\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001KHV|\200\006\356\000\000\000\022\"\314\000\000\236\377\210\277~\005~\214\264\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000D>$j\000\327\006D\002\002\235\377\210\277\002\000\207\277%| \325\007F\252\001\"j\000\327CE\002\002\235\377\210\277#| \325DG\252\001|\000\005\356\001\000\000\000$\320\000\000|\000\005\356$\000\000\000\"\320\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001MHV|\200\006\356\000\000\000\022\"\320\000\000\236\377\210\277~\005~\214\265\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000D>$j\000\327\006D\002\002\235\377\210\277\002\000\207\277%| \325\007F\252\001\"j\000\327CE\002\002\235\377\210\277#| \325DG\252\001|\000\005\356\001\000\000\000$\324\000\000|\000\005\356$\000\000\000\"\324\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001OHV|\200\006\356\000\000\000\022\"\324\000\000\236\377\210\277~\005~\214\266\000\0028~\000\205\276\001\000\207\277~\000\304\324\b\002\002\002\034\000\245\277\237\000\0024\221\000\207\277\202\000D>$j\000\327\006D\002\002\235\377\210\277\002\000\207\277%| \325\007F\252\001\"j\000\327CE\002\002\235\377\210\277#| \325DG\252\001|\000\005\356\001\000\000\000$\330\000\000|\000\005\356$\000\000\000\"\330\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001QHV|\200\006\356\000\000\000\022\"\330\000\000\236\377\210\277~\005~\214\267\000\0028\001\000\207\277\b\002\210|~j~\213\034\000\245\277\237\000\0024\221\000\207\277\202\000D>$j\000\327\006D\002\002\235\377\210\277\002\000\207\277%| \325\007F\252\001\"j\000\327CE\002\002\235\377\210\277#| \325DG\252\001|\000\005\356\001\000\000\000$\334\000\000|\000\005\356$\000\000\000\"\334\000\000\001\000\300\277B\003\002\020\000\000\300\277\001\000\207\277\001SHV|\200\006\356\000\000\000\022\"\334\000\000\236\377\210\277~\004~\214\260\004\0038~\000\204\276\001\000\207\277~\000\304\324\n\002\002\002Y\004\245\277\202\004E>%\000,\327\r\002\002\002\000\000\307\277\242\001\207\277\"j\000\327\000D\002\002\235\377\210\277#| \325\001F\252\001~\000\200\276|\000\005\356\"\000\000\000\"\300\000\000\237\002F41\001\207\277&\000,\327\fF\002\002#|\376\326\f\002\002\002\237\000\0024$\000U\326$M\226\004\221\000\207\277\202FF>#j\000\327\002F\002\002\235\377\210\277\002\000\207\277$| \325\003H\252\001~\000\304\324\b\000\002\002\032\000\245\277\202\000J>!\001\207\277'j\000\327\006J\002\002\235\377\210\277(| \325\007L\252\001%j\000\327#K\002\002\235\377\210\277&| \325$M\252\001|\000\005\356'\000\000\000'\000\000\000|\000\005\356(\000\000\000%\000\000\000\001\000\300\277\"ON\020\000\000\300\277\001\000\207\277'5PV|\200\006\356\000\000\000\024%\000\000\000\236\377\210\277~\000~\214\201\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\000J>!\001\207\277'j\000\327\006J\002\002\235\377\210\277(| \325\007L\252\001%j\000\327#K\002\002\235\377\210\277&| \325$M\252\001|\000\005\356\032\000\000\000'\004\000\000|\000\005\356'\000\000\000%\004\000\000\001\000\300\277\"54\020\000\000\300\277\001\000\207\277\0327NV|\200\006\356\000\000\200\023%\004\000\000\236\377\210\277~\000~\214\202\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277%j\000\327\0064\002\002\235\377\210\277&| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356%\000\000\000%\b\000\000|\000\005\356&\000\000\000\032\b\000\000\001\000\300\277\"KJ\020\000\000\300\277\001\000\207\277%9LV|\200\006\356\000\000\000\023\032\b\000\000\236\377\210\277~\000~\214\203\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277%j\000\327\0064\002\002\235\377\210\277&| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\034\000\000\000%\f\000\000|\000\005\356%\000\000\000\032\f\000\000\001\000\300\277\"98\020\000\000\300\277\001\000\207\277\034;JV|\200\006\356\000\000\200\022\032\f\000\000\236\377\210\277~\000~\214\204\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277\034j\000\327\0064\002\002\235\377\210\277\035| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\034\000\000\000\034\020\000\000|\000\005\356\035\000\000\000\032\020\000\000\001\000\300\277\"98\020\000\000\300\277\001\000\207\277\034=:V|\200\006\356\000\000\200\016\032\020\000\000\236\377\210\277~\000~\214\205\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277\034j\000\327\0064\002\002\235\377\210\277\035| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\034\000\000\000\034\024\000\000|\000\005\356\035\000\000\000\032\024\000\000\001\000\300\277\"98\020\000\000\300\277\001\000\207\277\034?:V|\200\006\356\000\000\200\016\032\024\000\000\236\377\210\277~\000~\214\206\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277\034j\000\327\0064\002\002\235\377\210\277\035| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\034\000\000\000\034\030\000\000|\000\005\356\035\000\000\000\032\030\000\000\001\000\300\277\"98\020\000\000\300\277\001\000\207\277\034A:V|\200\006\356\000\000\200\016\032\030\000\000\236\377\210\277~\000~\214\207\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277\034j\000\327\0064\002\002\235\377\210\277\035| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\034\000\000\000\034\034\000\000|\000\005\356\035\000\000\000\032\034\000\000\001\000\300\277\"98\020\000\000\300\277\001\000\207\277\034C:V|\200\006\356\000\000\200\016\032\034\000\000\236\377\210\277~\000~\214\220\00048~\000\200\276\001\000\207\277~\000\304\324\b4\002\002\032\000\245\277\202\0004>!\001\207\277\034j\000\327\0064\002\002\235\377\210\277\035| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\034\000\000\000\034@\000\000|\000\005\356\035\000\000\000\032@\000\000\001\000\300\277\"98\020\000\000\300\277\001\000\207\277\034%:V|\200\006\356\000\000\200\016\032@\000\000\236\377\210\277~\000~\214\221\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\0004>!\001\207\277\034j\000\327\0064\002\002\235\377\210\277\035| \325\0076\252\001\032j\000\327#5\002\002\235\377\210\277\033| \325$7\252\001|\000\005\356\022\000\000\000\034D\000\000|\000\005\356\034\000\000\000\032D\000\000\001\000\300\277\"%$\020\000\000\300\277\001\000\207\277\022'8V|\200\006\356\000\000\000\016\032D\000\000\236\377\210\277~\000~\214\222\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\032j\000\327\006$\002\002\235\377\210\277\033| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\032\000\000\000\032H\000\000|\000\005\356\033\000\000\000\022H\000\000\001\000\300\277\"54\020\000\000\300\277\001\000\207\277\032)6V|\200\006\356\000\000\200\r\022H\000\000\236\377\210\277~\000~\214\223\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\032j\000\327\006$\002\002\235\377\210\277\033| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\024\000\000\000\032L\000\000|\000\005\356\032\000\000\000\022L\000\000\001\000\300\277\")(\020\000\000\300\277\001\000\207\277\024+4V|\200\006\356\000\000\000\r\022L\000\000\236\377\210\277~\000~\214\224\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\024j\000\327\006$\002\002\235\377\210\277\025| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\024\000\000\000\024P\000\000|\000\005\356\025\000\000\000\022P\000\000\001\000\300\277\")(\020\000\000\300\277\001\000\207\277\024-*V|\200\006\356\000\000\200\n\022P\000\000\236\377\210\277~\000~\214\225\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\024j\000\327\006$\002\002\235\377\210\277\025| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\024\000\000\000\024T\000\000|\000\005\356\025\000\000\000\022T\000\000\001\000\300\277\")(\020\000\000\300\277\001\000\207\277\024/*V|\200\006\356\000\000\200\n\022T\000\000\236\377\210\277~\000~\214\226\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\024j\000\327\006$\002\002\235\377\210\277\025| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\024\000\000\000\024X\000\000|\000\005\356\025\000\000\000\022X\000\000\001\000\300\277\")(\020\000\000\300\277\001\000\207\277\0241*V|\200\006\356\000\000\200\n\022X\000\000\236\377\210\277~\000~\214\227\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\024j\000\327\006$\002\002\235\377\210\277\025| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\024\000\000\000\024\\\000\000|\000\005\356\025\000\000\000\022\\\000\000\001\000\300\277\")(\020\000\000\300\277\001\000\207\277\0243*V|\200\006\356\000\000\200\n\022\\\000\000\236\377\210\277~\000~\214\240\000$8~\000\200\276\001\000\207\277~\000\304\324\b$\002\002\032\000\245\277\202\000$>!\001\207\277\024j\000\327\006$\002\002\235\377\210\277\025| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\024\000\000\000\024\200\000\000|\000\005\356\025\000\000\000\022\200\000\000\001\000\300\277\")(\020\000\000\300\277\001\000\207\277\024\025*V|\200\006\356\000\000\200\n\022\200\000\000\236\377\210\277~\000~\214\241\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000$>!\001\207\277\024j\000\327\006$\002\002\235\377\210\277\025| \325\007&\252\001\022j\000\327#%\002\002\235\377\210\277\023| \325$'\252\001|\000\005\356\n\000\000\000\024\204\000\000|\000\005\356\024\000\000\000\022\204\000\000\001\000\300\277\"\025\024\020\000\000\300\277\001\000\207\277\n\027(V|\200\006\356\000\000\000\n\022\204\000\000\236\377\210\277~\000~\214\242\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\022j\000\327\006\024\002\002\235\377\210\277\023| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\022\000\000\000\022\210\000\000|\000\005\356\023\000\000\000\n\210\000\000\001\000\300\277\"%$\020\000\000\300\277\001\000\207\277\022\031&V|\200\006\356\000\000\200\t\n\210\000\000\236\377\210\277~\000~\214\243\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\022j\000\327\006\024\002\002\235\377\210\277\023| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\f\000\000\000\022\214\000\000|\000\005\356\022\000\000\000\n\214\000\000\001\000\300\277\"\031\030\020\000\000\300\277\001\000\207\277\f\033$V|\200\006\356\000\000\000\t\n\214\000\000\236\377\210\277~\000~\214\244\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\fj\000\327\006\024\002\002\235\377\210\277\r| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\f\000\000\000\f\220\000\000|\000\005\356\r\000\000\000\n\220\000\000\001\000\300\277\"\031\030\020\000\000\300\277\001\000\207\277\f\035\032V|\200\006\356\000\000\200\006\n\220\000\000\236\377\210\277~\000~\214\245\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\fj\000\327\006\024\002\002\235\377\210\277\r| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\f\000\000\000\f\224\000\000|\000\005\356\r\000\000\000\n\224\000\000\001\000\300\277\"\031\030\020\000\000\300\277\001\000\207\277\f\037\032V|\200\006\356\000\000\200\006\n\224\000\000\236\377\210\277~\000~\214\246\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\fj\000\327\006\024\002\002\235\377\210\277\r| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\f\000\000\000\f\230\000\000|\000\005\356\r\000\000\000\n\230\000\000\001\000\300\277\"\031\030\020\000\000\300\277\001\000\207\277\f!\032V|\200\006\356\000\000\200\006\n\230\000\000\236\377\210\277~\000~\214\247\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\fj\000\327\006\024\002\002\235\377\210\277\r| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\f\000\000\000\f\234\000\000|\000\005\356\r\000\000\000\n\234\000\000\001\000\300\277\"\031\030\020\000\000\300\277\001\000\207\277\f#\032V|\200\006\356\000\000\200\006\n\234\000\000\236\377\210\277~\000~\214\260\000\0248~\000\200\276\001\000\207\277~\000\304\324\b\024\002\002\032\000\245\277\202\000\024>!\001\207\277\fj\000\327\006\024\002\002\235\377\210\277\r| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\f\000\000\000\f\300\000\000|\000\005\356\r\000\000\000\n\300\000\000\001\000\300\277\"\031\030\020\000\000\300\277\001\000\207\277\f\005\032V|\200\006\356\000\000\200\006\n\300\000\000\236\377\210\277~\000~\214\261\000\0048~\000\200\276\001\000\207\277~\000\304\324\b\004\002\002\032\000\245\277\202\000\024>!\001\207\277\fj\000\327\006\024\002\002\235\377\210\277\r| \325\007\026\252\001\nj\000\327#\025\002\002\235\377\210\277\013| \325$\027\252\001|\000\005\356\002\000\000\000\f\304\000\000|\000\005\356\f\000\000\000\n\304\000\000\001\000\300\277\"\005\004\020\000\000\300\277\001\000\207\277\002\007\030V|\200\006\356\000\000\000\006\n\304\000\000\236\377\210\277~\000~\214\262\000\0048~\000\200\276\001\000\207\277~\000\304\324\b\004\002\002\032\000\245\277\202\000\004>!\001\207\277\nj\000\327\006\004\002\002\235\377\210\277\013| \325\007\006\252\001\002j\000\327#\005\002\002\235\377\210\277\003| \325$\007\252\001|\000\005\356\n\000\000\000\n\310\000\000|\000\005\356\013\000\000\000\002\310\000\000\001\000\300\277\"\025\024\020\000\000\300\277\001\000\207\277\n\t\026V|\200\006\356\000\000\200\005\002\310\000\000\236\377\210\277~\000~\214\263\000\0048~\000\200\276\001\000\207\277~\000\304\324\b\004\002\002\032\000\245\277\202\000\004>!\001\207\277\nj\000\327\006\004\002\002\235\377\210\277\013| \325\007\006\252\001\002j\000\327#\005\002\002\235\377\210\277\003| \325$\007\252\001|\000\005\356\004\000\000\000\n\314\000\000|\000\005\356\n\000\000\000\002\314\000\000\001\000\300\277\"\t\b\020\000\000\300\277\001\000\207\277\004\013\024V|\200\006\356\000\000\000\005\002\314\000\000\236\377\210\277~\000~\214\264\000\0048~\000\200\276\001\000\207\277~\000\304\324\b\004\002\002\032\000\245\277\202\000\004>!\001\207\277\004j\000\327\006\004\002\002\235\377\210\277\005| \325\007\006\252\001\002j\000\327#\005\002\002\235\377\210\277\003| \325$\007\252\001|\000\005\356\004\000\000\000\004\320\000\000|\000\005\356\005\000\000\000\002\320\000\000\001\000\300\277\"\t\b\020\000\000\300\277\001\000\207\277\004\r\nV|\200\006\356\000\000\200\002\002\320\000\000\236\377\210\277~\000~\214\265\000\0048~\000\200\276\001\000\207\277~\000\304\324\b\004\002\002\032\000\245\277\202\000\004>!\001\207\277\004j\000\327\006\004\002\002\235\377\210\277\005| \325\007\006\252\001\002j\000\327#\005\002\002\235\377\210\277\003| \325$\007\252\001|\000\005\356\004\000\000\000\004\324\000\000|\000\005\356\005\000\000\000\002\324\000\000\001\000\300\277\"\t\b\020\000\000\300\277\001\000\207\277\004\017\nV|\200\006\356\000\000\200\002\002\324\000\000\236\377\210\277~\000~\214\266\000\0048~\000\200\276\001\000\207\277~\000\304\324\b\004\002\002\032\000\245\277\202\000\004>!\001\207\277\004j\000\327\006\004\002\002\235\377\210\277\005| \325\007\006\252\001\002j\000\327#\005\002\002\235\377\210\277\003| \325$\007\252\001|\000\005\356\004\000\000\000\004\330\000\000|\000\005\356\005\000\000\000\002\330\000\000\001\000\300\277\"\t\b\020\000\000\300\277\001\000\207\277\004\021\nV|\200\006\356\000\000\200\002\002\330\000\000\236\377\210\277~\000~\214\267\000\0048\001\000\207\277\b\004\210|~j~\213\032\000\245\277\202\000\000>!\001\207\277\002j\000\327\006\000\002\002\235\377\210\277\003| \325\007\002\252\001\000j\000\327#\001\002\002\235\377\210\277\001| \325$\003\252\001|\000\005\356\002\000\000\000\002\334\000\000|\000\005\356\003\000\000\000\000\334\000\000\001\000\300\277\"\005\004\020\000\000\300\277\001\000\207\277\002\023\006V|\200\006\356\000\000\200\001\000\334\000\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\220\006\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000d\007\000\000\000\000\000\000\n\000\000\000\000\000\000\000\226\000\000\000\000\000\000\000\365\376\377o\000\000\000\000\b\007\000\000\000\000\000\000\004\000\000\000\000\000\000\0004\007\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 23.0.0 (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000)\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\357\000\000\000\000\002\t\000\200\204\000\000\000\000\000\000\000\000\000\000\000\000\000\000Z\000\000\000\022\003\b\000\000\033\000\000\000\000\000\000\234W\000\000\000\000\000\000\217\000\000\000\021\003\006\000@\b\000\000\000\000\000\000\000\002\000\000\000\000\000\000\233\000\000\000\021\003\006\000\000\b\000\000\000\000\000\000@\000\000\000\000\000\000\000\323\000\000\000\021\000\013\000\360\224\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.eh_frame\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000amdgpu.max_num_named_barrier\000gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201\000ff_fold_lut\000gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.kd\000__hip_cuid_4dbe48b018d034c5\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000T\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\220\006\000\000\000\000\000\000\220\006\000\000\000\000\000\000x\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\b\007\000\000\000\000\000\000\b\007\000\000\000\000\000\000,\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\0004\007\000\000\000\000\000\0004\007\000\000\000\000\000\0000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000d\007\000\000\000\000\000\000d\007\000\000\000\000\000\000\226\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000@\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000@\n\000\000\000\000\000\000@\n\000\000\000\000\000\0004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0009\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\033\000\000\000\000\000\000\000\013\000\000\000\000\000\000\200Y\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000?\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\200\204\000\000\000\000\000\000\200d\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000H\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360\204\000\000\000\000\000\000\360d\000\000\000\000\000\000\020\013\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000W\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360\224\000\000\000\000\000\000\360d\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\\\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360d\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000k\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360d\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360d\000\000\000\000\000\000\334\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\211\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320e\000\000\000\000\000\000\360\000\000\000\000\000\000\000\021\000\000\000\006\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300f\000\000\000\000\000\000\243\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\233\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000cg\000\000\000\000\000\000\370\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_3, 31968

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_3
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_4dbe48b018d034c5,@object # @__hip_gpubin_handle_4dbe48b018d034c5
	.local	__hip_gpubin_handle_4dbe48b018d034c5
	.comm	__hip_gpubin_handle_4dbe48b018d034c5,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_4dbe48b018d034c5,@object # @__hip_cuid_4dbe48b018d034c5
	.bss
	.globl	__hip_cuid_4dbe48b018d034c5
__hip_cuid_4dbe48b018d034c5:
	.byte	0                               # 0x0
	.size	__hip_cuid_4dbe48b018d034c5, 1

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __device_stub__gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym ff_fold_lut
	.addrsig_sym gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.addrsig_sym .L__unnamed_3
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_4dbe48b018d034c5
