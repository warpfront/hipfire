	.att_syntax
	.file	"iu4_ladder.hip"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x428fe00000000000              # double 4380866641920
.LCPI0_1:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.globl	main
	.prefalign	4, .Lfunc_end0, nop
	.type	main,@function
main:                                   # @main
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
	subq	$88, %rsp
	.cfi_def_cfa_offset 144
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leaq	.L__const.main.gate(%rip), %r14
	leaq	48(%rsp), %rdi
	movq	%r14, %rsi
	callq	_ZL14allocate_shapeRK5Shape
	leaq	.L__const.main.down(%rip), %rsi
	leaq	8(%rsp), %rdi
	callq	_ZL14allocate_shapeRK5Shape
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	72(%rsp), %r8
	movq	80(%rsp), %r9
	movq	8(%rsp), %r15
	movq	16(%rsp), %r12
	movq	24(%rsp), %r13
	movq	32(%rsp), %rbp
	movq	40(%rsp), %rbx
	movq	%r14, %rdi
	callq	_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	leaq	.L__const.main.down(%rip), %rdi
	movq	%r15, %rsi
	movq	%r12, %rdx
	movq	%r13, %rcx
	movq	%rbp, %r8
	movq	%rbx, %r9
	callq	_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy
	addsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = [4.38086664192E+12,0.0E+0]
	divsd	%xmm0, %xmm1
	divsd	.LCPI0_1(%rip), %xmm1
	leaq	.L.str.4(%rip), %r14
	movq	%r14, %rdi
	xorl	%esi, %esi
	movb	$2, %al
	callq	printf@PLT
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	72(%rsp), %r8
	movq	80(%rsp), %r9
	movq	8(%rsp), %rbx
	movq	16(%rsp), %r12
	movq	24(%rsp), %r13
	movq	32(%rsp), %rbp
	movq	40(%rsp), %r15
	leaq	.L__const.main.gate(%rip), %rdi
	callq	_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	leaq	.L__const.main.down(%rip), %rdi
	movq	%rbx, %rsi
	movq	%r12, %rdx
	movq	%r13, %rcx
	movq	%rbp, %r8
	movq	%r15, %r9
	callq	_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy
	addsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = [4.38086664192E+12,0.0E+0]
	divsd	%xmm0, %xmm1
	divsd	.LCPI0_1(%rip), %xmm1
	movq	%r14, %rdi
	movl	$1, %esi
	movb	$2, %al
	callq	printf@PLT
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	72(%rsp), %r8
	movq	80(%rsp), %r9
	movq	8(%rsp), %rbx
	movq	16(%rsp), %r15
	movq	24(%rsp), %r12
	movq	32(%rsp), %r13
	movq	40(%rsp), %rbp
	leaq	.L__const.main.gate(%rip), %rdi
	callq	_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	leaq	.L__const.main.down(%rip), %rdi
	movq	%rbx, %rsi
	movq	%r15, %rdx
	movq	%r12, %rcx
	movq	%r13, %r8
	movq	%rbp, %r9
	callq	_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy
	addsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = [4.38086664192E+12,0.0E+0]
	divsd	%xmm0, %xmm1
	divsd	.LCPI0_1(%rip), %xmm1
	movq	%r14, %rdi
	movl	$2, %esi
	movb	$2, %al
	callq	printf@PLT
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	72(%rsp), %r8
	movq	80(%rsp), %r9
	movq	8(%rsp), %rbx
	movq	16(%rsp), %r15
	movq	24(%rsp), %r12
	movq	32(%rsp), %r13
	movq	40(%rsp), %rbp
	leaq	.L__const.main.gate(%rip), %rdi
	callq	_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	leaq	.L__const.main.down(%rip), %rdi
	movq	%rbx, %rsi
	movq	%r15, %rdx
	movq	%r12, %rcx
	movq	%r13, %r8
	movq	%rbp, %r9
	callq	_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy
	addsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = [4.38086664192E+12,0.0E+0]
	divsd	%xmm0, %xmm1
	divsd	.LCPI0_1(%rip), %xmm1
	movq	%r14, %rdi
	movl	$3, %esi
	movb	$2, %al
	callq	printf@PLT
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	72(%rsp), %r8
	movq	80(%rsp), %r9
	movq	8(%rsp), %rbx
	movq	16(%rsp), %r15
	movq	24(%rsp), %r12
	movq	32(%rsp), %r13
	movq	40(%rsp), %rbp
	leaq	.L__const.main.gate(%rip), %rdi
	callq	_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	leaq	.L__const.main.down(%rip), %rdi
	movq	%rbx, %rsi
	movq	%r15, %rdx
	movq	%r12, %rcx
	movq	%r13, %r8
	movq	%rbp, %r9
	callq	_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy
	addsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = [4.38086664192E+12,0.0E+0]
	divsd	%xmm0, %xmm1
	divsd	.LCPI0_1(%rip), %xmm1
	movq	%r14, %rdi
	movl	$4, %esi
	movb	$2, %al
	callq	printf@PLT
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	72(%rsp), %r8
	movq	80(%rsp), %r9
	movq	8(%rsp), %rbx
	movq	16(%rsp), %r15
	movq	24(%rsp), %r12
	movq	32(%rsp), %r13
	movq	40(%rsp), %rbp
	leaq	.L__const.main.gate(%rip), %rdi
	callq	_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	leaq	.L__const.main.down(%rip), %rdi
	movq	%rbx, %rsi
	movq	%r15, %rdx
	movq	%r12, %rcx
	movq	%r13, %r8
	movq	%rbp, %r9
	callq	_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy
	addsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	movsd	.LCPI0_0(%rip), %xmm1           # xmm1 = [4.38086664192E+12,0.0E+0]
	divsd	%xmm0, %xmm1
	divsd	.LCPI0_1(%rip), %xmm1
	movq	%r14, %rdi
	movl	$5, %esi
	movb	$2, %al
	callq	printf@PLT
	xorl	%eax, %eax
	addq	$88, %rsp
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
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end1, nop     # -- Begin function _ZL14allocate_shapeRK5Shape
	.type	_ZL14allocate_shapeRK5Shape,@function
_ZL14allocate_shapeRK5Shape:            # @_ZL14allocate_shapeRK5Shape
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %r15
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rdi)
	movups	%xmm0, (%rdi)
	movq	$0, 32(%rdi)
	movslq	8(%rsi), %rbp
	movslq	16(%rsi), %r14
	movq	%rbp, %r12
	imulq	%r14, %r12
	shlq	$13, %r12
	movslq	12(%rsi), %rbx
	movq	%r12, %rsi
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB1_1
# %bb.3:
	movq	%r14, %r13
	imulq	%rbx, %r13
	shlq	$13, %r13
	leaq	8(%r15), %rdi
	movq	%rdi, 32(%rsp)                  # 8-byte Spill
	movq	%r13, %rsi
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB1_1
# %bb.4:
	leaq	(%rbx,%rbp), %rax
	imulq	%rax, %r14
	shlq	$10, %r14
	leaq	16(%r15), %rdi
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	movq	%r14, %rsi
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB1_1
# %bb.5:
	imulq	%rbp, %rbx
	movq	%rbx, %rbp
	shlq	$16, %rbp
	leaq	24(%r15), %rdi
	movq	%rdi, 16(%rsp)                  # 8-byte Spill
	movq	%rbp, %rsi
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB1_6
# %bb.7:
	shlq	$11, %rbx
	leaq	32(%r15), %rdi
	movq	%rdi, 8(%rsp)                   # 8-byte Spill
	movq	%rbx, %rsi
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB1_6
# %bb.8:
	movq	(%r15), %rdi
	movl	$17, %esi
	movq	%r12, %rdx
	callq	hipMemset@PLT
	testl	%eax, %eax
	jne	.LBB1_9
# %bb.10:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rdi
	movl	$34, %esi
	movq	%r13, %rdx
	callq	hipMemset@PLT
	testl	%eax, %eax
	jne	.LBB1_9
# %bb.11:
	movq	24(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rdi
	movl	$63, %esi
	movq	%r14, %rdx
	callq	hipMemset@PLT
	testl	%eax, %eax
	jne	.LBB1_12
# %bb.13:
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rdi
	xorl	%esi, %esi
	movq	%rbp, %rdx
	callq	hipMemset@PLT
	testl	%eax, %eax
	jne	.LBB1_12
# %bb.14:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	(%rax), %rdi
	xorl	%esi, %esi
	movq	%rbx, %rdx
	callq	hipMemset@PLT
	testl	%eax, %eax
	jne	.LBB1_12
# %bb.15:
	addq	$40, %rsp
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
.LBB1_1:
	.cfi_def_cfa_offset 96
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$326, %ecx                      # imm = 0x146
	jmp	.LBB1_2
.LBB1_12:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$329, %ecx                      # imm = 0x149
	jmp	.LBB1_2
.LBB1_6:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$327, %ecx                      # imm = 0x147
	jmp	.LBB1_2
.LBB1_9:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$328, %ecx                      # imm = 0x148
.LBB1_2:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$1, %edi
	callq	exit@PLT
.Lfunc_end1:
	.size	_ZL14allocate_shapeRK5Shape, .Lfunc_end1-_ZL14allocate_shapeRK5Shape
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy
.LCPI2_0:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI2_1:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI2_2:
	.quad	0x4020000000000000              # double 8
.LCPI2_3:
	.quad	0x4040000000000000              # double 32
.LCPI2_4:
	.quad	0x40d0000000000000              # double 16384
.LCPI2_5:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.prefalign	4, .Lfunc_end2, nop
	.type	_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy,@function
_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy: # @_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 112(%rsp)                  # 8-byte Spill
	movq	%r8, 216(%rsp)                  # 8-byte Spill
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	movq	%rdx, 200(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movq	%rdi, 120(%rsp)                 # 8-byte Spill
	movq	8(%rdi), %rbx
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB2_2
# %bb.1:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_2:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB2_4
# %bb.3:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_4:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB2_6
# %bb.5:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_6:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB2_8
# %bb.7:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_8:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, 240(%rsp)                 # 8-byte Spill
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB2_10
# %bb.9:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_10:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB2_11
# %bb.20:                               # %.preheader
	movl	$20, %ebx
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	movq	240(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB2_21:                               # =>This Inner Loop Header: Depth=1
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_23
# %bb.30:                               #   in Loop: Header=BB2_21 Depth=1
.Ltmp6:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp7:                                 # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_32
# %bb.36:                               #   in Loop: Header=BB2_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp12:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp13:                                # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_38
# %bb.42:                               #   in Loop: Header=BB2_21 Depth=1
.Ltmp18:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%r14, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp19:                                # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_46
# %bb.44:                               #   in Loop: Header=BB2_21 Depth=1
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movl	%eax, 4(%rsp)
	movl	%ecx, 236(%rsp)
	movl	%edx, 232(%rsp)
	movq	192(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 104(%rsp)
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	208(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	216(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp20:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp21:                                # EH_LABEL
# %bb.45:                               # %.noexc104
                                        #   in Loop: Header=BB2_21 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp22:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp23:                                # EH_LABEL
.LBB2_46:                               #   in Loop: Header=BB2_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp25:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp26:                                # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_48
# %bb.53:                               #   in Loop: Header=BB2_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp31:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp32:                                # EH_LABEL
# %bb.54:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_55
# %bb.59:                               #   in Loop: Header=BB2_21 Depth=1
	movl	$0, 128(%rsp)
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
.Ltmp37:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp38:                                # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_61
# %bb.65:                               #   in Loop: Header=BB2_21 Depth=1
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI2_0(%rip), %xmm0
	cmpq	%r12, %r15
	je	.LBB2_67
# %bb.66:                               #   in Loop: Header=BB2_21 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r13
	jmp	.LBB2_76
	.p2align	4
.LBB2_67:                               #   in Loop: Header=BB2_21 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%r12, %r15
	subq	%rbp, %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r15
	je	.LBB2_68
# %bb.70:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB2_21 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp43:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp44:                                # EH_LABEL
# %bb.71:                               # %.noexc108
                                        #   in Loop: Header=BB2_21 Depth=1
	movq	%rax, %r13
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB2_73
# %bb.72:                               #   in Loop: Header=BB2_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB2_73:                               # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB2_21 Depth=1
	testq	%rbp, %rbp
	je	.LBB2_75
# %bb.74:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB2_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB2_75:                               # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB2_21 Depth=1
	addq	%r13, %r15
	leaq	(,%r14,4), %r12
	addq	%r13, %r12
	movq	240(%rsp), %r14                 # 8-byte Reload
.LBB2_76:                               # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB2_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp46:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp47:                                # EH_LABEL
# %bb.77:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_78
# %bb.84:                               #   in Loop: Header=BB2_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp52:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp53:                                # EH_LABEL
# %bb.85:                               #   in Loop: Header=BB2_21 Depth=1
	testl	%eax, %eax
	jne	.LBB2_86
# %bb.12:                               #   in Loop: Header=BB2_21 Depth=1
	addq	$4, %r15
	movq	%r13, %rbp
	decl	%ebx
	jne	.LBB2_21
# %bb.13:
	cmpq	%r15, %r13
	je	.LBB2_16
# %bb.14:
	movq	%r15, %rax
	subq	%r13, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp58:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp59:                                # EH_LABEL
# %bb.15:                               # %.noexc
.Ltmp60:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp61:                                # EH_LABEL
.LBB2_16:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_.exit
	movss	36(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movss	40(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 112(%rsp)                # 4-byte Spill
	movl	$0, 128(%rsp)
.Ltmp63:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	128(%rsp), %rdi
	movl	$20480, %ecx                    # imm = 0x5000
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp64:                                # EH_LABEL
# %bb.17:                               # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKDv2_jS2_PKjPfPyiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB2_18
# %bb.92:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addss	112(%rsp), %xmm0                # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm3
	mulsd	.LCPI2_1(%rip), %xmm3
	movq	120(%rsp), %rax                 # 8-byte Reload
	cvtdq2pd	8(%rax), %xmm0
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	mulsd	%xmm0, %xmm2
	mulsd	.LCPI2_2(%rip), %xmm2
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	mulsd	.LCPI2_3(%rip), %xmm1
	mulsd	.LCPI2_4(%rip), %xmm1
	movq	(%rax), %rdx
	movapd	%xmm3, %xmm0
	movsd	%xmm3, 224(%rsp)                # 8-byte Spill
	divsd	%xmm3, %xmm1
	divsd	.LCPI2_5(%rip), %xmm1
	movl	128(%rsp), %ecx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rdi
	xorl	%esi, %esi
	movb	$2, %al
	callq	printf@PLT
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	224(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addq	$248, %rsp
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
.LBB2_61:
	.cfi_def_cfa_offset 304
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp40:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp41:                                # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$286, %ecx                      # imm = 0x11E
	jmp	.LBB2_26
.LBB2_55:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp34:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp35:                                # EH_LABEL
# %bb.56:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB2_50
.LBB2_48:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp28:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp29:                                # EH_LABEL
# %bb.49:
	.cfi_escape 0x2e, 0x00
.LBB2_50:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$285, %ecx                      # imm = 0x11D
	jmp	.LBB2_26
.LBB2_38:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp15:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp16:                                # EH_LABEL
# %bb.39:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$283, %ecx                      # imm = 0x11B
	jmp	.LBB2_26
.LBB2_32:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp9:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp10:                                # EH_LABEL
# %bb.33:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB2_25
.LBB2_23:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp3:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp4:                                 # EH_LABEL
# %bb.24:
	.cfi_escape 0x2e, 0x00
.LBB2_25:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$282, %ecx                      # imm = 0x11A
	jmp	.LBB2_26
.LBB2_78:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp49:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp50:                                # EH_LABEL
# %bb.79:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB2_80
.LBB2_86:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp55:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp56:                                # EH_LABEL
# %bb.87:
	.cfi_escape 0x2e, 0x00
.LBB2_80:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$287, %ecx                      # imm = 0x11F
.LBB2_26:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB2_68:
.Ltmp68:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp69:                                # EH_LABEL
# %bb.69:                               # %.noexc107
.LBB2_11:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$279, %ecx                      # imm = 0x117
	jmp	.LBB2_26
.LBB2_18:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp65:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp66:                                # EH_LABEL
# %bb.19:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$292, %ecx                      # imm = 0x124
	jmp	.LBB2_26
.LBB2_90:
.Ltmp62:                                # EH_LABEL
	jmp	.LBB2_94
.LBB2_91:                               # %.thread
.Ltmp67:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB2_96
.LBB2_82:                               # %.loopexit.split-lp163
.Ltmp70:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_81:                               # %.loopexit162
.Ltmp45:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_89:                               # %.loopexit.split-lp173
.Ltmp57:                                # EH_LABEL
	jmp	.LBB2_94
.LBB2_83:                               # %.loopexit.split-lp168
.Ltmp51:                                # EH_LABEL
	jmp	.LBB2_94
.LBB2_29:                               # %.loopexit.split-lp
.Ltmp5:                                 # EH_LABEL
	jmp	.LBB2_28
.LBB2_35:                               # %.loopexit.split-lp138
.Ltmp11:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_41:                               # %.loopexit.split-lp143
.Ltmp17:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_52:                               # %.loopexit.split-lp148
.Ltmp30:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_58:                               # %.loopexit.split-lp153
.Ltmp36:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_64:                               # %.loopexit.split-lp158
.Ltmp42:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_88:                               # %.loopexit172
.Ltmp54:                                # EH_LABEL
	jmp	.LBB2_94
.LBB2_93:                               # %.loopexit167
.Ltmp48:                                # EH_LABEL
.LBB2_94:
	movq	%rax, %rbx
	jmp	.LBB2_95
.LBB2_27:                               # %.loopexit
.Ltmp2:                                 # EH_LABEL
	jmp	.LBB2_28
.LBB2_34:                               # %.loopexit137
.Ltmp8:                                 # EH_LABEL
	jmp	.LBB2_28
.LBB2_40:                               # %.loopexit142
.Ltmp14:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_51:                               # %.loopexit147
.Ltmp27:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_57:                               # %.loopexit152
.Ltmp33:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_63:                               # %.loopexit157
.Ltmp39:                                # EH_LABEL
	jmp	.LBB2_28
.LBB2_98:
.Ltmp24:                                # EH_LABEL
.LBB2_28:
	movq	%rax, %rbx
	movq	%rbp, %r13
.LBB2_95:
	testq	%r13, %r13
	je	.LBB2_97
.LBB2_96:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB2_97:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit111
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end2:
	.size	_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy, .Lfunc_end2-_ZL9run_shapeILi0EEdRK5ShapePKDv2_jS5_PKjPfPy
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table2:
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
	.uleb128 .Ltmp8-.Lfunc_begin0           #     jumps to .Ltmp8
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          # >> Call Site 4 <<
	.uleb128 .Ltmp13-.Ltmp12                #   Call between .Ltmp12 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp18-.Lfunc_begin0          # >> Call Site 5 <<
	.uleb128 .Ltmp23-.Ltmp18                #   Call between .Ltmp18 and .Ltmp23
	.uleb128 .Ltmp24-.Lfunc_begin0          #     jumps to .Ltmp24
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp25-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp26-.Ltmp25                #   Call between .Ltmp25 and .Ltmp26
	.uleb128 .Ltmp27-.Lfunc_begin0          #     jumps to .Ltmp27
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp31-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp32-.Ltmp31                #   Call between .Ltmp31 and .Ltmp32
	.uleb128 .Ltmp33-.Lfunc_begin0          #     jumps to .Ltmp33
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp37-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp38-.Ltmp37                #   Call between .Ltmp37 and .Ltmp38
	.uleb128 .Ltmp39-.Lfunc_begin0          #     jumps to .Ltmp39
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp43-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp44-.Ltmp43                #   Call between .Ltmp43 and .Ltmp44
	.uleb128 .Ltmp45-.Lfunc_begin0          #     jumps to .Ltmp45
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp44-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp46-.Ltmp44                #   Call between .Ltmp44 and .Ltmp46
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp46-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp47-.Ltmp46                #   Call between .Ltmp46 and .Ltmp47
	.uleb128 .Ltmp48-.Lfunc_begin0          #     jumps to .Ltmp48
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp52-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp53-.Ltmp52                #   Call between .Ltmp52 and .Ltmp53
	.uleb128 .Ltmp54-.Lfunc_begin0          #     jumps to .Ltmp54
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp58-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp61-.Ltmp58                #   Call between .Ltmp58 and .Ltmp61
	.uleb128 .Ltmp62-.Lfunc_begin0          #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp63-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp64-.Ltmp63                #   Call between .Ltmp63 and .Ltmp64
	.uleb128 .Ltmp67-.Lfunc_begin0          #     jumps to .Ltmp67
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp40-.Lfunc_begin0          # >> Call Site 15 <<
	.uleb128 .Ltmp41-.Ltmp40                #   Call between .Ltmp40 and .Ltmp41
	.uleb128 .Ltmp42-.Lfunc_begin0          #     jumps to .Ltmp42
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp34-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Ltmp35-.Ltmp34                #   Call between .Ltmp34 and .Ltmp35
	.uleb128 .Ltmp36-.Lfunc_begin0          #     jumps to .Ltmp36
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp28-.Lfunc_begin0          # >> Call Site 17 <<
	.uleb128 .Ltmp29-.Ltmp28                #   Call between .Ltmp28 and .Ltmp29
	.uleb128 .Ltmp30-.Lfunc_begin0          #     jumps to .Ltmp30
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp16-.Ltmp15                #   Call between .Ltmp15 and .Ltmp16
	.uleb128 .Ltmp17-.Lfunc_begin0          #     jumps to .Ltmp17
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin0           # >> Call Site 19 <<
	.uleb128 .Ltmp10-.Ltmp9                 #   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp11-.Lfunc_begin0          #     jumps to .Ltmp11
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 20 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp49-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Ltmp50-.Ltmp49                #   Call between .Ltmp49 and .Ltmp50
	.uleb128 .Ltmp51-.Lfunc_begin0          #     jumps to .Ltmp51
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp55-.Lfunc_begin0          # >> Call Site 22 <<
	.uleb128 .Ltmp56-.Ltmp55                #   Call between .Ltmp55 and .Ltmp56
	.uleb128 .Ltmp57-.Lfunc_begin0          #     jumps to .Ltmp57
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp68-.Lfunc_begin0          # >> Call Site 23 <<
	.uleb128 .Ltmp69-.Ltmp68                #   Call between .Ltmp68 and .Ltmp69
	.uleb128 .Ltmp70-.Lfunc_begin0          #     jumps to .Ltmp70
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp69-.Lfunc_begin0          # >> Call Site 24 <<
	.uleb128 .Ltmp65-.Ltmp69                #   Call between .Ltmp69 and .Ltmp65
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp65-.Lfunc_begin0          # >> Call Site 25 <<
	.uleb128 .Ltmp66-.Ltmp65                #   Call between .Ltmp65 and .Ltmp66
	.uleb128 .Ltmp67-.Lfunc_begin0          #     jumps to .Ltmp67
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp66-.Lfunc_begin0          # >> Call Site 26 <<
	.uleb128 .Lfunc_end2-.Ltmp66            #   Call between .Ltmp66 and .Lfunc_end2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii # -- Begin function _Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.prefalign	4, .Lfunc_end3, nop
	.type	_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,@function
_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii: # @_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
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
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end3:
	.size	_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end3-_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.prefalign	4, .Lfunc_end4, nop
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
	jl	.LBB4_40
# %bb.1:                                # %.lr.ph
	movq	%rdx, %r14
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.LBB4_8
# %bb.2:                                # %.lr.ph43.preheader
	movq	$-4, %r13
	subq	%rbx, %r13
	.p2align	4
.LBB4_3:                                # %.lr.ph43
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_33 Depth 2
                                        #       Child Loop BB4_34 Depth 3
                                        #       Child Loop BB4_36 Depth 3
	shrq	%rbp
	movss	4(%rbx), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%rbx,%rbp,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm2
	movss	-4(%rsi), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	jbe	.LBB4_27
# %bb.4:                                #   in Loop: Header=BB4_3 Depth=1
	ucomiss	%xmm2, %xmm0
	jbe	.LBB4_24
# %bb.5:                                #   in Loop: Header=BB4_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm2, (%rbx)
	movss	%xmm0, (%rbx,%rbp,4)
	jmp	.LBB4_32
	.p2align	4
.LBB4_27:                               #   in Loop: Header=BB4_3 Depth=1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB4_29
# %bb.28:                               #   in Loop: Header=BB4_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx)
	movss	%xmm0, 4(%rbx)
	jmp	.LBB4_32
	.p2align	4
.LBB4_24:                               #   in Loop: Header=BB4_3 Depth=1
	ucomiss	%xmm1, %xmm0
	movss	(%rbx), %xmm2                   # xmm2 = mem[0],zero,zero,zero
	jbe	.LBB4_26
# %bb.25:                               #   in Loop: Header=BB4_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm2, -4(%rsi)
	jmp	.LBB4_32
	.p2align	4
.LBB4_29:                               #   in Loop: Header=BB4_3 Depth=1
	ucomiss	%xmm2, %xmm0
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	jbe	.LBB4_31
# %bb.30:                               #   in Loop: Header=BB4_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm1, -4(%rsi)
	jmp	.LBB4_32
.LBB4_26:                               #   in Loop: Header=BB4_3 Depth=1
	movss	%xmm1, (%rbx)
	movss	%xmm2, 4(%rbx)
	jmp	.LBB4_32
.LBB4_31:                               #   in Loop: Header=BB4_3 Depth=1
	movss	%xmm2, (%rbx)
	movss	%xmm1, (%rbx,%rbp,4)
	.p2align	4
.LBB4_32:                               # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i.preheader
                                        #   in Loop: Header=BB4_3 Depth=1
	decq	%r14
	leaq	4(%rbx), %r12
	movq	%rsi, %rax
	.p2align	4
.LBB4_33:                               # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i
                                        #   Parent Loop BB4_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB4_34 Depth 3
                                        #       Child Loop BB4_36 Depth 3
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	leaq	(%r12,%r13), %rbp
	.p2align	4
.LBB4_34:                               #   Parent Loop BB4_3 Depth=1
                                        #     Parent Loop BB4_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%r12), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	addq	$4, %r12
	addq	$4, %rbp
	ucomiss	%xmm1, %xmm0
	ja	.LBB4_34
# %bb.35:                               # %.preheader.i.i.preheader
                                        #   in Loop: Header=BB4_33 Depth=2
	leaq	-4(%r12), %r15
	.p2align	4
.LBB4_36:                               # %.preheader.i.i
                                        #   Parent Loop BB4_3 Depth=1
                                        #     Parent Loop BB4_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	-4(%rax), %xmm2                 # xmm2 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm2
	ja	.LBB4_36
# %bb.37:                               #   in Loop: Header=BB4_33 Depth=2
	cmpq	%rax, %r15
	jae	.LBB4_39
# %bb.38:                               #   in Loop: Header=BB4_33 Depth=2
	movss	%xmm2, (%r15)
	movss	%xmm1, (%rax)
	jmp	.LBB4_33
	.p2align	4
.LBB4_39:                               # %_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_.exit
                                        #   in Loop: Header=BB4_3 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	sarq	$2, %rbp
	cmpq	$16, %rbp
	jle	.LBB4_40
# %bb.6:                                #   in Loop: Header=BB4_3 Depth=1
	movq	%r15, %rsi
	testq	%r14, %r14
	jne	.LBB4_3
# %bb.7:                                # %._crit_edge.loopexit
	addq	$-4, %r12
	movq	%r12, %rsi
.LBB4_8:                                # %._crit_edge
	leaq	7(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%rsi, %r14
	callq	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	jmp	.LBB4_9
	.p2align	4
.LBB4_22:                               #   in Loop: Header=BB4_9 Depth=1
	xorl	%ecx, %ecx
.LBB4_23:                               # %_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_.exit.i.i
                                        #   in Loop: Header=BB4_9 Depth=1
	movss	%xmm0, (%rbx,%rcx,4)
	cmpq	$4, %rax
	jle	.LBB4_40
.LBB4_9:                                # %.lr.ph.i.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_12 Depth 2
                                        #     Child Loop BB4_20 Depth 2
	movss	-4(%r14), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, -4(%r14)
	addq	$-4, %r14
	movq	%r14, %rax
	subq	%rbx, %rax
	movq	%rax, %rdx
	sarq	$2, %rdx
	cmpq	$3, %rdx
	jl	.LBB4_10
# %bb.11:                               # %.lr.ph.i.i.i.i.preheader
                                        #   in Loop: Header=BB4_9 Depth=1
	leaq	-1(%rdx), %rcx
	shrq	$63, %rcx
	leaq	(%rdx,%rcx), %rsi
	decq	%rsi
	sarq	%rsi
	xorl	%edi, %edi
	jmp	.LBB4_12
	.p2align	4
.LBB4_14:                               # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB4_12 Depth=2
	leaq	2(,%rdi,2), %rcx
.LBB4_15:                               # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB4_12 Depth=2
	movss	(%rbx,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rdi,4)
	movq	%rcx, %rdi
	cmpq	%rsi, %rcx
	jge	.LBB4_16
.LBB4_12:                               # %.lr.ph.i.i.i.i
                                        #   Parent Loop BB4_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rdi,%rdi), %rcx
	movss	4(%rbx,%rcx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rbx,%rcx,4), %xmm1
	jbe	.LBB4_14
# %bb.13:                               #   in Loop: Header=BB4_12 Depth=2
	leaq	1(,%rdi,2), %rcx
	jmp	.LBB4_15
	.p2align	4
.LBB4_10:                               #   in Loop: Header=BB4_9 Depth=1
	xorl	%ecx, %ecx
.LBB4_16:                               # %._crit_edge.i.i.i.i
                                        #   in Loop: Header=BB4_9 Depth=1
	testb	$4, %al
	jne	.LBB4_19
# %bb.17:                               #   in Loop: Header=BB4_9 Depth=1
	addq	$-2, %rdx
	sarq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB4_19
# %bb.18:                               # %.thread.i.i.i
                                        #   in Loop: Header=BB4_9 Depth=1
	leaq	(%rcx,%rcx), %rdx
	movss	4(%rbx,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rcx,4)
	leaq	1(,%rcx,2), %rcx
	jmp	.LBB4_20
	.p2align	4
.LBB4_19:                               #   in Loop: Header=BB4_9 Depth=1
	testq	%rcx, %rcx
	je	.LBB4_22
	.p2align	4
.LBB4_20:                               # %.lr.ph.i.i.i.i.i
                                        #   Parent Loop BB4_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rcx), %rdx
	shrq	%rdx
	movss	(%rbx,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB4_23
# %bb.21:                               #   in Loop: Header=BB4_20 Depth=2
	movss	%xmm1, (%rbx,%rcx,4)
	movq	%rdx, %rcx
	testq	%rdx, %rdx
	jne	.LBB4_20
	jmp	.LBB4_22
.LBB4_40:                               # %_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_.exit
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
.Lfunc_end4:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end4-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ # -- Begin function _ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.prefalign	4, .Lfunc_end5, nop
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
	jl	.LBB5_17
# %bb.1:                                # %.lr.ph.i
	leaq	4(%r14), %r15
	movl	$4, %r12d
	movq	%r15, %r13
	movq	%r14, %rbp
	jmp	.LBB5_2
.LBB5_17:
	cmpq	%rbx, %r14
	je	.LBB5_29
# %bb.18:
	leaq	4(%r14), %rax
	cmpq	%rbx, %rax
	je	.LBB5_29
# %bb.19:                               # %.lr.ph.i15.preheader
	movq	%r14, %r15
	jmp	.LBB5_20
	.p2align	4
.LBB5_28:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i18
                                        #   in Loop: Header=BB5_20 Depth=1
	movss	%xmm1, (%rax)
	leaq	4(%r15), %rax
	cmpq	%rbx, %rax
	je	.LBB5_29
.LBB5_20:                               # %.lr.ph.i15
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_27 Depth 2
	movq	%r15, %rdi
	movq	%rax, %r15
	movss	4(%rdi), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB5_25
# %bb.21:                               # %_ZSt7advanceIPflEvRT_T0_.exit.i.i.i.i.i28
                                        #   in Loop: Header=BB5_20 Depth=1
	movss	%xmm1, 4(%rsp)                  # 4-byte Spill
	movq	%r15, %rdx
	subq	%r14, %rdx
	subq	%rdx, %rdi
	movq	%rdx, %rax
	sarq	$2, %rax
	addq	$8, %rdi
	cmpq	$2, %rax
	jl	.LBB5_23
# %bb.22:                               #   in Loop: Header=BB5_20 Depth=1
	movq	%r14, %rsi
	callq	memmove@PLT
	movq	%r14, %rax
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jmp	.LBB5_28
	.p2align	4
.LBB5_25:                               #   in Loop: Header=BB5_20 Depth=1
	movss	(%rdi), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%r15, %rax
	jbe	.LBB5_28
# %bb.26:                               # %.lr.ph.i.i22.preheader
                                        #   in Loop: Header=BB5_20 Depth=1
	movq	%r15, %rax
	.p2align	4
.LBB5_27:                               # %.lr.ph.i.i22
                                        #   Parent Loop BB5_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB5_27
	jmp	.LBB5_28
.LBB5_23:                               # %_ZSt7advanceIPflEvRT_T0_.exit.thread.i.i.i.i.i29
                                        #   in Loop: Header=BB5_20 Depth=1
	movq	%r14, %rax
	cmpq	$4, %rdx
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jne	.LBB5_28
# %bb.24:                               #   in Loop: Header=BB5_20 Depth=1
	movss	%xmm0, (%rdi)
	movq	%r14, %rax
	jmp	.LBB5_28
.LBB5_5:                                # %_ZSt7advanceIPflEvRT_T0_.exit.thread.i.i.i.i.i
                                        #   in Loop: Header=BB5_2 Depth=1
	movss	%xmm0, (%r15)
	.p2align	4
.LBB5_6:                                # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB5_2 Depth=1
	movq	%r14, %rax
	movss	4(%rsp), %xmm1                  # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB5_10:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB5_2 Depth=1
	movss	%xmm1, (%rax)
	addq	$4, %r12
	addq	$4, %r13
	cmpq	$64, %r12
	je	.LBB5_11
.LBB5_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_9 Depth 2
	movq	%rbp, %rax
	leaq	(%r14,%r12), %rbp
	movss	(%r14,%r12), %xmm1              # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB5_7
# %bb.3:                                # %_ZSt7advanceIPflEvRT_T0_.exit.i.i.i.i.i
                                        #   in Loop: Header=BB5_2 Depth=1
	movss	%xmm1, 4(%rsp)                  # 4-byte Spill
	cmpq	$5, %r12
	jb	.LBB5_5
# %bb.4:                                #   in Loop: Header=BB5_2 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
	jmp	.LBB5_6
	.p2align	4
.LBB5_7:                                #   in Loop: Header=BB5_2 Depth=1
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%rbp, %rax
	jbe	.LBB5_10
# %bb.8:                                # %.lr.ph.i.i.preheader
                                        #   in Loop: Header=BB5_2 Depth=1
	movq	%r13, %rax
	.p2align	4
.LBB5_9:                                # %.lr.ph.i.i
                                        #   Parent Loop BB5_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB5_9
	jmp	.LBB5_10
.LBB5_11:                               # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
	addq	$64, %r14
	jmp	.LBB5_12
	.p2align	4
.LBB5_16:                               # %_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops14_Val_less_iterEEvT_T0_.exit.i
                                        #   in Loop: Header=BB5_12 Depth=1
	movss	%xmm0, (%rax)
	addq	$4, %r14
.LBB5_12:                               # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_15 Depth 2
	cmpq	%rbx, %r14
	je	.LBB5_29
# %bb.13:                               # %.lr.ph.i6
                                        #   in Loop: Header=BB5_12 Depth=1
	movss	-4(%r14), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	movss	(%r14), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm0, %xmm1
	movq	%r14, %rax
	jbe	.LBB5_16
# %bb.14:                               # %.lr.ph.i.i8.preheader
                                        #   in Loop: Header=BB5_12 Depth=1
	movq	%r14, %rax
	.p2align	4
.LBB5_15:                               # %.lr.ph.i.i8
                                        #   Parent Loop BB5_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	%xmm1, (%rax)
	movss	-8(%rax), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm1
	ja	.LBB5_15
	jmp	.LBB5_16
.LBB5_29:                               # %_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
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
.Lfunc_end5:
	.size	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_, .Lfunc_end5-_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,"axG",@progbits,_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,comdat
	.weak	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ # -- Begin function _ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.prefalign	4, .Lfunc_end6, nop
	.type	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,@function
_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_: # @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_startproc
# %bb.0:
	subq	%rdi, %rsi
	movq	%rsi, %rax
	sarq	$2, %rax
	cmpq	$2, %rax
	jge	.LBB6_2
.LBB6_1:                                # %.loopexit
	retq
.LBB6_2:
	leaq	-2(%rax), %rdx
	movq	%rdx, %rcx
	shrq	%rcx
	decq	%rax
	shrq	%rax
	testb	$4, %sil
	jne	.LBB6_20
# %bb.3:                                # %.split.preheader
	incq	%rdx
	movq	%rcx, %rsi
	jmp	.LBB6_6
	.p2align	4
.LBB6_4:                                #   in Loop: Header=BB6_6 Depth=1
	movq	%r8, %r9
.LBB6_5:                                # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElfNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit
                                        #   in Loop: Header=BB6_6 Depth=1
	movss	%xmm0, (%rdi,%r9,4)
	subq	$1, %rsi
	jb	.LBB6_1
.LBB6_6:                                # %.split
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB6_10 Depth 2
                                        #     Child Loop BB6_15 Depth 2
	movss	(%rdi,%rsi,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rsi, %r8
	cmpq	%rax, %rsi
	jge	.LBB6_12
# %bb.7:                                # %.lr.ph.i.preheader
                                        #   in Loop: Header=BB6_6 Depth=1
	movq	%rsi, %r9
	jmp	.LBB6_10
	.p2align	4
.LBB6_8:                                # %.lr.ph.i
                                        #   in Loop: Header=BB6_10 Depth=2
	leaq	2(,%r9,2), %r8
.LBB6_9:                                # %.lr.ph.i
                                        #   in Loop: Header=BB6_10 Depth=2
	movss	(%rdi,%r8,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%r9,4)
	movq	%r8, %r9
	cmpq	%rax, %r8
	jge	.LBB6_12
.LBB6_10:                               # %.lr.ph.i
                                        #   Parent Loop BB6_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r9,%r9), %r8
	movss	4(%rdi,%r8,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rdi,%r8,4), %xmm1
	jbe	.LBB6_8
# %bb.11:                               #   in Loop: Header=BB6_10 Depth=2
	leaq	1(,%r9,2), %r8
	jmp	.LBB6_9
	.p2align	4
.LBB6_12:                               # %._crit_edge.i
                                        #   in Loop: Header=BB6_6 Depth=1
	cmpq	%rcx, %r8
	jne	.LBB6_14
# %bb.13:                               #   in Loop: Header=BB6_6 Depth=1
	movss	(%rdi,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%rcx,4)
	movq	%rdx, %r8
.LBB6_14:                               #   in Loop: Header=BB6_6 Depth=1
	cmpq	%rsi, %r8
	jle	.LBB6_4
	.p2align	4
.LBB6_15:                               # %.lr.ph.i.i
                                        #   Parent Loop BB6_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%r8), %r9
	shrq	$63, %r9
	addq	%r8, %r9
	decq	%r9
	sarq	%r9
	movss	(%rdi,%r9,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB6_4
# %bb.16:                               #   in Loop: Header=BB6_15 Depth=2
	movss	%xmm1, (%rdi,%r8,4)
	movq	%r9, %r8
	cmpq	%rsi, %r9
	jg	.LBB6_15
	jmp	.LBB6_5
	.p2align	4
.LBB6_18:                               #   in Loop: Header=BB6_20 Depth=1
	movq	%rdx, %rsi
.LBB6_19:                               # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElfNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit.us
                                        #   in Loop: Header=BB6_20 Depth=1
	movss	%xmm0, (%rdi,%rsi,4)
	subq	$1, %rcx
	jb	.LBB6_1
.LBB6_20:                               # %.split.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB6_24 Depth 2
                                        #     Child Loop BB6_27 Depth 2
	movss	(%rdi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rcx, %rsi
	cmpq	%rax, %rcx
	jge	.LBB6_19
# %bb.21:                               # %.lr.ph.i.us.preheader
                                        #   in Loop: Header=BB6_20 Depth=1
	movq	%rcx, %rsi
	jmp	.LBB6_24
	.p2align	4
.LBB6_22:                               # %.lr.ph.i.us
                                        #   in Loop: Header=BB6_24 Depth=2
	leaq	2(,%rsi,2), %rdx
.LBB6_23:                               # %.lr.ph.i.us
                                        #   in Loop: Header=BB6_24 Depth=2
	movss	(%rdi,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rdi,%rsi,4)
	movq	%rdx, %rsi
	cmpq	%rax, %rdx
	jge	.LBB6_26
.LBB6_24:                               # %.lr.ph.i.us
                                        #   Parent Loop BB6_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rsi,%rsi), %rdx
	movss	4(%rdi,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rdi,%rdx,4), %xmm1
	jbe	.LBB6_22
# %bb.25:                               #   in Loop: Header=BB6_24 Depth=2
	leaq	1(,%rsi,2), %rdx
	jmp	.LBB6_23
	.p2align	4
.LBB6_26:                               # %._crit_edge.i.us
                                        #   in Loop: Header=BB6_20 Depth=1
	cmpq	%rcx, %rdx
	jle	.LBB6_18
	.p2align	4
.LBB6_27:                               # %.lr.ph.i.i.us
                                        #   Parent Loop BB6_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rdx), %rsi
	shrq	$63, %rsi
	addq	%rdx, %rsi
	decq	%rsi
	sarq	%rsi
	movss	(%rdi,%rsi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB6_18
# %bb.28:                               #   in Loop: Header=BB6_27 Depth=2
	movss	%xmm1, (%rdi,%rdx,4)
	movq	%rsi, %rdx
	cmpq	%rcx, %rsi
	jg	.LBB6_27
	jmp	.LBB6_19
.Lfunc_end6:
	.size	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_, .Lfunc_end6-_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy
.LCPI7_0:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI7_1:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI7_2:
	.quad	0x4020000000000000              # double 8
.LCPI7_3:
	.quad	0x4040000000000000              # double 32
.LCPI7_4:
	.quad	0x40d0000000000000              # double 16384
.LCPI7_5:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.prefalign	4, .Lfunc_end7, nop
	.type	_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy,@function
_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy: # @_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 112(%rsp)                  # 8-byte Spill
	movq	%r8, 216(%rsp)                  # 8-byte Spill
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	movq	%rdx, 200(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movq	%rdi, 120(%rsp)                 # 8-byte Spill
	movq	8(%rdi), %rbx
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_2
# %bb.1:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_2:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_4
# %bb.3:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_4:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_6
# %bb.5:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_6:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_8
# %bb.7:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_8:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, 240(%rsp)                 # 8-byte Spill
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_10
# %bb.9:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_10:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB7_11
# %bb.20:                               # %.preheader
	movl	$20, %ebx
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	movq	240(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB7_21:                               # =>This Inner Loop Header: Depth=1
.Ltmp71:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp72:                                # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_23
# %bb.30:                               #   in Loop: Header=BB7_21 Depth=1
.Ltmp77:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp78:                                # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_32
# %bb.36:                               #   in Loop: Header=BB7_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp83:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp84:                                # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_38
# %bb.42:                               #   in Loop: Header=BB7_21 Depth=1
.Ltmp89:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%r14, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp90:                                # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_46
# %bb.44:                               #   in Loop: Header=BB7_21 Depth=1
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movl	%eax, 4(%rsp)
	movl	%ecx, 236(%rsp)
	movl	%edx, 232(%rsp)
	movq	192(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 104(%rsp)
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	208(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	216(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp91:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp92:                                # EH_LABEL
# %bb.45:                               # %.noexc104
                                        #   in Loop: Header=BB7_21 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp93:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp94:                                # EH_LABEL
.LBB7_46:                               #   in Loop: Header=BB7_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp96:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp97:                                # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_48
# %bb.53:                               #   in Loop: Header=BB7_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp102:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp103:                               # EH_LABEL
# %bb.54:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_55
# %bb.59:                               #   in Loop: Header=BB7_21 Depth=1
	movl	$0, 128(%rsp)
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
.Ltmp108:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp109:                               # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_61
# %bb.65:                               #   in Loop: Header=BB7_21 Depth=1
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI7_0(%rip), %xmm0
	cmpq	%r12, %r15
	je	.LBB7_67
# %bb.66:                               #   in Loop: Header=BB7_21 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r13
	jmp	.LBB7_76
	.p2align	4
.LBB7_67:                               #   in Loop: Header=BB7_21 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%r12, %r15
	subq	%rbp, %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r15
	je	.LBB7_68
# %bb.70:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB7_21 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp114:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp115:                               # EH_LABEL
# %bb.71:                               # %.noexc108
                                        #   in Loop: Header=BB7_21 Depth=1
	movq	%rax, %r13
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB7_73
# %bb.72:                               #   in Loop: Header=BB7_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB7_73:                               # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB7_21 Depth=1
	testq	%rbp, %rbp
	je	.LBB7_75
# %bb.74:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB7_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB7_75:                               # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB7_21 Depth=1
	addq	%r13, %r15
	leaq	(,%r14,4), %r12
	addq	%r13, %r12
	movq	240(%rsp), %r14                 # 8-byte Reload
.LBB7_76:                               # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB7_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp117:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp118:                               # EH_LABEL
# %bb.77:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_78
# %bb.84:                               #   in Loop: Header=BB7_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp123:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp124:                               # EH_LABEL
# %bb.85:                               #   in Loop: Header=BB7_21 Depth=1
	testl	%eax, %eax
	jne	.LBB7_86
# %bb.12:                               #   in Loop: Header=BB7_21 Depth=1
	addq	$4, %r15
	movq	%r13, %rbp
	decl	%ebx
	jne	.LBB7_21
# %bb.13:
	cmpq	%r15, %r13
	je	.LBB7_16
# %bb.14:
	movq	%r15, %rax
	subq	%r13, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp129:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp130:                               # EH_LABEL
# %bb.15:                               # %.noexc
.Ltmp131:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp132:                               # EH_LABEL
.LBB7_16:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_.exit
	movss	36(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movss	40(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 112(%rsp)                # 4-byte Spill
	movl	$0, 128(%rsp)
.Ltmp134:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	128(%rsp), %rdi
	movl	$20480, %ecx                    # imm = 0x5000
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp135:                               # EH_LABEL
# %bb.17:                               # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKDv2_jS2_PKjPfPyiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB7_18
# %bb.92:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addss	112(%rsp), %xmm0                # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm3
	mulsd	.LCPI7_1(%rip), %xmm3
	movq	120(%rsp), %rax                 # 8-byte Reload
	cvtdq2pd	8(%rax), %xmm0
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	mulsd	%xmm0, %xmm2
	mulsd	.LCPI7_2(%rip), %xmm2
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	mulsd	.LCPI7_3(%rip), %xmm1
	mulsd	.LCPI7_4(%rip), %xmm1
	movq	(%rax), %rdx
	movapd	%xmm3, %xmm0
	movsd	%xmm3, 224(%rsp)                # 8-byte Spill
	divsd	%xmm3, %xmm1
	divsd	.LCPI7_5(%rip), %xmm1
	movl	128(%rsp), %ecx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rdi
	movl	$1, %esi
	movb	$2, %al
	callq	printf@PLT
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	224(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addq	$248, %rsp
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
.LBB7_61:
	.cfi_def_cfa_offset 304
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp111:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp112:                               # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$286, %ecx                      # imm = 0x11E
	jmp	.LBB7_26
.LBB7_55:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp105:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp106:                               # EH_LABEL
# %bb.56:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB7_50
.LBB7_48:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp99:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp100:                               # EH_LABEL
# %bb.49:
	.cfi_escape 0x2e, 0x00
.LBB7_50:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$285, %ecx                      # imm = 0x11D
	jmp	.LBB7_26
.LBB7_38:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp86:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp87:                                # EH_LABEL
# %bb.39:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$283, %ecx                      # imm = 0x11B
	jmp	.LBB7_26
.LBB7_32:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp80:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp81:                                # EH_LABEL
# %bb.33:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB7_25
.LBB7_23:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp74:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp75:                                # EH_LABEL
# %bb.24:
	.cfi_escape 0x2e, 0x00
.LBB7_25:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$282, %ecx                      # imm = 0x11A
	jmp	.LBB7_26
.LBB7_78:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp120:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp121:                               # EH_LABEL
# %bb.79:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB7_80
.LBB7_86:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp126:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp127:                               # EH_LABEL
# %bb.87:
	.cfi_escape 0x2e, 0x00
.LBB7_80:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$287, %ecx                      # imm = 0x11F
.LBB7_26:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB7_68:
.Ltmp139:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp140:                               # EH_LABEL
# %bb.69:                               # %.noexc107
.LBB7_11:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$279, %ecx                      # imm = 0x117
	jmp	.LBB7_26
.LBB7_18:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp136:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp137:                               # EH_LABEL
# %bb.19:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$292, %ecx                      # imm = 0x124
	jmp	.LBB7_26
.LBB7_90:
.Ltmp133:                               # EH_LABEL
	jmp	.LBB7_94
.LBB7_91:                               # %.thread
.Ltmp138:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB7_96
.LBB7_82:                               # %.loopexit.split-lp163
.Ltmp141:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_81:                               # %.loopexit162
.Ltmp116:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_89:                               # %.loopexit.split-lp173
.Ltmp128:                               # EH_LABEL
	jmp	.LBB7_94
.LBB7_83:                               # %.loopexit.split-lp168
.Ltmp122:                               # EH_LABEL
	jmp	.LBB7_94
.LBB7_29:                               # %.loopexit.split-lp
.Ltmp76:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_35:                               # %.loopexit.split-lp138
.Ltmp82:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_41:                               # %.loopexit.split-lp143
.Ltmp88:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_52:                               # %.loopexit.split-lp148
.Ltmp101:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_58:                               # %.loopexit.split-lp153
.Ltmp107:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_64:                               # %.loopexit.split-lp158
.Ltmp113:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_88:                               # %.loopexit172
.Ltmp125:                               # EH_LABEL
	jmp	.LBB7_94
.LBB7_93:                               # %.loopexit167
.Ltmp119:                               # EH_LABEL
.LBB7_94:
	movq	%rax, %rbx
	jmp	.LBB7_95
.LBB7_27:                               # %.loopexit
.Ltmp73:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_34:                               # %.loopexit137
.Ltmp79:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_40:                               # %.loopexit142
.Ltmp85:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_51:                               # %.loopexit147
.Ltmp98:                                # EH_LABEL
	jmp	.LBB7_28
.LBB7_57:                               # %.loopexit152
.Ltmp104:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_63:                               # %.loopexit157
.Ltmp110:                               # EH_LABEL
	jmp	.LBB7_28
.LBB7_98:
.Ltmp95:                                # EH_LABEL
.LBB7_28:
	movq	%rax, %rbx
	movq	%rbp, %r13
.LBB7_95:
	testq	%r13, %r13
	je	.LBB7_97
.LBB7_96:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB7_97:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit111
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end7:
	.size	_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy, .Lfunc_end7-_ZL9run_shapeILi1EEdRK5ShapePKDv2_jS5_PKjPfPy
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table7:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp71-.Lfunc_begin1          #   Call between .Lfunc_begin1 and .Ltmp71
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp71-.Lfunc_begin1          # >> Call Site 2 <<
	.uleb128 .Ltmp72-.Ltmp71                #   Call between .Ltmp71 and .Ltmp72
	.uleb128 .Ltmp73-.Lfunc_begin1          #     jumps to .Ltmp73
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp77-.Lfunc_begin1          # >> Call Site 3 <<
	.uleb128 .Ltmp78-.Ltmp77                #   Call between .Ltmp77 and .Ltmp78
	.uleb128 .Ltmp79-.Lfunc_begin1          #     jumps to .Ltmp79
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp83-.Lfunc_begin1          # >> Call Site 4 <<
	.uleb128 .Ltmp84-.Ltmp83                #   Call between .Ltmp83 and .Ltmp84
	.uleb128 .Ltmp85-.Lfunc_begin1          #     jumps to .Ltmp85
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp89-.Lfunc_begin1          # >> Call Site 5 <<
	.uleb128 .Ltmp94-.Ltmp89                #   Call between .Ltmp89 and .Ltmp94
	.uleb128 .Ltmp95-.Lfunc_begin1          #     jumps to .Ltmp95
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp96-.Lfunc_begin1          # >> Call Site 6 <<
	.uleb128 .Ltmp97-.Ltmp96                #   Call between .Ltmp96 and .Ltmp97
	.uleb128 .Ltmp98-.Lfunc_begin1          #     jumps to .Ltmp98
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp102-.Lfunc_begin1         # >> Call Site 7 <<
	.uleb128 .Ltmp103-.Ltmp102              #   Call between .Ltmp102 and .Ltmp103
	.uleb128 .Ltmp104-.Lfunc_begin1         #     jumps to .Ltmp104
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp108-.Lfunc_begin1         # >> Call Site 8 <<
	.uleb128 .Ltmp109-.Ltmp108              #   Call between .Ltmp108 and .Ltmp109
	.uleb128 .Ltmp110-.Lfunc_begin1         #     jumps to .Ltmp110
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp114-.Lfunc_begin1         # >> Call Site 9 <<
	.uleb128 .Ltmp115-.Ltmp114              #   Call between .Ltmp114 and .Ltmp115
	.uleb128 .Ltmp116-.Lfunc_begin1         #     jumps to .Ltmp116
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp115-.Lfunc_begin1         # >> Call Site 10 <<
	.uleb128 .Ltmp117-.Ltmp115              #   Call between .Ltmp115 and .Ltmp117
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp117-.Lfunc_begin1         # >> Call Site 11 <<
	.uleb128 .Ltmp118-.Ltmp117              #   Call between .Ltmp117 and .Ltmp118
	.uleb128 .Ltmp119-.Lfunc_begin1         #     jumps to .Ltmp119
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp123-.Lfunc_begin1         # >> Call Site 12 <<
	.uleb128 .Ltmp124-.Ltmp123              #   Call between .Ltmp123 and .Ltmp124
	.uleb128 .Ltmp125-.Lfunc_begin1         #     jumps to .Ltmp125
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp129-.Lfunc_begin1         # >> Call Site 13 <<
	.uleb128 .Ltmp132-.Ltmp129              #   Call between .Ltmp129 and .Ltmp132
	.uleb128 .Ltmp133-.Lfunc_begin1         #     jumps to .Ltmp133
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp134-.Lfunc_begin1         # >> Call Site 14 <<
	.uleb128 .Ltmp135-.Ltmp134              #   Call between .Ltmp134 and .Ltmp135
	.uleb128 .Ltmp138-.Lfunc_begin1         #     jumps to .Ltmp138
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp111-.Lfunc_begin1         # >> Call Site 15 <<
	.uleb128 .Ltmp112-.Ltmp111              #   Call between .Ltmp111 and .Ltmp112
	.uleb128 .Ltmp113-.Lfunc_begin1         #     jumps to .Ltmp113
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp105-.Lfunc_begin1         # >> Call Site 16 <<
	.uleb128 .Ltmp106-.Ltmp105              #   Call between .Ltmp105 and .Ltmp106
	.uleb128 .Ltmp107-.Lfunc_begin1         #     jumps to .Ltmp107
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp99-.Lfunc_begin1          # >> Call Site 17 <<
	.uleb128 .Ltmp100-.Ltmp99               #   Call between .Ltmp99 and .Ltmp100
	.uleb128 .Ltmp101-.Lfunc_begin1         #     jumps to .Ltmp101
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp86-.Lfunc_begin1          # >> Call Site 18 <<
	.uleb128 .Ltmp87-.Ltmp86                #   Call between .Ltmp86 and .Ltmp87
	.uleb128 .Ltmp88-.Lfunc_begin1          #     jumps to .Ltmp88
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp80-.Lfunc_begin1          # >> Call Site 19 <<
	.uleb128 .Ltmp81-.Ltmp80                #   Call between .Ltmp80 and .Ltmp81
	.uleb128 .Ltmp82-.Lfunc_begin1          #     jumps to .Ltmp82
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp74-.Lfunc_begin1          # >> Call Site 20 <<
	.uleb128 .Ltmp75-.Ltmp74                #   Call between .Ltmp74 and .Ltmp75
	.uleb128 .Ltmp76-.Lfunc_begin1          #     jumps to .Ltmp76
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp120-.Lfunc_begin1         # >> Call Site 21 <<
	.uleb128 .Ltmp121-.Ltmp120              #   Call between .Ltmp120 and .Ltmp121
	.uleb128 .Ltmp122-.Lfunc_begin1         #     jumps to .Ltmp122
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp126-.Lfunc_begin1         # >> Call Site 22 <<
	.uleb128 .Ltmp127-.Ltmp126              #   Call between .Ltmp126 and .Ltmp127
	.uleb128 .Ltmp128-.Lfunc_begin1         #     jumps to .Ltmp128
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp139-.Lfunc_begin1         # >> Call Site 23 <<
	.uleb128 .Ltmp140-.Ltmp139              #   Call between .Ltmp139 and .Ltmp140
	.uleb128 .Ltmp141-.Lfunc_begin1         #     jumps to .Ltmp141
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp140-.Lfunc_begin1         # >> Call Site 24 <<
	.uleb128 .Ltmp136-.Ltmp140              #   Call between .Ltmp140 and .Ltmp136
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp136-.Lfunc_begin1         # >> Call Site 25 <<
	.uleb128 .Ltmp137-.Ltmp136              #   Call between .Ltmp136 and .Ltmp137
	.uleb128 .Ltmp138-.Lfunc_begin1         #     jumps to .Ltmp138
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp137-.Lfunc_begin1         # >> Call Site 26 <<
	.uleb128 .Lfunc_end7-.Ltmp137           #   Call between .Ltmp137 and .Lfunc_end7
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii # -- Begin function _Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.prefalign	4, .Lfunc_end8, nop
	.type	_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,@function
_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii: # @_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
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
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end8:
	.size	_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end8-_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy
.LCPI9_0:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI9_1:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI9_2:
	.quad	0x4020000000000000              # double 8
.LCPI9_3:
	.quad	0x4040000000000000              # double 32
.LCPI9_4:
	.quad	0x40d0000000000000              # double 16384
.LCPI9_5:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.prefalign	4, .Lfunc_end9, nop
	.type	_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy,@function
_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy: # @_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception2
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 112(%rsp)                  # 8-byte Spill
	movq	%r8, 216(%rsp)                  # 8-byte Spill
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	movq	%rdx, 200(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movq	%rdi, 120(%rsp)                 # 8-byte Spill
	movq	8(%rdi), %rbx
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_2
# %bb.1:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_2:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_4
# %bb.3:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_4:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_6
# %bb.5:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_6:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_8
# %bb.7:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_8:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, 240(%rsp)                 # 8-byte Spill
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_10
# %bb.9:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_10:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB9_11
# %bb.20:                               # %.preheader
	movl	$20, %ebx
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	movq	240(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB9_21:                               # =>This Inner Loop Header: Depth=1
.Ltmp142:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp143:                               # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_23
# %bb.30:                               #   in Loop: Header=BB9_21 Depth=1
.Ltmp148:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp149:                               # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_32
# %bb.36:                               #   in Loop: Header=BB9_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp154:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp155:                               # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_38
# %bb.42:                               #   in Loop: Header=BB9_21 Depth=1
.Ltmp160:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%r14, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp161:                               # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_46
# %bb.44:                               #   in Loop: Header=BB9_21 Depth=1
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movl	%eax, 4(%rsp)
	movl	%ecx, 236(%rsp)
	movl	%edx, 232(%rsp)
	movq	192(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 104(%rsp)
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	208(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	216(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp162:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp163:                               # EH_LABEL
# %bb.45:                               # %.noexc104
                                        #   in Loop: Header=BB9_21 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp164:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp165:                               # EH_LABEL
.LBB9_46:                               #   in Loop: Header=BB9_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp167:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp168:                               # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_48
# %bb.53:                               #   in Loop: Header=BB9_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp173:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp174:                               # EH_LABEL
# %bb.54:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_55
# %bb.59:                               #   in Loop: Header=BB9_21 Depth=1
	movl	$0, 128(%rsp)
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
.Ltmp179:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp180:                               # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_61
# %bb.65:                               #   in Loop: Header=BB9_21 Depth=1
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI9_0(%rip), %xmm0
	cmpq	%r12, %r15
	je	.LBB9_67
# %bb.66:                               #   in Loop: Header=BB9_21 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r13
	jmp	.LBB9_76
	.p2align	4
.LBB9_67:                               #   in Loop: Header=BB9_21 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%r12, %r15
	subq	%rbp, %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r15
	je	.LBB9_68
# %bb.70:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB9_21 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp185:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp186:                               # EH_LABEL
# %bb.71:                               # %.noexc108
                                        #   in Loop: Header=BB9_21 Depth=1
	movq	%rax, %r13
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB9_73
# %bb.72:                               #   in Loop: Header=BB9_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB9_73:                               # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB9_21 Depth=1
	testq	%rbp, %rbp
	je	.LBB9_75
# %bb.74:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB9_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB9_75:                               # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB9_21 Depth=1
	addq	%r13, %r15
	leaq	(,%r14,4), %r12
	addq	%r13, %r12
	movq	240(%rsp), %r14                 # 8-byte Reload
.LBB9_76:                               # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB9_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp188:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp189:                               # EH_LABEL
# %bb.77:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_78
# %bb.84:                               #   in Loop: Header=BB9_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp194:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp195:                               # EH_LABEL
# %bb.85:                               #   in Loop: Header=BB9_21 Depth=1
	testl	%eax, %eax
	jne	.LBB9_86
# %bb.12:                               #   in Loop: Header=BB9_21 Depth=1
	addq	$4, %r15
	movq	%r13, %rbp
	decl	%ebx
	jne	.LBB9_21
# %bb.13:
	cmpq	%r15, %r13
	je	.LBB9_16
# %bb.14:
	movq	%r15, %rax
	subq	%r13, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp200:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp201:                               # EH_LABEL
# %bb.15:                               # %.noexc
.Ltmp202:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp203:                               # EH_LABEL
.LBB9_16:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_.exit
	movss	36(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movss	40(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 112(%rsp)                # 4-byte Spill
	movl	$0, 128(%rsp)
.Ltmp205:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	128(%rsp), %rdi
	movl	$20480, %ecx                    # imm = 0x5000
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp206:                               # EH_LABEL
# %bb.17:                               # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKDv2_jS2_PKjPfPyiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB9_18
# %bb.92:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addss	112(%rsp), %xmm0                # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm3
	mulsd	.LCPI9_1(%rip), %xmm3
	movq	120(%rsp), %rax                 # 8-byte Reload
	cvtdq2pd	8(%rax), %xmm0
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	mulsd	%xmm0, %xmm2
	mulsd	.LCPI9_2(%rip), %xmm2
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	mulsd	.LCPI9_3(%rip), %xmm1
	mulsd	.LCPI9_4(%rip), %xmm1
	movq	(%rax), %rdx
	movapd	%xmm3, %xmm0
	movsd	%xmm3, 224(%rsp)                # 8-byte Spill
	divsd	%xmm3, %xmm1
	divsd	.LCPI9_5(%rip), %xmm1
	movl	128(%rsp), %ecx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rdi
	movl	$2, %esi
	movb	$2, %al
	callq	printf@PLT
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	224(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addq	$248, %rsp
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
.LBB9_61:
	.cfi_def_cfa_offset 304
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp182:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp183:                               # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$286, %ecx                      # imm = 0x11E
	jmp	.LBB9_26
.LBB9_55:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp176:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp177:                               # EH_LABEL
# %bb.56:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB9_50
.LBB9_48:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp170:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp171:                               # EH_LABEL
# %bb.49:
	.cfi_escape 0x2e, 0x00
.LBB9_50:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$285, %ecx                      # imm = 0x11D
	jmp	.LBB9_26
.LBB9_38:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp157:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp158:                               # EH_LABEL
# %bb.39:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$283, %ecx                      # imm = 0x11B
	jmp	.LBB9_26
.LBB9_32:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp151:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp152:                               # EH_LABEL
# %bb.33:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB9_25
.LBB9_23:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp145:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp146:                               # EH_LABEL
# %bb.24:
	.cfi_escape 0x2e, 0x00
.LBB9_25:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$282, %ecx                      # imm = 0x11A
	jmp	.LBB9_26
.LBB9_78:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp191:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp192:                               # EH_LABEL
# %bb.79:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB9_80
.LBB9_86:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp197:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp198:                               # EH_LABEL
# %bb.87:
	.cfi_escape 0x2e, 0x00
.LBB9_80:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$287, %ecx                      # imm = 0x11F
.LBB9_26:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB9_68:
.Ltmp210:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp211:                               # EH_LABEL
# %bb.69:                               # %.noexc107
.LBB9_11:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$279, %ecx                      # imm = 0x117
	jmp	.LBB9_26
.LBB9_18:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp207:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp208:                               # EH_LABEL
# %bb.19:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$292, %ecx                      # imm = 0x124
	jmp	.LBB9_26
.LBB9_90:
.Ltmp204:                               # EH_LABEL
	jmp	.LBB9_94
.LBB9_91:                               # %.thread
.Ltmp209:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB9_96
.LBB9_82:                               # %.loopexit.split-lp163
.Ltmp212:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_81:                               # %.loopexit162
.Ltmp187:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_89:                               # %.loopexit.split-lp173
.Ltmp199:                               # EH_LABEL
	jmp	.LBB9_94
.LBB9_83:                               # %.loopexit.split-lp168
.Ltmp193:                               # EH_LABEL
	jmp	.LBB9_94
.LBB9_29:                               # %.loopexit.split-lp
.Ltmp147:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_35:                               # %.loopexit.split-lp138
.Ltmp153:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_41:                               # %.loopexit.split-lp143
.Ltmp159:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_52:                               # %.loopexit.split-lp148
.Ltmp172:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_58:                               # %.loopexit.split-lp153
.Ltmp178:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_64:                               # %.loopexit.split-lp158
.Ltmp184:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_88:                               # %.loopexit172
.Ltmp196:                               # EH_LABEL
	jmp	.LBB9_94
.LBB9_93:                               # %.loopexit167
.Ltmp190:                               # EH_LABEL
.LBB9_94:
	movq	%rax, %rbx
	jmp	.LBB9_95
.LBB9_27:                               # %.loopexit
.Ltmp144:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_34:                               # %.loopexit137
.Ltmp150:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_40:                               # %.loopexit142
.Ltmp156:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_51:                               # %.loopexit147
.Ltmp169:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_57:                               # %.loopexit152
.Ltmp175:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_63:                               # %.loopexit157
.Ltmp181:                               # EH_LABEL
	jmp	.LBB9_28
.LBB9_98:
.Ltmp166:                               # EH_LABEL
.LBB9_28:
	movq	%rax, %rbx
	movq	%rbp, %r13
.LBB9_95:
	testq	%r13, %r13
	je	.LBB9_97
.LBB9_96:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB9_97:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit111
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end9:
	.size	_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy, .Lfunc_end9-_ZL9run_shapeILi2EEdRK5ShapePKDv2_jS5_PKjPfPy
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table9:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Lfunc_begin2-.Lfunc_begin2    # >> Call Site 1 <<
	.uleb128 .Ltmp142-.Lfunc_begin2         #   Call between .Lfunc_begin2 and .Ltmp142
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp142-.Lfunc_begin2         # >> Call Site 2 <<
	.uleb128 .Ltmp143-.Ltmp142              #   Call between .Ltmp142 and .Ltmp143
	.uleb128 .Ltmp144-.Lfunc_begin2         #     jumps to .Ltmp144
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp148-.Lfunc_begin2         # >> Call Site 3 <<
	.uleb128 .Ltmp149-.Ltmp148              #   Call between .Ltmp148 and .Ltmp149
	.uleb128 .Ltmp150-.Lfunc_begin2         #     jumps to .Ltmp150
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp154-.Lfunc_begin2         # >> Call Site 4 <<
	.uleb128 .Ltmp155-.Ltmp154              #   Call between .Ltmp154 and .Ltmp155
	.uleb128 .Ltmp156-.Lfunc_begin2         #     jumps to .Ltmp156
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp160-.Lfunc_begin2         # >> Call Site 5 <<
	.uleb128 .Ltmp165-.Ltmp160              #   Call between .Ltmp160 and .Ltmp165
	.uleb128 .Ltmp166-.Lfunc_begin2         #     jumps to .Ltmp166
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp167-.Lfunc_begin2         # >> Call Site 6 <<
	.uleb128 .Ltmp168-.Ltmp167              #   Call between .Ltmp167 and .Ltmp168
	.uleb128 .Ltmp169-.Lfunc_begin2         #     jumps to .Ltmp169
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp173-.Lfunc_begin2         # >> Call Site 7 <<
	.uleb128 .Ltmp174-.Ltmp173              #   Call between .Ltmp173 and .Ltmp174
	.uleb128 .Ltmp175-.Lfunc_begin2         #     jumps to .Ltmp175
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp179-.Lfunc_begin2         # >> Call Site 8 <<
	.uleb128 .Ltmp180-.Ltmp179              #   Call between .Ltmp179 and .Ltmp180
	.uleb128 .Ltmp181-.Lfunc_begin2         #     jumps to .Ltmp181
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp185-.Lfunc_begin2         # >> Call Site 9 <<
	.uleb128 .Ltmp186-.Ltmp185              #   Call between .Ltmp185 and .Ltmp186
	.uleb128 .Ltmp187-.Lfunc_begin2         #     jumps to .Ltmp187
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp186-.Lfunc_begin2         # >> Call Site 10 <<
	.uleb128 .Ltmp188-.Ltmp186              #   Call between .Ltmp186 and .Ltmp188
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp188-.Lfunc_begin2         # >> Call Site 11 <<
	.uleb128 .Ltmp189-.Ltmp188              #   Call between .Ltmp188 and .Ltmp189
	.uleb128 .Ltmp190-.Lfunc_begin2         #     jumps to .Ltmp190
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp194-.Lfunc_begin2         # >> Call Site 12 <<
	.uleb128 .Ltmp195-.Ltmp194              #   Call between .Ltmp194 and .Ltmp195
	.uleb128 .Ltmp196-.Lfunc_begin2         #     jumps to .Ltmp196
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp200-.Lfunc_begin2         # >> Call Site 13 <<
	.uleb128 .Ltmp203-.Ltmp200              #   Call between .Ltmp200 and .Ltmp203
	.uleb128 .Ltmp204-.Lfunc_begin2         #     jumps to .Ltmp204
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp205-.Lfunc_begin2         # >> Call Site 14 <<
	.uleb128 .Ltmp206-.Ltmp205              #   Call between .Ltmp205 and .Ltmp206
	.uleb128 .Ltmp209-.Lfunc_begin2         #     jumps to .Ltmp209
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp182-.Lfunc_begin2         # >> Call Site 15 <<
	.uleb128 .Ltmp183-.Ltmp182              #   Call between .Ltmp182 and .Ltmp183
	.uleb128 .Ltmp184-.Lfunc_begin2         #     jumps to .Ltmp184
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp176-.Lfunc_begin2         # >> Call Site 16 <<
	.uleb128 .Ltmp177-.Ltmp176              #   Call between .Ltmp176 and .Ltmp177
	.uleb128 .Ltmp178-.Lfunc_begin2         #     jumps to .Ltmp178
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp170-.Lfunc_begin2         # >> Call Site 17 <<
	.uleb128 .Ltmp171-.Ltmp170              #   Call between .Ltmp170 and .Ltmp171
	.uleb128 .Ltmp172-.Lfunc_begin2         #     jumps to .Ltmp172
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp157-.Lfunc_begin2         # >> Call Site 18 <<
	.uleb128 .Ltmp158-.Ltmp157              #   Call between .Ltmp157 and .Ltmp158
	.uleb128 .Ltmp159-.Lfunc_begin2         #     jumps to .Ltmp159
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp151-.Lfunc_begin2         # >> Call Site 19 <<
	.uleb128 .Ltmp152-.Ltmp151              #   Call between .Ltmp151 and .Ltmp152
	.uleb128 .Ltmp153-.Lfunc_begin2         #     jumps to .Ltmp153
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp145-.Lfunc_begin2         # >> Call Site 20 <<
	.uleb128 .Ltmp146-.Ltmp145              #   Call between .Ltmp145 and .Ltmp146
	.uleb128 .Ltmp147-.Lfunc_begin2         #     jumps to .Ltmp147
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp191-.Lfunc_begin2         # >> Call Site 21 <<
	.uleb128 .Ltmp192-.Ltmp191              #   Call between .Ltmp191 and .Ltmp192
	.uleb128 .Ltmp193-.Lfunc_begin2         #     jumps to .Ltmp193
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp197-.Lfunc_begin2         # >> Call Site 22 <<
	.uleb128 .Ltmp198-.Ltmp197              #   Call between .Ltmp197 and .Ltmp198
	.uleb128 .Ltmp199-.Lfunc_begin2         #     jumps to .Ltmp199
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp210-.Lfunc_begin2         # >> Call Site 23 <<
	.uleb128 .Ltmp211-.Ltmp210              #   Call between .Ltmp210 and .Ltmp211
	.uleb128 .Ltmp212-.Lfunc_begin2         #     jumps to .Ltmp212
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp211-.Lfunc_begin2         # >> Call Site 24 <<
	.uleb128 .Ltmp207-.Ltmp211              #   Call between .Ltmp211 and .Ltmp207
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp207-.Lfunc_begin2         # >> Call Site 25 <<
	.uleb128 .Ltmp208-.Ltmp207              #   Call between .Ltmp207 and .Ltmp208
	.uleb128 .Ltmp209-.Lfunc_begin2         #     jumps to .Ltmp209
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp208-.Lfunc_begin2         # >> Call Site 26 <<
	.uleb128 .Lfunc_end9-.Ltmp208           #   Call between .Ltmp208 and .Lfunc_end9
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end2:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii # -- Begin function _Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.prefalign	4, .Lfunc_end10, nop
	.type	_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,@function
_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii: # @_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
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
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end10:
	.size	_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end10-_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy
.LCPI11_0:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI11_1:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI11_2:
	.quad	0x4020000000000000              # double 8
.LCPI11_3:
	.quad	0x4040000000000000              # double 32
.LCPI11_4:
	.quad	0x40d0000000000000              # double 16384
.LCPI11_5:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.prefalign	4, .Lfunc_end11, nop
	.type	_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy,@function
_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy: # @_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception3
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 112(%rsp)                  # 8-byte Spill
	movq	%r8, 216(%rsp)                  # 8-byte Spill
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	movq	%rdx, 200(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movq	%rdi, 120(%rsp)                 # 8-byte Spill
	movq	8(%rdi), %rbx
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB11_2
# %bb.1:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB11_2:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB11_4
# %bb.3:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB11_4:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB11_6
# %bb.5:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB11_6:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB11_8
# %bb.7:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB11_8:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, 240(%rsp)                 # 8-byte Spill
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB11_10
# %bb.9:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB11_10:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB11_11
# %bb.20:                               # %.preheader
	movl	$20, %ebx
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	movq	240(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB11_21:                              # =>This Inner Loop Header: Depth=1
.Ltmp213:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp214:                               # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_23
# %bb.30:                               #   in Loop: Header=BB11_21 Depth=1
.Ltmp219:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp220:                               # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_32
# %bb.36:                               #   in Loop: Header=BB11_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp225:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp226:                               # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_38
# %bb.42:                               #   in Loop: Header=BB11_21 Depth=1
.Ltmp231:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%r14, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp232:                               # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_46
# %bb.44:                               #   in Loop: Header=BB11_21 Depth=1
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movl	%eax, 4(%rsp)
	movl	%ecx, 236(%rsp)
	movl	%edx, 232(%rsp)
	movq	192(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 104(%rsp)
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	208(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	216(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp233:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp234:                               # EH_LABEL
# %bb.45:                               # %.noexc104
                                        #   in Loop: Header=BB11_21 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp235:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp236:                               # EH_LABEL
.LBB11_46:                              #   in Loop: Header=BB11_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp238:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp239:                               # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_48
# %bb.53:                               #   in Loop: Header=BB11_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp244:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp245:                               # EH_LABEL
# %bb.54:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_55
# %bb.59:                               #   in Loop: Header=BB11_21 Depth=1
	movl	$0, 128(%rsp)
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
.Ltmp250:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp251:                               # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_61
# %bb.65:                               #   in Loop: Header=BB11_21 Depth=1
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI11_0(%rip), %xmm0
	cmpq	%r12, %r15
	je	.LBB11_67
# %bb.66:                               #   in Loop: Header=BB11_21 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r13
	jmp	.LBB11_76
	.p2align	4
.LBB11_67:                              #   in Loop: Header=BB11_21 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%r12, %r15
	subq	%rbp, %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r15
	je	.LBB11_68
# %bb.70:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB11_21 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp256:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp257:                               # EH_LABEL
# %bb.71:                               # %.noexc108
                                        #   in Loop: Header=BB11_21 Depth=1
	movq	%rax, %r13
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB11_73
# %bb.72:                               #   in Loop: Header=BB11_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB11_73:                              # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB11_21 Depth=1
	testq	%rbp, %rbp
	je	.LBB11_75
# %bb.74:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB11_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB11_75:                              # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB11_21 Depth=1
	addq	%r13, %r15
	leaq	(,%r14,4), %r12
	addq	%r13, %r12
	movq	240(%rsp), %r14                 # 8-byte Reload
.LBB11_76:                              # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB11_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp259:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp260:                               # EH_LABEL
# %bb.77:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_78
# %bb.84:                               #   in Loop: Header=BB11_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp265:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp266:                               # EH_LABEL
# %bb.85:                               #   in Loop: Header=BB11_21 Depth=1
	testl	%eax, %eax
	jne	.LBB11_86
# %bb.12:                               #   in Loop: Header=BB11_21 Depth=1
	addq	$4, %r15
	movq	%r13, %rbp
	decl	%ebx
	jne	.LBB11_21
# %bb.13:
	cmpq	%r15, %r13
	je	.LBB11_16
# %bb.14:
	movq	%r15, %rax
	subq	%r13, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp271:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp272:                               # EH_LABEL
# %bb.15:                               # %.noexc
.Ltmp273:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp274:                               # EH_LABEL
.LBB11_16:                              # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_.exit
	movss	36(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movss	40(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 112(%rsp)                # 4-byte Spill
	movl	$0, 128(%rsp)
.Ltmp276:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	128(%rsp), %rdi
	movl	$20480, %ecx                    # imm = 0x5000
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp277:                               # EH_LABEL
# %bb.17:                               # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKDv2_jS2_PKjPfPyiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB11_18
# %bb.92:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addss	112(%rsp), %xmm0                # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm3
	mulsd	.LCPI11_1(%rip), %xmm3
	movq	120(%rsp), %rax                 # 8-byte Reload
	cvtdq2pd	8(%rax), %xmm0
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	mulsd	%xmm0, %xmm2
	mulsd	.LCPI11_2(%rip), %xmm2
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	mulsd	.LCPI11_3(%rip), %xmm1
	mulsd	.LCPI11_4(%rip), %xmm1
	movq	(%rax), %rdx
	movapd	%xmm3, %xmm0
	movsd	%xmm3, 224(%rsp)                # 8-byte Spill
	divsd	%xmm3, %xmm1
	divsd	.LCPI11_5(%rip), %xmm1
	movl	128(%rsp), %ecx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rdi
	movl	$3, %esi
	movb	$2, %al
	callq	printf@PLT
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	224(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addq	$248, %rsp
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
.LBB11_61:
	.cfi_def_cfa_offset 304
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp253:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp254:                               # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$286, %ecx                      # imm = 0x11E
	jmp	.LBB11_26
.LBB11_55:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp247:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp248:                               # EH_LABEL
# %bb.56:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB11_50
.LBB11_48:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp241:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp242:                               # EH_LABEL
# %bb.49:
	.cfi_escape 0x2e, 0x00
.LBB11_50:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$285, %ecx                      # imm = 0x11D
	jmp	.LBB11_26
.LBB11_38:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp228:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp229:                               # EH_LABEL
# %bb.39:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$283, %ecx                      # imm = 0x11B
	jmp	.LBB11_26
.LBB11_32:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp222:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp223:                               # EH_LABEL
# %bb.33:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB11_25
.LBB11_23:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp216:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp217:                               # EH_LABEL
# %bb.24:
	.cfi_escape 0x2e, 0x00
.LBB11_25:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$282, %ecx                      # imm = 0x11A
	jmp	.LBB11_26
.LBB11_78:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp262:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp263:                               # EH_LABEL
# %bb.79:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB11_80
.LBB11_86:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp268:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp269:                               # EH_LABEL
# %bb.87:
	.cfi_escape 0x2e, 0x00
.LBB11_80:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$287, %ecx                      # imm = 0x11F
.LBB11_26:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB11_68:
.Ltmp281:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp282:                               # EH_LABEL
# %bb.69:                               # %.noexc107
.LBB11_11:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$279, %ecx                      # imm = 0x117
	jmp	.LBB11_26
.LBB11_18:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp278:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp279:                               # EH_LABEL
# %bb.19:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$292, %ecx                      # imm = 0x124
	jmp	.LBB11_26
.LBB11_90:
.Ltmp275:                               # EH_LABEL
	jmp	.LBB11_94
.LBB11_91:                              # %.thread
.Ltmp280:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_96
.LBB11_82:                              # %.loopexit.split-lp163
.Ltmp283:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_81:                              # %.loopexit162
.Ltmp258:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_89:                              # %.loopexit.split-lp173
.Ltmp270:                               # EH_LABEL
	jmp	.LBB11_94
.LBB11_83:                              # %.loopexit.split-lp168
.Ltmp264:                               # EH_LABEL
	jmp	.LBB11_94
.LBB11_29:                              # %.loopexit.split-lp
.Ltmp218:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_35:                              # %.loopexit.split-lp138
.Ltmp224:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_41:                              # %.loopexit.split-lp143
.Ltmp230:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_52:                              # %.loopexit.split-lp148
.Ltmp243:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_58:                              # %.loopexit.split-lp153
.Ltmp249:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_64:                              # %.loopexit.split-lp158
.Ltmp255:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_88:                              # %.loopexit172
.Ltmp267:                               # EH_LABEL
	jmp	.LBB11_94
.LBB11_93:                              # %.loopexit167
.Ltmp261:                               # EH_LABEL
.LBB11_94:
	movq	%rax, %rbx
	jmp	.LBB11_95
.LBB11_27:                              # %.loopexit
.Ltmp215:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_34:                              # %.loopexit137
.Ltmp221:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_40:                              # %.loopexit142
.Ltmp227:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_51:                              # %.loopexit147
.Ltmp240:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_57:                              # %.loopexit152
.Ltmp246:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_63:                              # %.loopexit157
.Ltmp252:                               # EH_LABEL
	jmp	.LBB11_28
.LBB11_98:
.Ltmp237:                               # EH_LABEL
.LBB11_28:
	movq	%rax, %rbx
	movq	%rbp, %r13
.LBB11_95:
	testq	%r13, %r13
	je	.LBB11_97
.LBB11_96:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB11_97:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit111
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end11:
	.size	_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy, .Lfunc_end11-_ZL9run_shapeILi3EEdRK5ShapePKDv2_jS5_PKjPfPy
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table11:
.Lexception3:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Lfunc_begin3-.Lfunc_begin3    # >> Call Site 1 <<
	.uleb128 .Ltmp213-.Lfunc_begin3         #   Call between .Lfunc_begin3 and .Ltmp213
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp213-.Lfunc_begin3         # >> Call Site 2 <<
	.uleb128 .Ltmp214-.Ltmp213              #   Call between .Ltmp213 and .Ltmp214
	.uleb128 .Ltmp215-.Lfunc_begin3         #     jumps to .Ltmp215
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp219-.Lfunc_begin3         # >> Call Site 3 <<
	.uleb128 .Ltmp220-.Ltmp219              #   Call between .Ltmp219 and .Ltmp220
	.uleb128 .Ltmp221-.Lfunc_begin3         #     jumps to .Ltmp221
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp225-.Lfunc_begin3         # >> Call Site 4 <<
	.uleb128 .Ltmp226-.Ltmp225              #   Call between .Ltmp225 and .Ltmp226
	.uleb128 .Ltmp227-.Lfunc_begin3         #     jumps to .Ltmp227
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp231-.Lfunc_begin3         # >> Call Site 5 <<
	.uleb128 .Ltmp236-.Ltmp231              #   Call between .Ltmp231 and .Ltmp236
	.uleb128 .Ltmp237-.Lfunc_begin3         #     jumps to .Ltmp237
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp238-.Lfunc_begin3         # >> Call Site 6 <<
	.uleb128 .Ltmp239-.Ltmp238              #   Call between .Ltmp238 and .Ltmp239
	.uleb128 .Ltmp240-.Lfunc_begin3         #     jumps to .Ltmp240
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp244-.Lfunc_begin3         # >> Call Site 7 <<
	.uleb128 .Ltmp245-.Ltmp244              #   Call between .Ltmp244 and .Ltmp245
	.uleb128 .Ltmp246-.Lfunc_begin3         #     jumps to .Ltmp246
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp250-.Lfunc_begin3         # >> Call Site 8 <<
	.uleb128 .Ltmp251-.Ltmp250              #   Call between .Ltmp250 and .Ltmp251
	.uleb128 .Ltmp252-.Lfunc_begin3         #     jumps to .Ltmp252
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp256-.Lfunc_begin3         # >> Call Site 9 <<
	.uleb128 .Ltmp257-.Ltmp256              #   Call between .Ltmp256 and .Ltmp257
	.uleb128 .Ltmp258-.Lfunc_begin3         #     jumps to .Ltmp258
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp257-.Lfunc_begin3         # >> Call Site 10 <<
	.uleb128 .Ltmp259-.Ltmp257              #   Call between .Ltmp257 and .Ltmp259
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp259-.Lfunc_begin3         # >> Call Site 11 <<
	.uleb128 .Ltmp260-.Ltmp259              #   Call between .Ltmp259 and .Ltmp260
	.uleb128 .Ltmp261-.Lfunc_begin3         #     jumps to .Ltmp261
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp265-.Lfunc_begin3         # >> Call Site 12 <<
	.uleb128 .Ltmp266-.Ltmp265              #   Call between .Ltmp265 and .Ltmp266
	.uleb128 .Ltmp267-.Lfunc_begin3         #     jumps to .Ltmp267
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp271-.Lfunc_begin3         # >> Call Site 13 <<
	.uleb128 .Ltmp274-.Ltmp271              #   Call between .Ltmp271 and .Ltmp274
	.uleb128 .Ltmp275-.Lfunc_begin3         #     jumps to .Ltmp275
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp276-.Lfunc_begin3         # >> Call Site 14 <<
	.uleb128 .Ltmp277-.Ltmp276              #   Call between .Ltmp276 and .Ltmp277
	.uleb128 .Ltmp280-.Lfunc_begin3         #     jumps to .Ltmp280
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp253-.Lfunc_begin3         # >> Call Site 15 <<
	.uleb128 .Ltmp254-.Ltmp253              #   Call between .Ltmp253 and .Ltmp254
	.uleb128 .Ltmp255-.Lfunc_begin3         #     jumps to .Ltmp255
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp247-.Lfunc_begin3         # >> Call Site 16 <<
	.uleb128 .Ltmp248-.Ltmp247              #   Call between .Ltmp247 and .Ltmp248
	.uleb128 .Ltmp249-.Lfunc_begin3         #     jumps to .Ltmp249
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp241-.Lfunc_begin3         # >> Call Site 17 <<
	.uleb128 .Ltmp242-.Ltmp241              #   Call between .Ltmp241 and .Ltmp242
	.uleb128 .Ltmp243-.Lfunc_begin3         #     jumps to .Ltmp243
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp228-.Lfunc_begin3         # >> Call Site 18 <<
	.uleb128 .Ltmp229-.Ltmp228              #   Call between .Ltmp228 and .Ltmp229
	.uleb128 .Ltmp230-.Lfunc_begin3         #     jumps to .Ltmp230
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp222-.Lfunc_begin3         # >> Call Site 19 <<
	.uleb128 .Ltmp223-.Ltmp222              #   Call between .Ltmp222 and .Ltmp223
	.uleb128 .Ltmp224-.Lfunc_begin3         #     jumps to .Ltmp224
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp216-.Lfunc_begin3         # >> Call Site 20 <<
	.uleb128 .Ltmp217-.Ltmp216              #   Call between .Ltmp216 and .Ltmp217
	.uleb128 .Ltmp218-.Lfunc_begin3         #     jumps to .Ltmp218
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp262-.Lfunc_begin3         # >> Call Site 21 <<
	.uleb128 .Ltmp263-.Ltmp262              #   Call between .Ltmp262 and .Ltmp263
	.uleb128 .Ltmp264-.Lfunc_begin3         #     jumps to .Ltmp264
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp268-.Lfunc_begin3         # >> Call Site 22 <<
	.uleb128 .Ltmp269-.Ltmp268              #   Call between .Ltmp268 and .Ltmp269
	.uleb128 .Ltmp270-.Lfunc_begin3         #     jumps to .Ltmp270
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp281-.Lfunc_begin3         # >> Call Site 23 <<
	.uleb128 .Ltmp282-.Ltmp281              #   Call between .Ltmp281 and .Ltmp282
	.uleb128 .Ltmp283-.Lfunc_begin3         #     jumps to .Ltmp283
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp282-.Lfunc_begin3         # >> Call Site 24 <<
	.uleb128 .Ltmp278-.Ltmp282              #   Call between .Ltmp282 and .Ltmp278
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp278-.Lfunc_begin3         # >> Call Site 25 <<
	.uleb128 .Ltmp279-.Ltmp278              #   Call between .Ltmp278 and .Ltmp279
	.uleb128 .Ltmp280-.Lfunc_begin3         #     jumps to .Ltmp280
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp279-.Lfunc_begin3         # >> Call Site 26 <<
	.uleb128 .Lfunc_end11-.Ltmp279          #   Call between .Ltmp279 and .Lfunc_end11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end3:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii # -- Begin function _Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.prefalign	4, .Lfunc_end12, nop
	.type	_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,@function
_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii: # @_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
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
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end12:
	.size	_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end12-_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy
.LCPI13_0:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI13_1:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI13_2:
	.quad	0x4020000000000000              # double 8
.LCPI13_3:
	.quad	0x4040000000000000              # double 32
.LCPI13_4:
	.quad	0x40d0000000000000              # double 16384
.LCPI13_5:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.prefalign	4, .Lfunc_end13, nop
	.type	_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy,@function
_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy: # @_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception4
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 112(%rsp)                  # 8-byte Spill
	movq	%r8, 216(%rsp)                  # 8-byte Spill
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	movq	%rdx, 200(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movq	%rdi, 120(%rsp)                 # 8-byte Spill
	movq	8(%rdi), %rbx
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB13_2
# %bb.1:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB13_2:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB13_4
# %bb.3:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB13_4:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB13_6
# %bb.5:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB13_6:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB13_8
# %bb.7:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB13_8:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, 240(%rsp)                 # 8-byte Spill
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB13_10
# %bb.9:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB13_10:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB13_11
# %bb.20:                               # %.preheader
	movl	$20, %ebx
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	movq	240(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB13_21:                              # =>This Inner Loop Header: Depth=1
.Ltmp284:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp285:                               # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_23
# %bb.30:                               #   in Loop: Header=BB13_21 Depth=1
.Ltmp290:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp291:                               # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_32
# %bb.36:                               #   in Loop: Header=BB13_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp296:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp297:                               # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_38
# %bb.42:                               #   in Loop: Header=BB13_21 Depth=1
.Ltmp302:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%r14, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp303:                               # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_46
# %bb.44:                               #   in Loop: Header=BB13_21 Depth=1
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movl	%eax, 4(%rsp)
	movl	%ecx, 236(%rsp)
	movl	%edx, 232(%rsp)
	movq	192(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 104(%rsp)
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	208(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	216(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp304:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp305:                               # EH_LABEL
# %bb.45:                               # %.noexc104
                                        #   in Loop: Header=BB13_21 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp306:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp307:                               # EH_LABEL
.LBB13_46:                              #   in Loop: Header=BB13_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp309:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp310:                               # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_48
# %bb.53:                               #   in Loop: Header=BB13_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp315:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp316:                               # EH_LABEL
# %bb.54:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_55
# %bb.59:                               #   in Loop: Header=BB13_21 Depth=1
	movl	$0, 128(%rsp)
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
.Ltmp321:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp322:                               # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_61
# %bb.65:                               #   in Loop: Header=BB13_21 Depth=1
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI13_0(%rip), %xmm0
	cmpq	%r12, %r15
	je	.LBB13_67
# %bb.66:                               #   in Loop: Header=BB13_21 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r13
	jmp	.LBB13_76
	.p2align	4
.LBB13_67:                              #   in Loop: Header=BB13_21 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%r12, %r15
	subq	%rbp, %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r15
	je	.LBB13_68
# %bb.70:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB13_21 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp327:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp328:                               # EH_LABEL
# %bb.71:                               # %.noexc108
                                        #   in Loop: Header=BB13_21 Depth=1
	movq	%rax, %r13
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB13_73
# %bb.72:                               #   in Loop: Header=BB13_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB13_73:                              # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB13_21 Depth=1
	testq	%rbp, %rbp
	je	.LBB13_75
# %bb.74:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB13_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB13_75:                              # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB13_21 Depth=1
	addq	%r13, %r15
	leaq	(,%r14,4), %r12
	addq	%r13, %r12
	movq	240(%rsp), %r14                 # 8-byte Reload
.LBB13_76:                              # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB13_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp330:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp331:                               # EH_LABEL
# %bb.77:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_78
# %bb.84:                               #   in Loop: Header=BB13_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp336:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp337:                               # EH_LABEL
# %bb.85:                               #   in Loop: Header=BB13_21 Depth=1
	testl	%eax, %eax
	jne	.LBB13_86
# %bb.12:                               #   in Loop: Header=BB13_21 Depth=1
	addq	$4, %r15
	movq	%r13, %rbp
	decl	%ebx
	jne	.LBB13_21
# %bb.13:
	cmpq	%r15, %r13
	je	.LBB13_16
# %bb.14:
	movq	%r15, %rax
	subq	%r13, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp342:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp343:                               # EH_LABEL
# %bb.15:                               # %.noexc
.Ltmp344:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp345:                               # EH_LABEL
.LBB13_16:                              # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_.exit
	movss	36(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movss	40(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 112(%rsp)                # 4-byte Spill
	movl	$0, 128(%rsp)
.Ltmp347:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	128(%rsp), %rdi
	movl	$20480, %ecx                    # imm = 0x5000
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp348:                               # EH_LABEL
# %bb.17:                               # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKDv2_jS2_PKjPfPyiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB13_18
# %bb.92:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addss	112(%rsp), %xmm0                # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm3
	mulsd	.LCPI13_1(%rip), %xmm3
	movq	120(%rsp), %rax                 # 8-byte Reload
	cvtdq2pd	8(%rax), %xmm0
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	mulsd	%xmm0, %xmm2
	mulsd	.LCPI13_2(%rip), %xmm2
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	mulsd	.LCPI13_3(%rip), %xmm1
	mulsd	.LCPI13_4(%rip), %xmm1
	movq	(%rax), %rdx
	movapd	%xmm3, %xmm0
	movsd	%xmm3, 224(%rsp)                # 8-byte Spill
	divsd	%xmm3, %xmm1
	divsd	.LCPI13_5(%rip), %xmm1
	movl	128(%rsp), %ecx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rdi
	movl	$4, %esi
	movb	$2, %al
	callq	printf@PLT
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	224(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addq	$248, %rsp
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
.LBB13_61:
	.cfi_def_cfa_offset 304
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp324:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp325:                               # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$286, %ecx                      # imm = 0x11E
	jmp	.LBB13_26
.LBB13_55:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp318:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp319:                               # EH_LABEL
# %bb.56:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB13_50
.LBB13_48:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp312:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp313:                               # EH_LABEL
# %bb.49:
	.cfi_escape 0x2e, 0x00
.LBB13_50:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$285, %ecx                      # imm = 0x11D
	jmp	.LBB13_26
.LBB13_38:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp299:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp300:                               # EH_LABEL
# %bb.39:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$283, %ecx                      # imm = 0x11B
	jmp	.LBB13_26
.LBB13_32:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp293:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp294:                               # EH_LABEL
# %bb.33:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB13_25
.LBB13_23:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp287:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp288:                               # EH_LABEL
# %bb.24:
	.cfi_escape 0x2e, 0x00
.LBB13_25:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$282, %ecx                      # imm = 0x11A
	jmp	.LBB13_26
.LBB13_78:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp333:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp334:                               # EH_LABEL
# %bb.79:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB13_80
.LBB13_86:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp339:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp340:                               # EH_LABEL
# %bb.87:
	.cfi_escape 0x2e, 0x00
.LBB13_80:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$287, %ecx                      # imm = 0x11F
.LBB13_26:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB13_68:
.Ltmp352:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp353:                               # EH_LABEL
# %bb.69:                               # %.noexc107
.LBB13_11:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$279, %ecx                      # imm = 0x117
	jmp	.LBB13_26
.LBB13_18:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp349:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp350:                               # EH_LABEL
# %bb.19:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$292, %ecx                      # imm = 0x124
	jmp	.LBB13_26
.LBB13_90:
.Ltmp346:                               # EH_LABEL
	jmp	.LBB13_94
.LBB13_91:                              # %.thread
.Ltmp351:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB13_96
.LBB13_82:                              # %.loopexit.split-lp163
.Ltmp354:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_81:                              # %.loopexit162
.Ltmp329:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_89:                              # %.loopexit.split-lp173
.Ltmp341:                               # EH_LABEL
	jmp	.LBB13_94
.LBB13_83:                              # %.loopexit.split-lp168
.Ltmp335:                               # EH_LABEL
	jmp	.LBB13_94
.LBB13_29:                              # %.loopexit.split-lp
.Ltmp289:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_35:                              # %.loopexit.split-lp138
.Ltmp295:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_41:                              # %.loopexit.split-lp143
.Ltmp301:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_52:                              # %.loopexit.split-lp148
.Ltmp314:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_58:                              # %.loopexit.split-lp153
.Ltmp320:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_64:                              # %.loopexit.split-lp158
.Ltmp326:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_88:                              # %.loopexit172
.Ltmp338:                               # EH_LABEL
	jmp	.LBB13_94
.LBB13_93:                              # %.loopexit167
.Ltmp332:                               # EH_LABEL
.LBB13_94:
	movq	%rax, %rbx
	jmp	.LBB13_95
.LBB13_27:                              # %.loopexit
.Ltmp286:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_34:                              # %.loopexit137
.Ltmp292:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_40:                              # %.loopexit142
.Ltmp298:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_51:                              # %.loopexit147
.Ltmp311:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_57:                              # %.loopexit152
.Ltmp317:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_63:                              # %.loopexit157
.Ltmp323:                               # EH_LABEL
	jmp	.LBB13_28
.LBB13_98:
.Ltmp308:                               # EH_LABEL
.LBB13_28:
	movq	%rax, %rbx
	movq	%rbp, %r13
.LBB13_95:
	testq	%r13, %r13
	je	.LBB13_97
.LBB13_96:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB13_97:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit111
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end13:
	.size	_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy, .Lfunc_end13-_ZL9run_shapeILi4EEdRK5ShapePKDv2_jS5_PKjPfPy
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table13:
.Lexception4:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Lfunc_begin4-.Lfunc_begin4    # >> Call Site 1 <<
	.uleb128 .Ltmp284-.Lfunc_begin4         #   Call between .Lfunc_begin4 and .Ltmp284
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp284-.Lfunc_begin4         # >> Call Site 2 <<
	.uleb128 .Ltmp285-.Ltmp284              #   Call between .Ltmp284 and .Ltmp285
	.uleb128 .Ltmp286-.Lfunc_begin4         #     jumps to .Ltmp286
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp290-.Lfunc_begin4         # >> Call Site 3 <<
	.uleb128 .Ltmp291-.Ltmp290              #   Call between .Ltmp290 and .Ltmp291
	.uleb128 .Ltmp292-.Lfunc_begin4         #     jumps to .Ltmp292
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp296-.Lfunc_begin4         # >> Call Site 4 <<
	.uleb128 .Ltmp297-.Ltmp296              #   Call between .Ltmp296 and .Ltmp297
	.uleb128 .Ltmp298-.Lfunc_begin4         #     jumps to .Ltmp298
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp302-.Lfunc_begin4         # >> Call Site 5 <<
	.uleb128 .Ltmp307-.Ltmp302              #   Call between .Ltmp302 and .Ltmp307
	.uleb128 .Ltmp308-.Lfunc_begin4         #     jumps to .Ltmp308
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp309-.Lfunc_begin4         # >> Call Site 6 <<
	.uleb128 .Ltmp310-.Ltmp309              #   Call between .Ltmp309 and .Ltmp310
	.uleb128 .Ltmp311-.Lfunc_begin4         #     jumps to .Ltmp311
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp315-.Lfunc_begin4         # >> Call Site 7 <<
	.uleb128 .Ltmp316-.Ltmp315              #   Call between .Ltmp315 and .Ltmp316
	.uleb128 .Ltmp317-.Lfunc_begin4         #     jumps to .Ltmp317
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp321-.Lfunc_begin4         # >> Call Site 8 <<
	.uleb128 .Ltmp322-.Ltmp321              #   Call between .Ltmp321 and .Ltmp322
	.uleb128 .Ltmp323-.Lfunc_begin4         #     jumps to .Ltmp323
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp327-.Lfunc_begin4         # >> Call Site 9 <<
	.uleb128 .Ltmp328-.Ltmp327              #   Call between .Ltmp327 and .Ltmp328
	.uleb128 .Ltmp329-.Lfunc_begin4         #     jumps to .Ltmp329
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp328-.Lfunc_begin4         # >> Call Site 10 <<
	.uleb128 .Ltmp330-.Ltmp328              #   Call between .Ltmp328 and .Ltmp330
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp330-.Lfunc_begin4         # >> Call Site 11 <<
	.uleb128 .Ltmp331-.Ltmp330              #   Call between .Ltmp330 and .Ltmp331
	.uleb128 .Ltmp332-.Lfunc_begin4         #     jumps to .Ltmp332
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp336-.Lfunc_begin4         # >> Call Site 12 <<
	.uleb128 .Ltmp337-.Ltmp336              #   Call between .Ltmp336 and .Ltmp337
	.uleb128 .Ltmp338-.Lfunc_begin4         #     jumps to .Ltmp338
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp342-.Lfunc_begin4         # >> Call Site 13 <<
	.uleb128 .Ltmp345-.Ltmp342              #   Call between .Ltmp342 and .Ltmp345
	.uleb128 .Ltmp346-.Lfunc_begin4         #     jumps to .Ltmp346
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp347-.Lfunc_begin4         # >> Call Site 14 <<
	.uleb128 .Ltmp348-.Ltmp347              #   Call between .Ltmp347 and .Ltmp348
	.uleb128 .Ltmp351-.Lfunc_begin4         #     jumps to .Ltmp351
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp324-.Lfunc_begin4         # >> Call Site 15 <<
	.uleb128 .Ltmp325-.Ltmp324              #   Call between .Ltmp324 and .Ltmp325
	.uleb128 .Ltmp326-.Lfunc_begin4         #     jumps to .Ltmp326
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp318-.Lfunc_begin4         # >> Call Site 16 <<
	.uleb128 .Ltmp319-.Ltmp318              #   Call between .Ltmp318 and .Ltmp319
	.uleb128 .Ltmp320-.Lfunc_begin4         #     jumps to .Ltmp320
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp312-.Lfunc_begin4         # >> Call Site 17 <<
	.uleb128 .Ltmp313-.Ltmp312              #   Call between .Ltmp312 and .Ltmp313
	.uleb128 .Ltmp314-.Lfunc_begin4         #     jumps to .Ltmp314
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp299-.Lfunc_begin4         # >> Call Site 18 <<
	.uleb128 .Ltmp300-.Ltmp299              #   Call between .Ltmp299 and .Ltmp300
	.uleb128 .Ltmp301-.Lfunc_begin4         #     jumps to .Ltmp301
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp293-.Lfunc_begin4         # >> Call Site 19 <<
	.uleb128 .Ltmp294-.Ltmp293              #   Call between .Ltmp293 and .Ltmp294
	.uleb128 .Ltmp295-.Lfunc_begin4         #     jumps to .Ltmp295
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp287-.Lfunc_begin4         # >> Call Site 20 <<
	.uleb128 .Ltmp288-.Ltmp287              #   Call between .Ltmp287 and .Ltmp288
	.uleb128 .Ltmp289-.Lfunc_begin4         #     jumps to .Ltmp289
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp333-.Lfunc_begin4         # >> Call Site 21 <<
	.uleb128 .Ltmp334-.Ltmp333              #   Call between .Ltmp333 and .Ltmp334
	.uleb128 .Ltmp335-.Lfunc_begin4         #     jumps to .Ltmp335
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp339-.Lfunc_begin4         # >> Call Site 22 <<
	.uleb128 .Ltmp340-.Ltmp339              #   Call between .Ltmp339 and .Ltmp340
	.uleb128 .Ltmp341-.Lfunc_begin4         #     jumps to .Ltmp341
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp352-.Lfunc_begin4         # >> Call Site 23 <<
	.uleb128 .Ltmp353-.Ltmp352              #   Call between .Ltmp352 and .Ltmp353
	.uleb128 .Ltmp354-.Lfunc_begin4         #     jumps to .Ltmp354
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp353-.Lfunc_begin4         # >> Call Site 24 <<
	.uleb128 .Ltmp349-.Ltmp353              #   Call between .Ltmp353 and .Ltmp349
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp349-.Lfunc_begin4         # >> Call Site 25 <<
	.uleb128 .Ltmp350-.Ltmp349              #   Call between .Ltmp349 and .Ltmp350
	.uleb128 .Ltmp351-.Lfunc_begin4         #     jumps to .Ltmp351
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp350-.Lfunc_begin4         # >> Call Site 26 <<
	.uleb128 .Lfunc_end13-.Ltmp350          #   Call between .Ltmp350 and .Lfunc_end13
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end4:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii # -- Begin function _Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.prefalign	4, .Lfunc_end14, nop
	.type	_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,@function
_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii: # @_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
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
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end14:
	.size	_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end14-_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy
.LCPI15_0:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI15_1:
	.quad	0x3fe0000000000000              # double 0.5
.LCPI15_2:
	.quad	0x4020000000000000              # double 8
.LCPI15_3:
	.quad	0x4040000000000000              # double 32
.LCPI15_4:
	.quad	0x40d0000000000000              # double 16384
.LCPI15_5:
	.quad	0x412e848000000000              # double 1.0E+6
	.text
	.prefalign	4, .Lfunc_end15, nop
	.type	_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy,@function
_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy: # @_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy
.Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception5
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 304
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 112(%rsp)                  # 8-byte Spill
	movq	%r8, 216(%rsp)                  # 8-byte Spill
	movq	%rcx, 208(%rsp)                 # 8-byte Spill
	movq	%rdx, 200(%rsp)                 # 8-byte Spill
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movq	%rdi, 120(%rsp)                 # 8-byte Spill
	movq	8(%rdi), %rbx
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB15_2
# %bb.1:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB15_2:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB15_4
# %bb.3:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB15_4:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB15_6
# %bb.5:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB15_6:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB15_8
# %bb.7:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB15_8:
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%rbx, 240(%rsp)                 # 8-byte Spill
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB15_10
# %bb.9:
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movq	192(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 104(%rsp)
	movq	200(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 96(%rsp)
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 88(%rsp)
	movq	216(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 80(%rsp)
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movq	%rsi, 72(%rsp)
	movl	%eax, 16(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%edx, 4(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 184(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB15_10:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB15_11
# %bb.20:                               # %.preheader
	movl	$20, %ebx
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	movq	240(%rsp), %r14                 # 8-byte Reload
	.p2align	4
.LBB15_21:                              # =>This Inner Loop Header: Depth=1
.Ltmp355:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp356:                               # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_23
# %bb.30:                               #   in Loop: Header=BB15_21 Depth=1
.Ltmp361:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp362:                               # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_32
# %bb.36:                               #   in Loop: Header=BB15_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp367:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp368:                               # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_38
# %bb.42:                               #   in Loop: Header=BB15_21 Depth=1
.Ltmp373:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$20480, %r8d                    # imm = 0x5000
	movq	%r14, %rdi
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp374:                               # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_46
# %bb.44:                               #   in Loop: Header=BB15_21 Depth=1
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movl	16(%rdx), %eax
	movl	8(%rdx), %ecx
	movl	12(%rdx), %edx
	movl	%eax, 4(%rsp)
	movl	%ecx, 236(%rsp)
	movl	%edx, 232(%rsp)
	movq	192(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 104(%rsp)
	movq	200(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 96(%rsp)
	movq	208(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)
	movq	216(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 184(%rsp)
.Ltmp375:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp376:                               # EH_LABEL
# %bb.45:                               # %.noexc104
                                        #   in Loop: Header=BB15_21 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp377:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	128(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp378:                               # EH_LABEL
.LBB15_46:                              #   in Loop: Header=BB15_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp380:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp381:                               # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_48
# %bb.53:                               #   in Loop: Header=BB15_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp386:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp387:                               # EH_LABEL
# %bb.54:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_55
# %bb.59:                               #   in Loop: Header=BB15_21 Depth=1
	movl	$0, 128(%rsp)
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdx
.Ltmp392:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp393:                               # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_61
# %bb.65:                               #   in Loop: Header=BB15_21 Depth=1
	movss	128(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI15_0(%rip), %xmm0
	cmpq	%r12, %r15
	je	.LBB15_67
# %bb.66:                               #   in Loop: Header=BB15_21 Depth=1
	movss	%xmm0, (%r15)
	movq	%rbp, %r13
	jmp	.LBB15_76
	.p2align	4
.LBB15_67:                              #   in Loop: Header=BB15_21 Depth=1
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movq	%r12, %r15
	subq	%rbp, %r15
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r15
	je	.LBB15_68
# %bb.70:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB15_21 Depth=1
	movq	%r15, %r14
	sarq	$2, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	cmovaeq	%rax, %r14
	leaq	(,%r14,4), %rdi
.Ltmp398:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp399:                               # EH_LABEL
# %bb.71:                               # %.noexc108
                                        #   in Loop: Header=BB15_21 Depth=1
	movq	%rax, %r13
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	jle	.LBB15_73
# %bb.72:                               #   in Loop: Header=BB15_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB15_73:                              # %_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB15_21 Depth=1
	testq	%rbp, %rbp
	je	.LBB15_75
# %bb.74:                               # %_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm.exit.i.i.i.i
                                        #   in Loop: Header=BB15_21 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB15_75:                              # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB15_21 Depth=1
	addq	%r13, %r15
	leaq	(,%r14,4), %r12
	addq	%r13, %r12
	movq	240(%rsp), %r14                 # 8-byte Reload
.LBB15_76:                              # %_ZNSt6vectorIfSaIfEE9push_backEOf.exit
                                        #   in Loop: Header=BB15_21 Depth=1
	movq	16(%rsp), %rdi
.Ltmp401:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp402:                               # EH_LABEL
# %bb.77:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_78
# %bb.84:                               #   in Loop: Header=BB15_21 Depth=1
	movq	8(%rsp), %rdi
.Ltmp407:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp408:                               # EH_LABEL
# %bb.85:                               #   in Loop: Header=BB15_21 Depth=1
	testl	%eax, %eax
	jne	.LBB15_86
# %bb.12:                               #   in Loop: Header=BB15_21 Depth=1
	addq	$4, %r15
	movq	%r13, %rbp
	decl	%ebx
	jne	.LBB15_21
# %bb.13:
	cmpq	%r15, %r13
	je	.LBB15_16
# %bb.14:
	movq	%r15, %rax
	subq	%r13, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp413:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp414:                               # EH_LABEL
# %bb.15:                               # %.noexc
.Ltmp415:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp416:                               # EH_LABEL
.LBB15_16:                              # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_.exit
	movss	36(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 224(%rsp)                # 4-byte Spill
	movss	40(%r13), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 112(%rsp)                # 4-byte Spill
	movl	$0, 128(%rsp)
.Ltmp418:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	128(%rsp), %rdi
	movl	$20480, %ecx                    # imm = 0x5000
	movl	$256, %edx                      # imm = 0x100
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
.Ltmp419:                               # EH_LABEL
# %bb.17:                               # %_Z44hipOccupancyMaxActiveBlocksPerMultiprocessorIPFvPKDv2_jS2_PKjPfPyiiiEE10hipError_tPiT_im.exit
	testl	%eax, %eax
	jne	.LBB15_18
# %bb.92:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movss	224(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	addss	112(%rsp), %xmm0                # 4-byte Folded Reload
	cvtss2sd	%xmm0, %xmm3
	mulsd	.LCPI15_1(%rip), %xmm3
	movq	120(%rsp), %rax                 # 8-byte Reload
	cvtdq2pd	8(%rax), %xmm0
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	mulsd	%xmm0, %xmm2
	mulsd	.LCPI15_2(%rip), %xmm2
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	mulsd	.LCPI15_3(%rip), %xmm1
	mulsd	.LCPI15_4(%rip), %xmm1
	movq	(%rax), %rdx
	movapd	%xmm3, %xmm0
	movsd	%xmm3, 224(%rsp)                # 8-byte Spill
	divsd	%xmm3, %xmm1
	divsd	.LCPI15_5(%rip), %xmm1
	movl	128(%rsp), %ecx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rdi
	movl	$5, %esi
	movb	$2, %al
	callq	printf@PLT
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	224(%rsp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	addq	$248, %rsp
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
.LBB15_61:
	.cfi_def_cfa_offset 304
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp395:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp396:                               # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$286, %ecx                      # imm = 0x11E
	jmp	.LBB15_26
.LBB15_55:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp389:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp390:                               # EH_LABEL
# %bb.56:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB15_50
.LBB15_48:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp383:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp384:                               # EH_LABEL
# %bb.49:
	.cfi_escape 0x2e, 0x00
.LBB15_50:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$285, %ecx                      # imm = 0x11D
	jmp	.LBB15_26
.LBB15_38:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp370:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp371:                               # EH_LABEL
# %bb.39:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$283, %ecx                      # imm = 0x11B
	jmp	.LBB15_26
.LBB15_32:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp364:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp365:                               # EH_LABEL
# %bb.33:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB15_25
.LBB15_23:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp358:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp359:                               # EH_LABEL
# %bb.24:
	.cfi_escape 0x2e, 0x00
.LBB15_25:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$282, %ecx                      # imm = 0x11A
	jmp	.LBB15_26
.LBB15_78:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp404:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp405:                               # EH_LABEL
# %bb.79:
	.cfi_escape 0x2e, 0x00
	jmp	.LBB15_80
.LBB15_86:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp410:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp411:                               # EH_LABEL
# %bb.87:
	.cfi_escape 0x2e, 0x00
.LBB15_80:
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$287, %ecx                      # imm = 0x11F
.LBB15_26:
	movq	%rax, %r8
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB15_68:
.Ltmp423:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp424:                               # EH_LABEL
# %bb.69:                               # %.noexc107
.LBB15_11:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$279, %ecx                      # imm = 0x117
	jmp	.LBB15_26
.LBB15_18:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp420:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp421:                               # EH_LABEL
# %bb.19:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	movq	%rbx, %rdi
	movl	$292, %ecx                      # imm = 0x124
	jmp	.LBB15_26
.LBB15_90:
.Ltmp417:                               # EH_LABEL
	jmp	.LBB15_94
.LBB15_91:                              # %.thread
.Ltmp422:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB15_96
.LBB15_82:                              # %.loopexit.split-lp163
.Ltmp425:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_81:                              # %.loopexit162
.Ltmp400:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_89:                              # %.loopexit.split-lp173
.Ltmp412:                               # EH_LABEL
	jmp	.LBB15_94
.LBB15_83:                              # %.loopexit.split-lp168
.Ltmp406:                               # EH_LABEL
	jmp	.LBB15_94
.LBB15_29:                              # %.loopexit.split-lp
.Ltmp360:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_35:                              # %.loopexit.split-lp138
.Ltmp366:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_41:                              # %.loopexit.split-lp143
.Ltmp372:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_52:                              # %.loopexit.split-lp148
.Ltmp385:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_58:                              # %.loopexit.split-lp153
.Ltmp391:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_64:                              # %.loopexit.split-lp158
.Ltmp397:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_88:                              # %.loopexit172
.Ltmp409:                               # EH_LABEL
	jmp	.LBB15_94
.LBB15_93:                              # %.loopexit167
.Ltmp403:                               # EH_LABEL
.LBB15_94:
	movq	%rax, %rbx
	jmp	.LBB15_95
.LBB15_27:                              # %.loopexit
.Ltmp357:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_34:                              # %.loopexit137
.Ltmp363:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_40:                              # %.loopexit142
.Ltmp369:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_51:                              # %.loopexit147
.Ltmp382:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_57:                              # %.loopexit152
.Ltmp388:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_63:                              # %.loopexit157
.Ltmp394:                               # EH_LABEL
	jmp	.LBB15_28
.LBB15_98:
.Ltmp379:                               # EH_LABEL
.LBB15_28:
	movq	%rax, %rbx
	movq	%rbp, %r13
.LBB15_95:
	testq	%r13, %r13
	je	.LBB15_97
.LBB15_96:
	subq	%r13, %r12
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB15_97:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit111
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end15:
	.size	_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy, .Lfunc_end15-_ZL9run_shapeILi5EEdRK5ShapePKDv2_jS5_PKjPfPy
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table15:
.Lexception5:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Lfunc_begin5-.Lfunc_begin5    # >> Call Site 1 <<
	.uleb128 .Ltmp355-.Lfunc_begin5         #   Call between .Lfunc_begin5 and .Ltmp355
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp355-.Lfunc_begin5         # >> Call Site 2 <<
	.uleb128 .Ltmp356-.Ltmp355              #   Call between .Ltmp355 and .Ltmp356
	.uleb128 .Ltmp357-.Lfunc_begin5         #     jumps to .Ltmp357
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp361-.Lfunc_begin5         # >> Call Site 3 <<
	.uleb128 .Ltmp362-.Ltmp361              #   Call between .Ltmp361 and .Ltmp362
	.uleb128 .Ltmp363-.Lfunc_begin5         #     jumps to .Ltmp363
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp367-.Lfunc_begin5         # >> Call Site 4 <<
	.uleb128 .Ltmp368-.Ltmp367              #   Call between .Ltmp367 and .Ltmp368
	.uleb128 .Ltmp369-.Lfunc_begin5         #     jumps to .Ltmp369
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp373-.Lfunc_begin5         # >> Call Site 5 <<
	.uleb128 .Ltmp378-.Ltmp373              #   Call between .Ltmp373 and .Ltmp378
	.uleb128 .Ltmp379-.Lfunc_begin5         #     jumps to .Ltmp379
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp380-.Lfunc_begin5         # >> Call Site 6 <<
	.uleb128 .Ltmp381-.Ltmp380              #   Call between .Ltmp380 and .Ltmp381
	.uleb128 .Ltmp382-.Lfunc_begin5         #     jumps to .Ltmp382
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp386-.Lfunc_begin5         # >> Call Site 7 <<
	.uleb128 .Ltmp387-.Ltmp386              #   Call between .Ltmp386 and .Ltmp387
	.uleb128 .Ltmp388-.Lfunc_begin5         #     jumps to .Ltmp388
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp392-.Lfunc_begin5         # >> Call Site 8 <<
	.uleb128 .Ltmp393-.Ltmp392              #   Call between .Ltmp392 and .Ltmp393
	.uleb128 .Ltmp394-.Lfunc_begin5         #     jumps to .Ltmp394
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp398-.Lfunc_begin5         # >> Call Site 9 <<
	.uleb128 .Ltmp399-.Ltmp398              #   Call between .Ltmp398 and .Ltmp399
	.uleb128 .Ltmp400-.Lfunc_begin5         #     jumps to .Ltmp400
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp399-.Lfunc_begin5         # >> Call Site 10 <<
	.uleb128 .Ltmp401-.Ltmp399              #   Call between .Ltmp399 and .Ltmp401
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp401-.Lfunc_begin5         # >> Call Site 11 <<
	.uleb128 .Ltmp402-.Ltmp401              #   Call between .Ltmp401 and .Ltmp402
	.uleb128 .Ltmp403-.Lfunc_begin5         #     jumps to .Ltmp403
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp407-.Lfunc_begin5         # >> Call Site 12 <<
	.uleb128 .Ltmp408-.Ltmp407              #   Call between .Ltmp407 and .Ltmp408
	.uleb128 .Ltmp409-.Lfunc_begin5         #     jumps to .Ltmp409
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp413-.Lfunc_begin5         # >> Call Site 13 <<
	.uleb128 .Ltmp416-.Ltmp413              #   Call between .Ltmp413 and .Ltmp416
	.uleb128 .Ltmp417-.Lfunc_begin5         #     jumps to .Ltmp417
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp418-.Lfunc_begin5         # >> Call Site 14 <<
	.uleb128 .Ltmp419-.Ltmp418              #   Call between .Ltmp418 and .Ltmp419
	.uleb128 .Ltmp422-.Lfunc_begin5         #     jumps to .Ltmp422
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp395-.Lfunc_begin5         # >> Call Site 15 <<
	.uleb128 .Ltmp396-.Ltmp395              #   Call between .Ltmp395 and .Ltmp396
	.uleb128 .Ltmp397-.Lfunc_begin5         #     jumps to .Ltmp397
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp389-.Lfunc_begin5         # >> Call Site 16 <<
	.uleb128 .Ltmp390-.Ltmp389              #   Call between .Ltmp389 and .Ltmp390
	.uleb128 .Ltmp391-.Lfunc_begin5         #     jumps to .Ltmp391
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp383-.Lfunc_begin5         # >> Call Site 17 <<
	.uleb128 .Ltmp384-.Ltmp383              #   Call between .Ltmp383 and .Ltmp384
	.uleb128 .Ltmp385-.Lfunc_begin5         #     jumps to .Ltmp385
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp370-.Lfunc_begin5         # >> Call Site 18 <<
	.uleb128 .Ltmp371-.Ltmp370              #   Call between .Ltmp370 and .Ltmp371
	.uleb128 .Ltmp372-.Lfunc_begin5         #     jumps to .Ltmp372
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp364-.Lfunc_begin5         # >> Call Site 19 <<
	.uleb128 .Ltmp365-.Ltmp364              #   Call between .Ltmp364 and .Ltmp365
	.uleb128 .Ltmp366-.Lfunc_begin5         #     jumps to .Ltmp366
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp358-.Lfunc_begin5         # >> Call Site 20 <<
	.uleb128 .Ltmp359-.Ltmp358              #   Call between .Ltmp358 and .Ltmp359
	.uleb128 .Ltmp360-.Lfunc_begin5         #     jumps to .Ltmp360
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp404-.Lfunc_begin5         # >> Call Site 21 <<
	.uleb128 .Ltmp405-.Ltmp404              #   Call between .Ltmp404 and .Ltmp405
	.uleb128 .Ltmp406-.Lfunc_begin5         #     jumps to .Ltmp406
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp410-.Lfunc_begin5         # >> Call Site 22 <<
	.uleb128 .Ltmp411-.Ltmp410              #   Call between .Ltmp410 and .Ltmp411
	.uleb128 .Ltmp412-.Lfunc_begin5         #     jumps to .Ltmp412
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp423-.Lfunc_begin5         # >> Call Site 23 <<
	.uleb128 .Ltmp424-.Ltmp423              #   Call between .Ltmp423 and .Ltmp424
	.uleb128 .Ltmp425-.Lfunc_begin5         #     jumps to .Ltmp425
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp424-.Lfunc_begin5         # >> Call Site 24 <<
	.uleb128 .Ltmp420-.Ltmp424              #   Call between .Ltmp424 and .Ltmp420
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp420-.Lfunc_begin5         # >> Call Site 25 <<
	.uleb128 .Ltmp421-.Ltmp420              #   Call between .Ltmp420 and .Ltmp421
	.uleb128 .Ltmp422-.Lfunc_begin5         #     jumps to .Ltmp422
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp421-.Lfunc_begin5         # >> Call Site 26 <<
	.uleb128 .Lfunc_end15-.Ltmp421          #   Call between .Ltmp421 and .Lfunc_end15
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end5:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii # -- Begin function _Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.prefalign	4, .Lfunc_end16, nop
	.type	_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,@function
_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii: # @_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
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
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end16:
	.size	_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end16-_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end17, nop    # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_9a7978dd31cac309(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB17_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_9a7978dd31cac309(%rip)
.LBB17_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_5(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_6(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end17:
	.size	__hip_module_ctor, .Lfunc_end17-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end18, nop    # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_9a7978dd31cac309(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB18_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_9a7978dd31cac309(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB18_2:
	retq
.Lfunc_end18:
	.size	__hip_module_dtor, .Lfunc_end18-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"gate_up"
	.size	.L.str, 8

	.type	.L__const.main.gate,@object     # @__const.main.gate
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3, 0x0
.L__const.main.gate:
	.quad	.L.str
	.long	272                             # 0x110
	.long	64                              # 0x40
	.long	40                              # 0x28
	.zero	4
	.size	.L__const.main.gate, 24

	.type	.L.str.1,@object                # @.str.1
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.1:
	.asciz	"down"
	.size	.L.str.1, 5

	.type	.L__const.main.down,@object     # @__const.main.down
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3, 0x0
.L__const.main.down:
	.quad	.L.str.1
	.long	40                              # 0x28
	.long	64                              # 0x40
	.long	136                             # 0x88
	.zero	4
	.size	.L__const.main.down, 24

	.type	.L.str.2,@object                # @.str.2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.2:
	.asciz	"HIP %s:%d %s\n"
	.size	.L.str.2, 14

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"iu4_ladder.hip"
	.size	.L.str.3, 15

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"AGG,step=%d,median_us=%.3f,TOPS=%.3f\n"
	.size	.L.str.4, 38

	.type	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,@object # @_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.section	.data.rel.ro._Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,"awG",@progbits,_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.p2align	3, 0x0
_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii:
	.quad	_Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.size	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii, 8

	.type	.L.str.5,@object                # @.str.5
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.5:
	.asciz	"SHAPE,step=%d,name=%s,median_us=%.3f,TOPS=%.3f,occ=%d\n"
	.size	.L.str.5, 55

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"vector::_M_realloc_append"
	.size	.L.str.6, 26

	.type	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,@object # @_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.section	.data.rel.ro._Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,"awG",@progbits,_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.p2align	3, 0x0
_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii:
	.quad	_Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.size	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii, 8

	.type	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,@object # @_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.section	.data.rel.ro._Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,"awG",@progbits,_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.p2align	3, 0x0
_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii:
	.quad	_Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.size	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii, 8

	.type	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,@object # @_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.section	.data.rel.ro._Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,"awG",@progbits,_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.p2align	3, 0x0
_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii:
	.quad	_Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.size	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii, 8

	.type	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,@object # @_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.section	.data.rel.ro._Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,"awG",@progbits,_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.p2align	3, 0x0
_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii:
	.quad	_Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.size	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii, 8

	.type	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,@object # @_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.section	.data.rel.ro._Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,"awG",@progbits,_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,comdat
	.weak	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.p2align	3, 0x0
_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii:
	.quad	_Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.size	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii"
	.size	.L__unnamed_1, 42

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii"
	.size	.L__unnamed_2, 42

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii"
	.size	.L__unnamed_3, 42

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii"
	.size	.L__unnamed_4, 42

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii"
	.size	.L__unnamed_5, 42

	.type	.L__unnamed_6,@object           # @5
.L__unnamed_6:
	.asciz	"_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii"
	.size	.L__unnamed_6, 42

	.type	.L__unnamed_7,@object           # @6
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_7:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\000\200\202\000\000\000\000\000\000 \000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx1201\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000~\000\000\000\000\000\000N\000\000\000@\0008\000\t\000@\000\022\000\020\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\204!\000\000\000\000\000\000\204!\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\"\000\000\000\000\000\000\0002\000\000\000\000\000\000\0002\000\000\000\000\000\000\264U\000\000\000\000\000\000\264U\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\270w\000\000\000\000\000\000\270\227\000\000\000\000\000\000\270\227\000\000\000\000\000\000p\000\000\000\000\000\000\000H\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000(x\000\000\000\000\000\000(\250\000\000\000\000\000\000(\250\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\270w\000\000\000\000\000\000\270\227\000\000\000\000\000\000\270\227\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\270w\000\000\000\000\000\000\270\227\000\000\000\000\000\000\270\227\000\000\000\000\000\000p\000\000\000\000\000\000\000H\b\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000l\030\000\000\000\000\000\000l\030\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000U\030\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\226\336\000\023\245.args\230\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\203\247.offset0\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size4\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\331)_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii\273.private_segment_fixed_size\000\253.sgpr_count\f\261.sgpr_spill_count\000\247.symbol\331,_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countE\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\230\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\203\247.offset0\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size4\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\331)_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii\273.private_segment_fixed_size\000\253.sgpr_count\f\261.sgpr_spill_count\000\247.symbol\331,_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countL\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\230\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\203\247.offset0\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size4\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\331)_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii\273.private_segment_fixed_size\000\253.sgpr_count\f\261.sgpr_spill_count\000\247.symbol\331,_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countV\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\230\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\203\247.offset0\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size4\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\331)_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii\273.private_segment_fixed_size\000\253.sgpr_count\026\261.sgpr_spill_count\000\247.symbol\331,_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countl\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\230\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\203\247.offset0\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size4\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\331)_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii\273.private_segment_fixed_size\000\253.sgpr_count\036\261.sgpr_spill_count\000\247.symbol\331,_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\264\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\230\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\203\247.offset0\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size4\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\331)_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii\273.private_segment_fixed_size\000\253.sgpr_count \261.sgpr_spill_count\000\247.symbol\331,_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\267\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\272amdgcn-amd-amdhsa--gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000X\000\000\000\022\003\b\000\0008\000\000\000\000\000\000\f\006\000\000\000\000\000\000\257\000\000\000\022\003\b\000\000?\000\000\000\000\000\0004\b\000\000\000\000\000\000]\001\000\000\022\003\b\000\000P\000\000\000\000\000\000\254\024\000\000\000\000\000\000\001\000\000\000\022\003\b\000\0002\000\000\000\000\000\000l\005\000\000\000\000\000\000+\000\000\000\021\003\006\000@\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\202\000\000\000\021\003\006\000\200\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\331\000\000\000\021\003\006\000\300\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\006\001\000\000\022\003\b\000\000H\000\000\000\000\000\000\344\007\000\000\000\000\000\0000\001\000\000\021\003\006\000\000 \000\000\000\000\000\000@\000\000\000\000\000\000\000\207\001\000\000\021\003\006\000@ \000\000\000\000\000\000@\000\000\000\000\000\000\000\264\001\000\000\022\003\b\000\000e\000\000\000\000\000\000\264\"\000\000\000\000\000\000\336\001\000\000\021\003\006\000\200 \000\000\000\000\000\000@\000\000\000\000\000\000\000\013\002\000\000\021\000\013\000(\250\000\000\000\000\000\000\001\000\000\000\000\000\000\000\003\000\000\000\001\000\000\000\004\000\000\000\032\000\000\000\n\200\000\000\t\000\200\304\000\000\020P\000\001\000\250\000\000\000\002\000\000\000\020\000\004\000@\004\210\000 \001\000\000\000\002\000\000\000\004\000\000\000_Dbp>\273K?\001\251\036\335|\315x\241:\n'?z!\240R\2748\031f 25\016\374O\222y>g\013\215\342\037\b\254~~\204\240\357$\006*\016\000\000\000\016\000\000\000\005\000\000\000\000\000\000\000\f\000\000\000\002\000\000\000\000\000\000\000\r\000\000\000\n\000\000\000\001\000\000\000\000\000\000\000\013\000\000\000\006\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\b\000\000\000\t\000\000\000\000\000\000\000\000_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.kd\000__hip_cuid_9a7978dd31cac309\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\000\000\000\000\000\000\000\300\022\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\260\000\000\000\b\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\000\000\000\000\000\000\000\200\030\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320\000\000\000\t\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\000\000\000\000\000\000\000@\037\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\001\000\000\n\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\000\000\000\000\000\000\000\000(\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\r\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\000\000\000\000\000\000\000\300/\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\240\002\000\000\026\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\000\000\000\000\000\000\000\200D\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000`\004\000\000\026\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\020\000\000\000\000\000\000\000\001zR\000\004\004\020\001\033\000\000\000\030\000\000\000\030\000\000\000$\021\000\000l\005\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\0004\000\000\000\b\027\000\000\f\006\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000P\000\000\000\354\035\000\0004\b\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000l\000\000\000\320&\000\000\344\007\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000\210\000\000\000\264.\000\000\254\024\000\000\000\017\00406\351\002\007\020\000\000\000\034\000\000\000\244\000\000\000\230C\000\000\264\"\000\000\000\017\00406\351\002\007\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200!\000\364(\000\000\370\000\000\307\277\006\201\004\277p\000\242\277\200\002R~\377\000\2028\021\021\021\021\377\000\2048\"\"\"\"\377\000\20683333\377\000\2108DDDD)\001\020\312)\001**)\001\020\312)\001,,)\001\020\312)\001..)\001\020\312)\00180)\001\020\312)\001::)\001\020\312)\001<<)\001\020\312)\001>>)\001\020\312)\0010@)\001\020\312)\00122)\001\020\312)\00144)\001\020\312)\00166)\001\020\312)\001 8)\001\020\312)\001\"\")\001\020\312)\001$$)\001\020\312)\001&&)\001\020\312)\001\030()\001\020\312)\001\032\032)\001\020\312)\001\034\034)\001\020\312)\001\036\036)\001\020\312)\001\020 )\001\020\312)\001\022\022)\001\020\312)\001\024\024)\001\020\312)\001\026\026)\001\020\312)\001\b\030)\001\020\312)\001\n\n)\001\020\312)\001\f\f)\001\020\312)\001\016\016)\001\020\312)\001\000\020)\001\020\312)\001\002\002)\001\020\312)\001\004\004)\001\020\312)\001\006\006)\003\020~\006\201\002\204\t\000\207\277\002\201\002\212)@J\314A\207\246|9@J\314A\207\346|1@J\314A\207\306|!@J\314A\207\206|\031@J\314A\207f|\021@J\314A\207F|\t@J\314A\207&|\001@J\314A\207\006|)@J\314A\207\246|9@J\314A\207\346|1@J\314A\207\306|!@J\314A\207\206|\031@J\314A\207f|\021@J\314A\207F|\t@J\314A\207&|\001@J\314A\207\006|\002\301\002\201\t\000\207\277\002\200\006\277\334\377\241\277A\000\240\277\200\002\020~\001\000\207\277\b\001\020\312\b\001\006\007\b\001\020\312\b\001\004\005\b\001\020\312\b\001\002\003\b\001\020\312\b\001\020\001\b\001\020\312\b\001\016\017\b\001\020\312\b\001\f\r\b\001\020\312\b\001\n\013\b\001\020\312\b\001\030\t\b\001\020\312\b\001\026\027\b\001\020\312\b\001\024\025\b\001\020\312\b\001\022\023\b\001\020\312\b\001 \021\b\001\020\312\b\001\036\037\b\001\020\312\b\001\034\035\b\001\020\312\b\001\032\033\b\001\020\312\b\001(\031\b\001\020\312\b\001&'\b\001\020\312\b\001$%\b\001\020\312\b\001\"#\b\001\020\312\b\0018!\b\001\020\312\b\00167\b\001\020\312\b\00145\b\001\020\312\b\00123\b\001\020\312\b\001@1\b\001\020\312\b\001>?\b\001\020\312\b\001<=\b\001\020\312\b\001:;\b\001\020\312\b\00109\b\001\020\312\b\001./\b\001\020\312\b\001,-\b\001\020\312\b\001*+\b\003R~\207\000\2026\000\"\000\364 \000\000\370\203\000\0000\002\000\207\277\201\202\224|)UR\0029uT\002\000\000J\324\202\202\002\002\001\000J\324\203\202\002\002\t\025\022\002!EB\002\03152\002\n\000\001\325*w\002\000)\000\001\325)W\002\000\021%\"\002\001\005\002\002\002\000J\324\205\202\002\002\n\000\001\325\ny\006\0001eV\002\204\202\224|\002\000\001\325)Y\006\000\003\000J\324\206\202\002\002\004\000J\324\207\202\002\002\031\000\001\325\0317\002\000\235\377\210\277\n{\024\002\022\000\001\325+g\002\000\002[\004\002\t\000\001\325\t\027\002\000\021\000\001\325\021'\002\000\n\000\001\325\n}\n\000\022\000\001\325\022i\006\000\002\000\001\325\002]\n\000\023\000\001\325\0319\006\000\021\000\001\325\021)\006\000\n\000\001\325\n\177\016\000\022k$\002\032\000\001\325!G\002\000\002\000\001\325\002_\016\000\021+\"\002\n\000\001\325\n\201\022\000\022\000\001\325\022m\n\000\013\000\001\325\032I\006\000\002\000\001\325\002a\022\000\t\000\001\325\t\031\006\000\001\000\001\325\001\007\002\000\022\000\001\325\022o\016\000\013K\026\002\002\005\000\327\002\025\002\002\237\361\210\277\n| \325\200\000\025\000\022\000\001\325\022q\022\000\013\000\001\325\013M\n\000\f\000\001\325\021-\n\000\t\033\022\002\001\000\001\325\001\t\006\000\002\005\000\327\002%\002\002\023;$\002\003\000\001\325\013O\016\000\237\361\210\277\n| \325\200\024\026\000\t\000\001\325\t\035\n\000\013\000\001\325\022=\n\000\003\000\001\325\003Q\022\000\001\013\002\002\007s\000\226\t\000\001\325\t\037\016\000\004\000\001\325\013?\016\000\013\000\001\325\f/\016\000\002j\000\327\002\007\002\002\235\377\210\277\003| \325\200\024\252\001\004\000\001\325\004A\022\000\005\000\001\325\0131\022\000\001\000\001\325\001\r\n\000\236\377\210\277\000u\000\201\002j\000\327\002\t\002\002\235\377\210\277\003| \325\200\006\252\001\004\000\001\325\t!\022\000\001\000\001\325\001\017\016\000\002j\000\327\002\013\002\002\235\377\210\277\003| \325\200\006\252\001\223\001\207\277\001\000\001\325\001\021\022\000\002j\000\327\002\t\002\002\235\377\210\277\003\000\207\277\003| \325\200\006\252\001\236\377\210\277\000\237\001\206\001j\000\327\002\003\002\002\235\377\210\277\002| \325\200\006\252\001\236\377\210\277\000\213\200\204\000\000\307\277\236\377\210\277\b\000\200\251\000\300\006\356\000\000\200\000\000\000\000\000\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\377\000\0048\021\021\021\021\204\000\0020\377\000\0068\"\"\"\"\200!\000\364(\000\000\370\023\001\207\277\002\001\"\312\203\000J\004\003\001 \312\200\002\006\005\000\000|\333\006\002\000\000\000\000\306\277\301N\200\276\000\000\307\277\006\201\004\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\203\000\242\277\200\000$\312\377\226\002)\370\000\000\000\377\002\0066\000\016\000\000\377\002\0026\000\f\000\000\377\000\21283333D\002\207\277)\001 \312\200\004\002+)\003T~\377\006\b8\000\002\000\000\377\006\0068\000\003\000\000)\001 \312\002\003\000,)\003\\~\024\002\207\277)\001 \312\002\t\004-)\001 \312\002\007\002/\000 \334\331\001\000\000A\000\000\330\331\004\000\000G\000\000\330\331\002\000\000I\377\000\2148DDDD)\001\020\312)\00180)\001\020\312)\001::)\001\020\312)\001<<)\001\020\312)\001>>)\001\020\312)\0010@)\001\020\312)\00122)\001\020\312)\00144)\001\020\312)\00166)\001\020\312)\001 8)\001\020\312)\001\"\")\001\020\312)\001$$)\001\020\312)\001&&)\001\020\312)\001\030()\001\020\312)\001\032\032)\001\020\312)\001\034\034)\001\020\312)\001\036\036)\001\020\312)\001\020 )\001\020\312)\001\022\022)\001\020\312)\001\024\024)\001\020\312)\001\026\026)\001\020\312)\001\b\030)\001\020\312)\001\n\n)\001\020\312)\001\f\f)\001\020\312)\001\016\016)\001\020\312)\001\000\020)\001\020\312)\001\002\002)\001\020\312)\001\004\004)\001\020\312)\001\006\006)\003\020~\006\201\002\204\t\000\207\277\002\201\002\212\000\000\306\277)@J\314A\213\246|9@J\314A\213\346|1@J\314A\213\306|!@J\314A\213\206|\031@J\314G\213f|\021@J\314G\213F|\t@J\314G\213&|\001@J\314G\213\006|)@J\314C\213\246|9@J\314C\213\346|1@J\314C\213\306|!@J\314C\213\206|\031@J\314I\213f|\021@J\314I\213F|\t@J\314I\213&|\001@J\314I\213\006|\002\301\002\201\t\000\207\277\002\200\006\277\334\377\241\277A\000\240\277\200\002\020~\001\000\207\277\b\001\020\312\b\001\006\007\b\001\020\312\b\001\004\005\b\001\020\312\b\001\002\003\b\001\020\312\b\001\020\001\b\001\020\312\b\001\016\017\b\001\020\312\b\001\f\r\b\001\020\312\b\001\n\013\b\001\020\312\b\001\030\t\b\001\020\312\b\001\026\027\b\001\020\312\b\001\024\025\b\001\020\312\b\001\022\023\b\001\020\312\b\001 \021\b\001\020\312\b\001\036\037\b\001\020\312\b\001\034\035\b\001\020\312\b\001\032\033\b\001\020\312\b\001(\031\b\001\020\312\b\001&'\b\001\020\312\b\001$%\b\001\020\312\b\001\"#\b\001\020\312\b\0018!\b\001\020\312\b\00167\b\001\020\312\b\00145\b\001\020\312\b\00123\b\001\020\312\b\001@1\b\001\020\312\b\001>?\b\001\020\312\b\001<=\b\001\020\312\b\001:;\b\001\020\312\b\00109\b\001\020\312\b\001./\b\001\020\312\b\001,-\b\001\020\312\b\001*+\b\003R~\207\000\0006\000\"\000\364 \000\000\370\221\001\207\277\201\000\224|)UR\002\000\000J\324\202\000\002\0029uT\002\001\000J\324\203\000\002\002\t\025\022\002\03152\002)\000\001\325)W\002\000\n\000\001\325*w\002\000\021%\"\002\001\005\002\002\002\000J\324\205\000\002\002\002\000\001\325)Y\006\0001eV\002\n\000\001\325\ny\006\000!EB\002\204\000\224|\003\000J\324\206\000\002\002\004\000J\324\207\000\002\002\001\000\001\325\001\007\002\000\235\377\210\277\002[\004\002\022\000\001\325+g\002\000\n{\024\002\032\000\001\325!G\002\000\001\000\001\325\001\t\006\000\002\000\001\325\002]\n\000\022\000\001\325\022i\006\000\n\000\001\325\n}\n\000\024\002\207\277\001\013\002\002\002\000\001\325\002_\016\000\024\002\207\277\022k$\002\000\000\001\325\n\177\016\000\031\000\001\325\0317\002\000\n\000\001\325\032I\006\000\002\000\001\325\002a\022\000\022\000\001\325\022m\n\000\000\000\001\325\000\201\022\000\001\000\001\325\001\r\n\000\nK\024\002\021\000\001\325\021'\002\000\022\000\001\325\022o\016\000\023\000\001\325\0319\006\000\000\005\000\327\002\001\002\002\024\002\207\277\021\000\001\325\021)\006\000\022\000\001\325\022q\022\000\237\361\210\277\002| \325\200\000\025\000\n\000\001\325\nM\n\000\021+\"\002\000\005\000\327\000%\002\002\023;$\002\t\000\001\325\t\027\002\000\003\000\001\325\nO\016\000\013\000\001\325\021-\n\000\237\361\210\277\002| \325\200\004\026\000\n\000\001\325\022=\n\000\t\000\001\325\t\031\006\000\003\000\001\325\003Q\022\000\001\000\001\325\001\017\016\000\007s\000\226\004\000\001\325\n?\016\000\t\033\022\002\n\000\001\325\013/\016\000\000j\000\327\000\007\002\002\024\002\207\277\003\000\001\325\004A\022\000\005\000\001\325\t\035\n\000\235\377\210\277\002| \325\200\004\252\001\004\000\001\325\n1\022\000\000j\000\327\000\007\002\002\003\000\001\325\005\037\016\000\235\377\210\277\002| \325\200\004\252\001\223\001\207\277\000j\000\327\000\t\002\002\003\000\001\325\003!\022\000\235\377\210\277\003\000\207\277\002| \325\200\004\252\001\001\000\001\325\001\021\022\000\236\377\210\277\000u\000\201\000j\000\327\000\007\002\002\235\377\210\277\002| \325\200\004\252\001\236\377\210\277\000\237\001\206\000j\000\327\000\003\002\002\235\377\210\277\001| \325\200\004\252\001\236\377\210\277\000\213\200\204\000\000\307\277\236\377\210\277\b\000\200\251\000\300\006\356\000\000\000\000K\000\000\000\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\205\277\200!\000\364(\000\000\370\000\"\000\364\000\000\000\370u\000\204\276u\237\005\206\204\000\2040\261\004\207\277\200\204\224J\000\000\307\277\006\201\002\204\002\237\003\206\t\000\207\277\002\004\204\252\200\000\203\276\004\214\204\204\006\200\002\277\b\004\204\251\004\300\005\356\001\000\000\000B\000\000\000\000\000\300\277\000\000|\333J\001\000\000\000\000\306\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\002\000\242\277\200\002\002~\001\000\240\277\301\000\203\276\200\002\022~~\003j\221\001\000\207\277\t\001\020\312\t\001\006\b\t\001\020\312\t\001\004\006\t\001\020\312\t\001\002\004\t\001\020\312\t\001\020\002\t\001\020\312\t\001\016\020\t\001\020\312\t\001\f\016\t\001\020\312\t\001\n\f\t\001\020\312\t\001\030\n\t\001\020\312\t\001\026\030\t\001\020\312\t\001\024\026\t\001\020\312\t\001\022\024\t\001\020\312\t\001 \022\t\001\020\312\t\001\036 \t\001\020\312\t\001\034\036\t\001\020\312\t\001\032\034\t\001\020\312\t\001(\032\t\001\020\312\t\001&(\t\001\020\312\t\001$&\t\001\020\312\t\001\"$\t\001\020\312\t\0010\"\t\001\020\312\t\001.0\t\001\020\312\t\001,.\t\001\020\312\t\001*,\t\001\020\312\t\0018*\t\001\020\312\t\00168\t\001\020\312\t\00146\t\001\020\312\t\00124\t\001\020\312\t\001@2\t\001\020\312\t\001>@\t\001\020\312\t\001<>\t\001\020\312\t\001:<\t\003t~\365\000\244\277\203\000\0020\377\204\0046\000\016\000\000\006\003\000\327\004\204\002\002\021\002\207\277\007| \325\005\000\r\000\377\002\0026\370\000\000\0003\002\207\277Hj\000\327\377\f\002\002\b \000\000\377\000\21483333\377\000\2168DDDD\200\002\nJ\200\002\002~\377\204\0066\000\f\000\000\377\004\b8\000\002\000\000\377\004\0048\000\003\000\000I| \325\200\016\252\001\001\001\020\312\001\001::\001\003|~\001\003x~\005\007\226J\005\t\230J\005\005\232J\001\001\020\312\001\001@=\001\001\020\312\001\0012?\001\001\020\312\001\0014A\001\001\020\312\001\00163\001\001\020\312\001\00185\001\001\020\312\001\001*7\001\001\020\312\001\001,9\001\001\020\312\001\001.+\001\001\020\312\001\0010-\001\001\020\312\001\001\"/\001\001\020\312\001\001$1\001\001\020\312\001\001&#\001\001\020\312\001\001(%\001\001\020\312\001\001\032'\001\001\020\312\001\001\034)\001\001\020\312\001\001\036\033\001\001\020\312\001\001 \035\001\001\020\312\001\001\022\037\001\001\020\312\001\001\024!\001\001\020\312\001\001\026\023\001\001\020\312\001\001\030\025\001\001\020\312\001\001\n\027\001\001\020\312\001\001\f\031\001\001\020\312\001\001\016\013\001\001\020\312\001\001\020\r\001\001\020\312\001\001\002\017\001\001\020\312\001\001\004\021\001\001\020\312\001\001\006\003\001\001\020\312\001\001\b\005\001\003\016~\001\003\022~\201\000\203\276\200\002\204\201\022\000\240\277\000\000\310\277\301N\200\276Hj\000\327\377\220\002\002\000 \000\000\003\202\003\201\235\377\210\277I| \325\200\222\252\001\236\377\210\277\004\003\005\201\236\377\210\277\005\201\006\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\201\000\242\277\200\000\020\312\200\000BB\200\000\020\312\200\000DD\236\377\210\277\003\002\004\277\301\200\005\230\003\002\003\277\003\000\242\277|\300\005\356B\000\000\000H\370\357\377\000 \334\331K\000\000N\000\000\330\331L\000\000R\000\000\330\331M\000\000T\236\377\210\277~\005j\221\002\000\306\277:@J\314N\215\352|2@J\314N\215\312|*@J\314N\215\252|\"@J\314N\215\212|\001\000\306\277\032@J\314R\215j|\022@J\314R\215J|\n@J\314R\215*|\002@J\314R\215\n|:@J\314P\215\352|2@J\314P\215\312|*@J\314P\215\252|\"@J\314P\215\212|\000\000\306\277\032@J\314T\215j|\022@J\314T\215J|\n@J\314T\215*|\002@J\314T\215\n|\236\377\210\277\003\000\244\277\000\000\300\277\000\020|\333JB\000\000\000\000\310\277\301N\200\276\003\201\006\201\200\000\020\312\200\000BB\200\000\020\312\200\000DD\006\002\004\277\301\200\005\230\006\002\003\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\003\000\242\277|\300\005\356B\000\000\000H\370\377\377\377\226\234J\000\020\000\000\236\377\210\277~\005j\221\000\020\330\331L\000\000R\000 \334\331N\000\000N\000\020\330\331M\000\000T\002\000\306\277\032@J\314R\215j|\022@J\314R\215J|\n@J\314R\215*|\002@J\314R\215\n|\001\000\306\277:@J\314N\215\352|2@J\314N\215\312|*@J\314N\215\252|\"@J\314N\215\212|\000\000\306\277\032@J\314T\215j|\022@J\314T\215J|\n@J\314T\215*|:@J\314P\215\352|2@J\314P\215\312|*@J\314P\215\252|\"@J\314P\215\212|\002@J\314T\215\n|\236\377\210\277q\377\244\277\000\000\300\277\000\000|\333JB\000\000m\377\240\277\207\000\2046\000\"\000\364 \000\000\370\203\000\000>\002\000\207\277\201\204\224|\000\000J\324\202\204\002\002\001\000J\324\203\204\002\002\235\377\210\277*WT\002\022'$\002\n\027\024\002\"GD\002\03274\002\023\000\001\325*Y\002\000\002\007\004\002\022\000\001\325\022)\002\000\033\000\001\325\"I\002\000\032\000\001\325\0329\002\000\023\000\001\325\023[\006\0002gd\002\022\000\001\325\022+\006\000\024\000\001\325\033K\006\000\032\000\001\325\032;\006\000\n\000\001\325\n\031\002\000\013\000\001\3252i\002\000\002\000\001\325\002\t\002\000\223\001\207\277\n\000\001\325\n\033\006\000\013\000\001\325\013k\006\000:wt\002\204\204\224|\002\000\001\325\002\013\006\000\235\377\210\277\013m\026\002:\000\001\325:y\002\000\023]&\002\024M(\002\022-$\002\n\035\024\002\003\000\001\325:{\006\000\002\r\004\002\007s\000\226\236\377\210\277\000u\000\201\003}\006\002\002\000J\324\205\204\002\002\003\000J\324\206\204\002\002\004\000J\324\207\204\002\002\236\377\210\277\000\237\001\206\003\000\001\325\003\177\n\000\013\000\001\325\013o\n\000\023\000\001\325\023_\n\000\024\000\001\325\024O\n\000\022\000\001\325\022/\n\000\003\000\001\325\003\201\016\000\013\000\001\325\013q\016\000\023\000\001\325\023a\016\000\004\000\001\325\024Q\016\000\n\000\001\325\n\037\n\000\003\000\001\325\003\203\022\000\013\000\001\325\013s\022\000\023\000\001\325\023c\022\000\004\000\001\325\004S\022\000\002\000\001\325\002\017\n\000\236\377\210\277\000\213\200\204\003\005\000\327\003\027\002\002\237\361\210\277\013| \325\200\000\025\000\002\000\001\325\002\021\016\000\303\001\207\277\003\005\000\327\003'\002\002\032=&\002\237\361\210\277\013| \325\200\026\026\000\003j\000\327\003\t\002\002\243\001\207\277\f\000\001\325\023?\n\000\235\377\210\277\004| \325\200\026\252\001\002\000\001\325\002\023\022\000\000\000\307\277\236\377\210\277\b\000\200\251\005\000\001\325\fA\016\000\f\000\001\325\0221\016\000\022\001\207\277\005\000\001\325\005C\022\000\006\000\001\325\f3\022\000\302\001\207\277\003j\000\327\003\013\002\002\005\000\001\325\n!\016\000\235\377\210\277\004| \325\200\b\252\001\003j\000\327\003\r\002\002\243\001\207\277\005\000\001\325\005#\022\000\235\377\210\277\004| \325\200\b\252\001\"\001\207\277\003j\000\327\003\013\002\002\235\377\210\277\004| \325\200\b\252\001\"\001\207\277\002j\000\327\003\005\002\002\235\377\210\277\003| \325\200\b\252\001\236\377\210\277\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001|\300\006\356\000\000\000\001\000\000\000\000\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\205\277\000\"\000\364(\000\000\370\000A\000\364\000\000\000\370u\000\212\276u\237\013\206s\000\214\276s\237\r\206\200\000\"\312\204\000@\b\203\000\2220\002\000\207\277\b\001 \312\200\202J+\b\001\020\312\b\001\004\007\b\001\020\312\b\001\004\006\b\001\020\312\b\001\002\002\b\003 ~\000\000\307\277\b\201\002\204\b\001\020\312\b\001\016\001\002\237\003\206\b\001\020\312\b\001\f\017\002\n\212\252\002\f\214\252\n\214\216\204\f\214\220\204\004\016\216\251\006\020\220\251\001\000\205\277\016\300\005\3561\000\000\000A\000\000\000\020\300\005\3565\000\000\000A\000\000\000\b\001\020\312\b\001\n\r\b\001\020\312\b\001\030\013\b\001\020\312\b\001\026\t\b\001\020\312\b\001\024\027\b\001\020\312\b\001\022\025\b\001\020\312\b\001 \023\b\001\020\312\b\001\036\021\b\001\020\312\b\001\034\037\b\001\020\312\b\001\032\035\b\001\020\312\b\001(\033\b\001\020\312\b\001&\031\b\001\020\312\b\001$'\b\001\020\312\b\001\"%\b\001\020\312\b\0010#\b\001\020\312\b\001.!\b\001\020\312\b\001,/\b\001\020\312\b\001*-\b\001\020\312\b\001@)\b\001\020\312\b\001>?\b\001\020\312\b\001<=\b\001\020\312\b\001:;\b\003r~\200\000\217\276\b\201\004\277\001\000\300\277\000\000|\333J1\000\000\000\000\300\277\000 |\333J5\000\000\b\001\020\312\b\00168\b\001\020\312\b\00146\b\001\020\312\b\00124\b\001\020\312\b\00102\000\000\306\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\341\000\242\277\200\000\"\312\206\000\0021\377\222\0026\370\000\000\000\201\000\b0\377\202\0066\000\016\000\000\024\002\207\2771\001$\312\377\202L5\000\f\000\0001\001 \312\200\002J21\001$\312\377\004\0006\000\b\000\0001\001\020\3121\001281\001\"\312\203\b\0027\243\001\207\2771\001 \312K\003N:1\003h~P\003\000\327\004\004\002\002\001\000\207\277Q| \325\005\000\r\000R\003\000\327\006\004\002\002\377\006\2328\000\002\000\000\377\006\2348\000\003\000\000\237\361\210\277S| \325\007\000\r\0001\001\020\3121\001<91\001\020\3121\001>;1\001\020\3121\001@=1\001\020\3121\001*?1\001\020\3121\001,)1\001\020\3121\001.+1\001\020\3121\0010-1\001\020\3121\001\"/1\001\020\3121\001$!1\001\020\3121\001&#1\001\020\3121\001(%1\001\020\3121\001\032'1\001\020\3121\001\034\0311\001\020\3121\001\036\0331\001\020\3121\001 \0351\001\020\3121\001\022\0371\001\020\3121\001\024\0211\001\020\3121\001\026\0231\001\020\3121\001\030\0251\001\020\3121\001\n\0271\001\020\3121\001\f\t1\001\020\3121\001\016\0131\001\020\3121\001\020\r1\001\020\3121\001\002\0171\001\020\3121\001\004\0011\001\020\3121\001\006\0031\001\020\3121\001\b\0051\003\016~\000 \003\260\200\000\204\276\017\000\216\276\n\000\240\277\000\000\310\277\301N\200\276\004\201\004\215\005\000\216\276\005\002\006\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000v\000\242\277\016\201\005\201\200\000\020\312\200\000BA\200\000\020\312\200\000DC\200\000\020\312\200\000FE\200\000\020\312\200\000HG\236\377\210\277\005\002\004\277\301\200\006\230\005\002\003\277\027\000\242\277\n\016\220\251\f\016\222\251\236\377\210\277\020\214\220\204\236\377\210\277Aj\000\327P!\000\002\235\377\210\277B| \325\021\242\252\001\022\214\220\204\236\377\210\277Ej\000\327R!\000\002\235\377\210\277F| \325\021\246\252\001|\300\005\356A\000\000\000A\000\020\000|\300\005\356E\000\000\000E\000\020\000\004\200\006\277\200\377\007\230\000\020\000\000\236\377\210\277\007\226\324J\003\377\007\230\0000\000\000~\006j\221\236\377\210\277\007\236\310Jj\233\260Jj\231\300Jj\235\324J\000 \334\331d\000\000T\000\000\330\331X\000\000h@`\334\331d\000\000X\200\240\334\331d\000\000\\\000 \334\331`\000\000`\300\340\334\331d\000\000d\000\000\330\331j\000\000j\005\000\306\277\031@J\314h\251f|\004\000\306\277\021@J\314h\261F|\003\000\306\277\t@J\314h\271&|\002\000\306\2771@J\314`\251\306|9@J\314`\261\346|)@J\314`\271\246|\001\000\306\277!@J\314`\311\206|\001@J\314h\311\006|1@J\314b\255\306|9@J\314b\265\346|)@J\314b\275\246|!@J\314b\315\206|\000\000\306\277\031@J\314j\255f|\021@J\314j\265F|\t@J\314j\275&|\001@J\314j\315\006|\217\377\244\277\004\201\006\277\200\377\006\230\000\020\000\000\003\377\007\230\0000\000\000\236\377\210\277\006\224\250J\007\224\252J\001\000\300\277\000\000|\333TA\000\000\000\000\300\277\000\000|\333UE\000\000\200\377\240\277\207\000\0006\200!\000\364 \000\000\370\001\000\207\277\201\000\224|\235\377\210\2771eb\002\000\000J\324\202\000\002\0029ud\002\001\000J\324\203\000\002\002\t\025\022\002\03152\0021\000\001\3251g\002\000\n\000\001\3252w\002\000\021%\"\002\001\005\002\002\002\000J\324\205\000\002\002\002\000\001\3251i\006\000)UR\002\n\000\001\325\ny\006\000!EB\002\204\000\224|\003\000J\324\206\000\002\002\004\000J\324\207\000\002\002\001\000\001\325\001\007\002\000\235\377\210\277\002k\004\002\022\000\001\325)W\002\000\n{\024\002\032\000\001\325!G\002\000\001\000\001\325\001\t\006\000\237\361\210\277\002\000\001\325\002m\n\000\022\000\001\325\022Y\006\000\n\000\001\325\n}\n\000\001\013\002\002\024\002\207\277\002\000\001\325\002o\016\000\022[$\002\004\000\207\277\000\000\001\325\n\177\016\000\031\000\001\325\0317\002\000\n\000\001\325\032I\006\000\002\000\001\325\002q\022\000\022\000\001\325\022]\n\000\000\000\001\325\000\201\022\000\001\000\001\325\001\r\n\000\nK\024\002\021\000\001\325\021'\002\000\022\000\001\325\022_\016\000\023\000\001\325\0319\006\000\000\005\000\327\002\001\002\002\024\002\207\277\021\000\001\325\021)\006\000\022\000\001\325\022a\022\000\237\361\210\277\002| \325\200\000\025\000\n\000\001\325\nM\n\000\021+\"\002\000\005\000\327\000%\002\002\023;$\002\t\000\001\325\t\027\002\000\003\000\001\325\nO\016\000\013\000\001\325\021-\n\000\237\361\210\277\002| \325\200\004\026\000\n\000\001\325\022=\n\000\t\000\001\325\t\031\006\000\003\000\001\325\003Q\022\000\001\000\001\325\001\017\016\000\ts\000\226\004\000\001\325\n?\016\000\t\033\022\002\n\000\001\325\013/\016\000\000j\000\327\000\007\002\002\024\002\207\277\003\000\001\325\004A\022\000\005\000\001\325\t\035\n\000\235\377\210\277\002| \325\200\004\252\001\004\000\001\325\n1\022\000\000j\000\327\000\007\002\002\003\000\001\325\005\037\016\000\235\377\210\277\002| \325\200\004\252\001\223\001\207\277\000j\000\327\000\t\002\002\003\000\001\325\003!\022\000\235\377\210\277\003\000\207\277\002| \325\200\004\252\001\001\000\001\325\001\021\022\000\236\377\210\277\000u\000\201\000j\000\327\000\007\002\002\235\377\210\277\002| \325\200\004\252\001\236\377\210\277\000\237\001\206\000j\000\327\000\003\002\002\235\377\210\277\001| \325\200\004\252\001\236\377\210\277\000\213\200\204\000\000\307\277\236\377\210\277\006\000\200\251\000\300\006\356\000\000\000\000I\000\000\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\205\277\000\"\000\364(\000\000\370\000%\000\364\020\000\000\370\000A\000\364\000\000\000\370s\000\220\276s\237\021\206u\000\214\276u\237\r\206\200\000\"\312\204\000@\b\202\000\2040\203\000\2220#\002\207\277\b\001 \312\200\202\212w\b\003\016~\b\001 \312\200\204\212x\b\001\020\312\b\001\004\006\b\003\b~\000\000\307\277\t\237\023\206\t\000\222\276\b\201\002\204\b\237\017\206\b\000\216\276\022\020\222\251\002\237\003\206\016\f\212\252\022\016\216\252\002\f\214\252\002\020\220\252\n\212\226\204\016\212\222\204\f\214\230\204\020\214\232\204\024\026\226\251\024\022\222\251\004\030\230\251\006\032\232\251\001\000\205\277\026\000\005\356G\000\000\000B\000\000\000\022\000\005\356H\000\000\000B\000\000\000\001\000\205\277\030\300\005\356C\000\000\000A\000\000\000\032\300\005\356z\000\000\000A\000\000\000\b\001\020\312\b\001\002\003\b\001\020\312\b\001\020\001\b\001\020\312\b\001\016\017\b\001\020\312\b\001\f\r\b\001\020\312\b\001\n\013\b\001\020\312\b\001\030\t\b\001\020\312\b\001\026\027\b\001\020\312\b\001\024\025\b\001\020\312\b\001\022\023\b\001\020\312\b\001 \021\b\001\020\312\b\001\036\037\b\001\020\312\b\001\034\035\b\001\020\312\b\001\032\033\b\001\020\312\b\001(\031\b\001\020\312\b\001&'\b\001\020\312\b\001$%\b\001\020\312\b\001\"#\b\001\020\312\b\0010!\b\001\020\312\b\001./\b\001\020\312\b\001,-\b\001\020\312\b\001*+\b\001\020\312\b\0018)\b\001\020\312\b\00167\b\001\020\312\b\00145\b\001\020\312\b\00123\b\001\020\312\b\001@1\b\001\020\312\b\001>?\b\001\020\312\b\001<=\b\001\020\312\b\001:;\b\001\020\312\b\001\\9\b\001\020\312\b\001NY\b\001\020\312\b\001R]\b\001\020\312\b\001J_\b\001\020\312\b\001LK\b\001\020\312\b\001ZQ\b\001\020\312\b\001PO\b\001\020\312\b\001TS\b\001\020\312\b\001VU\b\001\020\312\b\001XW\b\001\020\312\b\001^[\b\001\020\312\b\001bM\b\001\020\312\b\001`a\b\001\020\312\b\001dc\b\001\020\312\b\001lf\b\001\020\312\b\001dh\b\001\020\312\b\001fj\b\001\020\312\b\001hl\b\001\020\312\b\001nk\b\001\020\312\b\001rp\b\001\020\312\b\001nr\b\001\020\312\b\001tq\b\001\020\312\b\001xv\b\001\020\312\b\001\202~\b\001\020\312\b\001t\202\b\001\020\312\b\001\210\177\b\001\020\312\b\001\200\207\b\003\002\177\b\001\020\312\b\001\206\205\b\003\022\177\200\000\223\276\b\201\004\277\001\000\300\277\000\000|\333\212C\000\000\000\000\300\277\000 |\333\212z\000\000@D<\330\213GH\000\b\001\020\312\b\001\204{\b\003\364~\b\001\020\312\b\001||\000\000\306\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000$\003\242\277\200\000$\312\377\222\002\211\370\000\000\000\200\000\"\312\201\000\000\206\200\000\"\312\206\000\002\200\303\001\207\277\200\000 \312\200\004\216\201\200\000$\312\217\000\002}\200\000$\312\377\202\004\205\000\016\000\000\214\003\000\327\024\204\002\002\002\000W\326\001\201\t\004\200\000\"\312\203\002\000\210\200\000$\312\377\006\002|\000\b\000\000\377\b 9\000\002\000\000\377\b\"9\000\003\000\000\377\204\b8\270\000\000\000\215| \325\025\000\r\000\226\003\000\327\004\002\002\002\237\361\210\277\227| \325\005\000\r\000\230\003\000\327\006\002\002\002\200\000$\312\377\202\216z\000\f\000\000\200\000 \312\216\007\222\207\200\000$\312\377\204\222\204@\003\000\000\200\000\"\312\203\004\224\177\200\000$\312\377\b\224x\370\003\000\000\237\361\210\277\231| \325\007\000\r\000\200\000\020\312\200\000\202{\200\000\020\312\200\000~w\200\000\020\312\200\000vu\200\000\020\312\200\000t\203\200\000\020\312\200\000ry\200\000\020\312\200\000pq\200\000\020\312\200\000no\200\000\020\312\200\000ls\200\000\020\312\200\000jk\200\000\020\312\200\000hi\200\000\020\312\200\000fg\200\000\020\312\200\000de\200\000\020\312\200\000`m\200\000\020\312\200\000bc\200\000\020\312\200\000^a\200\000\020\312\200\000LX\200\000\020\312\200\000ZV\200\000\020\312\200\000VT\200\000\020\312\200\000TP\200\000\020\312\200\000RZ\200\000\020\312\200\000LO\200\000\020\312\200\000PJ\200\000\020\312\200\000JR\200\000\020\312\200\0008N\206\001\020\312\206\001::\206\001\020\312\206\001<<\206\001\020\312\206\001>>\206\001\020\312\200\0000@\206\001\020\312\206\00122\206\001\020\312\206\00144\206\001\020\312\206\00166\206\001\020\312\200\000(8\206\001\020\312\206\001**\206\001\020\312\206\001,,\206\001\020\312\206\001..\206\001\020\312\200\000 0\206\001\020\312\206\001\"\"\206\001\020\312\206\001$$\206\001\020\312\206\001&&\206\001\020\312\200\000\030(\206\001\020\312\206\001\032\032\206\001\020\312\206\001\034\034\206\001\020\312\206\001\036\036\206\001\020\312\200\000\020 \206\001\020\312\206\001\022\022\206\001\020\312\206\001\024\024\206\001\020\312\206\001\026\026\206\001\020\312\200\000\b\030\206\001\020\312\206\001\n\n\206\001\020\312\206\001\f\f\206\001\020\312\206\001\016\016\206\001\020\312\200\000\000\020\206\001\020\312\206\001\002\002\206\001\020\312\206\001\004\004\206\001\020\312\206\001\006\006\206\001\020\312\200\000^\b\200\000\020\312\200\000\\]\200\002\262~\000 \003\260\000@\006\260\200\000\207\276\023\000\222\276\n\000\240\277\000\000\310\277\301N\200\276\007\201\007\215\b\000\222\276\b\002\006\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000k\002\242\277\022\201\b\201\301\000\204\276\b\002\004\277\301\200\024\230\236\377\210\277~\024j\213\236\377\210\277\003\000\244\277\022\200\r\277\200\000\204\276\301\200\025\230\200\000\020\312\200\000\232\232\200\000\020\312\200\000BA\200\000\020\312\200\000DC\200\000\020\312\200\000FE\200\000\020\312\200\000HG\236\377\210\277~\004j\221\236\377\210\277C\000\244\277\022\200\r\277\301\200\025\230\022\202\004\201\236\377\210\277\004\002\004\277\301\200\004\230\236\377\210\277\025\004\004\213\236\377\210\277~\004j\221\236\377\210\277\036\000\244\277\022\201\004\205\023\000\205\276\236\377\210\277\004\201\004\201\301\000\225\276\236\377\210\277\n\004\226\251\016\004\204\251\236\377\210\277\026\212\226\204\004\212\204\204\236\377\210\277Aj\000\327\214-\000\002\235\377\210\277B| \325\027\032\253\001Cj\000\327\214\t\000\002\235\377\210\277D| \325\005\032\253\001\001\000\205\277|\000\005\356\233\000\000\000A\000\000\000|\000\005\356\232\000\000\000C\000\000\000\002\000\240\277\200\000\020\312\200\000\232\233\f\022\204\251\020\022\226\251\236\377\210\277\004\214\204\204\236\377\210\277Aj\000\327\226\t\000\002\235\377\210\277B| \325\005.\253\001\026\214\204\204\236\377\210\277Ej\000\327\230\t\000\002\235\377\210\277F| \325\0052\253\001|\300\005\356A\000\000\000A\000\020\000|\300\005\356E\000\000\000E\000\020\000\007\200\006\277\200\377\004\230\000\020\000\000\236\377\210\277\004\034eK\003\377\004\230\0000\000\000~\025j\221\236\377\210\277\004$YK\262\037AK\262!aK\262#eK\000 \334\331\254\000\000\234\000 \334\331\240\000\000\240@`\334\331\254\000\000\244\200\240\334\331\254\000\000\250\300\340\334\331\254\000\000\254\000\000\330\331\260\000\000\260\000\000\330\331\262\000\000\262\005\000\306\2779@J\314\2409\347|\004\000\306\2771@J\314\240I\307|\003\000\306\277)@J\314\240Q\247|\002\000\306\277!@J\314\240Y\207|\001\000\306\277\031@J\314\2609g|\021@J\314\260IG|\t@J\314\260Q'|\001@J\314\260Y\007|9@J\314\242=\347|1@J\314\242M\307|)@J\314\242U\247|!@J\314\242]\207|\000\000\306\277\031@J\314\262=g|\021@J\314\262MG|\t@J\314\262U'|\001@J\314\262]\007|\253\001\244\277\022\201\f\277\353\001\204\276\006\377\026\230\000H\000\0009\013r~\236\377\210\277\005\000 \312\026&\235\235:\013t~<\013x~3\013f~\005\002<\177|\000\005\354\237\000\f\000\234\000\000\000\000\000\300\277\2108;K\026(OK2\013d~;\013v~1\013b~|\000\005\354\240\000\f\000\235\000\000\000\000\000\300\277\2208;K)\013R~4\013h~+\013V~!\013B~|\000\005\354\241\000\f\000\235\000\000\000\000\000\300\277\2308;K*\013T~#\013F~,\013X~$\013H~|\000\005\354\242\000\f\000\235\000\000\000\000\000\300\277\2408;K\031\0132~\"\013D~\033\0136~\034\0138~|\000\005\354\243\000\f\000\235\000\000\000\000\000\300\277\2508;K\022\013$~\023\013&~\t\013\022~\024\013(~|\000\005\354\244\000\f\000\235\000\000\000\000\000\300\277\2608;K\002\013\004~\032\0134~\021\013\"~\013\013\026~|\000\005\354\245\000\f\000\235\000\000\000\000\000\300\277\2708;K\n\013\024~\f\013\030~\003\013\006~\001\013\002~|\000\005\354\246\000\f\000\235\000\000\000\000\000\300\277\377N;K\000\004\000\000\004\013\b~|\000\005\354\250\000\f\000\235\000\000\000\000\000\310\277\250A\307\310\250?\251\252\241\000\207\277\252u\022W\250Eu\020:y\002W\250Iu\020>\013x~\261\000\207\277\251s\000\310:y|\206\250Mu\020@\013x~\250C\301\310:yz9A\001\207\2779w\nW\250Gs\020=\013v~\005\002t~9w\006\310\250K9\200?\013v~\321\000\207\2779w\370V\377NsJ\200\004\000\000|\000\005\354;\000\f\0009\000\000\000\000\000\310\277;?\307\310;A=<<c\000\310=e\210\207;C\307\310;E31\301\000\207\2771g\000\3102i~\204;G\307\310;I315\013f~6\013h~1g\000\3102ix{;K\307\310;M317\013f~8\013h~\022\001\207\2771g\356V2i\352V\005\000 \312\377N12\000\005\000\000|\000\005\3543\000\f\0001\000\000\000\000\000\310\2773?\307\3103A54\241\000\207\2774S\000\3105U\202\2023C\307\3103E+))W\000\310*Yx~3G\307\3103I+)-\013V~.\013X~\022\001\207\277)W\354V*Y\006\3103K)t3MU\020/\013V~0\013X~\022\001\207\277)W\342V*Y\336V\005\000 \312\377N)*\200\005\000\000|\000\005\354+\000\f\000)\000\000\000\000\000\310\277+?\307\310+A-,\241\000\207\277,C\000\310-Err+C\307\310+E#!!G\340VB\001\207\277\"I\006\310+G!n+IE\020%\013F~&\013H~!G\326VB\001\207\277\"I\322V+K\307\310+M#!'\013F~(\013H~!G\316V\002\000\207\277\"I\312V\005\000 \312\3778!\"\200\000\000\000|\000\005\354#\000\f\000!\000\000\000\000\000\300\277\3778CJ\210\000\000\000|\000\005\354$\000\f\000!\000\000\000\000\000\300\277\3778CJ\220\000\000\000|\000\005\354%\000\f\000!\000\000\000\000\000\300\277\3778CJ\230\000\000\000|\000\005\354&\000\f\000!\000\000\000\000\000\300\277\3778CJ\240\000\000\000|\000\005\354'\000\f\000!\000\000\000\000\000\300\277\3778CJ\250\000\000\000|\000\005\354(\000\f\000!\000\000\000\000\000\300\277\3778CJ\260\000\000\000|\000\005\354+\000\f\000!\000\000\000\000\000\300\277\026*CJ|\000\005\354!\000\f\000!\000\000\000\000\000\300\277|\000\005\354\"\000\f\000\235\000\000\000\000\000\310\277\"GX\020\241\000\207\277,3\330V\"K2\020\0317\324V\"O2\020\035\0136~\261\000\207\277\0317\314V\"W2\020\037\0136~\0317\306V|\000\005\354\031\000\f\0009\000\000\000\000\000\300\277\"IZ\020\200\002r~\001\000\207\2779\003v~9\001\020\3129\0014=9\001\020\3129\0016?9\003f~9\001\020\3129\001,59\001\020\3129\001.79\003^~9\003:~9\001\020\3129\0018\0379\003`~9\003t~9\003x~9\003|~9\003\200~\000\000\306\277\031I6\020\241\000\207\277\033%\304V\031M$\0209\001\000\312\022)X\033\031Q$\020\026\013(~9\003,~\002\000\207\277\022)\250V\031C$\020-5\332V\"M4\020\030\013(~9\001\020\3129\001\030-\323\001\207\277\0329\320V\"Q4\020\036\0138~\022)\020\3109\001\036M9\003(~\0329\310V\"C4\020 \0138~9\003D~9\003@~\243\000\207\277\0329\300V\031G4\0209\001\000\312\032#`\034\031K\"\0209\0034~B\001\207\277\021'\274V\031O\"\020\025\013&~9\003*~\021'\254V\031W\"\020\027\013&~9\0032~9\003.~\003\000\207\277\021'\240V|\000\005\354\021\000\f\0001\000\000\000\000\000\300\2779\001\020\3129\00121\000\000\306\277\021G\306\310\021I\022\022\241\001\207\277\022\023\264V\021K\022\020\023\025\266V\021M\024\0209\001\020\3129\001\022\022D\001\207\277\t\027\256V\021O\022\020\r\013\026~9\003\032~\t\027\246V\021W\022\020\017\013\026~9\003\036~\002\000\207\277\t\027\230V|\000\005\354\t\000\f\000)\000\000\000\000\000\300\277\n\031\252V\021Q\024\020\016\013\030~9\001\020\3129\001*)B\001\207\2779\001\000\312\n\031N\016\021C\024\020\020\013\030~9\001\020\3129\001\020\021\n\031\224V9\003\030~\000\000\306\277\tG\306\310\tI\n\n9\001\020\3129\001$#\302\001\207\277\n\003\000\310\013\005RQ\tK\306\310\tM\002\0019\001\020\3129\001&%9\003\024~\001\007\000\310\002\tJN\tO\306\310\tQ\002\001\005\013\006~\006\013\b~9\001\020\3129\001('9\003\026~\024\002\207\277\001\007\276V\002\t\272V\tW\306\310\tC\002\001\007\013\006~\b\013\b~9\003V~9\003B~9\003\022~\004\000\207\277\001\007\000\310\002\tX\\9\001\020\3129\001\b\0019\001\020\3129\001\002\0029\001\020\3129\001\004\0049\001\020\3129\001\006\006~\024j\221\236\377\210\277\252\375\244\277\007\201\006\277\200\377\004\230\000\020\000\000\003\377\005\230\0000\000\000\022\202\024\201\236\377\210\277\004\0249K\024\002\004\277\005\024;K\301\200\024\230\001\000\300\277\000\000|\333\234A\000\000\000\000\300\277\000\000|\333\235E\000\000\236\377\210\277\025\024\024\213\236\377\210\277~\024j\221\236\377\210\277\223\375\244\277\022\201\f\277\377\006\004\230\000H\000\000\236\377\210\277\004\026\203J\000\004<\330A\233\232\000\213\375\240\277\207\000\0006\200\"\000\364 \000\000\3701\001\207\277\002\000J\324\201\000\002\002\202\000\224|\237\361\210\2779\000\001\3259u\n\000:\000\001\325\206\023\013\0001\000\001\3251e\n\0002\000\001\325\207\021\013\000)\000\001\325)U\n\000\235\377\210\2779wr\002\000\000J\324\203\000\002\002:\013S\3121g0:\001\000J\324\204\000\002\0022\tS\312)W(2\024\002\207\2779\000\001\3259y\002\000+\000\001\325:\003\003\0001\000\001\3251i\002\000\003\000J\324\205\000\002\0022\000\001\3252\377\002\0003\000\001\3259{\006\000+\000\001\325+\001\007\0001\000\001\3251k\006\000\004\000J\324\206\000\002\002*\000\001\325\202\007\013\000\237\361\210\2773\000\001\3253}\016\000+\000\001\325+\373\016\000)\000\001\325)Y\002\0001\000\001\3251m\016\000\005\000J\324\207\000\002\002,\000\001\3253\177\022\000\000\000\001\325+\371\022\000+\000\001\3252\367\006\000!\000\001\325!E\n\000*\375T\0021\000\001\3251o\022\000\237\361\210\277,\000\001\325,\201\026\000\000\000\001\325\000\365\026\000)\000\001\325)[\006\000+\000\001\325+\361\016\000!GB\002*\000\001\325*\363\002\000\021\000\001\325\021%\n\000-\000\001\3251q\026\000\000\006\000\327,\001\002\002+\000\001\325+\357\022\000)\000\001\325)]\016\000\237\361\210\277,| \325\200\000\031\000*\000\001\325*\355\006\000\021'\"\002\000\006\000\327\000[\002\002+\000\001\325+\353\026\000)\000\001\325)_\022\000\237\361\210\277\"| \325\200X\032\000*\000\001\325*\351\016\000#\000\001\325r\347\n\000\021\000\001\325\021)\002\000\000\006\000\327\000W\002\002)\000\001\325)a\026\000\237\361\210\277\"| \325\200D\032\000*\000\001\325*\343\022\000!\000\001\325!I\002\000\031\000\001\325\0315\n\000\021\000\001\325\021+\006\000\025\000\001\325a\305\n\000#\341F\002\000\006\000\327\000S\002\002\237\361\210\277\032| \325\200D\032\000\"\000\001\325*\337\026\000!\000\001\325!K\006\000#\000\001\325#\335\002\000\001\000\001\325\001\005\n\000\002\000\001\325Q\245\n\000\03172\002\033\000\001\325l\333\n\000\000\006\000\327\000E\002\002!\000\001\325!M\016\000\"\000\001\325#\327\006\000\t\000\001\325\t\025\n\000\n\000\001\325Z\267\n\000\033\3256\002\002\235\004\002\031\000\001\325\0319\002\000\022\000\001\325!O\022\000\034\000\001\325\"\323\016\000\n\257\024\002\033\000\001\325\033\321\002\000\031\000\001\325\031;\006\000\022\000\001\325\022Q\026\000\023\000\001\325\034\317\022\000\237\361\210\277\032| \325\2004\032\000\031\000\001\325\031=\016\000\033\000\001\325\033\315\006\000\000\006\000\327\000%\002\002\023\000\001\325\023\313\026\000\024\002\207\277\024\000\001\325\031?\022\000\031\000\001\325\033\311\016\000\237\361\210\277\022| \325\2004\032\000\000\006\000\327\000'\002\002\023\000\001\325\024A\026\000\024\000\001\325\031\307\022\000\021\000\001\325\021-\016\000\237\361\210\277\022| \325\200$\032\000\000\006\000\327\000'\002\002\023\000\001\325\024\301\026\000\021\000\001\325\021/\022\000\025\275(\002\237\361\210\277\022| \325\200$\032\000\000\006\000\327\000'\002\002\021\000\001\325\0211\026\000\023\000\001\325\024\261\002\000\t\027\022\002\237\361\210\277\022| \325\200$\032\000\000\006\000\327\000#\002\002\021\000\001\325\023\255\006\000\t\000\001\325\t\031\002\000\n\000\001\325\n\253\002\000\001\007\002\002\002\000\001\325\002\227\002\000\f\000\001\325\021\251\016\000\t\000\001\325\t\033\006\000\n\000\001\325\n\247\006\000\001\000\001\325\001\t\002\000\237\361\210\277\013| \325\200$\032\000\003\000\001\325\f\241\022\000\t\000\001\325\t\035\016\000\n\000\001\325\n\237\016\000\001\000\001\325\001\013\006\000\002\000\001\325\002\277\006\000\003\000\001\325\003\233\026\000\004\000\001\325\t\037\022\000\t\000\001\325\n\231\022\000\001\000\001\325\001\r\016\000\002\000\001\325\002\273\016\000\000j\000\327\000\007\002\002\004\000\001\325\004!\026\000\235\377\210\277\003| \325\200\026\252\001\005\000\001\325\t\225\026\000\001\000\001\325\001\017\022\000\000j\000\327\000\t\002\002\235\377\210\277\003| \325\200\006\252\001\002\000\001\325\002\271\022\000\003\000\207\277\000j\000\327\000\013\002\002\001\000\001\325\001\021\026\000\235\377\210\277\003| \325\200\006\252\001\002\000\001\325\002\263\026\000\ts\000\226\000j\000\327\000\003\002\002\235\377\210\277\001| \325\200\006\252\001\236\377\210\277\000u\000\201\000j\000\327\000\005\002\002\236\377\210\277\000\237\001\206\235\377\210\277\001| \325\200\002\252\001\236\377\210\277\000\213\200\204\000\000\307\277\236\377\210\277\n\000\200\251\000\300\006\356\000\000\000\000I\000\000\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\205\277\000B\000\364 \000\000\370\000`\000\364\000\000\000\370s\000\224\276s\237\025\206u\000\220\276u\237\021\206\202\000\2060\204\000\0220\217\000\0267\201\000\0303\024\002\207\277\200\206\034K\200\022\032K\000\000\307\277\013\237\027\206\013\000\226\276\n\201\f\204\n\237\023\206\n\000\222\276\026\024\226\251\f\237\r\206\022\020\216\252\026\022\222\252\f\020\220\252\f\024\224\252\016\212\230\204\022\212\226\204\020\214\232\204\024\214\234\204\004\030\230\251\004\026\226\251\000\032\232\251\002\034\234\251\001\000\205\277\030\000\005\356\n\000\000\000C\000\000\000\026\000\005\356\013\000\000\000C\000\000\000\001\000\205\277\032\300\005\356\001\000\000\000\t\000\000\000\034\300\005\356\005\000\000\000\t\000\000\000\200\000\215\276\n\200\002\277\001\000\300\277\000\000|\333\215\001\000\000\000\000\300\277\000 |\333\215\005\000\000@D<\330\216\n\013\000\200\002\020~\000\000\306\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\003\000\242\277\217\000\2026\201\000\2042\001\000\240\277\301\000\215\276\205\000\0243\200\000\"\312\203\000H\006\200\000\020\312\200\000\004\007\200\000\020\312\200\000\002\005\200\000\020\312\200\000\020\003\200\000\020\312\200\000\016\001\200\000\020\312\200\000\f\017\200\000\020\312\200\000\n\r\200\000\020\312\200\000\030\013\200\000\020\312\200\000\026\t\200\000\020\312\200\000\024\027\200\000\020\312\200\000\022\025\200\000\020\312\200\000 \023\200\000\020\312\200\000\036\021\200\000\020\312\200\000\034\037\200\000\020\312\200\000\032\035\200\000\020\312\200\000(\033\200\000\020\312\200\000&\031\200\000\020\312\200\000$'\200\000\020\312\200\000\"%\200\000\020\312\200\0000#\200\000\020\312\200\000.!\200\000\020\312\200\000,/\200\000\020\312\200\000*-\200\000\020\312\200\0008+\200\000\020\312\200\0006)\200\000\020\312\200\00047\200\000\020\312\200\00025\200\000\020\312\200\000@3\200\000\020\312\200\000>1\200\000\020\312\200\000<?\200\000\020\312\200\000:=\200\000\020\312\200\000P;\200\000\020\312\200\000p9\200\000\020\312\200\000\210g\200\000\020\312\200\000x{\200\000\020\312\200\000J\207\200\000\020\312\200\000R\211\200\000\020\312\200\000`Y\200\000\020\312\200\000nh\200\000\020\312\200\000xz\200\000\020\312\200\000ZK\200\000\020\312\200\000|S\200\000\020\312\200\000La\200\000\020\312\200\000Ti\200\000\020\312\200\000bq\200\000\020\312\200\000j}\200\000\020\312\200\000r[\200\000\020\312\200\000~\177\200\000\020\312\200\000\\M\200\000\020\312\200\000\200U\200\000\020\312\200\000Nc\200\000\020\312\200\000Vk\200\000\020\312\200\000ds\200\000\020\312\200\000l\201\200\000\020\312\200\000t]\200\000\020\312\200\000\202\203\200\000\020\312\200\000^O\200\000\020\312\200\000vW\200\000\020\312\200\000\204e\200\000\020\312\200\000Xm\200\000\020\312\200\000f\205\200\000\020\312\200\000nQ\200\000\020\312\200\000\206_\200\002\356~\200\002\352~~\rj\221'\003\244\277\200\000$\312\377\222\002u\370\000\000\000\200\000\"\312\201\000\000\206\200\000\"\312\213\024\003n#\002\207\277\200\000 \312\200\004\220f\200\000\"\312\206\024\003_\200\000\"\312\203\002\000^\200\000\"\312\211\024\005w\200\000$\312\377\006\002X\000\b\000\000\200\000\"\312\203\030\005\204\002\000W\326\002\201-\006\231\000\000\327\000\002\002\002\217\004\000\327\004\206\002\002\237\361\210\277\232| \325\001\000\001\000\233\000\000\327\002\002\002\002\220| \325\005\000\021\000\200\000$\312\377\b\222Q\000\f\000\000\377\b&9\000\002\000\000\377\b(9\000\003\000\000\200\000 \312\221\007\224v\200\000$\312\377\n\226\205@\003\000\000\200\000\"\312\203\004\226\202\377\n09\270\000\000\000\237\361\210\277\234| \325\003\000\001\000\200\000\020\312\200\000tm\200\000\020\312\200\000le\200\000\020\312\200\000dW\200\000\020\312\200\000VO\200\000\020\312\200\000N\203\200\000\020\312\200\000\200]\200\000\020\312\200\000\\\201\200\000\020\312\200\000~s\200\000\020\312\200\000rk\200\000\020\312\200\000jc\200\000\020\312\200\000bU\200\000\020\312\200\000TM\200\000\020\312\200\000L\177\200\000\020\312\200\000|[\200\000\020\312\200\000Z}\200\000\020\312\200\000zq\200\000\020\312\200\000hi\200\000\020\312\200\000`a\200\000\020\312\200\000RS\200\000\020\312\200\000JK\200\000\020\312\200\000xy\200\000\020\312u\001:o\200\000\020\312u\001<Y\200\000\020\312u\001>9u\001\020\312u\001@;u\001\020\312u\0012=u\001\020\312u\0014?\200\000\020\312u\00161u\001\020\312u\00183u\001\020\312u\001*5u\001\020\312u\001,7\200\000\020\312u\001.)u\001\020\312u\0010+u\001\020\312u\001\"-u\001\020\312u\001$/\200\000\020\312u\001&!u\001\020\312u\001(#u\001\020\312u\001\032%u\001\020\312u\001\034'\200\000\020\312u\001\036\031u\001\020\312u\001 \033u\001\020\312u\001\022\035u\001\020\312u\001\024\037\200\000\020\312u\001\026\021u\001\020\312u\001\030\023u\001\020\312u\001\n\025u\001\020\312u\001\f\027\200\000\020\312u\001\016\tu\001\020\312u\001\020\013u\001\020\312u\001\002\ru\001\020\312u\001\004\017\200\000\020\312u\001\006\001u\001\020\312u\001\b\003u\001\020\312\200\000\210\005u\001\020\312\200\000p\007\200\000\020\312\200\000P\211\200\002\016\177\200\002\366~\200\002\316~\200\000\205\276\000 \002\260\000@\003\260\200\000\212\276\236\377\210\277\005\000\204\276\n\000\240\277\000\000\310\277\301N\200\276\n\201\n\215\r\000\204\276\r\f\006\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000l\002\242\277\236\377\210\277\004\201\r\201\301\000\200\276\r\f\004\277\301\200\026\230\t\000\207\277~\026j\213\236\377\210\277\003\000\244\277\004\200\r\277\200\000\200\276\301\200\027\230\200\000\020\312\200\000\236\235\200\000\020\312\200\000BA\200\000\020\312\200\000DC\200\000\020\312\200\000FE\200\000\020\312\200\000HG\236\377\210\277~\000j\221\236\377\210\277C\000\244\277\004\200\r\277\301\200\027\230\004\202\000\201\236\377\210\277\000\f\004\277\301\200\000\230\236\377\210\277\027\000\000\213\236\377\210\277~\000j\221\236\377\210\277\036\000\244\277\004\201\000\205\005\000\201\276\236\377\210\277\000\201\000\201\301\000\227\276\236\377\210\277\016\000\230\251\022\000\200\251\236\377\210\277\030\212\230\204\000\212\200\204\236\377\210\277Aj\000\327\2171\000\002\235\377\210\277B| \325\031 \253\001Cj\000\327\217\001\000\002\235\377\210\277D| \325\001 \253\001\001\000\205\277|\000\005\356\236\000\000\000A\000\000\000|\000\005\356\235\000\000\000C\000\000\000\002\000\240\277\200\000\020\312\200\000\234\236\020\004\200\251\024\004\230\251\236\377\210\277\000\214\200\204\236\377\210\277Aj\000\327\231\001\000\002\235\377\210\277B| \325\0014\253\001\030\214\200\204\236\377\210\277Ej\000\327\233\001\000\002\235\377\210\277F| \325\0018\253\001|\300\005\356A\000\000\000A\000\020\000|\300\005\356E\000\000\000E\000\020\000\n\200\006\277\200\377\000\230\000\020\000\000\236\377\210\277\000\"kK\002\377\000\230\0000\000\000~\027j\221\236\377\210\277\000*_K\265%GK\265'gK\265)kK\000 \334\331\257\000\000\237\000 \334\331\243\000\000\243@`\334\331\257\000\000\247\200\240\334\331\257\000\000\253\300\340\334\331\257\000\000\257\000\000\330\331\263\000\000\263\000\000\330\331\265\000\000\265\005\000\306\2779@J\314\243?\347|\004\000\306\2771@J\314\243O\307|\003\000\306\277)@J\314\243W\247|\002\000\306\277!@J\314\243_\207|\001\000\306\277\031@J\314\263?g|\021@J\314\263OG|\t@J\314\263W'|\001@J\314\263_\007|9@J\314\245C\347|1@J\314\245S\307|)@J\314\245[\247|!@J\314\245c\207|\000\000\306\277\031@J\314\265Cg|\021@J\314\265SG|\t@J\314\265['|\001@J\314\265c\007|\254\001\244\277\004\201\f\277\353\001\200\276\003\377\030\230\000H\000\0009\013r~\236\377\210\277\001\000 \312\030,\237\2402\013d~\030.UK;\013v~\001\002B\177|\000\005\354\242\000\f\000\237\000\000\000\000\000\300\277\210>AK:\013t~<\013x~4\013h~*\013T~|\000\005\354\243\000\f\000\240\000\000\000\000\000\300\277\220>AK1\013b~3\013f~)\013R~+\013V~|\000\005\354\244\000\f\000\240\000\000\000\000\000\300\277\230>AK,\013X~\"\013D~#\013F~$\013H~|\000\005\354\245\000\f\000\240\000\000\000\000\000\300\277\240>AK!\013B~\031\0132~\032\0134~\033\0136~|\000\005\354\246\000\f\000\240\000\000\000\000\000\300\277\250>AK\034\0138~\022\013$~\021\013\"~\024\013(~|\000\005\354\247\000\f\000\240\000\000\000\000\000\300\277\260>AK\023\013&~\n\013\024~\f\013\030~\t\013\022~|\000\005\354\250\000\f\000\240\000\000\000\000\000\300\277\270>AK\013\013\026~\002\013\004~\001\013\002~\003\013\006~|\000\005\354\251\000\f\000\240\000\000\000\000\000\300\277\377TAK\000\004\000\000\004\013\b~|\000\005\354\253\000\f\000\240\000\000\000\000\000\310\277\253E\307\310\253G\255\254\021\001\207\277\254s\352V\253I\301\310\255u\2069\261\000\207\277\253K\301\3109wv:\253Ms\020=\013v~9w\006\310\253Q9f?\013v~\301\000\207\2779w \310\377T9X\200\004\000\000:y\334V\253Ou\020>\013x~:y\006\310\253S;_@\013x~\301\000\207\277:y\020\310\001\000:Q|\000\005\354;\000\f\0009\000\000\000\000\000\310\277;G{\020=e\006\310;K3\205\241\000\207\2772i\006\310;O3m6\013h~2i\274V;Se\0208\013h~\221\000\207\277;E\301\3102iN<<c\006\310;I1\204\001\002d~\242\000\207\2771g\006\310;M1v5\013f~1g\312V;Qc\0207\013f~\321\000\207\2771g\256V\377TcJ\000\005\000\000|\000\005\3543\000\f\0001\000\000\000\000\000\310\2773E\307\3103G544S\000\3105U\202\2023I\307\3103K+)\021\001\207\277)W\350V*Y\006\3103M)l3OU\020-\013V~.\013X~A\001\207\277)W\000\310*Y\\d3Q\307\3103S+)/\013V~0\013X~)W\254V\322\000\207\277*Y \310\377T)N\200\005\000\000\001\002T~|\000\005\354+\000\f\000)\000\000\000\000\000\310\277+E\307\310+G-,,C\000\310-E\200\200+I\307\310+K#!\021\001\207\277!G\346V\"I\326V+M\307\310+O#!%\013F~&\013H~A\001\207\277!G\000\310\"I\\c+Q\307\310+S#!'\013F~(\013H~!G\252V\002\000\207\277\"I\232V\001\000 \312\377>!\"\200\000\000\000|\000\005\354#\000\f\000!\000\000\000\000\000\300\277\377>CJ\210\000\000\000|\000\005\354$\000\f\000!\000\000\000\000\000\300\277\377>CJ\220\000\000\000|\000\005\354%\000\f\000!\000\000\000\000\000\300\277\377>CJ\230\000\000\000|\000\005\354&\000\f\000!\000\000\000\000\000\300\277\377>CJ\240\000\000\000|\000\005\354'\000\f\000!\000\000\000\000\000\300\277\377>CJ\250\000\000\000|\000\005\354(\000\f\000!\000\000\000\000\000\300\277\377>CJ\260\000\000\000|\000\005\354+\000\f\000!\000\000\000\000\000\300\277\0300CJ|\000\005\354!\000\f\000!\000\000\000\000\000\300\277|\000\005\354\"\000\f\000\240\000\000\000\000\000\310\277\"G\306\310\"I,,\241\000\207\277,3\000\310-5~~\"K\306\310\"M\032\031\0329\324V\"Q4\020\036\0138~4\002\207\277\0317\344V\"O2\020\035\0136~\0329\266V\"C4\020 \0138~\264\000\207\277\0317\304V\"W2\020\037\0136~\0317\250V|\000\005\354\031\000\f\0009\000\000\000\000\000\300\277\0329\020\310\200\0008L\001\000\207\2779\001\020\3129\001::9\001\020\3129\001<<9\001\020\3129\001>>9\001\020\3129\0012@9\001\020\3129\001449\001\020\3129\001669\003p~9\001\020\3129\001,,9\001\020\3129\001..9\003`~9\003D~9\001\020\3129\001\034\0349\001\020\3129\001\036\0369\003@~\000\000\306\277\031I6\020\301\001\207\277\033%\372V\031M$\020\031G4\0209\0036~\022)\322VS\002\207\277\032#\370V\031K\306\310\031Q\022\021\026\013(~9\0034~9\003,~\021'\342V\031O\"\020\025\013&~9\003*~\002\000\207\277\021'\302V\031W\"\020\027\013&~\022)\264V\030\013(~9\001\020\3129\001\030\027\004\000\207\277\021'\246V|\000\005\354\021\000\f\0001\000\000\000\000\000\300\2779\001\020\3129\00121\000\000\306\277\021I&\020\241\000\207\277\023\025\364V\021M\024\0209\001\000\312\n\031h\023\021Q\024\020\016\013\030~\031C$\0209\001\020\3129\001\016\031\223\001\207\277\n\031\262V\022)\006\310\021G\022K\021C\024\020\020\013\030~9\003(~\224\001\207\2779\001\000\312\022\023x\020\021K\300\310\n\031J\t9\003$~9\003\030~C\001\207\277\t\027\336V\021O\022\020\r\013\026~9\003\032~\t\027\300V\021W\022\020\017\013\026~9\003\"~9\003\036~\003\000\207\277\t\027\244V|\000\005\354\t\000\f\000)\000\000\000\000\000\300\2779\001\020\3129\001*)\000\000\306\277\tG\306\310\tI\n\n9\001\020\3129\001$#\302\001\207\277\n\003\000\310\013\005\210x\tK\306\310\tM\002\0019\001\020\3129\001&%9\001\020\3129\001\n\n\001\007\000\310\002\t\206\210\tO\306\310\tQ\002\001\005\013\006~\006\013\b~9\001\020\3129\001('9\003\n~\024\002\207\277\001\007\366V\tW\300\310\002\tp\001\tC\004\020\007\013\006~\b\013\b~9\003V~9\003B~9\003\022~\004\000\207\277\001\007\000\310\002\tPg9\001\020\3129\001\002\0019\001\020\3129\001\004\0039\001\020\3129\001\006\0069\003\020~~\026j\221\236\377\210\277\250\375\244\277\n\201\006\277\200\377\000\230\000\020\000\000\002\377\001\230\0000\000\000\004\202\026\201\236\377\210\277\000\032?K\026\f\004\277\001\032AK\301\200\026\230\001\000\300\277\000\000|\333\237A\000\000\000\000\300\277\000\000|\333\240E\000\000\027\026\026\213\t\000\207\277~\026j\221\236\377\210\277\222\375\244\277\004\201\f\277\377\003\000\230\000H\000\000\236\377\210\277\000\034\203J\000\004<\330A\236\235\000\212\375\240\277\213\001\020\312\214\001BA\353\001\200\276\377\024\207\026\000\005\000\000\236\377\210\277\262\001\207\277\001\000\"\312\202\202D\213\001\000$\312\210\204B\215\001\002\036\177C\000U\326\200\206\022\005\001\002\210~\204\000,3\013s\000\226\001\002>\177C\000\013\326\377\204\016\005P\000\000\000\001\002\214~\001\000\020\312\001\000\220H\003\000\207\277\001\000 \312\377\206\214\223@\001\000\000\377\206\212JP\000\000\000\377\206\024K\360\000\000\000\377\206\216J\240\000\000\000\001\000 \312\377\206\216\225\220\001\000\000\001\000 \312\377\206\220\227\340\001\000\000\001\000 \312\377\206\222\2330\002\000\000\000\000\300\277|\200\006\354\000\000\214:C\000\000\000\000\000\301\277|\200\006\354\000\000\fCE\000\000\000\000\000\301\277|\200\006\354\000\000\214;G\000\000\000\000\000\301\277|\200\006\354\000\000\f7\212\000\000\000\000\000\301\277|\200\006\354\000\000\f3\214\000\000\000\000\000\301\277|\200\006\354\000\000\214/\216\000\000\000\000\000\301\277|\200\006\354\000\000\f,\220\000\000\000\000\000\301\277|\200\006\354\000\000\214(\222\000\000\000\000\000\311\277\301N\200\276B\000F\326\226\005\001\002\236\377\210\277\000u\002\201\211,Q1\236\377\210\277\002\237\003\206\001\002:\177\224\000\013\326\377\202\n\005P\000\000\000\200\002\204~\236\377\210\277\002\220\204\204\001\002B\177\236\377\210\277\006\004\204\251\207\000\0006\202\202\202>\236\377\210\277\226\000\000\327\004P\003\002\237\361\210\277\231| \325\005\000\001\000\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\232\000\f\000\224\000\000\000\000\000\300\277\230j\000\327\226\203\002\002\235\377\210\277\231| \325\231\205\252\001\001\000 \312\377(\227\243\000\005\000\000\001\002J\177\000\000\306\277|\200\006\356\000\000\000M\230\000\000\000|\000\005\354\234\000\f\000\226\000\000\000\000\000\300\277\377(5K\000\n\000\000\000\000\306\277|\200\006\356\000\000\000N\230\000\200\000|\000\005\354\236\000\f\000\232\000\000\000\000\000\300\277\377(9K\000\017\000\000\000\000\306\277|\200\006\356\000\000\000O\230\200\000\000|\000\005\354\240\000\f\000\234\000\000\000\000\000\300\277\377(=K\000\024\000\000\000\000\306\277|\200\006\356\000\000\000P\230\200\200\000|\000\005\354\242\000\f\000\236\000\000\000\000\000\300\277\377(AK\000\031\000\000\000\000\306\277|\200\006\356\000\000\000Q\230\000\001\000|\000\005\354\244\000\f\000\240\000\000\000\000\000\300\277\377(EK\000\036\000\000\000\000\306\277|\200\006\356\000\000\000R\230\000\201\000|\000\005\354\246\000\f\000\242\000\000\000\000\000\300\277\377(IK\000#\000\000\000\000\306\277|\200\006\356\000\000\000S\230\200\001\000|\000\005\354\246\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200\201\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\fBC\000\000\000\000\000\301\277|\200\006\354\000\000\214BE\000\000\000\000\000\301\277|\200\006\354\000\000\f;G\000\000\000\000\000\301\277|\200\006\354\000\000\2146\212\000\000\000\000\000\301\277|\200\006\354\000\000\2142\214\000\000\000\000\000\301\277|\200\006\354\000\000\f/\216\000\000\000\000\000\301\277|\200\006\354\000\000\214+\220\000\000\000\000\000\301\277|\200\006\354\000\000\214'\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\246\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000 \000|\000\005\354\246\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000\240\000|\000\005\354\246\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200 \000|\000\005\354\246\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200\240\000|\000\005\354\246\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000!\000|\000\005\354\246\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000\241\000|\000\005\354\246\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200!\000|\000\005\354\246\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200\241\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\fAC\000\000\000\000\000\301\277|\200\006\354\000\000\214AE\000\000\000\000\000\301\277|\200\006\354\000\000\f:G\000\000\000\000\000\301\277|\200\006\354\000\000\f6\212\000\000\000\000\000\301\277|\200\006\354\000\000\f2\214\000\000\000\000\000\301\277|\200\006\354\000\000\214.\216\000\000\000\000\000\301\277|\200\006\354\000\000\f+\220\000\000\000\000\000\301\277|\200\006\354\000\000\f'\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\246\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000@\000|\000\005\354\246\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000\300\000|\000\005\354\246\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200@\000|\000\005\354\246\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200\300\000|\000\005\354\246\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000A\000|\000\005\354\246\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\000\301\000|\000\005\354\246\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200A\000|\000\005\354\246\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000S\230\200\301\000\377PM9\000`\000\000\377PQ9\000\340\000\000\262\001\207\277\246\000\000\327\004L\003\002\237\361\210\277\247| \325\005\000\001\000\250\000\000\327\004P\003\002\243\001\207\277\246j\000\327\246\203\002\002\235\377\210\277\247| \325\247\205\252\001\237\361\210\277\252| \325\005\000\001\000Aj\000\327\250\203\002\002\235\377\210\277\002\000\207\277B| \325\252\205\252\001\201\000\224|\235\377\210\2779ur\0021eb\002)UR\002!ER\312\204\0133!\021%\"\002\03152\002\000\000J\324\202\000\002\002\202\007S\312\t\025\b*\200\003E\002~\377R\312\001\005\000\032|\373$\002\237\361\210\277\n\000\001\3259w\002\000y\365r\0021\000\001\3251g\002\000x\023\005\0022\000\001\3252\355\002\000u\ru\002\203\000\224|\031\000\001\325\0317\002\000\001\000J\324\204\000\002\002)\000\001\325)W\002\000*\000\001\325*\351\002\000\235\377\210\2772\3336\002:\000\001\325:\357\002\000!\000\001\325!G\002\000\"\000\001\325\"\347\002\000\032\000\001\325\032\345\002\000\021\000\001\325\021'\002\000\t\000\001\325\t\027\002\000\013\000\001\3259\337\002\000\001\000\001\325\001\007\002\000:\335\006\002\022\000\001\325\022\343\002\0001i&\002\002\000\001\325\002\021\003\000)YR\312\"\327\"#*\331R\002!IR\312\032\325\032!\0319R\312\022\323\022\031\021)R\312\002\017\003\021\013\321\026\002\001\t\002\002\237\361\210\277\003\000\001\325\003\315\006\000\t\031\022\002\f\000\001\325\033\313\006\000\ny\024\002\205\000\224|\001\000\001\325\001\013\006\000\033\000\001\325!K\006\000\034\000\001\325\"\307\006\000\031\000\001\325\031;\006\000\235\377\210\277\f\275\n\002\004\000\001\325\n{\006\000\n\000\001\325\023k\006\000\023\000\001\325#[\006\000\032\000\001\325\032\305\006\000\021\000\001\325\021+\006\000\004}\b\002\024\000\001\325)\311\006\000\022\000\001\325\022\303\006\000\t\000\001\325\t\033\006\000\002\000\001\325\002\367\006\000\000\000J\324\206\000\002\002\024\273\030\002\013\000\001\325\013\301\006\000\001\000J\324\207\000\002\002\003\277\000\002\237\361\210\277\004\000\001\325\004\177\002\000\nm\006\002\023]\024\002\033MR\312\002\341\002\r\000\000\001\325\000\261\002\000\034\271R\312\031=\024\023\032\267*\002\021-\"\002\022\265R\312\013\263\n\022\t\035\022\002\001\r\002\002\003\000\001\325\003o\002\000\004\000\001\325\004\201\006\000\000\000\001\325\000\243\006\000\005\000\001\325\005\257\002\000\006\000\001\325\n_\002\000\n\000\001\325\f\255\002\000\f\000\001\325\rO\002\000\r\000\001\325\023\253\002\000\016\000\001\325\024?\002\000\023\000\001\325\025\251\002\000\021\000\001\325\021/\002\000\022\000\001\325\022\247\002\000\t\000\001\325\t\037\002\000\013\000\001\325\013\245\002\000\001\000\001\325\001\017\002\000\002\000\001\325\002\317\002\000\003\000\001\325\003q\006\000\000\000\000\327\004\001\002\002\237\361\210\277\004| \325\200\000\001\000\005\000\001\325\005\237\006\000\243\001\207\277\000j\000\327\000\007\002\002\235\377\210\277\003| \325\200\b\252\001\006\000\001\325\006a\006\000\243\001\207\277\000j\000\327\000\013\002\002\235\377\210\277\003| \325\200\006\252\001\004\000\001\325\n\235\006\000\243\001\207\277\000j\000\327\000\r\002\002\235\377\210\277\003| \325\200\006\252\001\005\000\001\325\fQ\006\000\243\001\207\277\000j\000\327\000\t\002\002\235\377\210\277\003| \325\200\006\252\001\006\000\001\325\r\233\006\000\243\001\207\277\000j\000\327\000\013\002\002\235\377\210\277\003| \325\200\006\252\001\004\000\001\325\016A\006\000\243\001\207\277\000j\000\327\000\r\002\002\235\377\210\277\003| \325\200\006\252\001\005\000\001\325\023\231\006\000\243\001\207\277\000j\000\327\000\t\002\002\235\377\210\277\003| \325\200\006\252\001\006\000\001\325\0211\006\000\243\001\207\277\000j\000\327\000\013\002\002\235\377\210\277\003| \325\200\006\252\001\004\000\001\325\022\227\006\000\243\001\207\277\000j\000\327\000\r\002\002\235\377\210\277\003| \325\200\006\252\001\005\000\001\325\t!\006\000\243\001\207\277\000j\000\327\000\t\002\002\235\377\210\277\003| \325\200\006\252\001\006\000\001\325\013\225\006\000\243\001\207\277\000j\000\327\000\013\002\002\235\377\210\277\003| \325\200\006\252\001\001\000\001\325\001\021\006\000\243\001\207\277\000j\000\327\000\r\002\002\235\377\210\277\003| \325\200\006\252\001\002\000\001\325\002\241\006\000\243\001\207\277\000j\000\327\000\003\002\002\235\377\210\277\001| \325\200\006\252\001\002\213\200\204\000j\000\327\000\005\002\002\235\377\210\277\001| \325\200\002\252\001\236\377\210\277\b\000\200\251\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\f@C\000\000\000\000\000\301\277|\200\006\354\000\000\214@E\000\000\000\000\000\301\277|\200\006\354\000\000\2149G\000\000\000\000\000\301\277|\200\006\354\000\000\2145\212\000\000\000\000\000\301\277|\200\006\354\000\000\2141\214\000\000\000\000\000\301\277|\200\006\354\000\000\f.\216\000\000\000\000\000\301\277|\200\006\354\000\000\214*\220\000\000\000\000\000\301\277|\200\006\354\000\000\214&\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\251\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\200T\246\000\000\000|\000\005\354\251\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\200TA\000\000\000|\000\005\354\250\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\000T\246\200\000\000|\000\005\354\250\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\000TA\200\000\000|\000\005\354\250\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\000T\246\000\001\000|\000\005\354\250\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\000TA\000\001\000|\000\005\354\250\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\000T\246\200\001\000|\000\005\354\250\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000TA\200\001\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\f?C\000\000\000\000\000\301\277|\200\006\354\000\000\214?E\000\000\000\000\000\301\277|\200\006\354\000\000\f9G\000\000\000\000\000\301\277|\200\006\354\000\000\f5\212\000\000\000\000\000\301\277|\200\006\354\000\000\f1\214\000\000\000\000\000\301\277|\200\006\354\000\000\214-\216\000\000\000\000\000\301\277|\200\006\354\000\000\f*\220\000\000\000\000\000\301\277|\200\006\354\000\000\f&\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\250\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\000\000|\000\005\354\250\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\200\000|\000\005\354\250\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\000\000|\000\005\354\250\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\200\000|\000\005\354\250\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\001\000|\000\005\354\250\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\201\000|\000\005\354\250\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\001\000|\000\005\354\250\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\201\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\f>C\000\000\000\000\000\301\277|\200\006\354\000\000\214>E\000\000\000\000\000\301\277|\200\006\354\000\000\2148G\000\000\000\000\000\301\277|\200\006\354\000\000\2144\212\000\000\000\000\000\301\277|\200\006\354\000\000\2140\214\000\000\000\000\000\301\277|\200\006\354\000\000\f-\216\000\000\000\000\000\301\277|\200\006\354\000\000\214)\220\000\000\000\000\000\301\277|\200\006\354\000\000\214%\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\250\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@ \000|\000\005\354\250\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\240\000|\000\005\354\250\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300 \000|\000\005\354\250\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\240\000|\000\005\354\250\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@!\000|\000\005\354\250\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\241\000|\000\005\354\250\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300!\000|\000\005\354\250\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\241\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\214<C\000\000\000\000\000\301\277|\200\006\354\000\000\f=E\000\000\000\000\000\301\277|\200\006\354\000\000\2147G\000\000\000\000\000\301\277|\200\006\354\000\000\f4\212\000\000\000\000\000\301\277|\200\006\354\000\000\f0\214\000\000\000\000\000\301\277|\200\006\354\000\000\214,\216\000\000\000\000\000\301\277|\200\006\354\000\000\f)\220\000\000\000\000\000\301\277|\200\006\354\000\000\f%\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354\250\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@@\000|\000\005\354\250\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\300\000|\000\005\354\250\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300@\000|\000\005\354\250\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\300\000|\000\005\354\250\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@A\000|\000\005\354\250\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230@\301\000|\000\005\354\250\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300A\000|\000\005\354\250\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\000T\230\300\301\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\300\277|\200\006\354\000\000\f<C\000\000\000\000\000\301\277|\200\006\354\000\000\214DE\000\000\000\000\000\301\277|\200\006\354\000\000\fDG\000\000\000\000\000\301\277|\200\006\354\000\000\214C\212\000\000\000\000\000\301\277|\200\006\354\000\000\214=\214\000\000\000\000\000\301\277|\200\006\354\000\000\f8\216\000\000\000\000\000\301\277|\200\006\354\000\000\2143\220\000\000\000\000\000\301\277|\200\006\354\000\000\f(\222\000\000\000\000\000\311\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\000\005\354C\000\f\000\224\000\000\000\000\000\310\277|\200\006\356\000\000\200!\246@\000\000|\000\005\354C\000\f\000\226\000\000\000\000\000\310\277|\200\006\356\000\000\200!A@\000\000|\000\005\354C\000\f\000\232\000\000\000\000\000\310\277|\200\006\356\000\000\200!\246\300\000\000|\000\005\354C\000\f\000\234\000\000\000\000\000\310\277|\200\006\356\000\000\200!A\300\000\000|\000\005\354C\000\f\000\236\000\000\000\000\000\310\277|\200\006\356\000\000\200!\246@\001\000|\000\005\354C\000\f\000\240\000\000\000\000\000\310\277|\200\006\356\000\000\200!A@\001\000|\000\005\354C\000\f\000\242\000\000\000\000\000\310\277|\200\006\356\000\000\200!\246\300\001\000|\000\005\354C\000\f\000\244\000\000\000\000\000\310\277|\200\006\356\000\000\200!A\300\001\000\000\000\301\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\300\006\356\000\000\000\000I\000\000\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\006\000\000\000\000\000\000\000\250\032\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\340\034\000\000\000\000\000\000\n\000\000\000\000\000\000\000'\002\000\000\000\000\000\000\365\376\377o\000\000\000\000\370\033\000\000\000\000\000\000\004\000\000\000\000\000\000\000h\034\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 23.0.0 (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000)\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\002\000\000\000\002\t\000\270\227\000\000\000\000\000\000\000\000\000\000\000\000\000\000Z\000\000\000\022\003\b\000\0002\000\000\000\000\000\000l\005\000\000\000\000\000\000\204\000\000\000\021\003\006\000@\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\261\000\000\000\022\003\b\000\0008\000\000\000\000\000\000\f\006\000\000\000\000\000\000\333\000\000\000\021\003\006\000\200\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\b\001\000\000\022\003\b\000\000?\000\000\000\000\000\0004\b\000\000\000\000\000\0002\001\000\000\021\003\006\000\300\037\000\000\000\000\000\000@\000\000\000\000\000\000\000_\001\000\000\022\003\b\000\000H\000\000\000\000\000\000\344\007\000\000\000\000\000\000\211\001\000\000\021\003\006\000\000 \000\000\000\000\000\000@\000\000\000\000\000\000\000\266\001\000\000\022\003\b\000\000P\000\000\000\000\000\000\254\024\000\000\000\000\000\000\340\001\000\000\021\003\006\000@ \000\000\000\000\000\000@\000\000\000\000\000\000\000\r\002\000\000\022\003\b\000\000e\000\000\000\000\000\000\264\"\000\000\000\000\000\0007\002\000\000\021\003\006\000\200 \000\000\000\000\000\000@\000\000\000\000\000\000\000d\002\000\000\021\000\013\000(\250\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.eh_frame\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000amdgpu.max_num_named_barrier\000_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.kd\000_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii\000_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.kd\000__hip_cuid_9a7978dd31cac309\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000l\030\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\250\032\000\000\000\000\000\000\250\032\000\000\000\000\000\000P\001\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\370\033\000\000\000\000\000\000\370\033\000\000\000\000\000\000p\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000h\034\000\000\000\000\000\000h\034\000\000\000\000\000\000x\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\340\034\000\000\000\000\000\000\340\034\000\000\000\000\000\000'\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000@\037\000\000\000\000\000\000@\037\000\000\000\000\000\000\200\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\300 \000\000\000\000\000\000\300 \000\000\000\000\000\000\304\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0009\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\0002\000\000\000\000\000\000\000\"\000\000\000\000\000\000\264U\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000?\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\270\227\000\000\000\000\000\000\270w\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000H\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000(\230\000\000\000\000\000\000(x\000\000\000\000\000\000\330\007\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000W\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000(\250\000\000\000\000\000\000(x\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\\\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000(x\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000k\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000(x\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000(x\000\000\000\000\000\000\334\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\211\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\by\000\000\000\000\000\000\310\001\000\000\000\000\000\000\021\000\000\000\006\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320z\000\000\000\000\000\000\243\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\233\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000s{\000\000\000\000\000\000\211\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_7, 37504

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_7
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_9a7978dd31cac309,@object # @__hip_gpubin_handle_9a7978dd31cac309
	.local	__hip_gpubin_handle_9a7978dd31cac309
	.comm	__hip_gpubin_handle_9a7978dd31cac309,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_9a7978dd31cac309,@object # @__hip_cuid_9a7978dd31cac309
	.bss
	.globl	__hip_cuid_9a7978dd31cac309
__hip_cuid_9a7978dd31cac309:
	.byte	0                               # 0x0
	.size	__hip_cuid_9a7978dd31cac309, 1

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
	.addrsig_sym _Z25__device_stub__iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _Z25__device_stub__iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z25__device_stub__iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z25__device_stub__iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z25__device_stub__iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z25__device_stub__iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym _Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.addrsig_sym .L__unnamed_7
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_9a7978dd31cac309
