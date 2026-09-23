	.att_syntax
	.file	"wmma_rate.hip"
	.text
	.globl	__device_stub__peak_f16         # -- Begin function __device_stub__peak_f16
	.prefalign	4, .Lfunc_end0, nop
	.type	__device_stub__peak_f16,@function
__device_stub__peak_f16:                # @__device_stub__peak_f16
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movl	%esi, 4(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	peak_f16@GOTPCREL(%rip), %rdi
	leaq	64(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end0:
	.size	__device_stub__peak_f16, .Lfunc_end0-__device_stub__peak_f16
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__peak_iu8         # -- Begin function __device_stub__peak_iu8
	.prefalign	4, .Lfunc_end1, nop
	.type	__device_stub__peak_iu8,@function
__device_stub__peak_iu8:                # @__device_stub__peak_iu8
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movl	%esi, 4(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	peak_iu8@GOTPCREL(%rip), %rdi
	leaq	64(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end1:
	.size	__device_stub__peak_iu8, .Lfunc_end1-__device_stub__peak_iu8
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__peak_fp8         # -- Begin function __device_stub__peak_fp8
	.prefalign	4, .Lfunc_end2, nop
	.type	__device_stub__peak_fp8,@function
__device_stub__peak_fp8:                # @__device_stub__peak_fp8
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movl	%esi, 4(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	peak_fp8@GOTPCREL(%rip), %rdi
	leaq	64(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end2:
	.size	__device_stub__peak_fp8, .Lfunc_end2-__device_stub__peak_fp8
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__peak_iu4         # -- Begin function __device_stub__peak_iu4
	.prefalign	4, .Lfunc_end3, nop
	.type	__device_stub__peak_iu4,@function
__device_stub__peak_iu4:                # @__device_stub__peak_iu4
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movl	%esi, 4(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	peak_iu4@GOTPCREL(%rip), %rdi
	leaq	64(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end3:
	.size	__device_stub__peak_iu4, .Lfunc_end3-__device_stub__peak_iu4
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI4_0:
	.quad	0x4020000000000000              # double 8
.LCPI4_1:
	.quad	0x40c3880000000000              # double 1.0E+4
.LCPI4_2:
	.quad	0x41cdcd6500000000              # double 1.0E+9
.LCPI4_3:
	.quad	0x4059000000000000              # double 100
	.text
	.globl	main
	.prefalign	4, .Lfunc_end4, nop
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
	subq	$1688, %rsp                     # imm = 0x698
	.cfi_def_cfa_offset 1744
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	cmpl	$1, %edi
	je	.LBB4_4
# %bb.1:
	movq	%rsi, %rbx
	cmpl	$2, %edi
	jne	.LBB4_3
# %bb.2:
	movq	8(%rbx), %rdi
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	callq	strcmp@PLT
	movb	$1, %bpl
	testl	%eax, %eax
	je	.LBB4_5
.LBB4_3:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	movq	(%rbx), %rdx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.1(%rip), %rsi
	jmp	.LBB4_94
.LBB4_4:
	xorl	%ebp, %ebp
.LBB4_5:
	movl	$0, 68(%rsp)
	movl	$0, 20(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	216(%rsp), %rdi
	movl	$1472, %edx                     # imm = 0x5C0
	xorl	%esi, %esi
	callq	memset@PLT
	.cfi_escape 0x2e, 0x00
	leaq	68(%rsp), %rdi
	callq	hipGetDevice@PLT
	testl	%eax, %eax
	jne	.LBB4_123
# %bb.6:
	movl	68(%rsp), %esi
	.cfi_escape 0x2e, 0x00
	leaq	216(%rsp), %rdi
	callq	hipGetDevicePropertiesR0600@PLT
	testl	%eax, %eax
	jne	.LBB4_124
# %bb.7:
	movl	68(%rsp), %edx
	.cfi_escape 0x2e, 0x00
	leaq	20(%rsp), %rdi
	movl	$63, %esi
	callq	hipDeviceGetAttribute@PLT
	testl	%eax, %eax
	jne	.LBB4_125
# %bb.8:
	movl	$829974119, %eax                # imm = 0x31786667
	xorl	1376(%rsp), %eax
	movl	$825242161, %ecx                # imm = 0x31303231
	xorl	1379(%rsp), %ecx
	orl	%eax, %ecx
	jne	.LBB4_93
# %bb.9:
	leaq	1376(%rsp), %rdx
	movl	20(%rsp), %ecx
	leaq	.L.str.10(%rip), %rax
	leaq	.L.str.11(%rip), %r8
	testb	%bpl, %bpl
	cmovneq	%rax, %r8
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.9(%rip), %rdi
	leaq	216(%rsp), %rsi
	xorl	%eax, %eax
	callq	printf@PLT
	movq	$0, 56(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	movl	$8, %esi
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB4_126
# %bb.10:
	movq	56(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	movl	$8, %edx
	xorl	%esi, %esi
	callq	hipMemset@PLT
	testl	%eax, %eax
	jne	.LBB4_127
# %bb.11:
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	xorps	%xmm0, %xmm0
	movaps	%xmm0, 144(%rsp)
	movaps	%xmm0, 160(%rsp)
	movaps	%xmm0, 176(%rsp)
	movl	$1, 144(%rsp)
	movq	$65536, 152(%rsp)               # imm = 0x10000
	movl	$3, 160(%rsp)
	leaq	.L__const.main.backward(%rip), %rax
	leaq	.L__const.main.forward(%rip), %rcx
	testb	%bpl, %bpl
	cmovneq	%rax, %rcx
	movq	%rcx, 120(%rsp)                 # 8-byte Spill
	movq	peak_iu4@GOTPCREL(%rip), %rsi
	movq	peak_f16@GOTPCREL(%rip), %rax
	movq	%rax, %r12
	cmovneq	%rsi, %r12
	movq	$21845, 168(%rsp)               # imm = 0x5555
	movq	peak_fp8@GOTPCREL(%rip), %r15
	movq	peak_iu8@GOTPCREL(%rip), %rcx
	movq	%rcx, %r13
	cmovneq	%r15, %r13
	cmovneq	%rcx, %r15
	movl	$8, 176(%rsp)
	cmovneq	%rax, %rsi
	movq	%rsi, 8(%rsp)                   # 8-byte Spill
	xorl	%ebx, %ebx
	leaq	256(%rdx), %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	%r13, 48(%rsp)                  # 8-byte Spill
	.p2align	4
.LBB4_12:                               # %.preheader79
                                        # =>This Inner Loop Header: Depth=1
	movq	152(%rsp,%rbx), %rbp
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$8, %esi
	movl	%ebp, %edx
	callq	hipFuncSetAttribute@PLT
	testl	%eax, %eax
	jne	.LBB4_122
# %bb.13:                               #   in Loop: Header=BB4_12 Depth=1
	movl	144(%rsp,%rbx), %r14d
	movl	20(%rsp), %esi
	imull	%r14d, %esi
	movq	56(%rsp), %rax
	movq	%rax, 24(%rsp)
	movl	$10000, (%rsp)                  # imm = 0x2710
	leaq	24(%rsp), %rax
	movq	%rax, 32(%rsp)
	movq	%rsp, %rax
	movq	%rax, 40(%rsp)
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rsi
	.cfi_escape 0x2e, 0x10
	movq	%r12, %rdi
	movl	$1, %edx
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movl	$1, %r8d
	leaq	32(%rsp), %r9
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	testl	%eax, %eax
	jne	.LBB4_117
# %bb.14:                               # %_ZL6launchPFvPyiEimS_i.exit
                                        #   in Loop: Header=BB4_12 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movl	$8, %esi
	movl	%ebp, %edx
	callq	hipFuncSetAttribute@PLT
	testl	%eax, %eax
	jne	.LBB4_122
# %bb.15:                               #   in Loop: Header=BB4_12 Depth=1
	movl	20(%rsp), %esi
	imull	%r14d, %esi
	movq	56(%rsp), %rax
	movq	%rax, 24(%rsp)
	movl	$10000, (%rsp)                  # imm = 0x2710
	leaq	24(%rsp), %rax
	movq	%rax, 32(%rsp)
	movq	%rsp, %rax
	movq	%rax, 40(%rsp)
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rsi
	.cfi_escape 0x2e, 0x10
	movq	%r13, %rdi
	movl	$1, %edx
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movl	$1, %r8d
	leaq	32(%rsp), %r9
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	testl	%eax, %eax
	jne	.LBB4_117
# %bb.16:                               # %_ZL6launchPFvPyiEimS_i.exit.1
                                        #   in Loop: Header=BB4_12 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$8, %esi
	movl	%ebp, %edx
	callq	hipFuncSetAttribute@PLT
	testl	%eax, %eax
	jne	.LBB4_122
# %bb.17:                               #   in Loop: Header=BB4_12 Depth=1
	movq	%r15, %rdi
	movq	%r12, %r13
	movl	20(%rsp), %esi
	imull	%r14d, %esi
	movq	56(%rsp), %rax
	movq	%rax, 24(%rsp)
	movl	$10000, (%rsp)                  # imm = 0x2710
	leaq	24(%rsp), %rax
	movq	%rax, 32(%rsp)
	movq	%rsp, %rax
	movq	%rax, 40(%rsp)
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rsi
	.cfi_escape 0x2e, 0x10
	movl	$1, %edx
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movl	$1, %r8d
	leaq	32(%rsp), %r9
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	testl	%eax, %eax
	jne	.LBB4_117
# %bb.18:                               # %_ZL6launchPFvPyiEimS_i.exit.2
                                        #   in Loop: Header=BB4_12 Depth=1
	.cfi_escape 0x2e, 0x00
	movq	8(%rsp), %r12                   # 8-byte Reload
	movq	%r12, %rdi
	movl	$8, %esi
	movl	%ebp, %edx
	callq	hipFuncSetAttribute@PLT
	testl	%eax, %eax
	jne	.LBB4_122
# %bb.19:                               #   in Loop: Header=BB4_12 Depth=1
	imull	20(%rsp), %r14d
	movq	56(%rsp), %rax
	movq	%rax, 24(%rsp)
	movl	$10000, (%rsp)                  # imm = 0x2710
	leaq	24(%rsp), %rax
	movq	%rax, 32(%rsp)
	movq	%rsp, %rax
	movq	%rax, 40(%rsp)
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %r14
	.cfi_escape 0x2e, 0x10
	movq	%r12, %rdi
	movq	%r14, %rsi
	movl	$1, %edx
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movl	$1, %r8d
	leaq	32(%rsp), %r9
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	testl	%eax, %eax
	jne	.LBB4_117
# %bb.20:                               # %_ZL6launchPFvPyiEimS_i.exit.3
                                        #   in Loop: Header=BB4_12 Depth=1
	addq	$16, %rbx
	cmpq	$48, %rbx
	movq	%rsp, %r14
	movq	%r13, %r12
	movq	48(%rsp), %r13                  # 8-byte Reload
	jne	.LBB4_12
# %bb.21:
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB4_128
# %bb.22:                               # %.preheader78
	movq	$0, 96(%rsp)                    # 8-byte Folded Spill
	jmp	.LBB4_24
	.p2align	4
.LBB4_23:                               #   in Loop: Header=BB4_24 Depth=1
	movq	96(%rsp), %rcx                  # 8-byte Reload
	addq	$16, %rcx
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	cmpq	$48, %rcx
	je	.LBB4_90
.LBB4_24:                               # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_27 Depth 2
                                        #       Child Loop BB4_34 Depth 3
                                        #       Child Loop BB4_67 Depth 3
                                        #         Child Loop BB4_72 Depth 4
                                        #       Child Loop BB4_79 Depth 3
                                        #         Child Loop BB4_84 Depth 4
                                        #       Child Loop BB4_87 Depth 3
                                        #         Child Loop BB4_89 Depth 4
	xorl	%r15d, %r15d
	jmp	.LBB4_27
	.p2align	4
.LBB4_25:                               #   in Loop: Header=BB4_27 Depth=2
	movq	%rsp, %r14
.LBB4_26:                               # %_ZL8run_cellRK7VariantRK9OccupancyiPy.exit
                                        #   in Loop: Header=BB4_27 Depth=2
	movss	12(%rbx), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	cvtsi2sdl	192(%rsp), %xmm2        # 4-byte Folded Reload
	movsd	.LCPI4_0(%rip), %xmm3           # xmm3 = [8.0E+0,0.0E+0]
	mulsd	%xmm3, %xmm2
	mulsd	.LCPI4_1(%rip), %xmm2
	movq	112(%rsp), %rax                 # 8-byte Reload
	xorps	%xmm1, %xmm1
	cvtsi2sdl	16(%rax), %xmm1
	mulsd	%xmm3, %xmm2
	mulsd	%xmm2, %xmm1
	divsd	%xmm0, %xmm1
	movss	(%r13), %xmm2                   # xmm2 = mem[0],zero,zero,zero
	subss	(%rbx), %xmm2
	cvtss2sd	%xmm2, %xmm2
	divsd	.LCPI4_2(%rip), %xmm1
	mulsd	.LCPI4_3(%rip), %xmm2
	divsd	%xmm0, %xmm2
	movq	(%rax), %rsi
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.33(%rip), %rdi
	movl	104(%rsp), %edx                 # 4-byte Reload
	movq	128(%rsp), %rcx                 # 8-byte Reload
	movl	$7, %r8d
	movb	$3, %al
	callq	printf@PLT
	movq	8(%rsp), %rsi                   # 8-byte Reload
	subq	%rbx, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
	movq	200(%rsp), %r15                 # 8-byte Reload
	incq	%r15
	cmpq	$4, %r15
	je	.LBB4_23
.LBB4_27:                               #   Parent Loop BB4_24 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB4_34 Depth 3
                                        #       Child Loop BB4_67 Depth 3
                                        #         Child Loop BB4_72 Depth 4
                                        #       Child Loop BB4_79 Depth 3
                                        #         Child Loop BB4_84 Depth 4
                                        #       Child Loop BB4_87 Depth 3
                                        #         Child Loop BB4_89 Depth 4
	leaq	(%r15,%r15,2), %rcx
	movl	20(%rsp), %r13d
	movq	56(%rsp), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	96(%rsp), %rax                  # 8-byte Reload
	movl	144(%rsp,%rax), %ebp
	movq	152(%rsp,%rax), %r12
	movq	120(%rsp), %rax                 # 8-byte Reload
	movq	8(%rax,%rcx,8), %rbx
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movl	$8, %esi
	movl	%r12d, %edx
	callq	hipFuncSetAttribute@PLT
	testl	%eax, %eax
	jne	.LBB4_121
# %bb.28:                               #   in Loop: Header=BB4_27 Depth=2
	movl	$0, 84(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	84(%rsp), %rdi
	movq	%rbx, 136(%rsp)                 # 8-byte Spill
	movq	%rbx, %rsi
	movl	$256, %edx                      # imm = 0x100
	movq	%r12, %rcx
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
	testl	%eax, %eax
	jne	.LBB4_120
# %bb.29:                               #   in Loop: Header=BB4_27 Depth=2
	movq	120(%rsp), %rax                 # 8-byte Reload
	leaq	(%r15,%r15,2), %rcx
	leaq	(%rax,%rcx,8), %rax
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	movl	84(%rsp), %r8d
	cmpl	%ebp, %r8d
	jne	.LBB4_119
# %bb.30:                               #   in Loop: Header=BB4_27 Depth=2
	imull	%ebp, %r13d
	movq	48(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 24(%rsp)
	movl	$10000, (%rsp)                  # imm = 0x2710
	leaq	24(%rsp), %rax
	movq	%rax, 32(%rsp)
	movq	%r14, 40(%rsp)
	movabsq	$4294967296, %rax               # imm = 0x100000000
	leaq	(%r13,%rax), %rsi
	.cfi_escape 0x2e, 0x10
	movq	136(%rsp), %rdi                 # 8-byte Reload
	movq	%rsi, 208(%rsp)                 # 8-byte Spill
	movl	$1, %edx
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movl	$1, %r8d
	leaq	32(%rsp), %r9
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	testl	%eax, %eax
	jne	.LBB4_117
# %bb.31:                               # %_ZL6launchPFvPyiEimS_i.exit.i
                                        #   in Loop: Header=BB4_27 Depth=2
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB4_118
# %bb.32:                               # %_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm.exit.i.i
                                        #   in Loop: Header=BB4_27 Depth=2
.Ltmp0:                                 # EH_LABEL
	movq	%r13, 192(%rsp)                 # 8-byte Spill
	movl	%ebp, 104(%rsp)                 # 4-byte Spill
	movq	%r12, 128(%rsp)                 # 8-byte Spill
	movq	%r15, 200(%rsp)                 # 8-byte Spill
	.cfi_escape 0x2e, 0x00
	movl	$28, %edi
	xorl	%ebx, %ebx
	movq	$0, 8(%rsp)                     # 8-byte Folded Spill
	callq	_Znwm@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.33:                               # %_ZNSt6vectorIfSaIfEE7reserveEm.exit.i
                                        #   in Loop: Header=BB4_27 Depth=2
	movq	%rax, %r15
	addq	$28, %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movl	$7, %r12d
	movq	%r15, %rbp
	movq	%rsp, %r14
	.p2align	4
.LBB4_34:                               #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
.Ltmp2:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	hipEventCreate@PLT
.Ltmp3:                                 # EH_LABEL
# %bb.35:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_102
# %bb.36:                               #   in Loop: Header=BB4_34 Depth=3
.Ltmp8:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp9:                                 # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_97
# %bb.38:                               #   in Loop: Header=BB4_34 Depth=3
	movq	(%rsp), %rdi
.Ltmp14:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	movq	8(%rsp), %r13                   # 8-byte Reload
	callq	hipEventRecord@PLT
.Ltmp15:                                # EH_LABEL
# %bb.39:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_108
# %bb.40:                               #   in Loop: Header=BB4_34 Depth=3
	movq	48(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 24(%rsp)
	movl	$10000, 108(%rsp)               # imm = 0x2710
	leaq	24(%rsp), %rax
	movq	%rax, 32(%rsp)
	leaq	108(%rsp), %rax
	movq	%rax, 40(%rsp)
.Ltmp20:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	136(%rsp), %rdi                 # 8-byte Reload
	movq	208(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %edx
	movabsq	$4294967552, %rcx               # imm = 0x100000100
	movl	$1, %r8d
	leaq	32(%rsp), %r9
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	136(%rsp)                       # 8-byte Folded Reload
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp21:                                # EH_LABEL
# %bb.41:                               # %.noexc94.i
                                        #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_99
# %bb.42:                               #   in Loop: Header=BB4_34 Depth=3
	movq	88(%rsp), %rdi
.Ltmp26:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp27:                                # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_106
# %bb.44:                               #   in Loop: Header=BB4_34 Depth=3
	movq	88(%rsp), %rdi
.Ltmp32:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp33:                                # EH_LABEL
# %bb.45:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_95
# %bb.46:                               #   in Loop: Header=BB4_34 Depth=3
	movl	$0, 32(%rsp)
	movq	(%rsp), %rsi
	movq	88(%rsp), %rdx
.Ltmp38:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	32(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp39:                                # EH_LABEL
# %bb.47:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_104
# %bb.48:                               #   in Loop: Header=BB4_34 Depth=3
	cmpq	%r13, %r15
	je	.LBB4_50
# %bb.49:                               #   in Loop: Header=BB4_34 Depth=3
	movss	32(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%r15)
	movq	%rbp, %rbx
	movq	%r15, %r13
	jmp	.LBB4_55
	.p2align	4
.LBB4_50:                               #   in Loop: Header=BB4_34 Depth=3
	movq	%r13, %r14
	subq	%rbp, %r14
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	cmpq	%rax, %r14
	je	.LBB4_115
# %bb.51:                               # %_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB4_34 Depth=3
	movq	%r14, %r15
	sarq	$2, %r15
	cmpq	$1, %r15
	adcq	%r15, %r15
	movabsq	$2305843009213693951, %rax      # imm = 0x1FFFFFFFFFFFFFFF
	cmpq	%rax, %r15
	cmovaeq	%rax, %r15
	leaq	(,%r15,4), %rdi
.Ltmp44:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
.Ltmp45:                                # EH_LABEL
# %bb.52:                               # %.noexc99.i
                                        #   in Loop: Header=BB4_34 Depth=3
	movq	%rax, %rbx
	movss	32(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, (%rax,%r14)
	testq	%r14, %r14
	jle	.LBB4_54
# %bb.53:                               #   in Loop: Header=BB4_34 Depth=3
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movq	%rbp, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
.LBB4_54:                               # %_ZNSt6vectorIfSaIfEE17_M_realloc_appendIJRKfEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB4_34 Depth=3
	leaq	(%rbx,%r14), %r13
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
	leaq	(%rbx,%r15,4), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%rsp, %r14
.LBB4_55:                               # %_ZNSt6vectorIfSaIfEE9push_backERKf.exit.i
                                        #   in Loop: Header=BB4_34 Depth=3
	movq	(%rsp), %rdi
.Ltmp47:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp48:                                # EH_LABEL
# %bb.56:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_110
# %bb.57:                               #   in Loop: Header=BB4_34 Depth=3
	movq	88(%rsp), %rdi
.Ltmp53:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp54:                                # EH_LABEL
# %bb.58:                               #   in Loop: Header=BB4_34 Depth=3
	testl	%eax, %eax
	jne	.LBB4_112
# %bb.59:                               #   in Loop: Header=BB4_34 Depth=3
	leaq	4(%r13), %r15
	movq	%rbx, %rbp
	decl	%r12d
	jne	.LBB4_34
# %bb.60:                               #   in Loop: Header=BB4_27 Depth=2
	cmpq	%r15, %rbx
	je	.LBB4_26
# %bb.61:                               #   in Loop: Header=BB4_27 Depth=2
	movq	%r15, %r14
	subq	%rbx, %r14
	movq	%r14, %rax
	sarq	$2, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp59:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp60:                                # EH_LABEL
# %bb.62:                               # %.noexc89.i
                                        #   in Loop: Header=BB4_27 Depth=2
	cmpq	$65, %r14
	jl	.LBB4_64
# %bb.63:                               # %.lr.ph.i.i
                                        #   in Loop: Header=BB4_27 Depth=2
	leaq	4(%rbx), %r12
	movl	$4, %ebp
	movq	%rbx, %r14
	jmp	.LBB4_79
	.p2align	4
.LBB4_64:                               #   in Loop: Header=BB4_27 Depth=2
	cmpq	%r13, %rbx
	je	.LBB4_25
# %bb.65:                               # %.lr.ph.i15.i.preheader
                                        #   in Loop: Header=BB4_27 Depth=2
	leaq	4(%rbx), %r14
	movq	%rbx, %r15
	jmp	.LBB4_67
	.p2align	4
.LBB4_66:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i18.i
                                        #   in Loop: Header=BB4_67 Depth=3
	movss	%xmm1, (%rax)
	addq	$4, %r14
	cmpq	%r13, %r15
	je	.LBB4_25
.LBB4_67:                               # %.lr.ph.i15.i
                                        #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB4_72 Depth 4
	movq	%r15, %rdi
	addq	$4, %r15
	movss	4(%rdi), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB4_70
# %bb.68:                               # %_ZSt7advanceIPflEvRT_T0_.exit.i.i.i.i.i28.i
                                        #   in Loop: Header=BB4_67 Depth=3
	movss	%xmm1, 48(%rsp)                 # 4-byte Spill
	movq	%r15, %rdx
	subq	%rbx, %rdx
	subq	%rdx, %rdi
	movq	%rdx, %rax
	sarq	$2, %rax
	addq	$8, %rdi
	cmpq	$2, %rax
	jl	.LBB4_73
# %bb.69:                               #   in Loop: Header=BB4_67 Depth=3
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rsi
	callq	memmove@PLT
	movq	%rbx, %rax
	movss	48(%rsp), %xmm1                 # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jmp	.LBB4_66
	.p2align	4
.LBB4_70:                               #   in Loop: Header=BB4_67 Depth=3
	movss	(%rdi), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%r15, %rax
	jbe	.LBB4_66
# %bb.71:                               # %.lr.ph.i.i22.i.preheader
                                        #   in Loop: Header=BB4_67 Depth=3
	movq	%r14, %rax
	.p2align	4
.LBB4_72:                               # %.lr.ph.i.i22.i
                                        #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        #       Parent Loop BB4_67 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB4_72
	jmp	.LBB4_66
.LBB4_73:                               # %_ZSt7advanceIPflEvRT_T0_.exit.thread.i.i.i.i.i29.i
                                        #   in Loop: Header=BB4_67 Depth=3
	movq	%rbx, %rax
	cmpq	$4, %rdx
	movss	48(%rsp), %xmm1                 # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	jne	.LBB4_66
# %bb.74:                               # %_ZSt7advanceIPflEvRT_T0_.exit.thread.thread.i.i.i.i.i31.i
                                        #   in Loop: Header=BB4_67 Depth=3
	movss	%xmm0, (%rdi)
	movq	%rbx, %rax
	jmp	.LBB4_66
.LBB4_76:                               # %_ZSt7advanceIPflEvRT_T0_.exit.thread.i.i.i.i.i.i
                                        #   in Loop: Header=BB4_79 Depth=3
	leaq	4(%rbx), %rax
	movss	%xmm0, (%rax)
	.p2align	4
.LBB4_77:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i.i
                                        #   in Loop: Header=BB4_79 Depth=3
	movq	%rbx, %rax
	movss	48(%rsp), %xmm1                 # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
.LBB4_78:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEES6_ET0_T_S8_S7_.exit.i.i
                                        #   in Loop: Header=BB4_79 Depth=3
	movss	%xmm1, (%rax)
	addq	$4, %rbp
	addq	$4, %r12
	cmpq	$64, %rbp
	je	.LBB4_85
.LBB4_79:                               #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB4_84 Depth 4
	movq	%r14, %rax
	leaq	(%rbx,%rbp), %r14
	movss	(%rbx,%rbp), %xmm1              # xmm1 = mem[0],zero,zero,zero
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB4_82
# %bb.80:                               # %_ZSt7advanceIPflEvRT_T0_.exit.i.i.i.i.i.i
                                        #   in Loop: Header=BB4_79 Depth=3
	movss	%xmm1, 48(%rsp)                 # 4-byte Spill
	cmpq	$5, %rbp
	jb	.LBB4_76
# %bb.81:                               #   in Loop: Header=BB4_79 Depth=3
	.cfi_escape 0x2e, 0x00
	leaq	4(%rbx), %rdi
	movq	%rbx, %rsi
	movq	%rbp, %rdx
	callq	memmove@PLT
	jmp	.LBB4_77
	.p2align	4
.LBB4_82:                               #   in Loop: Header=BB4_79 Depth=3
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	movq	%r14, %rax
	jbe	.LBB4_78
# %bb.83:                               # %.lr.ph.i.i.i.preheader
                                        #   in Loop: Header=BB4_79 Depth=3
	movq	%r12, %rax
	.p2align	4
.LBB4_84:                               # %.lr.ph.i.i.i
                                        #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        #       Parent Loop BB4_79 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movss	%xmm0, (%rax)
	movss	-8(%rax), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm1, %xmm0
	ja	.LBB4_84
	jmp	.LBB4_78
	.p2align	4
.LBB4_85:                               # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit.i
                                        #   in Loop: Header=BB4_27 Depth=2
	leaq	64(%rbx), %rax
	movq	%rsp, %r14
	cmpq	%r15, %rax
	jne	.LBB4_87
	jmp	.LBB4_26
	.p2align	4
.LBB4_86:                               # %_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops14_Val_less_iterEEvT_T0_.exit.i.i
                                        #   in Loop: Header=BB4_87 Depth=3
	movss	%xmm0, (%rcx)
	cmpq	%r13, %rax
	leaq	4(%rax), %rax
	je	.LBB4_26
.LBB4_87:                               # %.lr.ph.i6.i
                                        #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB4_89 Depth 4
	movss	-4(%rax), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	ucomiss	%xmm0, %xmm1
	movq	%rax, %rcx
	jbe	.LBB4_86
# %bb.88:                               # %.lr.ph.i.i8.i.preheader
                                        #   in Loop: Header=BB4_87 Depth=3
	movq	%rax, %rcx
	.p2align	4
.LBB4_89:                               # %.lr.ph.i.i8.i
                                        #   Parent Loop BB4_24 Depth=1
                                        #     Parent Loop BB4_27 Depth=2
                                        #       Parent Loop BB4_87 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movss	%xmm1, (%rcx)
	movss	-8(%rcx), %xmm1                 # xmm1 = mem[0],zero,zero,zero
	addq	$-4, %rcx
	ucomiss	%xmm0, %xmm1
	ja	.LBB4_89
	jmp	.LBB4_86
.LBB4_90:
	movq	56(%rsp), %rdi
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
	testl	%eax, %eax
	jne	.LBB4_129
# %bb.91:
	xorl	%eax, %eax
.LBB4_92:
	addq	$1688, %rsp                     # imm = 0x698
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
.LBB4_93:
	.cfi_def_cfa_offset 1744
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	leaq	1376(%rsp), %rdx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.8(%rip), %rsi
.LBB4_94:
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$2, %eax
	jmp	.LBB4_92
.LBB4_95:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp35:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp36:                                # EH_LABEL
# %bb.96:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.29(%rip), %r8
	movq	%rbx, %rdi
	movl	$195, %ecx
	jmp	.LBB4_114
.LBB4_97:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp11:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	movq	8(%rsp), %r13                   # 8-byte Reload
	callq	hipGetErrorString@PLT
.Ltmp12:                                # EH_LABEL
# %bb.98:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.26(%rip), %r8
	movq	%rbx, %rdi
	movl	$191, %ecx
	jmp	.LBB4_114
.LBB4_99:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp23:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp24:                                # EH_LABEL
# %bb.100:                              # %.noexc95.i
	.cfi_escape 0x2e, 0x00
.LBB4_101:                              # %.noexc95.i
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.21(%rip), %r8
	movq	%rbx, %rdi
	movl	$162, %ecx
	jmp	.LBB4_114
.LBB4_102:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp5:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp6:                                 # EH_LABEL
# %bb.103:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.25(%rip), %r8
	movq	%rbx, %rdi
	movl	$190, %ecx
	jmp	.LBB4_114
.LBB4_104:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp41:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp42:                                # EH_LABEL
# %bb.105:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.30(%rip), %r8
	movq	%rbx, %rdi
	movl	$197, %ecx
	jmp	.LBB4_114
.LBB4_106:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp29:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp30:                                # EH_LABEL
# %bb.107:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.28(%rip), %r8
	movq	%rbx, %rdi
	movl	$194, %ecx
	jmp	.LBB4_114
.LBB4_108:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp17:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp18:                                # EH_LABEL
# %bb.109:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.27(%rip), %r8
	movq	%rbx, %rdi
	movl	$192, %ecx
	jmp	.LBB4_114
.LBB4_110:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp50:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	movq	8(%rsp), %r13                   # 8-byte Reload
	callq	hipGetErrorString@PLT
.Ltmp51:                                # EH_LABEL
# %bb.111:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.31(%rip), %r8
	movq	%r14, %rdi
	movl	$199, %ecx
	jmp	.LBB4_114
.LBB4_112:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %r14
.Ltmp56:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp57:                                # EH_LABEL
# %bb.113:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.32(%rip), %r8
	movq	%r14, %rdi
	movl	$200, %ecx
.LBB4_114:
	movq	%rax, %r9
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$1, %edi
	callq	exit@PLT
.LBB4_115:
.Ltmp62:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.35(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp63:                                # EH_LABEL
# %bb.116:                              # %.noexc98.i
.LBB4_117:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	jmp	.LBB4_101
.LBB4_118:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.19(%rip), %r8
	movq	%rbx, %rdi
	movl	$183, %ecx
	jmp	.LBB4_114
.LBB4_119:
	movq	stderr@GOTPCREL(%rip), %rax
	movq	(%rax), %rdi
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	(%rax), %rdx
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.24(%rip), %rsi
	movl	%ebp, %ecx
	movq	%r12, %r9
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB4_120:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.23(%rip), %r8
	movq	%rbx, %rdi
	movl	$172, %ecx
	jmp	.LBB4_114
.LBB4_121:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.22(%rip), %r8
	movq	%rbx, %rdi
	movl	$169, %ecx
	jmp	.LBB4_114
.LBB4_122:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.18(%rip), %r8
	movq	%rbx, %rdi
	movl	$262, %ecx                      # imm = 0x106
	jmp	.LBB4_114
.LBB4_123:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.4(%rip), %r8
	movq	%rbx, %rdi
	movl	$229, %ecx
	jmp	.LBB4_114
.LBB4_124:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.5(%rip), %r8
	movq	%rbx, %rdi
	movl	$230, %ecx
	jmp	.LBB4_114
.LBB4_125:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.6(%rip), %r8
	movq	%rbx, %rdi
	movl	$232, %ecx
	jmp	.LBB4_114
.LBB4_126:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.12(%rip), %r8
	movq	%rbx, %rdi
	movl	$241, %ecx
	jmp	.LBB4_114
.LBB4_127:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.13(%rip), %r8
	movq	%rbx, %rdi
	movl	$242, %ecx
	jmp	.LBB4_114
.LBB4_128:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.19(%rip), %r8
	movq	%rbx, %rdi
	movl	$267, %ecx                      # imm = 0x10B
	jmp	.LBB4_114
.LBB4_129:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.20(%rip), %r8
	movq	%rbx, %rdi
	movl	$275, %ecx                      # imm = 0x113
	jmp	.LBB4_114
.LBB4_130:
.Ltmp61:                                # EH_LABEL
	jmp	.LBB4_144
.LBB4_131:                              # %.loopexit.split-lp45.i
.Ltmp64:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_132:                              # %.loopexit44.i
.Ltmp46:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_133:                              # %.loopexit.split-lp55.i
.Ltmp58:                                # EH_LABEL
	jmp	.LBB4_144
.LBB4_134:                              # %.loopexit.split-lp50.i
.Ltmp52:                                # EH_LABEL
	movq	%rax, %r14
	movq	%rbx, %rbp
	jmp	.LBB4_154
.LBB4_135:                              # %.loopexit.split-lp20.i
.Ltmp19:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_136:                              # %.loopexit.split-lp30.i
.Ltmp31:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_137:                              # %.loopexit.split-lp40.i
.Ltmp43:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_138:                              # %.loopexit.split-lp.i
.Ltmp7:                                 # EH_LABEL
	jmp	.LBB4_150
.LBB4_139:                              # %.loopexit.split-lp25.i
.Ltmp25:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_140:                              # %.loopexit.split-lp15.i
.Ltmp13:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_141:                              # %.loopexit.split-lp35.i
.Ltmp37:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_142:                              # %.loopexit54.i
.Ltmp55:                                # EH_LABEL
	jmp	.LBB4_144
.LBB4_143:                              # %.loopexit49.i
.Ltmp49:                                # EH_LABEL
.LBB4_144:
	movq	%rax, %r14
	movq	%rbx, %rbp
	movq	8(%rsp), %r13                   # 8-byte Reload
	jmp	.LBB4_154
.LBB4_145:                              # %.loopexit19.i
.Ltmp16:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_146:                              # %.loopexit24.i
.Ltmp22:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_147:                              # %.loopexit.i
.Ltmp4:                                 # EH_LABEL
	jmp	.LBB4_150
.LBB4_148:                              # %.loopexit39.i
.Ltmp40:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_149:                              # %.loopexit14.i
.Ltmp10:                                # EH_LABEL
.LBB4_150:
	movq	%rax, %r14
	movq	8(%rsp), %r13                   # 8-byte Reload
	jmp	.LBB4_154
.LBB4_151:                              # %.loopexit29.i
.Ltmp28:                                # EH_LABEL
	jmp	.LBB4_153
.LBB4_152:                              # %.loopexit34.i
.Ltmp34:                                # EH_LABEL
.LBB4_153:
	movq	%rax, %r14
.LBB4_154:
	testq	%rbp, %rbp
	je	.LBB4_156
# %bb.155:
	subq	%rbp, %r13
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
.LBB4_156:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit102.i
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end4:
	.size	main, .Lfunc_end4-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table4:
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
	.uleb128 .Ltmp61-.Lfunc_begin0          #     jumps to .Ltmp61
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp2-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp3-.Ltmp2                  #   Call between .Ltmp2 and .Ltmp3
	.uleb128 .Ltmp4-.Lfunc_begin0           #     jumps to .Ltmp4
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp8-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp9-.Ltmp8                  #   Call between .Ltmp8 and .Ltmp9
	.uleb128 .Ltmp10-.Lfunc_begin0          #     jumps to .Ltmp10
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp14-.Lfunc_begin0          # >> Call Site 5 <<
	.uleb128 .Ltmp15-.Ltmp14                #   Call between .Ltmp14 and .Ltmp15
	.uleb128 .Ltmp16-.Lfunc_begin0          #     jumps to .Ltmp16
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp20-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp21-.Ltmp20                #   Call between .Ltmp20 and .Ltmp21
	.uleb128 .Ltmp22-.Lfunc_begin0          #     jumps to .Ltmp22
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp26-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp27-.Ltmp26                #   Call between .Ltmp26 and .Ltmp27
	.uleb128 .Ltmp28-.Lfunc_begin0          #     jumps to .Ltmp28
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp32-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp33-.Ltmp32                #   Call between .Ltmp32 and .Ltmp33
	.uleb128 .Ltmp34-.Lfunc_begin0          #     jumps to .Ltmp34
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp38-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp39-.Ltmp38                #   Call between .Ltmp38 and .Ltmp39
	.uleb128 .Ltmp40-.Lfunc_begin0          #     jumps to .Ltmp40
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp44-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp45-.Ltmp44                #   Call between .Ltmp44 and .Ltmp45
	.uleb128 .Ltmp46-.Lfunc_begin0          #     jumps to .Ltmp46
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp45-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp47-.Ltmp45                #   Call between .Ltmp45 and .Ltmp47
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp47-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp48-.Ltmp47                #   Call between .Ltmp47 and .Ltmp48
	.uleb128 .Ltmp49-.Lfunc_begin0          #     jumps to .Ltmp49
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp53-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp54-.Ltmp53                #   Call between .Ltmp53 and .Ltmp54
	.uleb128 .Ltmp55-.Lfunc_begin0          #     jumps to .Ltmp55
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp59-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp60-.Ltmp59                #   Call between .Ltmp59 and .Ltmp60
	.uleb128 .Ltmp61-.Lfunc_begin0          #     jumps to .Ltmp61
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp60-.Lfunc_begin0          # >> Call Site 15 <<
	.uleb128 .Ltmp35-.Ltmp60                #   Call between .Ltmp60 and .Ltmp35
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp35-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Ltmp36-.Ltmp35                #   Call between .Ltmp35 and .Ltmp36
	.uleb128 .Ltmp37-.Lfunc_begin0          #     jumps to .Ltmp37
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          # >> Call Site 17 <<
	.uleb128 .Ltmp12-.Ltmp11                #   Call between .Ltmp11 and .Ltmp12
	.uleb128 .Ltmp13-.Lfunc_begin0          #     jumps to .Ltmp13
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp23-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp24-.Ltmp23                #   Call between .Ltmp23 and .Ltmp24
	.uleb128 .Ltmp25-.Lfunc_begin0          #     jumps to .Ltmp25
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp5-.Lfunc_begin0           # >> Call Site 19 <<
	.uleb128 .Ltmp6-.Ltmp5                  #   Call between .Ltmp5 and .Ltmp6
	.uleb128 .Ltmp7-.Lfunc_begin0           #     jumps to .Ltmp7
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp41-.Lfunc_begin0          # >> Call Site 20 <<
	.uleb128 .Ltmp42-.Ltmp41                #   Call between .Ltmp41 and .Ltmp42
	.uleb128 .Ltmp43-.Lfunc_begin0          #     jumps to .Ltmp43
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp29-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Ltmp30-.Ltmp29                #   Call between .Ltmp29 and .Ltmp30
	.uleb128 .Ltmp31-.Lfunc_begin0          #     jumps to .Ltmp31
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp17-.Lfunc_begin0          # >> Call Site 22 <<
	.uleb128 .Ltmp18-.Ltmp17                #   Call between .Ltmp17 and .Ltmp18
	.uleb128 .Ltmp19-.Lfunc_begin0          #     jumps to .Ltmp19
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp50-.Lfunc_begin0          # >> Call Site 23 <<
	.uleb128 .Ltmp51-.Ltmp50                #   Call between .Ltmp50 and .Ltmp51
	.uleb128 .Ltmp52-.Lfunc_begin0          #     jumps to .Ltmp52
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp56-.Lfunc_begin0          # >> Call Site 24 <<
	.uleb128 .Ltmp57-.Ltmp56                #   Call between .Ltmp56 and .Ltmp57
	.uleb128 .Ltmp58-.Lfunc_begin0          #     jumps to .Ltmp58
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp62-.Lfunc_begin0          # >> Call Site 25 <<
	.uleb128 .Ltmp63-.Ltmp62                #   Call between .Ltmp62 and .Ltmp63
	.uleb128 .Ltmp64-.Lfunc_begin0          #     jumps to .Ltmp64
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp63-.Lfunc_begin0          # >> Call Site 26 <<
	.uleb128 .Lfunc_end4-.Ltmp63            #   Call between .Ltmp63 and .Lfunc_end4
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.prefalign	4, .Lfunc_end5, nop
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
	jl	.LBB5_40
# %bb.1:                                # %.lr.ph
	movq	%rdx, %r14
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.LBB5_8
# %bb.2:                                # %.lr.ph43.preheader
	movq	$-4, %r13
	subq	%rbx, %r13
	.p2align	4
.LBB5_3:                                # %.lr.ph43
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_33 Depth 2
                                        #       Child Loop BB5_34 Depth 3
                                        #       Child Loop BB5_36 Depth 3
	shrq	%rbp
	movss	4(%rbx), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	movss	(%rbx,%rbp,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm2
	movss	-4(%rsi), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	jbe	.LBB5_27
# %bb.4:                                #   in Loop: Header=BB5_3 Depth=1
	ucomiss	%xmm2, %xmm0
	jbe	.LBB5_24
# %bb.5:                                #   in Loop: Header=BB5_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm2, (%rbx)
	movss	%xmm0, (%rbx,%rbp,4)
	jmp	.LBB5_32
	.p2align	4
.LBB5_27:                               #   in Loop: Header=BB5_3 Depth=1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB5_29
# %bb.28:                               #   in Loop: Header=BB5_3 Depth=1
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx)
	movss	%xmm0, 4(%rbx)
	jmp	.LBB5_32
	.p2align	4
.LBB5_24:                               #   in Loop: Header=BB5_3 Depth=1
	ucomiss	%xmm1, %xmm0
	movss	(%rbx), %xmm2                   # xmm2 = mem[0],zero,zero,zero
	jbe	.LBB5_26
# %bb.25:                               #   in Loop: Header=BB5_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm2, -4(%rsi)
	jmp	.LBB5_32
	.p2align	4
.LBB5_29:                               #   in Loop: Header=BB5_3 Depth=1
	ucomiss	%xmm2, %xmm0
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	jbe	.LBB5_31
# %bb.30:                               #   in Loop: Header=BB5_3 Depth=1
	movss	%xmm0, (%rbx)
	movss	%xmm1, -4(%rsi)
	jmp	.LBB5_32
.LBB5_26:                               #   in Loop: Header=BB5_3 Depth=1
	movss	%xmm1, (%rbx)
	movss	%xmm2, 4(%rbx)
	jmp	.LBB5_32
.LBB5_31:                               #   in Loop: Header=BB5_3 Depth=1
	movss	%xmm2, (%rbx)
	movss	%xmm1, (%rbx,%rbp,4)
	.p2align	4
.LBB5_32:                               # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i.preheader
                                        #   in Loop: Header=BB5_3 Depth=1
	decq	%r14
	leaq	4(%rbx), %r12
	movq	%rsi, %rax
	.p2align	4
.LBB5_33:                               # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i
                                        #   Parent Loop BB5_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB5_34 Depth 3
                                        #       Child Loop BB5_36 Depth 3
	movss	(%rbx), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	leaq	(%r12,%r13), %rbp
	.p2align	4
.LBB5_34:                               #   Parent Loop BB5_3 Depth=1
                                        #     Parent Loop BB5_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	(%r12), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	addq	$4, %r12
	addq	$4, %rbp
	ucomiss	%xmm1, %xmm0
	ja	.LBB5_34
# %bb.35:                               # %.preheader.i.i.preheader
                                        #   in Loop: Header=BB5_33 Depth=2
	leaq	-4(%r12), %r15
	.p2align	4
.LBB5_36:                               # %.preheader.i.i
                                        #   Parent Loop BB5_3 Depth=1
                                        #     Parent Loop BB5_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	-4(%rax), %xmm2                 # xmm2 = mem[0],zero,zero,zero
	addq	$-4, %rax
	ucomiss	%xmm0, %xmm2
	ja	.LBB5_36
# %bb.37:                               #   in Loop: Header=BB5_33 Depth=2
	cmpq	%rax, %r15
	jae	.LBB5_39
# %bb.38:                               #   in Loop: Header=BB5_33 Depth=2
	movss	%xmm2, (%r15)
	movss	%xmm1, (%rax)
	jmp	.LBB5_33
	.p2align	4
.LBB5_39:                               # %_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_.exit
                                        #   in Loop: Header=BB5_3 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	sarq	$2, %rbp
	cmpq	$16, %rbp
	jle	.LBB5_40
# %bb.6:                                #   in Loop: Header=BB5_3 Depth=1
	movq	%r15, %rsi
	testq	%r14, %r14
	jne	.LBB5_3
# %bb.7:                                # %._crit_edge.loopexit
	addq	$-4, %r12
	movq	%r12, %rsi
.LBB5_8:                                # %._crit_edge
	leaq	7(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%rsi, %r14
	callq	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	jmp	.LBB5_9
	.p2align	4
.LBB5_22:                               #   in Loop: Header=BB5_9 Depth=1
	xorl	%ecx, %ecx
.LBB5_23:                               # %_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_.exit.i.i
                                        #   in Loop: Header=BB5_9 Depth=1
	movss	%xmm0, (%rbx,%rcx,4)
	cmpq	$4, %rax
	jle	.LBB5_40
.LBB5_9:                                # %.lr.ph.i.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_12 Depth 2
                                        #     Child Loop BB5_20 Depth 2
	movss	-4(%r14), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	movss	(%rbx), %xmm1                   # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, -4(%r14)
	addq	$-4, %r14
	movq	%r14, %rax
	subq	%rbx, %rax
	movq	%rax, %rdx
	sarq	$2, %rdx
	cmpq	$3, %rdx
	jl	.LBB5_10
# %bb.11:                               # %.lr.ph.i.i.i.i.preheader
                                        #   in Loop: Header=BB5_9 Depth=1
	leaq	-1(%rdx), %rcx
	shrq	$63, %rcx
	leaq	(%rdx,%rcx), %rsi
	decq	%rsi
	sarq	%rsi
	xorl	%edi, %edi
	jmp	.LBB5_12
	.p2align	4
.LBB5_14:                               # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB5_12 Depth=2
	leaq	2(,%rdi,2), %rcx
.LBB5_15:                               # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB5_12 Depth=2
	movss	(%rbx,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rdi,4)
	movq	%rcx, %rdi
	cmpq	%rsi, %rcx
	jge	.LBB5_16
.LBB5_12:                               # %.lr.ph.i.i.i.i
                                        #   Parent Loop BB5_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rdi,%rdi), %rcx
	movss	4(%rbx,%rcx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	ucomiss	8(%rbx,%rcx,4), %xmm1
	jbe	.LBB5_14
# %bb.13:                               #   in Loop: Header=BB5_12 Depth=2
	leaq	1(,%rdi,2), %rcx
	jmp	.LBB5_15
	.p2align	4
.LBB5_10:                               #   in Loop: Header=BB5_9 Depth=1
	xorl	%ecx, %ecx
.LBB5_16:                               # %._crit_edge.i.i.i.i
                                        #   in Loop: Header=BB5_9 Depth=1
	testb	$4, %al
	jne	.LBB5_19
# %bb.17:                               #   in Loop: Header=BB5_9 Depth=1
	addq	$-2, %rdx
	sarq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB5_19
# %bb.18:                               # %.thread.i.i.i
                                        #   in Loop: Header=BB5_9 Depth=1
	leaq	(%rcx,%rcx), %rdx
	movss	4(%rbx,%rdx,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	movss	%xmm1, (%rbx,%rcx,4)
	leaq	1(,%rcx,2), %rcx
	jmp	.LBB5_20
	.p2align	4
.LBB5_19:                               #   in Loop: Header=BB5_9 Depth=1
	testq	%rcx, %rcx
	je	.LBB5_22
	.p2align	4
.LBB5_20:                               # %.lr.ph.i.i.i.i.i
                                        #   Parent Loop BB5_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rcx), %rdx
	shrq	%rdx
	movss	(%rbx,%rdx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	ucomiss	%xmm1, %xmm0
	jbe	.LBB5_23
# %bb.21:                               #   in Loop: Header=BB5_20 Depth=2
	movss	%xmm1, (%rbx,%rcx,4)
	movq	%rdx, %rcx
	testq	%rdx, %rdx
	jne	.LBB5_20
	jmp	.LBB5_22
.LBB5_40:                               # %_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_.exit
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
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end5-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
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
	.text
	.prefalign	4, .Lfunc_end7, nop     # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_dec4f8244316d2d0(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB7_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_dec4f8244316d2d0(%rip)
.LBB7_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	peak_f16@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	peak_iu8@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	peak_fp8@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	peak_iu4@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
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
.Lfunc_end7:
	.size	__hip_module_ctor, .Lfunc_end7-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end8, nop     # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_dec4f8244316d2d0(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB8_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_dec4f8244316d2d0(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB8_2:
	retq
.Lfunc_end8:
	.size	__hip_module_dtor, .Lfunc_end8-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	peak_f16,@object                # @peak_f16
	.section	.data.rel.ro,"aw",@progbits
	.globl	peak_f16
	.p2align	3, 0x0
peak_f16:
	.quad	__device_stub__peak_f16
	.size	peak_f16, 8

	.type	peak_iu8,@object                # @peak_iu8
	.globl	peak_iu8
	.p2align	3, 0x0
peak_iu8:
	.quad	__device_stub__peak_iu8
	.size	peak_iu8, 8

	.type	peak_fp8,@object                # @peak_fp8
	.globl	peak_fp8
	.p2align	3, 0x0
peak_fp8:
	.quad	__device_stub__peak_fp8
	.size	peak_fp8, 8

	.type	peak_iu4,@object                # @peak_iu4
	.globl	peak_iu4
	.p2align	3, 0x0
peak_iu4:
	.quad	__device_stub__peak_iu4
	.size	peak_iu4, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"--reverse"
	.size	.L.str, 10

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"usage: %s [--reverse]\n"
	.size	.L.str.1, 23

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"%s:%d: %s failed: %s\n"
	.size	.L.str.2, 22

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"wmma_rate.hip"
	.size	.L.str.3, 14

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"hipGetDevice(&device)"
	.size	.L.str.4, 22

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"hipGetDeviceProperties(&props, device)"
	.size	.L.str.5, 39

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, device)"
	.size	.L.str.6, 74

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"expected gfx1201, found %s\n"
	.size	.L.str.8, 28

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"DEVICE,name=%s,arch=%s,cu=%d,order=%s\n"
	.size	.L.str.9, 39

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"reverse"
	.size	.L.str.10, 8

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"forward"
	.size	.L.str.11, 8

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"hipMalloc(&out, sizeof(*out))"
	.size	.L.str.12, 30

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"hipMemset(out, 0, sizeof(*out))"
	.size	.L.str.13, 32

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"f16"
	.size	.L.str.14, 4

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"iu8"
	.size	.L.str.15, 4

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"fp8"
	.size	.L.str.16, 4

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"iu4"
	.size	.L.str.17, 4

	.type	.L__const.main.forward,@object  # @__const.main.forward
	.section	.data.rel.ro,"aw",@progbits
	.p2align	4, 0x0
.L__const.main.forward:
	.quad	.L.str.14
	.quad	peak_f16
	.long	8192                            # 0x2000
	.zero	4
	.quad	.L.str.15
	.quad	peak_iu8
	.long	8192                            # 0x2000
	.zero	4
	.quad	.L.str.16
	.quad	peak_fp8
	.long	8192                            # 0x2000
	.zero	4
	.quad	.L.str.17
	.quad	peak_iu4
	.long	16384                           # 0x4000
	.zero	4
	.size	.L__const.main.forward, 96

	.type	.L__const.main.backward,@object # @__const.main.backward
	.p2align	4, 0x0
.L__const.main.backward:
	.quad	.L.str.17
	.quad	peak_iu4
	.long	16384                           # 0x4000
	.zero	4
	.quad	.L.str.16
	.quad	peak_fp8
	.long	8192                            # 0x2000
	.zero	4
	.quad	.L.str.15
	.quad	peak_iu8
	.long	8192                            # 0x2000
	.zero	4
	.quad	.L.str.14
	.quad	peak_f16
	.long	8192                            # 0x2000
	.zero	4
	.size	.L__const.main.backward, 96

	.type	.L.str.18,@object               # @.str.18
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.18:
	.asciz	"hipFuncSetAttribute( (const void*)variants[dtype].kernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)occupancy.dynamic_lds)"
	.size	.L.str.18, 130

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"hipDeviceSynchronize()"
	.size	.L.str.19, 23

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"hipFree(out)"
	.size	.L.str.20, 13

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"hipLaunchKernel((const void*)kernel, dim3(blocks), dim3(kThreads), args, dynamic_lds, nullptr)"
	.size	.L.str.21, 95

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"hipFuncSetAttribute((const void*)variant.kernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)requested.dynamic_lds)"
	.size	.L.str.22, 121

	.type	.L.str.23,@object               # @.str.23
.L.str.23:
	.asciz	"hipOccupancyMaxActiveBlocksPerMultiprocessor( &available, (const void*)variant.kernel, kThreads, requested.dynamic_lds)"
	.size	.L.str.23, 120

	.type	.L.str.24,@object               # @.str.24
.L.str.24:
	.asciz	"occupancy mismatch for %s: requested=%d available=%d lds=%zu\n"
	.size	.L.str.24, 62

	.type	.L.str.25,@object               # @.str.25
.L.str.25:
	.asciz	"hipEventCreate(&begin)"
	.size	.L.str.25, 23

	.type	.L.str.26,@object               # @.str.26
.L.str.26:
	.asciz	"hipEventCreate(&end)"
	.size	.L.str.26, 21

	.type	.L.str.27,@object               # @.str.27
.L.str.27:
	.asciz	"hipEventRecord(begin)"
	.size	.L.str.27, 22

	.type	.L.str.28,@object               # @.str.28
.L.str.28:
	.asciz	"hipEventRecord(end)"
	.size	.L.str.28, 20

	.type	.L.str.29,@object               # @.str.29
.L.str.29:
	.asciz	"hipEventSynchronize(end)"
	.size	.L.str.29, 25

	.type	.L.str.30,@object               # @.str.30
.L.str.30:
	.asciz	"hipEventElapsedTime(&elapsed, begin, end)"
	.size	.L.str.30, 42

	.type	.L.str.31,@object               # @.str.31
.L.str.31:
	.asciz	"hipEventDestroy(begin)"
	.size	.L.str.31, 23

	.type	.L.str.32,@object               # @.str.32
.L.str.32:
	.asciz	"hipEventDestroy(end)"
	.size	.L.str.32, 21

	.type	.L.str.33,@object               # @.str.33
.L.str.33:
	.asciz	"RESULT,dtype=%s,blocks_per_cu=%d,lds=%zu,median_ms=%.6f,rate=%.6f,launch_spread_pct=%.4f,samples=%d\n"
	.size	.L.str.33, 101

	.type	.L.str.35,@object               # @.str.35
.L.str.35:
	.asciz	"vector::_M_realloc_append"
	.size	.L.str.35, 26

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"peak_f16"
	.size	.L__unnamed_1, 9

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"peak_iu8"
	.size	.L__unnamed_2, 9

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"peak_fp8"
	.size	.L__unnamed_3, 9

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"peak_iu4"
	.size	.L__unnamed_4, 9

	.type	.L__unnamed_5,@object           # @4
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_5:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\0000+\000\000\000\000\000\000 \000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx1201\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\260&\000\000\000\000\000\000N\000\000\000@\0008\000\t\000@\000\022\000\020\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\f\016\000\000\000\000\000\000\f\016\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\017\000\000\000\000\000\000\000\037\000\000\000\000\000\000\000\037\000\000\000\000\000\000\200\023\000\000\000\000\000\000\200\023\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\200\"\000\000\000\000\000\000\200B\000\000\000\000\000\000\200B\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\r\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\360\"\000\000\000\000\000\000\360R\000\000\000\000\000\000\360R\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\200\"\000\000\000\000\000\000\200B\000\000\000\000\000\000\200B\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\200\"\000\000\000\000\000\000\200B\000\000\000\000\000\000\200B\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\r\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\034\b\000\000\000\000\000\000\034\b\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000\b\b\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\224\336\000\023\245.args\222\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\203\247.offset\b\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\f\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\250peak_f16\273.private_segment_fixed_size\000\253.sgpr_count\f\261.sgpr_spill_count\000\247.symbol\253peak_f16.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countI\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\222\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\203\247.offset\b\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\f\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\250peak_iu8\273.private_segment_fixed_size\000\253.sgpr_count\f\261.sgpr_spill_count\000\247.symbol\253peak_iu8.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countF\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\222\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\203\247.offset\b\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\f\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\250peak_fp8\273.private_segment_fixed_size\000\253.sgpr_count\013\261.sgpr_spill_count\000\247.symbol\253peak_fp8.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countF\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\222\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\203\247.offset\b\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\f\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\250peak_iu4\273.private_segment_fixed_size\000\253.sgpr_count\n\261.sgpr_spill_count\000\247.symbol\253peak_iu4.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countE\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\272amdgcn-amd-amdhsa--gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\b\000\000\037\000\000\000\000\000\000\324\002\000\000\000\000\000\000\037\000\000\000\021\003\006\000\300\f\000\000\000\000\000\000@\000\000\000\000\000\000\0004\000\000\000\021\003\006\000\000\r\000\000\000\000\000\000@\000\000\000\000\000\000\000I\000\000\000\021\003\006\000@\r\000\000\000\000\000\000@\000\000\000\000\000\000\000\n\000\000\000\021\003\006\000\200\f\000\000\000\000\000\000@\000\000\000\000\000\000\000\026\000\000\000\022\003\b\000\000\"\000\000\000\000\000\000\234\005\000\000\000\000\000\000+\000\000\000\022\003\b\000\000(\000\000\000\000\000\000\220\005\000\000\000\000\000\000@\000\000\000\022\003\b\000\000.\000\000\000\000\000\000\330\002\000\000\000\000\000\000U\000\000\000\021\000\013\000\360R\000\000\000\000\000\000\001\000\000\000\000\000\000\000\002\000\000\000\001\000\000\000\002\000\000\000\032\000\000\000\000\201\013\000\b\000\020\000\000\001\b\000\000\000\224\t\001\000\000\000\005\000\000\000\362\327D!x\021\376L\020\035\244E\365\337\373L\216\374.Az\355D!\022\340D!v\355D!#]\023\320\n\000\000\000\n\000\000\000\006\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\t\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\004\000\000\000\002\000\000\000\007\000\000\000\000peak_f16\000peak_f16.kd\000peak_iu8\000peak_iu8.kd\000peak_fp8\000peak_fp8.kd\000peak_iu4\000peak_iu4.kd\000__hip_cuid_dec4f8244316d2d0\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000\200\022\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000`\000\000\000\t\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000@\025\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\000\000\000\b\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000\000\033\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\000\000\000\b\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000\300 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000`\000\000\000\b\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\020\000\000\000\000\000\000\000\001zR\000\004\004\020\001\033\000\000\000\030\000\000\000\030\000\000\000d\021\000\000\324\002\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\0004\000\000\000H\024\000\000\234\005\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000P\000\000\000,\032\000\000\220\005\000\000\000\017\00406\351\002\007\020\000\000\000\034\000\000\000l\000\000\000\020 \000\000\330\002\000\000\000\017\00406\351\002\007\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\364\b\000\000\370\000\000\307\277\002\201\004\277m\000\242\277\377\000\213\276\000?\200?\377\000\212\276\000>\200>\377\000\211\276\000=\200=\377\000\210\276\000<\200<\200\000\020\312\013\000D\001\377\000\207\276\000C\200C\377\000\206\276\000B\200B\377\000\205\276\000A\200A\377\000\204\276\000@\200@\b\000\020\312\007\000HA\n\000\020\312\t\000BC\006\000\020\312\005\000FG\004\000\020\312\001\001\002E\001\001\020\312\001\001\004\003\001\001\020\312\001\001\006\005\001\001\020\312\001\001\b\007\001\001\020\312\001\001\n\t\001\001\020\312\001\001\f\013\001\001\020\312\001\001\016\r\001\001\020\312\001\001\020\017\001\001\020\312\001\001\022\021\001\001\020\312\001\001\024\023\001\001\020\312\001\001\026\025\001\001\020\312\001\001\030\027\001\001\020\312\001\001\032\031\001\001\020\312\001\001\034\033\001\001\020\312\001\001\036\035\001\001\020\312\001\001 \037\001\001\020\312\001\001\"!\001\001\020\312\001\001$#\001\001\020\312\001\001&%\001\001\020\312\001\001('\001\001\020\312\001\001*)\001\001\020\312\001\001,+\001\001\020\312\001\001.-\001\001\020\312\001\0010/\001\001\020\312\001\00121\001\001\020\312\001\00143\001\001\020\312\001\00165\001\001\020\312\001\00187\001\001\020\312\001\001:9\001\001\020\312\001\001<;\001\001\020\312\001\001>=\001\001\020\312\001\001@?\001@@\314A\213\006\034\t@@\314A\213&\034\021@@\314A\213F\034\031@@\314A\213f\034!@@\314A\213\206\034)@@\314A\213\246\0341@@\314A\213\306\0349@@\314A\213\346\034\002\301\002\201\t\000\207\277\002\200\007\277\354\377\242\277\b\000\240\277\200\002r~\200\002b~\200\002R~\200\002B~\200\0022~\200\002\"~\200\002\022~\200\002\002~\237\000\0006~\000\202\276\001\000\207\277\200\000\224}5\000\245\277\200\002\000\006~\000\204\276\200\001\202\276\221\000\207\277\000\023\000\006\000#\000\006\221\000\207\277\0003\000\006\000C\000\006\221\000\207\277\000S\000\006\000c\000\006\221\000\207\277\000s\000\006\000C\000~\221\000\207\277\377\000\002\020\000\000\200/\001I\002~!\001\207\277\001\001\000X\000\000\200\317\001\017\002~\000\017\000~\236\377\210\277\004\b\205\276\236\377\210\277\007\000`\327\001\013\000\002\006\000`\327\000\013\000\002\201\005\005\204\236\377\210\277\004\005\004\221\002\006\202\251\364\377\242\277\000\000\037\327~\000\001\002~\000\204\276\001\000\207\277\200\000\224}\236\377\210\277~\004\004\215\t\000\245\277\000 \000\364\000\000\000\370\002\002\000~\200\000\020\312\003\000\000\002\000\000\307\277\000\300\020\356\000\000\b\000\002\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\200\000\000\364\b\000\000\370\207\000\2126\000\000\307\277\002\201\004\277]\000\242\277\200\002R~\377\212\20288888\377\212\2048@@@@\377\212\2068BBBB\377\212\2108DDDD)\001\020\312)\001**)\001\020\312)\001,,)\001\020\312)\001..)\001\020\312)\00180)\001\020\312)\001::)\001\020\312)\001<<)\001\020\312)\001>>)\001\020\312)\0010@)\001\020\312)\00122)\001\020\312)\00144)\001\020\312)\00166)\001\020\312)\001 8)\001\020\312)\001\"\")\001\020\312)\001$$)\001\020\312)\001&&)\001\020\312)\001\030()\001\020\312)\001\032\032)\001\020\312)\001\034\034)\001\020\312)\001\036\036)\001\020\312)\001\020 )\001\020\312)\001\022\022)\001\020\312)\001\024\024)\001\020\312)\001\026\026)\001\020\312)\001\b\030)\001\020\312)\001\n\n)\001\020\312)\001\f\f)\001\020\312)\001\016\016)\001\020\312)\001\000\020)\001\020\312)\001\002\002)\001\020\312)\001\004\004)\001\020\312)\001\006\006)\003\020~)@D\314A\207\246|9@D\314A\207\346|1@D\314A\207\306|!@D\314A\207\206|\031@D\314A\207f|\021@D\314A\207F|\t@D\314A\207&|\001@D\314A\207\006|\002\301\002\201\t\000\207\277\002\200\007\277\354\377\242\277A\000\240\277\200\002\020~\001\000\207\277\b\001\020\312\b\001\006\007\b\001\020\312\b\001\004\005\b\001\020\312\b\001\002\003\b\001\020\312\b\001\020\001\b\001\020\312\b\001\016\017\b\001\020\312\b\001\f\r\b\001\020\312\b\001\n\013\b\001\020\312\b\001\030\t\b\001\020\312\b\001\026\027\b\001\020\312\b\001\024\025\b\001\020\312\b\001\022\023\b\001\020\312\b\001 \021\b\001\020\312\b\001\036\037\b\001\020\312\b\001\034\035\b\001\020\312\b\001\032\033\b\001\020\312\b\001(\031\b\001\020\312\b\001&'\b\001\020\312\b\001$%\b\001\020\312\b\001\"#\b\001\020\312\b\0018!\b\001\020\312\b\00167\b\001\020\312\b\00145\b\001\020\312\b\00123\b\001\020\312\b\001@1\b\001\020\312\b\001>?\b\001\020\312\b\001<=\b\001\020\312\b\001:;\b\001\020\312\b\00109\b\001\020\312\b\001./\b\001\020\312\b\001,-\b\001\020\312\b\001*+\b\003R~\237\000\0006~\000\202\276\001\000\207\277\200\000\224}\275\000\245\277\201\212\224|\002\000J\324\202\212\002\002\003\000J\324\203\212\002\002\004\000J\324\204\212\002\002\005\000J\324\205\212\002\002)U\000\0029uR\0021eT\002!EB\002\03152\002\000\000\001\325\000W\n\000)\000\001\325)w\n\000\"\000\001\325*g\n\000\021%\"\002\006\000J\324\206\212\002\002\000\000\001\325\000Y\016\000)\000\001\325)y\016\000\032\000\001\325\"i\016\000\t\025\022\002\007\000J\324\207\212\002\002\000\000\001\325\000[\022\000\"\000\001\325){\022\000\022\000\001\325\032k\022\000\031\000\001\325\0317\n\000\021\000\001\325\021'\n\000\000\000\001\325\000]\026\000\032\000\001\325\"}\026\000\n\000\001\325\022m\026\000\t\000\001\325\t\027\n\000\021\000\001\325\021)\016\000\000\000\001\325\000_\032\000\022\000\001\325\032\177\032\000\n\000\001\325\no\032\000\032\000\001\325!G\n\000\001\005\002\002\000\000\001\325\000a\036\000\022\000\001\325\022\201\036\000\n\000\001\325\nq\036\000\023\000\001\325\032I\016\000\021\000\001\325\021+\022\000\t\000\001\325\t\031\016\000\000\b\000\327\000%\002\002\241\001\207\277\022| \325\200\000!\000\023\000\001\325\023K\022\000\000\b\000\327\000\025\002\002\237\361\210\277\003\000\207\277\n| \325\200$\"\000\022\000\001\325\0319\016\000\002\000\001\325\023M\026\000\001\000\001\325\001\007\n\000\t\000\001\325\t\033\022\000~\000\211\276\013\000\001\325\022;\022\000\002\000\001\325\002O\032\000\001\000\001\325\001\t\016\000\t\000\001\325\t\035\026\000\200\001\202\276\003\000\001\325\013=\026\000\013\000\001\325\021-\026\000\002\000\001\325\002Q\036\000\001\000\001\325\001\013\022\000\005\000\001\325\t\037\032\000\003\000\001\325\003?\032\000\004\000\001\325\013/\032\000\000j\000\327\000\005\002\002\235\377\210\277\002| \325\200\024\252\001\003\000\001\325\003A\036\000\004\000\001\325\0041\036\000\001\000\001\325\001\r\026\000\003\000\207\277\000j\000\327\000\007\002\002\235\377\210\277\002| \325\200\004\252\001\003\000\001\325\005!\036\000\001\000\001\325\001\017\032\000\000j\000\327\000\t\002\002\235\377\210\277\002| \325\200\004\252\001\223\001\207\277\001\000\001\325\001\021\036\000\000j\000\327\000\007\002\002\235\377\210\277\023\001\207\277\002| \325\200\004\252\001\000j\000\327\000\003\002\002\235\377\210\277\002\000\207\277\001| \325\200\004\252\001\236\377\210\277\t\b\206\276\236\377\210\277\001\000\207\277\005\000`\327\001\r\000\002\004\000`\327\000\r\000\002\201\006\006\204\236\377\210\277\t\006\t\221\002\004\202\251\363\377\242\277\000\000\037\327~\000\001\002~\000\204\276\001\000\207\277\200\000\224}\236\377\210\277~\004\004\215\t\000\245\277\000 \000\364\000\000\000\370\002\002\000~\200\000\020\312\003\000\000\002\000\000\307\277\000\300\020\356\000\000\b\000\002\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\200\000\000\364\b\000\000\370\207\000\2126\000\000\307\277\002\201\004\277]\000\242\277\200\002r~\377\212\20288888\377\212\2048@@@@\377\212\2068BBBB\377\212\2108DDDD9\001\020\3129\001::9\001\020\3129\001<<9\001\020\3129\001>>9\001\020\3129\0010@9\001\020\3129\001229\001\020\3129\001449\001\020\3129\001669\001\020\3129\001(89\001\020\3129\001**9\001\020\3129\001,,9\001\020\3129\001..9\001\020\3129\001 09\001\020\3129\001\"\"9\001\020\3129\001$$9\001\020\3129\001&&9\001\020\3129\001\030(9\001\020\3129\001\032\0329\001\020\3129\001\034\0349\001\020\3129\001\036\0369\001\020\3129\001\020 9\001\020\3129\001\022\0229\001\020\3129\001\024\0249\001\020\3129\001\026\0269\001\020\3129\001\b\0309\001\020\3129\001\n\n9\001\020\3129\001\f\f9\001\020\3129\001\016\0169\001\020\3129\001\000\0209\001\020\3129\001\002\0029\001\020\3129\001\004\0049\001\020\3129\001\006\0069\003\020~9@F\314A\207\346\0341@F\314A\207\306\034)@F\314A\207\246\034!@F\314A\207\206\034\031@F\314A\207f\034\021@F\314A\207F\034\t@F\314A\207&\034\001@F\314A\207\006\034\002\301\002\201\t\000\207\277\002\200\007\277\354\377\242\277A\000\240\277\200\002\020~\001\000\207\277\b\001\020\312\b\001\006\007\b\001\020\312\b\001\004\005\b\001\020\312\b\001\002\003\b\001\020\312\b\001\020\001\b\001\020\312\b\001\016\017\b\001\020\312\b\001\f\r\b\001\020\312\b\001\n\013\b\001\020\312\b\001\030\t\b\001\020\312\b\001\026\027\b\001\020\312\b\001\024\025\b\001\020\312\b\001\022\023\b\001\020\312\b\001 \021\b\001\020\312\b\001\036\037\b\001\020\312\b\001\034\035\b\001\020\312\b\001\032\033\b\001\020\312\b\001(\031\b\001\020\312\b\001&'\b\001\020\312\b\001$%\b\001\020\312\b\001\"#\b\001\020\312\b\0010!\b\001\020\312\b\001./\b\001\020\312\b\001,-\b\001\020\312\b\001*+\b\001\020\312\b\0018)\b\001\020\312\b\00167\b\001\020\312\b\00145\b\001\020\312\b\00123\b\001\020\312\b\001@1\b\001\020\312\b\001>?\b\001\020\312\b\001<=\b\001\020\312\b\001:;\b\003r~\237\000\0006~\000\202\276\001\000\207\277\200\000\224}\272\000\245\277\201\212\224|\002\000J\324\202\212\002\002\003\000J\324\203\212\002\002\004\000J\324\204\212\002\002\005\000J\324\205\212\002\0029u\000\002\001\005\002\002\006\000J\324\206\212\002\002\007\000J\324\207\212\002\002!EB\002\000\000\001\325\000w\n\000\001\000\001\325\001\007\n\000\03152\002\t\025\022\002~\000\210\276\000\000\001\325\000y\016\000\001\000\001\325\001\t\016\000\031\000\001\325\0317\n\000\223\001\207\277\000\000\001\325\000{\022\000\001\000\001\325\001\013\022\000\223\001\207\277\031\000\001\325\0319\016\000\000\000\001\325\000}\026\000\023\001\207\277\001\000\001\325\001\r\026\000\000\000\001\325\000\177\032\000\022\001\207\277\001\000\001\325\001\017\032\000\000\000\001\325\000\201\036\0001eb\002\223\001\207\277\001\000\001\325\001\021\036\000)UH\312\200\000\000)\243\001\207\277\"\000\001\3251g\n\000\021%\"\002\032\000\001\325)W\n\000\023\001\207\277\"\000\001\325\"i\016\000\022\000\001\325\032Y\016\000\022\001\207\277\032\000\001\325\"k\022\000\n\000\001\325\022[\022\000\242\001\207\277\022\000\001\325\032m\026\000\032\000\001\325!G\n\000\n\000\001\325\n]\026\000\223\001\207\277\022\000\001\325\022o\032\000\032\000\001\325\032I\016\000\223\001\207\277\n\000\001\325\n_\032\000\022\000\001\325\022q\036\000\022\001\207\277\n\000\001\325\na\036\000\000%\000\006\021\000\001\325\021'\n\000\023\000\001\325\032K\022\000\022\000\001\325\031;\022\000$\002\207\277\000\025\000\006\t\000\001\325\t\027\n\000\013\000\001\325\023M\026\000\021\000\001\325\021)\016\000\n\000\001\325\022=\026\000\024\002\207\277\t\000\001\325\t\031\016\000\002\000\001\325\013O\032\000\024\002\207\277\013\000\001\325\021+\022\000\003\000\001\325\n?\032\000\200\001\202\276\t\000\001\325\t\033\022\000\002\000\001\325\002Q\036\000\n\000\001\325\013-\026\000\223\001\207\277\004\000\001\325\t\035\026\000\000\005\000\006\002\000\001\325\003A\036\000\024\001\207\277\003\000\001\325\n/\032\000\000\005\000\006\"\001\207\277\002\000\001\325\0031\036\000\003\000\001\325\004\037\032\000\000\005\000\006\222\000\207\277\002\000\001\325\003!\036\000\000\005\000\006\221\000\207\277\000\003\000\006\000C\000~\221\000\207\277\377\000\002\020\000\000\200/\001I\002~!\001\207\277\001\001\000X\000\000\200\317\001\017\002~\000\017\000~\b\b\206\276\236\377\210\277\022\001\207\277\005\000`\327\001\r\000\002\004\000`\327\000\r\000\002\201\006\006\204\236\377\210\277\b\006\b\221\002\004\202\251\364\377\242\277\000\000\037\327~\000\001\002~\000\204\276\001\000\207\277\200\000\224}\236\377\210\277~\004\004\215\t\000\245\277\000 \000\364\000\000\000\370\002\002\000~\200\000\020\312\003\000\000\002\000\000\307\277\000\300\020\356\000\000\b\000\002\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\200\000\000\364\b\000\000\370\000\000\307\277\002\201\004\277]\000\242\277\200\002\002~\377\000\2028\021\021\021\021\377\000\2048\"\"\"\"\377\000\20683333\377\000\2108DDDD\001\001\020\312\001\001\002\002\001\001\020\312\001\001\004\004\001\001\020\312\001\001\006\006\001\001\020\312\001\001\b\b\001\001\020\312\001\001\n\n\001\001\020\312\001\001\f\f\001\001\020\312\001\001\016\016\001\001\020\312\001\001\020\020\001\001\020\312\001\001\022\022\001\001\020\312\001\001\024\024\001\001\020\312\001\001\026\026\001\001\020\312\001\001\030\030\001\001\020\312\001\001\032\032\001\001\020\312\001\001\034\034\001\001\020\312\001\001\036\036\001\001\020\312\001\001  \001\001\020\312\001\001\"\"\001\001\020\312\001\001$$\001\001\020\312\001\001&&\001\001\020\312\001\001((\001\001\020\312\001\001**\001\001\020\312\001\001,,\001\001\020\312\001\001..\001\001\020\312\001\00100\001\001\020\312\001\00122\001\001\020\312\001\00144\001\001\020\312\001\00166\001\001\020\312\001\00188\001\001\020\312\001\001::\001\001\020\312\001\001<<\001\001\020\312\001\001>>\001\003\200~\001@J\314A\207\006|\t@J\314A\207&|\021@J\314A\207F|\031@J\314A\207f|!@J\314A\207\206|)@J\314A\207\246|1@J\314A\207\306|9@J\314A\207\346|\002\301\002\201\t\000\207\277\002\200\007\277\354\377\242\277\b\000\240\277\200\002r~\200\002b~\200\002R~\200\002B~\200\0022~\200\002\"~\200\002\022~\200\002\002~\237\000\0006~\000\202\276\001\000\207\277\200\000\224}F\000\245\277\000\002\000\327\001\023\002\002\261\000\207\277\001| \325\200\000\t\000~\000\204\276\000j\000\327\000#\002\002\001| \325\200\002\252\001\200\001\202\276\000j\000\327\0003\002\002\235\377\210\277\001| \325\200\002\252\001\"\001\207\277\000j\000\327\000C\002\002\235\377\210\277\001| \325\200\002\252\001\"\001\207\277\000j\000\327\000S\002\002\235\377\210\277\001| \325\200\002\252\001\"\001\207\277\000j\000\327\000c\002\002\235\377\210\277\001| \325\200\002\252\001\"\001\207\277\000j\000\327\000s\002\002\235\377\210\277\001| \325\200\002\252\001\236\377\210\277\004\b\205\276\236\377\210\277\001\000\207\277\007\000`\327\001\013\000\002\006\000`\327\000\013\000\002\201\005\005\204\236\377\210\277\004\005\004\221\002\006\202\251\363\377\242\277\000\000\037\327~\000\001\002~\000\204\276\001\000\207\277\200\000\224}\236\377\210\277~\004\004\215\t\000\245\277\000 \000\364\000\000\000\370\002\002\000~\200\000\020\312\003\000\000\002\000\000\307\277\000\300\020\356\000\000\b\000\002\000\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000X\n\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\354\013\000\000\000\000\000\000\n\000\000\000\000\000\000\000q\000\000\000\000\000\000\000\365\376\377o\000\000\000\000H\013\000\000\000\000\000\000\004\000\000\000\000\000\000\000\224\013\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 23.0.0 (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000)\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\312\000\000\000\000\002\t\000\200B\000\000\000\000\000\000\000\000\000\000\000\000\000\000Z\000\000\000\022\003\b\000\000\037\000\000\000\000\000\000\324\002\000\000\000\000\000\000c\000\000\000\021\003\006\000\200\f\000\000\000\000\000\000@\000\000\000\000\000\000\000o\000\000\000\022\003\b\000\000\"\000\000\000\000\000\000\234\005\000\000\000\000\000\000x\000\000\000\021\003\006\000\300\f\000\000\000\000\000\000@\000\000\000\000\000\000\000\204\000\000\000\022\003\b\000\000(\000\000\000\000\000\000\220\005\000\000\000\000\000\000\215\000\000\000\021\003\006\000\000\r\000\000\000\000\000\000@\000\000\000\000\000\000\000\231\000\000\000\022\003\b\000\000.\000\000\000\000\000\000\330\002\000\000\000\000\000\000\242\000\000\000\021\003\006\000@\r\000\000\000\000\000\000@\000\000\000\000\000\000\000\256\000\000\000\021\000\013\000\360R\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.eh_frame\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000amdgpu.max_num_named_barrier\000peak_f16\000peak_f16.kd\000peak_iu8\000peak_iu8.kd\000peak_fp8\000peak_fp8.kd\000peak_iu4\000peak_iu4.kd\000__hip_cuid_dec4f8244316d2d0\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\034\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000X\n\000\000\000\000\000\000X\n\000\000\000\000\000\000\360\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000H\013\000\000\000\000\000\000H\013\000\000\000\000\000\000L\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000\224\013\000\000\000\000\000\000\224\013\000\000\000\000\000\000X\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\354\013\000\000\000\000\000\000\354\013\000\000\000\000\000\000q\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\200\f\000\000\000\000\000\000\200\f\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\200\r\000\000\000\000\000\000\200\r\000\000\000\000\000\000\214\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0009\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\037\000\000\000\000\000\000\000\017\000\000\000\000\000\000\200\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000?\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\200B\000\000\000\000\000\000\200\"\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000H\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360B\000\000\000\000\000\000\360\"\000\000\000\000\000\000\020\r\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000W\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360R\000\000\000\000\000\000\360\"\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\\\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\"\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000k\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\"\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\"\000\000\000\000\000\000\334\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\211\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320#\000\000\000\000\000\000h\001\000\000\000\000\000\000\021\000\000\000\006\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0008%\000\000\000\000\000\000\243\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\233\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\333%\000\000\000\000\000\000\323\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_5, 15152

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_5
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_dec4f8244316d2d0,@object # @__hip_gpubin_handle_dec4f8244316d2d0
	.local	__hip_gpubin_handle_dec4f8244316d2d0
	.comm	__hip_gpubin_handle_dec4f8244316d2d0,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_dec4f8244316d2d0,@object # @__hip_cuid_dec4f8244316d2d0
	.bss
	.globl	__hip_cuid_dec4f8244316d2d0
__hip_cuid_dec4f8244316d2d0:
	.byte	0                               # 0x0
	.size	__hip_cuid_dec4f8244316d2d0, 1

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
	.addrsig_sym __device_stub__peak_f16
	.addrsig_sym __device_stub__peak_iu8
	.addrsig_sym __device_stub__peak_fp8
	.addrsig_sym __device_stub__peak_iu4
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym peak_f16
	.addrsig_sym peak_iu8
	.addrsig_sym peak_fp8
	.addrsig_sym peak_iu4
	.addrsig_sym .L__unnamed_5
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_dec4f8244316d2d0
