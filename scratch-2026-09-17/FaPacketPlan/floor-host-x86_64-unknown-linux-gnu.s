	.att_syntax
	.file	"floor.hip"
	.text
	.globl	_Z6launchiiiiPKjPfi             # -- Begin function _Z6launchiiiiPKjPfi
	.prefalign	4, .Lfunc_end0, nop
	.type	_Z6launchiiiiPKjPfi,@function
_Z6launchiiiiPKjPfi:                    # @_Z6launchiiiiPKjPfi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$112, %rsp
	.cfi_def_cfa_offset 144
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	cmpl	$4, %edi
	ja	.LBB0_13
# %bb.1:
	movq	%r9, %rbx
	movq	%r8, %r14
	movl	144(%rsp), %ebp
	movl	%edi, %eax
	leaq	.LJTI0_0(%rip), %rdi
	movslq	(%rdi,%rax,4), %rax
	addq	%rdi, %rax
	jmpq	*%rax
.LBB0_2:
	movslq	%ecx, %r8
	movl	%esi, %edi
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rdi
	movl	%edx, %edx
	orq	%rax, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.3:
	movq	%r14, 72(%rsp)
	movq	%rbx, 64(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi0EEvPKjPfi@GOTPCREL(%rip), %rdi
	jmp	.LBB0_12
.LBB0_10:
	movslq	%ecx, %r8
	movl	%esi, %edi
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rdi
	movl	%edx, %edx
	orq	%rax, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.11:
	movq	%r14, 72(%rsp)
	movq	%rbx, 64(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi4EEvPKjPfi@GOTPCREL(%rip), %rdi
	jmp	.LBB0_12
.LBB0_6:
	movslq	%ecx, %r8
	movl	%esi, %edi
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rdi
	movl	%edx, %edx
	orq	%rax, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.7:
	movq	%r14, 72(%rsp)
	movq	%rbx, 64(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi2EEvPKjPfi@GOTPCREL(%rip), %rdi
	jmp	.LBB0_12
.LBB0_8:
	movslq	%ecx, %r8
	movl	%esi, %edi
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rdi
	movl	%edx, %edx
	orq	%rax, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.9:
	movq	%r14, 72(%rsp)
	movq	%rbx, 64(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi3EEvPKjPfi@GOTPCREL(%rip), %rdi
	jmp	.LBB0_12
.LBB0_4:
	movslq	%ecx, %r8
	movl	%esi, %edi
	movabsq	$4294967296, %rax               # imm = 0x100000000
	orq	%rax, %rdi
	movl	%edx, %edx
	orq	%rax, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.5:
	movq	%r14, 72(%rsp)
	movq	%rbx, 64(%rsp)
	movl	%ebp, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi1EEvPKjPfi@GOTPCREL(%rip), %rdi
.LBB0_12:
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB0_13:
	callq	hipGetLastError@PLT
	testl	%eax, %eax
	jne	.LBB0_15
# %bb.14:
	addq	$112, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_15:
	.cfi_def_cfa_offset 144
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.1(%rip), %rdx
	movq	%rbx, %rdi
	movq	%rax, %rcx
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$2, %edi
	callq	exit@PLT
.Lfunc_end0:
	.size	_Z6launchiiiiPKjPfi, .Lfunc_end0-_Z6launchiiiiPKjPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.LJTI0_0:
	.long	.LBB0_2-.LJTI0_0
	.long	.LBB0_4-.LJTI0_0
	.long	.LBB0_6-.LJTI0_0
	.long	.LBB0_8-.LJTI0_0
	.long	.LBB0_10-.LJTI0_0
                                        # -- End function
	.section	.text._Z27__device_stub__floor_kernelILi0EEvPKjPfi,"axG",@progbits,_Z27__device_stub__floor_kernelILi0EEvPKjPfi,comdat
	.weak	_Z27__device_stub__floor_kernelILi0EEvPKjPfi # -- Begin function _Z27__device_stub__floor_kernelILi0EEvPKjPfi
	.prefalign	4, .Lfunc_end1, nop
	.type	_Z27__device_stub__floor_kernelILi0EEvPKjPfi,@function
_Z27__device_stub__floor_kernelILi0EEvPKjPfi: # @_Z27__device_stub__floor_kernelILi0EEvPKjPfi
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi0EEvPKjPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end1:
	.size	_Z27__device_stub__floor_kernelILi0EEvPKjPfi, .Lfunc_end1-_Z27__device_stub__floor_kernelILi0EEvPKjPfi
	.cfi_endproc
                                        # -- End function
	.section	.text._Z27__device_stub__floor_kernelILi1EEvPKjPfi,"axG",@progbits,_Z27__device_stub__floor_kernelILi1EEvPKjPfi,comdat
	.weak	_Z27__device_stub__floor_kernelILi1EEvPKjPfi # -- Begin function _Z27__device_stub__floor_kernelILi1EEvPKjPfi
	.prefalign	4, .Lfunc_end2, nop
	.type	_Z27__device_stub__floor_kernelILi1EEvPKjPfi,@function
_Z27__device_stub__floor_kernelILi1EEvPKjPfi: # @_Z27__device_stub__floor_kernelILi1EEvPKjPfi
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi1EEvPKjPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end2:
	.size	_Z27__device_stub__floor_kernelILi1EEvPKjPfi, .Lfunc_end2-_Z27__device_stub__floor_kernelILi1EEvPKjPfi
	.cfi_endproc
                                        # -- End function
	.section	.text._Z27__device_stub__floor_kernelILi2EEvPKjPfi,"axG",@progbits,_Z27__device_stub__floor_kernelILi2EEvPKjPfi,comdat
	.weak	_Z27__device_stub__floor_kernelILi2EEvPKjPfi # -- Begin function _Z27__device_stub__floor_kernelILi2EEvPKjPfi
	.prefalign	4, .Lfunc_end3, nop
	.type	_Z27__device_stub__floor_kernelILi2EEvPKjPfi,@function
_Z27__device_stub__floor_kernelILi2EEvPKjPfi: # @_Z27__device_stub__floor_kernelILi2EEvPKjPfi
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi2EEvPKjPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end3:
	.size	_Z27__device_stub__floor_kernelILi2EEvPKjPfi, .Lfunc_end3-_Z27__device_stub__floor_kernelILi2EEvPKjPfi
	.cfi_endproc
                                        # -- End function
	.section	.text._Z27__device_stub__floor_kernelILi3EEvPKjPfi,"axG",@progbits,_Z27__device_stub__floor_kernelILi3EEvPKjPfi,comdat
	.weak	_Z27__device_stub__floor_kernelILi3EEvPKjPfi # -- Begin function _Z27__device_stub__floor_kernelILi3EEvPKjPfi
	.prefalign	4, .Lfunc_end4, nop
	.type	_Z27__device_stub__floor_kernelILi3EEvPKjPfi,@function
_Z27__device_stub__floor_kernelILi3EEvPKjPfi: # @_Z27__device_stub__floor_kernelILi3EEvPKjPfi
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi3EEvPKjPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end4:
	.size	_Z27__device_stub__floor_kernelILi3EEvPKjPfi, .Lfunc_end4-_Z27__device_stub__floor_kernelILi3EEvPKjPfi
	.cfi_endproc
                                        # -- End function
	.section	.text._Z27__device_stub__floor_kernelILi4EEvPKjPfi,"axG",@progbits,_Z27__device_stub__floor_kernelILi4EEvPKjPfi,comdat
	.weak	_Z27__device_stub__floor_kernelILi4EEvPKjPfi # -- Begin function _Z27__device_stub__floor_kernelILi4EEvPKjPfi
	.prefalign	4, .Lfunc_end5, nop
	.type	_Z27__device_stub__floor_kernelILi4EEvPKjPfi,@function
_Z27__device_stub__floor_kernelILi4EEvPKjPfi: # @_Z27__device_stub__floor_kernelILi4EEvPKjPfi
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12floor_kernelILi4EEvPKjPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end5:
	.size	_Z27__device_stub__floor_kernelILi4EEvPKjPfi, .Lfunc_end5-_Z27__device_stub__floor_kernelILi4EEvPKjPfi
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI6_0:
	.quad	2                               # 0x2
	.quad	3                               # 0x3
.LCPI6_1:
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
.LCPI6_2:
	.long	13                              # 0xd
	.long	13                              # 0xd
	.long	13                              # 0xd
	.long	13                              # 0xd
.LCPI6_3:
	.long	2863311531                      # 0xaaaaaaab
	.long	2863311531                      # 0xaaaaaaab
	.long	2863311531                      # 0xaaaaaaab
	.long	2863311531                      # 0xaaaaaaab
.LCPI6_4:
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
.LCPI6_5:
	.long	3                               # 0x3
	.long	3                               # 0x3
	.long	3                               # 0x3
	.long	3                               # 0x3
.LCPI6_6:
	.long	6                               # 0x6
	.long	6                               # 0x6
	.long	6                               # 0x6
	.long	6                               # 0x6
.LCPI6_7:
	.long	9                               # 0x9
	.long	9                               # 0x9
	.long	9                               # 0x9
	.long	9                               # 0x9
.LCPI6_8:
	.zero	16,32
.LCPI6_9:
	.quad	4                               # 0x4
	.quad	4                               # 0x4
.LCPI6_11:
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
.LCPI6_12:
	.quad	0x7fffffffffffffff              # double NaN
	.quad	0x7fffffffffffffff              # double NaN
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI6_10:
	.long	0x447a0000                      # float 1000
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI6_13:
	.quad	0x39b4484bfeebc2a0              # double 1.0000000000000001E-30
	.text
	.globl	main
	.prefalign	4, .Lfunc_end6, nop
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
	subq	$2056, %rsp                     # imm = 0x808
	.cfi_def_cfa_offset 2112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$32768, %r15d                   # imm = 0x8000
	movl	$64, %r12d
	cmpl	$2, %edi
	jl	.LBB6_2
# %bb.1:
	movq	%rsi, %rbx
	movl	%edi, %ebp
	movq	8(%rsi), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	__isoc23_strtol@PLT
	movq	%rax, %r12
	cmpl	$2, %ebp
	jne	.LBB6_85
.LBB6_2:
	movl	$128, %r14d
.LBB6_3:                                # %.thread299
	movl	$128, %ebp
.LBB6_4:                                # %.thread299
	leaq	584(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600@PLT
	testl	%eax, %eax
	jne	.LBB6_103
# %bb.5:
	leaq	1744(%rsp), %rsi
	movl	972(%rsp), %edx
	movl	%r15d, (%rsp)
	leaq	.L.str.3(%rip), %rdi
	movl	%r12d, %ecx
	movq	%r14, %r13
	movl	%r13d, %r8d
	movl	%ebp, %r9d
	xorl	%eax, %eax
	callq	printf@PLT
	movslq	%r15d, %rbx
	movq	_Z12floor_kernelILi0EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	140(%rsp), %rdi
	movl	%r13d, %edx
	movq	%rbx, %rcx
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
	testl	%eax, %eax
	jne	.LBB6_104
# %bb.6:
	movq	_Z12floor_kernelILi1EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	136(%rsp), %rdi
	movl	%r14d, %edx
	movq	%rbx, %rcx
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
	testl	%eax, %eax
	jne	.LBB6_105
# %bb.7:
	movq	_Z12floor_kernelILi2EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	132(%rsp), %rdi
	movl	%r14d, %edx
	movq	%rbx, %rcx
	callq	hipOccupancyMaxActiveBlocksPerMultiprocessor@PLT
	testl	%eax, %eax
	jne	.LBB6_106
# %bb.8:                                # %vector.ph
	movl	140(%rsp), %esi
	movl	136(%rsp), %edx
	movl	132(%rsp), %ecx
	leaq	.L.str.7(%rip), %rdi
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	callq	printf@PLT
	movaps	.LCPI6_0(%rip), %xmm0           # xmm0 = [2,3]
	movaps	.LCPI6_1(%rip), %xmm1           # xmm1 = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
	movdqa	.LCPI6_2(%rip), %xmm2           # xmm2 = [13,13,13,13]
	movdqa	.LCPI6_3(%rip), %xmm3           # xmm3 = [2863311531,2863311531,2863311531,2863311531]
	movdqa	.LCPI6_4(%rip), %xmm4           # xmm4 = [24,24,24,24]
	movdqa	.LCPI6_5(%rip), %xmm5           # xmm5 = [3,3,3,3]
	movdqa	.LCPI6_6(%rip), %xmm6           # xmm6 = [6,6,6,6]
	movdqa	.LCPI6_7(%rip), %xmm7           # xmm7 = [9,9,9,9]
	movdqa	.LCPI6_8(%rip), %xmm8           # xmm8 = [538976288,538976288,538976288,538976288]
	movdqa	.LCPI6_9(%rip), %xmm9           # xmm9 = [4,4]
	.p2align	4
.LBB6_9:                                # %vector.body
                                        # =>This Inner Loop Header: Depth=1
	movaps	%xmm1, %xmm10
	shufps	$136, %xmm0, %xmm10             # xmm10 = xmm10[0,2],xmm0[0,2]
	movaps	%xmm1, %xmm11
	shufps	$170, %xmm0, %xmm11             # xmm11 = xmm11[2,2],xmm0[2,2]
	pmuludq	%xmm2, %xmm11
	pshufd	$232, %xmm11, %xmm12            # xmm12 = xmm11[0,2,2,3]
	pmuludq	%xmm2, %xmm10
	pshufd	$232, %xmm10, %xmm10            # xmm10 = xmm10[0,2,2,3]
	punpckldq	%xmm12, %xmm10          # xmm10 = xmm10[0],xmm12[0],xmm10[1],xmm12[1]
	movdqa	%xmm10, %xmm12
	pmuludq	%xmm3, %xmm12
	pshufd	$237, %xmm12, %xmm12            # xmm12 = xmm12[1,3,2,3]
	pshufd	$160, %xmm11, %xmm11            # xmm11 = xmm11[0,0,2,2]
	pmuludq	%xmm3, %xmm11
	pshufd	$237, %xmm11, %xmm11            # xmm11 = xmm11[1,3,2,3]
	punpckldq	%xmm11, %xmm12          # xmm12 = xmm12[0],xmm11[0],xmm12[1],xmm11[1]
	psrld	$4, %xmm12
	pshufd	$245, %xmm12, %xmm11            # xmm11 = xmm12[1,1,3,3]
	pmuludq	%xmm4, %xmm12
	pshufd	$232, %xmm12, %xmm12            # xmm12 = xmm12[0,2,2,3]
	pmuludq	%xmm4, %xmm11
	pshufd	$232, %xmm11, %xmm11            # xmm11 = xmm11[0,2,2,3]
	punpckldq	%xmm11, %xmm12          # xmm12 = xmm12[0],xmm11[0],xmm12[1],xmm11[1]
	movdqa	%xmm10, %xmm13
	psubd	%xmm12, %xmm13
	movdqa	%xmm10, %xmm11
	paddd	%xmm5, %xmm11
	movdqa	%xmm11, %xmm12
	pmuludq	%xmm3, %xmm12
	pshufd	$237, %xmm12, %xmm12            # xmm12 = xmm12[1,3,2,3]
	pshufd	$245, %xmm11, %xmm14            # xmm14 = xmm11[1,1,3,3]
	pmuludq	%xmm3, %xmm14
	pshufd	$237, %xmm14, %xmm14            # xmm14 = xmm14[1,3,2,3]
	punpckldq	%xmm14, %xmm12          # xmm12 = xmm12[0],xmm14[0],xmm12[1],xmm14[1]
	psrld	$4, %xmm12
	pshufd	$245, %xmm12, %xmm14            # xmm14 = xmm12[1,1,3,3]
	pmuludq	%xmm4, %xmm12
	pshufd	$232, %xmm12, %xmm12            # xmm12 = xmm12[0,2,2,3]
	pmuludq	%xmm4, %xmm14
	pshufd	$232, %xmm14, %xmm14            # xmm14 = xmm14[0,2,2,3]
	punpckldq	%xmm14, %xmm12          # xmm12 = xmm12[0],xmm14[0],xmm12[1],xmm14[1]
	psubd	%xmm12, %xmm11
	pslld	$8, %xmm11
	por	%xmm13, %xmm11
	movdqa	%xmm10, %xmm12
	paddd	%xmm6, %xmm12
	movdqa	%xmm12, %xmm13
	pmuludq	%xmm3, %xmm13
	pshufd	$237, %xmm13, %xmm13            # xmm13 = xmm13[1,3,2,3]
	pshufd	$245, %xmm12, %xmm14            # xmm14 = xmm12[1,1,3,3]
	pmuludq	%xmm3, %xmm14
	pshufd	$237, %xmm14, %xmm14            # xmm14 = xmm14[1,3,2,3]
	punpckldq	%xmm14, %xmm13          # xmm13 = xmm13[0],xmm14[0],xmm13[1],xmm14[1]
	psrld	$4, %xmm13
	pshufd	$245, %xmm13, %xmm14            # xmm14 = xmm13[1,1,3,3]
	pmuludq	%xmm4, %xmm13
	pshufd	$232, %xmm13, %xmm13            # xmm13 = xmm13[0,2,2,3]
	pmuludq	%xmm4, %xmm14
	pshufd	$232, %xmm14, %xmm14            # xmm14 = xmm14[0,2,2,3]
	punpckldq	%xmm14, %xmm13          # xmm13 = xmm13[0],xmm14[0],xmm13[1],xmm14[1]
	psubd	%xmm13, %xmm12
	pslld	$16, %xmm12
	paddd	%xmm7, %xmm10
	movdqa	%xmm10, %xmm13
	pmuludq	%xmm3, %xmm13
	pshufd	$237, %xmm13, %xmm13            # xmm13 = xmm13[1,3,2,3]
	pshufd	$245, %xmm10, %xmm14            # xmm14 = xmm10[1,1,3,3]
	pmuludq	%xmm3, %xmm14
	pshufd	$237, %xmm14, %xmm14            # xmm14 = xmm14[1,3,2,3]
	punpckldq	%xmm14, %xmm13          # xmm13 = xmm13[0],xmm14[0],xmm13[1],xmm14[1]
	psrld	$4, %xmm13
	pshufd	$245, %xmm13, %xmm14            # xmm14 = xmm13[1,1,3,3]
	pmuludq	%xmm4, %xmm13
	pshufd	$232, %xmm13, %xmm13            # xmm13 = xmm13[0,2,2,3]
	pmuludq	%xmm4, %xmm14
	pshufd	$232, %xmm14, %xmm14            # xmm14 = xmm14[0,2,2,3]
	punpckldq	%xmm14, %xmm13          # xmm13 = xmm13[0],xmm14[0],xmm13[1],xmm14[1]
	psubd	%xmm13, %xmm10
	pslld	$24, %xmm10
	por	%xmm12, %xmm10
	por	%xmm11, %xmm10
	por	%xmm8, %xmm10
	movdqa	%xmm10, 320(%rsp,%rbx,4)
	addq	$4, %rbx
	paddq	%xmm9, %xmm1
	paddq	%xmm9, %xmm0
	cmpq	$64, %rbx
	jne	.LBB6_9
# %bb.10:                               # %middle.block
	leaq	24(%rsp), %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB6_107
# %bb.11:
	movq	24(%rsp), %rdi
	leaq	320(%rsp), %rsi
	movl	$256, %edx                      # imm = 0x100
	movl	$1, %ecx
	callq	hipMemcpy@PLT
	testl	%eax, %eax
	jne	.LBB6_108
# %bb.12:
	movslq	%r12d, %rax
	movslq	%r14d, %rbx
	imulq	%rax, %rbx
	imulq	$560, %rbx, %rsi                # imm = 0x230
	leaq	16(%rsp), %rdi
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	callq	hipMalloc@PLT
	testl	%eax, %eax
	jne	.LBB6_109
# %bb.13:                               # %.preheader347.preheader
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	xorl	%edi, %edi
	movl	%r12d, %esi
	movq	%r14, %r13
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$1, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$2, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$3, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$4, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	xorl	%edi, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$1, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$2, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$3, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$4, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	xorl	%edi, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$1, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$2, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$3, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$4, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	xorl	%edi, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$1, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$2, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$3, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	movl	%ebp, (%rsp)
	movl	$4, %edi
	movl	%r12d, %esi
	movl	%r13d, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
	callq	hipDeviceSynchronize@PLT
	testl	%eax, %eax
	jne	.LBB6_110
# %bb.14:
	leaq	152(%rsp), %rdi
	callq	hipEventCreate@PLT
	testl	%eax, %eax
	jne	.LBB6_111
# %bb.15:
	movq	%r14, 112(%rsp)                 # 8-byte Spill
	leaq	120(%rsp), %rdi
	callq	hipEventCreate@PLT
	testl	%eax, %eax
	jne	.LBB6_112
# %bb.16:                               # %.preheader320.preheader
	imulq	$140, %rbx, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	xorps	%xmm0, %xmm0
	movaps	%xmm0, 256(%rsp)
	movaps	%xmm0, 240(%rsp)
	movaps	%xmm0, 224(%rsp)
	movaps	%xmm0, 208(%rsp)
	movaps	%xmm0, 192(%rsp)
	movaps	%xmm0, 176(%rsp)
	movaps	%xmm0, 160(%rsp)
	movq	$0, 272(%rsp)
	xorl	%ebx, %ebx
	movq	%rbp, 144(%rsp)                 # 8-byte Spill
	movq	%r15, 312(%rsp)                 # 8-byte Spill
	movq	%r12, 304(%rsp)                 # 8-byte Spill
	jmp	.LBB6_18
	.p2align	4
.LBB6_17:                               #   in Loop: Header=BB6_18 Depth=1
	incl	%ebx
	cmpl	$15, %ebx
	je	.LBB6_38
.LBB6_18:                               # %.preheader320
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB6_21 Depth 2
	movl	$4, %eax
	xorl	%r13d, %r13d
	movq	%rbx, 40(%rsp)                  # 8-byte Spill
	jmp	.LBB6_21
	.p2align	4
.LBB6_19:                               #   in Loop: Header=BB6_21 Depth=2
	movsd	%xmm0, (%rbx)
	addq	$8, %rbx
	movq	%rbx, 8(%rcx)
.LBB6_20:                               # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit
                                        #   in Loop: Header=BB6_21 Depth=2
	leaq	.L.str.18(%rip), %rdi
	movq	40(%rsp), %rbx                  # 8-byte Reload
	movl	%ebx, %esi
	movl	%r14d, %edx
	movb	$1, %al
	callq	printf@PLT
	incl	%r13d
	movl	32(%rsp), %eax                  # 4-byte Reload
	decl	%eax
	cmpl	$5, %r13d
	je	.LBB6_17
.LBB6_21:                               #   Parent Loop BB6_18 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	testb	$1, %bl
	movl	%r13d, %r14d
	movl	%eax, 32(%rsp)                  # 4-byte Spill
	cmovel	%eax, %r14d
	movq	152(%rsp), %rdi
.Ltmp0:                                 # EH_LABEL
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.22:                               #   in Loop: Header=BB6_21 Depth=2
	testl	%eax, %eax
	jne	.LBB6_94
# %bb.23:                               #   in Loop: Header=BB6_21 Depth=2
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
.Ltmp6:                                 # EH_LABEL
	movl	%ebp, (%rsp)
	movl	%r14d, %edi
	movl	%r12d, %esi
	movq	112(%rsp), %rdx                 # 8-byte Reload
                                        # kill: def $edx killed $edx killed $rdx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
.Ltmp7:                                 # EH_LABEL
# %bb.24:                               # %_ZZ4mainENKUliE_clEi.exit
                                        #   in Loop: Header=BB6_21 Depth=2
	movq	120(%rsp), %rdi
.Ltmp9:                                 # EH_LABEL
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp10:                                # EH_LABEL
# %bb.25:                               #   in Loop: Header=BB6_21 Depth=2
	testl	%eax, %eax
	jne	.LBB6_96
# %bb.26:                               #   in Loop: Header=BB6_21 Depth=2
	movq	120(%rsp), %rdi
.Ltmp15:                                # EH_LABEL
	callq	hipEventSynchronize@PLT
.Ltmp16:                                # EH_LABEL
# %bb.27:                               #   in Loop: Header=BB6_21 Depth=2
	testl	%eax, %eax
	jne	.LBB6_90
# %bb.28:                               #   in Loop: Header=BB6_21 Depth=2
	movq	152(%rsp), %rsi
	movq	120(%rsp), %rdx
.Ltmp21:                                # EH_LABEL
	leaq	76(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp22:                                # EH_LABEL
# %bb.29:                               #   in Loop: Header=BB6_21 Depth=2
	testl	%eax, %eax
	jne	.LBB6_92
# %bb.30:                               #   in Loop: Header=BB6_21 Depth=2
	movl	%r14d, %eax
	leaq	(%rax,%rax,2), %rax
	movss	76(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI6_10(%rip), %xmm0
	cvtss2sd	%xmm0, %xmm0
	leaq	(%rsp,%rax,8), %rcx
	addq	$160, %rcx
	movq	168(%rsp,%rax,8), %rbx
	cmpq	176(%rsp,%rax,8), %rbx
	jne	.LBB6_19
# %bb.31:                               #   in Loop: Header=BB6_21 Depth=2
	movq	(%rcx), %r12
	subq	%r12, %rbx
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %rbx
	je	.LBB6_99
# %bb.32:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB6_21 Depth=2
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movsd	%xmm0, 48(%rsp)                 # 8-byte Spill
	movq	%rbx, %r15
	sarq	$3, %r15
	cmpq	$1, %r15
	adcq	%r15, %r15
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r15
	cmovaeq	%rax, %r15
	leaq	(,%r15,8), %rdi
.Ltmp27:                                # EH_LABEL
	callq	_Znwm@PLT
.Ltmp28:                                # EH_LABEL
# %bb.33:                               # %.noexc185
                                        #   in Loop: Header=BB6_21 Depth=2
	movq	%rax, %rbp
	movsd	48(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%rbx)
	testq	%rbx, %rbx
	jle	.LBB6_35
# %bb.34:                               #   in Loop: Header=BB6_21 Depth=2
	movq	%rbp, %rdi
	movq	%r12, %rsi
	movq	%rbx, %rdx
	callq	memcpy@PLT
	movsd	48(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
.LBB6_35:                               # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB6_21 Depth=2
	testq	%r12, %r12
	je	.LBB6_37
# %bb.36:                               # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i
                                        #   in Loop: Header=BB6_21 Depth=2
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	movss	76(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	mulss	.LCPI6_10(%rip), %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBB6_37:                               # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB6_21 Depth=2
	addq	%rbp, %rbx
	addq	$8, %rbx
	movq	64(%rsp), %rcx                  # 8-byte Reload
	movq	%rbp, (%rcx)
	movq	%rbx, 8(%rcx)
	leaq	(,%r15,8), %rax
	addq	%rbp, %rax
	movq	%rax, 16(%rcx)
	movq	312(%rsp), %r15                 # 8-byte Reload
	movq	304(%rsp), %r12                 # 8-byte Reload
	movq	144(%rsp), %rbp                 # 8-byte Reload
	jmp	.LBB6_20
.LBB6_38:                               # %.preheader319.preheader
	movq	160(%rsp), %r14
	movq	168(%rsp), %rbx
	cmpq	%rbx, %r14
	je	.LBB6_41
# %bb.39:
	movq	%rbx, %rax
	subq	%r14, %rax
	sarq	$3, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp30:                                # EH_LABEL
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp31:                                # EH_LABEL
# %bb.40:                               # %.noexc190
.Ltmp32:                                # EH_LABEL
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp33:                                # EH_LABEL
.LBB6_41:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEvT_S7_.exit
	movsd	(%r14), %xmm1                   # xmm1 = mem[0],zero
	movsd	56(%r14), %xmm0                 # xmm0 = mem[0],zero
	movq	112(%r14), %xmm2                # xmm2 = mem[0],zero
	leaq	.L.str.24(%rip), %rdi
	leaq	.L.str.19(%rip), %rsi
	movb	$3, %al
	callq	printf@PLT
	movq	184(%rsp), %r13
	movq	192(%rsp), %rbx
	cmpq	%rbx, %r13
	je	.LBB6_44
# %bb.42:
	movq	%rbx, %rax
	subq	%r13, %rax
	sarq	$3, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp34:                                # EH_LABEL
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp35:                                # EH_LABEL
# %bb.43:                               # %.noexc190.1
.Ltmp36:                                # EH_LABEL
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp37:                                # EH_LABEL
.LBB6_44:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEvT_S7_.exit.1
	movsd	(%r13), %xmm1                   # xmm1 = mem[0],zero
	movsd	56(%r13), %xmm0                 # xmm0 = mem[0],zero
	movq	112(%r13), %xmm2                # xmm2 = mem[0],zero
	leaq	.L.str.24(%rip), %rdi
	leaq	.L.str.20(%rip), %rsi
	movb	$3, %al
	callq	printf@PLT
	movq	208(%rsp), %rax
	movq	216(%rsp), %rbx
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	cmpq	%rbx, %rax
	je	.LBB6_47
# %bb.45:
	movq	%rbx, %rax
	movq	48(%rsp), %rdi                  # 8-byte Reload
	subq	%rdi, %rax
	sarq	$3, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp38:                                # EH_LABEL
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp39:                                # EH_LABEL
# %bb.46:                               # %.noexc190.2
.Ltmp40:                                # EH_LABEL
	movq	48(%rsp), %rdi                  # 8-byte Reload
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp41:                                # EH_LABEL
.LBB6_47:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEvT_S7_.exit.2
	movq	48(%rsp), %rax                  # 8-byte Reload
	movsd	(%rax), %xmm1                   # xmm1 = mem[0],zero
	movsd	56(%rax), %xmm0                 # xmm0 = mem[0],zero
	movq	112(%rax), %xmm2                # xmm2 = mem[0],zero
	leaq	.L.str.24(%rip), %rdi
	leaq	.L.str.21(%rip), %rsi
	movb	$3, %al
	callq	printf@PLT
	movq	232(%rsp), %rax
	movq	240(%rsp), %rbx
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	cmpq	%rbx, %rax
	je	.LBB6_50
# %bb.48:
	movq	%rbx, %rax
	movq	64(%rsp), %rdi                  # 8-byte Reload
	subq	%rdi, %rax
	sarq	$3, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp42:                                # EH_LABEL
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp43:                                # EH_LABEL
# %bb.49:                               # %.noexc190.3
.Ltmp44:                                # EH_LABEL
	movq	64(%rsp), %rdi                  # 8-byte Reload
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp45:                                # EH_LABEL
.LBB6_50:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEvT_S7_.exit.3
	movq	64(%rsp), %rax                  # 8-byte Reload
	movsd	(%rax), %xmm1                   # xmm1 = mem[0],zero
	movsd	56(%rax), %xmm0                 # xmm0 = mem[0],zero
	movq	112(%rax), %xmm2                # xmm2 = mem[0],zero
	leaq	.L.str.24(%rip), %rdi
	leaq	.L.str.22(%rip), %rsi
	movb	$3, %al
	callq	printf@PLT
	movq	256(%rsp), %rcx
	movq	264(%rsp), %rbx
	cmpq	%rbx, %rcx
	movq	%rcx, 80(%rsp)                  # 8-byte Spill
	je	.LBB6_53
# %bb.51:
	movq	%rbx, %rax
	subq	%rcx, %rax
	sarq	$3, %rax
	bsrq	%rax, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp46:                                # EH_LABEL
	movq	80(%rsp), %rdi                  # 8-byte Reload
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp47:                                # EH_LABEL
# %bb.52:                               # %.noexc190.4
.Ltmp48:                                # EH_LABEL
	movq	80(%rsp), %rdi                  # 8-byte Reload
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	movq	80(%rsp), %rcx                  # 8-byte Reload
.Ltmp49:                                # EH_LABEL
.LBB6_53:                               # %_ZSt4sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEvT_S7_.exit.4
	movq	%r13, 296(%rsp)                 # 8-byte Spill
	movsd	(%rcx), %xmm1                   # xmm1 = mem[0],zero
	movsd	56(%rcx), %xmm0                 # xmm0 = mem[0],zero
	movsd	112(%rcx), %xmm2                # xmm2 = mem[0],zero
	leaq	.L.str.24(%rip), %rdi
	leaq	.L.str.23(%rip), %rsi
	movb	$3, %al
	callq	printf@PLT
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rax
	shrq	$61, %rax
	movq	112(%rsp), %rbx                 # 8-byte Reload
	movq	104(%rsp), %r13                 # 8-byte Reload
	jne	.LBB6_113
# %bb.54:                               # %_ZNSt6vectorIfSaIfEE17_S_check_init_lenEmRKS0_.exit.i
	testq	%rcx, %rcx
	je	.LBB6_58
# %bb.55:
.Ltmp51:                                # EH_LABEL
	movq	%r14, 288(%rsp)                 # 8-byte Spill
	movq	%r13, %rdi
	callq	_Znwm@PLT
.Ltmp52:                                # EH_LABEL
# %bb.56:
	movq	56(%rsp), %r14                  # 8-byte Reload
	leaq	(%rax,%r14,4), %rcx
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	movl	$0, (%rax)
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	addq	$4, %rdi
	leaq	-4(%r13), %rbx
	xorl	%esi, %esi
	movq	%rbx, %rdx
	callq	memset@PLT
.Ltmp53:                                # EH_LABEL
	movq	%r13, %rdi
	callq	_Znwm@PLT
.Ltmp54:                                # EH_LABEL
# %bb.57:                               # %.noexc198
	leaq	(%rax,%r14,4), %rcx
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	movl	$0, (%rax)
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	addq	$4, %rdi
	xorl	%esi, %esi
	movq	%rbx, %rdx
	callq	memset@PLT
	movq	288(%rsp), %r14                 # 8-byte Reload
	movq	112(%rsp), %rbx                 # 8-byte Reload
	jmp	.LBB6_59
.LBB6_58:
	movq	$0, 40(%rsp)                    # 8-byte Folded Spill
	movq	$0, 96(%rsp)                    # 8-byte Folded Spill
	movq	$0, 88(%rsp)                    # 8-byte Folded Spill
	movq	$0, 32(%rsp)                    # 8-byte Folded Spill
.LBB6_59:                               # %_ZNSt6vectorIfSaIfEEC2EmRKS0_.exit199
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
.Ltmp56:                                # EH_LABEL
	movl	%ebp, (%rsp)
	xorl	%edi, %edi
	movl	%r12d, %esi
	movl	%ebx, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
.Ltmp57:                                # EH_LABEL
# %bb.60:                               # %_ZZ4mainENKUliE_clEi.exit201
	movq	16(%rsp), %rsi
.Ltmp59:                                # EH_LABEL
	movq	40(%rsp), %rdi                  # 8-byte Reload
	movq	%r13, %rdx
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp60:                                # EH_LABEL
# %bb.61:
	testl	%eax, %eax
	jne	.LBB6_115
# %bb.62:                               # %.preheader318
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
.Ltmp64:                                # EH_LABEL
	movl	%ebp, (%rsp)
	movl	$1, %edi
	movl	%r12d, %esi
	movl	%ebx, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
.Ltmp65:                                # EH_LABEL
# %bb.63:                               # %_ZZ4mainENKUliE_clEi.exit203
	movq	16(%rsp), %rsi
.Ltmp66:                                # EH_LABEL
	movq	32(%rsp), %rdi                  # 8-byte Reload
	movq	104(%rsp), %rdx                 # 8-byte Reload
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp67:                                # EH_LABEL
# %bb.64:
	testl	%eax, %eax
	jne	.LBB6_101
# %bb.65:                               # %.preheader
	cmpq	$0, 56(%rsp)                    # 8-byte Folded Reload
	movq	40(%rsp), %r8                   # 8-byte Reload
	movq	32(%rsp), %r9                   # 8-byte Reload
	je	.LBB6_68
# %bb.66:                               # %.lr.ph.preheader
	xorpd	%xmm1, %xmm1
	xorl	%r13d, %r13d
	movaps	.LCPI6_11(%rip), %xmm2          # xmm2 = [NaN,NaN,NaN,NaN]
	movapd	.LCPI6_12(%rip), %xmm3          # xmm3 = [NaN,NaN]
	movsd	.LCPI6_13(%rip), %xmm4          # xmm4 = [1.0000000000000001E-30,0.0E+0]
	xorl	%eax, %eax
	xorl	%ebp, %ebp
	xorpd	%xmm0, %xmm0
	movq	56(%rsp), %rdx                  # 8-byte Reload
	.p2align	4
.LBB6_67:                               # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
	movapd	%xmm0, %xmm6
	movapd	%xmm1, %xmm5
	movl	(%r8,%rax,4), %edi
	movl	(%r9,%rax,4), %ecx
	xorl	%esi, %esi
	cmpl	%ecx, %edi
	setne	%sil
	addq	%rsi, %rbp
	movd	%edi, %xmm7
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm7, %xmm1
	andps	%xmm2, %xmm7
	addl	%edi, %edi
	cmpl	$-16777216, %edi                # imm = 0xFF000000
	setae	%sil
	movd	%ecx, %xmm0
	addl	%ecx, %ecx
	cmpl	$-16777216, %ecx                # imm = 0xFF000000
	setae	%cl
	orb	%sil, %cl
	movzbl	%cl, %ecx
	cvtss2sd	%xmm0, %xmm0
	addq	%rcx, %r13
	subsd	%xmm0, %xmm1
	andpd	%xmm3, %xmm1
	movapd	%xmm1, %xmm0
	maxsd	%xmm6, %xmm0
	xorps	%xmm6, %xmm6
	cvtss2sd	%xmm7, %xmm6
	maxsd	%xmm4, %xmm6
	divsd	%xmm6, %xmm1
	maxsd	%xmm5, %xmm1
	incq	%rax
	cmpq	%rax, %rdx
	jne	.LBB6_67
	jmp	.LBB6_69
.LBB6_68:
	xorpd	%xmm0, %xmm0
	xorl	%ebp, %ebp
	xorpd	%xmm1, %xmm1
	xorl	%r13d, %r13d
	movq	56(%rsp), %rdx                  # 8-byte Reload
.LBB6_69:                               # %._crit_edge
	leaq	.L.str.27(%rip), %rdi
	leaq	.L.str.20(%rip), %rsi
	movq	%rbp, %rcx
	movq	%r13, %r8
	movb	$2, %al
	callq	printf@PLT
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
.Ltmp68:                                # EH_LABEL
	movq	144(%rsp), %rax                 # 8-byte Reload
	movl	%eax, (%rsp)
	movl	$2, %edi
	movl	%r12d, %esi
	movl	%ebx, %edx
	movl	%r15d, %ecx
	callq	_Z6launchiiiiPKjPfi
.Ltmp69:                                # EH_LABEL
# %bb.70:                               # %_ZZ4mainENKUliE_clEi.exit203.1
	movq	16(%rsp), %rsi
.Ltmp71:                                # EH_LABEL
	movq	32(%rsp), %rdi                  # 8-byte Reload
	movq	104(%rsp), %rdx                 # 8-byte Reload
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp72:                                # EH_LABEL
# %bb.71:
	testl	%eax, %eax
	jne	.LBB6_101
# %bb.72:                               # %.preheader.1
	movq	56(%rsp), %rdx                  # 8-byte Reload
	testq	%rdx, %rdx
	movq	40(%rsp), %r10                  # 8-byte Reload
	movq	32(%rsp), %r11                  # 8-byte Reload
	je	.LBB6_75
# %bb.73:                               # %.lr.ph.1.preheader
	xorpd	%xmm1, %xmm1
	xorl	%r8d, %r8d
	movaps	.LCPI6_11(%rip), %xmm2          # xmm2 = [NaN,NaN,NaN,NaN]
	movapd	.LCPI6_12(%rip), %xmm3          # xmm3 = [NaN,NaN]
	movsd	.LCPI6_13(%rip), %xmm4          # xmm4 = [1.0000000000000001E-30,0.0E+0]
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	xorpd	%xmm0, %xmm0
	.p2align	4
.LBB6_74:                               # %.lr.ph.1
                                        # =>This Inner Loop Header: Depth=1
	movapd	%xmm0, %xmm6
	movapd	%xmm1, %xmm5
	movl	(%r10,%rax,4), %esi
	movl	(%r11,%rax,4), %r9d
	xorl	%edi, %edi
	cmpl	%r9d, %esi
	setne	%dil
	addq	%rdi, %rcx
	movd	%esi, %xmm7
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm7, %xmm1
	andps	%xmm2, %xmm7
	addl	%esi, %esi
	cmpl	$-16777216, %esi                # imm = 0xFF000000
	setae	%sil
	movd	%r9d, %xmm0
	addl	%r9d, %r9d
	cmpl	$-16777216, %r9d                # imm = 0xFF000000
	setae	%dil
	orb	%sil, %dil
	movzbl	%dil, %esi
	cvtss2sd	%xmm0, %xmm0
	addq	%rsi, %r8
	subsd	%xmm0, %xmm1
	andpd	%xmm3, %xmm1
	movapd	%xmm1, %xmm0
	maxsd	%xmm6, %xmm0
	xorps	%xmm6, %xmm6
	cvtss2sd	%xmm7, %xmm6
	maxsd	%xmm4, %xmm6
	divsd	%xmm6, %xmm1
	maxsd	%xmm5, %xmm1
	incq	%rax
	cmpq	%rax, %rdx
	jne	.LBB6_74
	jmp	.LBB6_76
.LBB6_75:
	xorpd	%xmm0, %xmm0
	xorl	%ecx, %ecx
	xorpd	%xmm1, %xmm1
	xorl	%r8d, %r8d
.LBB6_76:                               # %._crit_edge.1
	leaq	.L.str.27(%rip), %rdi
	leaq	.L.str.21(%rip), %rsi
	movb	$2, %al
	callq	printf@PLT
	movq	24(%rsp), %rdi
.Ltmp77:                                # EH_LABEL
	callq	hipFree@PLT
.Ltmp78:                                # EH_LABEL
# %bb.77:
	testl	%eax, %eax
	jne	.LBB6_117
# %bb.78:
	movq	16(%rsp), %rdi
.Ltmp82:                                # EH_LABEL
	callq	hipFree@PLT
.Ltmp83:                                # EH_LABEL
# %bb.79:
	testl	%eax, %eax
	jne	.LBB6_119
# %bb.80:
	movq	32(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB6_82
# %bb.81:
	movq	88(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB6_82:                               # %_ZNSt6vectorIfSaIfEED2Ev.exit
	movq	40(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB6_84
# %bb.83:
	movq	96(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB6_84:                               # %_ZNSt6vectorIdSaIdEED2Ev.exit.4
	movq	272(%rsp), %rsi
	movq	80(%rsp), %rdi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	248(%rsp), %rsi
	movq	64(%rsp), %rdi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	224(%rsp), %rsi
	movq	48(%rsp), %rdi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	200(%rsp), %rsi
	movq	296(%rsp), %rdi                 # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	176(%rsp), %rsi
	subq	%r14, %rsi
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
	xorl	%eax, %eax
	orq	%r13, %rbp
	setne	%al
	addq	$2056, %rsp                     # imm = 0x808
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
.LBB6_85:
	.cfi_def_cfa_offset 2112
	movq	16(%rbx), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	__isoc23_strtol@PLT
	cmpl	$4, %ebp
	jb	.LBB6_89
# %bb.86:
	movq	%rax, %r13
	movq	24(%rbx), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	__isoc23_strtol@PLT
	movq	%rax, %r14
	cmpl	$4, %ebp
	je	.LBB6_88
# %bb.87:
	movq	32(%rbx), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	__isoc23_strtol@PLT
	movq	%rax, %r15
.LBB6_88:                               # %.thread299
	movq	%r14, %rbp
	movq	%r13, %r14
	jmp	.LBB6_4
.LBB6_89:
	movq	%rax, %r14
	jmp	.LBB6_3
.LBB6_90:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp18:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp19:                                # EH_LABEL
# %bb.91:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.16(%rip), %rdx
	jmp	.LBB6_98
.LBB6_92:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp24:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp25:                                # EH_LABEL
# %bb.93:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.17(%rip), %rdx
	jmp	.LBB6_98
.LBB6_94:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp3:                                 # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp4:                                 # EH_LABEL
# %bb.95:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.14(%rip), %rdx
	jmp	.LBB6_98
.LBB6_96:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp12:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp13:                                # EH_LABEL
# %bb.97:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.15(%rip), %rdx
.LBB6_98:
	movq	%rbx, %rdi
	movq	%rax, %rcx
	xorl	%eax, %eax
	callq	fprintf@PLT
	movl	$2, %edi
	callq	exit@PLT
.LBB6_99:
.Ltmp90:                                # EH_LABEL
	leaq	.L.str.30(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp91:                                # EH_LABEL
# %bb.100:                              # %.noexc
.LBB6_101:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp74:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp75:                                # EH_LABEL
# %bb.102:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.26(%rip), %rdx
	jmp	.LBB6_98
.LBB6_103:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.2(%rip), %rdx
	jmp	.LBB6_98
.LBB6_104:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.4(%rip), %rdx
	jmp	.LBB6_98
.LBB6_105:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	jmp	.LBB6_98
.LBB6_106:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.6(%rip), %rdx
	jmp	.LBB6_98
.LBB6_107:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.8(%rip), %rdx
	jmp	.LBB6_98
.LBB6_108:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.9(%rip), %rdx
	jmp	.LBB6_98
.LBB6_109:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.10(%rip), %rdx
	jmp	.LBB6_98
.LBB6_110:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.11(%rip), %rdx
	jmp	.LBB6_98
.LBB6_111:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.12(%rip), %rdx
	jmp	.LBB6_98
.LBB6_112:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.13(%rip), %rdx
	jmp	.LBB6_98
.LBB6_113:
.Ltmp87:                                # EH_LABEL
	leaq	.L.str.31(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp88:                                # EH_LABEL
# %bb.114:                              # %.noexc187
.LBB6_115:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp61:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp62:                                # EH_LABEL
# %bb.116:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.25(%rip), %rdx
	jmp	.LBB6_98
.LBB6_117:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp79:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp80:                                # EH_LABEL
# %bb.118:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.28(%rip), %rdx
	jmp	.LBB6_98
.LBB6_119:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp84:                                # EH_LABEL
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp85:                                # EH_LABEL
# %bb.120:
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.29(%rip), %rdx
	jmp	.LBB6_98
.LBB6_121:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit207.thread
.Ltmp55:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB6_139
.LBB6_122:                              # %.loopexit.split-lp
.Ltmp76:                                # EH_LABEL
	jmp	.LBB6_130
.LBB6_123:
.Ltmp58:                                # EH_LABEL
	jmp	.LBB6_130
.LBB6_124:
.Ltmp86:                                # EH_LABEL
	jmp	.LBB6_130
.LBB6_125:
.Ltmp81:                                # EH_LABEL
	jmp	.LBB6_130
.LBB6_126:
.Ltmp63:                                # EH_LABEL
	jmp	.LBB6_130
.LBB6_127:
.Ltmp89:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_128:                              # %.loopexit
.Ltmp73:                                # EH_LABEL
	jmp	.LBB6_130
.LBB6_129:
.Ltmp70:                                # EH_LABEL
.LBB6_130:
	movq	%rax, %rbx
	cmpq	$0, 32(%rsp)                    # 8-byte Folded Reload
	jne	.LBB6_138
# %bb.131:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit207
	cmpq	$0, 40(%rsp)                    # 8-byte Folded Reload
	jne	.LBB6_139
.LBB6_132:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit209
	movq	256(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB6_153
.LBB6_133:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit212
	movq	232(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB6_154
.LBB6_134:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit212.1
	movq	208(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB6_155
.LBB6_135:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit212.2
	movq	184(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB6_156
.LBB6_136:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit212.3
	movq	160(%rsp), %rdi
	testq	%rdi, %rdi
	jne	.LBB6_157
.LBB6_137:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit212.4
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB6_138:
	movq	32(%rsp), %rdi                  # 8-byte Reload
	movq	88(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	cmpq	$0, 40(%rsp)                    # 8-byte Folded Reload
	je	.LBB6_132
.LBB6_139:
	movq	40(%rsp), %rdi                  # 8-byte Reload
	movq	96(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	256(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB6_133
	jmp	.LBB6_153
.LBB6_140:
.Ltmp50:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_141:                              # %.loopexit.split-lp342
.Ltmp92:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_142:                              # %.loopexit341
.Ltmp29:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_143:                              # %.loopexit.split-lp327
.Ltmp14:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_144:                              # %.loopexit.split-lp322
.Ltmp5:                                 # EH_LABEL
	jmp	.LBB6_152
.LBB6_145:                              # %.loopexit.split-lp337
.Ltmp26:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_146:                              # %.loopexit.split-lp332
.Ltmp20:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_147:                              # %.loopexit326
.Ltmp11:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_148:                              # %.loopexit321
.Ltmp2:                                 # EH_LABEL
	jmp	.LBB6_152
.LBB6_149:                              # %.loopexit336
.Ltmp23:                                # EH_LABEL
	jmp	.LBB6_152
.LBB6_150:
.Ltmp8:                                 # EH_LABEL
	jmp	.LBB6_152
.LBB6_151:                              # %.loopexit331
.Ltmp17:                                # EH_LABEL
.LBB6_152:                              # %_ZNSt6vectorIfSaIfEED2Ev.exit209
	movq	%rax, %rbx
	movq	256(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB6_133
.LBB6_153:
	movq	272(%rsp), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	232(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB6_134
.LBB6_154:
	movq	248(%rsp), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	208(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB6_135
.LBB6_155:
	movq	224(%rsp), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	184(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB6_136
.LBB6_156:
	movq	200(%rsp), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	160(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB6_137
.LBB6_157:
	movq	176(%rsp), %rsi
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end6:
	.size	main, .Lfunc_end6-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table6:
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
	.uleb128 .Ltmp9-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp10-.Ltmp9                 #   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp11-.Lfunc_begin0          #     jumps to .Ltmp11
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin0          # >> Call Site 5 <<
	.uleb128 .Ltmp16-.Ltmp15                #   Call between .Ltmp15 and .Ltmp16
	.uleb128 .Ltmp17-.Lfunc_begin0          #     jumps to .Ltmp17
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp21-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp22-.Ltmp21                #   Call between .Ltmp21 and .Ltmp22
	.uleb128 .Ltmp23-.Lfunc_begin0          #     jumps to .Ltmp23
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp27-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp28-.Ltmp27                #   Call between .Ltmp27 and .Ltmp28
	.uleb128 .Ltmp29-.Lfunc_begin0          #     jumps to .Ltmp29
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp28-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp30-.Ltmp28                #   Call between .Ltmp28 and .Ltmp30
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp30-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp49-.Ltmp30                #   Call between .Ltmp30 and .Ltmp49
	.uleb128 .Ltmp50-.Lfunc_begin0          #     jumps to .Ltmp50
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp51-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp52-.Ltmp51                #   Call between .Ltmp51 and .Ltmp52
	.uleb128 .Ltmp89-.Lfunc_begin0          #     jumps to .Ltmp89
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp52-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp53-.Ltmp52                #   Call between .Ltmp52 and .Ltmp53
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp53-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp54-.Ltmp53                #   Call between .Ltmp53 and .Ltmp54
	.uleb128 .Ltmp55-.Lfunc_begin0          #     jumps to .Ltmp55
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp54-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp56-.Ltmp54                #   Call between .Ltmp54 and .Ltmp56
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp56-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp57-.Ltmp56                #   Call between .Ltmp56 and .Ltmp57
	.uleb128 .Ltmp58-.Lfunc_begin0          #     jumps to .Ltmp58
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp59-.Lfunc_begin0          # >> Call Site 15 <<
	.uleb128 .Ltmp60-.Ltmp59                #   Call between .Ltmp59 and .Ltmp60
	.uleb128 .Ltmp63-.Lfunc_begin0          #     jumps to .Ltmp63
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp64-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Ltmp65-.Ltmp64                #   Call between .Ltmp64 and .Ltmp65
	.uleb128 .Ltmp70-.Lfunc_begin0          #     jumps to .Ltmp70
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp66-.Lfunc_begin0          # >> Call Site 17 <<
	.uleb128 .Ltmp67-.Ltmp66                #   Call between .Ltmp66 and .Ltmp67
	.uleb128 .Ltmp73-.Lfunc_begin0          #     jumps to .Ltmp73
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp68-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp69-.Ltmp68                #   Call between .Ltmp68 and .Ltmp69
	.uleb128 .Ltmp70-.Lfunc_begin0          #     jumps to .Ltmp70
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp71-.Lfunc_begin0          # >> Call Site 19 <<
	.uleb128 .Ltmp72-.Ltmp71                #   Call between .Ltmp71 and .Ltmp72
	.uleb128 .Ltmp73-.Lfunc_begin0          #     jumps to .Ltmp73
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp77-.Lfunc_begin0          # >> Call Site 20 <<
	.uleb128 .Ltmp78-.Ltmp77                #   Call between .Ltmp77 and .Ltmp78
	.uleb128 .Ltmp81-.Lfunc_begin0          #     jumps to .Ltmp81
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp82-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Ltmp83-.Ltmp82                #   Call between .Ltmp82 and .Ltmp83
	.uleb128 .Ltmp86-.Lfunc_begin0          #     jumps to .Ltmp86
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp18-.Lfunc_begin0          # >> Call Site 22 <<
	.uleb128 .Ltmp19-.Ltmp18                #   Call between .Ltmp18 and .Ltmp19
	.uleb128 .Ltmp20-.Lfunc_begin0          #     jumps to .Ltmp20
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp24-.Lfunc_begin0          # >> Call Site 23 <<
	.uleb128 .Ltmp25-.Ltmp24                #   Call between .Ltmp24 and .Ltmp25
	.uleb128 .Ltmp26-.Lfunc_begin0          #     jumps to .Ltmp26
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 24 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          # >> Call Site 25 <<
	.uleb128 .Ltmp13-.Ltmp12                #   Call between .Ltmp12 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp90-.Lfunc_begin0          # >> Call Site 26 <<
	.uleb128 .Ltmp91-.Ltmp90                #   Call between .Ltmp90 and .Ltmp91
	.uleb128 .Ltmp92-.Lfunc_begin0          #     jumps to .Ltmp92
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp74-.Lfunc_begin0          # >> Call Site 27 <<
	.uleb128 .Ltmp75-.Ltmp74                #   Call between .Ltmp74 and .Ltmp75
	.uleb128 .Ltmp76-.Lfunc_begin0          #     jumps to .Ltmp76
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp75-.Lfunc_begin0          # >> Call Site 28 <<
	.uleb128 .Ltmp87-.Ltmp75                #   Call between .Ltmp75 and .Ltmp87
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp87-.Lfunc_begin0          # >> Call Site 29 <<
	.uleb128 .Ltmp88-.Ltmp87                #   Call between .Ltmp87 and .Ltmp88
	.uleb128 .Ltmp89-.Lfunc_begin0          #     jumps to .Ltmp89
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp61-.Lfunc_begin0          # >> Call Site 30 <<
	.uleb128 .Ltmp62-.Ltmp61                #   Call between .Ltmp61 and .Ltmp62
	.uleb128 .Ltmp63-.Lfunc_begin0          #     jumps to .Ltmp63
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp79-.Lfunc_begin0          # >> Call Site 31 <<
	.uleb128 .Ltmp80-.Ltmp79                #   Call between .Ltmp79 and .Ltmp80
	.uleb128 .Ltmp81-.Lfunc_begin0          #     jumps to .Ltmp81
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp84-.Lfunc_begin0          # >> Call Site 32 <<
	.uleb128 .Ltmp85-.Ltmp84                #   Call between .Ltmp84 and .Ltmp85
	.uleb128 .Ltmp86-.Lfunc_begin0          #     jumps to .Ltmp86
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp85-.Lfunc_begin0          # >> Call Site 33 <<
	.uleb128 .Lfunc_end6-.Ltmp85            #   Call between .Ltmp85 and .Lfunc_end6
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.prefalign	4, .Lfunc_end7, nop
	.type	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,@function
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_: # @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
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
	sarq	$3, %rbp
	cmpq	$17, %rbp
	jl	.LBB7_40
# %bb.1:                                # %.lr.ph
	movq	%rdx, %r14
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.LBB7_8
# %bb.2:                                # %.lr.ph43.preheader
	movq	$-8, %r13
	subq	%rbx, %r13
	.p2align	4
.LBB7_3:                                # %.lr.ph43
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB7_33 Depth 2
                                        #       Child Loop BB7_34 Depth 3
                                        #       Child Loop BB7_36 Depth 3
	shrq	%rbp
	movsd	8(%rbx), %xmm1                  # xmm1 = mem[0],zero
	movsd	(%rbx,%rbp,8), %xmm2            # xmm2 = mem[0],zero
	ucomisd	%xmm1, %xmm2
	movsd	-8(%rsi), %xmm0                 # xmm0 = mem[0],zero
	jbe	.LBB7_27
# %bb.4:                                #   in Loop: Header=BB7_3 Depth=1
	ucomisd	%xmm2, %xmm0
	jbe	.LBB7_24
# %bb.5:                                #   in Loop: Header=BB7_3 Depth=1
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	movsd	%xmm2, (%rbx)
	movsd	%xmm0, (%rbx,%rbp,8)
	jmp	.LBB7_32
	.p2align	4
.LBB7_27:                               #   in Loop: Header=BB7_3 Depth=1
	ucomisd	%xmm1, %xmm0
	jbe	.LBB7_29
# %bb.28:                               #   in Loop: Header=BB7_3 Depth=1
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	movsd	%xmm1, (%rbx)
	movsd	%xmm0, 8(%rbx)
	jmp	.LBB7_32
	.p2align	4
.LBB7_24:                               #   in Loop: Header=BB7_3 Depth=1
	ucomisd	%xmm1, %xmm0
	movsd	(%rbx), %xmm2                   # xmm2 = mem[0],zero
	jbe	.LBB7_26
# %bb.25:                               #   in Loop: Header=BB7_3 Depth=1
	movsd	%xmm0, (%rbx)
	movsd	%xmm2, -8(%rsi)
	jmp	.LBB7_32
	.p2align	4
.LBB7_29:                               #   in Loop: Header=BB7_3 Depth=1
	ucomisd	%xmm2, %xmm0
	movsd	(%rbx), %xmm1                   # xmm1 = mem[0],zero
	jbe	.LBB7_31
# %bb.30:                               #   in Loop: Header=BB7_3 Depth=1
	movsd	%xmm0, (%rbx)
	movsd	%xmm1, -8(%rsi)
	jmp	.LBB7_32
.LBB7_26:                               #   in Loop: Header=BB7_3 Depth=1
	movsd	%xmm1, (%rbx)
	movsd	%xmm2, 8(%rbx)
	jmp	.LBB7_32
.LBB7_31:                               #   in Loop: Header=BB7_3 Depth=1
	movsd	%xmm2, (%rbx)
	movsd	%xmm1, (%rbx,%rbp,8)
	.p2align	4
.LBB7_32:                               # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i.preheader
                                        #   in Loop: Header=BB7_3 Depth=1
	decq	%r14
	leaq	8(%rbx), %r12
	movq	%rsi, %rax
	.p2align	4
.LBB7_33:                               # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i
                                        #   Parent Loop BB7_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB7_34 Depth 3
                                        #       Child Loop BB7_36 Depth 3
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	leaq	(%r12,%r13), %rbp
	.p2align	4
.LBB7_34:                               #   Parent Loop BB7_3 Depth=1
                                        #     Parent Loop BB7_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsd	(%r12), %xmm1                   # xmm1 = mem[0],zero
	addq	$8, %r12
	addq	$8, %rbp
	ucomisd	%xmm1, %xmm0
	ja	.LBB7_34
# %bb.35:                               # %.preheader.i.i.preheader
                                        #   in Loop: Header=BB7_33 Depth=2
	leaq	-8(%r12), %r15
	.p2align	4
.LBB7_36:                               # %.preheader.i.i
                                        #   Parent Loop BB7_3 Depth=1
                                        #     Parent Loop BB7_33 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsd	-8(%rax), %xmm2                 # xmm2 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm0, %xmm2
	ja	.LBB7_36
# %bb.37:                               #   in Loop: Header=BB7_33 Depth=2
	cmpq	%rax, %r15
	jae	.LBB7_39
# %bb.38:                               #   in Loop: Header=BB7_33 Depth=2
	movsd	%xmm2, (%r15)
	movsd	%xmm1, (%rax)
	jmp	.LBB7_33
	.p2align	4
.LBB7_39:                               # %_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_.exit
                                        #   in Loop: Header=BB7_3 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	sarq	$3, %rbp
	cmpq	$16, %rbp
	jle	.LBB7_40
# %bb.6:                                #   in Loop: Header=BB7_3 Depth=1
	movq	%r15, %rsi
	testq	%r14, %r14
	jne	.LBB7_3
# %bb.7:                                # %._crit_edge.loopexit
	addq	$-8, %r12
	movq	%r12, %rsi
.LBB7_8:                                # %._crit_edge
	leaq	7(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%rsi, %r14
	callq	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	jmp	.LBB7_9
	.p2align	4
.LBB7_22:                               #   in Loop: Header=BB7_9 Depth=1
	xorl	%ecx, %ecx
.LBB7_23:                               # %_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_.exit.i.i
                                        #   in Loop: Header=BB7_9 Depth=1
	movsd	%xmm0, (%rbx,%rcx,8)
	cmpq	$8, %rax
	jle	.LBB7_40
.LBB7_9:                                # %.lr.ph.i.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB7_12 Depth 2
                                        #     Child Loop BB7_20 Depth 2
	movsd	-8(%r14), %xmm0                 # xmm0 = mem[0],zero
	movsd	(%rbx), %xmm1                   # xmm1 = mem[0],zero
	movsd	%xmm1, -8(%r14)
	addq	$-8, %r14
	movq	%r14, %rax
	subq	%rbx, %rax
	movq	%rax, %rdx
	sarq	$3, %rdx
	cmpq	$3, %rdx
	jl	.LBB7_10
# %bb.11:                               # %.lr.ph.i.i.i.i.preheader
                                        #   in Loop: Header=BB7_9 Depth=1
	leaq	-1(%rdx), %rcx
	shrq	$63, %rcx
	leaq	(%rdx,%rcx), %rsi
	decq	%rsi
	sarq	%rsi
	xorl	%edi, %edi
	jmp	.LBB7_12
	.p2align	4
.LBB7_14:                               # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB7_12 Depth=2
	leaq	2(,%rdi,2), %rcx
.LBB7_15:                               # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB7_12 Depth=2
	movsd	(%rbx,%rcx,8), %xmm1            # xmm1 = mem[0],zero
	movsd	%xmm1, (%rbx,%rdi,8)
	movq	%rcx, %rdi
	cmpq	%rsi, %rcx
	jge	.LBB7_16
.LBB7_12:                               # %.lr.ph.i.i.i.i
                                        #   Parent Loop BB7_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rdi,%rdi), %rcx
	movsd	8(%rbx,%rcx,8), %xmm1           # xmm1 = mem[0],zero
	ucomisd	16(%rbx,%rcx,8), %xmm1
	jbe	.LBB7_14
# %bb.13:                               #   in Loop: Header=BB7_12 Depth=2
	leaq	1(,%rdi,2), %rcx
	jmp	.LBB7_15
	.p2align	4
.LBB7_10:                               #   in Loop: Header=BB7_9 Depth=1
	xorl	%ecx, %ecx
.LBB7_16:                               # %._crit_edge.i.i.i.i
                                        #   in Loop: Header=BB7_9 Depth=1
	testb	$8, %al
	jne	.LBB7_19
# %bb.17:                               #   in Loop: Header=BB7_9 Depth=1
	addq	$-2, %rdx
	sarq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB7_19
# %bb.18:                               # %.thread.i.i.i
                                        #   in Loop: Header=BB7_9 Depth=1
	leaq	(%rcx,%rcx), %rdx
	movsd	8(%rbx,%rdx,8), %xmm1           # xmm1 = mem[0],zero
	movsd	%xmm1, (%rbx,%rcx,8)
	leaq	1(,%rcx,2), %rcx
	jmp	.LBB7_20
	.p2align	4
.LBB7_19:                               #   in Loop: Header=BB7_9 Depth=1
	testq	%rcx, %rcx
	je	.LBB7_22
	.p2align	4
.LBB7_20:                               # %.lr.ph.i.i.i.i.i
                                        #   Parent Loop BB7_9 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rcx), %rdx
	shrq	%rdx
	movsd	(%rbx,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB7_23
# %bb.21:                               #   in Loop: Header=BB7_20 Depth=2
	movsd	%xmm1, (%rbx,%rcx,8)
	movq	%rdx, %rcx
	testq	%rdx, %rdx
	jne	.LBB7_20
	jmp	.LBB7_22
.LBB7_40:                               # %_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_.exit
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
.Lfunc_end7:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end7-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ # -- Begin function _ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.prefalign	4, .Lfunc_end8, nop
	.type	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,@function
_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_: # @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
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
	cmpq	$129, %rax
	jl	.LBB8_17
# %bb.1:                                # %.lr.ph.i
	leaq	8(%r14), %r15
	movl	$8, %r12d
	movq	%r15, %r13
	movq	%r14, %rbp
	jmp	.LBB8_2
.LBB8_17:
	cmpq	%rbx, %r14
	je	.LBB8_29
# %bb.18:
	leaq	8(%r14), %rax
	cmpq	%rbx, %rax
	je	.LBB8_29
# %bb.19:                               # %.lr.ph.i15.preheader
	movq	%r14, %r15
	jmp	.LBB8_20
	.p2align	4
.LBB8_28:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEES6_ET0_T_S8_S7_.exit.i18
                                        #   in Loop: Header=BB8_20 Depth=1
	movsd	%xmm1, (%rax)
	leaq	8(%r15), %rax
	cmpq	%rbx, %rax
	je	.LBB8_29
.LBB8_20:                               # %.lr.ph.i15
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB8_27 Depth 2
	movq	%r15, %rdi
	movq	%rax, %r15
	movsd	8(%rdi), %xmm1                  # xmm1 = mem[0],zero
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB8_25
# %bb.21:                               # %_ZSt7advanceIPdlEvRT_T0_.exit.i.i.i.i.i28
                                        #   in Loop: Header=BB8_20 Depth=1
	movsd	%xmm1, (%rsp)                   # 8-byte Spill
	movq	%r15, %rdx
	subq	%r14, %rdx
	subq	%rdx, %rdi
	movq	%rdx, %rax
	sarq	$3, %rax
	addq	$16, %rdi
	cmpq	$2, %rax
	jl	.LBB8_23
# %bb.22:                               #   in Loop: Header=BB8_20 Depth=1
	movq	%r14, %rsi
	callq	memmove@PLT
	movq	%r14, %rax
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
	jmp	.LBB8_28
	.p2align	4
.LBB8_25:                               #   in Loop: Header=BB8_20 Depth=1
	movsd	(%rdi), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	movq	%r15, %rax
	jbe	.LBB8_28
# %bb.26:                               # %.lr.ph.i.i22.preheader
                                        #   in Loop: Header=BB8_20 Depth=1
	movq	%r15, %rax
	.p2align	4
.LBB8_27:                               # %.lr.ph.i.i22
                                        #   Parent Loop BB8_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm0, (%rax)
	movsd	-16(%rax), %xmm0                # xmm0 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm1, %xmm0
	ja	.LBB8_27
	jmp	.LBB8_28
.LBB8_23:                               # %_ZSt7advanceIPdlEvRT_T0_.exit.thread.i.i.i.i.i29
                                        #   in Loop: Header=BB8_20 Depth=1
	movq	%r14, %rax
	cmpq	$8, %rdx
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
	jne	.LBB8_28
# %bb.24:                               #   in Loop: Header=BB8_20 Depth=1
	movsd	%xmm0, (%rdi)
	movq	%r14, %rax
	jmp	.LBB8_28
.LBB8_5:                                # %_ZSt7advanceIPdlEvRT_T0_.exit.thread.i.i.i.i.i
                                        #   in Loop: Header=BB8_2 Depth=1
	movsd	%xmm0, (%r15)
	.p2align	4
.LBB8_6:                                # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB8_2 Depth=1
	movq	%r14, %rax
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB8_10:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB8_2 Depth=1
	movsd	%xmm1, (%rax)
	addq	$8, %r12
	addq	$8, %r13
	cmpq	$128, %r12
	je	.LBB8_11
.LBB8_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB8_9 Depth 2
	movq	%rbp, %rax
	leaq	(%r14,%r12), %rbp
	movsd	(%r14,%r12), %xmm1              # xmm1 = mem[0],zero
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB8_7
# %bb.3:                                # %_ZSt7advanceIPdlEvRT_T0_.exit.i.i.i.i.i
                                        #   in Loop: Header=BB8_2 Depth=1
	movsd	%xmm1, (%rsp)                   # 8-byte Spill
	cmpq	$9, %r12
	jb	.LBB8_5
# %bb.4:                                #   in Loop: Header=BB8_2 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
	jmp	.LBB8_6
	.p2align	4
.LBB8_7:                                #   in Loop: Header=BB8_2 Depth=1
	movsd	(%rax), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	movq	%rbp, %rax
	jbe	.LBB8_10
# %bb.8:                                # %.lr.ph.i.i.preheader
                                        #   in Loop: Header=BB8_2 Depth=1
	movq	%r13, %rax
	.p2align	4
.LBB8_9:                                # %.lr.ph.i.i
                                        #   Parent Loop BB8_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm0, (%rax)
	movsd	-16(%rax), %xmm0                # xmm0 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm1, %xmm0
	ja	.LBB8_9
	jmp	.LBB8_10
.LBB8_11:                               # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
	subq	$-128, %r14
	jmp	.LBB8_12
	.p2align	4
.LBB8_16:                               # %_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops14_Val_less_iterEEvT_T0_.exit.i
                                        #   in Loop: Header=BB8_12 Depth=1
	movsd	%xmm0, (%rax)
	addq	$8, %r14
.LBB8_12:                               # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB8_15 Depth 2
	cmpq	%rbx, %r14
	je	.LBB8_29
# %bb.13:                               # %.lr.ph.i6
                                        #   in Loop: Header=BB8_12 Depth=1
	movsd	-8(%r14), %xmm1                 # xmm1 = mem[0],zero
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm0, %xmm1
	movq	%r14, %rax
	jbe	.LBB8_16
# %bb.14:                               # %.lr.ph.i.i8.preheader
                                        #   in Loop: Header=BB8_12 Depth=1
	movq	%r14, %rax
	.p2align	4
.LBB8_15:                               # %.lr.ph.i.i8
                                        #   Parent Loop BB8_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm1, (%rax)
	movsd	-16(%rax), %xmm1                # xmm1 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm0, %xmm1
	ja	.LBB8_15
	jmp	.LBB8_16
.LBB8_29:                               # %_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
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
.Lfunc_end8:
	.size	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_, .Lfunc_end8-_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,"axG",@progbits,_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,comdat
	.weak	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ # -- Begin function _ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.prefalign	4, .Lfunc_end9, nop
	.type	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,@function
_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_: # @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_startproc
# %bb.0:
	subq	%rdi, %rsi
	movq	%rsi, %rax
	sarq	$3, %rax
	cmpq	$2, %rax
	jge	.LBB9_2
.LBB9_1:                                # %.loopexit
	retq
.LBB9_2:
	leaq	-2(%rax), %rdx
	movq	%rdx, %rcx
	shrq	%rcx
	decq	%rax
	shrq	%rax
	testb	$8, %sil
	jne	.LBB9_20
# %bb.3:                                # %.split.preheader
	incq	%rdx
	movq	%rcx, %rsi
	jmp	.LBB9_6
	.p2align	4
.LBB9_4:                                #   in Loop: Header=BB9_6 Depth=1
	movq	%r8, %r9
.LBB9_5:                                # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEldNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit
                                        #   in Loop: Header=BB9_6 Depth=1
	movsd	%xmm0, (%rdi,%r9,8)
	subq	$1, %rsi
	jb	.LBB9_1
.LBB9_6:                                # %.split
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB9_10 Depth 2
                                        #     Child Loop BB9_15 Depth 2
	movsd	(%rdi,%rsi,8), %xmm0            # xmm0 = mem[0],zero
	movq	%rsi, %r8
	cmpq	%rax, %rsi
	jge	.LBB9_12
# %bb.7:                                # %.lr.ph.i.preheader
                                        #   in Loop: Header=BB9_6 Depth=1
	movq	%rsi, %r9
	jmp	.LBB9_10
	.p2align	4
.LBB9_8:                                # %.lr.ph.i
                                        #   in Loop: Header=BB9_10 Depth=2
	leaq	2(,%r9,2), %r8
.LBB9_9:                                # %.lr.ph.i
                                        #   in Loop: Header=BB9_10 Depth=2
	movsd	(%rdi,%r8,8), %xmm1             # xmm1 = mem[0],zero
	movsd	%xmm1, (%rdi,%r9,8)
	movq	%r8, %r9
	cmpq	%rax, %r8
	jge	.LBB9_12
.LBB9_10:                               # %.lr.ph.i
                                        #   Parent Loop BB9_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r9,%r9), %r8
	movsd	8(%rdi,%r8,8), %xmm1            # xmm1 = mem[0],zero
	ucomisd	16(%rdi,%r8,8), %xmm1
	jbe	.LBB9_8
# %bb.11:                               #   in Loop: Header=BB9_10 Depth=2
	leaq	1(,%r9,2), %r8
	jmp	.LBB9_9
	.p2align	4
.LBB9_12:                               # %._crit_edge.i
                                        #   in Loop: Header=BB9_6 Depth=1
	cmpq	%rcx, %r8
	jne	.LBB9_14
# %bb.13:                               #   in Loop: Header=BB9_6 Depth=1
	movsd	(%rdi,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	movsd	%xmm1, (%rdi,%rcx,8)
	movq	%rdx, %r8
.LBB9_14:                               #   in Loop: Header=BB9_6 Depth=1
	cmpq	%rsi, %r8
	jle	.LBB9_4
	.p2align	4
.LBB9_15:                               # %.lr.ph.i.i
                                        #   Parent Loop BB9_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%r8), %r9
	shrq	$63, %r9
	addq	%r8, %r9
	decq	%r9
	sarq	%r9
	movsd	(%rdi,%r9,8), %xmm1             # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB9_4
# %bb.16:                               #   in Loop: Header=BB9_15 Depth=2
	movsd	%xmm1, (%rdi,%r8,8)
	movq	%r9, %r8
	cmpq	%rsi, %r9
	jg	.LBB9_15
	jmp	.LBB9_5
	.p2align	4
.LBB9_18:                               #   in Loop: Header=BB9_20 Depth=1
	movq	%rdx, %rsi
.LBB9_19:                               # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEldNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit.us
                                        #   in Loop: Header=BB9_20 Depth=1
	movsd	%xmm0, (%rdi,%rsi,8)
	subq	$1, %rcx
	jb	.LBB9_1
.LBB9_20:                               # %.split.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB9_24 Depth 2
                                        #     Child Loop BB9_27 Depth 2
	movsd	(%rdi,%rcx,8), %xmm0            # xmm0 = mem[0],zero
	movq	%rcx, %rsi
	cmpq	%rax, %rcx
	jge	.LBB9_19
# %bb.21:                               # %.lr.ph.i.us.preheader
                                        #   in Loop: Header=BB9_20 Depth=1
	movq	%rcx, %rsi
	jmp	.LBB9_24
	.p2align	4
.LBB9_22:                               # %.lr.ph.i.us
                                        #   in Loop: Header=BB9_24 Depth=2
	leaq	2(,%rsi,2), %rdx
.LBB9_23:                               # %.lr.ph.i.us
                                        #   in Loop: Header=BB9_24 Depth=2
	movsd	(%rdi,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	movsd	%xmm1, (%rdi,%rsi,8)
	movq	%rdx, %rsi
	cmpq	%rax, %rdx
	jge	.LBB9_26
.LBB9_24:                               # %.lr.ph.i.us
                                        #   Parent Loop BB9_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%rsi,%rsi), %rdx
	movsd	8(%rdi,%rdx,8), %xmm1           # xmm1 = mem[0],zero
	ucomisd	16(%rdi,%rdx,8), %xmm1
	jbe	.LBB9_22
# %bb.25:                               #   in Loop: Header=BB9_24 Depth=2
	leaq	1(,%rsi,2), %rdx
	jmp	.LBB9_23
	.p2align	4
.LBB9_26:                               # %._crit_edge.i.us
                                        #   in Loop: Header=BB9_20 Depth=1
	cmpq	%rcx, %rdx
	jle	.LBB9_18
	.p2align	4
.LBB9_27:                               # %.lr.ph.i.i.us
                                        #   Parent Loop BB9_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rdx), %rsi
	shrq	$63, %rsi
	addq	%rdx, %rsi
	decq	%rsi
	sarq	%rsi
	movsd	(%rdi,%rsi,8), %xmm1            # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB9_18
# %bb.28:                               #   in Loop: Header=BB9_27 Depth=2
	movsd	%xmm1, (%rdi,%rdx,8)
	movq	%rsi, %rdx
	cmpq	%rcx, %rsi
	jg	.LBB9_27
	jmp	.LBB9_19
.Lfunc_end9:
	.size	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_, .Lfunc_end9-_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end10, nop    # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_2edb253f577969c9(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB10_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_2edb253f577969c9(%rip)
.LBB10_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z12floor_kernelILi0EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z12floor_kernelILi1EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z12floor_kernelILi2EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z12floor_kernelILi3EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z12floor_kernelILi4EEvPKjPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_5(%rip), %rcx
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
.Lfunc_end10:
	.size	__hip_module_ctor, .Lfunc_end10-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end11, nop    # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_2edb253f577969c9(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB11_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_2edb253f577969c9(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB11_2:
	retq
.Lfunc_end11:
	.size	__hip_module_dtor, .Lfunc_end11-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z12floor_kernelILi0EEvPKjPfi,@object # @_Z12floor_kernelILi0EEvPKjPfi
	.section	.data.rel.ro._Z12floor_kernelILi0EEvPKjPfi,"awG",@progbits,_Z12floor_kernelILi0EEvPKjPfi,comdat
	.weak	_Z12floor_kernelILi0EEvPKjPfi
	.p2align	3, 0x0
_Z12floor_kernelILi0EEvPKjPfi:
	.quad	_Z27__device_stub__floor_kernelILi0EEvPKjPfi
	.size	_Z12floor_kernelILi0EEvPKjPfi, 8

	.type	_Z12floor_kernelILi1EEvPKjPfi,@object # @_Z12floor_kernelILi1EEvPKjPfi
	.section	.data.rel.ro._Z12floor_kernelILi1EEvPKjPfi,"awG",@progbits,_Z12floor_kernelILi1EEvPKjPfi,comdat
	.weak	_Z12floor_kernelILi1EEvPKjPfi
	.p2align	3, 0x0
_Z12floor_kernelILi1EEvPKjPfi:
	.quad	_Z27__device_stub__floor_kernelILi1EEvPKjPfi
	.size	_Z12floor_kernelILi1EEvPKjPfi, 8

	.type	_Z12floor_kernelILi2EEvPKjPfi,@object # @_Z12floor_kernelILi2EEvPKjPfi
	.section	.data.rel.ro._Z12floor_kernelILi2EEvPKjPfi,"awG",@progbits,_Z12floor_kernelILi2EEvPKjPfi,comdat
	.weak	_Z12floor_kernelILi2EEvPKjPfi
	.p2align	3, 0x0
_Z12floor_kernelILi2EEvPKjPfi:
	.quad	_Z27__device_stub__floor_kernelILi2EEvPKjPfi
	.size	_Z12floor_kernelILi2EEvPKjPfi, 8

	.type	_Z12floor_kernelILi3EEvPKjPfi,@object # @_Z12floor_kernelILi3EEvPKjPfi
	.section	.data.rel.ro._Z12floor_kernelILi3EEvPKjPfi,"awG",@progbits,_Z12floor_kernelILi3EEvPKjPfi,comdat
	.weak	_Z12floor_kernelILi3EEvPKjPfi
	.p2align	3, 0x0
_Z12floor_kernelILi3EEvPKjPfi:
	.quad	_Z27__device_stub__floor_kernelILi3EEvPKjPfi
	.size	_Z12floor_kernelILi3EEvPKjPfi, 8

	.type	_Z12floor_kernelILi4EEvPKjPfi,@object # @_Z12floor_kernelILi4EEvPKjPfi
	.section	.data.rel.ro._Z12floor_kernelILi4EEvPKjPfi,"awG",@progbits,_Z12floor_kernelILi4EEvPKjPfi,comdat
	.weak	_Z12floor_kernelILi4EEvPKjPfi
	.p2align	3, 0x0
_Z12floor_kernelILi4EEvPKjPfi:
	.quad	_Z27__device_stub__floor_kernelILi4EEvPKjPfi
	.size	_Z12floor_kernelILi4EEvPKjPfi, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%s: %s\n"
	.size	.L.str, 8

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"hipGetLastError()"
	.size	.L.str.1, 18

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"hipGetDeviceProperties(&prop,0)"
	.size	.L.str.2, 32

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"device=%s CUs=%d blocks=%d threads=%d KT16iters=%d LDS=%d\n"
	.size	.L.str.3, 59

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"hipOccupancyMaxActiveBlocksPerMultiprocessor(&active0,floor_kernel<0>,threads,lds)"
	.size	.L.str.4, 83

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"hipOccupancyMaxActiveBlocksPerMultiprocessor(&active1,floor_kernel<1>,threads,lds)"
	.size	.L.str.5, 83

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"hipOccupancyMaxActiveBlocksPerMultiprocessor(&active2,floor_kernel<2>,threads,lds)"
	.size	.L.str.6, 83

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"occupancy_API_resident_WG_per_reported_MP exact=%d partner=%d rcp=%d\n"
	.size	.L.str.7, 70

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"hipMalloc(&seed,sizeof(h))"
	.size	.L.str.8, 27

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"hipMemcpy(seed,h,sizeof(h),hipMemcpyHostToDevice)"
	.size	.L.str.9, 50

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"hipMalloc(&out,n*4)"
	.size	.L.str.10, 20

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"hipDeviceSynchronize()"
	.size	.L.str.11, 23

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"hipEventCreate(&a)"
	.size	.L.str.12, 19

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"hipEventCreate(&b)"
	.size	.L.str.13, 19

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"hipEventRecord(a)"
	.size	.L.str.14, 18

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"hipEventRecord(b)"
	.size	.L.str.15, 18

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"hipEventSynchronize(b)"
	.size	.L.str.16, 23

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"hipEventElapsedTime(&ms,a,b)"
	.size	.L.str.17, 29

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"sample,%d,%d,%.6f\n"
	.size	.L.str.18, 19

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"exact_shfl"
	.size	.L.str.19, 11

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"exact_permlane"
	.size	.L.str.20, 15

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"approx_rcp"
	.size	.L.str.21, 11

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"matrix"
	.size	.L.str.22, 7

	.type	.L.str.23,@object               # @.str.23
.L.str.23:
	.asciz	"softmax"
	.size	.L.str.23, 8

	.type	.L.str.24,@object               # @.str.24
.L.str.24:
	.asciz	"median,%s,%.6f,min=%.6f,max=%.6f\n"
	.size	.L.str.24, 34

	.type	.L.str.25,@object               # @.str.25
.L.str.25:
	.asciz	"hipMemcpy(base.data(),out,n*4,hipMemcpyDeviceToHost)"
	.size	.L.str.25, 53

	.type	.L.str.26,@object               # @.str.26
.L.str.26:
	.asciz	"hipMemcpy(x.data(),out,n*4,hipMemcpyDeviceToHost)"
	.size	.L.str.26, 50

	.type	.L.str.27,@object               # @.str.27
.L.str.27:
	.asciz	"compare,%s,components=%zu,bitdiff=%zu,nonfinite=%zu,maxabs=%.9g,maxrel=%.9g\n"
	.size	.L.str.27, 77

	.type	.L.str.28,@object               # @.str.28
.L.str.28:
	.asciz	"hipFree(seed)"
	.size	.L.str.28, 14

	.type	.L.str.29,@object               # @.str.29
.L.str.29:
	.asciz	"hipFree(out)"
	.size	.L.str.29, 13

	.type	.L.str.30,@object               # @.str.30
.L.str.30:
	.asciz	"vector::_M_realloc_append"
	.size	.L.str.30, 26

	.type	.L.str.31,@object               # @.str.31
.L.str.31:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.31, 49

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z12floor_kernelILi0EEvPKjPfi"
	.size	.L__unnamed_1, 30

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z12floor_kernelILi1EEvPKjPfi"
	.size	.L__unnamed_2, 30

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z12floor_kernelILi2EEvPKjPfi"
	.size	.L__unnamed_3, 30

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z12floor_kernelILi3EEvPKjPfi"
	.size	.L__unnamed_4, 30

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_Z12floor_kernelILi4EEvPKjPfi"
	.size	.L__unnamed_5, 30

	.type	.L__unnamed_6,@object           # @5
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_6:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\000\210s\000\000\000\000\000\000 \000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx1201\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\bo\000\000\000\000\000\000N\000\000\000@\0008\000\t\000@\000\022\000\020\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000d!\000\000\000\000\000\000d!\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\"\000\000\000\000\000\000\0002\000\000\000\000\000\000\0002\000\000\000\000\000\000\274G\000\000\000\000\000\000\274G\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\300i\000\000\000\000\000\000\300\211\000\000\000\000\000\000\300\211\000\000\000\000\000\000p\000\000\000\000\000\000\000@\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\0000j\000\000\000\000\000\0000\232\000\000\000\000\000\0000\232\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\300i\000\000\000\000\000\000\300\211\000\000\000\000\000\000\300\211\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\300i\000\000\000\000\000\000\300\211\000\000\000\000\000\000\300\211\000\000\000\000\000\000p\000\000\000\000\000\000\000@\006\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\340\031\000\000\000\000\000\000\340\031\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000\311\031\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\225\336\000\023\245.args\334\000\021\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\203\247.offset\020\245.size\004\253.value_kind\250by_value\203\247.offset\030\245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset\034\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset \245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset$\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset&\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset(\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset*\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset,\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset.\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offset@\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetH\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offsetX\245.size\002\253.value_kind\260hidden_grid_dims\203\247.offset\314\220\245.size\004\253.value_kind\267hidden_dynamic_lds_size\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001\030\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\002\000\245.name\275_Z12floor_kernelILi0EEvPKjPfi\273.private_segment_fixed_size\000\253.sgpr_count\022\261.sgpr_spill_count\000\247.symbol\331 _Z12floor_kernelILi0EEvPKjPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\273\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\334\000\021\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\203\247.offset\020\245.size\004\253.value_kind\250by_value\203\247.offset\030\245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset\034\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset \245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset$\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset&\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset(\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset*\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset,\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset.\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offset@\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetH\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offsetX\245.size\002\253.value_kind\260hidden_grid_dims\203\247.offset\314\220\245.size\004\253.value_kind\267hidden_dynamic_lds_size\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001\030\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\002\000\245.name\275_Z12floor_kernelILi1EEvPKjPfi\273.private_segment_fixed_size\000\253.sgpr_count\022\261.sgpr_spill_count\000\247.symbol\331 _Z12floor_kernelILi1EEvPKjPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\271\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\334\000\021\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\203\247.offset\020\245.size\004\253.value_kind\250by_value\203\247.offset\030\245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset\034\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset \245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset$\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset&\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset(\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset*\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset,\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset.\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offset@\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetH\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offsetX\245.size\002\253.value_kind\260hidden_grid_dims\203\247.offset\314\220\245.size\004\253.value_kind\267hidden_dynamic_lds_size\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001\030\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\002\000\245.name\275_Z12floor_kernelILi2EEvPKjPfi\273.private_segment_fixed_size\000\253.sgpr_count\022\261.sgpr_spill_count\000\247.symbol\331 _Z12floor_kernelILi2EEvPKjPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\271\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\334\000\021\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\203\247.offset\020\245.size\004\253.value_kind\250by_value\203\247.offset\030\245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset\034\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset \245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset$\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset&\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset(\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset*\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset,\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset.\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offset@\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetH\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offsetX\245.size\002\253.value_kind\260hidden_grid_dims\203\247.offset\314\220\245.size\004\253.value_kind\267hidden_dynamic_lds_size\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001\030\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\002\000\245.name\275_Z12floor_kernelILi3EEvPKjPfi\273.private_segment_fixed_size\000\253.sgpr_count\022\261.sgpr_spill_count\000\247.symbol\331 _Z12floor_kernelILi3EEvPKjPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\230\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\334\000\021\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\203\247.offset\020\245.size\004\253.value_kind\250by_value\203\247.offset\030\245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset\034\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset \245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset$\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset&\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset(\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset*\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset,\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset.\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offset@\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetH\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offsetX\245.size\002\253.value_kind\260hidden_grid_dims\203\247.offset\314\220\245.size\004\253.value_kind\267hidden_dynamic_lds_size\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001\030\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\002\000\245.name\275_Z12floor_kernelILi4EEvPKjPfi\273.private_segment_fixed_size\000\253.sgpr_count\016\261.sgpr_spill_count\000\247.symbol\331 _Z12floor_kernelILi4EEvPKjPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\256\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\272amdgcn-amd-amdhsa--gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\037\000\000\000\021\003\006\000\200\037\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\022\003\b\000\000D\000\000\000\000\000\000\324\021\000\000\000\000\000\000\235\000\000\000\021\003\006\000\000 \000\000\000\000\000\000@\000\000\000\000\000\000\000\276\000\000\000\022\003\b\000\000e\000\000\000\000\000\000d\006\000\000\000\000\000\000\033\001\000\000\021\003\006\000\200 \000\000\000\000\000\000@\000\000\000\000\000\000\000\001\000\000\000\022\003\b\000\0002\000\000\000\000\000\000\330\021\000\000\000\000\000\000^\000\000\000\021\003\006\000\300\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\177\000\000\000\022\003\b\000\000V\000\000\000\000\000\000\270\016\000\000\000\000\000\000\334\000\000\000\021\003\006\000@ \000\000\000\000\000\000@\000\000\000\000\000\000\000\375\000\000\000\022\003\b\000\000l\000\000\000\000\000\000\274\r\000\000\000\000\000\000<\001\000\000\021\000\013\0000\232\000\000\000\000\000\000\001\000\000\000\000\000\000\000\002\000\000\000\001\000\000\000\004\000\000\000\032\000\000\000\000\000 \000\000\000@`\001\b\250\000\001\000\000\201\002\000\000\002\000\000\000\000\000\020T\000\020\000\b\000\001\000\000\000\006\000\000\000\322\221\354\2216Mt\373\324\244\2520xo\211\003\327\267h\317\024\274i\367R\233K\341V\336~\377T\256\t\200\230\000\224\007K[ ]\f\000\000\000\f\000\000\000\t\000\000\000\013\000\000\000\000\000\000\000\000\000\000\000\003\000\000\000\n\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\005\000\000\000\000\000\000\000\001\000\000\000\002\000\000\000\b\000\000\000\000_Z12floor_kernelILi0EEvPKjPfi\000_Z12floor_kernelILi0EEvPKjPfi.kd\000_Z12floor_kernelILi1EEvPKjPfi\000_Z12floor_kernelILi1EEvPKjPfi.kd\000_Z12floor_kernelILi2EEvPKjPfi\000_Z12floor_kernelILi2EEvPKjPfi.kd\000_Z12floor_kernelILi3EEvPKjPfi\000_Z12floor_kernelILi3EEvPKjPfi.kd\000_Z12floor_kernelILi4EEvPKjPfi\000_Z12floor_kernelILi4EEvPKjPfi.kd\000__hip_cuid_2edb253f577969c9\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\001\000\000\000\000\000\000\200\022\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\002\000\000\027\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\001\000\000\000\000\000\000@$\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\002\000\000\027\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\001\000\000\000\000\000\000\0006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\340\001\000\000\027\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\001\000\000\000\000\000\000\300D\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320\000\000\000\022\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\001\000\000\000\000\000\000\200K\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\001\000\000\025\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\020\000\000\000\000\000\000\000\001zR\000\004\004\020\001\033\000\000\000\030\000\000\000\030\000\000\000$\021\000\000\330\021\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\0004\000\000\000\b#\000\000\324\021\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000P\000\000\000\3544\000\000\270\016\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000l\000\000\000\320C\000\000d\006\000\000\000\017\00406\351\002\007\020\000\000\000\030\000\000\000\210\000\000\000\264J\000\000\274\r\000\000\000\017\00406\351\002\007\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\214\000F\326\000\005\001\002\001\000\205\277\000A\000\364\000\000\000\370\200\000\000\364\020\000\000\370\200\000\210\276\t\000\207\277\b\000\211\276\000\0004\330\214\000\000\000\000\000\306\277\301N\200\276\b\000\212\276\b\000\213\276\b\000\214\276\b\000\215\276\b\000\216\276\b\000\217\276\203\000\0020\001\000\207\277\377\002\0026\370\000\000\000\000\000\307\277\002\201\004\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\004@\005\356\225\000\000\000\001\000\000\000\b\000\020\312\r\000\006\001\n\000\020\312\017\000\b\003\t\002\004~\013\000\020\312\f\000\004\004\016\002\016~4\002\207\277\b\003\000\177\b\003\360~\b\003\340~\b\001\020\312\007\001fh\b\001\020\312\007\001^`\b\001\020\312\007\001VX\b\001\020\312\007\001NP\b\001\020\312\007\001FH\b\001\020\312\007\001>@\b\001\020\312\007\00168\b\001\020\312\007\001.0\b\001\020\312\007\001&(\b\001\020\312\007\001\036 \b\001\020\312\007\001\026\030\b\001\020\312\007\001\016\020\007\001\020\312\006\001~\177\005\001\020\312\004\001|}\003\001\020\312\002\001z{\001\003\362~\007\001\020\312\006\001vw\005\001\020\312\004\001tu\003\001\020\312\002\001rs\001\003\342~\007\001\020\312\006\001no\005\001\020\312\004\001lm\003\001\020\312\002\001jk\001\001\020\312\006\001fi\005\001\020\312\004\001de\003\001\020\312\002\001bc\001\001\020\312\006\001^a\005\001\020\312\004\001\\]\003\001\020\312\002\001Z[\001\001\020\312\006\001VY\005\001\020\312\004\001TU\003\001\020\312\002\001RS\001\001\020\312\006\001NQ\005\001\020\312\004\001LM\003\001\020\312\002\001JK\001\001\020\312\006\001FI\005\001\020\312\004\001DE\003\001\020\312\002\001BC\001\001\020\312\006\001>A\005\001\020\312\004\001<=\003\001\020\312\002\001:;\001\001\020\312\006\00169\005\001\020\312\004\00145\003\001\020\312\002\00123\001\001\020\312\006\001.1\005\001\020\312\004\001,-\003\001\020\312\002\001*+\001\001\020\312\006\001&)\005\001\020\312\004\001$%\003\001\020\312\002\001\"#\001\001\020\312\006\001\036!\005\001\020\312\004\001\034\035\003\001\020\312\002\001\032\033\001\001\020\312\006\001\026\031\005\001\020\312\004\001\024\025\003\001\020\312\002\001\022\023\001\001\020\312\006\001\016\021\005\001\020\312\004\001\f\r\003\001\020\312\002\001\n\013\001\003\022~\300\002\242\277\201\000\0023\253\000\037\327\301\000\001\002\362\000\020\312\377\000\256\255\000\000\200\377\000\000\300\277\225\001\020\312\226\001\232\227\210\002\0037\220VY;\225\001\020\312\226\001\230\231\223\000\207\277\203\002\0079\203#\006\177\001\000\207\277\377\0069\021o\022\203:\204\002\0059\202\002\t9\201\002\0139\207\002\0179\206\002\0219\202#\004\177\210\002\rK\205\002\0039\205#\n\177\210#\020\177\204#\b\177\207#\016\177\201#\002\177\377\0047\021o\022\203:\206#\f\177\377\020C\021o\022\203:\377\b\307\310\377\n\237\235o\022\203:\377\002E\021o\022\203:\377\002\307\310\377\020\251\247\n\327#<\200\000\006\312\377\f\253\201\n\327#<\377\f\307\310\377\016\241\237o\022\203:\377\n\307\310\377\b\245\243\n\327#<\377\006\307\310\377\004\247\245\n\327#<\004\000\207\277\377\016\321\310\201\001\202\251\n\327#<\201\001\020\312\201\001\204\203\201\001\020\312\201\001\206\205\201\001\020\312\201\001\210\207\201\003^\177\231\001\240\277~\003~\214\000\000\306\277\212a\025\007\256\023c\t\377\\;|\000\000\200\377\377bc\021;\252\270?\261Kb\177\235\377\210\277\200b]\003\002\301\002\201\231\000\207\277\002\200\006\277\255]\301\310\257]\213\255\256|\374\326\213\0277\006\221\002\207\277\256U^\177\260\000\023\326\256_\313#\241\000\207\277\260__W\260j\374\326\215\0277\006\260_c\021\221\000\207\277\262\000\023\326\256c\303&\262_cW\241\000\207\277\256\000\023\326\256c\303&\235\377\210\277\256\0007\326\256_\307\006\241\000\207\277\257\000'\326\256\0277\006\256|\374\326\213\027;\006\256U`\177\225\000\207\277\261\000\023\326\256a\313#\261aaW\261j\374\326\216\027;\006\221\000\207\277\261ae\021\263\000\023\326\256e\307&\221\000\207\277\263aeW\256\000\023\326\256e\307&\235\377\210\277\221\000\207\277\256\0007\326\256a\313\006\260\000'\326\256\027;\006\256|\374\326\213\027?\006\221\002\207\277\256Ub\177\262\000\023\326\256c\313#\241\000\207\277\262ccW\262j\374\326\217\027?\006\262cg\021\221\000\207\277\264\000\023\326\256g\313&\264cgW\241\000\207\277\256\000\023\326\256g\313&\235\377\210\277\256\0007\326\256c\317\006\241\000\207\277\261\000'\326\256\027?\006\256|\374\326\213\027C\006\256Ud\177\225\000\207\277\263\000\023\326\256e\313#\263eeW\263j\374\326\220\027C\006\221\000\207\277\263ei\021\265\000\023\326\256i\317&\221\000\207\277\265eiW\256\000\023\326\256i\317&\235\377\210\277\221\000\207\277\256\0007\326\256e\323\006\262\000'\326\256\027C\006\256|\374\326\213\027G\006\221\002\207\277\256Uf\177\264\000\023\326\256g\313#\241\000\207\277\264ggW\264j\374\326\221\027G\006\264gk\021\221\000\207\277\266\000\023\326\256k\323&\266gkW\241\000\207\277\256\000\023\326\256k\323&\235\377\210\277\256\0007\326\256g\327\006\241\000\207\277\263\000'\326\256\027G\006\256|\374\326\213\027K\006\256Uh\177\225\000\207\277\265\000\023\326\256i\313#\265iiW\265j\374\326\222\027K\006\221\000\207\277\265im\021\267\000\023\326\256m\327&\221\000\207\277\267imW\256\000\023\326\256m\327&\235\377\210\277\221\000\207\277\256\0007\326\256i\333\006\264\000'\326\256\027K\006\256|\374\326\213\027O\006\221\002\207\277\256Uj\177\266\000\023\326\256k\313#\241\000\207\277\266kkW\266j\374\326\223\027O\006\266ko\021\221\000\207\277\270\000\023\326\256o\333&\270koW\241\000\207\277\256\000\023\326\256o\333&\235\377\210\277\256\0007\326\256k\337\006\241\000\207\277\265\000'\326\256\027O\006\256|\374\326\213\027S\006\256Ul\177\225\000\207\277\267\000\023\326\256m\313#\267mmW\267j\374\326\224\027S\006\221\000\207\277\267mq\021\271\000\023\326\256q\337&\221\000\207\277\271mqW\256\000\023\326\256q\337&\235\377\210\277\221\000\207\277\256\0007\326\256m\343\006\266\000'\326\256\027S\006\256|\374\326\213\027\267\006\221\002\207\277\256Un\177\270\000\023\326\256o\313#\241\000\207\277\270ooW\270j\374\326\255\027\267\006\270os\021\221\000\207\277\272\000\023\326\256s\343&\272osW\241\000\207\277\256\000\023\326\256s\343&\235\377\210\277\256\0007\326\256o\347\006\261\001\207\277\267\000'\326\256\027\267\006\256\000\234\325\201\001\001\002\256@\234\325\200\000\001\002\267\001\001\021\223\001\207\277\255\000\234\325\256\001\001\002\255H\234\325\256\001\001\002\256\000i\327\263i\003\002\256@i\327\265m\003\002\267\377\306\310\267\371|\177\255\000i\327\257a\003\002\255@i\327\261e\003\002\267\375\306\310\267\373|~\267\365\306\310\267\367zz\267\361\306\310\267\363xx\267\355\306\310\267\357vv\267\351\306\310\267\353tt\267\345\306\310\267\347rr\267\341\306\310\267\343pp\267\335\334\020\267\337\306\310\267\331lo\267\333\306\310\267\325jm\267\327\306\310\267\321hk\267\323\306\310\267\315fiy@F\314\225[\347\035\267\317\306\310\267\311dg\267\313\306\310\267\305be\267\307\306\310\267\301`c\267\303\306\310\267\275^aq@F\314\225[\307\035\267\277\306\310\267\271\\_\267\273\306\310\267\265Z]\267\267\306\310\267\261X[\267\263\306\310\267\255VYi@F\314\225[\247\035\267\257\306\310\267\251TW\267\253\306\310\267\245RU\267\247\306\310\267\241PS\267\243\306\310\267\235NQa@F\314\225[\207\035\267\237\306\310\267\231LO\267\233\306\310\267\225JM\267\227\306\310\267\221HK\267\223\306\310\267\215FIY@F\314\225[g\035\267\217\306\310\267\211DG\267\213\306\310\267\205BE\267\207\306\310\267\201@C\267\203\306\310\267}>AQ@F\314\225[G\035\267\177\306\310\267y<?\267{\306\310\267u:=\267w\306\310\267q8;\267s\306\310\267m69I@F\314\225['\035\267o\306\310\267i47\267k\306\310\267e25\267g\306\310\267a03\267c\306\310\267].1A@F\314\225[\007\035\267_\306\310\267Y,/\267[\306\310\267U*-\267W\306\310\267Q(+\267S\306\310\267M&)9@F\314\225[\347\034\267O\306\310\267I$'\267K\306\310\267E\"%\267G\306\310\267A #\267C\306\310\267=\036!1@F\314\225[\307\034\267?\306\310\2679\034\037\267;\306\310\2675\032\035\2677\306\310\2671\030\033\2673\306\310\267-\026\031)@F\314\225[\247\034\267/\306\310\267)\024\027\267+\306\310\267%\022\025\267'\306\310\267!\020\023\267#\306\310\267\035\016\021!@F\314\225[\207\034\267\037\306\310\267\031\f\017\267\033\306\310\267\025\n\r\267\027\306\310\267\021\b\013\267\023\306\310\267\r\006\t\031@F\314\225[g\034\267\017\306\310\267\t\004\007\267\013\306\310\267\005\002\005\267\007\006\020\267\003\002\020\021@F\314\225[G\034\t@F\314\225['\034\001@F\314\225[\007\034\211\001\020\312\212\001\256\256\213\003Z\177\356\000\242\277\322\000\207\277\210\001\020\312\207\001\222\224\206\001\020\312\205\001\220\222\204\001\020\312\203\001\216\220\202\001\020\312\201\001\214\216\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\0361\001\207\277\215@F\314\22737\036\240X\231|~\000\203\276\377\034\307\310\377\032\213\212\000\000\000>\243\001\207\277\377\036\307\310\377 \217\215\000\000\000>\377\"!\021\000\000\000>\212;\023\021\024\002\207\277\213=\037\021\377$\307\310\2167\223\221\000\000\000>\224\001\207\277\2159\307\310\220E\261\260\211\000*\326\217\377%\006\000\000\200\377\377&\037\021\000\000\000>\224\001\207\277\377(\307\310\221C\225\223\000\000\000>\211\000*\326\211aK\006\235\377\210\277\253Y%\0033\002\207\277\223?a\021\217Ae\021\211\000*\326\211cS\006\202$c1\302\000\207\277\211\000*\326\211e\303\006\000\000\314\332\261\211\000\222\000\000\306\277\211\000*\326\256\023K\006\212\000\023\326\212;'\206\215\000\023\326\2159'\206\216\000\023\326\2167'\206\213\000\023\326\213='\206\221\000\023\326\221C'\206\024\002\207\277\377\024\307\310\377\032\215\212;\252\270?\377\034\035\021;\252\270?\220\000\023\326\220E'\206\217\000\023\326\217A'\206\004\000\207\277\212K\024\177\377\022%|\000\000\200\377\377\026\027\021;\252\270?\377\"#\021;\252\270?\216K\034\177\377 \307\310\377\036\217\220;\252\270?\003\000\207\277\213K\026\177\215K\032\177\235\377\210\277\212\000\001\325\212\001\251\001\220K \177\217K\036\177\221K\"\177\260\000\001\325\216\001\251\001\244\025\035\021\222\000\023\326\223?'\206\213\000\001\325\213\001\251\001\223\000\001\325\215\001\251\001\227\001\207\277\262\000\001\325\220\001\251\001\377$\307\310\243\027\215\222;\252\270?\264\000\001\325\217\001\251\001\213\025\025\007\263\000\001\325\221\001\251\001\324\001\207\277\222K$\177\245'\307\310\246a\221\217\224\000*\326\215\0019\006\212'\007\311\247e\221\212\251i'\021\265\000*\326\224\037C\006\265\001\207\277\213\000\001\325\222\001\251\001\250g%\021\212a\025\007\252\027)\021\223\001\207\277\265\000*\326\265#K\006\212e\025\007\022\001\207\277\265\000*\326\265'S\006\212g\025\007\000\000\314\332\261\265\000\262\212i\025\007\301\000\207\277\212\027\025\007\000\000\314\332\261\212\000\260\001\000\306\277\262e\027-\265\027c-\255\003\026\177\002\000\207\277\200b#}\243\375\245\277\213|\374\326\377\376\305\006\000\000\340C\221\002\207\277\213Ud\177\263\000\023\326\213e\313#\241\000\207\277\263eeW\263j\374\326\261\377\305\006\000\000\340C\263ei\021\221\000\207\277\265\000\023\326\213i\317&\265eiW\241\000\207\277\213\000\023\326\213i\317&\235\377\210\277\213\0007\326\213e\323\006\221\000\207\277\213\000'\326\213\377\305\006\000\000\340C\377\026\027-\000\000\200\037\205\375\240\277\362\000\020\312\200\000\224\213\377\000\020\312\200\000\222\211\000\000\200\377\200\000\020\312\200\000\220\223\200\000\020\312\200\000\216\221\200\000\020\312\200\000\212\217\200\002\032\177\000\000\000\364$\000\000\370\2133\306\310\2135\032\031\2137\306\310\2139\034\033\213;\306\310\213=\036\035\213?\306\310\213A \037\213\363\306\310\213\365zy\213\367\306\310\213\371|{\213\303\306\310\213\305ba\213\307\306\310\213\311dc\213\223\306\310\213\225JI\213\227\306\310\213\231LK\000\000\307\277\377\000\000\213\377\377\000\000\213c\306\310\213e21\201|\376\326u\000\000\004\213g\306\310\213i43\213#\306\310\213%\022\021\213'\306\310\213)\024\023\213\023\306\310\213\025\n\t\201\000,\327\377\002\003\002\214\000\000\000\213\027\306\310\213\031\f\013\213\003\306\310\213\005\000\000\213\007\306\310\213\t\002\002\213\373\306\310\213\375~}\237\002\0055\213\377\306\310\213\001\201\177\213\313\306\310\213\315feC\002\207\277\202\002\003?\213\317\306\310\213\321hg\213\233\306\310\213\235NM\213\237\306\310\213\241PO\201j\000\327\006\002\003\002\235\377\210\277\202| \325\007\004\253\001\001\000\205\277|@\007\356\000\000\200\f\201\200\001\000|@\007\356\000\000\200\016\201\220\001\000\000\000\330\330\214\000\000\031\213k\306\310\213m65\213o\306\310\213q87\213+\306\310\213-\026\025\213/\306\310\2131\030\027\213\033\306\310\213\035\016\r\213\037\306\310\213!\020\017\213\013\306\310\213\r\004\004\213\017\306\310\213\021\006\006\213\343\306\310\213\345rq\213\347\306\310\213\351ts\213\263\306\310\213\265ZY\213\267\306\310\213\271\\[\213\203\306\310\213\205BA\213\207\306\310\213\211DC\213S\306\310\213U*)\213W\306\310\213Y,+\213\353\306\310\213\355vu\213\357\306\310\213\361xw\213\273\306\310\213\275^]\213\277\306\310\213\301`_\213\213\306\310\213\215FE\213\217\306\310\213\221HG\213[\306\310\213].-\213_\306\310\213a0/\213\323\306\310\213\325ji\213\327\306\310\213\331lk\213\243\306\310\213\245RQ\213\247\306\310\213\251TS\213s\306\310\213u:9\213w\306\310\213y<;\213C\306\310\213E\"!\213G\306\310\213I$#\000\000\306\277\031\r\030\177\213\333\306\310\213\335nm\213\337\306\310\213\341po\005\000\205\277|@\007\356\000\000\200<\201\000\000\000|@\007\356\000\000\200>\201\020\000\000|@\007\356\000\000\2008\201 \000\000|@\007\356\000\000\200:\2010\000\000|@\007\356\000\000\2004\201@\000\000|@\007\356\000\000\2006\201P\000\000\213\253\306\310\213\255VU\213\257\306\310\213\261XW\005\000\205\277|@\007\356\000\000\2000\201`\000\000|@\007\356\000\000\2002\201p\000\000|@\007\356\000\000\200,\201\200\000\000|@\007\356\000\000\200.\201\220\000\000|@\007\356\000\000\200(\201\240\000\000|@\007\356\000\000\200*\201\260\000\000\213{\306\310\213}>=\213\177\306\310\213\201@?\005\000\205\277|@\007\356\000\000\200$\201\300\000\000|@\007\356\000\000\200&\201\320\000\000|@\007\356\000\000\200 \201\340\000\000|@\007\356\000\000\200\"\201\360\000\000|@\007\356\000\000\200\034\201\000\001\000|@\007\356\000\000\200\036\201\020\001\000\213K\306\310\213M&%\213O\306\310\213Q('\016\000\205\277|@\007\356\000\000\200\030\201 \001\000|@\007\356\000\000\200\032\2010\001\000|@\007\356\000\000\200\024\201@\001\000|@\007\356\000\000\200\026\201P\001\000|@\007\356\000\000\200\020\201`\001\000|@\007\356\000\000\200\022\201p\001\000|@\007\356\000\000\200\b\201\240\001\000|@\007\356\000\000\200\n\201\260\001\000|@\007\356\000\000\200\004\201\300\001\000|@\007\356\000\000\200\006\201\320\001\000|@\007\356\000\000\000\000\201\340\001\000|@\007\356\000\000\000\002\201\360\001\000|@\007\356\000\000\200F\201\000\002\000|@\007\356\000\000\200H\201\020\002\000|@\007\356\000\000\200D\201 \002\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\214\000F\326\000\005\001\002\001\000\205\277\000A\000\364\000\000\000\370\200\000\000\364\020\000\000\370\200\000\210\276\t\000\207\277\b\000\211\276\000\0004\330\214\000\000\000\000\000\306\277\301N\200\276\b\000\212\276\b\000\213\276\b\000\214\276\b\000\215\276\b\000\216\276\b\000\217\276\203\000\0020\001\000\207\277\377\002\0026\370\000\000\000\000\000\307\277\002\201\004\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\004@\005\356\225\000\000\000\001\000\000\000\b\000\020\312\r\000\006\001\n\000\020\312\017\000\b\003\t\002\004~\013\000\020\312\f\000\004\004\016\002\016~4\002\207\277\b\003\000\177\b\003\360~\b\003\340~\b\001\020\312\007\001fh\b\001\020\312\007\001^`\b\001\020\312\007\001VX\b\001\020\312\007\001NP\b\001\020\312\007\001FH\b\001\020\312\007\001>@\b\001\020\312\007\00168\b\001\020\312\007\001.0\b\001\020\312\007\001&(\b\001\020\312\007\001\036 \b\001\020\312\007\001\026\030\b\001\020\312\007\001\016\020\007\001\020\312\006\001~\177\005\001\020\312\004\001|}\003\001\020\312\002\001z{\001\003\362~\007\001\020\312\006\001vw\005\001\020\312\004\001tu\003\001\020\312\002\001rs\001\003\342~\007\001\020\312\006\001no\005\001\020\312\004\001lm\003\001\020\312\002\001jk\001\001\020\312\006\001fi\005\001\020\312\004\001de\003\001\020\312\002\001bc\001\001\020\312\006\001^a\005\001\020\312\004\001\\]\003\001\020\312\002\001Z[\001\001\020\312\006\001VY\005\001\020\312\004\001TU\003\001\020\312\002\001RS\001\001\020\312\006\001NQ\005\001\020\312\004\001LM\003\001\020\312\002\001JK\001\001\020\312\006\001FI\005\001\020\312\004\001DE\003\001\020\312\002\001BC\001\001\020\312\006\001>A\005\001\020\312\004\001<=\003\001\020\312\002\001:;\001\001\020\312\006\00169\005\001\020\312\004\00145\003\001\020\312\002\00123\001\001\020\312\006\001.1\005\001\020\312\004\001,-\003\001\020\312\002\001*+\001\001\020\312\006\001&)\005\001\020\312\004\001$%\003\001\020\312\002\001\"#\001\001\020\312\006\001\036!\005\001\020\312\004\001\034\035\003\001\020\312\002\001\032\033\001\001\020\312\006\001\026\031\005\001\020\312\004\001\024\025\003\001\020\312\002\001\022\023\001\001\020\312\006\001\016\021\005\001\020\312\004\001\f\r\003\001\020\312\002\001\n\013\001\003\022~\277\002\242\277\201\000\0023\362\000\020\312\377\000\254\253\000\000\200\377\000\000\300\277\225\001\020\312\226\001\232\227\303\001\207\277\210\002\0037\225\0032\177\377\000\203\276\0202Tv\226\0030\177\203\002\0079\221\000\207\277\203#\006\177\377\0069\021o\022\203:\204\002\0059\202\002\t9\201\002\0139\207\002\0179\206\002\0219\202#\004\177\210\002\rK\205\002\0039\205#\n\177\210#\020\177\204#\b\177\207#\016\177\201#\002\177\377\0047\021o\022\203:\206#\f\177\377\020C\021o\022\203:\377\b\307\310\377\n\237\235o\022\203:\377\002E\021o\022\203:\377\002\307\310\377\020\251\247\n\327#<\200\000\006\312\377\f\253\201\n\327#<\377\f\307\310\377\016\241\237o\022\203:\377\n\307\310\377\b\245\243\n\327#<\377\006\307\310\377\004\247\245\n\327#<\004\000\207\277\377\016\321\310\201\001\202\251\n\327#<\201\001\020\312\201\001\204\203\201\001\020\312\201\001\206\205\201\001\020\312\201\001\210\207\201\003Z\177\231\001\240\277~\004~\214\254\023I\311\212]\213\257\377X;|\000\000\200\377\002\000\207\277\377^_\021;\252\270?\257K^\177\235\377\210\277\200^Y\003\002\301\002\201\236\377\210\277\002\200\006\277\255Y\007\310\253Y\253\212\254|\374\326\213\0277\006\221\002\207\277\254UZ\177\256\000\023\326\254[\313#\241\000\207\277\256[[W\256j\374\326\215\0277\006\256[_\021\221\000\207\277\260\000\023\326\254_\273&\260[_W\241\000\207\277\254\000\023\326\254_\273&\235\377\210\277\254\0007\326\254[\277\006\241\000\207\277\255\000'\326\254\0277\006\254|\374\326\213\027;\006\254U\\\177\225\000\207\277\257\000\023\326\254]\313#\257]]W\257j\374\326\216\027;\006\221\000\207\277\257]a\021\261\000\023\326\254a\277&\221\000\207\277\261]aW\254\000\023\326\254a\277&\235\377\210\277\221\000\207\277\254\0007\326\254]\303\006\256\000'\326\254\027;\006\254|\374\326\213\027?\006\221\002\207\277\254U^\177\260\000\023\326\254_\313#\241\000\207\277\260__W\260j\374\326\217\027?\006\260_c\021\221\000\207\277\262\000\023\326\254c\303&\262_cW\241\000\207\277\254\000\023\326\254c\303&\235\377\210\277\254\0007\326\254_\307\006\241\000\207\277\257\000'\326\254\027?\006\254|\374\326\213\027C\006\254U`\177\225\000\207\277\261\000\023\326\254a\313#\261aaW\261j\374\326\220\027C\006\221\000\207\277\261ae\021\263\000\023\326\254e\307&\221\000\207\277\263aeW\254\000\023\326\254e\307&\235\377\210\277\221\000\207\277\254\0007\326\254a\313\006\260\000'\326\254\027C\006\254|\374\326\213\027G\006\221\002\207\277\254Ub\177\262\000\023\326\254c\313#\241\000\207\277\262ccW\262j\374\326\221\027G\006\262cg\021\221\000\207\277\264\000\023\326\254g\313&\264cgW\241\000\207\277\254\000\023\326\254g\313&\235\377\210\277\254\0007\326\254c\317\006\241\000\207\277\261\000'\326\254\027G\006\254|\374\326\213\027K\006\254Ud\177\225\000\207\277\263\000\023\326\254e\313#\263eeW\263j\374\326\222\027K\006\221\000\207\277\263ei\021\265\000\023\326\254i\317&\221\000\207\277\265eiW\254\000\023\326\254i\317&\235\377\210\277\221\000\207\277\254\0007\326\254e\323\006\262\000'\326\254\027K\006\254|\374\326\213\027O\006\221\002\207\277\254Uf\177\264\000\023\326\254g\313#\241\000\207\277\264ggW\264j\374\326\223\027O\006\264gk\021\221\000\207\277\266\000\023\326\254k\323&\266gkW\241\000\207\277\254\000\023\326\254k\323&\235\377\210\277\254\0007\326\254g\327\006\241\000\207\277\263\000'\326\254\027O\006\254|\374\326\213\027S\006\254Uh\177\225\000\207\277\265\000\023\326\254i\313#\265iiW\265j\374\326\224\027S\006\221\000\207\277\265im\021\267\000\023\326\254m\327&\221\000\207\277\267imW\254\000\023\326\254m\327&\235\377\210\277\221\000\207\277\254\0007\326\254i\333\006\264\000'\326\254\027S\006\254|\374\326\213\027\257\006\221\002\207\277\254Uj\177\266\000\023\326\254k\313#\241\000\207\277\266kkW\266j\374\326\253\027\257\006\266ko\021\221\000\207\277\270\000\023\326\254o\333&\270koW\241\000\207\277\254\000\023\326\254o\333&\235\377\210\277\254\0007\326\254k\337\006\261\001\207\277\265\000'\326\254\027\257\006\254\000\234\325\201\001\001\002\254@\234\325\200\000\001\002\265\001\001\021\223\001\207\277\253\000\234\325\254\001\001\002\253H\234\325\254\001\001\002\254\000i\327\261e\003\002\254@i\327\263i\003\002\265\377\306\310\265\371|\177\253\000i\327\255]\003\002\253@i\327\257a\003\002\265\375\306\310\265\373|~\265\365\306\310\265\367zz\265\361\306\310\265\363xx\265\355\306\310\265\357vv\265\351\306\310\265\353tt\265\345\306\310\265\347rr\265\341\306\310\265\343pp\265\335\334\020\265\337\306\310\265\331lo\265\333\306\310\265\325jm\265\327\306\310\265\321hk\265\323\306\310\265\315fiy@F\314\225W\347\035\265\317\306\310\265\311dg\265\313\306\310\265\305be\265\307\306\310\265\301`c\265\303\306\310\265\275^aq@F\314\225W\307\035\265\277\306\310\265\271\\_\265\273\306\310\265\265Z]\265\267\306\310\265\261X[\265\263\306\310\265\255VYi@F\314\225W\247\035\265\257\306\310\265\251TW\265\253\306\310\265\245RU\265\247\306\310\265\241PS\265\243\306\310\265\235NQa@F\314\225W\207\035\265\237\306\310\265\231LO\265\233\306\310\265\225JM\265\227\306\310\265\221HK\265\223\306\310\265\215FIY@F\314\225Wg\035\265\217\306\310\265\211DG\265\213\306\310\265\205BE\265\207\306\310\265\201@C\265\203\306\310\265}>AQ@F\314\225WG\035\265\177\306\310\265y<?\265{\306\310\265u:=\265w\306\310\265q8;\265s\306\310\265m69I@F\314\225W'\035\265o\306\310\265i47\265k\306\310\265e25\265g\306\310\265a03\265c\306\310\265].1A@F\314\225W\007\035\265_\306\310\265Y,/\265[\306\310\265U*-\265W\306\310\265Q(+\265S\306\310\265M&)9@F\314\225W\347\034\265O\306\310\265I$'\265K\306\310\265E\"%\265G\306\310\265A #\265C\306\310\265=\036!1@F\314\225W\307\034\265?\306\310\2659\034\037\265;\306\310\2655\032\035\2657\306\310\2651\030\033\2653\306\310\265-\026\031)@F\314\225W\247\034\265/\306\310\265)\024\027\265+\306\310\265%\022\025\265'\306\310\265!\020\023\265#\306\310\265\035\016\021!@F\314\225W\207\034\265\037\306\310\265\031\f\017\265\033\306\310\265\025\n\r\265\027\306\310\265\021\b\013\265\023\306\310\265\r\006\t\031@F\314\225Wg\034\265\017\306\310\265\t\004\007\265\013\306\310\265\005\002\005\265\007\006\020\265\003\002\020\021@F\314\225WG\034\t@F\314\225W'\034\001@F\314\225W\007\034\211\001\020\312\212\001\254\254\213\003V\177\356\000\242\277\322\000\207\277\210\001\020\312\207\001\222\224\206\001\020\312\205\001\220\222\204\001\020\312\203\001\216\220\202\001\020\312\201\001\214\216\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036~\000\204\276\377\034\307\310\377\032\213\212\000\000\000>\222\001\207\277\377\036\307\310\377 \217\215\000\000\000>\377\"!\021\000\000\000>\023\002\207\277\212;\023\021\213=\037\021\244\001\207\277\377$\307\310\2167\223\221\000\000\000>\2159]\021\211\000*\326\217\377%\006\000\000\200\377\377&\037\021\000\000\000>$\002\207\277\377(\307\310\221C\225\223\000\000\000>\220E_\021\211\000*\326\211]K\006\024\002\207\277\217A]\021\223?%\021\223\000\207\277\211\000*\326\211_S\006\211\000*\326\211]K\006\221\000\207\277\211\003$\177\222\000\\\326\222\007\374\003\230\272\334\376\221\000\207\277\211\000*\326\254\023K\006\213\000\023\326\213='\206\212\000\023\326\212;'\206\215\000\023\326\2159'\206\216\000\023\326\2167'\206\220\000\023\326\220E'\206\217\000\023\326\217A'\206\377\026\307\310\377\024\213\213;\252\270?\377\032\033\021;\252\270?\221\000\023\326\221C'\206\377\034\035\021;\252\270?\377 \307\310\377\036\217\220;\252\270?\213K\026\177\212K\024\177\377\022%|\000\000\200\377\222\000\023\326\223?'\206\377\"#\021;\252\270?\215K\032\177\216K\034\177\217K\036\177\220K \177\235\377\210\277\213\000\001\325\213\001\251\001\212\000\001\325\212\001\251\001\377$%\021;\252\270?\221K\"\177\223\000\001\325\215\001\251\001\256\000\001\325\216\001\251\001\243\027\307\310\244\025\217\215\261\000\001\325\217\001\251\001\213\025\025\007\222K$\177\257\000\001\325\220\001\251\001\006\000\207\277\260\000\001\325\221\001\251\001\245'\307\310\246]\221\217\213\000*\326\215\0019\006\212'\025\007\247_#\021\251c'\021E\002\207\277\262\000\001\325\222\001\251\001\250a%\021\213\000*\326\213\037C\006\212]\025\007\252e)\021\223\001\207\277\213\000*\326\213#K\006\212_\025\007\022\001\207\277\213\000*\326\213'S\006\212a\025\007\022\001\207\277\213\003\\\177\212c\025\007\022\001\207\277\256\000\\\326\256\007\374\003\230\272\334\376\212e\025\007\222\000\207\277\256]_-\212\001\024\312\213_\257\256\241\001\207\277\256\000\\\326\256\007\374\003\230\272\334\376\253\003\026\177\200^#}\243\375\245\277\213|\374\326\377\376\275\006\000\000\340C\221\002\207\277\213U`\177\261\000\023\326\213a\313#\241\000\207\277\261aaW\261j\374\326\257\377\275\006\000\000\340C\261ae\021\221\000\207\277\263\000\023\326\213e\307&\263aeW\241\000\207\277\213\000\023\326\213e\307&\235\377\210\277\213\0007\326\213a\313\006\221\000\207\277\213\000'\326\213\377\275\006\000\000\340C\377\026\027-\000\000\200\037\205\375\240\277\362\000\020\312\200\000\224\213\377\000\020\312\200\000\222\211\000\000\200\377\200\000\020\312\200\000\220\223\200\000\020\312\200\000\216\221\200\000\020\312\200\000\212\217\200\002\032\177\000\000\000\364$\000\000\370\2133\306\310\2135\032\031\2137\306\310\2139\034\033\213;\306\310\213=\036\035\213?\306\310\213A \037\213\363\306\310\213\365zy\213\367\306\310\213\371|{\213\303\306\310\213\305ba\213\307\306\310\213\311dc\213\223\306\310\213\225JI\213\227\306\310\213\231LK\000\000\307\277\377\000\000\213\377\377\000\000\213c\306\310\213e21\201|\376\326u\000\000\004\213g\306\310\213i43\213#\306\310\213%\022\021\213'\306\310\213)\024\023\213\023\306\310\213\025\n\t\201\000,\327\377\002\003\002\214\000\000\000\213\027\306\310\213\031\f\013\213\003\306\310\213\005\000\000\213\007\306\310\213\t\002\002\213\373\306\310\213\375~}\237\002\0055\213\377\306\310\213\001\201\177\213\313\306\310\213\315feC\002\207\277\202\002\003?\213\317\306\310\213\321hg\213\233\306\310\213\235NM\213\237\306\310\213\241PO\201j\000\327\006\002\003\002\235\377\210\277\202| \325\007\004\253\001\001\000\205\277|@\007\356\000\000\200\f\201\200\001\000|@\007\356\000\000\200\016\201\220\001\000\000\000\330\330\214\000\000\031\213k\306\310\213m65\213o\306\310\213q87\213+\306\310\213-\026\025\213/\306\310\2131\030\027\213\033\306\310\213\035\016\r\213\037\306\310\213!\020\017\213\013\306\310\213\r\004\004\213\017\306\310\213\021\006\006\213\343\306\310\213\345rq\213\347\306\310\213\351ts\213\263\306\310\213\265ZY\213\267\306\310\213\271\\[\213\203\306\310\213\205BA\213\207\306\310\213\211DC\213S\306\310\213U*)\213W\306\310\213Y,+\213\353\306\310\213\355vu\213\357\306\310\213\361xw\213\273\306\310\213\275^]\213\277\306\310\213\301`_\213\213\306\310\213\215FE\213\217\306\310\213\221HG\213[\306\310\213].-\213_\306\310\213a0/\213\323\306\310\213\325ji\213\327\306\310\213\331lk\213\243\306\310\213\245RQ\213\247\306\310\213\251TS\213s\306\310\213u:9\213w\306\310\213y<;\213C\306\310\213E\"!\213G\306\310\213I$#\000\000\306\277\031\r\030\177\213\333\306\310\213\335nm\213\337\306\310\213\341po\005\000\205\277|@\007\356\000\000\200<\201\000\000\000|@\007\356\000\000\200>\201\020\000\000|@\007\356\000\000\2008\201 \000\000|@\007\356\000\000\200:\2010\000\000|@\007\356\000\000\2004\201@\000\000|@\007\356\000\000\2006\201P\000\000\213\253\306\310\213\255VU\213\257\306\310\213\261XW\005\000\205\277|@\007\356\000\000\2000\201`\000\000|@\007\356\000\000\2002\201p\000\000|@\007\356\000\000\200,\201\200\000\000|@\007\356\000\000\200.\201\220\000\000|@\007\356\000\000\200(\201\240\000\000|@\007\356\000\000\200*\201\260\000\000\213{\306\310\213}>=\213\177\306\310\213\201@?\005\000\205\277|@\007\356\000\000\200$\201\300\000\000|@\007\356\000\000\200&\201\320\000\000|@\007\356\000\000\200 \201\340\000\000|@\007\356\000\000\200\"\201\360\000\000|@\007\356\000\000\200\034\201\000\001\000|@\007\356\000\000\200\036\201\020\001\000\213K\306\310\213M&%\213O\306\310\213Q('\016\000\205\277|@\007\356\000\000\200\030\201 \001\000|@\007\356\000\000\200\032\2010\001\000|@\007\356\000\000\200\024\201@\001\000|@\007\356\000\000\200\026\201P\001\000|@\007\356\000\000\200\020\201`\001\000|@\007\356\000\000\200\022\201p\001\000|@\007\356\000\000\200\b\201\240\001\000|@\007\356\000\000\200\n\201\260\001\000|@\007\356\000\000\200\004\201\300\001\000|@\007\356\000\000\200\006\201\320\001\000|@\007\356\000\000\000\000\201\340\001\000|@\007\356\000\000\000\002\201\360\001\000|@\007\356\000\000\200F\201\000\002\000|@\007\356\000\000\200H\201\020\002\000|@\007\356\000\000\200D\201 \002\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\214\000F\326\000\005\001\002\001\000\205\277\000A\000\364\000\000\000\370\200\000\000\364\020\000\000\370\200\000\210\276\t\000\207\277\b\000\211\276\000\0004\330\214\000\000\000\000\000\306\277\301N\200\276\b\000\212\276\b\000\213\276\b\000\214\276\b\000\215\276\b\000\216\276\b\000\217\276\203\000\0020\001\000\207\277\377\002\0026\370\000\000\000\000\000\307\277\002\201\004\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\004@\005\356\225\000\000\000\001\000\000\000\b\000\020\312\r\000\006\001\n\000\020\312\017\000\b\003\t\002\004~\013\000\020\312\f\000\004\004\016\002\016~4\002\207\277\b\003\000\177\b\003\360~\b\003\340~\b\001\020\312\007\001fh\b\001\020\312\007\001^`\b\001\020\312\007\001VX\b\001\020\312\007\001NP\b\001\020\312\007\001FH\b\001\020\312\007\001>@\b\001\020\312\007\00168\b\001\020\312\007\001.0\b\001\020\312\007\001&(\b\001\020\312\007\001\036 \b\001\020\312\007\001\026\030\b\001\020\312\007\001\016\020\007\001\020\312\006\001~\177\005\001\020\312\004\001|}\003\001\020\312\002\001z{\001\003\362~\007\001\020\312\006\001vw\005\001\020\312\004\001tu\003\001\020\312\002\001rs\001\003\342~\007\001\020\312\006\001no\005\001\020\312\004\001lm\003\001\020\312\002\001jk\001\001\020\312\006\001fi\005\001\020\312\004\001de\003\001\020\312\002\001bc\001\001\020\312\006\001^a\005\001\020\312\004\001\\]\003\001\020\312\002\001Z[\001\001\020\312\006\001VY\005\001\020\312\004\001TU\003\001\020\312\002\001RS\001\001\020\312\006\001NQ\005\001\020\312\004\001LM\003\001\020\312\002\001JK\001\001\020\312\006\001FI\005\001\020\312\004\001DE\003\001\020\312\002\001BC\001\001\020\312\006\001>A\005\001\020\312\004\001<=\003\001\020\312\002\001:;\001\001\020\312\006\00169\005\001\020\312\004\00145\003\001\020\312\002\00123\001\001\020\312\006\001.1\005\001\020\312\004\001,-\003\001\020\312\002\001*+\001\001\020\312\006\001&)\005\001\020\312\004\001$%\003\001\020\312\002\001\"#\001\001\020\312\006\001\036!\005\001\020\312\004\001\034\035\003\001\020\312\002\001\032\033\001\001\020\312\006\001\026\031\005\001\020\312\004\001\024\025\003\001\020\312\002\001\022\023\001\001\020\312\006\001\016\021\005\001\020\312\004\001\f\r\003\001\020\312\002\001\n\013\001\003\022~\370\001\242\277\201\000\0023\362\000\020\312\377\000\254\253\000\000\200\377\000\000\300\277\225\001\020\312\226\001\232\227\303\001\207\277\210\002\0037\225\0032\177\377\000\203\276\0202Tv\226\0030\177\203\002\0079\221\000\207\277\203#\006\177\377\0069\021o\022\203:\204\002\0059\202\002\t9\201\002\0139\207\002\0179\206\002\0219\202#\004\177\210\002\rK\205\002\0039\205#\n\177\210#\020\177\204#\b\177\207#\016\177\201#\002\177\377\0047\021o\022\203:\206#\f\177\377\020C\021o\022\203:\377\b\307\310\377\n\237\235o\022\203:\377\002E\021o\022\203:\377\002\307\310\377\020\251\247\n\327#<\200\000\006\312\377\f\253\201\n\327#<\377\f\307\310\377\016\241\237o\022\203:\377\n\307\310\377\b\245\243\n\327#<\377\006\307\310\377\004\247\245\n\327#<\004\000\207\277\377\016\321\310\201\001\202\251\n\327#<\201\001\020\312\201\001\204\203\201\001\020\312\201\001\206\205\201\001\020\312\201\001\210\207\201\003Z\177\322\000\240\277~\004~\214\254\023I\311\212]\213\257\377X;|\000\000\200\377\213U`\177\002\000\207\277\377^_\021;\252\270?\257Kb\177\257\000\234\325\201\001\001\002\257@\234\325\200\000\001\002\260\037\307\310\260!\265\263\260#\307\310\260)\271\265\024\002\207\277\256\000\234\325\257\001\001\002\256H\234\325\257\001\001\002\235\377\210\277E\002\207\277\200bY\003\260\033\307\310\260\035\263\261\260'o\021\256@i\327\263i\003\002\253Y\307\310\260%\267\2534\002\207\277\256\000i\327\261e\003\002\255Y\025W\257@i\327\267q\003\002\253a\321\310\211\001\254\253\257\000i\327\265m\003\002\002\000\207\277\253\001\307\310\253\377~\200\253\375\306\310\253\373|~\253\371\306\310\253\367z|\253\365\306\310\253\363xz\253\361\306\310\253\357vx\253\355\306\310\253\353tv\253\351\306\310\253\347rt\253\345\306\310\253\343pr\253\341\306\310\253\337np\253\335\306\310\253\333ln\253\331\306\310\253\327jl\253\325\306\310\253\323hjy@F\314\225]\347\035\253\321\306\310\253\317fh\253\315\306\310\253\313df\253\311\306\310\253\307bd\253\305\306\310\253\303`bq@F\314\225]\307\035\253\301\306\310\253\277^`\253\275\306\310\253\273\\^\253\271\306\310\253\267Z\\\253\265\306\310\253\263XZi@F\314\225]\247\035\253\261\306\310\253\257VX\253\255\306\310\253\253TV\253\251\306\310\253\247RT\253\245\306\310\253\243PRa@F\314\225]\207\035\253\241\306\310\253\237NP\253\235\306\310\253\233LN\253\231\306\310\253\227JL\253\225\306\310\253\223HJY@F\314\225]g\035\253\221\306\310\253\217FH\253\215\306\310\253\213DF\253\211\306\310\253\207BD\253\205\306\310\253\203@BQ@F\314\225]G\035\253\201\306\310\253\177>@\253}\306\310\253{<>\253y\306\310\253w:<\253u\306\310\253s8:I@F\314\225]'\035\253q\306\310\253o68\253m\306\310\253k46\253i\306\310\253g24\253e\306\310\253c02A@F\314\225]\007\035\253a\306\310\253_.0\253]\306\310\253[,.\253Y\306\310\253W*,\253U\306\310\253S(*9@F\314\225]\347\034\253Q\306\310\253O&(\253M\306\310\253K$&\253I\306\310\253G\"$\253E\306\310\253C \"1@F\314\225]\307\034\253A\306\310\253?\036 \253=\306\310\253;\034\036\2539\306\310\2537\032\034\2535\306\310\2533\030\032)@F\314\225]\247\034\2531\306\310\253/\026\030\253-\306\310\253+\024\026\253)\306\310\253'\022\024\253%\306\310\253#\020\022!@F\314\225]\207\034\253!\306\310\253\037\016\020\253\035\306\310\253\033\f\016\253\031\306\310\253\027\n\f\253\025\306\310\253\023\b\n\031@F\314\225]g\034\253\021\306\310\253\017\006\b\253\r\306\310\253\013\004\006\253\t\306\310\253\007\002\004\253\005\306\310\253\003\000\002\021@F\314\225]G\034\t@F\314\225]'\034\001@F\314\225]\007\034\212\003Z\177\213\003V\177\002\301\002\201\236\377\210\277\002\200\006\277\356\000\242\277\322\000\207\277\210\001\020\312\207\001\222\224\206\001\020\312\205\001\220\222\204\001\020\312\203\001\216\220\202\001\020\312\201\001\214\216\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036\215@F\314\22737\036\241\000\207\277\215@F\314\22737\036~\000\204\276\377\034\307\310\377\032\213\212\000\000\000>\222\001\207\277\377\036\307\310\377 \217\215\000\000\000>\377\"!\021\000\000\000>\023\002\207\277\212;\023\021\213=\037\021\244\001\207\277\377$\307\310\2167\223\221\000\000\000>\2159]\021\211\000*\326\217\377%\006\000\000\200\377\377&\037\021\000\000\000>$\002\207\277\377(\307\310\221C\225\223\000\000\000>\220E_\021\211\000*\326\211]K\006\024\002\207\277\217A]\021\223?%\021\223\000\207\277\211\000*\326\211_S\006\211\000*\326\211]K\006\221\000\207\277\211\003$\177\222\000\\\326\222\007\374\003\230\272\334\376\221\000\207\277\211\000*\326\254\023K\006\213\000\023\326\213='\206\212\000\023\326\212;'\206\215\000\023\326\2159'\206\216\000\023\326\2167'\206\220\000\023\326\220E'\206\217\000\023\326\217A'\206\377\026\307\310\377\024\213\213;\252\270?\377\032\033\021;\252\270?\221\000\023\326\221C'\206\377\034\035\021;\252\270?\377 \307\310\377\036\217\220;\252\270?\213K\026\177\212K\024\177\377\022%|\000\000\200\377\222\000\023\326\223?'\206\377\"#\021;\252\270?\215K\032\177\216K\034\177\217K\036\177\220K \177\235\377\210\277\213\000\001\325\213\001\251\001\212\000\001\325\212\001\251\001\377$%\021;\252\270?\221K\"\177\223\000\001\325\215\001\251\001\256\000\001\325\216\001\251\001\243\027\307\310\244\025\217\215\261\000\001\325\217\001\251\001\213\025\025\007\222K$\177\257\000\001\325\220\001\251\001\006\000\207\277\260\000\001\325\221\001\251\001\245'\307\310\246]\221\217\213\000*\326\215\0019\006\212'\025\007\247_#\021\251c'\021E\002\207\277\262\000\001\325\222\001\251\001\250a%\021\213\000*\326\213\037C\006\212]\025\007\252e)\021\223\001\207\277\213\000*\326\213#K\006\212_\025\007\022\001\207\277\213\000*\326\213'S\006\212a\025\007\022\001\207\277\213\003\\\177\212c\025\007\022\001\207\277\256\000\\\326\256\007\374\003\230\272\334\376\212e\025\007\222\000\207\277\256]_-\212\001\024\312\213_\257\256\241\001\207\277\256\000\\\326\256\007\374\003\230\272\334\376\253\003\026\177\200^#}j\376\245\277\213|\374\326\377\376\275\006\000\000\340C\221\002\207\277\213U`\177\261\000\023\326\213a\313#\241\000\207\277\261aaW\261j\374\326\257\377\275\006\000\000\340C\261ae\021\221\000\207\277\263\000\023\326\213e\307&\263aeW\241\000\207\277\213\000\023\326\213e\307&\235\377\210\277\213\0007\326\213a\313\006\221\000\207\277\213\000'\326\213\377\275\006\000\000\340C\377\026\027-\000\000\200\037L\376\240\277\362\000\020\312\200\000\224\213\377\000\020\312\200\000\222\211\000\000\200\377\200\000\020\312\200\000\220\223\200\000\020\312\200\000\216\221\200\000\020\312\200\000\212\217\200\002\032\177\000\000\000\364$\000\000\370\2133\306\310\2135\032\031\2137\306\310\2139\034\033\213;\306\310\213=\036\035\213?\306\310\213A \037\213\363\306\310\213\365zy\213\367\306\310\213\371|{\213\303\306\310\213\305ba\213\307\306\310\213\311dc\213\223\306\310\213\225JI\213\227\306\310\213\231LK\000\000\307\277\377\000\000\213\377\377\000\000\213c\306\310\213e21\201|\376\326u\000\000\004\213g\306\310\213i43\213#\306\310\213%\022\021\213'\306\310\213)\024\023\213\023\306\310\213\025\n\t\201\000,\327\377\002\003\002\214\000\000\000\213\027\306\310\213\031\f\013\213\003\306\310\213\005\000\000\213\007\306\310\213\t\002\002\213\373\306\310\213\375~}\237\002\0055\213\377\306\310\213\001\201\177\213\313\306\310\213\315feC\002\207\277\202\002\003?\213\317\306\310\213\321hg\213\233\306\310\213\235NM\213\237\306\310\213\241PO\201j\000\327\006\002\003\002\235\377\210\277\202| \325\007\004\253\001\001\000\205\277|@\007\356\000\000\200\f\201\200\001\000|@\007\356\000\000\200\016\201\220\001\000\000\000\330\330\214\000\000\031\213k\306\310\213m65\213o\306\310\213q87\213+\306\310\213-\026\025\213/\306\310\2131\030\027\213\033\306\310\213\035\016\r\213\037\306\310\213!\020\017\213\013\306\310\213\r\004\004\213\017\306\310\213\021\006\006\213\343\306\310\213\345rq\213\347\306\310\213\351ts\213\263\306\310\213\265ZY\213\267\306\310\213\271\\[\213\203\306\310\213\205BA\213\207\306\310\213\211DC\213S\306\310\213U*)\213W\306\310\213Y,+\213\353\306\310\213\355vu\213\357\306\310\213\361xw\213\273\306\310\213\275^]\213\277\306\310\213\301`_\213\213\306\310\213\215FE\213\217\306\310\213\221HG\213[\306\310\213].-\213_\306\310\213a0/\213\323\306\310\213\325ji\213\327\306\310\213\331lk\213\243\306\310\213\245RQ\213\247\306\310\213\251TS\213s\306\310\213u:9\213w\306\310\213y<;\213C\306\310\213E\"!\213G\306\310\213I$#\000\000\306\277\031\r\030\177\213\333\306\310\213\335nm\213\337\306\310\213\341po\005\000\205\277|@\007\356\000\000\200<\201\000\000\000|@\007\356\000\000\200>\201\020\000\000|@\007\356\000\000\2008\201 \000\000|@\007\356\000\000\200:\2010\000\000|@\007\356\000\000\2004\201@\000\000|@\007\356\000\000\2006\201P\000\000\213\253\306\310\213\255VU\213\257\306\310\213\261XW\005\000\205\277|@\007\356\000\000\2000\201`\000\000|@\007\356\000\000\2002\201p\000\000|@\007\356\000\000\200,\201\200\000\000|@\007\356\000\000\200.\201\220\000\000|@\007\356\000\000\200(\201\240\000\000|@\007\356\000\000\200*\201\260\000\000\213{\306\310\213}>=\213\177\306\310\213\201@?\005\000\205\277|@\007\356\000\000\200$\201\300\000\000|@\007\356\000\000\200&\201\320\000\000|@\007\356\000\000\200 \201\340\000\000|@\007\356\000\000\200\"\201\360\000\000|@\007\356\000\000\200\034\201\000\001\000|@\007\356\000\000\200\036\201\020\001\000\213K\306\310\213M&%\213O\306\310\213Q('\016\000\205\277|@\007\356\000\000\200\030\201 \001\000|@\007\356\000\000\200\032\2010\001\000|@\007\356\000\000\200\024\201@\001\000|@\007\356\000\000\200\026\201P\001\000|@\007\356\000\000\200\020\201`\001\000|@\007\356\000\000\200\022\201p\001\000|@\007\356\000\000\200\b\201\240\001\000|@\007\356\000\000\200\n\201\260\001\000|@\007\356\000\000\200\004\201\300\001\000|@\007\356\000\000\200\006\201\320\001\000|@\007\356\000\000\000\000\201\340\001\000|@\007\356\000\000\000\002\201\360\001\000|@\007\356\000\000\200F\201\000\002\000|@\007\356\000\000\200H\201\020\002\000|@\007\356\000\000\200D\201 \002\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\227\000F\326\000\005\001\002\001\000\205\277\000A\000\364\000\000\000\370\200\000\000\364\020\000\000\370\200\000\210\276\t\000\207\277\b\000\211\276\000\0004\330\227\000\000\000\000\000\306\277\301N\200\276\b\000\212\276\b\000\213\276\b\000\214\276\b\000\215\276\b\000\216\276\b\000\217\276\203\000\0020\001\000\207\277\377\002\0026\370\000\000\000\000\000\307\277\002\201\004\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\004@\005\356\221\000\000\000\001\000\000\000\b\000\020\312\r\000\006\001\n\000\020\312\017\000\b\003\t\002\004~\013\000\020\312\f\000\004\004\016\002\016~4\002\207\277\b\003 ~\b\0030~\b\003@~\b\001\020\312\007\001&(\b\001\020\312\007\001.0\b\001\020\312\007\00168\b\001\020\312\007\001>@\b\001\020\312\007\001FH\b\001\020\312\007\001NP\b\001\020\312\007\001VX\b\001\020\312\007\001^`\b\001\020\312\007\001fh\b\001\020\312\007\001np\b\001\020\312\007\001vx\b\001\020\312\007\001~\200\007\001\020\312\006\001\016\017\005\001\020\312\004\001\f\r\003\001\020\312\002\001\n\013\001\003\022~\007\001\020\312\006\001\026\027\005\001\020\312\004\001\024\025\003\001\020\312\002\001\022\023\001\003\"~\007\001\020\312\006\001\036\037\005\001\020\312\004\001\034\035\003\001\020\312\002\001\032\033\001\001\020\312\006\001&\031\005\001\020\312\004\001$%\003\001\020\312\002\001\"#\001\001\020\312\006\001.!\005\001\020\312\004\001,-\003\001\020\312\002\001*+\001\001\020\312\006\0016)\005\001\020\312\004\00145\003\001\020\312\002\00123\001\001\020\312\006\001>1\005\001\020\312\004\001<=\003\001\020\312\002\001:;\001\001\020\312\006\001F9\005\001\020\312\004\001DE\003\001\020\312\002\001BC\001\001\020\312\006\001NA\005\001\020\312\004\001LM\003\001\020\312\002\001JK\001\001\020\312\006\001VI\005\001\020\312\004\001TU\003\001\020\312\002\001RS\001\001\020\312\006\001^Q\005\001\020\312\004\001\\]\003\001\020\312\002\001Z[\001\001\020\312\006\001fY\005\001\020\312\004\001de\003\001\020\312\002\001bc\001\001\020\312\006\001na\005\001\020\312\004\001lm\003\001\020\312\002\001jk\001\001\020\312\006\001vi\005\001\020\312\004\001tu\003\001\020\312\002\001rs\001\001\020\312\006\001~q\005\001\020\312\004\001|}\003\001\020\312\002\001z{\001\003\362~e\000\242\277\000\000\300\277\200\000\020\312\222\001\224\211\221\001\020\312\222\001\226\225\002\000\207\277\221\001\020\312\211\001\220\223\211\001\020\312\211\001\212\212\211\001\020\312\211\001\214\214\211\001\020\312\211\001\216\216\223\001\207\277\211\001\020\312\212\001\202\201\213\001\020\312\214\001\204\203\023\002\207\277\215\001\020\312\216\001\206\205\217\001\020\312\220\001\210\207\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\241\000\207\277\201@F\314\223+\007\036\201@F\314\223+\007\036\001\000\207\277\201@F\314\223+\007\036\t@F\314\221+'\034\021@F\314\221+G\034\031@F\314\221+g\034!@F\314\221+\207\034)@F\314\221+\247\0341@F\314\221+\307\0349@F\314\221+\347\034A@F\314\221+\007\035I@F\314\221+'\035Q@F\314\221+G\035Y@F\314\221+g\035a@F\314\221+\207\035i@F\314\221+\247\035q@F\314\221+\307\035y@F\314\221+\347\035\201@F\314\223+\007\036\001@F\314\221+\007\034\002\301\002\201\t\000\207\277\002\200\007\277\252\377\242\277\b\000\240\277\200\000\020\312\200\000\206\210\200\000\020\312\200\000\204\206\200\000\020\312\200\000\202\204\200\000\020\312\200\000\200\202\000\000\000\364$\000\000\370\000\000\307\277\377\000\000\213\377\377\000\000\271\000\207\277\211|\376\326u\000\000\004\000\000\330\330\227\000\000\000\211\000,\327\377\022\003\002\214\000\000\000\237\022\0255\221\000\207\277\202\022\023?\211j\000\327\006\022\003\002\001\000\207\277\212| \325\007\024\253\001\035\000\205\277|@\007\356\000\000\200\004\211\000\000\000|@\007\356\000\000\200\006\211\020\000\000|@\007\356\000\000\200\b\211 \000\000|@\007\356\000\000\200\n\2110\000\000|@\007\356\000\000\200\f\211@\000\000|@\007\356\000\000\200\016\211P\000\000|@\007\356\000\000\200\020\211`\000\000|@\007\356\000\000\200\022\211p\000\000|@\007\356\000\000\200\024\211\200\000\000|@\007\356\000\000\200\026\211\220\000\000|@\007\356\000\000\200\030\211\240\000\000|@\007\356\000\000\200\032\211\260\000\000|@\007\356\000\000\200\034\211\300\000\000|@\007\356\000\000\200\036\211\320\000\000|@\007\356\000\000\200 \211\340\000\000|@\007\356\000\000\200\"\211\360\000\000|@\007\356\000\000\200$\211\000\001\000|@\007\356\000\000\200&\211\020\001\000|@\007\356\000\000\200(\211 \001\000|@\007\356\000\000\200*\2110\001\000|@\007\356\000\000\200,\211@\001\000|@\007\356\000\000\200.\211P\001\000|@\007\356\000\000\2000\211`\001\000|@\007\356\000\000\2002\211p\001\000|@\007\356\000\000\2004\211\200\001\000|@\007\356\000\000\2006\211\220\001\000|@\007\356\000\000\2008\211\240\001\000|@\007\356\000\000\200:\211\260\001\000|@\007\356\000\000\200<\211\300\001\000|@\007\356\000\000\200>\211\320\001\000\000\000\306\277\000\r\030~\362\000\020\312\200\000\n\013\377\002\022~\000\000\200\377\004\000\205\277|@\007\356\000\000\200\000\211\340\001\000|@\007\356\000\000\200\002\211\360\001\000|@\007\356\000\000\200@\211\000\002\000|@\007\356\000\000\200B\211\020\002\000|@\007\356\000\000\200\004\211 \002\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\364\020\000\000\370\204\000F\326\000\005\001\002\200\000\204\276\t\000\207\277\004\000\212\276\004\000\213\276\000\0004\330\204\000\000\000\000\000\306\277\301N\200\276\004\000\205\276\004\000\206\276\004\000\207\276\004\000\210\276\004\000\211\276\004\000\020\312\005\000\002\001\n\000\020\312\013\000\b\007\006\000\020\312\007\000\004\003\b\000\020\312\t\000\006\005\003\000\207\277\b\001\020\312\007\001~\200\b\001\020\312\007\001vx\b\001\020\312\007\001np\b\001\020\312\007\001fh\b\001\020\312\007\001^`\b\001\020\312\007\001VX\b\001\020\312\007\001NP\b\001\020\312\007\001FH\b\001\020\312\007\001>@\b\001\020\312\007\00168\b\001\020\312\007\001.0\b\001\020\312\007\001&(\b\001\020\312\007\001\036 \b\001\020\312\007\001\026\030\b\001\020\312\007\001\016\020\000\000\307\277\002\201\004\277\006\001\020\312\005\001|~\004\001\020\312\003\001z|\002\001\020\312\001\001xz\006\001\020\312\005\001tv\004\001\020\312\003\001rt\002\001\020\312\001\001pr\006\001\020\312\005\001ln\004\001\020\312\003\001jl\002\001\020\312\001\001hj\006\001\020\312\005\001df\004\001\020\312\003\001bd\002\001\020\312\001\001`b\006\001\020\312\005\001\\^\004\001\020\312\003\001Z\\\002\001\020\312\001\001XZ\006\001\020\312\005\001TV\004\001\020\312\003\001RT\002\001\020\312\001\001PR\006\001\020\312\005\001LN\004\001\020\312\003\001JL\002\001\020\312\001\001HJ\006\001\020\312\005\001DF\004\001\020\312\003\001BD\002\001\020\312\001\001@B\006\001\020\312\005\001<>\004\001\020\312\003\001:<\002\001\020\312\001\0018:\006\001\020\312\005\00146\004\001\020\312\003\00124\002\001\020\312\001\00102\006\001\020\312\005\001,.\004\001\020\312\003\001*,\002\001\020\312\001\001(*\006\001\020\312\005\001$&\004\001\020\312\003\001\"$\002\001\020\312\001\001 \"\006\001\020\312\005\001\034\036\004\001\020\312\003\001\032\034\002\001\020\312\001\001\030\032\006\001\020\312\005\001\024\026\004\001\020\312\003\001\022\024\002\001\020\312\001\001\020\022\006\001\020\312\005\001\f\016\004\001\020\312\003\001\n\f\002\001\020\312\001\001\b\n\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\277\001\242\277\201\000\0043\377\000\203\276\n\327#<\377\000\020\312\362\000\246\247\000\000\200\377\222\000\207\277\200\000$\312\210\004\203\245\203\004\0139\221\000\207\277\205#\n\177\377\n\345\310\237\000\200\226o\022\203:\221\000\207\277\201#\002\177\377\002\033\021\n\327#<\201\004\r9\202\004\0179\003\002\035[\217\302\365<\003\002\037[\353Q\270=\003\002![\217\302u=\003\002#[=\nW>\003\002%[\353Q8>\204\004\0079\003\002'[\231\231\031>\003\002)[\217\302\365=\206#\002\177\207#\f\177\210\004\017K\207\004\0219\206\004\0239\205\004\0059\203#\006\177\207#\016\177\210#\020\177\211#\022\177\202#\004\177\377\006\307\310\377\002\231\225o\022\203:\224\001\207\277\377\f\307\310\377\020\233\227o\022\203:\377\016\307\310\377\004\235\231o\022\203:\377\0227\021o\022\203:\377\002\307\310\377\f\237\235\n\327#<\377\n\307\310\377\006\241\237\n\327#<\377\004\307\310\377\022\243\241\n\327#<\377\020\307\310\377\016\245\243\n\327#<\377\000\203\276\0202Tv\255\000\240\277\236\377\210\277~\005~\214\247\003S\t\377N;|\000\000\200\377\377RS\021;\252\270?\251KR\177\235\377\210\277\200RI\312\202Q\203\247\004\201\004\201\236\377\210\277\002\004\006\277\246OM\021\245O\005W\201\003N\177#\002\207\277\251|\374\326\203\007\233\006\254j\374\326\246\007\233\006\202\003J\177\223\002\207\277\251UT\177\253\000\023\326\251U\313#\221\000\207\277\253UUW\254UW\021\221\000\207\277\255\000\023\326\251W\263&\255UWW\241\000\207\277\251\000\023\326\251W\263&\235\377\210\277\250\0007\326\251U\257\006\221\000\207\277\246\000'\326\250\007\233\006\246\001\307\310\246\377~\200\246\375\306\310\246\373|~\246\371\306\310\246\367z|\246\365\306\310\246\363xz\246\361\306\310\246\357vx\246\355\306\310\246\353tv\246\351\306\310\246\347rt\246\345\306\310\246\343pr\246\341\306\310\246\337np\246\335\306\310\246\333ln\246\331\306\310\246\327jl\246\325\306\310\246\323hj\246\321\306\310\246\317fh\246\315\306\310\246\313df\246\311\306\310\246\307bd\246\305\306\310\246\303`b\246\301\306\310\246\277^`\246\275\306\310\246\273\\^\246\271\306\310\246\267Z\\\246\265\306\310\246\263XZ\246\261\306\310\246\257VX\246\255\306\310\246\253TV\246\251\306\310\246\247RT\246\245\306\310\246\243PR\246\241\306\310\246\237NP\246\235\306\310\246\233LN\246\231\306\310\246\227JL\246\225\306\310\246\223HJ\246\221\306\310\246\217FH\246\215\306\310\246\213DF\246\211\306\310\246\207BD\246\205\306\310\246\203@B\246\201\306\310\246\177>@\246}\306\310\246{<>\246y\306\310\246w:<\246u\306\310\246s8:\246q\306\310\246o68\246m\306\310\246k46\246i\306\310\246g24\246e\306\310\246c02\246a\306\310\246_.0\246]\306\310\246[,.\246Y\306\310\246W*,\246U\306\310\246S(*\246Q\306\310\246O&(\246M\306\310\246K$&\246I\306\310\246G\"$\246E\306\310\246C \"\246A\306\310\246?\036 \246=\306\310\246;\034\036\2469\306\310\2467\032\034\2465\306\310\2463\030\032\2461\306\310\246/\026\030\246-\306\310\246+\024\026\246)\306\310\246'\022\024\246%\306\310\246#\020\022\246!\306\310\246\037\016\020\246\035\306\310\246\033\f\016\246\031\306\310\246\027\n\f\246\025\306\310\246\023\b\n\246\021\306\310\246\017\006\b\246\r\306\310\246\013\004\006\246\t\306\310\246\007\002\004\246\005\306\310\246\003\000\002\203\003L\177\324\000\242\277\004e\205\276\236\377\210\277\272\001\207\277\005\034\205\310\005\"\207\203\027\267\3218\005\036\205\310\005 \203\201\027\267\3218\005$\205\310\005&\211\207\027\267\3218\377\006\307\310\377\f\207\203\000\000\000>\023\002\207\277\377\004\005\021\000\000\000>\377\002\025\021\000\000\000>#\002\207\277\377\020\307\310\203/\201\210\000\000\000>\005\032\013Y\027\267\3218\005(\207\310\202-\251\211\027\267\3218\224\001\207\277\212+\307\310\377\016\207\214\000\000\000>\377\n\013\021\000\000\000>~\000\205\276\221\000\207\277\2051\027\021\201\000*\326\213\377\005\006\000\000\200\377\2107\027\021\262\000\207\277\201\000*\326\201Q3\006\2063\031\021\377\022\307\310\2075\251\211\000\000\000>\2119S\021\221\000\207\277\201\000*\326\201S/\006\201\000*\326\201Q3\006\221\000\207\277\201\003\026\177\213\000\\\326\213\007\374\003\230\272\334\376\221\000\207\277\201\000*\326\247\003/\006\212\000\023\326\212+\007\206\210\000\023\326\2107\007\206\203\000\023\326\203/\007\206\377\002%|\000\000\200\377\205\000\023\326\2051\007\206\377\024\025\021;\252\270?\377\020\021\021;\252\270?\202\000\023\326\202-\007\206\206\000\023\326\2063\007\206\377\n\013\021;\252\270?\212K\024\177\210K\020\177\211\000\023\326\2119\007\206\207\000\023\326\2075\007\206\377\004\005\021;\252\270?\377\f\r\021;\252\270?\205K\n\177\235\377\210\277\007\000\207\277\250\000\001\325\212\001\251\001\377\006\007\021;\252\270?\252\000\001\325\210\001\251\001\377\016\017\021;\252\270?\202K\004\177\240Q\021\021\203K\006\177\214\000\001\325\205\001\251\001\206K\026\177\242U\025\021\207K\016\177\202\000\001\325\202\001\251\001'\003\207\277\203\000\001\325\203\001\251\001\377\022\023\021;\252\270?\254\000\001\325\213\001\251\001\025\002\207\277\253\000\001\325\207\001\251\001\236\007\r\021\304\001\207\277\211K\022\177\214\007\007\007\235\031\013\021\237\005\017\021\203\005\005\007\223\002\207\277\214\000*\326\205\001\031\006\251\000\001\325\211\001\251\001\223\001\207\277\202Q\005\007\203\000*\326\214\017#\006\223\001\207\277\244Y\307\310\241S\211\214\202S\007\311\243W\213\202\022\001\207\277\203\000*\326\203\023+\006\202U\005\007\222\000\207\277\203\000*\326\203\0273\006\203\003P\177\221\000\207\277\250\000\\\326\250\007\374\003\230\272\334\376\202W\025\311\250Q\251\202\221\000\207\277\202Y\025\311\203S\251\202\202\001\020\312\246\001\202\250\221\001\207\277\250\000\\\326\250\007\374\003\230\272\334\376\200R#}\251\376\245\277\203|\374\326\377\376\245\006\000\000\340C\221\002\207\277\203UT\177\253\000\023\326\203U\313#\241\000\207\277\253UUW\253j\374\326\251\377\245\006\000\000\340C\253UY\021\221\000\207\277\255\000\023\326\203Y\257&\255UYW\241\000\207\277\203\000\023\326\203Y\257&\235\377\210\277\203\0007\326\203U\263\006\221\000\207\277\203\000'\326\203\377\245\006\000\000\340C\377\006\007-\000\000\200\037\213\376\240\277\362\000\020\312\200\000\214\203\377\000\020\312\200\000\212\201\000\000\200\377\200\000\020\312\200\000\210\213\200\000\020\312\200\000\206\211\200\000\020\312\200\000\202\207\200\002\n\177\001\000\205\277\200\000\000\364$\000\000\370\000 \000\364\b\000\000\370\2033\306\310\2035\032\031\2037\306\310\2039\034\033\203;\306\310\203=\036\035\203?\306\310\203A \037\203\363\306\310\203\365zy\203\367\306\310\203\371|{\203\303\306\310\203\305ba\203\307\306\310\203\311dc\203\223\306\310\203\225JI\000\000\307\277\377\002\002\213\377\377\000\000\203\227\306\310\203\231LK\236\377\210\277\215|\376\326u\004\000\004\203c\306\310\203e21\203g\306\310\203i43\203#\306\310\203%\022\021\203'\306\310\203)\024\023\215\000,\327\377\032\003\002\214\000\000\000\203\023\306\310\203\025\n\t\203\027\306\310\203\031\f\013\203\003\306\310\203\005\000\000\203\007\306\310\203\t\002\002\237\032\0355\203\373\306\310\203\375~}\203\377\306\310\203\001\201\177C\002\207\277\202\032\033?\203\313\306\310\203\315fe\203\317\306\310\203\321hg\203\233\306\310\203\235NM\215j\000\327\000\032\003\002\235\377\210\277\216| \325\001\034\253\001\001\000\205\277|@\007\356\000\000\200\f\215\200\001\000|@\007\356\000\000\200\016\215\220\001\000\000\000\330\330\204\000\000\031\203\237\306\310\203\241PO\203k\306\310\203m65\203o\306\310\203q87\203+\306\310\203-\026\025\203/\306\310\2031\030\027\203\033\306\310\203\035\016\r\203\037\306\310\203!\020\017\203\013\306\310\203\r\004\004\203\017\306\310\203\021\006\006\203\343\306\310\203\345rq\203\347\306\310\203\351ts\203\263\306\310\203\265ZY\203\267\306\310\203\271\\[\203\203\306\310\203\205BA\203\207\306\310\203\211DC\203S\306\310\203U*)\203W\306\310\203Y,+\203\353\306\310\203\355vu\203\357\306\310\203\361xw\203\273\306\310\203\275^]\203\277\306\310\203\301`_\203\213\306\310\203\215FE\203\217\306\310\203\221HG\203[\306\310\203].-\203_\306\310\203a0/\203\323\306\310\203\325ji\203\327\306\310\203\331lk\203\243\306\310\203\245RQ\203\247\306\310\203\251TS\203s\306\310\203u:9\203w\306\310\203y<;\203C\306\310\203E\"!\203G\306\310\203I$#\000\000\306\277\031\r\b\177\203\333\306\310\203\335nm\203\337\306\310\203\341po\005\000\205\277|@\007\356\000\000\200<\215\000\000\000|@\007\356\000\000\200>\215\020\000\000|@\007\356\000\000\2008\215 \000\000|@\007\356\000\000\200:\2150\000\000|@\007\356\000\000\2004\215@\000\000|@\007\356\000\000\2006\215P\000\000\203\253\306\310\203\255VU\203\257\306\310\203\261XW\005\000\205\277|@\007\356\000\000\2000\215`\000\000|@\007\356\000\000\2002\215p\000\000|@\007\356\000\000\200,\215\200\000\000|@\007\356\000\000\200.\215\220\000\000|@\007\356\000\000\200(\215\240\000\000|@\007\356\000\000\200*\215\260\000\000\203{\306\310\203}>=\203\177\306\310\203\201@?\005\000\205\277|@\007\356\000\000\200$\215\300\000\000|@\007\356\000\000\200&\215\320\000\000|@\007\356\000\000\200 \215\340\000\000|@\007\356\000\000\200\"\215\360\000\000|@\007\356\000\000\200\034\215\000\001\000|@\007\356\000\000\200\036\215\020\001\000\203K\306\310\203M&%\203O\306\310\203Q('\016\000\205\277|@\007\356\000\000\200\030\215 \001\000|@\007\356\000\000\200\032\2150\001\000|@\007\356\000\000\200\024\215@\001\000|@\007\356\000\000\200\026\215P\001\000|@\007\356\000\000\200\020\215`\001\000|@\007\356\000\000\200\022\215p\001\000|@\007\356\000\000\200\b\215\240\001\000|@\007\356\000\000\200\n\215\260\001\000|@\007\356\000\000\200\004\215\300\001\000|@\007\356\000\000\200\006\215\320\001\000|@\007\356\000\000\000\000\215\340\001\000|@\007\356\000\000\000\002\215\360\001\000|@\007\356\000\000\200B\215\000\002\000|@\007\356\000\000\200D\215\020\002\000|@\007\356\000\000\200@\215 \002\000\000\000\200\277\003\000\266\277\000\000\260\277\000\000\000\000\006\000\000\000\000\000\000\000\030\034\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\004\036\000\000\000\000\000\000\n\000\000\000\000\000\000\000X\001\000\000\000\000\000\000\365\376\377o\000\000\000\0008\035\000\000\000\000\000\000\004\000\000\000\000\000\000\000\234\035\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 23.0.0 (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000)\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\261\001\000\000\000\002\t\000\300\211\000\000\000\000\000\000\000\000\000\000\000\000\000\000Z\000\000\000\022\003\b\000\0002\000\000\000\000\000\000\330\021\000\000\000\000\000\000x\000\000\000\021\003\006\000\200\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\231\000\000\000\022\003\b\000\000D\000\000\000\000\000\000\324\021\000\000\000\000\000\000\267\000\000\000\021\003\006\000\300\037\000\000\000\000\000\000@\000\000\000\000\000\000\000\330\000\000\000\022\003\b\000\000V\000\000\000\000\000\000\270\016\000\000\000\000\000\000\366\000\000\000\021\003\006\000\000 \000\000\000\000\000\000@\000\000\000\000\000\000\000\027\001\000\000\022\003\b\000\000e\000\000\000\000\000\000d\006\000\000\000\000\000\0005\001\000\000\021\003\006\000@ \000\000\000\000\000\000@\000\000\000\000\000\000\000V\001\000\000\022\003\b\000\000l\000\000\000\000\000\000\274\r\000\000\000\000\000\000t\001\000\000\021\003\006\000\200 \000\000\000\000\000\000@\000\000\000\000\000\000\000\225\001\000\000\021\000\013\0000\232\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.eh_frame\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000amdgpu.max_num_named_barrier\000_Z12floor_kernelILi0EEvPKjPfi\000_Z12floor_kernelILi0EEvPKjPfi.kd\000_Z12floor_kernelILi1EEvPKjPfi\000_Z12floor_kernelILi1EEvPKjPfi.kd\000_Z12floor_kernelILi2EEvPKjPfi\000_Z12floor_kernelILi2EEvPKjPfi.kd\000_Z12floor_kernelILi3EEvPKjPfi\000_Z12floor_kernelILi3EEvPKjPfi.kd\000_Z12floor_kernelILi4EEvPKjPfi\000_Z12floor_kernelILi4EEvPKjPfi.kd\000__hip_cuid_2edb253f577969c9\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\340\031\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\030\034\000\000\000\000\000\000\030\034\000\000\000\000\000\000 \001\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\0008\035\000\000\000\000\000\0008\035\000\000\000\000\000\000d\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000\234\035\000\000\000\000\000\000\234\035\000\000\000\000\000\000h\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\004\036\000\000\000\000\000\000\004\036\000\000\000\000\000\000X\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\200\037\000\000\000\000\000\000\200\037\000\000\000\000\000\000@\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\300 \000\000\000\000\000\000\300 \000\000\000\000\000\000\244\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0009\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\0002\000\000\000\000\000\000\000\"\000\000\000\000\000\000\274G\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000?\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\300\211\000\000\000\000\000\000\300i\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000H\000\000\000\b\000\000\000\003\000\000\000\000\000\000\0000\212\000\000\000\000\000\0000j\000\000\000\000\000\000\320\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000W\000\000\000\b\000\000\000\003\000\000\000\000\000\000\0000\232\000\000\000\000\000\0000j\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\\\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0000j\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000k\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0000j\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0000j\000\000\000\000\000\000\334\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\211\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020k\000\000\000\000\000\000\230\001\000\000\000\000\000\000\021\000\000\000\006\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\250l\000\000\000\000\000\000\243\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\233\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Km\000\000\000\000\000\000\272\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_6, 33672

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_6
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_2edb253f577969c9,@object # @__hip_gpubin_handle_2edb253f577969c9
	.local	__hip_gpubin_handle_2edb253f577969c9
	.comm	__hip_gpubin_handle_2edb253f577969c9,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_2edb253f577969c9,@object # @__hip_cuid_2edb253f577969c9
	.bss
	.globl	__hip_cuid_2edb253f577969c9
__hip_cuid_2edb253f577969c9:
	.byte	0                               # 0x0
	.size	__hip_cuid_2edb253f577969c9, 1

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
	.addrsig_sym _Z27__device_stub__floor_kernelILi0EEvPKjPfi
	.addrsig_sym _Z27__device_stub__floor_kernelILi1EEvPKjPfi
	.addrsig_sym _Z27__device_stub__floor_kernelILi2EEvPKjPfi
	.addrsig_sym _Z27__device_stub__floor_kernelILi3EEvPKjPfi
	.addrsig_sym _Z27__device_stub__floor_kernelILi4EEvPKjPfi
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _Z12floor_kernelILi0EEvPKjPfi
	.addrsig_sym _Z12floor_kernelILi1EEvPKjPfi
	.addrsig_sym _Z12floor_kernelILi2EEvPKjPfi
	.addrsig_sym _Z12floor_kernelILi3EEvPKjPfi
	.addrsig_sym _Z12floor_kernelILi4EEvPKjPfi
	.addrsig_sym .L__unnamed_6
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_2edb253f577969c9
