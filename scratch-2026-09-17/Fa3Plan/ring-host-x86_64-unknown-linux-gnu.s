	.att_syntax
	.file	"ring.hip"
	.text
	.globl	_Z19__device_stub__ringPKjPji   # -- Begin function _Z19__device_stub__ringPKjPji
	.prefalign	4, .Lfunc_end0, nop
	.type	_Z19__device_stub__ringPKjPji,@function
_Z19__device_stub__ringPKjPji:          # @_Z19__device_stub__ringPKjPji
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
	movq	_Z4ringPKjPji@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end0:
	.size	_Z19__device_stub__ringPKjPji, .Lfunc_end0-_Z19__device_stub__ringPKjPji
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI1_0:
	.long	1013904223                      # 0x3c6ef35f
	.long	1015568748                      # 0x3c88596c
	.long	1017233273                      # 0x3ca1bf79
	.long	1018897798                      # 0x3cbb2586
.LCPI1_1:
	.long	1020562323                      # 0x3cd48b93
	.long	1022226848                      # 0x3cedf1a0
	.long	1023891373                      # 0x3d0757ad
	.long	1025555898                      # 0x3d20bdba
.LCPI1_2:
	.long	1027220423                      # 0x3d3a23c7
	.long	1028884948                      # 0x3d5389d4
	.long	1030549473                      # 0x3d6cefe1
	.long	1032213998                      # 0x3d8655ee
.LCPI1_3:
	.long	1033878523                      # 0x3d9fbbfb
	.long	1035543048                      # 0x3db92208
	.long	1037207573                      # 0x3dd28815
	.long	1038872098                      # 0x3debee22
.LCPI1_4:
	.long	1040536623                      # 0x3e05542f
	.long	1042201148                      # 0x3e1eba3c
	.long	1043865673                      # 0x3e382049
	.long	1045530198                      # 0x3e518656
.LCPI1_5:
	.long	1047194723                      # 0x3e6aec63
	.long	1048859248                      # 0x3e845270
	.long	1050523773                      # 0x3e9db87d
	.long	1052188298                      # 0x3eb71e8a
.LCPI1_6:
	.long	1053852823                      # 0x3ed08497
	.long	1055517348                      # 0x3ee9eaa4
	.long	1057181873                      # 0x3f0350b1
	.long	1058846398                      # 0x3f1cb6be
.LCPI1_7:
	.long	1060510923                      # 0x3f361ccb
	.long	1062175448                      # 0x3f4f82d8
	.long	1063839973                      # 0x3f68e8e5
	.long	1065504498                      # 0x3f824ef2
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
	subq	$120, %rsp
	.cfi_def_cfa_offset 176
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	xorl	%ecx, %ecx
	xorl	%r13d, %r13d
	.p2align	4
.LBB1_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_23 Depth 2
                                        #     Child Loop BB1_57 Depth 2
	leaq	.Lconstinit(%rip), %rax
	movslq	(%r13,%rax), %rbp
	testq	%rbp, %rbp
	js	.LBB1_86
# %bb.2:                                # %_ZNSt6vectorIjSaIjEE17_S_check_init_lenEmRKS0_.exit.i
                                        #   in Loop: Header=BB1_1 Depth=1
	movl	%ecx, 24(%rsp)                  # 4-byte Spill
	testl	%ebp, %ebp
	je	.LBB1_3
# %bb.4:                                # %.noexc79
                                        #   in Loop: Header=BB1_1 Depth=1
	movq	%rbp, %rbx
	shlq	$12, %rbx
	leaq	(,%rbx,4), %rdi
	.cfi_escape 0x2e, 0x00
	callq	_Znwm@PLT
	movq	%rax, %r14
	leaq	(%rax,%rbx,4), %r15
	movl	$0, (%rax)
	leaq	4(%rax), %rdi
	leaq	-4(,%rbx,4), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r15, (%rsp)                    # 8-byte Spill
	movq	%r15, %rbx
	jmp	.LBB1_5
	.p2align	4
.LBB1_3:                                #   in Loop: Header=BB1_1 Depth=1
	movq	$0, (%rsp)                      # 8-byte Folded Spill
	xorl	%r14d, %r14d
	xorl	%ebx, %ebx
.LBB1_5:                                # %_ZNSt6vectorIjSaIjEEC2EmRKS0_.exit
                                        #   in Loop: Header=BB1_1 Depth=1
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$49152, %edi                    # imm = 0xC000
	callq	_Znwm@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.6:                                #   in Loop: Header=BB1_1 Depth=1
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$49152, %edx                    # imm = 0xC000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
.Ltmp3:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$128, %edi
	callq	_Znwm@PLT
.Ltmp4:                                 # EH_LABEL
# %bb.7:                                #   in Loop: Header=BB1_1 Depth=1
	movq	%rax, %r12
	movq	%rbx, %rax
	subq	%r14, %rax
	pxor	%xmm0, %xmm0
	movdqu	%xmm0, 112(%r12)
	movdqu	%xmm0, 96(%r12)
	movdqu	%xmm0, 80(%r12)
	movdqu	%xmm0, 64(%r12)
	movdqu	%xmm0, 48(%r12)
	movdqu	%xmm0, 32(%r12)
	movdqu	%xmm0, 16(%r12)
	movdqu	%xmm0, (%r12)
	subq	%r14, %rbx
	je	.LBB1_24
# %bb.8:                                # %.lr.ph.preheader
                                        #   in Loop: Header=BB1_1 Depth=1
	sarq	$2, %rax
	leaq	-33(%rax), %rcx
	cmpq	$-29, %rcx
	jae	.LBB1_10
# %bb.9:                                #   in Loop: Header=BB1_1 Depth=1
	xorl	%ecx, %ecx
	jmp	.LBB1_19
	.p2align	4
.LBB1_10:                               # %vector.ph
                                        #   in Loop: Header=BB1_1 Depth=1
	movl	%eax, %ecx
	andl	$60, %ecx
	movdqa	.LCPI1_0(%rip), %xmm1           # xmm1 = [1013904223,1015568748,1017233273,1018897798]
	movdqu	%xmm1, (%r14)
	movdqu	(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, (%r12)
	cmpq	$4, %rcx
	je	.LBB1_18
# %bb.11:                               # %vector.body.1
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_1(%rip), %xmm1           # xmm1 = [1020562323,1022226848,1023891373,1025555898]
	movdqu	%xmm1, 16(%r14)
	movdqu	16(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 16(%r12)
	cmpl	$8, %ecx
	je	.LBB1_18
# %bb.12:                               # %vector.body.2
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_2(%rip), %xmm1           # xmm1 = [1027220423,1028884948,1030549473,1032213998]
	movdqu	%xmm1, 32(%r14)
	movdqu	32(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 32(%r12)
	cmpl	$12, %ecx
	je	.LBB1_18
# %bb.13:                               # %vector.body.3
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_3(%rip), %xmm1           # xmm1 = [1033878523,1035543048,1037207573,1038872098]
	movdqu	%xmm1, 48(%r14)
	movdqu	48(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 48(%r12)
	cmpl	$16, %ecx
	je	.LBB1_18
# %bb.14:                               # %vector.body.4
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_4(%rip), %xmm1           # xmm1 = [1040536623,1042201148,1043865673,1045530198]
	movdqu	%xmm1, 64(%r14)
	movdqu	64(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 64(%r12)
	cmpl	$20, %ecx
	je	.LBB1_18
# %bb.15:                               # %vector.body.5
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_5(%rip), %xmm1           # xmm1 = [1047194723,1048859248,1050523773,1052188298]
	movdqu	%xmm1, 80(%r14)
	movdqu	80(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 80(%r12)
	cmpl	$24, %ecx
	je	.LBB1_18
# %bb.16:                               # %vector.body.6
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_6(%rip), %xmm1           # xmm1 = [1053852823,1055517348,1057181873,1058846398]
	movdqu	%xmm1, 96(%r14)
	movdqu	96(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 96(%r12)
	cmpl	$28, %ecx
	je	.LBB1_18
# %bb.17:                               # %vector.body.7
                                        #   in Loop: Header=BB1_1 Depth=1
	movdqa	.LCPI1_7(%rip), %xmm1           # xmm1 = [1060510923,1062175448,1063839973,1065504498]
	movdqu	%xmm1, 112(%r14)
	movdqu	112(%r12), %xmm0
	paddd	%xmm1, %xmm0
	movdqu	%xmm0, 112(%r12)
	.p2align	4
.LBB1_18:                               # %middle.block
                                        #   in Loop: Header=BB1_1 Depth=1
	cmpq	%rcx, %rax
	je	.LBB1_24
.LBB1_19:                               # %.lr.ph.preheader710
                                        #   in Loop: Header=BB1_1 Depth=1
	leaq	1(%rcx), %rdx
	testb	$4, %bl
	je	.LBB1_21
# %bb.20:                               # %.lr.ph.prol
                                        #   in Loop: Header=BB1_1 Depth=1
	imull	$1664525, %ecx, %esi            # imm = 0x19660D
	addl	$1013904223, %esi               # imm = 0x3C6EF35F
	movl	%esi, (%r14,%rcx,4)
	andl	$28, %ecx
	addl	%esi, (%r12,%rcx,4)
	movq	%rdx, %rcx
.LBB1_21:                               # %.lr.ph.prol.loopexit
                                        #   in Loop: Header=BB1_1 Depth=1
	cmpq	%rdx, %rax
	je	.LBB1_24
# %bb.22:                               # %.lr.ph.preheader971
                                        #   in Loop: Header=BB1_1 Depth=1
	imulq	$1664525, %rcx, %rdx            # imm = 0x19660D
	addq	$1015568748, %rdx               # imm = 0x3C88596C
	.p2align	4
.LBB1_23:                               # %.lr.ph
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leal	-1664525(%rdx), %esi
	movq	%rcx, %rdi
	shrq	$12, %rdi
	imull	$97, %edi, %edi
	xorl	%esi, %edi
	movl	%edi, (%r14,%rcx,4)
	movl	%ecx, %esi
	andl	$31, %esi
	addl	%edi, (%r12,%rsi,4)
	leaq	1(%rcx), %rsi
	movq	%rsi, %rdi
	shrq	$12, %rdi
	imull	$97, %edi, %edi
	xorl	%edx, %edi
	movl	%edi, 4(%r14,%rcx,4)
	andl	$31, %esi
	addl	%edi, (%r12,%rsi,4)
	addq	$2, %rcx
	addq	$3329050, %rdx                  # imm = 0x32CC1A
	cmpq	%rcx, %rax
	jne	.LBB1_23
	.p2align	4
.LBB1_24:                               # %._crit_edge
                                        #   in Loop: Header=BB1_1 Depth=1
.Ltmp6:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movq	%rbx, %rsi
	callq	hipMalloc@PLT
.Ltmp7:                                 # EH_LABEL
# %bb.25:                               # %_ZL9hipMallocIjE10hipError_tPPT_m.exit
                                        #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_26
# %bb.32:                               #   in Loop: Header=BB1_1 Depth=1
.Ltmp12:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$49152, %esi                    # imm = 0xC000
	leaq	8(%rsp), %rdi
	callq	hipMalloc@PLT
.Ltmp13:                                # EH_LABEL
# %bb.33:                               # %_ZL9hipMallocIjE10hipError_tPPT_m.exit90
                                        #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_34
# %bb.38:                               #   in Loop: Header=BB1_1 Depth=1
	movq	16(%rsp), %rdi
.Ltmp18:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rsi
	movq	%rbx, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp19:                                # EH_LABEL
# %bb.39:                               #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_40
# %bb.44:                               #   in Loop: Header=BB1_1 Depth=1
.Ltmp24:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$41216, %r8d                    # imm = 0xA100
	movabsq	$4294967360, %rdi               # imm = 0x100000040
	movl	$1, %esi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %ecx
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp25:                                # EH_LABEL
# %bb.45:                               #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_48
# %bb.46:                               #   in Loop: Header=BB1_1 Depth=1
	movq	16(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	%ebp, 28(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	28(%rsp), %rax
	movq	%rax, 112(%rsp)
.Ltmp26:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp27:                                # EH_LABEL
# %bb.47:                               # %.noexc91
                                        #   in Loop: Header=BB1_1 Depth=1
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
.Ltmp28:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z4ringPKjPji@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp29:                                # EH_LABEL
.LBB1_48:                               #   in Loop: Header=BB1_1 Depth=1
.Ltmp31:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError@PLT
.Ltmp32:                                # EH_LABEL
# %bb.49:                               #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_50
# %bb.54:                               #   in Loop: Header=BB1_1 Depth=1
	movq	8(%rsp), %rsi
.Ltmp37:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$49152, %edx                    # imm = 0xC000
	movq	%r15, %rdi
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp38:                                # EH_LABEL
# %bb.55:                               #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_63
# %bb.56:                               # %.preheader.preheader
                                        #   in Loop: Header=BB1_1 Depth=1
	xorl	%ebx, %ebx
	movl	$2, %eax
	.p2align	4
.LBB1_57:                               # %.preheader
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leal	-2(%rax), %ecx
	andl	$31, %ecx
	movl	-8(%r15,%rax,4), %edx
	movl	-4(%r15,%rax,4), %esi
	xorl	%edi, %edi
	cmpl	(%r12,%rcx,4), %edx
	setne	%dil
	movl	%ebx, %ecx
	addl	%edi, %ecx
	leal	-1(%rax), %edx
	andl	$31, %edx
	xorl	%edi, %edi
	cmpl	(%r12,%rdx,4), %esi
	setne	%dil
	movl	(%r15,%rax,4), %edx
	movl	%eax, %esi
	andl	$31, %esi
	xorl	%ebx, %ebx
	cmpl	(%r12,%rsi,4), %edx
	setne	%bl
	addl	%edi, %ebx
	addl	%ecx, %ebx
	addq	$3, %rax
	cmpq	$12290, %rax                    # imm = 0x3002
	jne	.LBB1_57
# %bb.58:                               #   in Loop: Header=BB1_1 Depth=1
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.6(%rip), %rdi
	movl	%ebp, %esi
	movl	%ebx, %edx
	xorl	%eax, %eax
	callq	printf@PLT
	movl	$1, %ebp
	testl	%ebx, %ebx
	jne	.LBB1_75
# %bb.59:                               #   in Loop: Header=BB1_1 Depth=1
	movq	16(%rsp), %rdi
.Ltmp43:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp44:                                # EH_LABEL
# %bb.60:                               #   in Loop: Header=BB1_1 Depth=1
	testl	%eax, %eax
	jne	.LBB1_61
# %bb.69:                               #   in Loop: Header=BB1_1 Depth=1
	movq	8(%rsp), %rdi
.Ltmp49:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp50:                                # EH_LABEL
# %bb.70:                               #   in Loop: Header=BB1_1 Depth=1
	movl	24(%rsp), %ebp                  # 4-byte Reload
	testl	%eax, %eax
	jne	.LBB1_71
.LBB1_75:                               # %_ZNSt6vectorIjSaIjEED2Ev.exit94
                                        #   in Loop: Header=BB1_1 Depth=1
	.cfi_escape 0x2e, 0x00
	movl	$128, %esi
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$49152, %esi                    # imm = 0xC000
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	testq	%r14, %r14
	je	.LBB1_77
# %bb.76:                               #   in Loop: Header=BB1_1 Depth=1
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
.LBB1_77:                               # %_ZNSt6vectorIjSaIjEED2Ev.exit96
                                        #   in Loop: Header=BB1_1 Depth=1
	testl	%ebx, %ebx
	jne	.LBB1_79
# %bb.78:                               # %_ZNSt6vectorIjSaIjEED2Ev.exit96
                                        #   in Loop: Header=BB1_1 Depth=1
	addq	$4, %r13
	movl	%ebp, %ecx
	cmpq	$24, %r13
	jne	.LBB1_1
.LBB1_79:
	movl	%ebp, %eax
	addq	$120, %rsp
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
.LBB1_40:
	.cfi_def_cfa_offset 176
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp21:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp22:                                # EH_LABEL
# %bb.41:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.3(%rip), %rdx
	jmp	.LBB1_28
.LBB1_34:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp15:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp16:                                # EH_LABEL
# %bb.35:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.2(%rip), %rdx
	jmp	.LBB1_28
.LBB1_26:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp9:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp10:                                # EH_LABEL
# %bb.27:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.1(%rip), %rdx
	jmp	.LBB1_28
.LBB1_86:                               # %.noexc
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.9(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB1_63:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp40:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp41:                                # EH_LABEL
# %bb.64:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.5(%rip), %rdx
	jmp	.LBB1_28
.LBB1_50:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp34:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp35:                                # EH_LABEL
# %bb.51:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.4(%rip), %rdx
	jmp	.LBB1_28
.LBB1_61:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp46:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp47:                                # EH_LABEL
# %bb.62:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.7(%rip), %rdx
	jmp	.LBB1_28
.LBB1_71:
	movq	stderr@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rbx
.Ltmp52:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%eax, %edi
	callq	hipGetErrorString@PLT
.Ltmp53:                                # EH_LABEL
# %bb.72:
	.cfi_escape 0x2e, 0x00
	leaq	.L.str(%rip), %rsi
	leaq	.L.str.8(%rip), %rdx
.LBB1_28:
	movq	%rbx, %rdi
	movq	%rax, %rcx
	xorl	%eax, %eax
	callq	fprintf@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2, %edi
	callq	exit@PLT
.LBB1_74:                               # %.loopexit.split-lp163
.Ltmp54:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_68:                               # %.loopexit.split-lp158
.Ltmp48:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_73:                               # %.loopexit162
.Ltmp51:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_67:                               # %.loopexit157
.Ltmp45:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_53:                               # %.loopexit.split-lp148
.Ltmp36:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_66:                               # %.loopexit.split-lp153
.Ltmp42:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_31:                               # %.loopexit.split-lp
.Ltmp11:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_37:                               # %.loopexit.split-lp138
.Ltmp17:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_43:                               # %.loopexit.split-lp143
.Ltmp23:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_52:                               # %.loopexit147
.Ltmp33:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_65:                               # %.loopexit152
.Ltmp39:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_29:
.Ltmp2:                                 # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB1_83
.LBB1_30:
.Ltmp5:                                 # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB1_82
.LBB1_80:                               # %.loopexit
.Ltmp8:                                 # EH_LABEL
	jmp	.LBB1_81
.LBB1_36:                               # %.loopexit137
.Ltmp14:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_42:                               # %.loopexit142
.Ltmp20:                                # EH_LABEL
	jmp	.LBB1_81
.LBB1_87:
.Ltmp30:                                # EH_LABEL
.LBB1_81:                               # %_ZNSt6vectorIjSaIjEED2Ev.exit98
	movq	%rax, %rbx
	.cfi_escape 0x2e, 0x00
	movl	$128, %esi
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
.LBB1_82:                               # %_ZNSt6vectorIjSaIjEED2Ev.exit100
	.cfi_escape 0x2e, 0x00
	movl	$49152, %esi                    # imm = 0xC000
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
.LBB1_83:
	testq	%r14, %r14
	je	.LBB1_85
# %bb.84:
	movq	(%rsp), %rsi                    # 8-byte Reload
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
.LBB1_85:                               # %_ZNSt6vectorIjSaIjEED2Ev.exit102
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
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
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp3-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin0           # >> Call Site 5 <<
	.uleb128 .Ltmp7-.Ltmp6                  #   Call between .Ltmp6 and .Ltmp7
	.uleb128 .Ltmp8-.Lfunc_begin0           #     jumps to .Ltmp8
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp13-.Ltmp12                #   Call between .Ltmp12 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin0          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp18-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp19-.Ltmp18                #   Call between .Ltmp18 and .Ltmp19
	.uleb128 .Ltmp20-.Lfunc_begin0          #     jumps to .Ltmp20
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp24-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp29-.Ltmp24                #   Call between .Ltmp24 and .Ltmp29
	.uleb128 .Ltmp30-.Lfunc_begin0          #     jumps to .Ltmp30
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp31-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp32-.Ltmp31                #   Call between .Ltmp31 and .Ltmp32
	.uleb128 .Ltmp33-.Lfunc_begin0          #     jumps to .Ltmp33
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp37-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp38-.Ltmp37                #   Call between .Ltmp37 and .Ltmp38
	.uleb128 .Ltmp39-.Lfunc_begin0          #     jumps to .Ltmp39
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp43-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp44-.Ltmp43                #   Call between .Ltmp43 and .Ltmp44
	.uleb128 .Ltmp45-.Lfunc_begin0          #     jumps to .Ltmp45
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp49-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp50-.Ltmp49                #   Call between .Ltmp49 and .Ltmp50
	.uleb128 .Ltmp51-.Lfunc_begin0          #     jumps to .Ltmp51
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp21-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp22-.Ltmp21                #   Call between .Ltmp21 and .Ltmp22
	.uleb128 .Ltmp23-.Lfunc_begin0          #     jumps to .Ltmp23
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp16-.Ltmp15                #   Call between .Ltmp15 and .Ltmp16
	.uleb128 .Ltmp17-.Lfunc_begin0          #     jumps to .Ltmp17
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin0           # >> Call Site 15 <<
	.uleb128 .Ltmp10-.Ltmp9                 #   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp11-.Lfunc_begin0          #     jumps to .Ltmp11
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp10-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Ltmp40-.Ltmp10                #   Call between .Ltmp10 and .Ltmp40
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp40-.Lfunc_begin0          # >> Call Site 17 <<
	.uleb128 .Ltmp41-.Ltmp40                #   Call between .Ltmp40 and .Ltmp41
	.uleb128 .Ltmp42-.Lfunc_begin0          #     jumps to .Ltmp42
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp34-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp35-.Ltmp34                #   Call between .Ltmp34 and .Ltmp35
	.uleb128 .Ltmp36-.Lfunc_begin0          #     jumps to .Ltmp36
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp46-.Lfunc_begin0          # >> Call Site 19 <<
	.uleb128 .Ltmp47-.Ltmp46                #   Call between .Ltmp46 and .Ltmp47
	.uleb128 .Ltmp48-.Lfunc_begin0          #     jumps to .Ltmp48
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp52-.Lfunc_begin0          # >> Call Site 20 <<
	.uleb128 .Ltmp53-.Ltmp52                #   Call between .Ltmp52 and .Ltmp53
	.uleb128 .Ltmp54-.Lfunc_begin0          #     jumps to .Ltmp54
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp53-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Lfunc_end1-.Ltmp53            #   Call between .Ltmp53 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end2, nop     # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_f70806c800e446c8(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB2_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_f70806c800e446c8(%rip)
.LBB2_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z4ringPKjPji@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end2:
	.size	__hip_module_ctor, .Lfunc_end2-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end3, nop     # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_f70806c800e446c8(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_f70806c800e446c8(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z4ringPKjPji,@object           # @_Z4ringPKjPji
	.section	.data.rel.ro,"aw",@progbits
	.globl	_Z4ringPKjPji
	.p2align	3, 0x0
_Z4ringPKjPji:
	.quad	_Z19__device_stub__ringPKjPji
	.size	_Z4ringPKjPji, 8

	.type	.Lconstinit,@object             # @constinit
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.Lconstinit:
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.long	7                               # 0x7
	.long	65                              # 0x41
	.long	513                             # 0x201
	.size	.Lconstinit, 24

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%s %s\n"
	.size	.L.str, 7

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"hipMalloc(&s,input.size()*4)"
	.size	.L.str.1, 29

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"hipMalloc(&d,output.size()*4)"
	.size	.L.str.2, 30

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"hipMemcpy(s,input.data(),input.size()*4,hipMemcpyHostToDevice)"
	.size	.L.str.3, 63

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"hipGetLastError()"
	.size	.L.str.4, 18

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"hipMemcpy(output.data(),d,output.size()*4,hipMemcpyDeviceToHost)"
	.size	.L.str.5, 65

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"tiles=%d blocks=64 consumers=6 producers=2 LDS=41216 mismatches=%d\n"
	.size	.L.str.6, 68

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"hipFree(s)"
	.size	.L.str.7, 11

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"hipFree(d)"
	.size	.L.str.8, 11

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.9, 49

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z4ringPKjPji"
	.size	.L__unnamed_1, 14

	.type	.L__unnamed_2,@object           # @1
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_2:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\000\360\026\000\000\000\000\000\000 \000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx1201\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000p\022\000\000\000\000\000\000N\000\000\000@\0008\000\t\000@\000\022\000\020\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000t\006\000\000\000\000\000\000t\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\007\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\000\017\000\000\000\000\000\000\000/\000\000\000\000\000\000\000/\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000p\017\000\000\000\000\000\000p?\000\000\000\000\000\000p?\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\000\017\000\000\000\000\000\000\000/\000\000\000\000\000\000\000/\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\000\017\000\000\000\000\000\000\000/\000\000\000\000\000\000\000/\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\234\002\000\000\000\000\000\000\234\002\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000\206\002\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\336\000\023\245.args\223\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\203\247.offset\020\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\024\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\001\000\245.name\255_Z4ringPKjPji\273.private_segment_fixed_size\000\253.sgpr_count\017\261.sgpr_spill_count\000\247.symbol\260_Z4ringPKjPji.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\021\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\272amdgcn-amd-amdhsa--gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\b\000\000\027\000\000\000\000\000\000X\006\000\000\000\000\000\000\017\000\000\000\021\003\006\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000 \000\000\000\021\000\013\000p?\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\200\000\020\000\000\f\000\200\001\000\000\000j\310\177\376\006\272\005P+\316X\034\004\000\000\000\004\000\000\000\003\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\000_Z4ringPKjPji\000_Z4ringPKjPji.kd\000__hip_cuid_f70806c800e446c8\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320\000\000\000\002\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\020\000\000\000\000\000\000\000\001zR\000\004\004\020\001\033\000\000\000\030\000\000\000\030\000\000\000\244\020\000\000X\006\000\000\000\017\00406\351\002\007\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000~\000\202\276\220\000\230}\001\000F\326\000\005\001\002\200\002\004~\000\2404\330\001\002\000\000~\002~\214\001\000\205\277\300\000\000\364\020\000\000\370\000A\000\364\000\000\000\370\000\000\306\277\301N\200\276\001\000\020\326\000\013\t\002\207\000\0042\237\000\b6\223\001\207\277\203\002\232|\005\000\013\326\002\007\005\004\377\377\224\277\000\000\307\277\003\201\004\277|\300\n\356\000\000\004\000\000\000\000\000\\\001\242\277\200\000\"\312\202\b\006\001\007\000F\326\005\005\001\002\b\000F\326\002\005\001\002\000\000J\324\200\b\002\002D\002\207\277\013\000V\326\002\033\031\004\377\b\0228\340\377\377\377\000\000V\326\002\027\021\004\200\000 \312\200\f\n\003\001\001 \312\200\026\n\006\200\000\210\276\200\000\211\276\016\000\240\277\236\377\210\277~\001~\214\t\000\207\277~\n~\214\001\000J\324\003\030\002\002\f\001 \312\377\000\000\003\000\020\000\000\t\301\t\215\001\b\b\214\236\377\210\277~\b~\2216\001\245\277\002\000\001\325\200\002%\000\201\006\0326\201\006\030J\003\000\207\277\216\004\0040j \201\276\236\377\210\277~\001\002\215U\000\245\277\203\032\0340\201\006\030J\002\000\207\277\200\034\006J\000 \212\276\030\000\245\277\000\240\330\330\003\000\000\016\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\016\031\002\002~\001~\213\016\000\245\277\200\000\213\276\001\000\203\277\000\240\330\330\003\000\000\016\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\016\031\002\002\001\013\013\214\t\000\207\277~\013~\221\363\377\246\277~\n~\214\000 \212\276\030\000\245\277\004\240\330\330\003\000\000\016\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\016\031\002\002~\001~\213\016\000\245\277\200\000\213\276\001\000\203\277\004\240\330\330\003\000\000\016\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\016\031\002\002\001\013\013\214\t\000\207\277~\013~\221\363\377\246\277~\n~\214\t\001 \312\n\005\002\003\200\000\212\276\000\000\330\330\002\000\000\016\240\006\006J\377\004\004J\200\000\000\000\002\000\207\277\001\000I\324\377\006\002\002\337\017\000\000\001\n\n\214\000\000\306\277\016\r\fJ~\n~\221\362\377\246\277~\n~\214\000 \201\276\005\000\245\277\002\000\013\326\r1\035\004\000\000\300\277\020\2404\330\002\f\000\000\236\377\210\277~\001~\214\236\377\210\277\0020\212\276\222\377\245\277~\000\202\276\201\006\222}\246\000\245\277\230\032\034\026\301\006\006J\002\000\207\277\200\034\034J\000 \213\276\030\000\245\277\020\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\017\007\002\002~\001~\213\016\000\245\277\200\000\214\276\001\000\203\277\020\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\017\007\002\002\001\f\f\214\t\000\207\277~\f~\221\363\377\246\277~\013~\214\000 \213\276\030\000\245\277\024\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\017\007\002\002~\001~\213\016\000\245\277\200\000\214\276\001\000\203\277\024\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\017\007\002\002\001\f\f\214\t\000\207\277~\f~\221\363\377\246\277~\013~\214\000 \213\276\030\000\245\277\030\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\017\007\002\002~\001~\213\016\000\245\277\200\000\214\276\001\000\203\277\030\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\017\007\002\002\001\f\f\214\t\000\207\277~\f~\221\363\377\246\277~\013~\214\000 \213\276\030\000\245\277\034\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\017\007\002\002~\001~\213\016\000\245\277\200\000\214\276\001\000\203\277\034\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\017\007\002\002\001\f\f\214\t\000\207\277~\f~\221\363\377\246\277~\013~\214\000 \213\276\030\000\245\277 \240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\017\007\002\002~\001~\213\016\000\245\277\200\000\214\276\001\000\203\277 \240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\017\007\002\002\001\f\f\214\t\000\207\277~\f~\221\363\377\246\277~\013~\214\t\000\207\277~\000~\213\030\000\245\277$\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000M\324\017\007\002\002~\001~\213\016\000\245\277\200\000\213\276\001\000\203\277$\240\330\330\016\000\000\017\000\000\306\277|\300\n\356\000\000\004\000\000\000\000\000\001\000J\324\017\007\002\002\001\013\013\214\t\000\207\277~\013~\221\363\377\246\277\236\377\210\277~\002~\214\202\000\036>\013\005\034J\200\000\213\276\242\001\207\277\002\001\000\327\004\036\002\002\237\361\210\277\003| \325\005 \006\000\t\003\036~|\000\005\356\020\000\000\000\002\000\000\000\240\036\036J\002\001\000\327\377\004\002\002\200\000\000\000\237\361\210\277\003| \325\200\006\006\000\003\000\207\277\002\000I\324\377\036\002\002\337\007\000\000\002\013\013\214\000\000\300\277\000\0004\330\016\020\000\000\377\034\034J\200\000\000\000~\013~\221\352\377\246\277~\013~\214\000 \201\276\302\376\245\277\002\000F\326\r\007!\004\000\000\306\277\000\2404\330\002\f\000\000\274\376\240\277~\b~\214\001\000\240\277\200\002\f~\000\000\310\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000j \200\276\022\000\245\277\236\377\210\277u\377\000\226\300\000\000\000\200\002\002~\236\377\210\277\000\000F\326\005\013\001\000\221\000\207\277\000\t\0008\202\000\000>\221\000\207\277\000j\000\327\006\000\002\002\001| \325\007\002\252\001|\200\006\356\000\000\000\003\000\000\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\330\004\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\210\005\000\000\000\000\000\000\n\000\000\000\000\000\000\000<\000\000\000\000\000\000\000\365\376\377o\000\000\000\0008\005\000\000\000\000\000\000\004\000\000\000\000\000\000\000`\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 23.0.0 (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000)\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\225\000\000\000\000\002\t\000\000/\000\000\000\000\000\000\000\000\000\000\000\000\000\000Z\000\000\000\022\003\b\000\000\027\000\000\000\000\000\000X\006\000\000\000\000\000\000h\000\000\000\021\003\006\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000y\000\000\000\021\000\013\000p?\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.eh_frame\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000amdgpu.max_num_named_barrier\000_Z4ringPKjPji\000_Z4ringPKjPji.kd\000__hip_cuid_f70806c800e446c8\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\234\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\330\004\000\000\000\000\000\000\330\004\000\000\000\000\000\000`\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\0008\005\000\000\000\000\000\0008\005\000\000\000\000\000\000(\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000`\005\000\000\000\000\000\000`\005\000\000\000\000\000\000(\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\210\005\000\000\000\000\000\000\210\005\000\000\000\000\000\000<\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000@\006\000\000\000\000\000\000@\006\000\000\000\000\000\0004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0009\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\007\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000?\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\000/\000\000\000\000\000\000\000\017\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000H\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p/\000\000\000\000\000\000p\017\000\000\000\000\000\000\220\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000W\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p?\000\000\000\000\000\000p\017\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\\\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\017\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000k\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\017\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\017\000\000\000\000\000\000\334\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\211\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000P\020\000\000\000\000\000\000\330\000\000\000\000\000\000\000\021\000\000\000\006\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000(\021\000\000\000\000\000\000\243\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\233\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\313\021\000\000\000\000\000\000\236\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_2, 9968

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_2
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_f70806c800e446c8,@object # @__hip_gpubin_handle_f70806c800e446c8
	.local	__hip_gpubin_handle_f70806c800e446c8
	.comm	__hip_gpubin_handle_f70806c800e446c8,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_f70806c800e446c8,@object # @__hip_cuid_f70806c800e446c8
	.bss
	.globl	__hip_cuid_f70806c800e446c8
__hip_cuid_f70806c800e446c8:
	.byte	0                               # 0x0
	.size	__hip_cuid_f70806c800e446c8, 1

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
	.addrsig_sym _Z19__device_stub__ringPKjPji
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _Z4ringPKjPji
	.addrsig_sym .L__unnamed_2
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_f70806c800e446c8
