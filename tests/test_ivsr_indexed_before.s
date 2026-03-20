.section .text
.globl sum_array
.type sum_array, @function
sum_array:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $48, %rsp
    movq %rbx, -48(%rbp)
    movq %r12, -40(%rbp)
    movq %r13, -32(%rbp)
    movq %r14, -24(%rbp)
    movq %r15, -16(%rbp)
    movq %rdi, %r14
    movq %rsi, %r15
    movslq %esi, %rax
    movq %rax, %r13
    xorl %eax, %eax
    movq %rax, -8(%rbp)
    movq %rax, %rbx
    movq %rdi, %r12
.LBB1:
    movslq %ebx, %rax
    movq %rax, %r14
    cmpl %r13d, %eax
    jge .LBB4
.LBB2:
    movsd -8(%rbp), %xmm0
    addsd (%r12), %xmm0
    movsd %xmm0, -8(%rbp)
    leaq 1(%rbx), %r14
    movslq %r14d, %rax
    movq %rax, %r15
    leaq 8(%r12), %rax
    movq %r15, %rbx
    movq %rax, %r12
    jmp .LBB1
.LBB4:
    movsd -8(%rbp), %xmm0
    movq -48(%rbp), %rbx
    movq -40(%rbp), %r12
    movq -32(%rbp), %r13
    movq -24(%rbp), %r14
    movq -16(%rbp), %r15
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size sum_array, .-sum_array

.globl add_arrays
.type add_arrays, @function
add_arrays:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $96, %rsp
    movq %rbx, -96(%rbp)
    movq %r12, -88(%rbp)
    movq %r13, -80(%rbp)
    movq %r14, -72(%rbp)
    movq %r15, -64(%rbp)
    movq %r11, -56(%rbp)
    movq %r10, -48(%rbp)
    movq %r8, -40(%rbp)
    movq %rdi, -8(%rbp)
    movq %rsi, -16(%rbp)
    movq %rdx, %r13
    movq %rcx, %r14
    movq %rdi, %r11
    movq %rsi, %r10
    movslq %ecx, %rax
    movq %rax, %r8
    xorl %eax, %eax
    movq %rax, %rbx
    movq %rdx, %r12
.LBB7:
    movslq %ebx, %rax
    movq %rax, %r13
    cmpl %r8d, %eax
    jge .LBB10
.LBB8:
    movq %r13, %r14
    shlq $3, %r14
    movq %r14, %rax
    addq %r11, %rax
    movsd (%r11,%r13,8), %xmm0
    movq %rax, -24(%rbp)
    movq %r14, %rax
    addq %r10, %rax
    movsd (%r10,%r13,8), %xmm0
    movq %rax, -32(%rbp)
    movsd -24(%rbp), %xmm0
    addsd -32(%rbp), %xmm0
    movsd %xmm0, (%r12)
    leaq 1(%rbx), %r14
    movslq %r14d, %rax
    movq %rax, %r15
    leaq 8(%r12), %rax
    movq %r15, %rbx
    movq %rax, %r12
    jmp .LBB7
.LBB10:
    movq -96(%rbp), %rbx
    movq -88(%rbp), %r12
    movq -80(%rbp), %r13
    movq -72(%rbp), %r14
    movq -64(%rbp), %r15
    movq -56(%rbp), %r11
    movq -48(%rbp), %r10
    movq -40(%rbp), %r8
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size add_arrays, .-add_arrays

.globl sum_longs
.type sum_longs, @function
sum_longs:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $64, %rsp
    movq %rbx, -64(%rbp)
    movq %r12, -56(%rbp)
    movq %r13, -48(%rbp)
    movq %r14, -40(%rbp)
    movq %r15, -32(%rbp)
    movq %r11, -24(%rbp)
    movq %r10, -16(%rbp)
    movq %rdi, %r15
    movq %rsi, %r14
    movslq %esi, %rax
    movq %rax, %r10
    xorl %eax, %eax
    movq %rax, %r11
    movq %rax, %r12
    movq %rdi, %r13
.LBB12:
    movslq %r12d, %rax
    movq %rax, %r14
    cmpl %r10d, %eax
    jge .LBB15
.LBB13:
    movq (%r13), %rax
    movq %r11, %r15
    addq %rax, %r15
    leaq 1(%r12), %r14
    movslq %r14d, %rax
    movq %rax, %rbx
    leaq 8(%r13), %rax
    movq %r15, %r11
    movq %rbx, %r12
    movq %rax, %r13
    jmp .LBB12
.LBB15:
    movq %r11, %rax
    movq -64(%rbp), %rbx
    movq -56(%rbp), %r12
    movq -48(%rbp), %r13
    movq -40(%rbp), %r14
    movq -32(%rbp), %r15
    movq -24(%rbp), %r11
    movq -16(%rbp), %r10
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size sum_longs, .-sum_longs

.globl sum_ints
.type sum_ints, @function
sum_ints:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $64, %rsp
    movq %rbx, -64(%rbp)
    movq %r12, -56(%rbp)
    movq %r13, -48(%rbp)
    movq %r14, -40(%rbp)
    movq %r15, -32(%rbp)
    movq %r11, -24(%rbp)
    movq %r10, -16(%rbp)
    movq %rdi, %r15
    movq %rsi, %r14
    movslq %esi, %rax
    movq %rax, %r10
    xorl %eax, %eax
    movq %rax, %r11
    movq %rax, %r12
    movq %rdi, %r13
.LBB18:
    movslq %r12d, %rax
    movq %rax, %r14
    cmpl %r10d, %eax
    jge .LBB21
.LBB19:
    movq %r13, %rcx
    movslq (%rcx), %rax
    movq %r11, %r15
    addl %eax, %r15d
    movslq %r15d, %r15
    leaq 1(%r12), %r14
    movslq %r14d, %rax
    movq %rax, %rbx
    leaq 4(%r13), %rax
    movq %r15, %r11
    movq %rbx, %r12
    movq %rax, %r13
    jmp .LBB18
.LBB21:
    movq %r11, %rax
    movq -64(%rbp), %rbx
    movq -56(%rbp), %r12
    movq -48(%rbp), %r13
    movq -40(%rbp), %r14
    movq -32(%rbp), %r15
    movq -24(%rbp), %r11
    movq -16(%rbp), %r10
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size sum_ints, .-sum_ints

.globl sum_points_x
.type sum_points_x, @function
sum_points_x:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $48, %rsp
    movq %rbx, -48(%rbp)
    movq %r12, -40(%rbp)
    movq %r13, -32(%rbp)
    movq %r14, -24(%rbp)
    movq %r15, -16(%rbp)
    movq %rdi, %r14
    movq %rsi, %r15
    movslq %esi, %rax
    movq %rax, %r13
    xorl %eax, %eax
    movq %rax, -8(%rbp)
    movq %rax, %rbx
    movq %rdi, %r12
.LBB24:
    movslq %ebx, %rax
    movq %rax, %r14
    cmpl %r13d, %eax
    jge .LBB27
.LBB25:
    movsd -8(%rbp), %xmm0
    addsd (%r12), %xmm0
    movsd %xmm0, -8(%rbp)
    leaq 1(%rbx), %r14
    movslq %r14d, %rax
    movq %rax, %r15
    leaq 24(%r12), %rax
    movq %r15, %rbx
    movq %rax, %r12
    jmp .LBB24
.LBB27:
    movsd -8(%rbp), %xmm0
    movq -48(%rbp), %rbx
    movq -40(%rbp), %r12
    movq -32(%rbp), %r13
    movq -24(%rbp), %r14
    movq -16(%rbp), %r15
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size sum_points_x, .-sum_points_x

.globl fill_array
.type fill_array, @function
fill_array:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $48, %rsp
    movq %rbx, -48(%rbp)
    movq %r12, -40(%rbp)
    movq %r13, -32(%rbp)
    movq %r14, -24(%rbp)
    movq %r15, -16(%rbp)
    movq %rdi, %r14
    movq %rsi, %r15
    movq %xmm0, -8(%rbp)
    movq -8(%rbp), %rax
    movq %rax, -8(%rbp)
    movslq %esi, %rax
    movq %rax, %r13
    xorl %eax, %eax
    movq %rax, %rbx
    movq %rdi, %r12
.LBB30:
    movslq %ebx, %rax
    movq %rax, %r14
    cmpl %r13d, %eax
    jge .LBB33
.LBB31:
    movq -8(%rbp), %rax
    movq %rax, %rdx
    movq %r12, %rcx
    movq %rdx, (%rcx)
    leaq 1(%rbx), %r14
    movslq %r14d, %rax
    movq %rax, %r15
    leaq 8(%r12), %rax
    movq %r15, %rbx
    movq %rax, %r12
    jmp .LBB30
.LBB33:
    movq -48(%rbp), %rbx
    movq -40(%rbp), %r12
    movq -32(%rbp), %r13
    movq -24(%rbp), %r14
    movq -16(%rbp), %r15
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size fill_array, .-fill_array

.globl scale_array
.type scale_array, @function
scale_array:
.cfi_startproc
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .cfi_def_cfa_register %rbp
    subq $48, %rsp
    movq %rbx, -48(%rbp)
    movq %r12, -40(%rbp)
    movq %r13, -32(%rbp)
    movq %r14, -24(%rbp)
    movq %r15, -16(%rbp)
    movq %rdi, %r14
    movq %rsi, %r15
    movq %xmm0, -8(%rbp)
    movq -8(%rbp), %rax
    movq %rax, -8(%rbp)
    movslq %esi, %rax
    movq %rax, %r13
    xorl %eax, %eax
    movq %rax, %r12
    movq %rdi, %rbx
.LBB35:
    movslq %r12d, %rax
    movq %rax, %r14
    cmpl %r13d, %eax
    jge .LBB38
.LBB36:
    movsd (%rbx), %xmm0
    mulsd -8(%rbp), %xmm0
    movsd %xmm0, (%rbx)
    leaq 1(%r12), %r14
    movslq %r14d, %rax
    movq %rax, %r15
    leaq 8(%rbx), %rax
    movq %r15, %r12
    movq %rax, %rbx
    jmp .LBB35
.LBB38:
    movq -48(%rbp), %rbx
    movq -40(%rbp), %r12
    movq -32(%rbp), %r13
    movq -24(%rbp), %r14
    movq -16(%rbp), %r15
    movq %rbp, %rsp
    popq %rbp
    ret
.cfi_endproc
.size scale_array, .-scale_array


.section .note.GNU-stack,"",@progbits
