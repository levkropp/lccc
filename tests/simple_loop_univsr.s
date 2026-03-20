.section .text
.globl test
.type test, @function
test:
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
.size test, .-test


.section .note.GNU-stack,"",@progbits
