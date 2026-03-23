//! x86-64 SSE/AES/CRC intrinsic emission and floating-point math intrinsics.
//!
//! Handles the `emit_intrinsic` trait method for the x86-64 backend, covering:
//! - Memory fences (lfence, mfence, sfence, pause, clflush)
//! - Non-temporal stores (movnti, movntdq, movntpd)
//! - SSE/SSE2 128-bit packed operations (arithmetic, compare, shuffle, shift)
//! - SSE2 element insertion/extraction and type conversion
//! - AES-NI encryption/decryption and key generation
//! - CLMUL carry-less multiplication
//! - CRC32 instructions
//! - Frame/return address intrinsics
//! - SSE scalar float math (sqrt, fabs) for F32/F64

use crate::ir::reexports::{
    IntrinsicOp,
    IrConst,
    Operand,
    Value,
};
use crate::backend::state::StackSlot;
use super::emit::X86Codegen;

impl X86Codegen {
    /// Load a float operand into %xmm0. Handles both Value operands (from stack)
    /// and float constants (loaded via their bit pattern into rax first).
    fn float_operand_to_xmm0(&mut self, op: &Operand, is_f32: bool) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::F64(v) => {
                        let bits = v.to_bits() as i64;
                        if bits == 0 {
                            self.state.emit("    xorpd %xmm0, %xmm0");
                        } else if bits >= i32::MIN as i64 && bits <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", bits, "rax");
                            self.state.emit("    movq %rax, %xmm0");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", bits, "rax");
                            self.state.emit("    movq %rax, %xmm0");
                        }
                    }
                    IrConst::F32(v) => {
                        let bits = v.to_bits() as i32;
                        if bits == 0 {
                            self.state.emit("    xorps %xmm0, %xmm0");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movl", bits as i64, "eax");
                            self.state.emit("    movd %eax, %xmm0");
                        }
                    }
                    _ => {
                        // Integer or other constants - load to rax and move to xmm
                        self.operand_to_reg(op, "rax");
                        if is_f32 {
                            self.state.emit("    movd %eax, %xmm0");
                        } else {
                            self.state.emit("    movq %rax, %xmm0");
                        }
                    }
                }
            }
            Operand::Value(_) => {
                // Load from stack slot to rax, then to xmm0
                self.operand_to_reg(op, "rax");
                if is_f32 {
                    self.state.emit("    movd %eax, %xmm0");
                } else {
                    self.state.emit("    movq %rax, %xmm0");
                }
            }
        }
    }

    fn emit_nontemporal_store(&mut self, op: &IntrinsicOp, dest_ptr: &Option<Value>, args: &[Operand]) {
        let Some(ptr) = dest_ptr else { return };
        match op {
            IntrinsicOp::Movnti => {
                self.operand_to_reg(&args[0], "rcx");
                self.value_to_reg(ptr, "rax");
                self.state.emit("    movnti %ecx, (%rax)");
            }
            IntrinsicOp::Movnti64 => {
                self.operand_to_reg(&args[0], "rcx");
                self.value_to_reg(ptr, "rax");
                self.state.emit("    movnti %rcx, (%rax)");
            }
            IntrinsicOp::Movntdq => {
                self.operand_to_reg(&args[0], "rcx");
                self.state.emit("    movdqu (%rcx), %xmm0");
                self.value_to_reg(ptr, "rax");
                self.state.emit("    movntdq %xmm0, (%rax)");
            }
            IntrinsicOp::Movntpd => {
                self.operand_to_reg(&args[0], "rcx");
                self.state.emit("    movupd (%rcx), %xmm0");
                self.value_to_reg(ptr, "rax");
                self.state.emit("    movntpd %xmm0, (%rax)");
            }
            _ => {}
        }
    }

    /// Emit SSE binary 128-bit op: load xmm0 from arg0 ptr, xmm1 from arg1 ptr,
    /// apply the given SSE instruction, store result xmm0 to dest_ptr.
    fn emit_sse_binary_128(&mut self, dest_ptr: &Value, args: &[Operand], sse_inst: &str) {
        // Load destination address FIRST into a dedicated register to avoid clobbering
        self.value_to_reg(dest_ptr, "rdx");
        // Load operands into separate registers
        self.operand_to_reg(&args[0], "rax");
        self.state.emit("    movdqu (%rax), %xmm0");
        self.operand_to_reg(&args[1], "rcx");
        self.state.emit("    movdqu (%rcx), %xmm1");
        // Perform vector operation
        self.state.emit_fmt(format_args!("    {} %xmm1, %xmm0", sse_inst));
        // Store result using dedicated destination register
        self.state.emit("    movdqu %xmm0, (%rdx)");
    }

    /// Emit SSE unary 128-bit op with immediate: load xmm0 from arg0 ptr,
    /// apply `inst $imm, %xmm0`, store result xmm0 to dest_ptr.
    fn emit_sse_unary_imm_128(&mut self, dest_ptr: &Value, args: &[Operand], sse_inst: &str) {
        self.operand_to_reg(&args[0], "rax");
        self.state.emit("    movdqu (%rax), %xmm0");
        let imm = self.operand_to_imm_i64(&args[1]);
        self.state.emit_fmt(format_args!("    {} ${}, %xmm0", sse_inst, imm));
        self.value_to_reg(dest_ptr, "rax");
        self.state.emit("    movdqu %xmm0, (%rax)");
    }

    /// Emit SSE shuffle with immediate: load xmm0, apply `inst $imm, %xmm0, %xmm0`,
    /// store result. Used for pshufd/pshuflw/pshufhw which read and write same register.
    fn emit_sse_shuffle_imm_128(&mut self, dest_ptr: &Value, args: &[Operand], sse_inst: &str) {
        self.operand_to_reg(&args[0], "rax");
        self.state.emit("    movdqu (%rax), %xmm0");
        let imm = self.operand_to_imm_i64(&args[1]);
        self.state.emit_fmt(format_args!("    {} ${}, %xmm0, %xmm0", sse_inst, imm));
        self.value_to_reg(dest_ptr, "rax");
        self.state.emit("    movdqu %xmm0, (%rax)");
    }

    pub(super) fn emit_intrinsic_impl(&mut self, dest: &Option<Value>, op: &IntrinsicOp, dest_ptr: &Option<Value>, args: &[Operand]) {
        match op {
            IntrinsicOp::Lfence => { self.state.emit("    lfence"); }
            IntrinsicOp::Mfence => { self.state.emit("    mfence"); }
            IntrinsicOp::Sfence => { self.state.emit("    sfence"); }
            IntrinsicOp::Pause => { self.state.emit("    pause"); }
            IntrinsicOp::Clflush => {
                // args[0] = pointer to flush
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    clflush (%rax)");
            }
            IntrinsicOp::Movnti | IntrinsicOp::Movnti64
            | IntrinsicOp::Movntdq | IntrinsicOp::Movntpd => {
                self.emit_nontemporal_store(op, dest_ptr, args);
            }
            IntrinsicOp::Loaddqu => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Storedqu => {
                if let Some(ptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rcx");
                    self.state.emit("    movdqu (%rcx), %xmm0");
                    self.value_to_reg(ptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Pcmpeqb128 | IntrinsicOp::Pcmpeqd128
            | IntrinsicOp::Psubusb128 | IntrinsicOp::Psubsb128
            | IntrinsicOp::Por128
            | IntrinsicOp::Pand128 | IntrinsicOp::Pxor128 => {
                if let Some(dptr) = dest_ptr {
                    let inst = match op {
                        IntrinsicOp::Pcmpeqb128 => "pcmpeqb",
                        IntrinsicOp::Pcmpeqd128 => "pcmpeqd",
                        IntrinsicOp::Psubusb128 => "psubusb",
                        IntrinsicOp::Psubsb128 => "psubsb",
                        IntrinsicOp::Por128 => "por",
                        IntrinsicOp::Pand128 => "pand",
                        IntrinsicOp::Pxor128 => "pxor",
                        _ => unreachable!("unexpected SSE binary op: {:?}", op),
                    };
                    self.emit_sse_binary_128(dptr, args, inst);
                }
            }
            IntrinsicOp::Pmovmskb128 => {
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                self.state.emit("    pmovmskb %xmm0, %eax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::SetEpi8 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movd %eax, %xmm0");
                    self.state.emit("    punpcklbw %xmm0, %xmm0");
                    self.state.emit("    punpcklwd %xmm0, %xmm0");
                    self.state.emit("    pshufd $0, %xmm0, %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::SetEpi32 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movd %eax, %xmm0");
                    self.state.emit("    pshufd $0, %xmm0, %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Crc32_8 | IntrinsicOp::Crc32_16
            | IntrinsicOp::Crc32_32 | IntrinsicOp::Crc32_64 => {
                self.operand_to_reg(&args[0], "rax");
                self.operand_to_reg(&args[1], "rcx");
                let inst = match op {
                    IntrinsicOp::Crc32_8  => "crc32b %cl, %eax",
                    IntrinsicOp::Crc32_16 => "crc32w %cx, %eax",
                    IntrinsicOp::Crc32_32 => "crc32l %ecx, %eax",
                    IntrinsicOp::Crc32_64 => "crc32q %rcx, %rax",
                    _ => unreachable!("CRC32 dispatch matched non-CRC32 op: {:?}", op),
                };
                self.state.emit_fmt(format_args!("    {}", inst));
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::FrameAddress => {
                // __builtin_frame_address(0): return current frame pointer (rbp)
                self.state.emit("    movq %rbp, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::ReturnAddress => {
                // __builtin_return_address(0): return address is at (%rbp)+8
                self.state.emit("    movq 8(%rbp), %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::ThreadPointer => {
                // __builtin_thread_pointer(): read the TLS base from %fs:0
                self.state.emit("    movq %fs:0, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::SqrtF64 => {
                // sqrtsd: scalar double-precision square root
                self.float_operand_to_xmm0(&args[0], false);
                self.state.emit("    sqrtsd %xmm0, %xmm0");
                self.state.emit("    movq %xmm0, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::SqrtF32 => {
                // sqrtss: scalar single-precision square root
                self.float_operand_to_xmm0(&args[0], true);
                self.state.emit("    sqrtss %xmm0, %xmm0");
                self.state.emit("    movd %xmm0, %eax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::FabsF64 => {
                // Clear sign bit for double-precision absolute value
                self.float_operand_to_xmm0(&args[0], false);
                self.state.emit("    movabsq $0x7FFFFFFFFFFFFFFF, %rcx");
                self.state.emit("    movq %rcx, %xmm1");
                self.state.emit("    andpd %xmm1, %xmm0");
                self.state.emit("    movq %xmm0, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::FabsF32 => {
                // Clear sign bit for single-precision absolute value
                self.float_operand_to_xmm0(&args[0], true);
                self.state.emit("    movl $0x7FFFFFFF, %ecx");
                self.state.emit("    movd %ecx, %xmm1");
                self.state.emit("    andps %xmm1, %xmm0");
                self.state.emit("    movd %xmm0, %eax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            // AES-NI binary ops: aesenc, aesenclast, aesdec, aesdeclast
            IntrinsicOp::Aesenc128 | IntrinsicOp::Aesenclast128
            | IntrinsicOp::Aesdec128 | IntrinsicOp::Aesdeclast128 => {
                if let Some(dptr) = dest_ptr {
                    let inst = match op {
                        IntrinsicOp::Aesenc128 => "aesenc",
                        IntrinsicOp::Aesenclast128 => "aesenclast",
                        IntrinsicOp::Aesdec128 => "aesdec",
                        IntrinsicOp::Aesdeclast128 => "aesdeclast",
                        _ => unreachable!("AES-NI dispatch matched non-AES op: {:?}", op),
                    };
                    self.emit_sse_binary_128(dptr, args, inst);
                }
            }
            // AES-NI unary: aesimc
            IntrinsicOp::Aesimc128 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.state.emit("    aesimc %xmm0, %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            // AES-NI: aeskeygenassist with immediate
            IntrinsicOp::Aeskeygenassist128 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    // args[1] is the immediate value
                    let imm = self.operand_to_imm_i64(&args[1]);
                    self.state.emit_fmt(format_args!("    aeskeygenassist ${}, %xmm0, %xmm0", imm));
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            // CLMUL: pclmulqdq with immediate
            IntrinsicOp::Pclmulqdq128 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    movdqu (%rcx), %xmm1");
                    let imm = self.operand_to_imm_i64(&args[2]);
                    self.state.emit_fmt(format_args!("    pclmulqdq ${}, %xmm1, %xmm0", imm));
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            // SSE2 shift-by-immediate operations
            IntrinsicOp::Pslldqi128 | IntrinsicOp::Psrldqi128
            | IntrinsicOp::Psllqi128 | IntrinsicOp::Psrlqi128 => {
                if let Some(dptr) = dest_ptr {
                    let inst = match op {
                        IntrinsicOp::Pslldqi128 => "pslldq",
                        IntrinsicOp::Psrldqi128 => "psrldq",
                        IntrinsicOp::Psllqi128 => "psllq",
                        IntrinsicOp::Psrlqi128 => "psrlq",
                        _ => unreachable!("unexpected SSE shift-by-immediate op: {:?}", op),
                    };
                    self.emit_sse_unary_imm_128(dptr, args, inst);
                }
            }
            // SSE2 shuffle with immediate (3-operand form: inst $imm, %src, %dst)
            IntrinsicOp::Pshufd128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_sse_shuffle_imm_128(dptr, args, "pshufd");
                }
            }
            // Load low 64 bits, zero upper (MOVQ)
            IntrinsicOp::Loadldi128 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movq (%rax), %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }

            // SSE2 binary 128-bit operations
            IntrinsicOp::Paddw128 | IntrinsicOp::Psubw128 | IntrinsicOp::Pmulhw128
            | IntrinsicOp::Pmaddwd128 | IntrinsicOp::Pcmpgtw128 | IntrinsicOp::Pcmpgtb128
            | IntrinsicOp::Paddd128 | IntrinsicOp::Psubd128
            | IntrinsicOp::Packssdw128 | IntrinsicOp::Packsswb128 | IntrinsicOp::Packuswb128
            | IntrinsicOp::Punpcklbw128 | IntrinsicOp::Punpckhbw128
            | IntrinsicOp::Punpcklwd128 | IntrinsicOp::Punpckhwd128 => {
                if let Some(dptr) = dest_ptr {
                    let inst = match op {
                        IntrinsicOp::Paddw128 => "paddw",
                        IntrinsicOp::Psubw128 => "psubw",
                        IntrinsicOp::Pmulhw128 => "pmulhw",
                        IntrinsicOp::Pmaddwd128 => "pmaddwd",
                        IntrinsicOp::Pcmpgtw128 => "pcmpgtw",
                        IntrinsicOp::Pcmpgtb128 => "pcmpgtb",
                        IntrinsicOp::Paddd128 => "paddd",
                        IntrinsicOp::Psubd128 => "psubd",
                        IntrinsicOp::Packssdw128 => "packssdw",
                        IntrinsicOp::Packsswb128 => "packsswb",
                        IntrinsicOp::Packuswb128 => "packuswb",
                        IntrinsicOp::Punpcklbw128 => "punpcklbw",
                        IntrinsicOp::Punpckhbw128 => "punpckhbw",
                        IntrinsicOp::Punpcklwd128 => "punpcklwd",
                        IntrinsicOp::Punpckhwd128 => "punpckhwd",
                        _ => unreachable!("unexpected SSE binary op: {:?}", op),
                    };
                    self.emit_sse_binary_128(dptr, args, inst);
                }
            }

            // SSE2 element shift-by-immediate operations
            IntrinsicOp::Psllwi128 | IntrinsicOp::Psrlwi128 | IntrinsicOp::Psrawi128
            | IntrinsicOp::Psradi128 | IntrinsicOp::Pslldi128 | IntrinsicOp::Psrldi128 => {
                if let Some(dptr) = dest_ptr {
                    let inst = match op {
                        IntrinsicOp::Psllwi128 => "psllw",
                        IntrinsicOp::Psrlwi128 => "psrlw",
                        IntrinsicOp::Psrawi128 => "psraw",
                        IntrinsicOp::Psradi128 => "psrad",
                        IntrinsicOp::Pslldi128 => "pslld",
                        IntrinsicOp::Psrldi128 => "psrld",
                        _ => unreachable!("unexpected SSE element shift op: {:?}", op),
                    };
                    self.emit_sse_unary_imm_128(dptr, args, inst);
                }
            }

            // --- SSE2 set/insert/extract/convert ---
            IntrinsicOp::SetEpi16 => {
                // Broadcast 16-bit value to all 8 lanes
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movd %eax, %xmm0");
                    self.state.emit("    punpcklwd %xmm0, %xmm0");
                    self.state.emit("    pshufd $0, %xmm0, %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Pinsrw128 => {
                // Insert 16-bit value at lane: pinsrw $imm, %eax, %xmm0
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.operand_to_reg(&args[1], "rcx");
                    let imm = self.operand_to_imm_i64(&args[2]);
                    self.state.emit_fmt(format_args!("    pinsrw ${}, %ecx, %xmm0", imm));
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Pextrw128 => {
                // Extract 16-bit value at lane: pextrw $imm, %xmm0, %eax
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                let imm = self.operand_to_imm_i64(&args[1]);
                self.state.emit_fmt(format_args!("    pextrw ${}, %xmm0, %eax", imm));
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::Pinsrd128 => {
                // Insert 32-bit value at lane: pinsrd $imm, %eax, %xmm0 (SSE4.1)
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.operand_to_reg(&args[1], "rcx");
                    let imm = self.operand_to_imm_i64(&args[2]);
                    self.state.emit_fmt(format_args!("    pinsrd ${}, %ecx, %xmm0", imm));
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Pextrd128 => {
                // Extract 32-bit value at lane: pextrd $imm, %xmm0, %eax (SSE4.1)
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                let imm = self.operand_to_imm_i64(&args[1]);
                self.state.emit_fmt(format_args!("    pextrd ${}, %xmm0, %eax", imm));
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::Pinsrb128 => {
                // Insert 8-bit value at lane: pinsrb $imm, %eax, %xmm0 (SSE4.1)
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.operand_to_reg(&args[1], "rcx");
                    let imm = self.operand_to_imm_i64(&args[2]);
                    self.state.emit_fmt(format_args!("    pinsrb ${}, %ecx, %xmm0", imm));
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Pextrb128 => {
                // Extract 8-bit value at lane: pextrb $imm, %xmm0, %eax (SSE4.1)
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                let imm = self.operand_to_imm_i64(&args[1]);
                self.state.emit_fmt(format_args!("    pextrb ${}, %xmm0, %eax", imm));
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::Pinsrq128 => {
                // Insert 64-bit value at lane: pinsrq $imm, %rax, %xmm0 (SSE4.1)
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                    self.operand_to_reg(&args[1], "rcx");
                    let imm = self.operand_to_imm_i64(&args[2]);
                    self.state.emit_fmt(format_args!("    pinsrq ${}, %rcx, %xmm0", imm));
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Pextrq128 => {
                // Extract 64-bit value at lane: pextrq $imm, %xmm0, %rax (SSE4.1)
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                let imm = self.operand_to_imm_i64(&args[1]);
                self.state.emit_fmt(format_args!("    pextrq ${}, %xmm0, %rax", imm));
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::Storeldi128 => {
                // Store low 64 bits to memory (MOVQ)
                if let Some(ptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rcx");
                    self.state.emit("    movdqu (%rcx), %xmm0");
                    self.value_to_reg(ptr, "rax");
                    self.state.emit("    movq %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Cvtsi128Si32 => {
                // Extract low 32-bit integer (MOVD)
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                self.state.emit("    movd %xmm0, %eax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::Cvtsi32Si128 => {
                // Convert int to __m128i (MOVD, zero-extends upper bits)
                if let Some(dptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movd %eax, %xmm0");
                    self.value_to_reg(dptr, "rax");
                    self.state.emit("    movdqu %xmm0, (%rax)");
                }
            }
            IntrinsicOp::Cvtsi128Si64 => {
                // Extract low 64-bit integer (MOVQ)
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");
                self.state.emit("    movq %xmm0, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::Pshuflw128 | IntrinsicOp::Pshufhw128 => {
                if let Some(dptr) = dest_ptr {
                    let inst = match op {
                        IntrinsicOp::Pshuflw128 => "pshuflw",
                        IntrinsicOp::Pshufhw128 => "pshufhw",
                        _ => unreachable!("unexpected SSE shuffle op: {:?}", op),
                    };
                    self.emit_sse_shuffle_imm_128(dptr, args, inst);
                }
            }
            IntrinsicOp::FmaF64x2 => {
                // dest_ptr[0..2] += broadcast(args[0]) * args[1][0..2]
                // args[0] = A pointer (scalar F64, broadcast to both lanes)
                // args[1] = B pointer (2×F64)
                // dest_ptr = C pointer (read+write, 2×F64)
                if let Some(c_ptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rcx");      // A ptr → %rcx
                    self.operand_to_reg(&args[1], "rdx");      // B ptr → %rdx
                    self.value_to_reg(c_ptr, "rax");           // C ptr → %rax
                    self.state.emit("    movsd (%rcx), %xmm1");       // xmm1 = A scalar
                    self.state.emit("    unpcklpd %xmm1, %xmm1");     // xmm1 = {A, A}
                    self.state.emit("    movupd (%rdx), %xmm0");      // xmm0 = {B[j], B[j+1]}
                    self.state.emit("    mulpd %xmm1, %xmm0");        // xmm0 = {A*Bj, A*Bj1}
                    self.state.emit("    addpd (%rax), %xmm0");       // xmm0 += {C[j], C[j+1]}
                    self.state.emit("    movupd %xmm0, (%rax)");      // store back
                }
            }
            IntrinsicOp::FmaF64x4 => {
                // dest_ptr[0..4] += broadcast(args[0]) * args[1][0..4]
                // args[0] = A pointer (scalar F64, broadcast to all 4 lanes)
                // args[1] = B pointer (4×F64)
                // dest_ptr = C pointer (read+write, 4×F64)
                if let Some(c_ptr) = dest_ptr {
                    self.operand_to_reg(&args[0], "rcx");      // A ptr → %rcx
                    self.operand_to_reg(&args[1], "rdx");      // B ptr → %rdx
                    self.value_to_reg(c_ptr, "rax");           // C ptr → %rax

                    // AVX2 instructions (VEX-encoded, 256-bit ymm registers)
                    self.state.emit("    movsd (%rcx), %xmm1");          // Load A scalar (64-bit)
                    self.state.emit("    vbroadcastsd %xmm1, %ymm1");    // Broadcast to {A, A, A, A}
                    self.state.emit("    vmovupd (%rdx), %ymm0");        // Load 4 doubles unaligned
                    self.state.emit("    vmulpd %ymm1, %ymm0, %ymm0");   // ymm0 = A * {B[j..j+3]}
                    self.state.emit("    vaddpd (%rax), %ymm0, %ymm0");  // ymm0 += {C[j..j+3]}
                    self.state.emit("    vmovupd %ymm0, (%rax)");        // Write 4 results back
                }
            }

            // --- Vector loads for reduction patterns ---
            IntrinsicOp::LoadF64x4 => {
                // Load 4 packed doubles: vmovupd (%base + %offset), %ymm0
                if let Some(dptr) = dest_ptr {
                    self.value_to_reg(dptr, "rdx");          // Load dest FIRST into %rdx
                    self.operand_to_reg(&args[0], "rax");    // base pointer
                    self.operand_to_reg(&args[1], "rcx");    // byte offset
                    self.state.emit("    vmovupd (%rax,%rcx), %ymm0");
                    self.state.emit("    vmovupd %ymm0, (%rdx)");  // Store to %rdx
                }
            }
            IntrinsicOp::LoadF64x2 => {
                // Load 2 packed doubles: movupd (%base + %offset), %xmm0
                if let Some(dptr) = dest_ptr {
                    self.value_to_reg(dptr, "rdx");          // Load dest FIRST into %rdx
                    self.operand_to_reg(&args[0], "rax");
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    movupd (%rax,%rcx), %xmm0");
                    self.state.emit("    movupd %xmm0, (%rdx)");  // Store to %rdx
                }
            }
            IntrinsicOp::LoadI32x8 => {
                // Load 8 packed ints: vmovdqu (%base + %offset), %ymm0
                if let Some(dptr) = dest_ptr {
                    self.value_to_reg(dptr, "rdx");          // Load dest FIRST into %rdx
                    self.operand_to_reg(&args[0], "rax");
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    vmovdqu (%rax,%rcx), %ymm0");
                    self.state.emit("    vmovdqu %ymm0, (%rdx)");  // Store to %rdx
                }
            }
            IntrinsicOp::LoadI32x4 => {
                // Load 4 packed ints: movdqu (%base + %offset), %xmm0
                if let Some(dptr) = dest_ptr {
                    self.value_to_reg(dptr, "rdx");          // Load dest FIRST into %rdx
                    self.operand_to_reg(&args[0], "rax");
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    movdqu (%rax,%rcx), %xmm0");
                    self.state.emit("    movdqu %xmm0, (%rdx)");  // Store to %rdx
                }
            }

            // --- Vector arithmetic ---
            IntrinsicOp::AddF64x4 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_avx_binary_256(dptr, args, "vaddpd");
                }
            }
            IntrinsicOp::AddF64x2 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_sse_binary_128(dptr, args, "addpd");
                }
            }
            IntrinsicOp::MulF64x4 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_avx_binary_256(dptr, args, "vmulpd");
                }
            }
            IntrinsicOp::MulF64x2 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_sse_binary_128(dptr, args, "mulpd");
                }
            }
            IntrinsicOp::AddI32x8 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_avx_binary_256(dptr, args, "vpaddd");
                }
            }
            IntrinsicOp::AddI32x4 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_sse_binary_128(dptr, args, "paddd");
                }
            }

            // --- Horizontal reduction ---
            IntrinsicOp::HorizontalAddF64x4 => {
                // Reduce 4×F64 → 1×F64
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    vmovupd (%rax), %ymm0");        // Load 4 doubles
                self.state.emit("    vextractf128 $1, %ymm0, %xmm1"); // Extract upper 128 bits
                self.state.emit("    vaddpd %xmm1, %xmm0, %xmm0");    // Add upper + lower (4→2)
                self.state.emit("    vunpckhpd %xmm0, %xmm0, %xmm1"); // Shuffle element 1 to position 0
                self.state.emit("    vaddsd %xmm1, %xmm0, %xmm0");    // Final scalar add (2→1)
                self.state.emit("    vmovq %xmm0, %rax");             // Extract to GPR
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::HorizontalAddF64x2 => {
                // Reduce 2×F64 → 1×F64
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movupd (%rax), %xmm0");          // Load 2 doubles
                self.state.emit("    unpckhpd %xmm0, %xmm0, %xmm1");  // Duplicate upper to xmm1
                self.state.emit("    addsd %xmm1, %xmm0");            // Add (2→1)
                self.state.emit("    movq %xmm0, %rax");              // Extract to GPR
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::HorizontalAddI32x8 => {
                // Reduce 8×I32 → 1×I32
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    vmovdqu (%rax), %ymm0");         // Load 8 ints
                self.state.emit("    vextracti128 $1, %ymm0, %xmm1"); // Extract upper 128 (8→4)
                self.state.emit("    vpaddd %xmm1, %xmm0, %xmm0");    // Add halves (8→4)
                self.state.emit("    vpsrldq $8, %xmm0, %xmm1");      // Shift 8 bytes (4→2)
                self.state.emit("    vpaddd %xmm1, %xmm0, %xmm0");    // Add (4→2)
                self.state.emit("    vpsrldq $4, %xmm0, %xmm1");      // Shift 4 bytes (2→1)
                self.state.emit("    vpaddd %xmm1, %xmm0, %xmm0");    // Add (2→1)
                self.state.emit("    vmovd %xmm0, %eax");             // Extract to GPR
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::HorizontalAddI32x4 => {
                // Reduce 4×I32 → 1×I32
                self.operand_to_reg(&args[0], "rax");
                self.state.emit("    movdqu (%rax), %xmm0");          // Load 4 ints
                self.state.emit("    psrldq $8, %xmm0, %xmm1");       // Shift 8 bytes (4→2)
                self.state.emit("    paddd %xmm1, %xmm0");            // Add (4→2)
                self.state.emit("    psrldq $4, %xmm0, %xmm1");       // Shift 4 bytes (2→1)
                self.state.emit("    paddd %xmm1, %xmm0");            // Add (2→1)
                self.state.emit("    movd %xmm0, %eax");              // Extract to GPR
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }

            // --- Register-based vector operations (SSA-friendly) ---

            IntrinsicOp::VecLoadF64x4 => {
                // %dest_vec = load_vector(base_ptr, offset) - AVX2 4×F64
                // Load from memory array into ymm0, then store directly to stack slot
                self.operand_to_reg(&args[0], "rax");  // base pointer
                self.operand_to_reg(&args[1], "rcx");  // offset
                self.state.emit("    vmovupd (%rax,%rcx), %ymm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store vector directly to stack slot (not via pointer indirection)
                        self.state.out.emit_instr_reg_rbp("    vmovupd", "ymm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecLoadF64x2 => {
                // %dest_vec = load_vector(base_ptr, offset) - SSE2 2×F64
                self.operand_to_reg(&args[0], "rax");
                self.operand_to_reg(&args[1], "rcx");
                self.state.emit("    movupd (%rax,%rcx), %xmm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store vector directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    movupd", "xmm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecLoadI32x8 => {
                // %dest_vec = load_vector(base_ptr, offset) - AVX2 8×I32
                self.operand_to_reg(&args[0], "rax");
                self.operand_to_reg(&args[1], "rcx");
                self.state.emit("    vmovdqu (%rax,%rcx), %ymm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rdx");
                        self.state.emit("    vmovdqu %ymm0, (%rdx)");
                    }
                }
            }
            IntrinsicOp::VecLoadI32x4 => {
                // %dest_vec = load_vector(base_ptr, offset) - SSE2 4×I32
                self.operand_to_reg(&args[0], "rax");
                self.operand_to_reg(&args[1], "rcx");
                self.state.emit("    movdqu (%rax,%rcx), %xmm0");
                if let Some(d) = dest {
                    if let Some(slot) = self.state.get_slot(d.0) {
                        self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rdx");
                        self.state.emit("    movdqu %xmm0, (%rdx)");
                    }
                }
            }

            IntrinsicOp::VecAddF64x4 => {
                // %dest_vec = %src1_vec + %src2_vec - AVX2 4×F64
                // Load both source vectors directly from their stack slots and add.
                // Vector values are stored directly in stack slots (not as pointers),
                // so we load them with offset(%rbp) addressing, not pointer indirection.
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    // Vector operand: load directly from stack slot
                    self.state.out.emit_instr_rbp_reg("    vmovupd", slot.0 as i64, "ymm0");
                } else {
                    // Non-vector operand (shouldn't happen for VecAdd, but handle gracefully)
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    vmovupd (%rax), %ymm0");
                }
                if let Some(slot) = self.get_slot_for_operand(&args[1]) {
                    // Vector operand: load directly from stack slot
                    self.state.out.emit_instr_rbp_reg("    vmovupd", slot.0 as i64, "ymm1");
                } else {
                    // Non-vector operand (shouldn't happen for VecAdd, but handle gracefully)
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    vmovupd (%rcx), %ymm1");
                }
                self.state.emit("    vaddpd %ymm1, %ymm0, %ymm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store result directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    vmovupd", "ymm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecAddF64x2 => {
                // %dest_vec = %src1_vec + %src2_vec - SSE2 2×F64
                // Load both source vectors directly from stack slots
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    self.state.out.emit_instr_rbp_reg("    movupd", slot.0 as i64, "xmm0");
                } else {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movupd (%rax), %xmm0");
                }
                if let Some(slot) = self.get_slot_for_operand(&args[1]) {
                    self.state.out.emit_instr_rbp_reg("    movupd", slot.0 as i64, "xmm1");
                } else {
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    movupd (%rcx), %xmm1");
                }
                self.state.emit("    addpd %xmm1, %xmm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store result directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    movupd", "xmm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecMulF64x4 => {
                // %dest_vec = %src1_vec * %src2_vec - AVX2 4×F64
                // Load both source vectors directly from stack slots
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    self.state.out.emit_instr_rbp_reg("    vmovupd", slot.0 as i64, "ymm0");
                } else {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    vmovupd (%rax), %ymm0");
                }
                if let Some(slot) = self.get_slot_for_operand(&args[1]) {
                    self.state.out.emit_instr_rbp_reg("    vmovupd", slot.0 as i64, "ymm1");
                } else {
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    vmovupd (%rcx), %ymm1");
                }
                self.state.emit("    vmulpd %ymm1, %ymm0, %ymm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store result directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    vmovupd", "ymm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecMulF64x2 => {
                // %dest_vec = %src1_vec * %src2_vec - SSE2 2×F64
                // Load both source vectors directly from stack slots
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    self.state.out.emit_instr_rbp_reg("    movupd", slot.0 as i64, "xmm0");
                } else {
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movupd (%rax), %xmm0");
                }
                if let Some(slot) = self.get_slot_for_operand(&args[1]) {
                    self.state.out.emit_instr_rbp_reg("    movupd", slot.0 as i64, "xmm1");
                } else {
                    self.operand_to_reg(&args[1], "rcx");
                    self.state.emit("    movupd (%rcx), %xmm1");
                }
                self.state.emit("    mulpd %xmm1, %xmm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store result directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    movupd", "xmm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecAddI32x8 | IntrinsicOp::VecAddI32x4 => {
                // Vector integer add (similar pattern)
                let (load_inst, add_inst, store_inst, reg) = match op {
                    IntrinsicOp::VecAddI32x8 => ("vmovdqu", "vpaddd", "vmovdqu", "ymm"),
                    IntrinsicOp::VecAddI32x4 => ("movdqu", "paddd", "movdqu", "xmm"),
                    _ => unreachable!(),
                };
                self.operand_to_reg(&args[0], "rax");
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rax");
                }
                self.state.emit_fmt(format_args!("    {} (%rax), %{}0", load_inst, reg));
                self.operand_to_reg(&args[1], "rcx");
                if let Some(slot) = self.get_slot_for_operand(&args[1]) {
                    self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rcx");
                }
                self.state.emit_fmt(format_args!("    {} (%rcx), %{}1", load_inst, reg));
                self.state.emit_fmt(format_args!("    {} %{}1, %{}0, %{}0", add_inst, reg, reg, reg));
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rdx");
                        self.state.emit_fmt(format_args!("    {} %{}0, (%rdx)", store_inst, reg));
                    }
                }
            }

            IntrinsicOp::VecHorizontalAddF64x4 => {
                // %scalar = horizontal_add(%vec) - AVX2 4×F64 → F64
                // Load vector from operand and reduce
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    // Direct load from slot
                    self.state.out.emit_instr_rbp_reg("    vmovupd", slot.0 as i64, "ymm0");
                } else {
                    // Fallback: load pointer then dereference
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    vmovupd (%rax), %ymm0");
                }
                self.state.emit("    vextractf128 $1, %ymm0, %xmm1");
                self.state.emit("    vaddpd %xmm1, %xmm0, %xmm0");
                self.state.emit("    vunpckhpd %xmm0, %xmm0, %xmm1");
                self.state.emit("    vaddsd %xmm1, %xmm0, %xmm0");
                self.state.emit("    vmovq %xmm0, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::VecHorizontalAddF64x2 => {
                // %scalar = horizontal_add(%vec) - SSE2 2×F64 → F64
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    // Direct load from slot
                    self.state.out.emit_instr_rbp_reg("    movupd", slot.0 as i64, "xmm0");
                } else {
                    // Fallback: load pointer then dereference
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movupd (%rax), %xmm0");
                }
                self.state.emit("    unpckhpd %xmm0, %xmm0, %xmm1");
                self.state.emit("    addsd %xmm1, %xmm0");
                self.state.emit("    movq %xmm0, %rax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::VecHorizontalAddI32x8 => {
                // %scalar = horizontal_add(%vec) - AVX2 8×I32 → I32
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    // Direct load from slot
                    self.state.out.emit_instr_rbp_reg("    vmovdqu", slot.0 as i64, "ymm0");
                } else {
                    // Fallback: load pointer then dereference
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    vmovdqu (%rax), %ymm0");
                }
                self.state.emit("    vextracti128 $1, %ymm0, %xmm1");
                self.state.emit("    vpaddd %xmm1, %xmm0, %xmm0");
                self.state.emit("    vpsrldq $8, %xmm0, %xmm1");
                self.state.emit("    vpaddd %xmm1, %xmm0, %xmm0");
                self.state.emit("    vpsrldq $4, %xmm0, %xmm1");
                self.state.emit("    vpaddd %xmm1, %xmm0, %xmm0");
                self.state.emit("    vmovd %xmm0, %eax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }
            IntrinsicOp::VecHorizontalAddI32x4 => {
                // %scalar = horizontal_add(%vec) - SSE2 4×I32 → I32
                if let Some(slot) = self.get_slot_for_operand(&args[0]) {
                    // Direct load from slot
                    self.state.out.emit_instr_rbp_reg("    movdqu", slot.0 as i64, "xmm0");
                } else {
                    // Fallback: load pointer then dereference
                    self.operand_to_reg(&args[0], "rax");
                    self.state.emit("    movdqu (%rax), %xmm0");
                }
                self.state.emit("    psrldq $8, %xmm0, %xmm1");
                self.state.emit("    paddd %xmm1, %xmm0");
                self.state.emit("    psrldq $4, %xmm0, %xmm1");
                self.state.emit("    paddd %xmm1, %xmm0");
                self.state.emit("    movd %xmm0, %eax");
                if let Some(d) = dest {
                    self.store_rax_to(d);
                }
            }

            IntrinsicOp::VecZeroF64x4 => {
                // %dest_vec = {0.0, 0.0, 0.0, 0.0} - AVX2 4×F64
                self.state.emit("    vxorpd %ymm0, %ymm0, %ymm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store zero vector directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    vmovupd", "ymm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecZeroF64x2 => {
                // %dest_vec = {0.0, 0.0} - SSE2 2×F64
                self.state.emit("    xorpd %xmm0, %xmm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        // Store zero vector directly to stack slot
                        self.state.out.emit_instr_reg_rbp("    movupd", "xmm0", slot.0 as i64);
                    }
                }
            }
            IntrinsicOp::VecZeroI32x8 => {
                // %dest_vec = {0, 0, 0, 0, 0, 0, 0, 0} - AVX2 8×I32
                self.state.emit("    vpxor %ymm0, %ymm0, %ymm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rdx");
                        self.state.emit("    vmovdqu %ymm0, (%rdx)");
                    }
                }
            }
            IntrinsicOp::VecZeroI32x4 => {
                // %dest_vec = {0, 0, 0, 0} - SSE2 4×I32
                self.state.emit("    pxor %xmm0, %xmm0");
                if let Some(d) = dest {
                    self.state.vector_values.insert(d.0);
                    if let Some(slot) = self.state.get_slot(d.0) {
                        self.state.out.emit_instr_rbp_reg("    leaq", slot.0 as i64, "rdx");
                        self.state.emit("    movdqu %xmm0, (%rdx)");
                    }
                }
            }
        }
    }

    /// Helper: Get stack slot for an operand if it's a Value
    fn get_slot_for_operand(&self, op: &Operand) -> Option<StackSlot> {
        match op {
            Operand::Value(v) => self.state.get_slot(v.0),
            _ => None,
        }
    }

    /// Emit AVX binary 256-bit op: load ymm0 from arg0 ptr, ymm1 from arg1 ptr,
    /// apply the given AVX instruction, store result ymm0 to dest_ptr.
    fn emit_avx_binary_256(&mut self, dest_ptr: &Value, args: &[Operand], avx_inst: &str) {
        // Load destination address FIRST into a dedicated register to avoid clobbering
        self.value_to_reg(dest_ptr, "rdx");
        // Load operands into separate registers
        self.operand_to_reg(&args[0], "rax");
        self.state.emit("    vmovupd (%rax), %ymm0");
        self.operand_to_reg(&args[1], "rcx");
        self.state.emit("    vmovupd (%rcx), %ymm1");
        // Perform vector operation
        self.state.emit_fmt(format_args!("    {} %ymm1, %ymm0, %ymm0", avx_inst));
        // Store result using dedicated destination register
        self.state.emit("    vmovupd %ymm0, (%rdx)");
    }
}
