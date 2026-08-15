//! AArch64 NEON/SIMD intrinsic emission and F128 (quad-precision) soft-float helpers.
//!
//! NEON intrinsics: SSE-equivalent operations via 128-bit NEON instructions.
//! F128: IEEE 754 binary128 via compiler-rt/libgcc soft-float libcalls.

use crate::ir::reexports::{IntrinsicOp, Operand, Value};
use crate::common::types::IrType;
use super::emit::{ArmCodegen, arm_fp_name, arm_vector_name, callee_saved_name, is_arm_fp_phys};

impl ArmCodegen {
    /// If `val_id` is register-allocated to a NEON register, return its 128-bit
    /// `vN` name (allocator IDs 40..55 map to v16..v31).
    fn assigned_vector_reg(&self, val_id: u32) -> Option<String> {
        self.reg_assignments.get(&val_id).and_then(|&phys| {
            if is_arm_fp_phys(phys) { Some(arm_vector_name(phys)) } else { None }
        })
    }

    /// Make a 128-bit vector operand available in a NEON register and return
    /// its `vN` name: the assigned register when the value is register-allocated,
    /// otherwise the value is loaded from its stack slot into `qreg`.
    fn load_vector_value_128(&mut self, op: &Operand, qreg: &str) -> String {
        if let Operand::Value(v) = op {
            if let Some(name) = self.assigned_vector_reg(v.0) {
                return name;
            }
            if let Some(slot) = self.state.get_slot(v.0) {
                self.emit_load_from_sp(qreg, slot.0, "ldr");
            }
        }
        qreg.replacen('q', "v", 1)
    }

    fn store_vector_value_128(&mut self, dest: &Value, qreg: &str) {
        self.state.vector_values.insert(dest.0);
        if let Some(name) = self.assigned_vector_reg(dest.0) {
            let src = qreg.replacen('q', "v", 1);
            self.state.emit_fmt(format_args!("    mov {}.16b, {}.16b", name, src));
        } else if let Some(slot) = self.state.get_slot(dest.0) {
            self.emit_store_to_sp(qreg, slot.0, "str");
        }
    }

    pub(super) fn emit_neon_binary_128(&mut self, dest_ptr: &Value, args: &[Operand], neon_inst: &str) {
        // Load first 128-bit operand pointer into x0, then load q0
        self.operand_to_x0(&args[0]);
        self.state.emit("    ldr q0, [x0]");
        // Load second 128-bit operand pointer into x1, then load q1
        match &args[1] {
            Operand::Value(v) => {
                if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        self.emit_alloca_addr("x1", v.0, slot.0);
                    } else {
                        self.emit_load_from_sp("x1", slot.0, "ldr");
                    }
                }
            }
            Operand::Const(_) => {
                self.operand_to_x0(&args[1]);
                self.state.emit("    mov x1, x0");
            }
        }
        self.state.emit("    ldr q1, [x1]");
        // Apply the binary NEON operation
        self.state.emit_fmt(format_args!("    {} v0.16b, v0.16b, v1.16b", neon_inst));
        // Store result to dest_ptr
        self.load_ptr_to_reg(dest_ptr, "x0");
        self.state.emit("    str q0, [x0]");
    }

    /// Store a scalar result from x0 (or w0) into the dest stack slot.
    fn store_scalar_dest(&mut self, dest: &Option<Value>, reg: &str) {
        if let Some(d) = dest {
            if let Some(slot) = self.state.get_slot(d.0) {
                self.emit_store_to_sp(reg, slot.0, "str");
            }
        }
    }

    /// Emit a unary F64 operation: apply `op_inst` with the operand in d0, store result.
    /// Register-aware: reads/writes allocated FP registers directly when present.
    fn emit_f64_unary_neon(&mut self, dest: &Option<Value>, args: &[Operand], op_inst: &str) {
        // When the dest has an FP register, compute straight into it.
        let dest_reg = dest.and_then(|d| {
            self.reg_assignments.get(&d.0).and_then(|&phys| {
                if is_arm_fp_phys(phys) { Some(arm_fp_name(phys, IrType::F64)) } else { None }
            })
        });
        if let Some(dreg) = dest_reg {
            let src = self.float_operand_reg(&args[0], IrType::F64, "d0");
            self.state.emit_fmt(format_args!("    {} {}, {}", op_inst, dreg, src));
            return;
        }
        self.float_operand_to_reg(&args[0], IrType::F64, "d0");
        self.state.emit_fmt(format_args!("    {} d0, d0", op_inst));
        if let Some(d) = dest {
            self.store_float_reg(d, IrType::F64, "d0");
        }
    }

    /// Emit a unary F32 operation: apply `op_inst` with the operand in s0, store result.
    fn emit_f32_unary_neon(&mut self, dest: &Option<Value>, args: &[Operand], op_inst: &str) {
        self.float_operand_to_reg(&args[0], IrType::F32, "s0");
        self.state.emit_fmt(format_args!("    {} s0, s0", op_inst));
        if let Some(d) = dest {
            self.store_float_reg(d, IrType::F32, "s0");
        }
    }

    /// Emit a non-temporal store: load value from args[0], store to dest_ptr.
    fn emit_nontemporal_store(&mut self, dest_ptr: &Option<Value>, args: &[Operand], save_reg: &str, val_reg: &str) {
        if let Some(ptr) = dest_ptr {
            self.operand_to_x0(&args[0]);
            self.state.emit_fmt(format_args!("    mov {}, {}", save_reg, val_reg));
            self.load_ptr_to_reg(ptr, "x0");
            self.state.emit_fmt(format_args!("    str {}, [x0]", save_reg));
        }
    }

    pub(super) fn emit_intrinsic_arm(&mut self, dest: &Option<Value>, op: &IntrinsicOp, dest_ptr: &Option<Value>, args: &[Operand]) {
        match op {
            IntrinsicOp::Lfence | IntrinsicOp::Mfence => {
                self.state.emit("    dmb ish");
            }
            IntrinsicOp::Sfence => {
                self.state.emit("    dmb ishst");
            }
            IntrinsicOp::Pause => {
                self.state.emit("    yield");
            }
            IntrinsicOp::Clflush => {
                // ARM has no direct clflush; use dc civac (clean+invalidate to PoC)
                self.operand_to_x0(&args[0]);
                self.state.emit("    dc civac, x0");
            }
            IntrinsicOp::Movnti => {
                self.emit_nontemporal_store(dest_ptr, args, "w9", "w0");
            }
            IntrinsicOp::Movnti64 => {
                self.emit_nontemporal_store(dest_ptr, args, "x9", "x0");
            }
            IntrinsicOp::Movntdq | IntrinsicOp::Movntpd => {
                // Non-temporal 128-bit store: dest_ptr = target, args[0] = source ptr
                if let Some(ptr) = dest_ptr {
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    ldr q0, [x0]");
                    self.load_ptr_to_reg(ptr, "x0");
                    self.state.emit("    str q0, [x0]");
                }
            }
            IntrinsicOp::Loaddqu => {
                // Load 128-bit unaligned: args[0] = source ptr, dest_ptr = result storage
                if let Some(dptr) = dest_ptr {
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    ldr q0, [x0]");
                    self.load_ptr_to_reg(dptr, "x0");
                    self.state.emit("    str q0, [x0]");
                }
            }
            IntrinsicOp::Storedqu => {
                // Store 128-bit unaligned: dest_ptr = target ptr, args[0] = source data ptr
                if let Some(ptr) = dest_ptr {
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    ldr q0, [x0]");
                    self.load_ptr_to_reg(ptr, "x0");
                    self.state.emit("    str q0, [x0]");
                }
            }
            IntrinsicOp::Pcmpeqb128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_neon_binary_128(dptr, args, "cmeq");
                }
            }
            IntrinsicOp::Pcmpeqd128 => {
                if let Some(dptr) = dest_ptr {
                    // For 32-bit lane equality, load q regs, use cmeq with .4s arrangement
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    ldr q0, [x0]");
                    if let Operand::Value(v) = &args[1] {
                        self.load_ptr_to_reg(v, "x1");
                    } else {
                        self.operand_to_x0(&args[1]);
                        self.state.emit("    mov x1, x0");
                    }
                    self.state.emit("    ldr q1, [x1]");
                    self.state.emit("    cmeq v0.4s, v0.4s, v1.4s");
                    self.load_ptr_to_reg(dptr, "x0");
                    self.state.emit("    str q0, [x0]");
                }
            }
            IntrinsicOp::Psubusb128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_neon_binary_128(dptr, args, "uqsub");
                }
            }
            IntrinsicOp::Psubsb128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_neon_binary_128(dptr, args, "sqsub");
                }
            }
            IntrinsicOp::Por128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_neon_binary_128(dptr, args, "orr");
                }
            }
            IntrinsicOp::Pand128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_neon_binary_128(dptr, args, "and");
                }
            }
            IntrinsicOp::Pxor128 => {
                if let Some(dptr) = dest_ptr {
                    self.emit_neon_binary_128(dptr, args, "eor");
                }
            }
            IntrinsicOp::Pmovmskb128 => {
                // Extract the high bit of each byte in a 128-bit vector into a 16-bit mask.
                // NEON has no pmovmskb equivalent, so we use a multi-step sequence:
                //   1. Load 128-bit data into v0
                //   2. Shift right each byte by 7 to isolate the sign bit
                //   3. Multiply by power-of-2 bit positions, then add across lanes
                self.operand_to_x0(&args[0]);
                self.state.emit("    ldr q0, [x0]");
                self.state.emit("    ushr v0.16b, v0.16b, #7");
                // Load bit position constants: [1,2,4,8,16,32,64,128] repeated
                self.state.emit("    movz x0, #0x0201");
                self.state.emit("    movk x0, #0x0804, lsl #16");
                self.state.emit("    movk x0, #0x2010, lsl #32");
                self.state.emit("    movk x0, #0x8040, lsl #48");
                self.state.emit("    fmov d1, x0");
                self.state.emit("    mov v1.d[1], x0");
                self.state.emit("    mul v0.16b, v0.16b, v1.16b");
                // Split and sum each half
                self.state.emit("    ext v1.16b, v0.16b, v0.16b, #8");
                self.state.emit("    addv b0, v0.8b");
                self.state.emit("    umov w0, v0.b[0]");
                self.state.emit("    addv b1, v1.8b");
                self.state.emit("    umov w1, v1.b[0]");
                self.state.emit("    orr w0, w0, w1, lsl #8");
                self.store_scalar_dest(dest, "x0");
            }
            IntrinsicOp::SetEpi8 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    dup v0.16b, w0");
                    self.load_ptr_to_reg(dptr, "x0");
                    self.state.emit("    str q0, [x0]");
                }
            }
            IntrinsicOp::SetEpi32 => {
                if let Some(dptr) = dest_ptr {
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    dup v0.4s, w0");
                    self.load_ptr_to_reg(dptr, "x0");
                    self.state.emit("    str q0, [x0]");
                }
            }
            IntrinsicOp::Crc32_8 | IntrinsicOp::Crc32_16
            | IntrinsicOp::Crc32_32 | IntrinsicOp::Crc32_64 => {
                let is_64 = matches!(op, IntrinsicOp::Crc32_64);
                let (save_reg, crc_inst) = match op {
                    IntrinsicOp::Crc32_8  => ("w9", "crc32cb w9, w9, w0"),
                    IntrinsicOp::Crc32_16 => ("w9", "crc32ch w9, w9, w0"),
                    IntrinsicOp::Crc32_32 => ("w9", "crc32cw w9, w9, w0"),
                    IntrinsicOp::Crc32_64 => ("x9", "crc32cx w9, w9, x0"),
                    _ => unreachable!(),
                };
                self.operand_to_x0(&args[0]);
                self.state.emit_fmt(format_args!("    mov {}, {}", save_reg, if is_64 { "x0" } else { "w0" }));
                self.operand_to_x0(&args[1]);
                self.state.emit_fmt(format_args!("    {}", crc_inst));
                self.state.emit("    mov x0, x9");
                self.store_scalar_dest(dest, "x0");
            }
            IntrinsicOp::FrameAddress => {
                self.state.emit("    mov x0, x29");
                self.store_scalar_dest(dest, "x0");
            }
            IntrinsicOp::ReturnAddress => {
                // x30 (lr) is clobbered by bl instructions, so read from stack
                self.state.emit("    ldr x0, [x29, #8]");
                self.store_scalar_dest(dest, "x0");
            }
            IntrinsicOp::ThreadPointer => {
                // __builtin_thread_pointer(): read TLS base from tpidr_el0
                self.state.emit("    mrs x0, tpidr_el0");
                self.store_scalar_dest(dest, "x0");
            }
            IntrinsicOp::SqrtF64 => self.emit_f64_unary_neon(dest, args, "fsqrt"),
            IntrinsicOp::SqrtF32 => self.emit_f32_unary_neon(dest, args, "fsqrt"),
            IntrinsicOp::FabsF64 => self.emit_f64_unary_neon(dest, args, "fabs"),
            IntrinsicOp::FabsF32 => self.emit_f32_unary_neon(dest, args, "fabs"),
            // x86-specific SSE/AES-NI/CLMUL intrinsics - these are x86-only and should
            // not appear in ARM codegen in practice. Cross-compiled code that conditionally
            // uses these behind #ifdef __x86_64__ will have the calls dead-code eliminated.
            // TODO: consider emitting a runtime trap instead of silent zeros
            IntrinsicOp::Aesenc128 | IntrinsicOp::Aesenclast128
            | IntrinsicOp::Aesdec128 | IntrinsicOp::Aesdeclast128
            | IntrinsicOp::Aesimc128 | IntrinsicOp::Aeskeygenassist128
            | IntrinsicOp::Pclmulqdq128
            | IntrinsicOp::Pslldqi128 | IntrinsicOp::Psrldqi128
            | IntrinsicOp::Psllqi128 | IntrinsicOp::Psrlqi128
            | IntrinsicOp::Pshufd128 | IntrinsicOp::Loadldi128
            | IntrinsicOp::Paddw128 | IntrinsicOp::Psubw128
            | IntrinsicOp::Pmulhw128 | IntrinsicOp::Pmaddwd128
            | IntrinsicOp::Pcmpgtw128 | IntrinsicOp::Pcmpgtb128
            | IntrinsicOp::Psllwi128 | IntrinsicOp::Psrlwi128
            | IntrinsicOp::Psrawi128 | IntrinsicOp::Psradi128
            | IntrinsicOp::Pslldi128 | IntrinsicOp::Psrldi128
            | IntrinsicOp::Paddd128 | IntrinsicOp::Psubd128
            | IntrinsicOp::Packssdw128 | IntrinsicOp::Packsswb128 | IntrinsicOp::Packuswb128
            | IntrinsicOp::Punpcklbw128 | IntrinsicOp::Punpckhbw128
            | IntrinsicOp::Punpcklwd128 | IntrinsicOp::Punpckhwd128
            | IntrinsicOp::SetEpi16 | IntrinsicOp::Pinsrw128
            | IntrinsicOp::Pextrw128 | IntrinsicOp::Storeldi128
            | IntrinsicOp::Cvtsi128Si32 | IntrinsicOp::Cvtsi32Si128
            | IntrinsicOp::Cvtsi128Si64
            | IntrinsicOp::Pshuflw128 | IntrinsicOp::Pshufhw128
            | IntrinsicOp::Pinsrd128 | IntrinsicOp::Pextrd128
            | IntrinsicOp::Pinsrb128 | IntrinsicOp::Pextrb128
            | IntrinsicOp::Pinsrq128 | IntrinsicOp::Pextrq128
            | IntrinsicOp::FmaF64x4
            | IntrinsicOp::FmaF64x4Hoisted | IntrinsicOp::FmaF64x4SIB
            | IntrinsicOp::LoadF64x4 | IntrinsicOp::LoadF64x2
            | IntrinsicOp::LoadI32x8 | IntrinsicOp::LoadI32x4
            | IntrinsicOp::AddF64x4 | IntrinsicOp::AddF64x2
            | IntrinsicOp::MulF64x4 | IntrinsicOp::MulF64x2
            | IntrinsicOp::AddI32x8 | IntrinsicOp::AddI32x4
            | IntrinsicOp::HorizontalAddF64x4 | IntrinsicOp::HorizontalAddF64x2
            | IntrinsicOp::HorizontalAddI32x8 | IntrinsicOp::HorizontalAddI32x4 => {
                // x86-only: zero dest if present
                if let Some(dptr) = dest_ptr {
                    if let Some(slot) = self.state.get_slot(dptr.0) {
                        self.state.emit_fmt(format_args!("    add x9, sp, #{}", slot.0));
                        self.state.emit("    stp xzr, xzr, [x9]");
                    }
                }
            }

            IntrinsicOp::FmaF64x2 => {
                if let Some(c_ptr) = dest_ptr {
                    // Preserve all three addresses before using SIMD registers;
                    // operand_to_x0 may itself need x9 for an indirect value.
                    self.operand_to_x0(&args[0]);
                    self.state.emit("    mov x10, x0");
                    self.operand_to_x0(&args[1]);
                    self.state.emit("    mov x11, x0");
                    self.operand_to_x0(&Operand::Value(*c_ptr));
                    self.state.emit("    mov x12, x0");
                    self.state.emit("    ldr d0, [x10]");
                    self.state.emit("    ldr q1, [x11]");
                    self.state.emit("    ldr q2, [x12]");
                    self.state.emit("    dup v0.2d, v0.d[0]");
                    self.state.emit("    fmla v2.2d, v1.2d, v0.2d");
                    self.state.emit("    str q2, [x12]");
                }
            }

            IntrinsicOp::BroadcastLoadF64 => {
                self.operand_to_x0(&args[0]);
                self.state.emit("    ldr d15, [x0]");
                self.state.emit("    dup v15.2d, v15.d[0]");
            }

            IntrinsicOp::FmaF64x2Hoisted => {
                if let Some(c_ptr) = dest_ptr {
                    let b_phys = self.operand_reg(&args[0]).filter(|r| !is_arm_fp_phys(*r));
                    let c_phys = self.operand_reg(&Operand::Value(*c_ptr)).filter(|r| !is_arm_fp_phys(*r));
                    let b_addr = b_phys.map(callee_saved_name).unwrap_or("x11");
                    let c_addr = c_phys.map(callee_saved_name).unwrap_or("x12");
                    if b_phys.is_none() {
                        self.operand_to_x0(&args[0]);
                        self.state.emit("    mov x11, x0");
                    }
                    if c_phys.is_none() {
                        self.operand_to_x0(&Operand::Value(*c_ptr));
                        self.state.emit("    mov x12, x0");
                    }
                    self.state.emit_fmt(format_args!("    ldr q1, [{}]", b_addr));
                    self.state.emit_fmt(format_args!("    ldr q2, [{}]", c_addr));
                    self.state.emit("    fmla v2.2d, v1.2d, v15.2d");
                    self.state.emit_fmt(format_args!("    str q2, [{}]", c_addr));
                    self.state.emit_fmt(format_args!("    ldr q1, [{}, #16]", b_addr));
                    self.state.emit_fmt(format_args!("    ldr q2, [{}, #16]", c_addr));
                    self.state.emit("    fmla v2.2d, v1.2d, v15.2d");
                    self.state.emit_fmt(format_args!("    str q2, [{}, #16]", c_addr));
                }
            }

            IntrinsicOp::VecZeroI64x2 => {
                if let Some(d) = dest {
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    eor {0}.16b, {0}.16b, {0}.16b", name));
                    } else {
                        self.state.emit("    eor v0.16b, v0.16b, v0.16b");
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecLoadWidenI32ToI64x2 => {
                if let Some(d) = dest {
                    let base_phys = self.operand_reg(&args[0]).filter(|r| !is_arm_fp_phys(*r));
                    let addr = base_phys.map(callee_saved_name).unwrap_or("x10");
                    if base_phys.is_none() {
                        self.operand_to_x0(&args[0]);
                        self.state.emit("    mov x10, x0");
                    }
                    self.state.emit_fmt(format_args!("    ldr d0, [{}]", addr));
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    sxtl {}.2d, v0.2s", name));
                    } else {
                        self.state.emit("    sxtl v0.2d, v0.2s");
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecAddI64x2 | IntrinsicOp::VecMulI64x2 => {
                if let Some(d) = dest {
                    let a = self.load_vector_value_128(&args[0], "q0");
                    let b = self.load_vector_value_128(&args[1], "q1");
                    let mnemonic = if *op == IntrinsicOp::VecAddI64x2 { "add" } else { "mul" };
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    {} {}.2d, {}.2d, {}.2d", mnemonic, name, a, b));
                    } else {
                        self.state.emit_fmt(format_args!("    {} v0.2d, {}.2d, {}.2d", mnemonic, a, b));
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecHorizontalAddI64x2 => {
                let a = self.load_vector_value_128(&args[0], "q0");
                self.state.emit_fmt(format_args!("    addp d0, {}.2d", a));
                self.state.emit("    fmov x0, d0");
                if let Some(d) = dest { self.store_x0_to(d); }
            }

            // Register-based vector operations (NEON 128-bit).
            // Two-wide F64 (2×.2d) and four-wide I32 (4×.4s) for reductions.
            IntrinsicOp::VecLoadF64x2 | IntrinsicOp::VecLoadI32x4 => {
                if let Some(d) = dest {
                    let base_phys = self.operand_reg(&args[0]).filter(|r| !is_arm_fp_phys(*r));
                    let addr = base_phys.map(callee_saved_name).unwrap_or("x10");
                    if base_phys.is_none() {
                        self.operand_to_x0(&args[0]);
                        self.state.emit("    mov x10, x0");
                    }
                    self.state.emit_fmt(format_args!("    ldr q0, [{}]", addr));
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    mov {}.16b, v0.16b", name));
                    } else {
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecZeroF64x2 | IntrinsicOp::VecZeroI32x4 => {
                if let Some(d) = dest {
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    eor {0}.16b, {0}.16b, {0}.16b", name));
                    } else {
                        self.state.emit("    eor v0.16b, v0.16b, v0.16b");
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecAddF64x2 | IntrinsicOp::VecMulF64x2
            | IntrinsicOp::VecAddI32x4 | IntrinsicOp::VecMulI32x4 => {
                if let Some(d) = dest {
                    let a = self.load_vector_value_128(&args[0], "q0");
                    let b = self.load_vector_value_128(&args[1], "q1");
                    let (mnemonic, suffix) = match op {
                        IntrinsicOp::VecAddF64x2 => ("fadd", "2d"),
                        IntrinsicOp::VecMulF64x2 => ("fmul", "2d"),
                        IntrinsicOp::VecAddI32x4 => ("add", "4s"),
                        _ => ("mul", "4s"),
                    };
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    {} {}.{}, {}.{}, {}.{}", mnemonic, name, suffix, a, suffix, b, suffix));
                    } else {
                        self.state.emit_fmt(format_args!("    {} v0.{}, {}.{}, {}.{}", mnemonic, suffix, a, suffix, b, suffix));
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecBroadcastI32x4 => {
                if let Some(d) = dest {
                    self.operand_to_x0(&args[0]);
                    if let Some(name) = self.assigned_vector_reg(d.0) {
                        self.state.vector_values.insert(d.0);
                        self.state.emit_fmt(format_args!("    dup {}.4s, w0", name));
                    } else {
                        self.state.emit("    dup v0.4s, w0");
                        self.store_vector_value_128(d, "q0");
                    }
                }
            }

            IntrinsicOp::VecStoreI32x4 => {
                // Store a 4×I32 vector to dest_ptr.
                if let Some(ptr) = dest_ptr {
                    let src = self.load_vector_value_128(&args[0], "q0");
                    let base_phys = self.operand_reg(&Operand::Value(*ptr)).filter(|r| !is_arm_fp_phys(*r));
                    let addr = base_phys.map(callee_saved_name).unwrap_or("x10");
                    if base_phys.is_none() {
                        self.operand_to_x0(&Operand::Value(*ptr));
                        self.state.emit("    mov x10, x0");
                    }
                    self.state.emit_fmt(format_args!("    str {}, [{}]", src.replacen('v', "q", 1), addr));
                }
            }

            IntrinsicOp::VecHorizontalAddF64x2 => {
                let a = self.load_vector_value_128(&args[0], "q0");
                self.state.emit_fmt(format_args!("    faddp d0, {}.2d", a));
                if let Some(d) = dest {
                    self.store_float_reg(d, IrType::F64, "d0");
                }
            }

            IntrinsicOp::VecHorizontalAddI32x4 => {
                let a = self.load_vector_value_128(&args[0], "q0");
                self.state.emit_fmt(format_args!("    addv s0, {}.4s", a));
                self.state.emit("    fmov w0, s0");
                if let Some(d) = dest { self.store_x0_to(d); }
            }

            // Not-yet-implemented register-based vector intrinsics for ARM.
            IntrinsicOp::VecLoadF64x4 | IntrinsicOp::VecLoadI32x8 |
            IntrinsicOp::VecAddF64x4 | IntrinsicOp::VecMulF64x4 |
            IntrinsicOp::VecAddI32x8 |
            IntrinsicOp::VecHorizontalAddF64x4 |
            IntrinsicOp::VecHorizontalAddI32x8 |
            IntrinsicOp::VecZeroF64x4 | IntrinsicOp::VecZeroI32x8 => {
                unimplemented!("4-wide/AVX vector intrinsics not implemented for ARM");
            }
        }
    }

    // ---- F128 (long double / IEEE quad precision) soft-float helpers ----
    //
    // On AArch64, long double is IEEE 754 binary128 (16 bytes).
    // Hardware has no quad-precision FP ops, so we use compiler-rt/libgcc soft-float:
    //   Comparison: __eqtf2, __lttf2, __letf2, __gttf2, __getf2
    //   Arithmetic: __addtf3, __subtf3, __multf3, __divtf3
    //   Conversion: __extenddftf2 (f64->f128), __trunctfdf2 (f128->f64)
    // ABI: f128 passed/returned in Q registers (q0, q1). Int result in w0/x0.

}
