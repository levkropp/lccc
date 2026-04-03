//! Emit AT&T x86-64 assembly from allocated MachInst sequences.
//!
//! After register allocation rewrites all MachReg::Vreg to MachReg::Phys,
//! this module pattern-matches each MachInst to produce text assembly.

use crate::backend::regalloc::PhysReg;
use crate::backend::common::AsmOutput;
use super::machinst::*;

/// Map a PhysReg to its 64-bit register name.
/// Extends the existing phys_reg_name to also handle rax(0) and rcx(7).
fn reg_name(reg: PhysReg) -> &'static str {
    match reg.0 {
        0 => "rax",
        1 => "rbx", 2 => "r12", 3 => "r13", 4 => "r14", 5 => "r15",
        6 => "rbp",
        7 => "rcx",
        10 => "r11", 11 => "r10", 12 => "r8", 13 => "r9",
        14 => "rdi", 15 => "rsi", 16 => "rdx",
        // XMM registers — shouldn't normally appear in GPR MachInst, but handle
        // gracefully for values that got XMM allocation from the main allocator.
        20 => "xmm2", 21 => "xmm3", 22 => "xmm4", 23 => "xmm5",
        24 => "xmm6", 25 => "xmm7",
        _ => "rax", // fallback for unexpected register IDs
    }
}

/// Map a PhysReg to its 32-bit sub-register name.
fn reg_name_32(reg: PhysReg) -> &'static str {
    match reg.0 {
        0 => "eax",
        1 => "ebx", 2 => "r12d", 3 => "r13d", 4 => "r14d", 5 => "r15d",
        6 => "ebp",
        7 => "ecx",
        10 => "r11d", 11 => "r10d", 12 => "r8d", 13 => "r9d",
        14 => "edi", 15 => "esi", 16 => "edx",
        _ => unreachable!("invalid machinst register index {}", reg.0),
    }
}

/// Map a PhysReg to its 16-bit sub-register name.
fn reg_name_16(reg: PhysReg) -> &'static str {
    match reg.0 {
        0 => "ax",
        1 => "bx", 2 => "r12w", 3 => "r13w", 4 => "r14w", 5 => "r15w",
        6 => "bp",
        7 => "cx",
        10 => "r11w", 11 => "r10w", 12 => "r8w", 13 => "r9w",
        14 => "di", 15 => "si", 16 => "dx",
        _ => unreachable!("invalid machinst register index {}", reg.0),
    }
}

/// Map a PhysReg to its 8-bit sub-register name.
fn reg_name_8(reg: PhysReg) -> &'static str {
    match reg.0 {
        0 => "al",
        1 => "bl", 2 => "r12b", 3 => "r13b", 4 => "r14b", 5 => "r15b",
        6 => "bpl",
        7 => "cl",
        10 => "r11b", 11 => "r10b", 12 => "r8b", 13 => "r9b",
        14 => "dil", 15 => "sil", 16 => "dl",
        _ => unreachable!("invalid machinst register index {}", reg.0),
    }
}

/// Get the register name at a given operand size.
fn sized_reg_name(reg: PhysReg, size: OpSize) -> &'static str {
    match size {
        OpSize::S8 => reg_name_8(reg),
        OpSize::S16 => reg_name_16(reg),
        OpSize::S32 => reg_name_32(reg),
        OpSize::S64 => reg_name(reg),
    }
}

/// Format a MachReg as an AT&T register operand (%name).
fn fmt_reg(reg: &MachReg, size: OpSize) -> String {
    match reg {
        MachReg::Phys(r) => format!("%{}", sized_reg_name(*r, size)),
        MachReg::Vreg(id) => format!("%vreg{}", id), // shouldn't appear after allocation
    }
}

/// Format a MachOperand as AT&T assembly.
fn fmt_operand(op: &MachOperand, size: OpSize, out: &AsmOutput) -> String {
    match op {
        MachOperand::Reg(r) => fmt_reg(r, size),
        MachOperand::Imm(v) => format!("${}", v),
        MachOperand::Mem { base, offset } => {
            let base_name = fmt_reg(base, OpSize::S64);
            if *offset == 0 {
                format!("({})", base_name)
            } else {
                format!("{}({})", offset, base_name)
            }
        }
        MachOperand::MemIndex { base, index, scale, offset } => {
            let base_name = fmt_reg(base, OpSize::S64);
            let index_name = fmt_reg(index, OpSize::S64);
            if *offset == 0 {
                format!("({}, {}, {})", base_name, index_name, scale)
            } else {
                format!("{}({}, {}, {})", offset, base_name, index_name, scale)
            }
        }
        MachOperand::StackSlot(slot_offset) => {
            if out.use_rsp_addressing {
                let rsp_off = out.rsp_frame_size + slot_offset;
                format!("{}(%rsp)", rsp_off)
            } else {
                format!("{}(%rbp)", slot_offset)
            }
        }
        MachOperand::RipRel(sym) => format!("{}(%rip)", sym),
    }
}

/// ALU operation mnemonic.
fn alu_mnemonic(op: AluOp) -> &'static str {
    match op {
        AluOp::Add => "add",
        AluOp::Sub => "sub",
        AluOp::And => "and",
        AluOp::Or => "or",
        AluOp::Xor => "xor",
        AluOp::Imul => "imul",
    }
}

/// Shift operation mnemonic.
fn shift_mnemonic(op: ShiftOp, size: OpSize) -> &'static str {
    match (op, size) {
        (ShiftOp::Shl, OpSize::S32) => "shll",
        (ShiftOp::Shl, _) => "shlq",
        (ShiftOp::Shr, OpSize::S32) => "shrl",
        (ShiftOp::Shr, _) => "shrq",
        (ShiftOp::Sar, OpSize::S32) => "sarl",
        (ShiftOp::Sar, _) => "sarq",
    }
}

/// Condition code suffix for Jcc/SetCC/CMov.
fn cc_suffix(cc: CondCode) -> &'static str {
    match cc {
        CondCode::E => "e",
        CondCode::Ne => "ne",
        CondCode::L => "l",
        CondCode::Le => "le",
        CondCode::G => "g",
        CondCode::Ge => "ge",
        CondCode::B => "b",
        CondCode::Be => "be",
        CondCode::A => "a",
        CondCode::Ae => "ae",
    }
}

/// Emit a single allocated MachInst as AT&T assembly text.
pub fn emit_machinst(inst: &MachInst, out: &mut AsmOutput) {
    match inst {
        MachInst::Mov { src, dst, size } => {
            let suffix = size.suffix();
            let src_str = fmt_operand(src, *size, out);
            let dst_str = fmt_operand(dst, *size, out);
            // Special case: movq $0 → xorl (shorter encoding)
            if let MachOperand::Imm(0) = src {
                if let MachOperand::Reg(MachReg::Phys(r)) = dst {
                    let r32 = reg_name_32(*r);
                    out.emit_fmt(format_args!("    xorl %{}, %{}", r32, r32));
                    return;
                }
            }
            out.emit_fmt(format_args!("    mov{} {}, {}", suffix, src_str, dst_str));
        }

        MachInst::Alu { op, src, dst, size } => {
            let mnem = alu_mnemonic(*op);
            let suffix = size.suffix();
            let src_str = fmt_operand(src, *size, out);
            let dst_str = fmt_reg(dst, *size);
            out.emit_fmt(format_args!("    {}{} {}, {}", mnem, suffix, src_str, dst_str));
        }

        MachInst::Imul3 { imm, src, dst, size } => {
            let suffix = size.suffix();
            let src_str = fmt_reg(src, *size);
            let dst_str = fmt_reg(dst, *size);
            out.emit_fmt(format_args!("    imul{} ${}, {}, {}", suffix, imm, src_str, dst_str));
        }

        MachInst::Neg { dst, size } => {
            let suffix = size.suffix();
            let dst_str = fmt_reg(dst, *size);
            out.emit_fmt(format_args!("    neg{} {}", suffix, dst_str));
        }

        MachInst::Not { dst, size } => {
            let suffix = size.suffix();
            let dst_str = fmt_reg(dst, *size);
            out.emit_fmt(format_args!("    not{} {}", suffix, dst_str));
        }

        MachInst::Shift { op, amount, dst, size } => {
            let mnem = shift_mnemonic(*op, *size);
            let dst_str = fmt_reg(dst, *size);
            match amount {
                MachOperand::Imm(v) => {
                    out.emit_fmt(format_args!("    {} ${}, {}", mnem, v, dst_str));
                }
                _ => {
                    // Variable shift: amount must be in %cl
                    out.emit_fmt(format_args!("    {} %cl, {}", mnem, dst_str));
                }
            }
        }

        MachInst::Lea { base, index, offset, dst } => {
            let base_str = fmt_reg(base, OpSize::S64);
            let dst_str = fmt_reg(dst, OpSize::S64);
            if let Some((idx, scale)) = index {
                let idx_str = fmt_reg(idx, OpSize::S64);
                if *offset == 0 {
                    out.emit_fmt(format_args!("    leaq ({}, {}, {}), {}", base_str, idx_str, scale, dst_str));
                } else {
                    out.emit_fmt(format_args!("    leaq {}({}, {}, {}), {}", offset, base_str, idx_str, scale, dst_str));
                }
            } else if *offset == 0 {
                // lea (%base), %dst — just a mov
                out.emit_fmt(format_args!("    movq {}, {}", base_str, dst_str));
            } else {
                out.emit_fmt(format_args!("    leaq {}({}), {}", offset, base_str, dst_str));
            }
        }

        MachInst::Cqto { size } => {
            if *size == OpSize::S32 {
                out.emit("    cltd");
            } else {
                out.emit("    cqto");
            }
        }

        MachInst::XorRdx => {
            out.emit("    xorl %edx, %edx");
        }

        MachInst::Div { divisor, signed, size } => {
            let mnem = if *signed { "idiv" } else { "div" };
            let suffix = size.suffix();
            let div_str = fmt_operand(divisor, *size, out);
            out.emit_fmt(format_args!("    {}{} {}", mnem, suffix, div_str));
        }

        MachInst::Cmp { lhs, rhs, size } => {
            let suffix = size.suffix();
            // AT&T: cmp rhs, lhs (reversed from Intel)
            let rhs_str = fmt_operand(rhs, *size, out);
            let lhs_str = fmt_operand(lhs, *size, out);
            out.emit_fmt(format_args!("    cmp{} {}, {}", suffix, rhs_str, lhs_str));
        }

        MachInst::Test { lhs, rhs, size } => {
            let suffix = size.suffix();
            let rhs_str = fmt_operand(rhs, *size, out);
            let lhs_str = fmt_operand(lhs, *size, out);
            out.emit_fmt(format_args!("    test{} {}, {}", suffix, rhs_str, lhs_str));
        }

        MachInst::SetCC { cc, dst } => {
            let cc_str = cc_suffix(*cc);
            let dst_str = fmt_reg(dst, OpSize::S8);
            out.emit_fmt(format_args!("    set{} {}", cc_str, dst_str));
        }

        MachInst::Movzx { src, dst, from_size, to_size } => {
            let src_str = fmt_reg(src, *from_size);
            let dst_str = fmt_reg(dst, *to_size);
            // movzbl, movzwl, movzbq, movzwq
            let mnem = match (from_size, to_size) {
                (OpSize::S8, OpSize::S32) => "movzbl",
                (OpSize::S16, OpSize::S32) => "movzwl",
                (OpSize::S8, OpSize::S64) => "movzbq",
                (OpSize::S16, OpSize::S64) => "movzwq",
                _ => "movzbl", // fallback
            };
            out.emit_fmt(format_args!("    {} {}, {}", mnem, src_str, dst_str));
        }

        MachInst::Movsx { src, dst, from_size, to_size } => {
            let src_str = fmt_reg(src, *from_size);
            let dst_str = fmt_reg(dst, *to_size);
            let mnem = match (from_size, to_size) {
                (OpSize::S8, OpSize::S32) => "movsbl",
                (OpSize::S16, OpSize::S32) => "movswl",
                (OpSize::S8, OpSize::S64) => "movsbq",
                (OpSize::S16, OpSize::S64) => "movswq",
                (OpSize::S32, OpSize::S64) => "movslq",
                _ => "movslq", // fallback
            };
            out.emit_fmt(format_args!("    {} {}, {}", mnem, src_str, dst_str));
        }

        MachInst::Cmov { cc, src, dst, size } => {
            let cc_str = cc_suffix(*cc);
            let suffix = size.suffix();
            let src_str = fmt_operand(src, *size, out);
            let dst_str = fmt_reg(dst, *size);
            out.emit_fmt(format_args!("    cmov{}{} {}, {}", cc_str, suffix, src_str, dst_str));
        }

        MachInst::Jcc { cc, target } => {
            let cc_str = cc_suffix(*cc);
            out.emit_fmt(format_args!("    j{} {}", cc_str, target));
        }

        MachInst::Jmp { target } => {
            out.emit_fmt(format_args!("    jmp {}", target));
        }

        MachInst::Label(name) => {
            out.emit_fmt(format_args!("{}:", name));
        }

        MachInst::Call { target } => {
            out.emit_fmt(format_args!("    call {}", target));
        }

        MachInst::CallIndirect { reg } => {
            let reg_str = fmt_reg(reg, OpSize::S64);
            out.emit_fmt(format_args!("    call *{}", reg_str));
        }

        MachInst::Ret => {
            out.emit("    ret");
        }

        MachInst::Raw(text) => {
            out.emit(text);
        }
    }
}

/// Emit a sequence of allocated MachInsts as AT&T assembly.
pub fn emit_machinsts(insts: &[MachInst], out: &mut AsmOutput) {
    for inst in insts {
        emit_machinst(inst, out);
    }
}
