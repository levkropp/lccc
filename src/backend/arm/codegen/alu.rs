//! ArmCodegen: ALU operations (integer arithmetic, bitwise, unary).

use crate::ir::reexports::{IrBinOp, Operand, Value};
use crate::common::types::IrType;
use super::emit::{ArmCodegen, callee_saved_name, callee_saved_name_32, arm_alu_mnemonic, is_arm_fp_phys};

impl ArmCodegen {
    pub(super) fn emit_shifted_logical_impl(
        &mut self, shift_op: IrBinOp, shift_lhs: &Operand, shift_amount: &Operand,
        logical_op: IrBinOp, other: &Operand, dest: &Value, ty: IrType,
    ) {
        let use_32bit = matches!(ty, IrType::I32 | IrType::U32);
        let amount = Self::const_as_imm12(shift_amount)
            .expect("shifted logical fusion requires a constant shift");
        let width = if use_32bit { 32 } else { 64 };
        debug_assert!(amount < width);

        let mut materialize = |this: &mut Self, op: &Operand, scratch: &str| -> String {
            if let Some(reg) = this.operand_reg(op) {
                if use_32bit { callee_saved_name_32(reg).to_string() }
                else { callee_saved_name(reg).to_string() }
            } else {
                this.operand_to_x0(op);
                let acc = if use_32bit { "w0" } else { "x0" };
                this.state.emit_fmt(format_args!("    mov {}, {}", scratch, acc));
                scratch.to_string()
            }
        };
        let other_reg = materialize(self, other, if use_32bit { "w1" } else { "x1" });
        let shifted_reg = materialize(self, shift_lhs, if use_32bit { "w2" } else { "x2" });
        let output = if let Some(reg) = self.dest_reg(dest) {
            if use_32bit { callee_saved_name_32(reg) } else { callee_saved_name(reg) }
        } else if use_32bit { "w0" } else { "x0" };
        self.state.emit_fmt(format_args!("    {} {}, {}, {}, {} #{}",
            arm_alu_mnemonic(logical_op), output, other_reg, shifted_reg,
            arm_alu_mnemonic(shift_op), amount));
        self.state.reg_cache.invalidate_acc();
        if self.dest_reg(dest).is_none() {
            self.store_x0_to(dest);
        }
    }

    pub(super) fn emit_int_fused_mul_add_impl(
        &mut self, lhs: &Operand, rhs: &Operand, acc: &Operand, dest: &Value, ty: IrType,
    ) {
        let use_32bit = matches!(ty, IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 | IrType::I32 | IrType::U32);
        let mut materialize = |this: &mut Self, op: &Operand, scratch: &str| -> String {
            if let Some(reg) = this.operand_reg(op) {
                if use_32bit { callee_saved_name_32(reg).to_string() }
                else { callee_saved_name(reg).to_string() }
            } else {
                // Not register-assigned: load stack-homed values directly into
                // the scratch register, avoiding the ldr-into-x0 + mov detour.
                if let Operand::Value(v) = op {
                    if !this.state.is_alloca(v.0) {
                        if let Some(slot) = this.state.get_slot(v.0) {
                            this.emit_load_from_sp(scratch, slot.0, "ldr");
                            return scratch.to_string();
                        }
                    }
                }
                this.operand_to_x0(op);
                let acc_name = if use_32bit { "w0" } else { "x0" };
                this.state.emit_fmt(format_args!("    mov {}, {}", scratch, acc_name));
                scratch.to_string()
            }
        };
        let lhs_reg = materialize(self, lhs, if use_32bit { "w1" } else { "x1" });
        let rhs_reg = materialize(self, rhs, if use_32bit { "w2" } else { "x2" });
        let acc_reg = materialize(self, acc, if use_32bit { "w3" } else { "x3" });
        if let Some(dest_phys) = self.dest_reg(dest) {
            let output = if use_32bit { callee_saved_name_32(dest_phys) } else { callee_saved_name(dest_phys) };
            self.state.emit_fmt(format_args!("    madd {}, {}, {}, {}", output, lhs_reg, rhs_reg, acc_reg));
            self.state.reg_cache.invalidate_acc();
            return;
        }
        let output = if use_32bit { "w0" } else { "x0" };
        self.state.emit_fmt(format_args!("    madd {}, {}, {}, {}", output, lhs_reg, rhs_reg, acc_reg));
        self.store_x0_to(dest);
    }

    pub(super) fn emit_int_fused_mul_sub_impl(
        &mut self, lhs: &Operand, rhs: &Operand, acc: &Operand, dest: &Value, ty: IrType,
    ) {
        // a - b*c → msub dest, b, c, a (single instruction on AArch64).
        let use_32bit = matches!(ty, IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 | IrType::I32 | IrType::U32);
        let mut materialize = |this: &mut Self, op: &Operand, scratch: &str| -> String {
            if let Some(reg) = this.operand_reg(op) {
                if use_32bit { callee_saved_name_32(reg).to_string() }
                else { callee_saved_name(reg).to_string() }
            } else {
                if let Operand::Value(v) = op {
                    if !this.state.is_alloca(v.0) {
                        if let Some(slot) = this.state.get_slot(v.0) {
                            this.emit_load_from_sp(scratch, slot.0, "ldr");
                            return scratch.to_string();
                        }
                    }
                }
                this.operand_to_x0(op);
                let acc_name = if use_32bit { "w0" } else { "x0" };
                this.state.emit_fmt(format_args!("    mov {}, {}", scratch, acc_name));
                scratch.to_string()
            }
        };
        let lhs_reg = materialize(self, lhs, if use_32bit { "w1" } else { "x1" });
        let rhs_reg = materialize(self, rhs, if use_32bit { "w2" } else { "x2" });
        let acc_reg = materialize(self, acc, if use_32bit { "w3" } else { "x3" });
        if let Some(dest_phys) = self.dest_reg(dest) {
            let output = if use_32bit { callee_saved_name_32(dest_phys) } else { callee_saved_name(dest_phys) };
            self.state.emit_fmt(format_args!("    msub {}, {}, {}, {}", output, lhs_reg, rhs_reg, acc_reg));
            self.state.reg_cache.invalidate_acc();
            return;
        }
        let output = if use_32bit { "w0" } else { "x0" };
        self.state.emit_fmt(format_args!("    msub {}, {}, {}, {}", output, lhs_reg, rhs_reg, acc_reg));
        self.store_x0_to(dest);
    }

    pub(super) fn emit_float_neg_impl(&mut self, ty: IrType) {
        if ty == IrType::F32 {
            self.state.emit("    fmov s0, w0");
            self.state.emit("    fneg s0, s0");
            self.state.emit("    fmov w0, s0");
            self.state.emit("    mov w0, w0"); // zero-extend
        } else {
            self.state.emit("    fmov d0, x0");
            self.state.emit("    fneg d0, d0");
            self.state.emit("    fmov x0, d0");
        }
    }

    pub(super) fn emit_int_neg_impl(&mut self, _ty: IrType) {
        self.state.emit("    neg x0, x0");
    }

    pub(super) fn emit_int_not_impl(&mut self, _ty: IrType) {
        self.state.emit("    mvn x0, x0");
    }

    pub(super) fn emit_int_clz_impl(&mut self, ty: IrType) {
        if ty == IrType::I32 || ty == IrType::U32 {
            self.state.emit("    clz w0, w0");
        } else {
            self.state.emit("    clz x0, x0");
        }
    }

    pub(super) fn emit_int_ctz_impl(&mut self, ty: IrType) {
        if ty == IrType::I32 || ty == IrType::U32 {
            self.state.emit("    rbit w0, w0");
            self.state.emit("    clz w0, w0");
        } else {
            self.state.emit("    rbit x0, x0");
            self.state.emit("    clz x0, x0");
        }
    }

    pub(super) fn emit_int_bswap_impl(&mut self, ty: IrType) {
        if ty == IrType::I16 || ty == IrType::U16 {
            self.state.emit("    rev w0, w0");
            self.state.emit("    lsr w0, w0, #16");
        } else if ty == IrType::I32 || ty == IrType::U32 {
            self.state.emit("    rev w0, w0");
        } else {
            self.state.emit("    rev x0, x0");
        }
    }

    pub(super) fn emit_int_bitreverse_impl(&mut self, ty: IrType) {
        if ty == IrType::I32 || ty == IrType::U32 {
            self.state.emit("    rbit w0, w0");
        } else {
            self.state.emit("    rbit x0, x0");
        }
    }

    pub(super) fn emit_int_popcount_impl(&mut self, ty: IrType) {
        if ty == IrType::I32 || ty == IrType::U32 {
            self.state.emit("    fmov s0, w0");
        } else {
            self.state.emit("    fmov d0, x0");
        }
        self.state.emit("    cnt v0.8b, v0.8b");
        self.state.emit("    uaddlv h0, v0.8b");
        self.state.emit("    fmov w0, s0");
    }

    pub(super) fn emit_int_binop_impl(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        let use_32bit = ty == IrType::I32 || ty == IrType::U32;

        // Strength reduction: UDiv/URem by power-of-2 constant
        if let Some(shift) = Self::const_as_power_of_2(rhs) {
            if op == IrBinOp::UDiv {
                self.operand_to_x0(lhs);
                if use_32bit {
                    self.state.emit_fmt(format_args!("    lsr w0, w0, #{}", shift));
                } else {
                    self.state.emit_fmt(format_args!("    lsr x0, x0, #{}", shift));
                }
                self.store_x0_to(dest);
                return;
            }
            if op == IrBinOp::URem {
                self.operand_to_x0(lhs);
                let mask = (1u64 << shift) - 1;
                if use_32bit {
                    self.state.emit_fmt(format_args!("    and w0, w0, #{}", mask));
                } else {
                    self.state.emit_fmt(format_args!("    and x0, x0, #{}", mask));
                }
                self.store_x0_to(dest);
                return;
            }
        }

        // Register-direct path
        if let Some(dest_phys) = self.dest_reg(dest) {
            let dest_name = callee_saved_name(dest_phys);
            let dest_name_32 = callee_saved_name_32(dest_phys);

            let is_shift = matches!(op, IrBinOp::Shl | IrBinOp::AShr | IrBinOp::LShr);
            let is_simple_alu = matches!(op, IrBinOp::Add | IrBinOp::Sub | IrBinOp::And
                | IrBinOp::Or | IrBinOp::Xor | IrBinOp::Mul) || is_shift;
            if is_simple_alu {
                let mnemonic = arm_alu_mnemonic(op);
                let lhs_phys = self.operand_reg(lhs).filter(|r| !is_arm_fp_phys(*r));
                // Emit `op dest, lhs, rhs` with lhs either its assigned register
                // (three-operand form, no setup mov) or dest preloaded with lhs.
                macro_rules! emit3 {
                    ($rhs:expr) => {
                        if let Some(lp) = lhs_phys {
                            if use_32bit {
                                let l = callee_saved_name_32(lp);
                                self.state.emit_fmt(format_args!("    {} {}, {}, {}", mnemonic, dest_name_32, l, $rhs(lp, true)));
                            } else {
                                let l = callee_saved_name(lp);
                                self.state.emit_fmt(format_args!("    {} {}, {}, {}", mnemonic, dest_name, l, $rhs(lp, false)));
                            }
                        } else {
                            self.operand_to_callee_reg(lhs, dest_phys);
                            if use_32bit {
                                self.state.emit_fmt(format_args!("    {} {}, {}, {}", mnemonic, dest_name_32, dest_name_32, $rhs(dest_phys, true)));
                            } else {
                                self.state.emit_fmt(format_args!("    {} {}, {}, {}", mnemonic, dest_name, dest_name, $rhs(dest_phys, false)));
                            }
                        }
                    };
                }

                if matches!(op, IrBinOp::Add | IrBinOp::Sub) {
                    if let Some(imm) = Self::const_as_imm12(rhs) {
                        emit3!(|_, _| format!("#{}", imm));
                        self.state.reg_cache.invalidate_acc();
                        return;
                    }
                }

                if matches!(op, IrBinOp::And | IrBinOp::Or | IrBinOp::Xor) {
                    let width = if use_32bit { 32 } else { 64 };
                    if let Some(imm) = Self::const_as_logical_imm(rhs, width) {
                        emit3!(|_, _| format!("#{:#x}", imm));
                        self.state.reg_cache.invalidate_acc();
                        return;
                    }
                }

                // AArch64 encodes a constant shift directly in the instruction.
                // Keeping the value in its assigned register avoids several moves
                // through x0/x1/x2 for the common C bit-manipulation idiom.
                if is_shift {
                    if let Some(imm) = Self::const_as_imm12(rhs) {
                        let width = if use_32bit { 32 } else { 64 };
                        if imm < width {
                            emit3!(|_, _| format!("#{}", imm));
                            self.state.reg_cache.invalidate_acc();
                            return;
                        }
                    }
                }

                let rhs_phys = self.operand_reg(rhs).filter(|r| !is_arm_fp_phys(*r));
                if let Some(rp) = rhs_phys {
                    emit3!(|_, is32| if is32 { callee_saved_name_32(rp).to_string() } else { callee_saved_name(rp).to_string() });
                    self.state.reg_cache.invalidate_acc();
                    return;
                }
                // rhs not register-assigned: load it into x0 as scratch.
                // If lhs is also not register-assigned, its load into dest goes
                // through x0 too — so lhs must be loaded FIRST, otherwise it
                // clobbers the rhs already sitting in x0.
                if lhs_phys.is_none() {
                    self.operand_to_callee_reg(lhs, dest_phys);
                    self.operand_to_x0(rhs);
                    if use_32bit {
                        self.state.emit_fmt(format_args!("    {} {}, {}, w0", mnemonic, dest_name_32, dest_name_32));
                    } else {
                        self.state.emit_fmt(format_args!("    {} {}, {}, x0", mnemonic, dest_name, dest_name));
                    }
                    self.state.reg_cache.invalidate_acc();
                    return;
                }
                self.operand_to_x0(rhs);
                emit3!(|_, is32| if is32 { "w0".to_string() } else { "x0".to_string() });
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }

        // Immediate accumulator path for spilled/short-lived results.  The
        // previous fallback materialized every constant in x0, copied both
        // operands through x1/x2, then performed a register-register op.  GEP
        // scaling and loop increments commonly land here, so use AArch64's
        // immediate forms even when the result has no physical assignment.
        if matches!(op, IrBinOp::Add | IrBinOp::Sub) {
            if let Some(imm) = Self::const_as_imm12(rhs) {
                self.operand_to_x0(lhs);
                let mnemonic = arm_alu_mnemonic(op);
                if use_32bit {
                    self.state.emit_fmt(format_args!("    {} w0, w0, #{}", mnemonic, imm));
                } else {
                    self.state.emit_fmt(format_args!("    {} x0, x0, #{}", mnemonic, imm));
                }
                self.store_x0_to(dest);
                return;
            }
        }
        if matches!(op, IrBinOp::And | IrBinOp::Or | IrBinOp::Xor) {
            let width = if use_32bit { 32 } else { 64 };
            if let Some(imm) = Self::const_as_logical_imm(rhs, width) {
                self.operand_to_x0(lhs);
                let mnemonic = arm_alu_mnemonic(op);
                if use_32bit {
                    self.state.emit_fmt(format_args!("    {} w0, w0, #{:#x}", mnemonic, imm));
                } else {
                    self.state.emit_fmt(format_args!("    {} x0, x0, #{:#x}", mnemonic, imm));
                }
                self.store_x0_to(dest);
                return;
            }
        }
        if matches!(op, IrBinOp::Shl | IrBinOp::AShr | IrBinOp::LShr) {
            if let Some(imm) = Self::const_as_imm12(rhs) {
                let width = if use_32bit { 32 } else { 64 };
                if imm < width {
                    self.operand_to_x0(lhs);
                    let mnemonic = arm_alu_mnemonic(op);
                    if use_32bit {
                        self.state.emit_fmt(format_args!("    {} w0, w0, #{}", mnemonic, imm));
                    } else {
                        self.state.emit_fmt(format_args!("    {} x0, x0, #{}", mnemonic, imm));
                    }
                    self.store_x0_to(dest);
                    return;
                }
            }
        }

        // Fallback: accumulator path
        self.operand_to_x0(lhs);
        self.state.emit("    mov x1, x0");
        self.operand_to_x0(rhs);
        self.state.emit("    mov x2, x0");

        if use_32bit {
            match op {
                IrBinOp::Add => {
                    self.state.emit("    add w0, w1, w2");
                }
                IrBinOp::Sub => {
                    self.state.emit("    sub w0, w1, w2");
                }
                IrBinOp::Mul => {
                    self.state.emit("    mul w0, w1, w2");
                }
                IrBinOp::SDiv => {
                    self.state.emit("    sdiv w0, w1, w2");
                }
                IrBinOp::UDiv => self.state.emit("    udiv w0, w1, w2"),
                IrBinOp::SRem => {
                    self.state.emit("    sdiv w3, w1, w2");
                    self.state.emit("    msub w0, w3, w2, w1");
                }
                IrBinOp::URem => {
                    self.state.emit("    udiv w3, w1, w2");
                    self.state.emit("    msub w0, w3, w2, w1");
                }
                IrBinOp::And => self.state.emit("    and w0, w1, w2"),
                IrBinOp::Or => self.state.emit("    orr w0, w1, w2"),
                IrBinOp::Xor => self.state.emit("    eor w0, w1, w2"),
                IrBinOp::Shl => {
                    self.state.emit("    lsl w0, w1, w2");
                }
                IrBinOp::AShr => {
                    self.state.emit("    asr w0, w1, w2");
                }
                IrBinOp::LShr => self.state.emit("    lsr w0, w1, w2"),
            }
        } else {
            match op {
                IrBinOp::Add => self.state.emit("    add x0, x1, x2"),
                IrBinOp::Sub => self.state.emit("    sub x0, x1, x2"),
                IrBinOp::Mul => self.state.emit("    mul x0, x1, x2"),
                IrBinOp::SDiv => self.state.emit("    sdiv x0, x1, x2"),
                IrBinOp::UDiv => self.state.emit("    udiv x0, x1, x2"),
                IrBinOp::SRem => {
                    self.state.emit("    sdiv x3, x1, x2");
                    self.state.emit("    msub x0, x3, x2, x1");
                }
                IrBinOp::URem => {
                    self.state.emit("    udiv x3, x1, x2");
                    self.state.emit("    msub x0, x3, x2, x1");
                }
                IrBinOp::And => self.state.emit("    and x0, x1, x2"),
                IrBinOp::Or => self.state.emit("    orr x0, x1, x2"),
                IrBinOp::Xor => self.state.emit("    eor x0, x1, x2"),
                IrBinOp::Shl => self.state.emit("    lsl x0, x1, x2"),
                IrBinOp::AShr => self.state.emit("    asr x0, x1, x2"),
                IrBinOp::LShr => self.state.emit("    lsr x0, x1, x2"),
            }
        }

        self.store_x0_to(dest);
    }

    pub(super) fn emit_copy_i128_impl(&mut self, dest: &Value, src: &Operand) {
        self.operand_to_x0_x1(src);
        self.store_x0_x1_to(dest);
    }
}
