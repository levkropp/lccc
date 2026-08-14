//! ArmCodegen: memory operations (load, store, memcpy, GEP, stack).

use crate::ir::reexports::{Operand, Value};
use crate::common::types::IrType;
use crate::backend::state::{StackSlot, SlotAddr};
use super::emit::{ArmCodegen, arm_fp_name, callee_saved_name, is_arm_fp_phys};

impl ArmCodegen {
    // ---- Store/Load overrides ----

    pub(super) fn emit_store_impl(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        if ty == IrType::F128 {
            crate::backend::f128_softfloat::f128_emit_store(self, val, ptr);
            return;
        }
        if matches!(ty, IrType::F32 | IrType::F64) {
            if let Operand::Value(v) = val {
                if let Some(&phys) = self.reg_assignments.get(&v.0) {
                    if is_arm_fp_phys(phys) {
                        let fp = arm_fp_name(phys, ty);
                        if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                            match addr {
                                SlotAddr::OverAligned(slot, id) => {
                                    self.emit_alloca_aligned_addr_impl(slot, id);
                                    self.state.emit_fmt(format_args!("    str {}, [x9]", fp));
                                }
                                SlotAddr::Direct(slot) => self.emit_store_to_sp(&fp, slot.0, "str"),
                                SlotAddr::Indirect(slot) => {
                                    self.emit_load_ptr_from_slot_impl(slot, ptr.0);
                                    self.state.emit_fmt(format_args!("    str {}, [x9]", fp));
                                }
                            }
                            return;
                        }
                    }
                }
            }
        }
        crate::backend::traits::emit_store_default(self, val, ptr, ty);
    }

    pub(super) fn emit_load_impl(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if ty == IrType::F128 {
            crate::backend::f128_softfloat::f128_emit_load(self, dest, ptr);
            return;
        }
        if matches!(ty, IrType::F32 | IrType::F64) {
            if let Some(&phys) = self.reg_assignments.get(&dest.0) {
                if is_arm_fp_phys(phys) {
                    let fp = arm_fp_name(phys, ty);
                    if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                        match addr {
                            SlotAddr::OverAligned(slot, id) => {
                                self.emit_alloca_aligned_addr_impl(slot, id);
                                self.state.emit_fmt(format_args!("    ldr {}, [x9]", fp));
                            }
                            SlotAddr::Direct(slot) => self.emit_load_from_sp(&fp, slot.0, "ldr"),
                            SlotAddr::Indirect(slot) => {
                                self.emit_load_ptr_from_slot_impl(slot, ptr.0);
                                self.state.emit_fmt(format_args!("    ldr {}, [x9]", fp));
                            }
                        }
                        return;
                    }
                }
            }
        }
        crate::backend::traits::emit_load_default(self, dest, ptr, ty);
    }

    pub(super) fn emit_store_with_const_offset_impl(&mut self, val: &Operand, base: &Value, offset: i64, ty: IrType) {
        if ty == IrType::F128 {
            crate::backend::f128_softfloat::f128_emit_store_with_offset(self, val, base, offset);
            return;
        }
        // FP value with a register assignment: store the d/s register directly,
        // folding the constant offset into the addressing mode when encodable.
        if matches!(ty, IrType::F32 | IrType::F64) {
            if let Operand::Value(v) = val {
                if let Some(&phys) = self.reg_assignments.get(&v.0) {
                    if is_arm_fp_phys(phys) {
                        let fp = arm_fp_name(phys, ty);
                        let addr = self.state.resolve_slot_addr(base.0);
                        if let Some(addr) = addr {
                            let scale: i64 = if ty == IrType::F32 { 4 } else { 8 };
                            match addr {
                                SlotAddr::OverAligned(slot, id) => {
                                    self.emit_alloca_aligned_addr_impl(slot, id);
                                    self.emit_add_offset_to_addr_reg_impl(offset);
                                    self.state.emit_fmt(format_args!("    str {}, [x9]", fp));
                                }
                                SlotAddr::Direct(slot) => {
                                    let folded_slot = StackSlot(slot.0 + offset);
                                    self.emit_store_to_sp(&fp, folded_slot.0, "str");
                                }
                                SlotAddr::Indirect(slot) => {
                                    self.emit_load_ptr_from_slot_impl(slot, base.0);
                                    if offset > 0 && offset % scale == 0 && offset / scale <= 4095 {
                                        self.state.emit_fmt(format_args!("    str {}, [x9, #{}]", fp, offset));
                                    } else if (-256..=255).contains(&offset) {
                                        self.state.emit_fmt(format_args!("    stur {}, [x9, #{}]", fp, offset));
                                    } else {
                                        if offset != 0 {
                                            self.emit_add_offset_to_addr_reg_impl(offset);
                                        }
                                        self.state.emit_fmt(format_args!("    str {}, [x9]", fp));
                                    }
                                }
                            }
                            return;
                        }
                    }
                }
            }
        }
        self.operand_to_x0(val);
        let addr = self.state.resolve_slot_addr(base.0);
        if let Some(addr) = addr {
            let store_instr = self.store_instr_for_type_impl(ty);
            match addr {
                SlotAddr::OverAligned(slot, id) => {
                    self.state.emit("    mov x1, x0");
                    self.emit_alloca_aligned_addr_impl(slot, id);
                    self.emit_add_offset_to_addr_reg_impl(offset);
                    let reg = Self::reg_for_type("x1", ty);
                    self.state.emit_fmt(format_args!("    {} {}, [x9]", store_instr, reg));
                }
                SlotAddr::Direct(slot) => {
                    let folded_slot = StackSlot(slot.0 + offset);
                    let reg = Self::reg_for_type("x0", ty);
                    self.emit_store_to_sp(reg, folded_slot.0, store_instr);
                }
                SlotAddr::Indirect(slot) => {
                    self.state.emit("    mov x1, x0");
                    self.emit_load_ptr_from_slot_impl(slot, base.0);
                    if offset != 0 {
                        self.emit_add_offset_to_addr_reg_impl(offset);
                    }
                    let reg = Self::reg_for_type("x1", ty);
                    self.state.emit_fmt(format_args!("    {} {}, [x9]", store_instr, reg));
                }
            }
        }
    }

    pub(super) fn emit_load_with_const_offset_impl(&mut self, dest: &Value, base: &Value, offset: i64, ty: IrType) {
        if ty == IrType::F128 {
            crate::backend::f128_softfloat::f128_emit_load_with_offset(self, dest, base, offset);
            return;
        }
        if matches!(ty, IrType::F32 | IrType::F64) {
            if let (Some(&dest_phys), Some(&base_phys)) =
                (self.reg_assignments.get(&dest.0), self.reg_assignments.get(&base.0))
            {
                if is_arm_fp_phys(dest_phys) && !is_arm_fp_phys(base_phys) {
                    let fp = arm_fp_name(dest_phys, ty);
                    let base_reg = callee_saved_name(base_phys);
                    let scale = if ty == IrType::F32 { 4 } else { 8 };
                    if offset == 0 {
                        self.state.emit_fmt(format_args!("    ldr {}, [{}]", fp, base_reg));
                    } else if offset > 0 && offset % scale == 0 && offset / scale <= 4095 {
                        self.state.emit_fmt(format_args!("    ldr {}, [{}, #{}]", fp, base_reg, offset));
                    } else if (-256..=255).contains(&offset) {
                        self.state.emit_fmt(format_args!("    ldur {}, [{}, #{}]", fp, base_reg, offset));
                    } else {
                        self.state.emit_fmt(format_args!("    mov x9, {}", base_reg));
                        self.emit_add_offset_to_addr_reg_impl(offset);
                        self.state.emit_fmt(format_args!("    ldr {}, [x9]", fp));
                    }
                    return;
                }
            }
        }
        let addr = self.state.resolve_slot_addr(base.0);
        if let Some(addr) = addr {
            let load_instr = self.load_instr_for_type_impl(ty);
            match addr {
                SlotAddr::OverAligned(slot, id) => {
                    self.emit_alloca_aligned_addr_impl(slot, id);
                    self.emit_add_offset_to_addr_reg_impl(offset);
                    let (actual_instr, dest_reg) = Self::arm_parse_load(load_instr);
                    self.state.emit_fmt(format_args!("    {} {}, [x9]", actual_instr, dest_reg));
                }
                SlotAddr::Direct(slot) => {
                    let folded_slot = StackSlot(slot.0 + offset);
                    let (actual_instr, dest_reg) = Self::arm_parse_load(load_instr);
                    self.emit_load_from_sp(dest_reg, folded_slot.0, actual_instr);
                }
                SlotAddr::Indirect(slot) => {
                    self.emit_load_ptr_from_slot_impl(slot, base.0);
                    if offset != 0 {
                        self.emit_add_offset_to_addr_reg_impl(offset);
                    }
                    let (actual_instr, dest_reg) = Self::arm_parse_load(load_instr);
                    self.state.emit_fmt(format_args!("    {} {}, [x9]", actual_instr, dest_reg));
                }
            }
            self.store_x0_to(dest);
        }
    }

    pub(super) fn emit_typed_store_to_slot_impl(&mut self, instr: &'static str, ty: IrType, slot: StackSlot) {
        let reg = Self::reg_for_type("x0", ty);
        self.emit_store_to_sp(reg, slot.0, instr);
    }

    pub(super) fn emit_typed_load_from_slot_impl(&mut self, instr: &'static str, slot: StackSlot) {
        let (actual_instr, dest_reg) = Self::arm_parse_load(instr);
        self.emit_load_from_sp(dest_reg, slot.0, actual_instr);
    }

    pub(super) fn emit_load_ptr_from_slot_impl(&mut self, slot: StackSlot, val_id: u32) {
        if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x9, {}", reg_name));
        } else {
            self.emit_load_from_sp("x9", slot.0, "ldr");
        }
    }

    pub(super) fn emit_typed_store_indirect_impl(&mut self, instr: &'static str, ty: IrType) {
        // emit_store_default loads the pointer into x9 first and the value into
        // the accumulator (x0) second.  Using x1 here stores an unrelated value
        // for globals, heap fields, and all other indirect destinations.
        let reg = Self::reg_for_type("x0", ty);
        self.state.emit_fmt(format_args!("    {} {}, [x9]", instr, reg));
    }

    pub(super) fn emit_typed_load_indirect_impl(&mut self, instr: &'static str) {
        let (actual_instr, dest_reg) = Self::arm_parse_load(instr);
        self.state.emit_fmt(format_args!("    {} {}, [x9]", actual_instr, dest_reg));
    }

    pub(super) fn emit_add_offset_to_addr_reg_impl(&mut self, offset: i64) {
        if (0..=4095).contains(&offset) {
            self.state.emit_fmt(format_args!("    add x9, x9, #{}", offset));
        } else if offset < 0 && (-offset) <= 4095 {
            self.state.emit_fmt(format_args!("    sub x9, x9, #{}", -offset));
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit("    add x9, x9, x17");
        }
    }

    pub(super) fn emit_slot_addr_to_secondary_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr("x1", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x1, {}", reg_name));
        } else {
            self.emit_load_from_sp("x1", slot.0, "ldr");
        }
    }

    pub(super) fn emit_gep_direct_const_impl(&mut self, slot: StackSlot, offset: i64) {
        let folded = slot.0 + offset;
        self.emit_add_sp_offset("x0", folded);
    }

    pub(super) fn emit_gep_indirect_const_impl(&mut self, slot: StackSlot, offset: i64, val_id: u32) {
        if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x0, {}", reg_name));
        } else {
            self.emit_load_from_sp("x0", slot.0, "ldr");
        }
        if offset != 0 {
            self.emit_add_imm_to_acc_impl(offset);
        }
    }

    pub(super) fn emit_add_imm_to_acc_impl(&mut self, imm: i64) {
        if (0..=4095).contains(&imm) {
            self.state.emit_fmt(format_args!("    add x0, x0, #{}", imm));
        } else if imm < 0 && (-imm) <= 4095 {
            self.state.emit_fmt(format_args!("    sub x0, x0, #{}", -imm));
        } else {
            self.emit_load_imm64("x1", imm);
            self.state.emit("    add x0, x0, x1");
        }
    }

    pub(super) fn emit_round_up_acc_to_16_impl(&mut self) {
        self.state.emit("    add x0, x0, #15");
        self.state.emit("    and x0, x0, #-16");
    }

    pub(super) fn emit_sub_sp_by_acc_impl(&mut self) {
        self.state.emit("    sub sp, sp, x0");
    }

    pub(super) fn emit_mov_sp_to_acc_impl(&mut self) {
        self.state.emit("    mov x0, sp");
    }

    pub(super) fn emit_mov_acc_to_sp_impl(&mut self) {
        self.state.emit("    mov sp, x0");
    }

    pub(super) fn emit_align_acc_impl(&mut self, align: usize) {
        self.state.emit_fmt(format_args!("    add x0, x0, #{}", align - 1));
        self.state.emit_fmt(format_args!("    and x0, x0, #{}", -(align as i64)));
    }

    pub(super) fn emit_memcpy_load_dest_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr("x9", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x9, {}", reg_name));
        } else {
            self.emit_load_from_sp("x9", slot.0, "ldr");
        }
    }

    pub(super) fn emit_memcpy_load_src_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr("x10", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x10, {}", reg_name));
        } else {
            self.emit_load_from_sp("x10", slot.0, "ldr");
        }
    }

    pub(super) fn emit_alloca_aligned_addr_impl(&mut self, slot: StackSlot, val_id: u32) {
        let align = self.state.alloca_over_align(val_id)
            .expect("alloca must have over-alignment for aligned addr emission");
        self.emit_add_sp_offset("x9", slot.0);
        self.load_large_imm("x17", (align - 1) as i64);
        self.state.emit("    add x9, x9, x17");
        self.load_large_imm("x17", -(align as i64));
        self.state.emit("    and x9, x9, x17");
    }

    pub(super) fn emit_alloca_aligned_addr_to_acc_impl(&mut self, slot: StackSlot, val_id: u32) {
        let align = self.state.alloca_over_align(val_id)
            .expect("alloca must have over-alignment for aligned addr emission");
        self.emit_add_sp_offset("x0", slot.0);
        self.load_large_imm("x17", (align - 1) as i64);
        self.state.emit("    add x0, x0, x17");
        self.load_large_imm("x17", -(align as i64));
        self.state.emit("    and x0, x0, x17");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_memcpy_impl_impl(&mut self, size: usize) {
        // Struct assignments are overwhelmingly small fixed-size copies.  A
        // byte-at-a-time runtime loop is especially costly when the copy sits
        // in a hot loop, so use AArch64 pair transfers for sizes that can be
        // unrolled without excessive code growth.  x9 is the destination and
        // x10 the source; x12/x13 are reserved codegen scratch registers.
        if size <= 256 {
            let mut offset = 0usize;
            while offset + 16 <= size {
                self.state.emit_fmt(format_args!("    ldp x12, x13, [x10, #{}]", offset));
                self.state.emit_fmt(format_args!("    stp x12, x13, [x9, #{}]", offset));
                offset += 16;
            }
            if offset + 8 <= size {
                self.state.emit_fmt(format_args!("    ldr x12, [x10, #{}]", offset));
                self.state.emit_fmt(format_args!("    str x12, [x9, #{}]", offset));
                offset += 8;
            }
            if offset + 4 <= size {
                self.state.emit_fmt(format_args!("    ldr w12, [x10, #{}]", offset));
                self.state.emit_fmt(format_args!("    str w12, [x9, #{}]", offset));
                offset += 4;
            }
            if offset + 2 <= size {
                self.state.emit_fmt(format_args!("    ldrh w12, [x10, #{}]", offset));
                self.state.emit_fmt(format_args!("    strh w12, [x9, #{}]", offset));
                offset += 2;
            }
            if offset < size {
                self.state.emit_fmt(format_args!("    ldrb w12, [x10, #{}]", offset));
                self.state.emit_fmt(format_args!("    strb w12, [x9, #{}]", offset));
            }
            return;
        }

        let label_id = self.state.next_label_id();
        let loop_label = format!(".Lmemcpy_loop_{}", label_id);
        let done_label = format!(".Lmemcpy_done_{}", label_id);
        self.load_large_imm("x11", size as i64);
        self.state.emit_fmt(format_args!("{}:", loop_label));
        self.state.emit_fmt(format_args!("    cbz x11, {}", done_label));
        self.state.emit("    ldrb w12, [x10], #1");
        self.state.emit("    strb w12, [x9], #1");
        self.state.emit("    sub x11, x11, #1");
        self.state.emit_fmt(format_args!("    b {}", loop_label));
        self.state.emit_fmt(format_args!("{}:", done_label));
    }
}
