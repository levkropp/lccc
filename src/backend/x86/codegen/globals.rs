//! X86Codegen: global address, label address, TLS global address operations.

use crate::ir::reexports::{Operand, Value};
use crate::common::types::IrType;
use super::emit::{X86Codegen, phys_reg_name, phys_reg_name_32, typed_phys_reg_name, is_xmm_reg};

impl X86Codegen {
    pub(super) fn emit_global_addr_impl(&mut self, dest: &Value, name: &str) {
        // Register-direct: emit directly to dest register, skip %rax relay.
        if let Some(d_reg) = self.dest_reg(dest) {
            if !is_xmm_reg(d_reg) {
                let d_name = phys_reg_name(d_reg);
                if self.state.needs_got_for_addr(name) {
                    self.state.emit_fmt(format_args!("    movq {}@GOTPCREL(%rip), %{}", name, d_name));
                } else {
                    self.state.out.emit_instr_sym_base_reg("    leaq", name, "rip", d_name);
                }
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }
        if self.state.needs_got_for_addr(name) {
            self.state.emit_fmt(format_args!("    movq {}@GOTPCREL(%rip), %rax", name));
        } else {
            self.state.out.emit_instr_sym_base_reg("    leaq", name, "rip", "rax");
        }
        self.store_rax_to(dest);
    }

    pub(super) fn emit_tls_global_addr_impl(&mut self, dest: &Value, name: &str) {
        // TLS requires %rax for the fs:0 base — can't fully avoid accumulator.
        // But when dest has a register, we can skip the final store_rax_to relay.
        if self.state.pic_mode {
            self.state.emit_fmt(format_args!("    movq {}@GOTTPOFF(%rip), %rax", name));
            self.state.emit("    addq %fs:0, %rax");
        } else {
            self.state.emit("    movq %fs:0, %rax");
            self.state.emit_fmt(format_args!("    leaq {}@TPOFF(%rax), %rax", name));
        }
        self.store_rax_to(dest);
    }

    pub(super) fn emit_global_addr_absolute_impl(&mut self, dest: &Value, name: &str) {
        // Register-direct: emit directly to dest register.
        if let Some(d_reg) = self.dest_reg(dest) {
            if !is_xmm_reg(d_reg) {
                let d_name = phys_reg_name(d_reg);
                self.state.out.emit_instr_sym_imm_reg("    movq", name, d_name);
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }
        self.state.out.emit_instr_sym_imm_reg("    movq", name, "rax");
        self.store_rax_to(dest);
    }

    pub(super) fn emit_global_load_rip_rel_impl(&mut self, dest: &Value, sym: &str, ty: IrType) {
        // Register-direct: load directly to dest register.
        if let Some(d_reg) = self.dest_reg(dest) {
            if !is_xmm_reg(d_reg) {
                let load_instr = Self::mov_load_for_type(ty);
                let d_name = match ty {
                    IrType::U32 | IrType::F32 => phys_reg_name_32(d_reg),
                    _ => phys_reg_name(d_reg),
                };
                self.state.emit_fmt(format_args!("    {} {}(%rip), %{}", load_instr, sym, d_name));
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }
        let load_instr = Self::mov_load_for_type(ty);
        let dest_reg = Self::load_dest_reg(ty);
        self.state.emit_fmt(format_args!("    {} {}(%rip), {}", load_instr, sym, dest_reg));
        self.emit_store_result_impl(dest);
    }

    pub(super) fn emit_global_store_rip_rel_impl(&mut self, val: &Operand, sym: &str, ty: IrType) {
        // Register-direct: store directly from val register, skip operand_to_rax.
        if let Operand::Value(v) = val {
            if let Some(v_reg) = self.reg_assignments.get(&v.0).copied() {
                if !is_xmm_reg(v_reg) {
                    let store_instr = Self::mov_store_for_type(ty);
                    let v_name = typed_phys_reg_name(v_reg, ty);
                    self.state.emit_fmt(format_args!("    {} %{}, {}(%rip)", store_instr, v_name, sym));
                    return;
                }
            }
        }
        self.emit_load_operand_impl(val);
        let store_instr = Self::mov_store_for_type(ty);
        let store_reg = Self::reg_for_type("rax", ty);
        self.state.emit_fmt(format_args!("    {} %{}, {}(%rip)", store_instr, store_reg, sym));
    }

    pub(super) fn emit_label_addr_impl(&mut self, dest: &Value, label: &str) {
        // Register-direct: emit directly to dest register.
        if let Some(d_reg) = self.dest_reg(dest) {
            if !is_xmm_reg(d_reg) {
                let d_name = phys_reg_name(d_reg);
                self.state.out.emit_instr_sym_base_reg("    leaq", label, "rip", d_name);
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }
        self.state.out.emit_instr_sym_base_reg("    leaq", label, "rip", "rax");
        self.store_rax_to(dest);
    }

    // These thin helpers avoid circular delegation issues:
    fn emit_store_result_impl(&mut self, dest: &Value) {
        self.store_rax_to(dest);
    }

    fn emit_load_operand_impl(&mut self, op: &Operand) {
        self.operand_to_rax(op);
    }
}
