//! ArmCodegen: cast operations.

use crate::ir::reexports::{Operand, Value};
use crate::common::types::IrType;
use crate::backend::cast::{CastKind, classify_cast};
use crate::backend::traits::ArchCodegen;
use super::emit::{ArmCodegen, arm_fp_name, callee_saved_name, callee_saved_name_32, is_arm_fp_phys};

impl ArmCodegen {
    pub(super) fn emit_cast_instrs_impl(&mut self, from_ty: IrType, to_ty: IrType) {
        match classify_cast(from_ty, to_ty) {
            CastKind::Noop | CastKind::UnsignedToSignedSameSize { .. } => {}

            CastKind::FloatToSigned { from_f64 } => {
                if from_f64 {
                    self.state.emit("    fmov d0, x0");
                    self.state.emit("    fcvtzs x0, d0");
                } else {
                    self.state.emit("    fmov s0, w0");
                    self.state.emit("    fcvtzs x0, s0");
                }
                match to_ty {
                    IrType::I8 => self.state.emit("    sxtb x0, w0"),
                    IrType::I16 => self.state.emit("    sxth x0, w0"),
                    IrType::I32 => self.state.emit("    sxtw x0, w0"),
                    _ => {}
                }
            }

            CastKind::FloatToUnsigned { from_f64, .. } => {
                if from_f64 {
                    self.state.emit("    fmov d0, x0");
                    self.state.emit("    fcvtzu x0, d0");
                } else {
                    self.state.emit("    fmov s0, w0");
                    self.state.emit("    fcvtzu x0, s0");
                }
                match to_ty {
                    IrType::U8 => self.state.emit("    and x0, x0, #0xff"),
                    IrType::U16 => self.state.emit("    and x0, x0, #0xffff"),
                    IrType::U32 => self.state.emit("    mov w0, w0"),
                    _ => {}
                }
            }

            CastKind::SignedToFloat { to_f64, from_ty } => {
                match from_ty.size() {
                    1 => self.state.emit("    sxtb x0, w0"),
                    2 => self.state.emit("    sxth x0, w0"),
                    4 => self.state.emit("    sxtw x0, w0"),
                    _ => {}
                }
                if to_f64 {
                    self.state.emit("    scvtf d0, x0");
                    self.state.emit("    fmov x0, d0");
                } else {
                    self.state.emit("    scvtf s0, x0");
                    self.state.emit("    fmov w0, s0");
                }
            }

            CastKind::UnsignedToFloat { to_f64, .. } => {
                if to_f64 {
                    self.state.emit("    ucvtf d0, x0");
                    self.state.emit("    fmov x0, d0");
                } else {
                    self.state.emit("    ucvtf s0, x0");
                    self.state.emit("    fmov w0, s0");
                }
            }

            CastKind::FloatToFloat { widen } => {
                if widen {
                    self.state.emit("    fmov s0, w0");
                    self.state.emit("    fcvt d0, s0");
                    self.state.emit("    fmov x0, d0");
                } else {
                    self.state.emit("    fmov d0, x0");
                    self.state.emit("    fcvt s0, d0");
                    self.state.emit("    fmov w0, s0");
                }
            }

            CastKind::SignedToUnsignedSameSize { to_ty } => {
                match to_ty {
                    IrType::U8 => self.state.emit("    and x0, x0, #0xff"),
                    IrType::U16 => self.state.emit("    and x0, x0, #0xffff"),
                    IrType::U32 => self.state.emit("    mov w0, w0"),
                    _ => {}
                }
            }

            CastKind::IntWiden { from_ty, .. } => {
                if from_ty.is_unsigned() {
                    match from_ty {
                        IrType::U8 => self.state.emit("    and x0, x0, #0xff"),
                        IrType::U16 => self.state.emit("    and x0, x0, #0xffff"),
                        IrType::U32 => self.state.emit("    mov w0, w0"),
                        _ => {}
                    }
                } else {
                    match from_ty {
                        IrType::I8 => self.state.emit("    sxtb x0, w0"),
                        IrType::I16 => self.state.emit("    sxth x0, w0"),
                        IrType::I32 => self.state.emit("    sxtw x0, w0"),
                        _ => {}
                    }
                }
            }

            CastKind::IntNarrow { to_ty } => {
                match to_ty {
                    IrType::I8 => self.state.emit("    sxtb x0, w0"),
                    IrType::U8 => self.state.emit("    and x0, x0, #0xff"),
                    IrType::I16 => self.state.emit("    sxth x0, w0"),
                    IrType::U16 => self.state.emit("    and x0, x0, #0xffff"),
                    IrType::I32 => self.state.emit("    sxtw x0, w0"),
                    IrType::U32 => self.state.emit("    mov w0, w0"),
                    _ => {}
                }
            }

            CastKind::SignedToF128 { .. }
            | CastKind::UnsignedToF128 { .. }
            | CastKind::F128ToSigned { .. }
            | CastKind::F128ToUnsigned { .. }
            | CastKind::FloatToF128 { .. }
            | CastKind::F128ToFloat { .. } => {
                unreachable!("F128 cast variants not produced by classify_cast()");
            }
        }
    }

    pub(super) fn emit_cast_impl(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) {
        if crate::backend::f128_softfloat::f128_emit_cast(self, dest, src, from_ty, to_ty) {
            return;
        }
        // Integer-to-float casts whose result has an FP allocation can write
        // that register directly.  The generic accumulator convention would
        // otherwise round-trip through x0 (`scvtf d0; fmov x0,d0; fmov dN,x0`).
        if let Some(&phys) = self.reg_assignments.get(&dest.0) {
            if is_arm_fp_phys(phys) && matches!(to_ty, IrType::F32 | IrType::F64) {
                let kind = classify_cast(from_ty, to_ty);
                if matches!(kind, CastKind::SignedToFloat { .. } | CastKind::UnsignedToFloat { .. }) {
                    let signed = matches!(kind, CastKind::SignedToFloat { .. });
                    let mnemonic = if signed { "scvtf" } else { "ucvtf" };
                    let fp_dest = arm_fp_name(phys, to_ty);
                    // Source already in a register: convert directly from it,
                    // no x0 round-trip. Only whole-register source widths here;
                    // sub-register signed extension still needs a scratch reg.
                    let src_phys = self.operand_reg(src).filter(|r| !is_arm_fp_phys(*r));
                    if let Some(sp) = src_phys {
                        match from_ty.size() {
                            8 => {
                                self.state.emit_fmt(format_args!(
                                    "    {} {}, {}", mnemonic, fp_dest, callee_saved_name(sp)));
                            }
                            4 => {
                                self.state.emit_fmt(format_args!(
                                    "    {} {}, {}", mnemonic, fp_dest, callee_saved_name_32(sp)));
                            }
                            _ => {
                                self.emit_load_operand(src);
                                if signed {
                                    if from_ty.size() == 1 { self.state.emit("    sxtb x0, w0"); }
                                    else { self.state.emit("    sxth x0, w0"); }
                                } else {
                                    if from_ty.size() == 1 { self.state.emit("    and x0, x0, #0xff"); }
                                    else { self.state.emit("    and x0, x0, #0xffff"); }
                                }
                                self.state.emit_fmt(format_args!("    {} {}, x0", mnemonic, fp_dest));
                            }
                        }
                        self.state.reg_cache.invalidate_acc();
                        return;
                    }
                    self.emit_load_operand(src);
                    if signed {
                        match from_ty.size() {
                            1 => self.state.emit("    sxtb x0, w0"),
                            2 => self.state.emit("    sxth x0, w0"),
                            4 => self.state.emit("    sxtw x0, w0"),
                            _ => {}
                        }
                    } else {
                        match from_ty.size() {
                            1 => self.state.emit("    and x0, x0, #0xff"),
                            2 => self.state.emit("    and x0, x0, #0xffff"),
                            4 => self.state.emit("    mov w0, w0"),
                            _ => {}
                        }
                    }
                    self.state.emit_fmt(format_args!("    {} {}, x0", mnemonic, fp_dest));
                    self.state.reg_cache.invalidate_acc();
                    return;
                }
            }
        }
        crate::backend::traits::emit_cast_default(self, dest, src, from_ty, to_ty);
    }
}
