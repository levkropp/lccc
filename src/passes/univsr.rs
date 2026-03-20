/ Un-IVSR pass: Reverse pointer IV strength reduction when indexed addressing is beneficial.
//!
//! This pass runs after IVSR and detects pointer induction variables created by IVSR.
//! When the target architecture supports indexed addressing (like x86-64 SIB), it's often
//! more efficient to use `mov (%base,%index,scale)` than pointer arithmetic `mov (%ptr); add $stride, %ptr`.
//!
//! The pass transforms:
//!   %ptr = Phi(%initial_ptr, %ptr_next)
//!   %val = Load(%ptr)
//!   %ptr_next = GEP(%ptr, stride)
//!
//! Back to:
//!   %index = Phi(%init_index, %index_next)
//!   %offset = Mul/Shl(%index, stride)
//!   %addr = GEP(%base, %offset)
//!   %val = Load(%addr)
//!   %index_next = Add(%index, 1)
//!
//! This allows the backend's indexed addressing detection (Phase 9) to emit
//! efficient SIB-form instructions.

use crate::common::fx_hash::FxHashMap;
use crate::common::types::IrType;
use crate::ir::reexports::{
    BlockId,
    Instruction,
    IrBinOp,
    IrConst,
    IrFunction,
    Operand,
    Value,
};

/// Information about an IVSR-created pointer IV that should be un-transformed
struct IvsrPointerIV {
    /// The pointer phi's destination value
    ptr_phi_dest: Value,
    /// The original array base pointer
    base_ptr: Value,
    /// Element stride in bytes
    stride: i64,
    /// Initial offset (usually 0)
    init_offset: i64,
    /// The associated index IV (if found)
    index_iv: Option<Value>,
    /// Block where the phi resides
    header_block: BlockId,
}

/// Run the un-IVSR pass on a function.
/// Returns the number of pointer IVs that were reverted.
pub(crate) fn run_univsr(func: &mut IrFunction) -> usize {
    // Only run on x86_64 where indexed addressing is available
    // TODO: Make this target-aware when other architectures are added

    let ivsr_pointers = detect_ivsr_pointer_ivs(func);
    if ivsr_pointers.is_empty() {
        return 0;
    }

    let mut num_reverted = 0;
    for ptr_iv in ivsr_pointers {
        // Only revert if stride is a valid SIB scale (1, 2, 4, or 8)
        if !is_valid_sib_scale(ptr_iv.stride) {
            continue;
        }

        // Only revert if we found the index IV
        if ptr_iv.index_iv.is_none() {
            continue;
        }

        if revert_pointer_iv(func, &ptr_iv) {
            num_reverted += 1;
        }
    }

    num_reverted
}

/// Detect IVSR-created pointer IVs in the function
fn detect_ivsr_pointer_ivs(func: &IrFunction) -> Vec<IvsrPointerIV> {
    let mut result = Vec::new();

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Phi { dest, ty, incoming } = inst {
                // Only interested in pointer phis with exactly 2 incoming edges
                if *ty != IrType::Ptr || incoming.len() != 2 {
                    continue;
                }

                // Check if this matches IVSR pointer pattern
                if let Some(ptr_iv) = analyze_pointer_phi(func, dest, incoming, block.label) {
                    result.push(ptr_iv);
                }
            }
        }
    }

    result
}

/// Analyze a pointer phi to see if it's an IVSR pattern
fn analyze_pointer_phi(
    func: &IrFunction,
    dest: &Value,
    incoming: &[(Operand, BlockId)],
    header_block: BlockId,
) -> Option<IvsrPointerIV> {
    // Extract the two incoming edges
    let (init_op, _init_block) = &incoming[0];
    let (backedge_op, _backedge_block) = &incoming[1];

    // Backedge value should be defined by a GEP
    let backedge_val = match backedge_op {
        Operand::Value(v) => v,
        _ => return None,
    };

    // Find the GEP instruction that defines the backedge value
    let (gep_base, stride) = find_gep_with_const_offset(func, backedge_val.0)?;

    // Verify GEP base is the phi itself (recursive pattern: %ptr_next = GEP(%ptr, stride))
    if gep_base.0 != dest.0 {
        return None;
    }

    // Extract base pointer from init value
    let (base_ptr, init_offset) = extract_base_from_init(func, init_op)?;

    // Try to find the associated index IV
    let index_iv = find_index_iv_in_header(func, header_block);

    Some(IvsrPointerIV {
        ptr_phi_dest: *dest,
        base_ptr,
        stride,
        init_offset,
        index_iv,
        header_block,
    })
}

/// Find a GEP instruction that defines a value with a constant offset
fn find_gep_with_const_offset(func: &IrFunction, val_id: u32) -> Option<(Value, i64)> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, base, offset, .. } = inst {
                if dest.0 == val_id {
                    // Extract constant offset
                    let stride = match offset {
                        Operand::Const(c) => c.to_i64()?,
                        _ => return None,
                    };
                    return Some((*base, stride));
                }
            }
        }
    }
    None
}

/// Extract the base pointer from the init value of a pointer phi
fn extract_base_from_init(func: &IrFunction, init_op: &Operand) -> Option<(Value, i64)> {
    match init_op {
        Operand::Value(v) => {
            // Check if init value is a GEP (handles cases like %init = GEP(%arr, 0))
            for block in &func.blocks {
                for inst in &block.instructions {
                    if let Instruction::GetElementPtr { dest, base, offset, .. } = inst {
                        if dest.0 == v.0 {
                            let init_offset = match offset {
                                Operand::Const(c) => c.to_i64().unwrap_or(0),
                                _ => 0,
                            };
                            return Some((*base, init_offset));
                        }
                    }
                }
            }
            // Not a GEP - just a direct pointer value
            Some((*v, 0))
        }
        _ => None,
    }
}

/// Find an integer IV phi in the same header block
fn find_index_iv_in_header(func: &IrFunction, header: BlockId) -> Option<Value> {
    let header_block = &func.blocks[header.0 as usize];

    for inst in &header_block.instructions {
        if let Instruction::Phi { dest, ty, incoming } = inst {
            // Look for integer phis (I32, I64, U32, U64)
            if !matches!(ty, IrType::I32 | IrType::I64 | IrType::U32 | IrType::U64) {
                continue;
            }

            // Check if it's incremented by 1 in the backedge (typical loop counter)
            if is_unit_increment_iv(func, dest, incoming) {
                return Some(*dest);
            }
        }
    }

    None
}

/// Check if a phi is incremented by 1 per iteration
fn is_unit_increment_iv(func: &IrFunction, dest: &Value, incoming: &[(Operand, BlockId)]) -> bool {
    if incoming.len() != 2 {
        return false;
    }

    // Get the backedge value (typically the second incoming edge)
    let backedge_val = match &incoming[1].0 {
        Operand::Value(v) => v,
        _ => return false,
    };

    // Check if backedge value is defined by Add(%iv, 1) or Add(1, %iv)
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::BinOp { dest: bd, op: IrBinOp::Add, lhs, rhs, .. } = inst {
                if bd.0 == backedge_val.0 {
                    match (lhs, rhs) {
                        (Operand::Value(v), Operand::Const(c)) | (Operand::Const(c), Operand::Value(v)) => {
                            if v.0 == dest.0 && c.to_i64() == Some(1) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    false
}

/// Revert a pointer IV back to indexed form
fn revert_pointer_iv(func: &mut IrFunction, ptr_iv: &IvsrPointerIV) -> bool {
    // For now, just return false - full implementation would:
    // 1. Find all uses of the pointer phi
    // 2. Replace with GEP(base, index * stride)
    // 3. Update the index IV
    // 4. Remove the pointer phi

    // This is complex and would require significant IR manipulation.
    // For Phase 9b MVP, we'll handle this at codegen level instead.
    false
}

/// Check if a stride is valid for x86-64 SIB encoding
fn is_valid_sib_scale(stride: i64) -> bool {
    matches!(stride, 1 | 2 | 4 | 8)
}
