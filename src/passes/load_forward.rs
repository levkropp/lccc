//! Redundant reload forwarding across a single predecessor edge.
//!
//! When a block B has exactly one predecessor P, and P ends with a Load of a
//! given pointer, and B reloads the same pointer (same type) with no
//! intervening memory clobber on either side, the reload is redundant — the
//! value was just loaded on the only incoming path. Replace it with a Copy of
//! the earlier load's destination:
//!
//! ```text
//! P:  %a = load %ptr        B:  %b = load %ptr      (redundant)
//!     ...                     ...
//! ```
//!
//! This pattern shows up in conditional reductions like
//! `if (arr[i] > 0) s += arr[i]`, where the frontend emits one load for the
//! condition and a second load in the taken arm. Forwarding the value removes
//! a per-iteration load and, because the arm becomes load-free, lets
//! if-conversion turn the diamond into a branchless `select`.
//!
//! Safety: we require B to have a single predecessor (so the earlier load is
//! guaranteed to have executed), identical pointer value and load type, and no
//! memory clobber (store/call/memcpy/atomic/fence) between the two loads.

use crate::common::types::AddressSpace;
use crate::ir::analysis;
use crate::ir::reexports::{Instruction, IrFunction, Operand, Value};

/// Run redundant-load forwarding on a function. Returns the number of loads
/// replaced by copies.
pub(crate) fn run(func: &mut IrFunction) -> usize {
    // Map label -> block index for predecessor lookups.
    let label_to_idx = analysis::build_label_map(func);
    let (preds, _succs) = analysis::build_cfg(func, &label_to_idx);

    // dest value of a load -> (ptr value id, type) for loads that end a block
    // with no trailing memory clobber. Built per-block below.
    let mut rewrites: Vec<(usize, usize, Value, Value)> = Vec::new(); // (block, inst, dest, src)

    for (bi, block) in func.blocks.iter().enumerate() {
        // Single-predecessor requirement.
        let pred_row = preds.row(bi);
        if pred_row.len() != 1 {
            continue;
        }
        let pred_idx = pred_row[0] as usize;
        let pred_block = &func.blocks[pred_idx];

        // Find the last Load in the predecessor with no memory clobber after it.
        // Scan backwards from the end.
        let mut pred_load: Option<(u32, u32, crate::common::types::IrType)> = None; // (ptr id, dest id, ty)
        for inst in pred_block.instructions.iter().rev() {
            match inst {
                Instruction::Load { dest, ptr, ty, seg_override } => {
                    if *seg_override == AddressSpace::Default && !ty.is_float() && !ty.is_128bit() && !ty.is_long_double() {
                        pred_load = Some((ptr.0, dest.0, *ty));
                    }
                    break;
                }
                // A memory clobber after the load means the value may be stale.
                Instruction::Store { .. }
                | Instruction::Memcpy { .. }
                | Instruction::Call { .. }
                | Instruction::CallIndirect { .. }
                | Instruction::AtomicStore { .. }
                | Instruction::AtomicRmw { .. }
                | Instruction::AtomicCmpxchg { .. }
                | Instruction::Fence { .. } => break,
                _ => {}
            }
        }
        let Some((pred_ptr, pred_dest, pred_ty)) = pred_load else { continue };

        // Find the first Load in this block with no memory clobber before it.
        for (ii, inst) in block.instructions.iter().enumerate() {
            match inst {
                Instruction::Load { dest, ptr, ty, seg_override } => {
                    if *seg_override == AddressSpace::Default
                        && ptr.0 == pred_ptr
                        && *ty == pred_ty
                        && dest.0 != pred_dest
                    {
                        rewrites.push((bi, ii, *dest, Value(pred_dest)));
                    }
                    break; // Only the first load is safe to forward.
                }
                Instruction::Store { .. }
                | Instruction::Memcpy { .. }
                | Instruction::Call { .. }
                | Instruction::CallIndirect { .. }
                | Instruction::AtomicStore { .. }
                | Instruction::AtomicRmw { .. }
                | Instruction::AtomicCmpxchg { .. }
                | Instruction::Fence { .. } => break,
                _ => {}
            }
        }
    }

    let count = rewrites.len();
    for (bi, ii, dest, src) in rewrites {
        func.blocks[bi].instructions[ii] = Instruction::Copy { dest, src: Operand::Value(src) };
    }
    count
}
