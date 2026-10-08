//! Late redundant-load elimination within basic blocks.
//!
//! After strength reduction, array/struct field accesses appear as
//! `Load(GEP(base, const_off))` with plain constant offsets. When the frontend
//! emits the same field access in several statements (e.g. `bodies[j].mass`
//! in each of nbody's three velocity updates), the duplicate loads survive
//! GVN because GEP CSE is disabled. This pass merges loads of the same
//! (root, syms, konst, march) linear-form address within a block, as long as
//! every intervening store is provably non-aliasing (affine disjointness).

use crate::common::fx_hash::FxHashMap;
use crate::common::types::IrType;
use crate::ir::reexports::{Instruction, IrFunction, Operand, Value};
use super::alias;

pub(crate) fn run(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_NO_REDUNDANT_LOADS").is_ok() {
        return 0;
    }
    let cfg = crate::ir::analysis::CfgAnalysis::build(func);
    let frames = alias::LoopFrames::build_with_cfg(func, &cfg);

    let mut total = 0;
    // All dest rewrites, applied function-wide at the end (a duplicate dest
    // may be used in a later dominated block; the canonical load always
    // dominates it, being earlier in the same block).
    let mut all_rewrites: FxHashMap<u32, u32> = FxHashMap::default();

    for bi in 0..func.blocks.len() {
        let frame = frames.block_frame[bi];
        // Cached loads: (form, type, canonical dest). Small per block.
        let mut available: Vec<(alias::LinForm, IrType, Value)> = Vec::new();
        let mut removed: Vec<usize> = Vec::new();

        for (ii, inst) in func.blocks[bi].instructions.iter().enumerate() {
            match inst {
                Instruction::Load { dest, ptr, ty, seg_override }
                    if *seg_override == crate::common::types::AddressSpace::Default
                        && !ty.is_long_double()
                        && !ty.is_128bit() =>
                {
                    if let Some(form) = alias::resolve_in_frame(func, &frames, frame, *ptr) {
                        if let Some((_, _, canon)) = available
                            .iter()
                            .find(|(f, t, _)| *f == form && *t == *ty)
                        {
                            all_rewrites.insert(dest.0, canon.0);
                            removed.push(ii);
                            continue;
                        }
                        available.push((form, *ty, *dest));
                    }
                }
                Instruction::Store { ptr, ty, .. } => {
                    if available.is_empty() {
                        continue;
                    }
                    if let Some(sform) = alias::resolve_in_frame(func, &frames, frame, *ptr) {
                        let ssz = alias::byte_size(*ty);
                        available.retain(|(f, lty, _)| {
                            alias::forms_disjoint(f, alias::byte_size(*lty), &sform, ssz, true)
                        });
                    } else {
                        available.clear();
                    }
                }
                // Calls and other memory writers: clear everything.
                Instruction::Call { .. }
                | Instruction::CallIndirect { .. }
                | Instruction::Memcpy { .. }
                | Instruction::AtomicRmw { .. }
                | Instruction::AtomicCmpxchg { .. }
                | Instruction::AtomicStore { .. }
                | Instruction::Fence { .. }
                | Instruction::InlineAsm { .. }
                | Instruction::VaStart { .. }
                | Instruction::VaEnd { .. }
                | Instruction::VaCopy { .. } => {
                    available.clear();
                }
                Instruction::Intrinsic { dest_ptr: Some(_), .. } => {
                    available.clear();
                }
                _ => {}
            }
        }

        if !removed.is_empty() {
            let mut removed_iter = removed.iter().copied();
            let mut next_remove = removed_iter.next();
            let mut idx = 0usize;
            func.blocks[bi].instructions.retain(|_| {
                let cur = idx;
                idx += 1;
                if Some(cur) == next_remove {
                    next_remove = removed_iter.next();
                    total += 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    if !all_rewrites.is_empty() {
        for block in func.blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                super::tail_call_elim::replace_values_in_inst(inst, &all_rewrites);
            }
            // Terminator operands.
            match &mut block.terminator {
                crate::ir::reexports::Terminator::CondBranch { cond, .. } => {
                    if let Operand::Value(v) = cond {
                        if let Some(&to) = all_rewrites.get(&v.0) {
                            *v = Value(to);
                        }
                    }
                }
                crate::ir::reexports::Terminator::Return(Some(op)) => {
                    if let Operand::Value(v) = op {
                        if let Some(&to) = all_rewrites.get(&v.0) {
                            *v = Value(to);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    total
}
