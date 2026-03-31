//! Copy coalescing and immediately-consumed value analysis.
//!
//! Copy coalescing identifies Copy instructions where the destination can
//! share the source's stack slot (eliminating a separate allocation).
//! The immediately-consumed analysis identifies values that are produced
//! and consumed in adjacent instructions, allowing them to skip stack
//! slot allocation entirely by staying in the accumulator register cache.

use crate::ir::reexports::{
    Instruction,
    IrFunction,
    Operand,
    Terminator,
};
use crate::common::types::IrType;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::backend::regalloc::PhysReg;
use crate::backend::liveness::{
    for_each_operand_in_instruction, for_each_value_use_in_instruction,
    for_each_operand_in_terminator,
};

/// Build the copy alias map: dest_id -> root_id for Copy instructions where
/// dest and src can share the same stack slot.
///
/// Safety: only coalesces when the Copy is the SOLE use of the source value,
/// guaranteeing the source is dead after the Copy (avoids the "lost copy"
/// problem in phi parallel copy groups).
/// Returns `(copy_alias, phi_web_aliases)` where phi_web_aliases contains value IDs
/// that were coalesced via phi-web analysis and need force-overwrite in resolve_copy_aliases.
pub(super) fn build_copy_alias_map(
    func: &IrFunction,
    def_block: &FxHashMap<u32, usize>,
    multi_def_values: &FxHashSet<u32>,
    reg_assigned: &FxHashMap<u32, PhysReg>,
    use_blocks_map: &FxHashMap<u32, Vec<usize>>,
    cached_liveness: &Option<crate::backend::liveness::LivenessResult>,
) -> (FxHashMap<u32, u32>, FxHashSet<u32>) {
    // Count uses of each value across all instructions.
    let mut use_count: FxHashMap<u32, u32> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    *use_count.entry(v.0).or_insert(0) += 1;
                }
            });
            for_each_value_use_in_instruction(inst, |v| {
                *use_count.entry(v.0).or_insert(0) += 1;
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                *use_count.entry(v.0).or_insert(0) += 1;
            }
        });
    }

    // Build last-use instruction point map (NOT using LiveInterval.end which
    // includes live-through extensions from backward dataflow). We compute actual
    // last-instruction-use points by scanning all instructions directly.
    // This correctly identifies values whose last explicit use IS the Copy instruction.
    let mut last_use_instr: FxHashMap<u32, u32> = FxHashMap::default();
    let mut copy_program_points: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    if cached_liveness.is_some() {
        let mut pp: u32 = 0;
        for (blk_idx, block) in func.blocks.iter().enumerate() {
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                // Record program point for Copy instructions.
                if matches!(inst, Instruction::Copy { .. }) {
                    copy_program_points.insert((blk_idx, inst_idx), pp);
                }
                // Track last-use for all operands (reads of values).
                for_each_operand_in_instruction(inst, |op| {
                    if let Operand::Value(v) = op {
                        last_use_instr.insert(v.0, pp);
                    }
                });
                for_each_value_use_in_instruction(inst, |v| {
                    last_use_instr.insert(v.0, pp);
                });
                pp += 1;
            }
            // Terminators also use operands.
            for_each_operand_in_terminator(&block.terminator, |op| {
                if let Operand::Value(v) = op {
                    last_use_instr.insert(v.0, pp);
                }
            });
            pp += 1;
        }
    }

    // Collect Copy instructions eligible for aliasing.
    let mut raw_aliases: Vec<(u32, u32)> = Vec::new();
    for (blk_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                let d = dest.0;
                let s = src_val.0;
                // Never alias a multi-defined source (the aliased value must be
                // single-def; multi-def values have complex liveness that makes
                // slot sharing unsafe).
                if multi_def_values.contains(&s) {
                    continue;
                }
                if reg_assigned.contains_key(&d) || reg_assigned.contains_key(&s) {
                    continue;
                }
                // Coalesce if Copy is the sole use of the source, OR if the
                // source's live interval ends at/before this Copy (src is dead after).
                let sole_use = use_count.get(&s).copied().unwrap_or(0) == 1;
                if !sole_use {
                    // Block-local dead-after-copy check: is src unused after this
                    // Copy within this block, AND not used in any other block?
                    // (The global last_use_instr doesn't work because phi sources
                    // may have Copies in multiple predecessor blocks.)
                    let src_use_blocks = use_blocks_map.get(&s);
                    let src_only_in_this_block = src_use_blocks
                        .map(|blks| blks.iter().all(|&b| b == blk_idx))
                        .unwrap_or(true);
                    if !src_only_in_this_block {
                        continue;
                    }
                    // Check: is src used AFTER this Copy instruction in this block?
                    let mut used_after = false;
                    for later_inst in &block.instructions[inst_idx + 1..] {
                        let mut found = false;
                        for_each_operand_in_instruction(later_inst, |op| {
                            if let Operand::Value(v) = op {
                                if v.0 == s { found = true; }
                            }
                        });
                        for_each_value_use_in_instruction(later_inst, |v| {
                            if v.0 == s { found = true; }
                        });
                        if found { used_after = true; break; }
                    }
                    // Also check the terminator.
                    if !used_after {
                        for_each_operand_in_terminator(&block.terminator, |op| {
                            if let Operand::Value(v) = op {
                                if v.0 == s { used_after = true; }
                            }
                        });
                    }
                    if used_after {
                        continue;
                    }
                }

                let src_def_blk = def_block.get(&s).copied();
                let src_in_copy_block = src_def_blk == Some(blk_idx);
                let dest_cross_block = use_blocks_map.get(&d)
                    .map(|blks| blks.iter().any(|&b| b != blk_idx))
                    .unwrap_or(false);

                if src_in_copy_block && dest_cross_block {
                    // Phi-copy pattern: src is defined and killed in this block
                    // (sole use = this copy), but dest is used in other blocks.
                    //
                    // dest may be multi-defined (phi elimination creates one Copy
                    // per predecessor that defines the phi dest). That's fine here
                    // because dest is the slot OWNER (root), not the aliased value.
                    // All definitions of dest write to the same slot, making the
                    // backedge copy a same-slot no-op.
                    //
                    // Reversed aliasing (src → dest) is safe: src gets dest's slot,
                    // which is already live across all of dest's uses. After aliasing,
                    // the copy becomes a same-slot no-op (skipped by generate_copy).
                    // This eliminates the double-slot pattern that arises from phi
                    // elimination in loops with spilled variables.
                    raw_aliases.push((s, d)); // src uses dest's (wider-live) slot
                    continue;
                }

                // Standard same-block coalescing: dest uses src's slot.
                // Dest must not be multi-defined (would make slot-sharing unsafe
                // for the standard direction), and dest's uses must all be in the
                // same block as source's definition.
                if multi_def_values.contains(&d) {
                    continue;
                }
                if dest_cross_block {
                    continue;
                }
                raw_aliases.push((d, s));
            }
        }
    }

    // ── Phi-web coalescing ──────────────────────────────────────────────────
    //
    // Force phi web members to share the same stack slot. For Copy(dest, src)
    // where dest is multi-def (phi dest), coalesce src→dest if src is NOT live
    // in any block where dest is defined by a DIFFERENT Copy.
    let mut phi_web_aliases: FxHashSet<u32> = FxHashSet::default();
    if !std::env::var("CCC_NO_PHI_WEB_COALESCE").is_ok() {
        // Phase 1: Collect all Copy(dest, src) pairs grouped by dest.
        // Also record which blocks contain definitions of each dest.
        let mut phi_copies: FxHashMap<u32, Vec<(u32, usize)>> = FxHashMap::default();
        let mut dest_def_blocks: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for (blk_idx, block) in func.blocks.iter().enumerate() {
            for inst in &block.instructions {
                if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                    let d = dest.0;
                    let s = src_val.0;
                    if !multi_def_values.contains(&d) { continue; }
                    if reg_assigned.contains_key(&s) { continue; }
                    // Skip if already coalesced by per-Copy analysis.
                    if raw_aliases.iter().any(|&(a, _)| a == s) { continue; }
                    phi_copies.entry(d).or_default().push((s, blk_idx));
                    dest_def_blocks.entry(d).or_default().push(blk_idx);
                }
            }
        }

        // Phase 2: For each phi web, check interference and coalesce.
        for (dest_id, sources) in &phi_copies {
            if sources.len() < 2 { continue; }
            if reg_assigned.contains_key(dest_id) { continue; }

            // Get all blocks where dest is defined (by different Copies).
            let def_blks = match dest_def_blocks.get(dest_id) {
                Some(blks) => blks,
                None => continue,
            };

            for &(src_id, src_copy_blk) in sources {
                // Interference check: is src live in any block where dest is
                // defined by a DIFFERENT Copy (i.e., in any def block ≠ src_copy_blk)?
                //
                // "Live in block B" approximated by: src has uses in block B, OR
                // src is defined before B and has uses after B.
                // Simple conservative check: src's use_blocks must not overlap with
                // any of dest's other definition blocks.
                let src_use_blks = use_blocks_map.get(&src_id);
                let interferes = match src_use_blks {
                    Some(blks) => {
                        blks.iter().any(|use_blk| {
                            def_blks.iter().any(|&def_blk| {
                                def_blk == *use_blk && def_blk != src_copy_blk
                            })
                        })
                    }
                    None => false, // no uses = dead, no interference
                };

                if !interferes {
                    raw_aliases.push((src_id, *dest_id));
                    phi_web_aliases.insert(src_id);
                }
            }
        }
    }

    // Build alias map with transitive resolution: follow chains to find root.
    // Safety limit on chain depth guards against pathological cycles.
    const MAX_ALIAS_CHAIN_DEPTH: usize = 100;
    let mut copy_alias: FxHashMap<u32, u32> = FxHashMap::default();
    for (dest_id, src_id) in raw_aliases {
        let mut root = src_id;
        let mut depth = 0;
        while let Some(&parent) = copy_alias.get(&root) {
            root = parent;
            depth += 1;
            if depth > MAX_ALIAS_CHAIN_DEPTH { break; }
        }
        if root != dest_id {
            copy_alias.insert(dest_id, root);
        }
    }

    // Remove aliases where root or dest is an alloca (alloca slots are special).
    let alloca_ids: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|inst| {
            if let Instruction::Alloca { dest, .. } = inst { Some(dest.0) } else { None }
        })
        .collect();
    copy_alias.retain(|dest_id, root_id| {
        !alloca_ids.contains(root_id) && !alloca_ids.contains(dest_id)
    });

    // Remove aliases for InlineAsm output pointer values. InlineAsm Phase 4 reads
    // output pointers from stack slots AFTER the asm executes; if aliased, the
    // root's slot may be reused between the Copy and the InlineAsm, corrupting
    // the pointer read in Phase 4.
    let mut asm_output_ptrs: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::InlineAsm { outputs, .. } = inst {
                for (_, v, _) in outputs {
                    asm_output_ptrs.insert(v.0);
                }
            }
        }
    }
    if !asm_output_ptrs.is_empty() {
        copy_alias.retain(|dest_id, _| !asm_output_ptrs.contains(dest_id));
    }

    (copy_alias, phi_web_aliases)
}

/// Identify values that can skip stack slot allocation because they are
/// produced and consumed in adjacent instructions within the same block.
///
/// A value V defined at instruction I can skip its slot if:
/// 1. V has exactly one use as an Operand (loaded via operand_to_rax/rcx)
/// 2. That use is at instruction I+1 (or in the block terminator if I is last)
/// 3. V is the FIRST Operand of the consumer (loaded first into the accumulator)
/// 4. V is NOT used as a Value reference (ptr in Store/Load, base in GEP, etc.)
/// 5. V is not i128/f128 (these need 16-byte slots with special handling)
/// 6. V is not from a Copy instruction (copy aliasing needs the root's slot)
/// 7. V is not from an Alloca (allocas always need addressable slots)
///
/// The codegen accumulator cache ensures correctness: store_rax_to sets the
/// cache, and the next instruction's operand_to_rax finds V there.
pub(super) fn compute_immediately_consumed(func: &IrFunction, lhs_first_binop: bool) -> FxHashSet<u32> {
    let mut result = FxHashSet::default();

    // First pass: count uses per value (both Operand and Value-ref uses).
    let mut operand_use_count: FxHashMap<u32, u32> = FxHashMap::default();
    let mut has_value_ref_use: FxHashSet<u32> = FxHashSet::default();

    for block in &func.blocks {
        for inst in &block.instructions {
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    *operand_use_count.entry(v.0).or_insert(0) += 1;
                }
            });
            for_each_value_use_in_instruction(inst, |v| {
                has_value_ref_use.insert(v.0);
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                *operand_use_count.entry(v.0).or_insert(0) += 1;
            }
        });
    }

    // Collect all copy-alias roots: values that serve as the slot source for copies.
    // These must keep their slots since aliased copies will use them.
    let mut copy_alias_roots: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Copy { src: Operand::Value(v), .. } = inst {
                copy_alias_roots.insert(v.0);
            }
        }
    }

    // Second pass: check adjacency and first-operand conditions.
    for block in &func.blocks {
        let insts = &block.instructions;
        for (i, inst) in insts.iter().enumerate() {
            let dest = match inst.dest() {
                Some(d) => d,
                None => continue,
            };

            // Only acc-preserving producers are safe: after these execute,
            // the accumulator cache still holds the result. Cache-invalidating
            // instructions (Call, Atomic*, DynAlloca, etc.) clear the cache
            // after store_rax_to, so the next instruction can't find the value.
            if !is_acc_preserving_producer(inst) { continue; }
            // Skip i128/f128 (special 16-byte handling uses emit_load_acc_pair /
            // emit_store_acc_pair which bypass the normal accumulator cache).
            if involves_i128_or_f128(inst) { continue; }
            // Skip if value has Value-ref uses (ptr/base in Store/Load/GEP).
            if has_value_ref_use.contains(&dest.0) { continue; }
            // Skip if value is a copy-alias root (other values share its slot).
            if copy_alias_roots.contains(&dest.0) { continue; }
            // Must have exactly one Operand use.
            let use_cnt = operand_use_count.get(&dest.0).copied().unwrap_or(0);
            if use_cnt != 1 { continue; }
            if use_cnt != 1 { continue; }

            // Check if the single use is in the immediately next instruction
            // or in the block terminator (if this is the last instruction).
            if i + 1 < insts.len() {
                // Use must be in instruction i+1, as the first Operand.
                let next = &insts[i + 1];
                if is_safe_sole_consumer(next, dest.0, lhs_first_binop) {
                    result.insert(dest.0);
                }
            } else {
                // Last instruction: use must be in the terminator, as the sole Operand.
                if is_sole_operand_of_terminator(&block.terminator, dest.0) {
                    result.insert(dest.0);
                }
            }
        }
    }

    result
}

/// Check if an instruction is an "acc-preserving" producer: after execution,
/// the accumulator register cache still holds the result value. Only these
/// instructions can participate in the skip-slot optimization as producers.
///
/// Cache-invalidating instructions (Call, Store, Atomic*, DynAlloca, InlineAsm,
/// etc.) call invalidate_all() after execution, clearing the cache.
fn is_acc_preserving_producer(inst: &Instruction) -> bool {
    matches!(inst,
        Instruction::Load { .. }
        | Instruction::BinOp { .. }
        | Instruction::UnaryOp { .. }
        | Instruction::Cmp { .. }
        | Instruction::Cast { .. }
        | Instruction::GetElementPtr { .. }
        | Instruction::GlobalAddr { .. }
        | Instruction::Select { .. }
        | Instruction::LabelAddr { .. }
    )
}

/// Check if an instruction involves I128/U128/F128 types in any operand position.
/// These use emit_load_acc_pair / emit_store_acc_pair which bypass the normal
/// accumulator cache, so they cannot participate in skip-slot optimization.
fn involves_i128_or_f128(inst: &Instruction) -> bool {
    fn is_wide(ty: IrType) -> bool {
        matches!(ty, IrType::I128 | IrType::U128 | IrType::F128)
    }
    match inst {
        Instruction::Cast { from_ty, to_ty, .. } => is_wide(*from_ty) || is_wide(*to_ty),
        Instruction::UnaryOp { ty, .. } => is_wide(*ty),
        Instruction::BinOp { ty, .. } => is_wide(*ty),
        Instruction::Cmp { ty, .. } => is_wide(*ty),
        Instruction::Load { ty, .. } => is_wide(*ty),
        _ => {
            // For other instructions, just check the result type.
            matches!(inst.result_type(), Some(ty) if is_wide(ty))
        }
    }
}

/// Check if value_id is the sole Operand loaded by the given instruction,
/// with guaranteed loading order (no other operand loaded before it).
///
/// Only single-operand consumers are safe by default: Store (val loaded first),
/// Cast, UnaryOp, Copy. Two-operand instructions (BinOp, Cmp) are excluded on
/// x86/ARM because codegen may load the OTHER operand first (e.g. BinOp's
/// rhs_conflicts path, float Cmp's Lt/Le operand swap). GEP excluded because
/// OverAligned base computation clobbers %rax before offset is loaded.
///
/// When `lhs_first_binop` is true (RISC-V), BinOp and Cmp are also safe when
/// value_id is the lhs operand, because the RISC-V backend unconditionally
/// loads lhs before rhs with no register-direct conflict paths.
fn is_safe_sole_consumer(inst: &Instruction, value_id: u32, lhs_first_binop: bool) -> bool {
    match inst {
        // Store: val is always loaded first via emit_load_operand (operand_to_rax)
        Instruction::Store { val: Operand::Value(v), .. } => v.0 == value_id,
        // Single-operand instructions: loaded via operand_to_rax, no other operand
        Instruction::Cast { src: Operand::Value(v), .. } => v.0 == value_id,
        Instruction::UnaryOp { src: Operand::Value(v), .. } => v.0 == value_id,
        Instruction::Copy { src: Operand::Value(v), .. } => v.0 == value_id,
        // BinOp: safe on architectures that always load lhs first (RISC-V)
        Instruction::BinOp { lhs: Operand::Value(v), .. } if lhs_first_binop => v.0 == value_id,
        // Cmp: safe on architectures that always load lhs first (RISC-V)
        Instruction::Cmp { lhs: Operand::Value(v), .. } if lhs_first_binop => v.0 == value_id,
        // All other instructions: not safe (GEP, Call, Select, etc.)
        _ => false,
    }
}

/// Check if value_id is the sole operand of a block terminator.
fn is_sole_operand_of_terminator(term: &Terminator, value_id: u32) -> bool {
    match term {
        Terminator::Return(Some(Operand::Value(v))) => v.0 == value_id,
        Terminator::CondBranch { cond: Operand::Value(v), .. } => v.0 == value_id,
        Terminator::Switch { val: Operand::Value(v), .. } => v.0 == value_id,
        Terminator::IndirectBranch { target: Operand::Value(v), .. } => v.0 == value_id,
        _ => false,
    }
}
