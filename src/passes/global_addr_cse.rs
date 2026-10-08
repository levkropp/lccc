//! Global-address materialization CSE (substitution-based).
//!
//! The frontend emits one `GlobalAddr` per global access, so a loop over
//! several static arrays re-materializes the same `adrp`+`add` pair per
//! access site, each a distinct SSA value that pins its own register or
//! stack slot for the whole region (fannkuch carried 10+ live copies of
//! three array bases). A global's address is a linker constant, so every
//! same-symbol materialization is interchangeable: rewrite all uses of a
//! duplicate to the canonical value and delete the duplicate instruction.
//!
//! Unlike GVN's Copy-insertion this is pure substitution, so it cannot
//! create the pointer-valued Copy chains whose base register can be stale
//! at fold points (the hazard that keeps GEP CSE disabled).
//!
//! Canonical choice is dominance-safe by construction:
//! - an entry-block materialization is canonical for the whole function
//!   (the entry block dominates every reachable block);
//! - otherwise the first materialization within a block is canonical for
//!   later same-block duplicates (same-block-earlier dominates all of the
//!   duplicate's uses).
//!
//! Cross-block merging between two non-entry blocks is not attempted.
//! CCC_NO_GADDR_CSE disables for A/B.

use crate::common::fx_hash::FxHashMap;
use crate::ir::reexports::{Instruction, IrFunction, Value};

pub(crate) fn run(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_NO_GADDR_CSE").is_ok() || func.blocks.is_empty() {
        return 0;
    }

    // Entry-block canonicals: name -> value.
    let mut entry_canonical: FxHashMap<&str, Value> = FxHashMap::default();
    for inst in &func.blocks[0].instructions {
        if let Instruction::GlobalAddr { dest, name } = inst {
            entry_canonical.entry(name.as_str()).or_insert(*dest);
        }
    }

    // Duplicate value -> canonical value, plus the (block, index) of each
    // duplicate instruction to delete.
    let mut subst: FxHashMap<u32, u32> = FxHashMap::default();
    let mut dups: Vec<(usize, usize)> = Vec::new();

    for (bi, block) in func.blocks.iter().enumerate() {
        // Per-block canonicals for symbols the entry block never materialized.
        let mut block_canonical: FxHashMap<&str, Value> = FxHashMap::default();
        for (ii, inst) in block.instructions.iter().enumerate() {
            let Instruction::GlobalAddr { dest, name } = inst else { continue };
            if let Some(&canon) = entry_canonical.get(name.as_str()) {
                if canon != *dest {
                    subst.insert(dest.0, canon.0);
                    dups.push((bi, ii));
                }
                continue;
            }
            match block_canonical.get(name.as_str()) {
                Some(&canon) => {
                    subst.insert(dest.0, canon.0);
                    dups.push((bi, ii));
                }
                None => {
                    block_canonical.insert(name.as_str(), *dest);
                }
            }
        }
    }

    if dups.is_empty() {
        return 0;
    }

    // Rewrite all uses of duplicate values function-wide, then delete the
    // duplicate instructions (indices collected in ascending order per block;
    // remove in descending order per block to keep indices valid).
    for block in func.blocks.iter_mut() {
        for inst in &mut block.instructions {
            super::tail_call_elim::replace_values_in_inst(inst, &subst);
        }
        super::tail_call_elim::replace_values_in_terminator(&mut block.terminator, &subst);
    }
    let n = dups.len();
    dups.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    for (bi, ii) in dups {
        let block = &mut func.blocks[bi];
        block.instructions.remove(ii);
        if !block.source_spans.is_empty() && block.source_spans.len() > ii {
            block.source_spans.remove(ii);
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;
    use crate::ir::reexports::{BasicBlock, BlockId, Operand, Terminator};

    fn make_func(blocks: Vec<BasicBlock>) -> IrFunction {
        IrFunction {
            name: "test".to_string(),
            params: vec![],
            return_type: IrType::I32,
            blocks,
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 10,
            next_label: 0,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
            loop_promoted_f64_values: Vec::new(),
        }
    }

    #[test]
    fn merges_same_symbol_in_block() {
        let mut func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::GlobalAddr { dest: Value(1), name: "g".to_string() },
                Instruction::GlobalAddr { dest: Value(2), name: "g".to_string() },
                Instruction::Store {
                    val: Operand::Value(Value(0)),
                    ptr: Value(2),
                    ty: IrType::I32,
                    seg_override: crate::common::types::AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(None),
            source_spans: Vec::new(),
        }]);
        let n = run(&mut func);
        assert_eq!(n, 1);
        // Duplicate deleted; the store now uses the canonical value.
        assert_eq!(func.blocks[0].instructions.len(), 2);
        match &func.blocks[0].instructions[1] {
            Instruction::Store { ptr, .. } => assert_eq!(ptr.0, 1),
            _ => panic!("expected store"),
        }
    }

    #[test]
    fn entry_block_canonical_wins_cross_block() {
        let mut func = make_func(vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::GlobalAddr { dest: Value(1), name: "g".to_string() },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: Vec::new(),
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![
                    Instruction::GlobalAddr { dest: Value(5), name: "g".to_string() },
                    Instruction::Load {
                        dest: Value(6),
                        ptr: Value(5),
                        ty: IrType::I32,
                        seg_override: crate::common::types::AddressSpace::Default,
                    },
                ],
                terminator: Terminator::Return(None),
                source_spans: Vec::new(),
            },
        ]);
        let n = run(&mut func);
        assert_eq!(n, 1);
        assert_eq!(func.blocks[1].instructions.len(), 1);
        match &func.blocks[1].instructions[0] {
            Instruction::Load { ptr, .. } => assert_eq!(ptr.0, 1),
            _ => panic!("expected load"),
        }
    }

    #[test]
    fn distinct_symbols_untouched() {
        let mut func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::GlobalAddr { dest: Value(1), name: "a".to_string() },
                Instruction::GlobalAddr { dest: Value(2), name: "b".to_string() },
            ],
            terminator: Terminator::Return(None),
            source_spans: Vec::new(),
        }]);
        assert_eq!(run(&mut func), 0);
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }
}
