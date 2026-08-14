//! Promote simple loop-local memory recurrences to SSA.
//!
//! Recognizes a natural loop containing one load and one store of an invariant
//! scalar address, with the store in the sole latch.  When no other operation in
//! the loop can alias that address, the load is moved to the preheader, carried
//! by a phi, and the final value is stored once on loop exit.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::analysis::CfgAnalysis;
use crate::ir::reexports::{Instruction, IrFunction, Operand, Value};
use super::loop_analysis;

#[derive(Clone, Copy)]
struct Path { root: u32, offset: i64 }

fn pointer_paths(func: &IrFunction) -> FxHashMap<u32, Path> {
    let mut paths = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                paths.insert(dest.0, Path { root: dest.0, offset: 0 });
            }
        }
    }
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                let derived = match inst {
                    Instruction::GetElementPtr { dest, base, offset: Operand::Const(c), .. } => {
                        c.to_i64().and_then(|off| paths.get(&base.0).map(|p| (dest.0, Path { root: p.root, offset: p.offset + off })))
                    }
                    Instruction::Copy { dest, src: Operand::Value(src) } =>
                        paths.get(&src.0).copied().map(|p| (dest.0, p)),
                    _ => None,
                };
                if let Some((dest, path)) = derived {
                    if !paths.contains_key(&dest) { paths.insert(dest, path); changed = true; }
                }
            }
        }
        if !changed { break; }
    }
    paths
}

fn byte_size(ty: IrType) -> i64 {
    match ty {
        IrType::I8 | IrType::U8 => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::I64 | IrType::U64 | IrType::F64 | IrType::Ptr => 8,
        _ => 16,
    }
}

fn disjoint(paths: &FxHashMap<u32, Path>, a: Value, a_ty: IrType, b: Value, b_ty: IrType) -> bool {
    let (Some(pa), Some(pb)) = (paths.get(&a.0), paths.get(&b.0)) else { return false };
    if pa.root != pb.root { return true; }
    let (ae, be) = (pa.offset + byte_size(a_ty), pb.offset + byte_size(b_ty));
    ae <= pb.offset || be <= pa.offset
}

pub(crate) fn run(func: &mut IrFunction) -> usize {
    let cfg = CfgAnalysis::build(func);
    let loops = loop_analysis::merge_loops_by_header(loop_analysis::find_natural_loops(
        cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom));
    if loops.is_empty() { return 0; }
    let paths = pointer_paths(func);
    let volatile_roots: FxHashSet<u32> = func.blocks.iter().flat_map(|b| &b.instructions)
        .filter_map(|inst| match inst {
            Instruction::Alloca { dest, volatile: true, .. } => Some(dest.0),
            _ => None,
        }).collect();
    let mut def_block = FxHashMap::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() { def_block.insert(dest.0, bi); }
        }
    }

    for lp in loops {
        let Some(preheader) = loop_analysis::find_preheader(lp.header, &lp.body, &cfg.preds) else { continue };
        let latches: Vec<usize> = cfg.preds.row(lp.header).iter().map(|&v| v as usize)
            .filter(|b| lp.body.contains(b)).collect();
        if latches.len() != 1 { continue; }
        let latch = latches[0];
        let exits: Vec<(usize, usize)> = lp.body.iter().flat_map(|&from|
            cfg.succs.row(from).iter().map(move |&to| (from, to as usize)))
            .filter(|(_, to)| !lp.body.contains(to)).collect();
        if exits.len() != 1 { continue; }
        let (exit_from, exit_block) = exits[0];
        if cfg.preds.row(exit_block).iter().any(|&p| p as usize != exit_from) { continue; }

        let mut loads: FxHashMap<u32, Vec<(usize, usize, Value, IrType)>> = FxHashMap::default();
        let mut stores: FxHashMap<u32, Vec<(usize, usize, Operand, IrType)>> = FxHashMap::default();
        let mut has_call_or_memcpy = false;
        for &bi in &lp.body {
            for (ii, inst) in func.blocks[bi].instructions.iter().enumerate() {
                match inst {
                    Instruction::Load { dest, ptr, ty, .. } => loads.entry(ptr.0).or_default().push((bi, ii, *dest, *ty)),
                    Instruction::Store { val, ptr, ty, .. } => stores.entry(ptr.0).or_default().push((bi, ii, val.clone(), *ty)),
                    Instruction::Call { .. } | Instruction::CallIndirect { .. } | Instruction::Memcpy { .. } => has_call_or_memcpy = true,
                    _ => {}
                }
            }
        }
        if has_call_or_memcpy { continue; }

        for (&ptr_id, load_list) in &loads {
            let Some(store_list) = stores.get(&ptr_id) else { continue };
            if load_list.len() != 1 || store_list.len() != 1 { continue; }
            let (load_b, load_i, load_dest, load_ty) = load_list[0];
            let (store_b, store_i, store_val, store_ty) = &store_list[0];
            if *store_b != latch || load_ty != *store_ty { continue; }
            if def_block.get(&ptr_id).is_some_and(|b| lp.body.contains(b)) { continue; }
            let ptr = Value(ptr_id);
            if paths.get(&ptr_id).is_some_and(|p| volatile_roots.contains(&p.root)) { continue; }

            let mut alias = false;
            for (&other_ptr, other_stores) in &stores {
                if other_ptr == ptr_id { continue; }
                for (_, _, _, other_ty) in other_stores {
                    if !disjoint(&paths, ptr, load_ty, Value(other_ptr), *other_ty) { alias = true; }
                }
            }
            if alias { continue; }

            let init = Value(func.next_value_id); func.next_value_id += 1;
            let phi = Value(func.next_value_id); func.next_value_id += 1;
            func.blocks[preheader].instructions.push(Instruction::Load {
                dest: init, ptr, ty: load_ty, seg_override: crate::common::types::AddressSpace::Default,
            });
            let pre_label = func.blocks[preheader].label;
            let latch_label = func.blocks[latch].label;
            func.blocks[lp.header].instructions.insert(0, Instruction::Phi {
                dest: phi, ty: load_ty,
                incoming: vec![(Operand::Value(init), pre_label), (store_val.clone(), latch_label)],
            });
            if load_ty == IrType::F64 {
                func.loop_promoted_f64_values.push(phi);
            }

            let mut map = FxHashMap::default(); map.insert(load_dest.0, phi.0);
            for &bi in &lp.body {
                for inst in &mut func.blocks[bi].instructions {
                    super::tail_call_elim::replace_values_in_inst(inst, &map);
                }
            }
            func.blocks[load_b].instructions.remove(load_i + usize::from(load_b == lp.header));
            let adjusted_store_i = store_i + usize::from(*store_b == lp.header)
                - usize::from(*store_b == load_b && *store_i > load_i);
            func.blocks[*store_b].instructions.remove(adjusted_store_i);
            func.blocks[exit_block].instructions.insert(0, Instruction::Store {
                val: Operand::Value(phi), ptr, ty: load_ty,
                seg_override: crate::common::types::AddressSpace::Default,
            });
            return 1;
        }
    }
    0
}

/// Mark one ordinary F64 add-reduction for the dedicated ARM loop register
/// pool. This is intentionally enabled only after this pass has already found
/// a promotable memory recurrence in the function, keeping the policy narrow.
pub(crate) fn mark_f64_add_reduction(func: &mut IrFunction) -> usize {
    if func.loop_promoted_f64_values.is_empty() { return 0; }
    let mut add_defs: FxHashMap<u32, (Operand, Operand)> = FxHashMap::default();
    for block in &func.blocks { for inst in &block.instructions {
        if let Instruction::BinOp { dest, op: crate::ir::reexports::IrBinOp::Add, lhs, rhs, ty: IrType::F64 } = inst {
            add_defs.insert(dest.0, (lhs.clone(), rhs.clone()));
        }
    }}
    for block in &func.blocks { for inst in &block.instructions {
        let Instruction::Phi { dest, ty: IrType::F64, incoming } = inst else { continue };
        if func.loop_promoted_f64_values.iter().any(|v| v.0 == dest.0) { continue; }
        let has_zero = incoming.iter().any(|(op, _)| matches!(op, Operand::Const(crate::ir::reexports::IrConst::F64(v)) if *v == 0.0));
        if !has_zero { continue; }
        let is_reduction = incoming.iter().filter_map(|(op, _)| match op { Operand::Value(v) => Some(v.0), _ => None })
            .any(|back| add_defs.get(&back).is_some_and(|(lhs, rhs)|
                matches!(lhs, Operand::Value(v) if v.0 == dest.0)
                || matches!(rhs, Operand::Value(v) if v.0 == dest.0)));
        if is_reduction {
            func.loop_promoted_f64_values.push(*dest);
            return 1;
        }
    }}
    0
}
