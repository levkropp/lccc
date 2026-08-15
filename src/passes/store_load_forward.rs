//! Store-to-load forwarding for alloca fields.
//!
//! After inlining and full unrolling, aggregate code looks like:
//! `store v1 -> [agg + 0]; store v2 -> [agg + 8]; ...; load [agg + 0]`
//! spread across a chain of blocks. When the load's address is a
//! compile-time-constant offset of the same alloca root and the same value is
//! stored on every incoming path, the load can be replaced by a Copy of the
//! stored SSA value — turning struct memory traffic into register dataflow
//! (the SROA effect for fully-scalarized aggregates).
//!
//! A forward dataflow computes, per block, the map (root, offset) -> stored
//! value. At control-flow joins only entries on which ALL predecessors agree
//! (same field, same value) survive. Calls, unknown-pointer stores, and
//! overlapping memcpys kill affected entries. Deliberately narrow:
//! alloca-rooted constant-offset addresses and size-matched forwarding only.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::analysis;
use crate::ir::reexports::{Instruction, IrFunction, Operand, Value};

/// A constant byte path from an alloca root: (root value id, byte offset).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FieldPath {
    root: u32,
    offset: i64,
}

/// Resolve a pointer value to its (alloca root, constant byte offset).
/// Follows GEPs with constant offsets and Copies.
fn build_field_paths(func: &IrFunction) -> FxHashMap<u32, FieldPath> {
    let mut paths: FxHashMap<u32, FieldPath> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                paths.insert(dest.0, FieldPath { root: dest.0, offset: 0 });
            }
        }
    }
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                let derived = match inst {
                    Instruction::GetElementPtr { dest, base, offset: Operand::Const(c), .. } => {
                        c.to_i64().and_then(|off| {
                            paths.get(&base.0).map(|p| (dest.0, FieldPath { root: p.root, offset: p.offset + off }))
                        })
                    }
                    Instruction::Copy { dest, src: Operand::Value(src) } => {
                        paths.get(&src.0).copied().map(|p| (dest.0, p))
                    }
                    _ => None,
                };
                if let Some((dest, path)) = derived {
                    if !paths.contains_key(&dest) {
                        paths.insert(dest, path);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    paths
}

fn type_size(ty: IrType) -> i64 {
    use crate::common::types::IrType::*;
    match ty {
        I8 | U8 => 1,
        I16 | U16 => 2,
        I32 | U32 | F32 => 4,
        I64 | U64 | F64 | Ptr => 8,
        _ => 16,
    }
}

/// Apply one instruction's effect to the running store map, rewriting loads in
/// place. Returns false if the instruction made the instruction itself a Copy
/// (a forwarded load) — callers count those.
fn apply_inst(
    inst: &mut Instruction,
    paths: &FxHashMap<u32, FieldPath>,
    map: &mut FxHashMap<FieldPath, (Value, i64)>,
    changed: &mut usize,
) {
    match inst {
        Instruction::Store { val, ptr, ty, .. } => {
            if let Some(fp) = paths.get(&ptr.0).copied() {
                let size = type_size(*ty);
                // Kill overlapping field entries (a wide store covers narrower
                // fields starting within its range).
                map.retain(|ofp, &mut (_v, fs)| {
                    ofp.root != fp.root || ofp.offset + fs <= fp.offset || fp.offset + size <= ofp.offset
                });
                if let Operand::Value(v) = val {
                    map.insert(fp, (*v, size));
                }
            } else {
                map.clear();
            }
        }
        Instruction::Load { dest, ptr, ty, .. } => {
            if let Some(fp) = paths.get(&ptr.0) {
                if let Some(&(stored_v, store_size)) = map.get(fp) {
                    if store_size == type_size(*ty) {
                        *inst = Instruction::Copy { dest: *dest, src: Operand::Value(stored_v) };
                        *changed += 1;
                        return;
                    }
                }
            }
        }
        Instruction::Call { .. }
        | Instruction::CallIndirect { .. }
        | Instruction::InlineAsm { .. }
        | Instruction::AtomicRmw { .. }
        | Instruction::AtomicCmpxchg { .. }
        | Instruction::AtomicLoad { .. }
        | Instruction::AtomicStore { .. } => {
            map.clear();
        }
        Instruction::Memcpy { dest, size, .. } => {
            match paths.get(&dest.0).copied() {
                Some(d) => {
                    let sz = *size as i64;
                    map.retain(|fp, &mut (_v, fs)| {
                        fp.root != d.root || fp.offset + fs <= d.offset || d.offset + sz <= fp.offset
                    });
                }
                None => map.clear(),
            }
        }
        _ => {}
    }
}

/// Run store-to-load forwarding on a function. Returns the number of loads replaced.
pub(crate) fn run(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_NO_SL_FORWARD").is_ok() {
        return 0;
    }
    let paths = build_field_paths(func);
    if paths.is_empty() {
        return 0;
    }

    let label_to_idx = analysis::build_label_map(func);
    let (preds, _succs) = analysis::build_cfg(func, &label_to_idx);
    let n = func.blocks.len();

    // Forward dataflow: IN[b] = entries all predecessors agree on; OUT[b] =
    // the block's effect applied to IN[b]. Iterate to a fixpoint.
    let mut in_map: Vec<FxHashMap<FieldPath, (Value, i64)>> = (0..n).map(|_| FxHashMap::default()).collect();
    let mut out_map: Vec<FxHashMap<FieldPath, (Value, i64)>> = (0..n).map(|_| FxHashMap::default()).collect();
    let mut worklist: FxHashSet<usize> = (0..n).collect();
    while !worklist.is_empty() {
        let b = *worklist.iter().next().unwrap();
        worklist.remove(&b);
        // IN[b] = intersection of OUT[preds]: entries with identical values.
        let mut acc: Option<FxHashMap<FieldPath, (Value, i64)>> = None;
        for &p in preds.row(b).iter() {
            let p = p as usize;
            acc = Some(match acc {
                None => out_map[p].clone(),
                Some(mut a) => {
                    a.retain(|fp, v| out_map[p].get(fp) == Some(v));
                    a
                }
            });
        }
        let in_b = acc.unwrap_or_default();
        if in_b == in_map[b] {
            // Even if IN is unchanged, OUT may need one initial computation.
            if !out_map[b].is_empty() || in_b.is_empty() {
                continue;
            }
        }
        in_map[b] = in_b.clone();
        let mut m = in_b;
        for inst in &mut func.blocks[b].instructions {
            // Analysis only here; rewriting happens in the second pass.
            let mut dummy = 0;
            apply_inst_analysis(inst, &paths, &mut m, &mut dummy);
        }
        if m != out_map[b] {
            out_map[b] = m;
            for &s in _succs.row(b).iter() {
                worklist.insert(s as usize);
            }
        }
    }

    // Second pass: rewrite loads using IN maps.
    let mut changes = 0;
    for b in 0..n {
        let mut m = in_map[b].clone();
        for inst in &mut func.blocks[b].instructions {
            apply_inst(inst, &paths, &mut m, &mut changes);
        }
    }
    changes
}

/// The analysis-only variant of apply_inst (no rewriting).
fn apply_inst_analysis(
    inst: &Instruction,
    paths: &FxHashMap<u32, FieldPath>,
    map: &mut FxHashMap<FieldPath, (Value, i64)>,
    changed: &mut usize,
) {
    let _ = changed;
    match inst {
        Instruction::Store { ptr, ty, val, .. } => {
            if let Some(fp) = paths.get(&ptr.0).copied() {
                let size = type_size(*ty);
                map.retain(|ofp, &mut (_v, fs)| {
                    ofp.root != fp.root || ofp.offset + fs <= fp.offset || fp.offset + size <= ofp.offset
                });
                if let Operand::Value(v) = val {
                    map.insert(fp, (*v, size));
                }
            } else {
                map.clear();
            }
        }
        Instruction::Call { .. }
        | Instruction::CallIndirect { .. }
        | Instruction::InlineAsm { .. }
        | Instruction::AtomicRmw { .. }
        | Instruction::AtomicCmpxchg { .. }
        | Instruction::AtomicLoad { .. }
        | Instruction::AtomicStore { .. } => {
            map.clear();
        }
        Instruction::Memcpy { dest, size, .. } => match paths.get(&dest.0).copied() {
            Some(d) => {
                let sz = *size as i64;
                map.retain(|fp, &mut (_v, fs)| {
                    fp.root != d.root || fp.offset + fs <= d.offset || d.offset + sz <= fp.offset
                });
            }
            None => map.clear(),
        },
        _ => {}
    }
}
