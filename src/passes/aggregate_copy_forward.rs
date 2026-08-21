//! Forward reads from short-lived aggregate copies to their original storage.
//!
//! Inlining structure-returning functions often leaves IR like:
//! `memcpy(tmp, object); load(tmp.field)`.  When `tmp` never escapes or receives
//! another write, copying the complete aggregate is unnecessary.  This pass is
//! deliberately conservative: the copy and every read must be in the same
//! block, with the copy preceding the reads.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::reexports::{Instruction, IrFunction, Operand, Value};

fn type_size(ty: crate::common::types::IrType) -> i64 {
    use crate::common::types::IrType::*;
    match ty { I8 | U8 => 1, I16 | U16 => 2, I32 | U32 | F32 => 4,
        I64 | U64 | F64 | Ptr => 8, _ => 16 }
}

/// Remove stores to fields of non-escaping stack aggregates that are never read.
fn eliminate_dead_aggregate_field_stores(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_NO_AGG_DSE").is_ok() {
        return 0;
    }
    // Track aggregate roots separately from precise paths. Loop pointer phis
    // commonly merge offsets 0 and 48; they still share the same allocation
    // root even though `pointer_paths` deliberately rejects the differing paths.
    //
    // The suffix is the byte offset within the aggregate when it is statically
    // known. SUFFIX_UNKNOWN marks pointers whose offset cannot be proven
    // (variable GEP offset, or a phi merging different offsets): a load through
    // such a pointer may touch ANY field, so the whole aggregate counts as read
    // and no field store may be deleted. (Collapsing an unknown offset to 0
    // previously recorded only field 0 as read, deleting the init stores of
    // `long a[N] = {...}` when the array was read via a marching pointer.)
    const SUFFIX_UNKNOWN: i64 = i64::MIN;
    let mut root_suffix: FxHashMap<u32, (u32, i64)> = FxHashMap::default();
    let mut alloca_size: FxHashMap<u32, i64> = FxHashMap::default();
    // Marching-pointer phis: phi dest -> (start suffix, constant step). A load
    // through such a phi covers a one-sided range ([start, size) for a positive
    // step, [0, start] for a negative one) instead of the whole aggregate, so
    // stores outside the marched region remain eliminable.
    let mut march_range: FxHashMap<u32, (i64, i64)> = FxHashMap::default();
    for block in &func.blocks { for inst in &block.instructions {
        if let Instruction::Alloca { dest, size, .. } = inst {
            root_suffix.insert(dest.0, (dest.0, 0));
            alloca_size.insert(dest.0, *size as i64);
        }
    }}
    loop {
        let mut changed = false;
        for block in &func.blocks { for inst in &block.instructions {
            let derived = match inst {
                Instruction::GetElementPtr { dest, base, offset, .. } => {
                    // A constant-offset GEP off a marched pointer is marched
                    // with the same period, shifted by the offset.
                    if let (Some(&m), Operand::Const(c)) = (march_range.get(&base.0), offset) {
                        if let Some(k) = c.to_i64() {
                            if march_range.insert(dest.0, (m.0 + k, m.1)).is_none() { changed = true; }
                        }
                    }
                    root_suffix.get(&base.0).copied().map(|(root, suffix)| {
                        let next = match offset {
                            _ if suffix == SUFFIX_UNKNOWN => SUFFIX_UNKNOWN,
                            Operand::Const(c) => suffix + c.to_i64().unwrap_or(0),
                            Operand::Value(_) => SUFFIX_UNKNOWN,
                        };
                        (dest.0, (root, next))
                    })
                }
                Instruction::Copy { dest, src: Operand::Value(src) } => {
                    if let Some(&m) = march_range.get(&src.0) {
                        if march_range.insert(dest.0, m).is_none() { changed = true; }
                    }
                    root_suffix.get(&src.0).copied().map(|p| (dest.0, p))
                }
                Instruction::Phi { dest, incoming, .. } => {
                    let vals: Vec<(u32, i64)> = incoming.iter().filter_map(|(op, _)| match op {
                        Operand::Value(v) => root_suffix.get(&v.0).copied(), _ => None }).collect();
                    if !vals.is_empty() && vals.iter().all(|p| p.0 == vals[0].0) {
                        let suffix = if vals.iter().all(|p| p.1 == vals[0].1) { vals[0].1 } else { SUFFIX_UNKNOWN };
                        if suffix == SUFFIX_UNKNOWN && !march_range.contains_key(&dest.0) {
                            // Differing suffixes: check for a marching-pointer
                            // phi — non-backedge incoming carry one precise
                            // start suffix; backedges are GEP(this phi, k).
                            let is_self_gep = |v: &Value| func.blocks.iter().flat_map(|b| &b.instructions).any(|i| matches!(i,
                                Instruction::GetElementPtr { dest: d, base, offset: Operand::Const(_), .. }
                                    if *d == *v && base.0 == dest.0));
                            let self_gep_step = |v: &Value| func.blocks.iter().flat_map(|b| &b.instructions).find_map(|i| {
                                if let Instruction::GetElementPtr { dest: d, base, offset: Operand::Const(c), .. } = i {
                                    if *d == *v && base.0 == dest.0 { return c.to_i64(); }
                                }
                                None
                            });
                            let mut start: Option<i64> = None;
                            let mut step: i64 = 0;
                            let mut ok = true;
                            for (op, _) in incoming {
                                let Operand::Value(v) = op else { ok = false; break };
                                if is_self_gep(v) {
                                    match self_gep_step(v) {
                                        Some(k) if k != 0 && (step == 0 || step == k) => step = k,
                                        _ => { ok = false; break; }
                                    }
                                } else if let Some(&(_, s)) = root_suffix.get(&v.0) {
                                    if s == SUFFIX_UNKNOWN || (start.is_some() && start != Some(s)) {
                                        ok = false; break;
                                    }
                                    start = Some(s);
                                } else {
                                    ok = false; break;
                                }
                            }
                            if ok {
                                if let (Some(s0), true) = (start, step != 0) {
                                    march_range.insert(dest.0, (s0, step));
                                    changed = true;
                                }
                            }
                        }
                        Some((dest.0, (vals[0].0, suffix)))
                    } else { None }
                }
                _ => None,
            };
            if let Some((dest, path)) = derived {
                match root_suffix.get(&dest) {
                    None => { root_suffix.insert(dest, path); changed = true; }
                    Some(&existing) if existing != path => {
                        // Later information (e.g. a loop backedge not yet mapped
                        // on the first pass) contradicts the recorded suffix:
                        // downgrade to the imprecise root-only form. UNKNOWN is
                        // absorbing, so the fixpoint still terminates.
                        let downgraded = (existing.0, SUFFIX_UNKNOWN);
                        if existing != downgraded {
                            root_suffix.insert(dest, downgraded);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }}
        if !changed { break; }
    }
    let volatile_roots: FxHashSet<u32> = func.blocks.iter().flat_map(|b| &b.instructions)
        .filter_map(|inst| match inst { Instruction::Alloca { dest, volatile: true, .. } => Some(dest.0), _ => None })
        .collect();
    let mut escaping = FxHashSet::default();
    let mut loaded: FxHashMap<u32, Vec<(i64, i64)>> = FxHashMap::default();
    // Periodic loads through marching pointers: (root, first offset, period, size).
    let mut periodic_loads: Vec<(u32, i64, i64, i64)> = Vec::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Load { ptr, ty, .. } = inst {
                if let Some((root, suffix)) = root_suffix.get(&ptr.0) {
                    if *suffix == SUFFIX_UNKNOWN {
                        match march_range.get(&ptr.0).copied() {
                            // Marching-pointer load with positive period p:
                            // touches [l0 + p*j, l0 + p*j + size) for j >= 0.
                            Some((l0, p)) if p > 0 && l0 >= 0 => {
                                periodic_loads.push((*root, l0, p, type_size(*ty)));
                            }
                            _ => {
                                // Unknown offset: the load may touch any field.
                                let whole = alloca_size.get(root).copied().unwrap_or(i64::MAX);
                                loaded.entry(*root).or_default().push((0, whole));
                            }
                        }
                    } else {
                        loaded.entry(*root).or_default().push((*suffix, type_size(*ty)));
                    }
                }
            }
            match inst {
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    for arg in &info.args { if let Operand::Value(v) = arg {
                        if let Some((root, _)) = root_suffix.get(&v.0) { escaping.insert(*root); }
                    }}
                }
                Instruction::Memcpy { dest, src, .. } => {
                    if let Some((root, _)) = root_suffix.get(&dest.0) { escaping.insert(*root); }
                    if let Some((root, _)) = root_suffix.get(&src.0) { escaping.insert(*root); }
                }
                Instruction::Store { val: Operand::Value(v), .. } => {
                    if let Some((root, _)) = root_suffix.get(&v.0) { escaping.insert(*root); }
                }
                _ => {}
            }
        }
    }
    let mut changes = 0;
    for block in &mut func.blocks {
        let old = std::mem::take(&mut block.instructions);
        let old_spans = std::mem::take(&mut block.source_spans);
        let has_spans = !old_spans.is_empty();
        let mut kept = Vec::with_capacity(old.len());
        let mut spans = Vec::with_capacity(old_spans.len());
        for (ii, inst) in old.into_iter().enumerate() {
            let dead = if let Instruction::Store { ptr, ty, .. } = &inst {
                if let Some((root, off)) = root_suffix.get(&ptr.0) {
                    // A store through an unknown-offset pointer cannot be
                    // classified as a field store — keep it.
                    if *off == SUFFIX_UNKNOWN {
                        false
                    } else if !volatile_roots.contains(root) && !escaping.contains(root) {
                        let size = type_size(*ty);
                        let hit_precise = loaded.get(root).is_some_and(|ranges| ranges.iter().any(|(lo, ls)| *off < lo + ls && *lo < *off + size));
                        let whole = alloca_size.get(root).copied().unwrap_or(i64::MAX);
                        let hit_periodic = periodic_loads.iter().any(|&(r, l0, p, lsz)| {
                            r == *root && periodic_overlap(l0, p, lsz, *off, size, whole)
                        });
                        !(hit_precise || hit_periodic)
                    } else { false }
                } else { false }
            } else { false };
            if dead { changes += 1; continue; }
            kept.push(inst);
            if has_spans { spans.push(old_spans[ii]); }
        }
        block.instructions = kept;
        if has_spans { block.source_spans = spans; }
    }
    changes
}

/// Does a periodic load covering [l0 + p*j, l0 + p*j + lsz) for integer j >= 0
/// (bounded to the alloca's [0, size)) overlap the store range [off, off+ssz)?
/// p must be positive.
fn periodic_overlap(l0: i64, p: i64, lsz: i64, off: i64, ssz: i64, size: i64) -> bool {
    // Smallest j >= 0 with l0 + p*j + lsz > off.
    let j_min = ((off - l0 - lsz).div_euclid(p) + 1).max(0);
    // Largest j with l0 + p*j < off + ssz.
    let j_max = (off + ssz - l0 - 1).div_euclid(p);
    j_min <= j_max && l0 + p * j_min < size
}

#[derive(Clone)]
struct CopyCandidate {
    block: usize,
    inst: usize,
    source: Value,
    source_root: u32,
}

/// Return the alloca root and GEP path for pointer values derived from allocas.
fn pointer_paths(func: &IrFunction) -> FxHashMap<u32, (u32, Vec<(Operand, crate::common::types::IrType)>)> {
    let mut paths = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                paths.insert(dest.0, (dest.0, Vec::new()));
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::GetElementPtr { dest, base, offset, ty } => {
                        if paths.contains_key(&dest.0) { continue; }
                        if let Some((root, parent)) = paths.get(&base.0).cloned() {
                            let mut path = parent;
                            path.push((offset.clone(), *ty));
                            paths.insert(dest.0, (root, path));
                            changed = true;
                        }
                    }
                    Instruction::Copy { dest, src: Operand::Value(src) } => {
                        if !paths.contains_key(&dest.0) {
                            if let Some(path) = paths.get(&src.0).cloned() {
                                paths.insert(dest.0, path);
                                changed = true;
                            }
                        }
                    }
                    Instruction::Phi { dest, incoming, .. } => {
                        if paths.contains_key(&dest.0) { continue; }
                        let mut common = None;
                        let mut compatible = true;
                        for (op, _) in incoming {
                            match op {
                                Operand::Const(c) if c.to_i64() == Some(0) => {}
                                Operand::Value(v) => if let Some(path) = paths.get(&v.0) {
                                    if common.as_ref().is_some_and(|p: &(u32, Vec<(Operand, crate::common::types::IrType)>)|
                                        p.0 != path.0 || p.1.len() != path.1.len()) {
                                        compatible = false;
                                        break;
                                    }
                                    common = Some(path.clone());
                                } else { compatible = false; break; },
                                _ => { compatible = false; break; }
                            }
                        }
                        if compatible {
                            if let Some(path) = common { paths.insert(dest.0, path); changed = true; }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    paths
}

/// Redirect construction of a store-only temporary aggregate into the final
/// memcpy destination.  This turns `build(tmp); memcpy(dst, tmp)` into
/// `build(dst)` when every use of `tmp` is a same-block GEP/store or that copy.
/// Hoist the definition of `dest` (and its movable in-block def chain) to the
/// top of block `bi`, so redirected uses precede neither their new pointer's
/// definition nor any intermediate computation. Only pure address/value
/// computations (GEP/BinOp/Cast/Copy) are moved; anything else aborts the
/// hoist and the caller rejects the candidate.
fn try_hoist_def_chain(func: &mut IrFunction, bi: usize, dest: Value) -> bool {
    // Collect the in-block def chain (dependency order).
    let mut chain: Vec<usize> = Vec::new(); // instruction indices
    let mut worklist = vec![dest.0];
    let mut seen = FxHashSet::default();
    // Same-block terminal definitions (allocas, address constants) the chain
    // bottoms out in: available for the whole block, so they terminate the
    // chain — but the reinsertion point must stay after their positions.
    let mut terminal_dests: FxHashSet<u32> = FxHashSet::default();
    while let Some(v) = worklist.pop() {
        if !seen.insert(v) {
            continue;
        }
        let mut def_pos = None;
        for (di, inst) in func.blocks[bi].instructions.iter().enumerate() {
            if inst.dest().is_some_and(|d| d.0 == v) {
                def_pos = Some(di);
                break;
            }
        }
        let Some(di) = def_pos else { continue }; // defined elsewhere: available
        let inst = &func.blocks[bi].instructions[di];
        if matches!(
            inst,
            Instruction::Alloca { .. }
                | Instruction::ParamRef { .. }
                | Instruction::GlobalAddr { .. }
                | Instruction::LabelAddr { .. }
        ) {
            terminal_dests.insert(v);
            continue;
        }
        let movable = matches!(
            inst,
            Instruction::GetElementPtr { .. }
                | Instruction::BinOp { .. }
                | Instruction::Cast { .. }
                | Instruction::Copy { .. }
        );
        if !movable {
            return false;
        }
        chain.push(di);
        let mut ops = Vec::new();
        inst.for_each_used_value(|u| ops.push(u));
        for u in ops {
            worklist.push(u);
        }
    }
    // Remove from the block (descending index order), then reinsert in
    // dependency order after the leading Alloca/Phi instructions AND after
    // any same-block terminal definitions (a mid-block Alloca the chain
    // references must still precede the hoisted instructions).
    chain.sort_unstable();
    let mut moved = Vec::with_capacity(chain.len());
    for &di in chain.iter().rev() {
        moved.push(func.blocks[bi].instructions.remove(di));
    }
    moved.reverse();
    // Keep source_spans aligned if present (drop them; debug info only).
    let spans_present = !func.blocks[bi].source_spans.is_empty();
    if spans_present {
        func.blocks[bi].source_spans.clear();
    }
    let mut ins = 0;
    while ins < func.blocks[bi].instructions.len() {
        match &func.blocks[bi].instructions[ins] {
            Instruction::Alloca { .. } | Instruction::Phi { .. } => ins += 1,
            _ => break,
        }
    }
    for (idx, inst) in func.blocks[bi].instructions.iter().enumerate() {
        if let Some(d) = inst.dest() {
            if terminal_dests.contains(&d.0) {
                ins = ins.max(idx + 1);
            }
        }
    }
    for inst in moved.into_iter().rev() {
        func.blocks[bi].instructions.insert(ins, inst);
    }
    true
}

fn forward_store_only_temporaries(func: &mut IrFunction) -> usize {
    let paths = pointer_paths(func);
    let roots: FxHashSet<u32> = paths.iter()
        .filter_map(|(&v, (root, path))| (v == *root && path.is_empty()).then_some(v))
        .collect();
    let mut copies: FxHashMap<u32, (usize, usize, Value)> = FxHashMap::default();
    let mut duplicate = FxHashSet::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Memcpy { dest, src, .. } = inst {
                let Some((root, path)) = paths.get(&src.0) else { continue };
                if !path.is_empty() || !roots.contains(root) { continue; }
                if paths.get(&dest.0).is_some_and(|p| p.0 == *root) { continue; }
                if copies.insert(*root, (bi, ii, *dest)).is_some() { duplicate.insert(*root); }
            }
        }
    }
    for root in duplicate { copies.remove(&root); }
    if std::env::var("CCC_DEBUG_AGG").is_ok() {
        eprintln!("[AGG] func={} candidates={:?}", func.name, copies.keys().collect::<Vec<_>>());
    }

    // The memcpy dest must be defined before any use we redirect to it.
    // A dest computed immediately before the copy (e.g. a GEP for the target
    // field in a fully-unrolled loop body) would leave redirected stores with
    // a use-before-def. When the dest's definition is a pure address/value
    // computation, hoist it (and its def chain) to the top of the block so the
    // redirect is well-ordered.
    let mut dest_def_idx: FxHashMap<u32, usize> = FxHashMap::default();
    for (root, &(bi, _ii, dest)) in &copies {
        for (di, inst) in func.blocks[bi].instructions.iter().enumerate() {
            if inst.dest().is_some_and(|d| d.0 == dest.0) {
                dest_def_idx.insert(*root, di);
                break;
            }
        }
    }
    // Candidates whose dest def is late in the block: try to hoist, else drop.
    {
        let mut to_fix: Vec<(u32, usize, Value)> = Vec::new();
        for (root, &(bi, copy_i, dest)) in &copies {
            if let Some(&di) = dest_def_idx.get(root) {
                // Any root use before the def in the same block is a problem.
                let mut earliest_use: Option<usize> = None;
                for (ii, inst) in func.blocks[bi].instructions.iter().enumerate() {
                    if ii >= copy_i { break; }
                    let mut used = Vec::new();
                    inst.for_each_used_value(|v| used.push(v));
                    if used.iter().any(|v| paths.get(v).is_some_and(|p| p.0 == *root)) {
                        earliest_use = Some(ii);
                        break;
                    }
                }
                if earliest_use.is_some_and(|u| u < di) {
                    to_fix.push((*root, bi, dest));
                }
            }
        }
        for (root, bi, dest) in to_fix {
            if try_hoist_def_chain(func, bi, dest) {
                // Recompute the copy's index after the hoist.
                if let Some((new_i, _)) = func.blocks[bi]
                    .instructions
                    .iter()
                    .enumerate()
                    .find(|(_, inst)| matches!(inst, Instruction::Memcpy { dest: d, .. } if d.0 == dest.0))
                {
                    if let Some(entry) = copies.get_mut(&root) {
                        *entry = (bi, new_i, dest);
                    }
                }
                dest_def_idx.insert(root, 0);
            } else {
                if std::env::var("CCC_DEBUG_AGG").is_ok() {
                    eprintln!("[AGG] func={} root={} dropped: dest def chain not hoistable", func.name, root);
                }
                copies.remove(&root);
            }
        }
    }

    let mut invalid = FxHashSet::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            let mut used = Vec::new(); inst.for_each_used_value(|v| used.push(v));
            for value in used {
                let Some((root, _)) = paths.get(&value) else { continue };
                let Some((copy_b, copy_i, _)) = copies.get(root) else { continue };
                let allowed = match inst {
                    Instruction::GetElementPtr { base, .. } => base.0 == value,
                    Instruction::Store { ptr, .. } => ptr.0 == value,
                    Instruction::Copy { src: Operand::Value(v), .. } => v.0 == value,
                    Instruction::Phi { incoming, .. } => incoming.iter().any(|(op, _)| matches!(op, Operand::Value(v) if v.0 == value)),
                    Instruction::Memcpy { dest, src, .. } =>
                        (src.0 == value && bi == *copy_b && ii == *copy_i)
                        || (dest.0 == value && bi == *copy_b && ii < *copy_i),
                    _ => false,
                };
                let path_merge = matches!(inst, Instruction::Copy { .. } | Instruction::Phi { .. });
                let ordered = path_merge || ii <= *copy_i;
                let located = path_merge || bi == *copy_b;
                // A use redirected to dest must come after dest's definition.
                let after_def = match dest_def_idx.get(root) {
                    Some(&di) if bi == *copy_b && ii < *copy_i => di < ii,
                    _ => true,
                };
                if !allowed || !located || !ordered || !after_def {
                    if std::env::var("CCC_DEBUG_AGG").is_ok() {
                        eprintln!(
                            "[AGG] invalidate root={} at block={} inst={} allowed={} located={} ordered={} after_def={} inst={:?}",
                            root, bi, ii, allowed, located, ordered, after_def, inst
                        );
                    }
                    invalid.insert(*root);
                }
            }
        }
    }
    for root in invalid { copies.remove(&root); }
    if copies.is_empty() { return 0; }

    let mut changes = 0;
    for (&root, &(bi, copy_i, dest)) in &copies {
        for (ii, inst) in func.blocks[bi].instructions.iter_mut().enumerate() {
            match inst {
                Instruction::GetElementPtr { base, .. } if ii < copy_i && base.0 == root => { *base = dest; changes += 1; }
                Instruction::Store { ptr, .. } if ii < copy_i && ptr.0 == root => { *ptr = dest; changes += 1; }
                Instruction::Copy { src: Operand::Value(v), .. } if v.0 == root => {
                    *v = dest; changes += 1;
                }
                Instruction::Phi { incoming, .. } => {
                    for (op, _) in incoming {
                        if matches!(op, Operand::Value(v) if v.0 == root) {
                            *op = Operand::Value(dest); changes += 1;
                        }
                    }
                }
                Instruction::Memcpy { dest: copy_dest, .. } if ii < copy_i && copy_dest.0 == root => {
                    *copy_dest = dest; changes += 1;
                }
                _ => {}
            }
        }
    }
    let mut removals: Vec<(usize, usize)> = copies.values().map(|(b, i, _)| (*b, *i)).collect();
    removals.sort_unstable_by(|a, b| b.cmp(a));
    for (bi, ii) in removals {
        func.blocks[bi].instructions.remove(ii);
        if !func.blocks[bi].source_spans.is_empty() { func.blocks[bi].source_spans.remove(ii); }
        changes += 1;
    }
    if std::env::var("CCC_DEBUG_AGG").is_ok() {
        eprintln!("[AGG] func={} forwarded_changes={} surviving_candidates={}", func.name, changes, copies.len());
    }
    changes
}

pub(crate) fn run(func: &mut IrFunction) -> usize {
    let reverse_changes = forward_store_only_temporaries(func);
    let cfg = crate::ir::analysis::CfgAnalysis::build(func);
    let dominates = |a: usize, mut b: usize| {
        if a == b { return true; }
        for _ in 0..cfg.idom.len() {
            let parent = cfg.idom[b];
            if parent == b { break; }
            if parent == a { return true; }
            b = parent;
        }
        false
    };
    let paths = pointer_paths(func);
    let alloca_roots: FxHashSet<u32> = paths.iter()
        .filter_map(|(&v, (root, path))| (v == *root && path.is_empty()).then_some(v))
        .collect();

    let mut candidates: FxHashMap<u32, CopyCandidate> = FxHashMap::default();
    let mut duplicate = FxHashSet::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Memcpy { dest, src, .. } = inst {
                // An arbitrary pointer (global, parameter, or loaded pointer) may be
                // overwritten between this copy and a later read.  Restrict forwarding
                // to compiler-known stack objects so those writes can be checked below.
                if alloca_roots.contains(&dest.0) {
                    let Some((source_root, _)) = paths.get(&src.0) else { continue };
                    if *source_root == dest.0 {
                        continue;
                    }
                    if candidates.insert(dest.0, CopyCandidate {
                        block: bi,
                        inst: ii,
                        source: *src,
                        source_root: *source_root,
                    }).is_some() {
                        duplicate.insert(dest.0);
                    }
                }
            }
        }
    }
    for root in duplicate { candidates.remove(&root); }
    if candidates.is_empty() { return reverse_changes + eliminate_dead_aggregate_field_stores(func); }

    // Reject escaping, written, cross-block, or pre-copy uses of each temporary.
    let mut invalid = FxHashSet::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            let mut used = Vec::new();
            inst.for_each_used_value(|v| used.push(v));
            for value in used {
                let Some((root, _)) = paths.get(&value) else { continue };
                let Some(candidate) = candidates.get(root) else { continue };
                let allowed_shape = match inst {
                    Instruction::GetElementPtr { base, .. } => base.0 == value,
                    Instruction::Load { ptr, .. } => ptr.0 == value,
                    Instruction::Memcpy { dest, src, .. } => dest.0 == *root || src.0 == value,
                    Instruction::Copy { src: Operand::Value(v), .. } => v.0 == value,
                    Instruction::Phi { incoming, .. } => incoming.iter().any(|(op, _)| matches!(op, Operand::Value(v) if v.0 == value)),
                    _ => false,
                };
                let is_defining_copy = matches!(inst, Instruction::Memcpy { dest, .. } if dest.0 == *root)
                    && bi == candidate.block && ii == candidate.inst;
                let is_path_definition = matches!(inst, Instruction::GetElementPtr { base, .. } if base.0 == value)
                    || matches!(inst, Instruction::Copy { src: Operand::Value(v), .. } if v.0 == value)
                    || matches!(inst, Instruction::Phi { incoming, .. } if incoming.iter().any(|(op, _)| matches!(op, Operand::Value(v) if v.0 == value)));
                let ordered_read = if bi == candidate.block {
                    ii > candidate.inst
                } else {
                    dominates(candidate.block, bi)
                };
                // Keep the snapshot lifetime local to one block.  This makes the
                // source-mutation proof below exact even when the block is in a loop:
                // a write in the next iteration cannot precede a read in this one.
                let cross_block_read = !is_defining_copy && !is_path_definition
                    && bi != candidate.block;
                if !allowed_shape || cross_block_read
                    || (!is_defining_copy && !is_path_definition && !ordered_read) {
                    invalid.insert(*root);
                }
            }
        }
    }
    for root in invalid {
        candidates.remove(&root);
    }
    if candidates.is_empty() { return reverse_changes + eliminate_dead_aggregate_field_stores(func); }

    // The source must also remain unchanged after the copy.  Otherwise replacing
    // a temporary read with a source read changes snapshot semantics (TinyCC's
    // `tmp = *vtop; *vtop = ...; use(tmp)` exposed this).
    let mut invalid = FxHashSet::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            let written_root = match inst {
                Instruction::Store { ptr, .. } => paths.get(&ptr.0).map(|p| p.0),
                Instruction::Memcpy { dest, .. } => paths.get(&dest.0).map(|p| p.0),
                _ => None,
            };
            let Some(written_root) = written_root else { continue };
            for (&dest_root, candidate) in &candidates {
                let after_copy = bi == candidate.block && ii > candidate.inst;
                if after_copy && written_root == candidate.source_root {
                    invalid.insert(dest_root);
                }
            }
        }
    }
    for root in invalid { candidates.remove(&root); }
    if candidates.is_empty() { return reverse_changes + eliminate_dead_aggregate_field_stores(func); }

    fn resolve_source(mut source: Value, candidates: &FxHashMap<u32, CopyCandidate>) -> Value {
        let mut seen = FxHashSet::default();
        while seen.insert(source.0) {
            if let Some(next) = candidates.get(&source.0) { source = next.source; } else { break; }
        }
        source
    }

    let mut changes = 0;
    for bi in 0..func.blocks.len() {
        let old = std::mem::take(&mut func.blocks[bi].instructions);
        let old_spans = std::mem::take(&mut func.blocks[bi].source_spans);
        let has_spans = !old_spans.is_empty();
        let mut out = Vec::with_capacity(old.len());
        let mut spans = Vec::new();
        for (ii, mut inst) in old.into_iter().enumerate() {
            if let Instruction::Memcpy { dest, .. } = &inst {
                if candidates.get(&dest.0).is_some_and(|c| c.block == bi && c.inst == ii) {
                    changes += 1;
                    continue;
                }
            }
            if let Instruction::Memcpy { src, .. } = &mut inst {
                if let Some(candidate) = candidates.get(&src.0) {
                    *src = resolve_source(candidate.source, &candidates);
                    changes += 1;
                }
            }
            if let Instruction::Load { ptr, .. } = &mut inst {
                if let Some((root, path)) = paths.get(&ptr.0) {
                    if let Some(candidate) = candidates.get(root) {
                        let mut base = resolve_source(candidate.source, &candidates);
                        for (offset, ty) in path {
                            let dest = Value(func.next_value_id);
                            func.next_value_id += 1;
                            out.push(Instruction::GetElementPtr {
                                dest, base, offset: offset.clone(), ty: *ty,
                            });
                            if has_spans { spans.push(old_spans[ii]); }
                            base = dest;
                        }
                        *ptr = base;
                        changes += 1;
                    }
                }
            }
            out.push(inst);
            if has_spans { spans.push(old_spans[ii]); }
        }
        func.blocks[bi].instructions = out;
        if has_spans { func.blocks[bi].source_spans = spans; }
    }
    changes + reverse_changes + eliminate_dead_aggregate_field_stores(func)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::{AddressSpace, IrType};
    use crate::ir::reexports::{BasicBlock, BlockId, IrConst, Terminator};

    #[test]
    fn preserves_snapshot_when_copy_source_is_overwritten() {
        let mut func = IrFunction::new("snapshot".into(), IrType::I32, vec![], false);
        func.next_value_id = 3;
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 4, volatile: false },
                Instruction::Alloca { dest: Value(1), ty: IrType::I32, size: 4, align: 4, volatile: false },
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(1)), ptr: Value(0), ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Memcpy { dest: Value(1), src: Value(0), size: 4 },
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(2)), ptr: Value(0), ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(2), ptr: Value(1), ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: vec![],
        });

        assert_eq!(run(&mut func), 0);
        assert!(func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Memcpy { dest: Value(1), src: Value(0), .. })));
    }
}
