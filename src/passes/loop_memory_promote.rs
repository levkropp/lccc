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

/// Follow GEP/Copy/Add-const chains, accumulating a constant byte offset.
/// Returns the root value id and accumulated offset.
fn resolve_ptr_chain(func: &IrFunction, start: Value) -> Option<(u32, i64)> {
    let mut cur = start;
    let mut off: i64 = 0;
    for _ in 0..64 {
        let mut next = None;
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.dest() != Some(cur) { continue; }
                next = match inst {
                    Instruction::GetElementPtr { base, offset: Operand::Const(c), .. } => {
                        c.to_i64().map(|k| (base, k))
                    }
                    Instruction::Copy { src: Operand::Value(src), .. } => Some((src, 0)),
                    Instruction::BinOp { op: crate::ir::reexports::IrBinOp::Add, lhs, rhs, .. } => {
                        match (lhs, rhs) {
                            (Operand::Value(v), Operand::Const(c))
                            | (Operand::Const(c), Operand::Value(v)) => {
                                c.to_i64().map(|k| (v, k))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                break;
            }
            if next.is_some() { break; }
        }
        match next {
            Some((base, k)) => {
                off = off.checked_add(k)?;
                cur = *base;
            }
            None => return Some((cur.0, off)),
        }
    }
    None
}

/// A linear pointer form within the loop being analyzed:
///   address = root_base + S coeff*iv + konst + march*t
/// where `t` counts iterations of the current loop, and `syms` are
/// loop-invariant terms keyed by (outer) phi value id, sorted by id.
/// All arithmetic on byte offsets is checked; we bail out on overflow.
#[derive(Clone, PartialEq, Eq)]
struct LinForm {
    root: u64,
    syms: Vec<(u32, i64)>,
    konst: i64,
    march: i64,
}

/// A stable identity for a pointer root: global by name, alloca/param by id.
fn root_id(func: &IrFunction, v: Value) -> u64 {
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.dest() == Some(v) {
                return match inst {
                    Instruction::GlobalAddr { name, .. } => {
                        let mut h = 0xcbf29ce484222325u64;
                        for b in name.as_bytes() {
                            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
                        }
                        h
                    }
                    Instruction::Alloca { .. } => 0x1_0000_0000 + v.0 as u64,
                    Instruction::ParamRef { param_idx, .. } => 0x2_0000_0000 + *param_idx as u64,
                    _ => 0x3_0000_0000 + v.0 as u64,
                };
            }
        }
    }
    0x3_0000_0000 + v.0 as u64
}

/// Find the single definition of a value.
fn find_def<'a>(func: &'a IrFunction, v: Value) -> Option<&'a Instruction> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.dest() == Some(v) {
                return Some(inst);
            }
        }
    }
    None
}

/// Identify a simple striding phi `phi [init, phi + const]` (step via
/// GEP/add/copy chains). Returns (init_operand, stride), identified by FORM.
fn striding_phi(func: &IrFunction, phi_v: Value) -> Option<(Operand, i64)> {
    let Instruction::Phi { incoming, .. } = find_def(func, phi_v)? else { return None };
    if incoming.len() != 2 { return None; }
    let mut init = None;
    let mut stride = 0i64;
    for (op, _) in incoming {
        if let Operand::Value(v) = op {
            if let Some((root, off)) = resolve_ptr_chain(func, *v) {
                if root == phi_v.0 && off != 0 {
                    stride = off;
                    continue;
                }
            }
        }
        if init.is_some() { return None; } // two non-step incomings: bail
        init = Some(*op);
    }
    if stride == 0 { return None; }
    Some((init?, stride))
}

/// Resolve a value to a linear form relative to the current loop. Phis in the
/// current loop header become the marching term t; outer-loop phis are either
/// opaque symbolic terms (integer IVs) or expanded through a lockstep IV
/// (marching pointers). Other in-body definitions reject the resolution.
fn resolve_lin_form(
    func: &IrFunction,
    lp_body: &FxHashSet<usize>,
    def_block: &FxHashMap<u32, usize>,
    cur_header: usize,
    v: Value,
    fuel: u8,
) -> Option<LinForm> {
    if fuel == 0 { return None; }
    let fuel = fuel - 1;
    let inst = find_def(func, v)?;
    let def_bi = def_block.get(&v.0).copied().unwrap_or(usize::MAX);
    let debug = std::env::var("CCC_DEBUG_PROMOTE").is_ok();

    if def_bi == cur_header {
        // Current-loop phi: the marching variable. value = init + stride*t
        // for both pointer and integer phis.
        if matches!(inst, Instruction::Phi { .. }) {
            let (init_op, stride) = striding_phi(func, v)?;
            let mut f = match init_op {
                Operand::Value(init_v) => {
                    resolve_lin_form(func, lp_body, def_block, cur_header, init_v, fuel)?
                }
                Operand::Const(c) => LinForm { root: 0, syms: vec![], konst: c.to_i64()?, march: 0 },
            };
            f.march = f.march.checked_add(stride)?;
            return Some(f);
        }
        return None;
    }

    match inst {
        // Pure structural chains resolve soundly wherever they sit in the
        // loop: their value is always the same function of phi/inv parts.
        Instruction::Copy { src: Operand::Value(src), .. } => {
            resolve_lin_form(func, lp_body, def_block, cur_header, *src, fuel)
        }
        Instruction::Cast { src: Operand::Value(src), from_ty, to_ty, .. }
            if from_ty.size() <= to_ty.size() =>
        {
            resolve_lin_form(func, lp_body, def_block, cur_header, *src, fuel)
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            let mut f = resolve_lin_form(func, lp_body, def_block, cur_header, *base, fuel)?;
            match offset {
                Operand::Const(c) => {
                    f.konst = f.konst.checked_add(c.to_i64()?)?;
                }
                Operand::Value(ov) => {
                    let g = resolve_lin_form(func, lp_body, def_block, cur_header, *ov, fuel)?;
                    f = merge_forms(f, g)?;
                }
            }
            Some(f)
        }
        Instruction::BinOp { op: crate::ir::reexports::IrBinOp::Add, lhs, rhs, .. } => {
            match (lhs, rhs) {
                (Operand::Value(a), Operand::Value(b)) => {
                    let fa = resolve_lin_form(func, lp_body, def_block, cur_header, *a, fuel)?;
                    let fb = resolve_lin_form(func, lp_body, def_block, cur_header, *b, fuel)?;
                    merge_forms(fa, fb)
                }
                (Operand::Value(a), Operand::Const(c)) | (Operand::Const(c), Operand::Value(a)) => {
                    let mut fa = resolve_lin_form(func, lp_body, def_block, cur_header, *a, fuel)?;
                    fa.konst = fa.konst.checked_add(c.to_i64()?)?;
                    Some(fa)
                }
                _ => None,
            }
        }
        Instruction::BinOp { op: crate::ir::reexports::IrBinOp::Mul, lhs, rhs, .. } => {
            let (val_op, c) = match (lhs, rhs) {
                (Operand::Value(a), Operand::Const(c)) | (Operand::Const(c), Operand::Value(a)) => {
                    (*a, c.to_i64()?)
                }
                _ => return None,
            };
            let mut f = resolve_lin_form(func, lp_body, def_block, cur_header, val_op, fuel)?;
            if f.root != 0 { return None; } // scaling a pointer: not an address component
            f.konst = f.konst.checked_mul(c)?;
            f.march = f.march.checked_mul(c)?;
            for s in f.syms.iter_mut() {
                s.1 = s.1.checked_mul(c)?;
            }
            Some(f)
        }
        Instruction::Phi { ty, .. } => {
            // A phi inside the analyzed loop but outside its header belongs to
            // a nested loop: its value varies within one iteration — bail.
            if lp_body.contains(&def_bi) { return None; }
            // Phi of an OUTER loop (loop-invariant here).
            if *ty != crate::common::types::IrType::Ptr {
                // Integer outer phi: usable as an opaque symbolic term if it
                // is a simple striding IV; otherwise an opaque root.
                if striding_phi(func, v).is_some() {
                    return Some(LinForm { root: 0, syms: vec![(v.0, 1)], konst: 0, march: 0 });
                }
                return Some(LinForm { root: root_id(func, v), syms: vec![], konst: 0, march: 0 });
            }
            // Pointer outer phi: expand via a lockstep integer IV in the same
            // (outer) header: P = init + (S/S1)*(IV - IV_init).
            let (init_op, stride) = striding_phi(func, v)?;
            let Operand::Value(init_v) = init_op else { return None };
            let outer_header = def_bi;
            let mut iv_sym = None;
            for binst in &func.blocks[outer_header].instructions {
                let Instruction::Phi { dest, ty: ity, .. } = binst else { continue };
                if *ity == crate::common::types::IrType::Ptr { continue; }
                let Some((iv_init_op, iv_stride)) = striding_phi(func, *dest) else { continue };
                let Operand::Const(ivc) = iv_init_op else { continue };
                let Some(iv_c0) = ivc.to_i64() else { continue };
                if iv_stride == 0 || stride % iv_stride != 0 { continue; }
                iv_sym = Some((dest.0, stride / iv_stride, iv_c0));
                break;
            }
            let (iv_id, ratio, iv_c0) = iv_sym?;
            let mut f = resolve_lin_form(func, lp_body, def_block, cur_header, init_v, fuel)?;
            if f.march != 0 { return None; }
            // += ratio * (IV - iv_c0)
            f.konst = f.konst.checked_sub(ratio.checked_mul(iv_c0)?)?;
            match f.syms.iter_mut().find(|s| s.0 == iv_id) {
                Some(s) => s.1 = s.1.checked_add(ratio)?,
                None => f.syms.push((iv_id, ratio)),
            }
            f.syms.sort_by_key(|s| s.0);
            Some(f)
        }
        _ => {
            // Opaque root: only loop-invariant defs qualify.
            if lp_body.contains(&def_bi) { return None; }
            Some(LinForm { root: root_id(func, v), syms: vec![], konst: 0, march: 0 })
        }
    }
}

/// Merge two linear forms (at most one may carry a root); used for GEP
/// base+offset and pointer arithmetic addition.
fn merge_forms(mut a: LinForm, b: LinForm) -> Option<LinForm> {
    if a.root != 0 && b.root != 0 { return None; }
    if b.root != 0 { a.root = b.root; }
    a.konst = a.konst.checked_add(b.konst)?;
    a.march = a.march.checked_add(b.march)?;
    for (id, c) in b.syms {
        match a.syms.iter_mut().find(|s| s.0 == id) {
            Some(s) => s.1 = s.1.checked_add(c)?,
            None => a.syms.push((id, c)),
        }
    }
    a.syms.sort_by_key(|s| s.0);
    a.syms.retain(|s| s.1 != 0);
    Some(a)
}

/// Prove that the loop-invariant address `cand` never aliases `store` in any
/// iteration of the loop, using linear forms: both addresses are expressed as
/// root + coeff*iv + konst + march*t; with identical roots and symbolic
/// parts, a store marching away from the invariant candidate never reaches it.
/// Example: `bodies[i].vx` vs stores to `bodies[j].vx`, j marching +56 from
/// i+1: forms are bodies+56*iv+24 and bodies+56*iv+80+56*t — disjoint.
fn affine_disjoint(
    func: &IrFunction,
    lp_body: &FxHashSet<usize>,
    def_block: &FxHashMap<u32, usize>,
    header_idx: usize,
    cand: Value,
    cand_ty: IrType,
    store: Value,
    store_ty: IrType,
) -> bool {
    // Candidate must be loop-invariant.
    if def_block.get(&cand.0).is_some_and(|b| lp_body.contains(b)) { return false; }
    let debug = std::env::var("CCC_DEBUG_PROMOTE").is_ok();
    let cf = resolve_lin_form(func, lp_body, def_block, header_idx, cand, 32);
    let sf = resolve_lin_form(func, lp_body, def_block, header_idx, store, 32);
    if debug {
        eprintln!("[AFFINE] cand={} -> {:?}; store={} -> {:?}",
            cand.0, cf.as_ref().map(|f| (f.root, &f.syms, f.konst, f.march)),
            store.0, sf.as_ref().map(|f| (f.root, &f.syms, f.konst, f.march)));
    }
    let (Some(cf), Some(sf)) = (cf, sf) else { return false };
    if cf.root == 0 || cf.root != sf.root { return false; }

    let (cand_sz, store_sz) = (byte_size(cand_ty), byte_size(store_ty));
    if sf.march == 0 && cf.march == 0 {
        // Both invariant: plain constant range separation.
        if cf.syms != sf.syms { return false; }
        return cf.konst + cand_sz <= sf.konst || sf.konst + store_sz <= cf.konst;
    }
    if cf.march != 0 { return false; } // candidate itself marches
    if cf.syms != sf.syms { return false; }
    if sf.march > 0 {
        // Store range starts at or above the candidate's top and marches up.
        sf.konst >= cf.konst + cand_sz
    } else {
        // Store range ends at or below the candidate's bottom and marches down.
        sf.konst + store_sz <= cf.konst
    }
}

pub(crate) fn run(func: &mut IrFunction) -> usize {
    // Promote one pointer at a time, rescanning after each success: index
    // bookkeeping stays simple and already-promoted loops no longer match.
    // Each promotion strictly reduces the loop's memory-op count, so this
    // terminates quickly.
    let mut total = 0;
    for _ in 0..64 {
        let n = run_once(func);
        if n == 0 { break; }
        total += n;
    }
    total
}

fn run_once(func: &mut IrFunction) -> usize {
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
                    let other = Value(other_ptr);
                    if !disjoint(&paths, ptr, load_ty, other, *other_ty)
                        && !affine_disjoint(func, &lp.body, &def_block, lp.header,
                                            ptr, load_ty, other, *other_ty)
                    {
                        alias = true;
                    }
                }
            }
            if alias { continue; }
            if std::env::var("CCC_DEBUG_PROMOTE").is_ok() {
                eprintln!("[PROMOTE] func={} loop header={} ptr=Value({}) ty={:?}",
                    func.name, lp.header, ptr_id, load_ty);
            }

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
