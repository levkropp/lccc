//! Loop-aware alias analysis based on linear pointer forms.
//!
//! Pointers resolve to `root + Sum coeff*iv + konst + march*t` (SCEV-lite),
//! expanding GEP/copy/cast/add/mul chains, current-loop striding phis (the
//! march term), and outer-loop pointer phis via lockstep integer IVs.
//! Used by loop_memory_promote (invariant vs marching-store disjointness)
//! and GVN (selective load-CSE invalidation on provably-disjoint stores).

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::reexports::{Instruction, IrFunction, Operand, Value};
use super::loop_analysis;

/// Follow GEP/Copy/Add-const chains, accumulating a constant byte offset.
/// Returns the root value id and accumulated offset.
pub(crate) fn resolve_ptr_chain(func: &IrFunction, start: Value) -> Option<(u32, i64)> {
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

pub(crate) fn byte_size(ty: IrType) -> i64 {
    match ty {
        IrType::I8 | IrType::U8 => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::I64 | IrType::U64 | IrType::F64 | IrType::Ptr => 8,
        _ => 16,
    }
}

/// A linear pointer form within the loop being analyzed:
///   address = root_base + S coeff*iv + konst + march*t
/// where `t` counts iterations of the current loop, and `syms` are
/// loop-invariant terms keyed by (outer) phi value id, sorted by id.
/// All arithmetic on byte offsets is checked; we bail out on overflow.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct LinForm {
    pub(crate) root: u64,
    pub(crate) syms: Vec<(u32, i64)>,
    pub(crate) konst: i64,
    pub(crate) march: i64,
}

/// A stable identity for a pointer root: global by name, alloca/param by id.
pub(crate) fn root_id(func: &IrFunction, v: Value) -> u64 {
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
pub(crate) fn striding_phi(func: &IrFunction, phi_v: Value) -> Option<(Operand, i64)> {
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
pub(crate) fn resolve_lin_form(
    func: &IrFunction,
    lp_body: &FxHashSet<usize>,
    def_block: &FxHashMap<u32, usize>,
    cur_header: usize,
    v: Value,
    fuel: u8,
) -> Option<LinForm> {
    if fuel == 0 { return None; }
    let fuel = fuel - 1;
    if std::env::var("CCC_DEBUG_PROMOTE").is_ok() && find_def(func, v).is_none() {
        eprintln!("[RESOLVE] v{} bail: no def", v.0);
    }
    let inst = find_def(func, v)?;
    let def_bi = def_block.get(&v.0).copied().unwrap_or(usize::MAX);
    let debug = std::env::var("CCC_DEBUG_PROMOTE").is_ok();
    macro_rules! bail {
        ($why:expr) => {{
            if debug { eprintln!("[RESOLVE] v{} bail: {}", v.0, $why); }
            return None;
        }};
    }

    if def_bi == cur_header {
        // Current-loop phi: the marching variable. value = init + stride*t
        // for both pointer and integer phis.
        if matches!(inst, Instruction::Phi { .. }) {
            let Some((init_op, stride)) = striding_phi(func, v) else { bail!("cur-header phi not striding") };
            let mut f = match init_op {
                Operand::Value(init_v) => {
                    resolve_lin_form(func, lp_body, def_block, cur_header, init_v, fuel)?
                }
                Operand::Const(c) => LinForm { root: 0, syms: vec![], konst: c.to_i64()?, march: 0 },
            };
            f.march = f.march.checked_add(stride)?;
            return Some(f);
        }
        bail!("in cur header but not a phi");
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
pub(crate) fn merge_forms(mut a: LinForm, b: LinForm) -> Option<LinForm> {
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
pub(crate) fn affine_disjoint(
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
    if cf.root != sf.root {
        // Different roots can still be provably disjoint by storage kind:
        // an alloca never coincides with a global or with a parameter's
        // pointee (its frame slot did not exist when the arguments were
        // formed), and two distinct globals name distinct storage. A
        // param/global pair can genuinely alias (f(&g)); anything involving
        // an opaque or constant root stays conservative.
        if cf.root == 0 || sf.root == 0 { return false; }
        let kind = |r: u64| {
            if (0x1_0000_0000..0x2_0000_0000).contains(&r) { 1 }      // alloca
            else if (0x2_0000_0000..0x3_0000_0000).contains(&r) { 2 } // param
            else if (0x3_0000_0000..0x4_0000_0000).contains(&r) { 3 } // opaque
            else { 4 }                                               // global name hash
        };
        let (a, b) = (kind(cf.root), kind(sf.root));
        return matches!((a, b), (1, 4) | (4, 1) | (1, 2) | (2, 1)) || (a == 4 && b == 4);
    }

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

// ── Per-function loop frames for GVN-style point-to-point queries ───────────

/// Precomputed loop structure: each block's innermost natural loop.
/// GVN uses this to resolve load/store pointers under the frame of the block
/// they execute in, so marching-pointer forms compare coherently.
pub(crate) struct LoopFrames {
    pub def_block: FxHashMap<u32, usize>,
    /// (header, body) per loop, ordered so that inner loops come first
    /// (innermost containment wins when mapping blocks to frames).
    pub frames: Vec<(usize, FxHashSet<usize>)>,
    /// block index -> innermost frame index (u32::MAX when not in a loop).
    pub block_frame: Vec<u32>,
}

impl LoopFrames {
    pub(crate) fn build(func: &IrFunction) -> Self {
        let cfg = crate::ir::analysis::CfgAnalysis::build(func);
        Self::build_with_cfg(func, &cfg)
    }

    pub(crate) fn build_with_cfg(func: &IrFunction, cfg: &crate::ir::analysis::CfgAnalysis) -> Self {
        let mut def_block = FxHashMap::default();
        for (bi, block) in func.blocks.iter().enumerate() {
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() { def_block.insert(dest.0, bi); }
            }
        }
        let loops = loop_analysis::merge_loops_by_header(
            loop_analysis::find_natural_loops(cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom));
        let mut frames: Vec<(usize, FxHashSet<usize>)> =
            loops.iter().map(|lp| (lp.header, lp.body.clone())).collect();
        // Innermost first: sort by body size ascending.
        frames.sort_by_key(|(_, body)| body.len());
        let mut block_frame = vec![u32::MAX; func.blocks.len()];
        for (fi, (_, body)) in frames.iter().enumerate() {
            for &b in body.iter() {
                if block_frame[b] == u32::MAX {
                    block_frame[b] = fi as u32;
                }
            }
        }
        LoopFrames { def_block, frames, block_frame }
    }
}

/// Debug helper: dump loop frames for a function.
pub(crate) fn dump_frames(func: &IrFunction, lf: &LoopFrames) {
    for (fi, (h, body)) in lf.frames.iter().enumerate() {
        eprintln!("[FRAMES] fn={} frame {} header={} body={:?}", func.name, fi, h, {
            let mut v: Vec<_> = body.iter().copied().collect();
            v.sort();
            v
        });
    }
}

/// Resolve a pointer to its linear form under a frame (u32::MAX = no loop).
pub(crate) fn resolve_in_frame(
    func: &IrFunction,
    lf: &LoopFrames,
    frame: u32,
    v: Value,
) -> Option<LinForm> {
    static EMPTY: std::sync::OnceLock<FxHashSet<usize>> = std::sync::OnceLock::new();
    let (body_ref, header_idx) = if frame == u32::MAX {
        (EMPTY.get_or_init(FxHashSet::default), usize::MAX)
    } else {
        let (h, b) = &lf.frames[frame as usize];
        (b, *h)
    };
    resolve_lin_form(func, body_ref, &lf.def_block, header_idx, v, 32)
}

/// Same-iteration disjointness of two resolved forms.
/// `same_frame` must be true iff both were resolved under the same loop frame;
/// march terms from different frames are incomparable (returns false = may alias).
pub(crate) fn forms_disjoint(
    load: &LinForm,
    load_sz: i64,
    store: &LinForm,
    store_sz: i64,
    same_frame: bool,
) -> bool {
    if load.root == 0 || load.root != store.root { return false; }
    if load.syms != store.syms { return false; }
    if !same_frame && (load.march != 0 || store.march != 0) { return false; }
    let d = store.konst - load.konst;
    let dm = store.march - load.march;
    if dm == 0 {
        // Constant separation (also covers lockstep-equal marches).
        return store.konst >= load.konst + load_sz || load.konst >= store.konst + store_sz;
    }
    if dm > 0 {
        // Store - load grows with t: disjoint iff the store starts at or above
        // the load's top at t = 0.
        d >= load_sz
    } else {
        // Store - load shrinks: disjoint iff the store ends at or below the
        // load's bottom at t = 0.
        d + store_sz <= 0
    }
}

