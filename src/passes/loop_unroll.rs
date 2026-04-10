//! Loop unrolling pass.
//!
//! Unrolls small inner loops using "unroll with intermediate IV steps and
//! early exits". Replicates the loop body K times per unrolled cycle, with
//! an exit-condition check inserted between each copy. This handles
//! non-multiple-K trip counts without a separate cleanup loop: whichever
//! intermediate check fires first terminates the partial cycle.
//!
//! Example — 4× unrolled loop:
//!
//! ```text
//! header:   %iv = Phi [init, %iv_next]
//!           %cond = Cmp %iv, limit
//!           CondBranch %cond, exit, body_entry
//!
//! [original body blocks]  →  exit_check_1
//!
//! exit_check_1:
//!   %iv_1  = Add %iv, step
//!   %cond_1 = Cmp %iv_1, limit
//!   CondBranch %cond_1, exit, body_copy_2_entry
//!
//! [body_copy_2]  →  exit_check_2
//!   ...
//! exit_check_3  →  [body_copy_4]  →  latch
//!
//! latch:  %iv_next = Add %iv_3, step   ← was Add %iv, step
//!         Branch header
//! ```

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::analysis::CfgAnalysis;
use crate::ir::reexports::{
    BasicBlock, BlockId, Instruction, IrBinOp, IrCmpOp, IrConst, IrFunction, Operand, Terminator,
    Value,
};
use super::loop_analysis;

/// Maximum number of body-work blocks (body excluding header and latch) for
/// a loop to be eligible. Prevents excessive code size growth.
const MAX_UNROLL_BODY_BLOCKS: usize = 8;

/// Choose the unroll factor based on total instruction count in body-work blocks.
fn choose_unroll_factor(body_inst_count: usize) -> u32 {
    match body_inst_count {
        0..=8   => 8,
        9..=20  => 4,
        21..=60 => 2,
        _       => 1, // too large — skip
    }
}

/// All information needed to perform the unrolling transformation.
struct UnrollCandidate {
    /// Block index of the loop header (has the phi + condition check).
    header: usize,
    /// Block index of the single latch (has the IV increment + back-branch).
    latch: usize,
    /// Body blocks, excluding header and latch.
    body_work: Vec<usize>,
    /// Index into `body_work` whose label equals `body_entry`.
    body_entry_work_idx: usize,
    /// Index into `body_work` of the block that branches to the latch.
    pre_latch_work_idx: usize,
    /// Exit block label (outside the loop, target of the header's exit branch).
    exit_target: BlockId,
    /// First in-loop block label (target of the header's continue branch).
    body_entry: BlockId,
    /// The IV phi value defined in the header.
    iv_phi: Value,
    /// Type of the IV.
    iv_ty: IrType,
    /// Constant step added to IV per iteration.
    iv_step: i64,
    /// Comparison operator used in the exit condition.
    exit_cmp_op: IrCmpOp,
    /// Type of the exit comparison instruction.
    exit_cmp_ty: IrType,
    /// The loop-invariant operand of the exit comparison (the "limit").
    exit_limit: Operand,
    /// `true` if the IV is the left-hand operand of the exit Cmp.
    iv_is_lhs: bool,
    /// `true` if cond==true means exit (false means continue).
    exit_cond_positive: bool,
    /// Index of the `Add %iv, step` instruction inside the latch block.
    latch_iv_incr_idx: usize,
    /// Number of times to replicate the loop body (K). Always ≥ 2.
    unroll_factor: u32,
}

/// Run the loop-unrolling pass on one function. Returns the number of loops
/// that were successfully unrolled.
pub(crate) fn unroll_loops(func: &mut IrFunction) -> usize {
    if func.blocks.len() < 2 {
        return 0;
    }
    let cfg = CfgAnalysis::build(func);
    let raw = loop_analysis::find_natural_loops(
        cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom,
    );
    if raw.is_empty() {
        return 0;
    }
    let loops = loop_analysis::merge_loops_by_header(raw);

    // Set of all loop-header block indices (used for nested-loop detection).
    let all_headers: FxHashSet<usize> = loops.iter().map(|l| l.header).collect();

    // Collect and sort candidates by body size (smallest first = innermost first).
    let mut candidates: Vec<UnrollCandidate> = loops
        .iter()
        .filter_map(|lp| analyze_loop(func, lp, &cfg, &all_headers))
        .collect();
    candidates.sort_by_key(|c| c.body_work.len());

    let mut count = 0;
    for c in candidates {
        if do_unroll(func, c) {
            count += 1;
        }
    }
    count
}

// ── Eligibility analysis ──────────────────────────────────────────────────────

fn analyze_loop(
    func: &IrFunction,
    lp: &loop_analysis::NaturalLoop,
    cfg: &CfgAnalysis,
    all_headers: &FxHashSet<usize>,
) -> Option<UnrollCandidate> {
    let header = lp.header;

    // 1. Size check: body (header + latch + work blocks) must be small.
    if lp.body.len() > MAX_UNROLL_BODY_BLOCKS + 2 {
        return None;
    }

    // 2. Single latch: exactly one block in body has a back-edge to header.
    let back_preds: Vec<usize> = cfg
        .preds
        .row(header)
        .iter()
        .map(|&p| p as usize)
        .filter(|p| lp.body.contains(p))
        .collect();
    if back_preds.len() != 1 {
        return None;
    }
    let latch = back_preds[0];

    // Latch must terminate with an unconditional Branch back to the header.
    let header_label = func.blocks[header].label;
    match &func.blocks[latch].terminator {
        Terminator::Branch(lbl) if *lbl == header_label => {}
        _ => return None,
    }

    // 3. A unique preheader must exist.
    loop_analysis::find_preheader(header, &lp.body, &cfg.preds)?;

    // 4. body_work = body \ {header, latch}; must be non-empty.
    let body_work: Vec<usize> = lp
        .body
        .iter()
        .copied()
        .filter(|&b| b != header && b != latch)
        .collect();
    if body_work.is_empty() {
        return None;
    }

    // 5. No nested loops: body_work blocks must not be headers of other loops.
    for &b in &body_work {
        if all_headers.contains(&b) {
            return None;
        }
    }

    // 6. No disqualifying instructions in body_work.
    for &bi in &body_work {
        for inst in &func.blocks[bi].instructions {
            match inst {
                Instruction::Call { .. }
                | Instruction::CallIndirect { .. }
                | Instruction::InlineAsm { .. }
                | Instruction::AtomicRmw { .. }
                | Instruction::AtomicCmpxchg { .. }
                | Instruction::AtomicLoad { .. }
                | Instruction::AtomicStore { .. }
                | Instruction::DynAlloca { .. } => return None,
                _ => {}
            }
        }
    }

    // 7. Find basic IV: a phi in the header whose back-edge value is
    //    Add(%iv, const_step) in the latch.
    let latch_label = func.blocks[latch].label;
    let (iv_phi, iv_ty, iv_step, latch_iv_incr_idx) =
        find_iv_in_loop(func, header, latch, latch_label)?;

    // 8. Detect the exit condition from the header's CondBranch.
    let (exit_target, body_entry, exit_cmp_op, exit_cmp_ty, exit_limit, iv_is_lhs, exit_cond_positive) =
        find_exit_condition(func, header, &lp.body, iv_phi)?;

    // 9. Count body instructions and select the unroll factor.
    let body_inst_count: usize = body_work
        .iter()
        .map(|&bi| func.blocks[bi].instructions.len())
        .sum();
    let unroll_factor = choose_unroll_factor(body_inst_count);
    if unroll_factor <= 1 {
        return None;
    }

    // 10. Find body_entry_work_idx and ensure a unique pre-latch block.
    let body_entry_work_idx = body_work
        .iter()
        .position(|&bi| func.blocks[bi].label == body_entry)?;

    let mut pre_latch_work_idx: Option<usize> = None;
    for (j, &bi) in body_work.iter().enumerate() {
        if block_has_succ(&func.blocks[bi].terminator, latch_label) {
            if pre_latch_work_idx.is_some() {
                return None; // multiple blocks branch to latch — too complex
            }
            pre_latch_work_idx = Some(j);
        }
    }
    let pre_latch_work_idx = pre_latch_work_idx?;

    // 11. Exit-block phi eligibility: all incoming-from-header values must be
    //     loop-invariant (not defined in body_work), so each new exit edge can
    //     carry the same value without creating new definitions.
    if let Some(exit_bi) = func.blocks.iter().position(|b| b.label == exit_target) {
        for inst in &func.blocks[exit_bi].instructions {
            if let Instruction::Phi { incoming, .. } = inst {
                for (op, src_label) in incoming {
                    if *src_label == header_label {
                        if let Operand::Value(v) = op {
                            if is_defined_in_body(v.0, &lp.body, func) {
                                return None;
                            }
                        }
                    }
                }
            }
        }
    }

    // Skip unrolling for I32/U32 IV types on 64-bit targets when the loop body
    // contains Cast(I32→I64) or GEP instructions that widen the IV. The unroller
    // creates intermediate IV values at the narrow type, and in complex functions
    // (like SQLite's 255K-line amalgamation) the widened values can interact
    // incorrectly with subsequent optimization passes.
    // Simple loops without IV widening (pure I32 arithmetic) are safe to unroll.
    if !crate::common::types::target_is_32bit() && iv_ty.size() < 8 && iv_ty.is_integer() {
        let has_iv_widening = body_work.iter().any(|&bi| {
            func.blocks[bi].instructions.iter().any(|inst| {
                match inst {
                    Instruction::Cast { src: Operand::Value(v), from_ty, to_ty, .. } => {
                        v.0 == iv_phi.0
                            && matches!(from_ty, IrType::I32 | IrType::U32)
                            && matches!(to_ty, IrType::I64 | IrType::U64 | IrType::Ptr)
                    }
                    Instruction::GetElementPtr { offset: Operand::Value(v), .. } => {
                        v.0 == iv_phi.0
                    }
                    _ => false,
                }
            })
        });
        if has_iv_widening {
            return None;
        }
    }

    Some(UnrollCandidate {
        header,
        latch,
        body_work,
        body_entry_work_idx,
        pre_latch_work_idx,
        exit_target,
        body_entry,
        iv_phi,
        iv_ty,
        iv_step,
        exit_cmp_op,
        exit_cmp_ty,
        exit_limit,
        iv_is_lhs,
        exit_cond_positive,
        latch_iv_incr_idx,
        unroll_factor,
    })
}

/// Find a basic induction variable in the loop header and its increment in
/// the latch. Returns `(phi_dest, ty, step, latch_incr_idx)`.
fn find_iv_in_loop(
    func: &IrFunction,
    header: usize,
    latch: usize,
    latch_label: BlockId,
) -> Option<(Value, IrType, i64, usize)> {
    for inst in &func.blocks[header].instructions {
        let (phi_dest, ty, incoming) = match inst {
            Instruction::Phi { dest, ty, incoming } if ty.is_integer() => (dest, ty, incoming),
            _ => continue,
        };

        // Value flowing into the header from the latch (the back-edge value).
        let back_val = incoming
            .iter()
            .find(|(_, lbl)| *lbl == latch_label)
            .and_then(|(op, _)| {
                if let Operand::Value(v) = op { Some(*v) } else { None }
            });
        let back_val = back_val?;

        // Look for `Add(phi_dest, const_step)` or `Add(const_step, phi_dest)`
        // in the latch that produces `back_val`.
        let phi_id = phi_dest.0;
        for (idx, latch_inst) in func.blocks[latch].instructions.iter().enumerate() {
            if let Instruction::BinOp { dest, op: IrBinOp::Add, lhs, rhs, .. } = latch_inst {
                if *dest != back_val {
                    continue;
                }
                let step = match (lhs, rhs) {
                    (Operand::Value(v), Operand::Const(c)) if v.0 == phi_id => c.to_i64(),
                    (Operand::Const(c), Operand::Value(v)) if v.0 == phi_id => c.to_i64(),
                    _ => None,
                };
                if let Some(step) = step {
                    return Some((*phi_dest, *ty, step, idx));
                }
            }
        }
    }
    None
}

/// Detect the exit condition from the header's CondBranch terminator.
///
/// Returns `(exit_target, body_entry, cmp_op, cmp_ty, limit, iv_is_lhs, exit_cond_positive)`.
/// `exit_cond_positive` is `true` when the condition evaluating to `true` means "exit".
fn find_exit_condition(
    func: &IrFunction,
    header: usize,
    loop_body: &FxHashSet<usize>,
    iv_phi: Value,
) -> Option<(BlockId, BlockId, IrCmpOp, IrType, Operand, bool, bool)> {
    let header_block = &func.blocks[header];

    let (cond_op, true_label, false_label) = match &header_block.terminator {
        Terminator::CondBranch { cond, true_label, false_label } => {
            (*cond, *true_label, *false_label)
        }
        _ => return None,
    };

    // Map labels to block indices for in-loop membership check.
    let label_to_idx: FxHashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

    let true_in_loop = label_to_idx
        .get(&true_label)
        .map(|&bi| loop_body.contains(&bi))
        .unwrap_or(false);
    let false_in_loop = label_to_idx
        .get(&false_label)
        .map(|&bi| loop_body.contains(&bi))
        .unwrap_or(false);

    // Exactly one branch must be in-loop, the other is the exit.
    if true_in_loop == false_in_loop {
        return None;
    }

    let (exit_target, body_entry, exit_cond_positive) = if !true_in_loop {
        (true_label, false_label, true)
    } else {
        (false_label, true_label, false)
    };

    // Trace the condition value to a Cmp instruction (through at most one Cast).
    let cond_id = match cond_op {
        Operand::Value(v) => v.0,
        _ => return None,
    };

    // Build a map of value-id → instruction for the header.
    let mut hdr_defs: FxHashMap<u32, &Instruction> = FxHashMap::default();
    for inst in &header_block.instructions {
        if let Some(dest) = inst.dest() {
            hdr_defs.insert(dest.0, inst);
        }
    }

    // Look through one Cast.
    let cmp_id = match hdr_defs.get(&cond_id) {
        Some(Instruction::Cast { src: Operand::Value(v), .. }) => v.0,
        _ => cond_id,
    };

    let (cmp_op, cmp_lhs, cmp_rhs, cmp_ty) = match hdr_defs.get(&cmp_id) {
        Some(Instruction::Cmp { op, lhs, rhs, ty, .. }) => (*op, *lhs, *rhs, *ty),
        _ => return None,
    };

    let iv_id = iv_phi.0;

    // One Cmp operand must be exactly the IV phi; the other must be loop-invariant.
    let (iv_is_lhs, limit_op) =
        if matches!(cmp_lhs, Operand::Value(v) if v.0 == iv_id)
            && is_loop_invariant_op(cmp_rhs, loop_body, func)
        {
            (true, cmp_rhs)
        } else if matches!(cmp_rhs, Operand::Value(v) if v.0 == iv_id)
            && is_loop_invariant_op(cmp_lhs, loop_body, func)
        {
            (false, cmp_lhs)
        } else {
            return None;
        };

    Some((exit_target, body_entry, cmp_op, cmp_ty, limit_op, iv_is_lhs, exit_cond_positive))
}

// ── CFG helpers ───────────────────────────────────────────────────────────────

fn is_loop_invariant_op(op: Operand, loop_body: &FxHashSet<usize>, func: &IrFunction) -> bool {
    match op {
        Operand::Const(_) => true,
        Operand::Value(v) => !is_defined_in_body(v.0, loop_body, func),
    }
}

fn is_defined_in_body(val_id: u32, loop_body: &FxHashSet<usize>, func: &IrFunction) -> bool {
    for &bi in loop_body {
        if bi < func.blocks.len() {
            for inst in &func.blocks[bi].instructions {
                if let Some(dest) = inst.dest() {
                    if dest.0 == val_id {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn block_has_succ(term: &Terminator, target: BlockId) -> bool {
    match term {
        Terminator::Branch(lbl) => *lbl == target,
        Terminator::CondBranch { true_label, false_label, .. } => {
            *true_label == target || *false_label == target
        }
        _ => false,
    }
}

/// Replace `old` with `new` in one specific block-label slot of a terminator.
fn redirect_label(term: &mut Terminator, old: BlockId, new: BlockId) {
    match term {
        Terminator::Branch(lbl) if *lbl == old => *lbl = new,
        Terminator::CondBranch { true_label, false_label, .. } => {
            if *true_label == old {
                *true_label = new;
            }
            if *false_label == old {
                *false_label = new;
            }
        }
        _ => {}
    }
}

/// Apply a block-label rename map to all branch targets in a terminator.
fn replace_block_ids(term: &mut Terminator, map: &FxHashMap<BlockId, BlockId>) {
    match term {
        Terminator::Branch(lbl) => {
            if let Some(&new) = map.get(lbl) {
                *lbl = new;
            }
        }
        Terminator::CondBranch { true_label, false_label, .. } => {
            if let Some(&new) = map.get(true_label) {
                *true_label = new;
            }
            if let Some(&new) = map.get(false_label) {
                *false_label = new;
            }
        }
        Terminator::Switch { cases, default, .. } => {
            if let Some(&new) = map.get(default) {
                *default = new;
            }
            for (_, lbl) in cases {
                if let Some(&new) = map.get(lbl) {
                    *lbl = new;
                }
            }
        }
        _ => {}
    }
}

// ── Transformation ────────────────────────────────────────────────────────────

fn do_unroll(func: &mut IrFunction, c: UnrollCandidate) -> bool {
    let k = c.unroll_factor as usize; // total copies (1 original + k-1 clones)
    let num_new = k - 1; // number of clones = number of exit-check blocks
    if num_new == 0 {
        return false;
    }

    let header_label = func.blocks[c.header].label;
    let latch_label = func.blocks[c.latch].label;

    // ── Pre-allocate all new BlockIds and Values ──────────────────────────────
    let max_label = func.blocks.iter().map(|b| b.label.0).max().unwrap_or(0);
    let mut next_label = max_label + 1;
    let mut next_val = func.next_value_id;

    // iv_vals[j]    = %iv_{j+1}    (used in exit_check_{j+1} and clone[j])
    // cond_vals[j]  = %cond_{j+1}  (used in exit_check_{j+1})
    // ec_labels[j]  = label of exit_check_{j+1}
    // cl_labels[j]  = labels of clone[j]'s body_work blocks (parallel to body_work)
    let iv_vals: Vec<Value> = (0..num_new)
        .map(|_| { let v = Value(next_val); next_val += 1; v })
        .collect();
    let cond_vals: Vec<Value> = (0..num_new)
        .map(|_| { let v = Value(next_val); next_val += 1; v })
        .collect();
    let ec_labels: Vec<BlockId> = (0..num_new)
        .map(|_| { let l = BlockId(next_label); next_label += 1; l })
        .collect();
    let cl_labels: Vec<Vec<BlockId>> = (0..num_new)
        .map(|_| {
            (0..c.body_work.len())
                .map(|_| { let l = BlockId(next_label); next_label += 1; l })
                .collect()
        })
        .collect();

    // Build value-rename maps for each clone.
    // clone_vmaps[j]: old_value_id → fresh_value_id, seeded with iv_phi → iv_vals[j].
    let mut clone_vmaps: Vec<FxHashMap<u32, u32>> = Vec::with_capacity(num_new);
    for j in 0..num_new {
        let mut vmap: FxHashMap<u32, u32> = FxHashMap::default();
        vmap.insert(c.iv_phi.0, iv_vals[j].0);
        for &bi in &c.body_work {
            for inst in &func.blocks[bi].instructions {
                if let Some(dest) = inst.dest() {
                    vmap.entry(dest.0).or_insert_with(|| {
                        let v = next_val;
                        next_val += 1;
                        v
                    });
                }
            }
        }
        clone_vmaps.push(vmap);
    }
    func.next_value_id = next_val;

    // ── Build new blocks (read-only access to func.blocks) ───────────────────
    let mut new_blocks: Vec<BasicBlock> = Vec::new();

    for j in 0..num_new {
        // The IV value feeding into this exit check:
        //   j=0: prev_iv = %iv_phi (the header phi)
        //   j>0: prev_iv = iv_vals[j-1]
        let prev_iv: Operand = if j == 0 {
            Operand::Value(c.iv_phi)
        } else {
            Operand::Value(iv_vals[j - 1])
        };

        let iv_j = iv_vals[j];
        let cond_j = cond_vals[j];

        // Entry of clone[j] (the block exit_check_{j+1} jumps into on "continue").
        let clone_entry = cl_labels[j][c.body_entry_work_idx];

        // ── Build exit_check_{j+1} ────────────────────────────────────────
        let cmp_lhs = if c.iv_is_lhs { Operand::Value(iv_j) } else { c.exit_limit };
        let cmp_rhs = if c.iv_is_lhs { c.exit_limit } else { Operand::Value(iv_j) };
        let (ec_true, ec_false) = if c.exit_cond_positive {
            (c.exit_target, clone_entry)
        } else {
            (clone_entry, c.exit_target)
        };

        new_blocks.push(BasicBlock {
            label: ec_labels[j],
            instructions: vec![
                Instruction::BinOp {
                    dest: iv_j,
                    op: IrBinOp::Add,
                    lhs: prev_iv,
                    rhs: Operand::Const(IrConst::from_i64(c.iv_step, c.iv_ty)),
                    ty: c.iv_ty,
                },
                Instruction::Cmp {
                    dest: cond_j,
                    op: c.exit_cmp_op,
                    lhs: cmp_lhs,
                    rhs: cmp_rhs,
                    ty: c.exit_cmp_ty,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(cond_j),
                true_label: ec_true,
                false_label: ec_false,
            },
            source_spans: Vec::new(),
        });

        // ── Build clone[j] (cloned body_work blocks) ──────────────────────
        // Block-label rename map for internal branches within this clone.
        let mut blk_map: FxHashMap<BlockId, BlockId> = FxHashMap::default();
        for (i, &bi) in c.body_work.iter().enumerate() {
            blk_map.insert(func.blocks[bi].label, cl_labels[j][i]);
        }

        // Where does clone[j]'s pre-latch block redirect after "latch"?
        //   j < num_new-1: → exit_check_{j+2}  (= ec_labels[j+1])
        //   j = num_new-1: → original latch     (no redirect)
        let post_latch_redirect: Option<BlockId> = if j + 1 < num_new {
            Some(ec_labels[j + 1])
        } else {
            None // last clone keeps going to original latch
        };

        let vmap = &clone_vmaps[j];
        for (i, &bi) in c.body_work.iter().enumerate() {
            let orig = &func.blocks[bi];

            let new_insts: Vec<Instruction> = orig
                .instructions
                .iter()
                .map(|inst| {
                    let mut cloned = inst.clone();
                    replace_values_in_inst(&mut cloned, vmap);
                    rename_inst_dest(&mut cloned, vmap);
                    cloned
                })
                .collect();

            let mut new_term = orig.terminator.clone();
            replace_values_in_terminator(&mut new_term, vmap);
            replace_block_ids(&mut new_term, &blk_map);

            // Redirect latch edge from pre-latch block.
            if i == c.pre_latch_work_idx {
                if let Some(redirect_to) = post_latch_redirect {
                    redirect_label(&mut new_term, latch_label, redirect_to);
                }
                // else: last clone's pre-latch block stays pointing at original latch.
            }

            new_blocks.push(BasicBlock {
                label: cl_labels[j][i],
                instructions: new_insts,
                terminator: new_term,
                source_spans: Vec::new(),
            });
        }
    }

    // ── Mutate existing blocks ────────────────────────────────────────────────

    // Step 3: Redirect original body's pre-latch block from latch → exit_check_1.
    redirect_label(
        &mut func.blocks[c.body_work[c.pre_latch_work_idx]].terminator,
        latch_label,
        ec_labels[0],
    );

    // Step 4: Update latch's IV increment: swap iv_phi → iv_{K-1} (= iv_vals[num_new-1]).
    let last_iv = iv_vals[num_new - 1];
    if let Instruction::BinOp { op: IrBinOp::Add, lhs, rhs, .. } =
        &mut func.blocks[c.latch].instructions[c.latch_iv_incr_idx]
    {
        if matches!(lhs, Operand::Value(v) if v.0 == c.iv_phi.0) {
            *lhs = Operand::Value(last_iv);
        } else if matches!(rhs, Operand::Value(v) if v.0 == c.iv_phi.0) {
            *rhs = Operand::Value(last_iv);
        }
    }

    // Step 5: For any phi in the exit block that has an incoming from header,
    // add the same value as incoming from each new exit-check block.
    if let Some(exit_bi) = func.blocks.iter().position(|b| b.label == c.exit_target) {
        // Collect (phi_index, value) pairs where value came from the header.
        let phi_header_vals: Vec<(usize, Operand)> = func.blocks[exit_bi]
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(phi_idx, inst)| {
                if let Instruction::Phi { incoming, .. } = inst {
                    incoming
                        .iter()
                        .find(|(_, lbl)| *lbl == header_label)
                        .map(|(op, _)| (phi_idx, *op))
                } else {
                    None
                }
            })
            .collect();

        for (phi_idx, op) in phi_header_vals {
            for j in 0..num_new {
                if let Instruction::Phi { incoming, .. } =
                    &mut func.blocks[exit_bi].instructions[phi_idx]
                {
                    incoming.push((op, ec_labels[j]));
                }
            }
        }
    }

    // Step 6: Append all new blocks.
    func.blocks.extend(new_blocks);

    true
}

// ── Value-replacement helpers (adapted from tail_call_elim.rs) ────────────────

/// Rename the SSA *definition* site (dest) of an instruction using `map`.
/// Only variants that produce an SSA value are affected; others are a no-op.
fn rename_inst_dest(inst: &mut Instruction, map: &FxHashMap<u32, u32>) {
    match inst {
        Instruction::Alloca { dest, .. }
        | Instruction::DynAlloca { dest, .. }
        | Instruction::Load { dest, .. }
        | Instruction::BinOp { dest, .. }
        | Instruction::UnaryOp { dest, .. }
        | Instruction::Cmp { dest, .. }
        | Instruction::GetElementPtr { dest, .. }
        | Instruction::Cast { dest, .. }
        | Instruction::Copy { dest, .. }
        | Instruction::GlobalAddr { dest, .. }
        | Instruction::VaArg { dest, .. }
        | Instruction::AtomicRmw { dest, .. }
        | Instruction::AtomicCmpxchg { dest, .. }
        | Instruction::AtomicLoad { dest, .. }
        | Instruction::Phi { dest, .. }
        | Instruction::LabelAddr { dest, .. }
        | Instruction::GetReturnF64Second { dest }
        | Instruction::GetReturnF32Second { dest }
        | Instruction::GetReturnF128Second { dest }
        | Instruction::Select { dest, .. }
        | Instruction::StackSave { dest }
        | Instruction::ParamRef { dest, .. } => replace_val(dest, map),

        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            if let Some(dest) = &mut info.dest {
                replace_val(dest, map);
            }
        }

        Instruction::Intrinsic { dest, .. } => {
            if let Some(dest) = dest {
                replace_val(dest, map);
            }
        }

        // No SSA destination.
        Instruction::Store { .. }
        | Instruction::Memcpy { .. }
        | Instruction::VaArgStruct { .. }
        | Instruction::VaStart { .. }
        | Instruction::VaEnd { .. }
        | Instruction::VaCopy { .. }
        | Instruction::AtomicStore { .. }
        | Instruction::Fence { .. }
        | Instruction::SetReturnF64Second { .. }
        | Instruction::SetReturnF32Second { .. }
        | Instruction::SetReturnF128Second { .. }
        | Instruction::InlineAsm { .. }
        | Instruction::StackRestore { .. } => {}
    }
}

#[inline]
fn replace_val(v: &mut Value, map: &FxHashMap<u32, u32>) {
    if let Some(&new_id) = map.get(&v.0) {
        *v = Value(new_id);
    }
}

#[inline]
fn replace_op(op: &mut Operand, map: &FxHashMap<u32, u32>) {
    if let Operand::Value(v) = op {
        replace_val(v, map);
    }
}

fn replace_values_in_inst(inst: &mut Instruction, map: &FxHashMap<u32, u32>) {
    match inst {
        // Definitions with no operands to replace.
        Instruction::ParamRef { .. }
        | Instruction::Alloca { .. }
        | Instruction::GlobalAddr { .. }
        | Instruction::LabelAddr { .. }
        | Instruction::Fence { .. }
        | Instruction::StackSave { .. }
        | Instruction::GetReturnF64Second { .. }
        | Instruction::GetReturnF32Second { .. }
        | Instruction::GetReturnF128Second { .. } => {}

        // Memory.
        Instruction::Store { val, ptr, .. } => {
            replace_op(val, map);
            replace_val(ptr, map);
        }
        Instruction::Load { ptr, .. } => replace_val(ptr, map),
        Instruction::Memcpy { dest, src, .. } => {
            replace_val(dest, map);
            replace_val(src, map);
        }

        // Arithmetic / logic.
        Instruction::BinOp { lhs, rhs, .. } => {
            replace_op(lhs, map);
            replace_op(rhs, map);
        }
        Instruction::UnaryOp { src, .. } => replace_op(src, map),
        Instruction::Cmp { lhs, rhs, .. } => {
            replace_op(lhs, map);
            replace_op(rhs, map);
        }

        // Pointer / address.
        Instruction::GetElementPtr { base, offset, .. } => {
            replace_val(base, map);
            replace_op(offset, map);
        }
        Instruction::DynAlloca { size, .. } => replace_op(size, map),
        Instruction::StackRestore { ptr } => replace_val(ptr, map),

        // Conversions.
        Instruction::Cast { src, .. } => replace_op(src, map),
        Instruction::Copy { src, .. } => replace_op(src, map),

        // Calls.
        Instruction::Call { info, .. } => {
            for arg in &mut info.args {
                replace_op(arg, map);
            }
        }
        Instruction::CallIndirect { func_ptr, info } => {
            replace_op(func_ptr, map);
            for arg in &mut info.args {
                replace_op(arg, map);
            }
        }

        // Phi.
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming {
                replace_op(op, map);
            }
        }

        // Select.
        Instruction::Select { cond, true_val, false_val, .. } => {
            replace_op(cond, map);
            replace_op(true_val, map);
            replace_op(false_val, map);
        }

        // Atomics.
        Instruction::AtomicRmw { ptr, val, .. } => {
            replace_op(ptr, map);
            replace_op(val, map);
        }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => {
            replace_op(ptr, map);
            replace_op(expected, map);
            replace_op(desired, map);
        }
        Instruction::AtomicLoad { ptr, .. } => replace_op(ptr, map),
        Instruction::AtomicStore { ptr, val, .. } => {
            replace_op(ptr, map);
            replace_op(val, map);
        }

        // Varargs.
        Instruction::VaArg { va_list_ptr, .. } => replace_val(va_list_ptr, map),
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, .. } => {
            replace_val(dest_ptr, map);
            replace_val(va_list_ptr, map);
        }
        Instruction::VaStart { va_list_ptr } => replace_val(va_list_ptr, map),
        Instruction::VaEnd { va_list_ptr } => replace_val(va_list_ptr, map),
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            replace_val(dest_ptr, map);
            replace_val(src_ptr, map);
        }

        // Inline assembly.
        Instruction::InlineAsm { inputs, .. } => {
            for (_, op, _) in inputs {
                replace_op(op, map);
            }
        }

        // Intrinsics.
        Instruction::Intrinsic { args, .. } => {
            for arg in args {
                replace_op(arg, map);
            }
        }

        // Complex-return helpers.
        Instruction::SetReturnF64Second { src } => replace_op(src, map),
        Instruction::SetReturnF32Second { src } => replace_op(src, map),
        Instruction::SetReturnF128Second { src } => replace_op(src, map),
    }
}

fn replace_values_in_terminator(term: &mut Terminator, map: &FxHashMap<u32, u32>) {
    match term {
        Terminator::Return(Some(op)) => replace_op(op, map),
        Terminator::CondBranch { cond, .. } => replace_op(cond, map),
        Terminator::IndirectBranch { target, .. } => replace_op(target, map),
        Terminator::Switch { val, .. } => replace_op(val, map),
        Terminator::Return(None) | Terminator::Branch(_) | Terminator::Unreachable => {}
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::{AddressSpace, IrType};
    use crate::ir::reexports::{BasicBlock, BlockId, IrConst, Value};

    /// Build a simple counting loop:
    ///   preheader → header → body → latch → (back to header) / exit
    ///
    /// ```
    /// preheader (B0):
    ///   %0 = Copy 0i32
    ///   Branch B1
    ///
    /// header (B1):
    ///   %1 = Phi [(%0, B0), (%5, B3)]   // i
    ///   %3 = Cmp Slt %1, const(n_val)   // limit is a compile-time constant
    ///   CondBranch %3, B2(body), B4(exit)
    ///
    /// body (B2):
    ///   %4 = GEP(arr, %1)
    ///   Store(0, %4)
    ///   Branch B3
    ///
    /// latch (B3):
    ///   %5 = Add %1, 1
    ///   Branch B1
    ///
    /// exit (B4):
    ///   Return void
    /// ```
    ///
    /// The limit is a constant so it is loop-invariant (not defined in loop.body).
    fn make_counting_loop(n_val: i32) -> IrFunction {
        let mut func =
            IrFunction::new("loop_test".to_string(), IrType::Void, vec![], false);

        // B0: preheader — init i = 0
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![Instruction::Copy {
                dest: Value(0),
                src: Operand::Const(IrConst::I32(0)),
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // B1: header — %1 = phi(0, %5); %3 = cmp %1 < const(n_val)
        // Limit is a constant → loop-invariant → eligible for unrolling.
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(1),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(0)), BlockId(0)),
                        (Operand::Value(Value(5)), BlockId(3)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(3),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(n_val)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(3)),
                true_label: BlockId(2), // continue (body)
                false_label: BlockId(4), // exit
            },
            source_spans: Vec::new(),
        });

        // B2: body — GEP + store
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::GetElementPtr {
                    dest: Value(4),
                    base: Value(10), // arr (loop-invariant, defined outside)
                    offset: Operand::Value(Value(1)),
                    ty: IrType::I32,
                },
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(0)),
                    ptr: Value(4),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Branch(BlockId(3)), // → latch
            source_spans: Vec::new(),
        });

        // B3: latch — %5 = %1 + 1; Branch B1
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![Instruction::BinOp {
                dest: Value(5),
                op: IrBinOp::Add,
                lhs: Operand::Value(Value(1)),
                rhs: Operand::Const(IrConst::I32(1)),
                ty: IrType::I32,
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // B4: exit
        func.blocks.push(BasicBlock {
            label: BlockId(4),
            instructions: vec![],
            terminator: Terminator::Return(None),
            source_spans: Vec::new(),
        });

        func.next_value_id = 11; // 0–10 used (10 = arr placeholder)
        func
    }

    #[test]
    fn test_basic_unroll_8x() {
        let mut func = make_counting_loop(100);
        let n = unroll_loops(&mut func);
        assert_eq!(n, 1, "should unroll exactly one loop");

        // Original 5 blocks + 7 exit_check blocks + 7 body_work clones = 19.
        assert_eq!(
            func.blocks.len(),
            19,
            "expected 5 original + 7 exit_checks + 7 clones = 19 blocks"
        );

        // The latch's Add should now use one of the new IV values (not Value(1)).
        let latch = func.blocks.iter().find(|b| b.label == BlockId(3)).unwrap();
        let iv_incr = latch
            .instructions
            .iter()
            .find(|i| matches!(i, Instruction::BinOp { op: IrBinOp::Add, .. }))
            .unwrap();
        if let Instruction::BinOp { lhs, .. } = iv_incr {
            assert!(
                !matches!(lhs, Operand::Value(v) if v.0 == 1),
                "latch IV increment should use iv_7 (not original iv_phi Value(1))"
            );
        }
    }

    #[test]
    fn test_no_unroll_call_in_body() {
        let mut func = make_counting_loop(100);
        // Insert a Call instruction into the body (B2).
        func.blocks[2].instructions.push(Instruction::Call {
            func: "some_func".to_string(),
            info: crate::ir::reexports::CallInfo {
                dest: None,
                args: vec![],
                arg_types: vec![],
                return_type: IrType::Void,
                is_variadic: false,
                num_fixed_args: 0,
                struct_arg_sizes: vec![],
                struct_arg_aligns: vec![],
                struct_arg_classes: vec![],
                struct_arg_riscv_float_classes: vec![],
                is_sret: false,
                is_fastcall: false,
                ret_eightbyte_classes: vec![],
            },
        });
        let n = unroll_loops(&mut func);
        assert_eq!(n, 0, "loop with call should not be unrolled");
        assert_eq!(func.blocks.len(), 5, "block count should be unchanged");
    }

    #[test]
    fn test_no_unroll_large_body() {
        // Build a loop whose body has > 60 instructions → factor = 1 → no unroll.
        let mut func = make_counting_loop(100);
        // Pad body (B2) with NOPs (Copy %0 = %0) until > 60 instructions.
        for _ in 0..65 {
            func.blocks[2].instructions.push(Instruction::Copy {
                dest: Value(0),
                src: Operand::Value(Value(0)),
            });
        }
        let n = unroll_loops(&mut func);
        assert_eq!(n, 0, "loop with > 60 body instructions should not be unrolled");
    }

    #[test]
    fn test_no_unroll_no_preheader() {
        // Make the header have two entry predecessors (no unique preheader).
        let mut func = make_counting_loop(100);
        // Add a second predecessor to the header (B1) from B4 (exit).
        func.blocks[4].terminator = Terminator::Branch(BlockId(1));
        // Also extend B1's phi to include B4.
        if let Instruction::Phi { incoming, .. } = &mut func.blocks[1].instructions[0] {
            incoming.push((Operand::Value(Value(0)), BlockId(4)));
        }
        let n = unroll_loops(&mut func);
        assert_eq!(n, 0, "loop without unique preheader should not be unrolled");
    }

    #[test]
    fn test_no_unroll_nested_loop_outer() {
        // The outer loop's body_work contains the inner loop's header —
        // the outer loop must NOT be unrolled, but the inner loop IS unrolled.
        //
        // Structure:
        //   B0 (outer preheader) → B1 (outer header)
        //   B1: %i = phi, cmp i < 10 → B2(inner hdr) or B6(outer exit)
        //   B2 (inner header): %j = phi, cmp j < 10 → B2b(inner body) or B5(outer latch)
        //   B2b (inner body): a Copy instruction → B3(inner latch)
        //   B3 (inner latch): %j_next = j+1 → B2 (back-edge)
        //   B5 (outer latch): %i_next = i+1 → B1 (back-edge)
        //   B6 (outer exit): Return
        //
        // Inner loop: {B2, B2b, B3}, body_work={B2b}, header=B2, latch=B3 → can unroll.
        // Outer loop: {B1, B2, B2b, B3, B5}, body_work={B2, B2b, B3}, header=B1, latch=B5
        //   → body_work contains B2 which is a loop header → outer NOT unrolled.
        let mut func =
            IrFunction::new("nested".to_string(), IrType::Void, vec![], false);

        // B0: outer preheader
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![Instruction::Copy {
                dest: Value(0),
                src: Operand::Const(IrConst::I32(0)),
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // B1: outer header — %1 = phi(%0, %10); cmp %1 < 10
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(1),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(0)), BlockId(0)),
                        (Operand::Value(Value(10)), BlockId(5)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(2),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(10)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(2)),
                true_label: BlockId(2), // inner header
                false_label: BlockId(6), // outer exit
            },
            source_spans: Vec::new(),
        });

        // B2: inner header — %3 = phi(%1, %7); cmp %3 < 10
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(3),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(1)), BlockId(1)),
                        (Operand::Value(Value(7)), BlockId(3)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(4),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(10)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(4)),
                true_label: BlockId(20), // inner body (B2b)
                false_label: BlockId(5), // outer latch (inner exit)
            },
            source_spans: Vec::new(),
        });

        // B2b (BlockId 20): inner body — a single Copy; branches to inner latch
        func.blocks.push(BasicBlock {
            label: BlockId(20),
            instructions: vec![Instruction::Copy {
                dest: Value(20),
                src: Operand::Const(IrConst::I32(0)),
            }],
            terminator: Terminator::Branch(BlockId(3)), // → inner latch
            source_spans: Vec::new(),
        });

        // B3: inner latch — %7 = %3+1; back to inner header
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![Instruction::BinOp {
                dest: Value(7),
                op: IrBinOp::Add,
                lhs: Operand::Value(Value(3)),
                rhs: Operand::Const(IrConst::I32(1)),
                ty: IrType::I32,
            }],
            terminator: Terminator::Branch(BlockId(2)), // back to inner header
            source_spans: Vec::new(),
        });

        // B5: outer latch — %10 = %1+1; back to outer header
        func.blocks.push(BasicBlock {
            label: BlockId(5),
            instructions: vec![Instruction::BinOp {
                dest: Value(10),
                op: IrBinOp::Add,
                lhs: Operand::Value(Value(1)),
                rhs: Operand::Const(IrConst::I32(1)),
                ty: IrType::I32,
            }],
            terminator: Terminator::Branch(BlockId(1)), // back to outer header
            source_spans: Vec::new(),
        });

        // B6: outer exit
        func.blocks.push(BasicBlock {
            label: BlockId(6),
            instructions: vec![],
            terminator: Terminator::Return(None),
            source_spans: Vec::new(),
        });

        func.next_value_id = 21;

        let n = unroll_loops(&mut func);

        // Outer loop must NOT be unrolled (body_work contains inner header B2).
        let outer_latch = func.blocks.iter().find(|b| b.label == BlockId(5)).unwrap();
        assert!(
            matches!(outer_latch.terminator, Terminator::Branch(lbl) if lbl == BlockId(1)),
            "outer latch should still branch to outer header"
        );

        // Inner loop (body_work = {B2b} with 1 instruction) should be unrolled.
        assert_eq!(n, 1, "only the inner loop should be unrolled");
    }

    #[test]
    fn test_value_ids_unique_after_unroll() {
        // After unrolling, all Value IDs must be distinct (no duplicates in all
        // block instructions). This catches the "reuse old val IDs" bug.
        let mut func = make_counting_loop(16);
        unroll_loops(&mut func);

        let mut seen: FxHashSet<u32> = FxHashSet::default();
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    assert!(
                        seen.insert(dest.0),
                        "duplicate Value({}) after unrolling",
                        dest.0
                    );
                }
            }
        }
    }
}
