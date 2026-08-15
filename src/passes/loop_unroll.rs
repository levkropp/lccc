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
    // Full unrolling of constant-trip loops first: it eliminates loops
    // entirely (exposing constant GEP offsets for aggregate scalarization),
    // and iterating internally handles trip counts that only become constant
    // after an outer loop is unrolled.
    let mut count = full_unroll_constant_loops(func);

    let cfg = CfgAnalysis::build(func);
    let raw = loop_analysis::find_natural_loops(
        cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom,
    );
    if raw.is_empty() {
        return count;
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

// ── Full unrolling of constant-trip loops ────────────────────────────────────
//
// A counted loop with a constant trip count (`for (i = 0; i < 4; i++) ...`)
// is replaced by straight-line code: the body is cloned once per iteration,
// the IV is substituted with its constant value per iteration, and all loop
// control disappears. Unlike the partial unroller, this eliminates the loop
// entirely — which turns loop-variable GEP offsets (e.g. `particles[i].x`
// at `i * 48`) into constant offsets, the gateway to scalarizing aggregate
// locals (struct_copy's make/distance loops).
//
// Only the canonical shape is accepted: header = phis + one Cmp + CondBranch,
// single latch, unique preheader, no nested loops, no calls/atomics/asm,
// `iv < const_limit` continue-on-true, constant init and step, trip count
// 1..=16, and a bounded unrolled size. Header phis may carry arbitrary
// loop-carried values (accumulators); they resolve through the rename chain.

/// A planned full unroll.
struct FullUnrollPlan {
    header: usize,
    /// Ordered blocks cloned per iteration. Shape A (header != latch):
    /// body_work + [latch]. Shape B (single-block loop): [header].
    work: Vec<usize>,
    /// Index into `work` of the block entered at iteration start.
    entry_idx: usize,
    /// The block carrying the back-edge (shape A: the latch; shape B: header).
    latch: usize,
    /// Instruction dests stripped from every copy (header phis, the exit cmp,
    /// and the Copy/Cast chain feeding it).
    drop_dests: FxHashSet<u32>,
    /// Shape B: the work block is the header itself; strip drop_dests from it.
    single_block: bool,
    exit_target: BlockId,
    /// (phi dest, preheader incoming, latch incoming) for every header phi.
    phis: Vec<(Value, Operand, Operand)>,
    /// Resolvable header Copy dests: (dest, src operand) — dropped from the
    /// header and substituted per iteration.
    header_copies: Vec<(u32, Operand)>,
    /// Resolvable header const-Cast dests: (dest, folded value, ty).
    header_casts: Vec<(u32, i64, IrType)>,
    /// All values defined inside the loop (for outside-use substitution).
    loop_defs: FxHashSet<u32>,
    iv_phi: Value,
    iv_ty: IrType,
    init: i64,
    step: i64,
    trip: usize,
}

/// Run full unrolling on one function; returns the number of loops unrolled.
/// Iterates internally so that loops whose trip count becomes constant only
/// after an outer loop was unrolled (e.g. `j = i+1 .. 4`) are also unrolled.
/// Run full unrolling on one function; returns the number of loops unrolled.
/// Iterates internally so that loops whose trip count becomes constant only
/// after an outer loop was unrolled (e.g. `j = i+1 .. 4`) are also unrolled.
///
/// Currently OPT-IN (CCC_FULL_UNROLL=1): on the target workload (struct_copy's
/// inlined make/distance loops) unrolling alone does not eliminate aggregate
/// memory traffic and the code growth is a net loss, so it is off by default
/// pending a working store->load forwarding chain.
pub(crate) fn full_unroll_constant_loops(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_FULL_UNROLL").is_err() {
        return 0;
    }
    let mut total = 0;
    for _ in 0..32 {
        let n = full_unroll_once(func);
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}

fn full_unroll_once(func: &mut IrFunction) -> usize {
    if func.blocks.len() < 2 {
        return 0;
    }
    let cfg = CfgAnalysis::build(func);
    let raw = loop_analysis::find_natural_loops(cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom);
    if raw.is_empty() {
        return 0;
    }
    let loops = loop_analysis::merge_loops_by_header(raw);
    let all_headers: FxHashSet<usize> = loops.iter().map(|l| l.header).collect();

    // Smallest body first = innermost loops first.
    let mut plans: Vec<FullUnrollPlan> = loops
        .iter()
        .filter_map(|lp| analyze_full_unroll(func, lp, &cfg, &all_headers))
        .collect();
    if std::env::var("CCC_DEBUG_FULL_UNROLL").is_ok() {
        eprintln!(
            "[FULL_UNROLL] func={} loops={} plans={} [{}]",
            func.name,
            loops.len(),
            plans.len(),
            plans.iter().map(|p| format!("trip={}", p.trip)).collect::<Vec<_>>().join(",")
        );
        for p in &plans {
            eprintln!(
                "[FULL_UNROLL]   plan header={} trip={} work_labels={:?} entry_idx={} single={}",
                p.header,
                p.trip,
                p.work.iter().map(|&bi| func.blocks[bi].label.0).collect::<Vec<_>>(),
                p.entry_idx,
                p.single_block
            );
        }
    }
    plans.sort_by_key(|p| p.work.len());
    // Apply ONE plan per re-analysis round (innermost first). Applying several
    // plans computed from the same pre-transform state is unsound: the first
    // plan's mutations (iteration-0 substitution, escape rewrites) would be
    // re-cloned by later plans with stale maps. The driver loop re-analyzes.
    let mut n = 0;
    for plan in plans {
        if apply_full_unroll(func, &plan) {
            n += 1;
            break;
        }
    }
    n
}

/// Resolve a value to a compile-time constant, following Copy-of-const and
/// simple constant BinOp chains (bounded depth).
fn const_value_of(func: &IrFunction, val_id: u32, depth: u32) -> Option<i64> {
    if depth > 8 {
        return None;
    }
    let mut def: Option<&Instruction> = None;
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.dest().is_some_and(|d| d.0 == val_id) {
                def = Some(inst);
                break;
            }
        }
        if def.is_some() {
            break;
        }
    }
    let def = def?;
    let op_val = |op: &Operand, depth: u32| -> Option<i64> {
        match op {
            Operand::Const(c) => c.to_i64(),
            Operand::Value(v) => const_value_of(func, v.0, depth + 1),
        }
    };
    match def {
        Instruction::Copy { src, .. } => op_val(src, depth),
        Instruction::Cast { src, from_ty, to_ty, .. }
            if from_ty.is_integer() && to_ty.is_integer() =>
        {
            op_val(src, depth)
        }
        Instruction::BinOp { op, lhs, rhs, .. } => {
            let l = op_val(lhs, depth)?;
            let r = op_val(rhs, depth)?;
            match op {
                IrBinOp::Add => Some(l.wrapping_add(r)),
                IrBinOp::Sub => Some(l.wrapping_sub(r)),
                IrBinOp::Mul => Some(l.wrapping_mul(r)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Follow single-def Copy chains for a value, optionally restricted to a
/// defining block (`usize::MAX` = anywhere). Returns the final value id.
fn unwrap_copy_in(func: &IrFunction, mut val: u32, block: usize) -> u32 {
    for _ in 0..16 {
        let mut next = None;
        for (bi, b) in func.blocks.iter().enumerate() {
            if block != usize::MAX && bi != block {
                continue;
            }
            for inst in &b.instructions {
                if let Instruction::Copy { dest, src: Operand::Value(s) } = inst {
                    if dest.0 == val {
                        next = Some(s.0);
                    }
                }
            }
        }
        match next {
            Some(s) if s != val => val = s,
            _ => break,
        }
    }
    val
}

/// Is `v` defined by a Phi in block `header`?
fn plan_free_is_phi(func: &IrFunction, header: usize, v: u32) -> bool {
    func.blocks[header]
        .instructions
        .iter()
        .any(|i| matches!(i, Instruction::Phi { dest, .. } if dest.0 == v))
}

fn analyze_full_unroll(
    func: &IrFunction,
    lp: &loop_analysis::NaturalLoop,
    cfg: &CfgAnalysis,
    all_headers: &FxHashSet<usize>,
) -> Option<FullUnrollPlan> {
    let header = lp.header;
    let dbg = std::env::var("CCC_DEBUG_FULL_UNROLL").is_ok();
    macro_rules! bail {
        ($why:expr) => {{
            if dbg {
                eprintln!("[FULL_UNROLL]   func={} header={} rejected: {}", func.name, header, $why);
            }
            return None;
        }};
    }

    if lp.body.len() > MAX_UNROLL_BODY_BLOCKS + 2 {
        bail!("too many blocks");
    }

    // Single back-edge source.
    let back_preds: Vec<usize> = cfg
        .preds
        .row(header)
        .iter()
        .map(|&p| p as usize)
        .filter(|p| lp.body.contains(p))
        .collect();
    if back_preds.len() != 1 {
        bail!("multi latch");
    }
    let latch = back_preds[0];
    let single_block = latch == header;
    let header_label = func.blocks[header].label;

    // Back-edge shape: shape A latch ends in Branch(header); in shape B the
    // header's own CondBranch carries the self edge.
    if single_block {
        match &func.blocks[header].terminator {
            Terminator::CondBranch { true_label, false_label, .. }
                if *true_label == header_label || *false_label == header_label => {}
            _ => bail!("single-block loop without self edge"),
        }
    } else {
        match &func.blocks[latch].terminator {
            Terminator::Branch(lbl) if *lbl == header_label => {}
            _ => bail!("latch not back-branch"),
        }
    }

    let preheader = loop_analysis::find_preheader(header, &lp.body, &cfg.preds);
    let Some(preheader) = preheader else { bail!("no preheader") };
    let preheader_label = func.blocks[preheader].label;

    // Nested loops inside the body are fine: they are cloned wholesale with
    // label remapping (natural-loop bodies nest properly by construction).
    // Disqualifying instructions anywhere in the loop.
    for &bi in lp.body.iter() {
        for inst in &func.blocks[bi].instructions {
            match inst {
                Instruction::Call { .. }
                | Instruction::CallIndirect { .. }
                | Instruction::InlineAsm { .. }
                | Instruction::AtomicRmw { .. }
                | Instruction::AtomicCmpxchg { .. }
                | Instruction::AtomicLoad { .. }
                | Instruction::AtomicStore { .. }
                | Instruction::DynAlloca { .. } => bail!("disqualified instr"),
                _ => {}
            }
        }
    }

    // Collect header phis (two incomings: preheader + latch).
    let mut phis: Vec<(Value, Operand, Operand)> = Vec::new();
    let mut drop_dests: FxHashSet<u32> = FxHashSet::default();
    let mut cmp_dests: Vec<u32> = Vec::new();
    let mut header_copies: Vec<(u32, Operand)> = Vec::new();
    let mut header_casts: Vec<(u32, i64, IrType)> = Vec::new();
    let mut header_nonconst_casts: Vec<u32> = Vec::new();
    let mut header_other = false;
    for inst in &func.blocks[header].instructions {
        match inst {
            Instruction::Phi { dest, incoming, .. } => {
                if incoming.len() != 2 {
                    bail!("phi with != 2 incomings");
                }
                let pre_op = incoming.iter().find(|(_, l)| *l == preheader_label).map(|(op, _)| *op);
                let latch_op = incoming
                    .iter()
                    .find(|(_, l)| *l == func.blocks[latch].label)
                    .map(|(op, _)| *op);
                match (pre_op, latch_op) {
                    (Some(p), Some(l)) => {
                        phis.push((*dest, p, l));
                        drop_dests.insert(dest.0);
                    }
                    _ => bail!("phi incoming mismatch"),
                }
            }
            Instruction::Cmp { dest, .. } => cmp_dests.push(dest.0),
            Instruction::Copy { dest, src } => {
                header_copies.push((dest.0, *src));
                drop_dests.insert(dest.0);
            }
            Instruction::Cast { dest, src, to_ty, .. } => {
                let folded = match src {
                    Operand::Const(c) => c.to_i64().and_then(|v| fold_const_cast(v, *to_ty)),
                    _ => None,
                };
                match folded {
                    Some(v) => header_casts.push((dest.0, v, *to_ty)),
                    None => header_nonconst_casts.push(dest.0),
                }
                drop_dests.insert(dest.0);
            }
            _ => header_other = true,
        }
    }
    if cmp_dests.len() != 1 || phis.is_empty() {
        bail!("bad cmp/phi count");
    }
    let cmp_dest = cmp_dests[0];

    // The cond-aux chain: Copy/Cast dests in the header feeding the CondBranch's
    // condition through the exit Cmp. Only these may be dropped as control.
    let (cond_op, _, _) = match &func.blocks[header].terminator {
        Terminator::CondBranch { cond, true_label, false_label } => (*cond, *true_label, *false_label),
        _ => bail!("header not condbranch"),
    };
    let cond_id = match cond_op {
        Operand::Value(v) => v.0,
        _ => bail!("const cond"),
    };
    let mut aux_set: FxHashSet<u32> = FxHashSet::default();
    {
        let mut worklist = vec![cond_id];
        while let Some(v) = worklist.pop() {
            // Never treat phi dests as aux: they are resolved per iteration.
            let is_phi = plan_free_is_phi(func, header, v);
            if is_phi {
                continue;
            }
            if !aux_set.insert(v) {
                continue;
            }
            for inst in &func.blocks[header].instructions {
                if inst.dest().is_some_and(|d| d.0 == v) {
                    match inst {
                        Instruction::Copy { src, .. } | Instruction::Cast { src, .. } => {
                            if let Operand::Value(s) = src {
                                worklist.push(s.0);
                            }
                        }
                        Instruction::Cmp { lhs, rhs, .. } => {
                            for op in [lhs, rhs] {
                                if let Operand::Value(s) = op {
                                    worklist.push(s.0);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    if !aux_set.contains(&cmp_dest) {
        bail!("cond does not trace to the cmp");
    }
    // Aux dests must only be used inside the aux chain itself (and the
    // terminator). A use anywhere else means the value escapes the control
    // chain and cannot be dropped.
    for (bi, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            let inst_dest = inst.dest().map(|d| d.0);
            let mut bad = false;
            for_each_value_use_in_inst_simple(inst, &mut |vid| {
                if aux_set.contains(&vid) && !inst_dest.is_some_and(|d| aux_set.contains(&d)) {
                    bad = true;
                }
            });
            if bad {
                bail!("aux value escapes chain");
            }
            let _ = bi;
        }
    }
    drop_dests.extend(aux_set.iter().copied());

    // Non-const Casts in the header are only droppable as part of the cond
    // chain; any other use would need a real (sign-changing) cast value.
    for d in &header_nonconst_casts {
        if !aux_set.contains(d) {
            bail!("non-const cast in header outside cond chain");
        }
    }

    // For shape A (multi-block), the header must contain ONLY phis + droppable
    // plumbing (any other instruction would be loop work living in the header,
    // which shape A does not clone).
    if !single_block && header_other {
        if dbg {
            for inst in &func.blocks[header].instructions {
                if let Some(d) = inst.dest() {
                    if !drop_dests.contains(&d.0) {
                        eprintln!("[FULL_UNROLL]     header inst: {:?}", inst);
                    }
                }
            }
        }
        bail!("work instruction in header (shape A)");
    }

    // IV: a header phi whose latch incoming (through Copies) is
    // `Add(phi, const_step)` in the latch.
    let mut iv_found: Option<(Value, IrType, i64)> = None;
    for (phi_dest, _pre_op, latch_op) in &phis {
        let Some(phi_inst) = func.blocks[header]
            .instructions
            .iter()
            .find(|i| i.dest().is_some_and(|d| d.0 == phi_dest.0))
        else {
            continue;
        };
        let Instruction::Phi { ty, .. } = phi_inst else { continue };
        if !ty.is_integer() {
            continue;
        }
        let latch_val = match latch_op {
            Operand::Value(v) => v.0,
            _ => continue,
        };
        let incr_def = unwrap_copy_in(func, latch_val, latch);
        for latch_inst in &func.blocks[latch].instructions {
            let Instruction::BinOp { dest, op: IrBinOp::Add, lhs, rhs, .. } = latch_inst else {
                continue;
            };
            if dest.0 != incr_def {
                continue;
            }
            let step = match (lhs, rhs) {
                (Operand::Value(v), Operand::Const(c))
                    if unwrap_copy_in(func, v.0, usize::MAX) == phi_dest.0 =>
                    c.to_i64(),
                (Operand::Const(c), Operand::Value(v))
                    if unwrap_copy_in(func, v.0, usize::MAX) == phi_dest.0 =>
                    c.to_i64(),
                _ => None,
            };
            if let Some(step) = step {
                if step > 0 {
                    iv_found = Some((*phi_dest, *ty, step));
                }
                break;
            }
        }
        if iv_found.is_some() {
            break;
        }
    }
    let Some((iv_phi, iv_ty, iv_step)) = iv_found else { bail!("no iv") };

    // Exit condition: `iv < const_limit`, continue-on-true.
    let (true_label, false_label) = match &func.blocks[header].terminator {
        Terminator::CondBranch { true_label, false_label, .. } => (*true_label, *false_label),
        _ => bail!("no condbranch"),
    };
    let label_to_idx: FxHashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();
    let true_in_loop = label_to_idx.get(&true_label).map(|&bi| lp.body.contains(&bi)).unwrap_or(false);
    let false_in_loop = label_to_idx.get(&false_label).map(|&bi| lp.body.contains(&bi)).unwrap_or(false);
    if true_in_loop == false_in_loop {
        bail!("both/neither target in loop");
    }
    let (exit_target, body_entry_label, exit_cond_positive) = if !true_in_loop {
        (true_label, false_label, true)
    } else {
        (false_label, true_label, false)
    };
    let cmp_inst = func.blocks[header]
        .instructions
        .iter()
        .find(|i| matches!(i, Instruction::Cmp { dest, .. } if dest.0 == cmp_dest))
        .unwrap();
    let Instruction::Cmp { op: exit_cmp_op, lhs: cmp_lhs, rhs: cmp_rhs, .. } = cmp_inst else {
        bail!("cmp gone");
    };
    let (iv_is_lhs, limit) = {
        let l_iv = matches!(cmp_lhs, Operand::Value(v) if unwrap_copy_in(func, v.0, usize::MAX) == iv_phi.0);
        let r_iv = matches!(cmp_rhs, Operand::Value(v) if unwrap_copy_in(func, v.0, usize::MAX) == iv_phi.0);
        if l_iv && !r_iv {
            let lim = match cmp_rhs {
                Operand::Const(c) => c.to_i64()?,
                Operand::Value(v) => const_value_of(func, v.0, 0)?,
            };
            (true, lim)
        } else if r_iv && !l_iv {
            let lim = match cmp_lhs {
                Operand::Const(c) => c.to_i64()?,
                Operand::Value(v) => const_value_of(func, v.0, 0)?,
            };
            (false, lim)
        } else {
            bail!("cmp does not involve the iv");
        }
    };
    if *exit_cmp_op != IrCmpOp::Slt || !iv_is_lhs || exit_cond_positive {
        bail!("not iv < const continue-on-true");
    }
    let init_op = phis.iter().find(|(d, _, _)| d.0 == iv_phi.0)?.1;
    let init = match init_op {
        Operand::Const(c) => c.to_i64()?,
        Operand::Value(v) => const_value_of(func, v.0, 0)?,
    };
    if limit <= init {
        bail!("zero trip");
    }
    let trip64 = (limit - init + iv_step - 1) / iv_step;
    let max_trip: i64 = std::env::var("CCC_FULL_UNROLL_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    if !(1..=max_trip).contains(&trip64) {
        bail!("trip out of range");
    }
    let trip = trip64 as usize;

    // Ordered work sequence cloned per iteration.
    let work: Vec<usize> = if single_block {
        vec![header]
    } else {
        lp.body
            .iter()
            .copied()
            .filter(|&b| b != header && b != latch)
            .chain(std::iter::once(latch))
            .collect()
    };
    let entry_idx = work
        .iter()
        .position(|&bi| func.blocks[bi].label == body_entry_label)
        .or_else(|| if single_block { Some(0) } else { None });
    let Some(entry_idx) = entry_idx else { bail!("body entry not in work") };

    // Size budget.
    let work_insts: usize = work.iter().map(|&bi| func.blocks[bi].instructions.len()).sum();
    if work_insts * trip > 512 {
        bail!("too big");
    }

    // Escape analysis: a loop-defined value may be used outside the loop when
    // the use is dominated by the loop's exit edge — after unrolling it reads
    // the final iteration's value, which the transform substitutes. Outside
    // *definitions* of a loop value are allowed only as "init" writes that
    // dominate the loop header (the loop's writes always overwrite them before
    // any escape read).
    let mut loop_defs: FxHashSet<u32> = FxHashSet::default();
    for &bi in lp.body.iter() {
        for inst in &func.blocks[bi].instructions {
            if let Some(d) = inst.dest() {
                loop_defs.insert(d.0);
            }
        }
    }
    let Some(exit_bi) = func.blocks.iter().position(|b| b.label == exit_target) else {
        bail!("no exit block");
    };
    // The exit block must follow the loop in text order: clones are spliced
    // immediately after the last loop block, and liveness computes intervals
    // in text order, so any clone-defined value must precede its uses.
    if lp.body.iter().any(|&bi| bi >= exit_bi) {
        bail!("exit block precedes loop body in text order");
    }
    // Non-phi outside uses are substituted with the final loop value, which is
    // only correct if every path to them traversed the loop: require the exit
    // block's sole predecessor to be the header edge.
    let exit_single_pred = cfg
        .preds
        .row(exit_bi)
        .iter()
        .all(|&p| p as usize == header);
    for (bi, block) in func.blocks.iter().enumerate() {
        if lp.body.contains(&bi) {
            continue;
        }
        for inst in &block.instructions {
            if let Some(d) = inst.dest() {
                if loop_defs.contains(&d.0) && !dominates(&cfg.idom, bi, header) {
                    bail!("outside redef of loop value");
                }
            }
            match inst {
                Instruction::Phi { incoming, .. } => {
                    for (op, lbl) in incoming {
                        if let Operand::Value(v) = op {
                            if !loop_defs.contains(&v.0) {
                                continue;
                            }
                            // The exit-phi incoming along the header edge is
                            // rewritten to the final value by the transform.
                            if bi == exit_bi && *lbl == header_label {
                                continue;
                            }
                            let src_ok = label_to_idx
                                .get(lbl)
                                .map(|&sbi| dominates(&cfg.idom, exit_bi, sbi))
                                .unwrap_or(false);
                            if !src_ok || !exit_single_pred {
                                bail!("phi escape from non-dominated edge");
                            }
                        }
                    }
                }
                _ => {
                    let mut bad = false;
                    for_each_value_use_in_inst_simple(inst, &mut |vid| {
                        if loop_defs.contains(&vid)
                            && !(dominates(&cfg.idom, exit_bi, bi) && exit_single_pred)
                        {
                            bad = true;
                        }
                    });
                    if bad {
                        bail!("loop value used outside (non-dominated)");
                    }
                }
            }
        }
        let mut bad = false;
        for_each_value_use_in_term_simple(&block.terminator, &mut |vid| {
            if loop_defs.contains(&vid) && !(dominates(&cfg.idom, exit_bi, bi) && exit_single_pred) {
                bad = true;
            }
        });
        if bad {
            bail!("loop value used in terminator outside");
        }
    }

    Some(FullUnrollPlan {
        header,
        work,
        entry_idx,
        latch,
        drop_dests,
        single_block,
        exit_target,
        phis,
        header_copies,
        header_casts,
        loop_defs,
        iv_phi,
        iv_ty,
        init,
        step: iv_step,
        trip,
    })
}

/// Does block `a` dominate block `b`? (Walks b's immediate-dominator chain.)
fn dominates(idom: &[usize], a: usize, b: usize) -> bool {
    let mut cur = b;
    for _ in 0..=idom.len() {
        if cur == a {
            return true;
        }
        let next = idom.get(cur).copied().unwrap_or(usize::MAX);
        if next == usize::MAX || next == cur {
            break;
        }
        cur = next;
    }
    false
}

/// Fold an integer Cast of a constant to the destination type.
fn fold_const_cast(v: i64, to_ty: IrType) -> Option<i64> {
    match to_ty {
        IrType::I8 => Some(v as i8 as i64),
        IrType::U8 => Some(v as u8 as i64),
        IrType::I16 => Some(v as i16 as i64),
        IrType::U16 => Some(v as u16 as i64),
        IrType::I32 => Some(v as i32 as i64),
        IrType::U32 => Some(v as u32 as i64),
        IrType::I64 | IrType::U64 => Some(v),
        _ => None,
    }
}

/// Minimal per-instruction value-use walker for the safety check.
fn for_each_value_use_in_inst_simple(inst: &Instruction, f: &mut impl FnMut(u32)) {
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => {
            for op in [lhs, rhs] {
                if let Operand::Value(v) = op {
                    f(v.0);
                }
            }
        }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } => {
            if let Operand::Value(v) = src {
                f(v.0);
            }
        }
        Instruction::Copy { src, .. } => {
            if let Operand::Value(v) = src {
                f(v.0);
            }
        }
        Instruction::Store { val, ptr, .. } => {
            if let Operand::Value(v) = val {
                f(v.0);
            }
            f(ptr.0);
        }
        Instruction::Load { ptr, .. } => f(ptr.0),
        Instruction::GetElementPtr { base, offset, .. } => {
            f(base.0);
            if let Operand::Value(v) = offset {
                f(v.0);
            }
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            for op in [cond, true_val, false_val] {
                if let Operand::Value(v) = op {
                    f(v.0);
                }
            }
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming {
                if let Operand::Value(v) = op {
                    f(v.0);
                }
            }
        }
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            for a in &info.args {
                if let Operand::Value(v) = a {
                    f(v.0);
                }
            }
        }
        Instruction::Memcpy { dest, src, .. } => {
            f(dest.0);
            f(src.0);
        }
        _ => {}
    }
}

fn for_each_value_use_in_term_simple(term: &Terminator, f: &mut impl FnMut(u32)) {
    match term {
        Terminator::Return(Some(op)) | Terminator::CondBranch { cond: op, .. } => {
            if let Operand::Value(v) = op {
                f(v.0);
            }
        }
        Terminator::Switch { val, .. } => {
            if let Operand::Value(v) = val {
                f(v.0);
            }
        }
        Terminator::IndirectBranch { target, .. } => {
            if let Operand::Value(v) = target {
                f(v.0);
            }
        }
        _ => {}
    }
}

fn apply_full_unroll(func: &mut IrFunction, plan: &FullUnrollPlan) -> bool {
    let trip = plan.trip;
    let header = plan.header;
    let header_label = func.blocks[header].label;
    let clone_order: Vec<usize> = plan.work.clone();

    // ── Fresh labels for clones (iterations 1..trip) ──────────────────────
    let max_label = func.blocks.iter().map(|b| b.label.0).max().unwrap_or(0);
    let mut next_label = max_label + 1;
    let cl_labels: Vec<Vec<BlockId>> = (1..trip)
        .map(|_| {
            clone_order
                .iter()
                .map(|_| {
                    let l = BlockId(next_label);
                    next_label += 1;
                    l
                })
                .collect()
        })
        .collect();

    // ── Per-iteration operand maps ─────────────────────────────────────────
    let mut next_val = func.next_value_id;
    let mut op_maps: Vec<FxHashMap<u32, Operand>> = Vec::with_capacity(trip);
    for k in 0..trip {
        let mut m: FxHashMap<u32, Operand> = FxHashMap::default();
        if k > 0 {
            for &bi in &clone_order {
                for inst in &func.blocks[bi].instructions {
                    if let Some(d) = inst.dest() {
                        if plan.drop_dests.contains(&d.0) {
                            continue; // stripped from every copy
                        }
                        m.entry(d.0).or_insert_with(|| {
                            let v = Value(next_val);
                            next_val += 1;
                            Operand::Value(v)
                        });
                    }
                }
            }
        }
        for (dest, pre_op, latch_op) in &plan.phis {
            let resolved: Operand = if dest.0 == plan.iv_phi.0 {
                Operand::Const(IrConst::from_i64(plan.init + (k as i64) * plan.step, plan.iv_ty))
            } else if k == 0 {
                *pre_op
            } else {
                match latch_op {
                    Operand::Const(c) => Operand::Const(*c),
                    Operand::Value(v) => op_maps[k - 1].get(&v.0).copied().unwrap_or(Operand::Value(*v)),
                }
            };
            m.insert(dest.0, resolved);
        }
        // Resolvable header plumbing (copies of phis, const casts) gets the
        // per-iteration resolution too, so body uses substitute correctly.
        // Chains (copy of copy) are handled by iterating to a fixpoint.
        for _ in 0..3 {
            for (dest, src) in &plan.header_copies {
                let resolved = match src {
                    Operand::Const(c) => Operand::Const(*c),
                    Operand::Value(v) => m.get(&v.0).copied().unwrap_or(Operand::Value(*v)),
                };
                m.insert(*dest, resolved);
            }
        }
        for (dest, val, ty) in &plan.header_casts {
            m.insert(*dest, Operand::Const(IrConst::from_i64(*val, *ty)));
        }
        op_maps.push(m);
    }
    func.next_value_id = next_val;

    // Exit resolution per phi: latch incoming resolved through the final map.
    let mut exit_res: FxHashMap<u32, Operand> = FxHashMap::default();
    for _ in 0..=plan.phis.len() {
        for (dest, _pre_op, latch_op) in &plan.phis {
            let resolved = if dest.0 == plan.iv_phi.0 {
                Operand::Const(IrConst::from_i64(plan.init + (trip as i64) * plan.step, plan.iv_ty))
            } else {
                match latch_op {
                    Operand::Const(c) => Operand::Const(*c),
                    Operand::Value(v) => exit_res
                        .get(&v.0)
                        .copied()
                        .or_else(|| op_maps[trip - 1].get(&v.0).copied())
                        .unwrap_or(Operand::Value(*v)),
                }
            };
            exit_res.insert(dest.0, resolved);
        }
    }

    // ── Build clones for iterations 1..trip from the PRISTINE originals ────
    let mut new_blocks: Vec<BasicBlock> = Vec::new();
    for k in 1..trip {
        let cj = k - 1;
        let mut blk_map: FxHashMap<BlockId, BlockId> = FxHashMap::default();
        for (i, &bi) in clone_order.iter().enumerate() {
            blk_map.insert(func.blocks[bi].label, cl_labels[cj][i]);
        }
        for (i, &bi) in clone_order.iter().enumerate() {
            let orig = &func.blocks[bi];
            let new_insts: Vec<Instruction> = orig
                .instructions
                .iter()
                .filter(|inst| {
                    inst.dest().map(|d| !plan.drop_dests.contains(&d.0)).unwrap_or(true)
                })
                .map(|inst| {
                    let mut cloned = inst.clone();
                    substitute_operands_in_inst(&mut cloned, &op_maps[k]);
                    let mut dmap: FxHashMap<u32, u32> = FxHashMap::default();
                    if let Some(d) = cloned.dest() {
                        if let Some(Operand::Value(nv)) = op_maps[k].get(&d.0) {
                            dmap.insert(d.0, nv.0);
                        }
                    }
                    rename_inst_dest(&mut cloned, &dmap);
                    // Remap phi incoming labels inside cloned (nested-loop) blocks.
                    if let Instruction::Phi { incoming, .. } = &mut cloned {
                        for (_, lbl) in incoming.iter_mut() {
                            if let Some(&new) = blk_map.get(lbl) {
                                *lbl = new;
                            }
                        }
                    }
                    cloned
                })
                .collect();
            let mut new_term = orig.terminator.clone();
            substitute_operands_in_terminator(&mut new_term, &op_maps[k]);
            replace_block_ids(&mut new_term, &blk_map);
            if bi == plan.latch {
                let target = if k + 1 < trip {
                    cl_labels[k][plan.entry_idx]
                } else {
                    plan.exit_target
                };
                new_term = Terminator::Branch(target);
            }
            new_blocks.push(BasicBlock {
                label: cl_labels[cj][i],
                instructions: new_insts,
                terminator: new_term,
                source_spans: Vec::new(),
            });
        }
    }

    // ── Mutate iteration-0 blocks in place ────────────────────────────────
    if plan.single_block {
        // The header block keeps its work instructions (minus the dropped
        // control chain) and branches to the next iteration (or the exit).
        let block = &mut func.blocks[header];
        block.instructions.retain(|inst| {
            inst.dest().map(|d| !plan.drop_dests.contains(&d.0)).unwrap_or(true)
        });
        for inst in &mut block.instructions {
            substitute_operands_in_inst(inst, &op_maps[0]);
        }
        let target = if trip > 1 {
            cl_labels[0][plan.entry_idx]
        } else {
            plan.exit_target
        };
        block.terminator = Terminator::Branch(target);
    } else {
        // Header becomes a passthrough to the first work block.
        let entry_label = func.blocks[clone_order[plan.entry_idx]].label;
        let header_block = &mut func.blocks[header];
        header_block.instructions.clear();
        header_block.terminator = Terminator::Branch(entry_label);
        for &bi in &clone_order {
            for inst in &mut func.blocks[bi].instructions {
                substitute_operands_in_inst(inst, &op_maps[0]);
            }
            substitute_operands_in_terminator(&mut func.blocks[bi].terminator, &op_maps[0]);
        }
        let target = if trip > 1 {
            cl_labels[0][plan.entry_idx]
        } else {
            plan.exit_target
        };
        func.blocks[plan.latch].terminator = Terminator::Branch(target);
    }

    // ── Exit-block phis: rewrite the header incoming to the final value ────
    let last_latch_label = if trip > 1 {
        cl_labels[trip - 2][clone_order.len() - 1]
    } else {
        func.blocks[plan.latch].label
    };
    if let Some(exit_bi) = func.blocks.iter().position(|b| b.label == plan.exit_target) {
        for inst in &mut func.blocks[exit_bi].instructions {
            if let Instruction::Phi { incoming, .. } = inst {
                for (op, lbl) in incoming.iter_mut() {
                    if *lbl == header_label {
                        let new_op = match op {
                            Operand::Const(c) => Operand::Const(*c),
                            Operand::Value(v) => exit_res
                                .get(&v.0)
                                .copied()
                                .or_else(|| op_maps[trip - 1].get(&v.0).copied())
                                .unwrap_or(Operand::Value(*v)),
                        };
                        *op = new_op;
                        *lbl = last_latch_label;
                    }
                }
            }
        }
    }

    // ── Outside uses of loop-defined values: substitute the final value ───
    // Escape analysis has proven every such use is dominated by the exit
    // edge, where the final iteration's value is available.
    if !plan.loop_defs.is_empty() {
        let final_map: FxHashMap<u32, Operand> = plan
            .loop_defs
            .iter()
            .map(|&v| {
                let op = exit_res
                    .get(&v)
                    .copied()
                    .or_else(|| op_maps[trip - 1].get(&v).copied())
                    .unwrap_or(Operand::Value(Value(v)));
                (v, op)
            })
            .collect();
        for (bi, block) in func.blocks.iter_mut().enumerate() {
            if plan.single_block && bi == header {
                continue;
            }
            if clone_order.contains(&bi) {
                continue; // iteration-0 blocks already substituted per-iteration
            }
            if bi == header {
                continue; // passthrough now
            }
            for inst in &mut block.instructions {
                substitute_operands_in_inst(inst, &final_map);
            }
            substitute_operands_in_terminator(&mut block.terminator, &final_map);
        }
    }

    // Splice the clones into place right after the last loop block so that
    // text order matches control-flow order (liveness intervals are computed
    // in text order; appending at the end produced use-before-def intervals).
    let insert_at = clone_order.iter().copied().chain(std::iter::once(header)).max().unwrap() + 1;
    func.blocks.splice(insert_at..insert_at, new_blocks);

    // Debug verification: no use of a dropped (loop-control) value may remain.
    if std::env::var("CCC_DEBUG_FULL_UNROLL").is_ok() {
        for (bi, block) in func.blocks.iter().enumerate() {
            for inst in &block.instructions {
                let mut bad = false;
                for_each_value_use_in_inst_simple(inst, &mut |vid| {
                    if plan.drop_dests.contains(&vid) {
                        bad = true;
                    }
                });
                if bad {
                    eprintln!(
                        "[FULL_UNROLL] DANGLING USE in func={} block={} inst={:?}",
                        func.name, bi, inst
                    );
                }
            }
        }
    }
    true
}
/// Substitute operand (and plain-Value) positions per an operand map.
/// Each position is rewritten exactly once (operand resolutions may reference
/// value ids that also have renames in the map — they must NOT be re-mapped).
fn substitute_operands_in_inst(inst: &mut Instruction, map: &FxHashMap<u32, Operand>) {
    // Operand positions: full substitution (const or value resolution).
    // Plain-Value positions (Load ptr, GEP base, ...): value renames only.
    let sub = |op: &mut Operand| {
        if let Operand::Value(v) = op {
            if let Some(new) = map.get(&v.0) {
                *op = *new;
            }
        }
    };
    let subv = |v: &mut Value| {
        if let Some(Operand::Value(nv)) = map.get(&v.0) {
            *v = *nv;
        }
    };
    match inst {
        Instruction::ParamRef { .. }
        | Instruction::Alloca { .. }
        | Instruction::GlobalAddr { .. }
        | Instruction::LabelAddr { .. }
        | Instruction::Fence { .. }
        | Instruction::StackSave { .. }
        | Instruction::GetReturnF64Second { .. }
        | Instruction::GetReturnF32Second { .. }
        | Instruction::GetReturnF128Second { .. } => {}

        Instruction::Store { val, ptr, .. } => {
            sub(val);
            subv(ptr);
        }
        Instruction::Load { ptr, .. } => subv(ptr),
        Instruction::Memcpy { dest, src, .. } => {
            subv(dest);
            subv(src);
        }
        Instruction::BinOp { lhs, rhs, .. } => {
            sub(lhs);
            sub(rhs);
        }
        Instruction::UnaryOp { src, .. } => sub(src),
        Instruction::Cmp { lhs, rhs, .. } => {
            sub(lhs);
            sub(rhs);
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            subv(base);
            sub(offset);
        }
        Instruction::DynAlloca { size, .. } => sub(size),
        Instruction::StackRestore { ptr } => subv(ptr),
        Instruction::Cast { src, .. } => sub(src),
        Instruction::Copy { src, .. } => sub(src),
        Instruction::Call { info, .. } => {
            for arg in &mut info.args {
                sub(arg);
            }
        }
        Instruction::CallIndirect { func_ptr, info } => {
            sub(func_ptr);
            for arg in &mut info.args {
                sub(arg);
            }
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming {
                sub(op);
            }
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            sub(cond);
            sub(true_val);
            sub(false_val);
        }
        Instruction::AtomicRmw { ptr, val, .. } => {
            sub(ptr);
            sub(val);
        }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => {
            sub(ptr);
            sub(expected);
            sub(desired);
        }
        Instruction::AtomicLoad { ptr, .. } => sub(ptr),
        Instruction::AtomicStore { ptr, val, .. } => {
            sub(ptr);
            sub(val);
        }
        Instruction::InlineAsm { inputs, outputs, .. } => {
            for (_, op, _) in inputs {
                sub(op);
            }
            for (_, v, _) in outputs {
                subv(v);
            }
        }
        Instruction::Intrinsic { args, dest_ptr, .. } => {
            for a in args {
                sub(a);
            }
            if let Some(dp) = dest_ptr {
                subv(dp);
            }
        }
        Instruction::VaArg { va_list_ptr, .. } => subv(va_list_ptr),
        Instruction::VaStart { va_list_ptr } => subv(va_list_ptr),
        Instruction::VaEnd { va_list_ptr } => subv(va_list_ptr),
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            subv(dest_ptr);
            subv(src_ptr);
        }
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, .. } => {
            subv(dest_ptr);
            subv(va_list_ptr);
        }
        Instruction::SetReturnF64Second { src }
        | Instruction::SetReturnF32Second { src }
        | Instruction::SetReturnF128Second { src } => sub(src),
    }
}

fn substitute_operands_in_terminator(term: &mut Terminator, map: &FxHashMap<u32, Operand>) {
    let sub = |op: &mut Operand| {
        if let Operand::Value(v) = op {
            if let Some(new) = map.get(&v.0) {
                *op = *new;
            }
        }
    };
    match term {
        Terminator::Return(Some(op)) => sub(op),
        Terminator::CondBranch { cond, .. } => sub(cond),
        Terminator::Switch { val, .. } => sub(val),
        Terminator::IndirectBranch { target, .. } => sub(target),
        _ => {}
    }
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

        // B2: body — GEP + store. The GEP uses a constant offset (not the IV)
        // so the loop stays eligible under the IV-widening guard, which only
        // rejects GEPs that index directly by the narrow IV.
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::GetElementPtr {
                    dest: Value(4),
                    base: Value(10), // arr (loop-invariant, defined outside)
                    offset: Operand::Const(IrConst::I32(0)),
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
    fn test_no_unroll_iv_indexed_gep() {
        // A body that GEPs directly by the narrow (I32) IV must NOT be unrolled
        // on 64-bit targets: the unroller's intermediate IV values stay narrow
        // and would corrupt 64-bit pointer arithmetic after copy propagation.
        let mut func = make_counting_loop(100);
        for inst in &mut func.blocks[2].instructions {
            if let Instruction::GetElementPtr { offset, .. } = inst {
                *offset = Operand::Value(Value(1)); // index by the IV
            }
        }
        let n = unroll_loops(&mut func);
        if crate::common::types::target_is_32bit() {
            assert_eq!(n, 1, "32-bit targets have no widening hazard");
        } else {
            assert_eq!(n, 0, "IV-indexed GEP loop should not be unrolled");
            assert_eq!(func.blocks.len(), 5, "block count should be unchanged");
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
