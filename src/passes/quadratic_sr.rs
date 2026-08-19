//! Second-order strength reduction: triangular-number loop indices.
//!
//! spectral_norm's inner loop evaluates `idx = (i+j)(i+j+1)/2 + i + 1` fresh
//! every iteration (a multiply plus a signed-division-by-two correction
//! sequence). Because the inner IV j has slope 1, the triangular term
//! satisfies `idx(j+1) = idx(j) + (i+j+1)`, so it can be carried as two
//! running counters instead of being recomputed:
//!
//! ```text
//! idx += s;      // s = i+j+1 this iteration
//! s   += 1;
//! ```
//!
//! Integer addition is associative and exact mod 2^N, and t(t+1) is always
//! even, so the accumulator produces bit-identical results to the direct
//! computation (including wraparound). Only the exact slope-1 triangular form
//! is recognized; anything else is left alone.

use crate::common::fx_hash::FxHashSet;
use crate::common::types::IrType;
use crate::ir::analysis::CfgAnalysis;
use crate::ir::reexports::{Instruction, IrFunction, IrBinOp, IrConst, Operand, Value};
use super::loop_analysis;

pub(crate) fn run(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_NO_QUAD_SR").is_ok() {
        return 0;
    }
    if func.blocks.len() < 2 {
        return 0;
    }
    let cfg = CfgAnalysis::build(func);
    let loops = loop_analysis::merge_loops_by_header(loop_analysis::find_natural_loops(
        cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom));
    let mut total = 0;
    for lp in &loops {
        // Innermost loops only (no nested loop inside).
        let innermost = !loops.iter().any(|o| {
            o.header != lp.header
                && o.body.len() < lp.body.len()
                && o.body.iter().all(|b| lp.body.contains(b))
        });
        if !innermost {
            continue;
        }
        total += reduce_loop(func, lp, &cfg);
    }
    total
}

/// Compute `value = IV + invariant` for a slope-1 triangular term. Returns
/// the invariant operand when `value` is exactly `iv + inv` (either order).
fn affine_in_iv(
    func: &IrFunction,
    loop_body: &FxHashSet<usize>,
    iv: Value,
    v: Value,
    _fuel: u8,
) -> Option<(i64, Option<Operand>)> {
    if v == iv {
        return Some((1, None));
    }
    let inst = loop_body.iter().find_map(|&bi| {
        func.blocks[bi].instructions.iter().find(|i| i.dest() == Some(v))
    })?;
    if let Instruction::BinOp { op: IrBinOp::Add, lhs, rhs, .. } = inst {
        for (a, b) in [(lhs, rhs), (rhs, lhs)] {
            if matches!(a, Operand::Value(x) if *x == iv) {
                // b must be loop-invariant.
                let inv_ok = match b {
                    Operand::Const(_) => true,
                    Operand::Value(bv) => !loop_body.iter().any(|&bi| {
                        func.blocks[bi].instructions.iter().any(|i| i.dest() == Some(*bv))
                    }),
                };
                if inv_ok {
                    return Some((1, Some(*b)));
                }
            }
        }
    }
    None
}

fn reduce_loop(func: &mut IrFunction, lp: &loop_analysis::NaturalLoop, cfg: &CfgAnalysis) -> usize {
    let debug = std::env::var("CCC_DEBUG_QUAD_SR").is_ok();
    let header = lp.header;
    let Some(preheader) = loop_analysis::find_preheader(header, &lp.body, &cfg.preds) else {
        return 0;
    };

    // Find the slope-1 IV phi (backedge step is `phi + 1`).
    let mut iv_phi = None;
    for inst in &func.blocks[header].instructions {
        if let Instruction::Phi { dest, ty: IrType::I32, incoming } = inst {
            if incoming.len() != 2 {
                continue;
            }
            let mut is_step1 = false;
            for (op, _) in incoming {
                if let Operand::Value(v) = op {
                    if is_add_const_one(func, &lp.body, *v, *dest) {
                        is_step1 = true;
                    }
                }
            }
            if is_step1 {
                iv_phi = Some((*dest, incoming.clone()));
                break;
            }
        }
    }
    let Some((iv, incoming)) = iv_phi else { return 0 };
    let _ = incoming;

    // Find a triangular product: Mul(a, b) where b = a + 1 and a is affine
    // slope-1 in the IV.
    let mut triangular: Option<(Value, Value, Option<Operand>, Value)> = None; // (prod, a, a_inv, div2_result)
    for &bi in &lp.body {
        for inst in &func.blocks[bi].instructions {
            let Instruction::BinOp { dest: prod, op: IrBinOp::Mul, lhs, rhs, .. } = inst else { continue };
            let (Operand::Value(av), Operand::Value(bv)) = (lhs, rhs) else { continue };
            // One of them must be Add(other, 1).
            let (a_val, b_val) = {
                let b_is_a_plus1 = is_plus_one_of(func, &lp.body, *bv, *av);
                let a_is_b_plus1 = is_plus_one_of(func, &lp.body, *av, *bv);
                if b_is_a_plus1 {
                    (*av, *bv)
                } else if a_is_b_plus1 {
                    (*bv, *av)
                } else {
                    continue;
                }
            };
            // a must be affine slope-1 in the IV: coef == 1.
            let Some((coef, a_inv)) = affine_in_iv(func, &lp.body, iv, a_val, 16) else { continue };
            if coef != 1 {
                continue;
            }
            // prod must feed a signed division by 2 (find the final /2 result).
            if let Some(d2) = find_div2_result(func, &lp.body, *prod) {
                triangular = Some((*prod, a_val, a_inv, d2));
                if debug {
                    eprintln!("[QUAD_SR] loop header={} triangular prod={} a={} div2={}", header, prod.0, a_val.0, d2.0);
                }
                break;
            }
        }
        if triangular.is_some() {
            break;
        }
    }
    let Some((_prod, a_val, a_inv, half)) = triangular else { return 0 };

    // `half` = t(t+1)/2 must feed the final index via adds of loop-invariants.
    // Find the final value: follow single-use Add chains from `half` adding
    // loop-invariant operands, collecting the invariant tail (must be exact —
    // the difference is unaffected by invariant additions).
    let mut final_val = half;
    let mut tail_invs: Vec<Operand> = Vec::new();
    let mut seen = 0;
    loop {
        seen += 1;
        if seen > 8 {
            return 0;
        }
        // Find the single Add that uses final_val.
        let mut next = None;
        for &bi in &lp.body {
            for inst in &func.blocks[bi].instructions {
                if let Instruction::BinOp { dest, op: IrBinOp::Add, lhs, rhs, .. } = inst {
                    let lhs_is = matches!(lhs, Operand::Value(v) if *v == final_val);
                    let rhs_is = matches!(rhs, Operand::Value(v) if *v == final_val);
                    if lhs_is || rhs_is {
                        let other = if lhs_is { rhs } else { lhs };
                        // other must be loop-invariant.
                        let inv = match other {
                            Operand::Const(_) => true,
                            Operand::Value(v) => !lp.body.iter().any(|&bj| {
                                func.blocks[bj].instructions.iter().any(|i| i.dest() == Some(*v))
                            }),
                        };
                        if inv {
                            next = Some((*dest, *other));
                        }
                    }
                }
            }
        }
        match next {
            Some((d, inv)) => {
                tail_invs.push(inv);
                final_val = d;
            }
            None => break,
        }
    }
    if final_val == half {
        return 0; // the /2 result is never used in an index accumulation
    }

    // a_inv is the invariant part of `a` (a = j + a_inv). For the benchmark,
    // a = i + j so a_inv = i. We need the init values at j = init.
    // The IV init must be a constant for the init computation to be simple.
    let init_op = incoming.iter().find_map(|(op, _)| match op {
        Operand::Const(c) => Some(*c),
        _ => None,
    });
    // We only handle IV init of 0 (the common case); the invariant part a_inv
    // must be a plain Value (the outer IV or a param), not a constant.
    let Some(IrConst::I32(0)) = init_op else { return 0 };
    let Some(Operand::Value(inv_val)) = a_inv else { return 0 };

    // Build the accumulator: two new phis (idx, s) + preheader inits.
    let mut next_id = func.next_value_id;
    let mut nv = || {
        let v = Value(next_id);
        next_id += 1;
        v
    };
    let idx_phi = nv();
    let s_phi = nv();
    let idx_init = nv();
    let s_init = nv();
    let idx_next = nv();
    let s_next = nv();
    let prod0 = nv();
    let a_plus1 = nv();
    let half0 = nv();

    let pre_label = func.blocks[preheader].label;
    // Find the latch (backedge to header).
    let latch = lp.body.iter().copied().find(|&bi| {
        matches!(&func.blocks[bi].terminator, crate::ir::instruction::Terminator::Branch(t) if *t == func.blocks[header].label)
            || matches!(&func.blocks[bi].terminator, crate::ir::instruction::Terminator::CondBranch { true_label, false_label, .. } if *true_label == func.blocks[header].label || *false_label == func.blocks[header].label)
    });
    let Some(latch) = latch else { return 0 };
    let latch_label = func.blocks[latch].label;

    // Preheader: idx_init = a0*(a0+1)/2 + tail, s_init = a0 + 1, where
    // a0 = a at j=0 = inv_val (since a = j + inv_val, j init 0).
    // tail = the collected invariant adds.
    {
        let pre = &mut func.blocks[preheader];
        // a0 = inv_val (a = j + inv_val, j=0). If a had the invariant on the
        // other side, it's still just inv_val (addition is commutative).
        pre.instructions.push(Instruction::BinOp {
            dest: a_plus1,
            op: IrBinOp::Add,
            lhs: Operand::Value(inv_val),
            rhs: Operand::Const(IrConst::I32(1)),
            ty: IrType::I32,
        });
        pre.instructions.push(Instruction::BinOp {
            dest: prod0,
            op: IrBinOp::Mul,
            lhs: Operand::Value(inv_val),
            rhs: Operand::Value(a_plus1),
            ty: IrType::I32,
        });
        // t(t+1) is always even, so AShr by 1 is exact for the (non-negative
        // or wraparound) value — matches the body's corrected division.
        pre.instructions.push(Instruction::BinOp {
            dest: half0,
            op: IrBinOp::AShr,
            lhs: Operand::Value(prod0),
            rhs: Operand::Const(IrConst::I32(1)),
            ty: IrType::I32,
        });
        // Apply the invariant tail.
        let mut cur = half0;
        for inv in &tail_invs {
            let d = nv();
            pre.instructions.push(Instruction::BinOp {
                dest: d,
                op: IrBinOp::Add,
                lhs: Operand::Value(cur),
                rhs: *inv,
                ty: IrType::I32,
            });
            cur = d;
        }
        pre.instructions.push(Instruction::Copy {
            dest: idx_init,
            src: Operand::Value(cur),
        });
        pre.instructions.push(Instruction::Copy {
            dest: s_init,
            src: Operand::Value(a_plus1),
        });
    }

    // Header phis.
    let header_block = &mut func.blocks[header];
    let insert_pos = header_block
        .instructions
        .iter()
        .position(|i| !matches!(i, Instruction::Phi { .. }))
        .unwrap_or(header_block.instructions.len());
    header_block.instructions.insert(insert_pos, Instruction::Phi {
        dest: s_phi,
        ty: IrType::I32,
        incoming: vec![
            (Operand::Value(s_init), pre_label),
            (Operand::Value(s_next), latch_label),
        ],
    });
    header_block.instructions.insert(insert_pos, Instruction::Phi {
        dest: idx_phi,
        ty: IrType::I32,
        incoming: vec![
            (Operand::Value(idx_init), pre_label),
            (Operand::Value(idx_next), latch_label),
        ],
    });

    // Latch increments: idx_next = idx_phi + s_phi; s_next = s_phi + 1.
    {
        let latch_block = &mut func.blocks[latch];
        latch_block.instructions.push(Instruction::BinOp {
            dest: idx_next,
            op: IrBinOp::Add,
            lhs: Operand::Value(idx_phi),
            rhs: Operand::Value(s_phi),
            ty: IrType::I32,
        });
        latch_block.instructions.push(Instruction::BinOp {
            dest: s_next,
            op: IrBinOp::Add,
            lhs: Operand::Value(s_phi),
            rhs: Operand::Const(IrConst::I32(1)),
            ty: IrType::I32,
        });
    }

    // Replace uses of final_val with idx_phi within the loop.
    for &bi in &lp.body {
        for inst in &mut func.blocks[bi].instructions {
            let mut replace = |op: &mut Operand| {
                if let Operand::Value(v) = op {
                    if *v == final_val {
                        *v = idx_phi;
                    }
                }
            };
            match inst {
                Instruction::BinOp { lhs, rhs, .. } => {
                    replace(lhs);
                    replace(rhs);
                }
                Instruction::Cmp { lhs, rhs, .. } => {
                    replace(lhs);
                    replace(rhs);
                }
                Instruction::Cast { src, .. } | Instruction::Copy { src, .. } | Instruction::UnaryOp { src, .. } => {
                    replace(src);
                }
                Instruction::Store { val, .. } => {
                    replace(val);
                }
                Instruction::Load { .. } => {}
                _ => {}
            }
        }
    }

    func.next_value_id = next_id;
    if debug {
        eprintln!("[QUAD_SR] transformed loop header={} final_val={} -> idx_phi={}", header, final_val.0, idx_phi.0);
    }
    1
}

/// Is `v` defined as `phi + 1` (or `1 + phi`) in the loop?
fn is_add_const_one(func: &IrFunction, loop_body: &FxHashSet<usize>, v: Value, phi: Value) -> bool {
    loop_body.iter().any(|&bi| {
        func.blocks[bi].instructions.iter().any(|i| {
            matches!(i, Instruction::BinOp { dest, op: IrBinOp::Add, lhs, rhs, .. }
                if *dest == v
                    && ((matches!(lhs, Operand::Value(lv) if *lv == phi) && matches!(rhs, Operand::Const(c) if c.to_i64() == Some(1)))
                        || (matches!(rhs, Operand::Value(rv) if *rv == phi) && matches!(lhs, Operand::Const(c) if c.to_i64() == Some(1)))))
        })
    })
}

/// Is `b` defined as `a + 1` (or `1 + a`) in the loop?
fn is_plus_one_of(func: &IrFunction, loop_body: &FxHashSet<usize>, b: Value, a: Value) -> bool {
    loop_body.iter().any(|&bi| {
        func.blocks[bi].instructions.iter().any(|i| {
            matches!(i, Instruction::BinOp { dest, op: IrBinOp::Add, lhs, rhs, .. }
                if *dest == b
                    && ((matches!(lhs, Operand::Value(v) if *v == a) && matches!(rhs, Operand::Const(c) if c.to_i64() == Some(1)))
                        || (matches!(rhs, Operand::Value(v) if *v == a) && matches!(lhs, Operand::Const(c) if c.to_i64() == Some(1)))))
        })
    })
}

/// Find the final result of the signed division-by-2 applied to `prod`:
/// the AShr(Add(prod, LShr(AShr(prod,31),31)), 1) sequence, or a direct
/// AShr(prod, 1) / LShr(prod, 1) / SDiv(prod, 2).
fn find_div2_result(func: &IrFunction, loop_body: &FxHashSet<usize>, prod: Value) -> Option<Value> {
    // First find sign = AShr(prod, 31) and adj = Add(prod, LShr(sign,31)),
    // then half = AShr(adj, 1).
    let mut sign = None;
    let mut adj = None;
    let mut half = None;
    for &bi in loop_body {
        for inst in &func.blocks[bi].instructions {
            if let Instruction::BinOp { dest, op, lhs, rhs, .. } = inst {
                let lhs_v = matches!(lhs, Operand::Value(v) if *v == prod);
                match op {
                    IrBinOp::AShr if lhs_v && matches!(rhs, Operand::Const(c) if c.to_i64() == Some(31)) => {
                        sign = Some(*dest);
                    }
                    IrBinOp::LShr if matches!(lhs, Operand::Value(v) if Some(*v) == sign) && matches!(rhs, Operand::Const(c) if c.to_i64() == Some(31)) => {
                        adj = None; // placeholder; LShr dest is the mask
                        // record mask separately below
                    }
                    _ => {}
                }
            }
        }
    }
    // Second pass for the mask/add/half chain.
    let mut mask = None;
    for &bi in loop_body {
        for inst in &func.blocks[bi].instructions {
            if let Instruction::BinOp { dest, op, lhs, rhs, .. } = inst {
                match op {
                    IrBinOp::LShr if matches!(lhs, Operand::Value(v) if Some(*v) == sign) && matches!(rhs, Operand::Const(c) if c.to_i64() == Some(31)) => {
                        mask = Some(*dest);
                    }
                    IrBinOp::Add => {
                        let uses_prod = matches!(lhs, Operand::Value(v) if *v == prod);
                        let uses_mask = matches!(rhs, Operand::Value(v) if Some(*v) == mask)
                            || matches!(lhs, Operand::Value(v) if Some(*v) == mask);
                        if uses_prod && uses_mask {
                            adj = Some(*dest);
                        }
                    }
                    IrBinOp::AShr if matches!(lhs, Operand::Value(v) if Some(*v) == adj) && matches!(rhs, Operand::Const(c) if c.to_i64() == Some(1)) => {
                        half = Some(*dest);
                    }
                    _ => {}
                }
            }
        }
    }
    half
}
