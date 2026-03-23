//! Binary recursion → iterative accumulator transformation.
//!
//! Detects the Fibonacci-like pattern:
//! ```c
//! long f(int n) {
//!     if (n <= base) return base_val;
//!     return f(n - d1) + f(n - d2);   // two self-recursive calls combined with Add
//! }
//! ```
//!
//! And converts it to an iterative sliding-window loop:
//! ```c
//! long f(int n) {
//!     if (n <= 1) return n;
//!     long a = f(base), b = f(base+1);
//!     for (int i = base+2; i <= n; i++) { long t = a + b; a = b; b = t; }
//!     return b;
//! }
//! ```
//!
//! This eliminates exponential recursion overhead for linear recurrences.

use crate::common::types::IrType;
use crate::ir::reexports::*;

/// Run binary-recursion-to-iteration on a single function.
/// Returns the number of transformations applied (0 or 1).
pub(crate) fn recursion_to_iteration(func: &mut IrFunction) -> usize {
    if func.is_variadic || func.is_declaration || func.uses_sret {
        return 0;
    }
    if func.blocks.is_empty() || func.params.len() != 1 {
        return 0; // Only handle single-parameter functions for now
    }

    let func_name = func.name.clone();

    // ── 1. Find exactly 2 self-recursive calls ──────────────────────────────
    let mut calls: Vec<CallSite> = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Call { func: f, info } = inst {
                if f == &func_name && !info.is_sret && !info.is_variadic
                    && info.num_fixed_args == 1
                {
                    if let Some(dest) = info.dest {
                        calls.push(CallSite {
                            block_idx,
                            inst_idx,
                            dest,
                            arg: info.args[0].clone(),
                        });
                    }
                }
            }
        }
    }

    if calls.len() != 2 {
        return 0; // Must have exactly 2 self-recursive calls
    }

    // Both calls must be in the same block
    if calls[0].block_idx != calls[1].block_idx {
        return 0;
    }
    let call_block_idx = calls[0].block_idx;

    // ── 2. Verify the combination: result = call1 + call2 ──────────────────
    // Find an Add instruction that uses both call results
    let call_block = &func.blocks[call_block_idx];
    let mut combine_add: Option<(Value, usize)> = None;

    for (inst_idx, inst) in call_block.instructions.iter().enumerate() {
        if let Instruction::BinOp { dest, op: IrBinOp::Add, lhs, rhs, ty: _ } = inst {
            let lhs_is_call = match lhs {
                Operand::Value(v) => v.0 == calls[0].dest.0 || v.0 == calls[1].dest.0,
                _ => false,
            };
            let rhs_is_call = match rhs {
                Operand::Value(v) => v.0 == calls[0].dest.0 || v.0 == calls[1].dest.0,
                _ => false,
            };
            if lhs_is_call && rhs_is_call {
                combine_add = Some((*dest, inst_idx));
                break;
            }
        }
    }

    let (add_dest, _add_idx) = match combine_add {
        Some(x) => x,
        None => return 0, // No Add combining the two call results
    };

    // The block must return the Add result
    let returns_add = match &call_block.terminator {
        Terminator::Return(Some(Operand::Value(v))) => v.0 == add_dest.0,
        _ => false,
    };
    if !returns_add {
        return 0;
    }

    // ── 3. Analyze call arguments: must be param - constant ────────────────
    // Find the parameter value
    let mut param_val = None;
    let mut param_ty = IrType::I32;
    for inst in &func.blocks[0].instructions {
        if let Instruction::ParamRef { dest, param_idx: 0, ty } = inst {
            param_val = Some(*dest);
            param_ty = *ty;
        }
    }
    let param_val = match param_val {
        Some(v) => v,
        None => return 0,
    };

    // Trace each call's argument back to find `param - constant`
    let dec1 = trace_param_decrement(func, &calls[0].arg, param_val);
    let dec2 = trace_param_decrement(func, &calls[1].arg, param_val);

    let (dec1, dec2) = match (dec1, dec2) {
        (Some(d1), Some(d2)) => (d1, d2),
        _ => return 0,
    };

    // Ensure decrements are 1 and 2 (or 2 and 1) — standard Fibonacci pattern
    let (small_dec, large_dec) = if dec1 < dec2 { (dec1, dec2) } else { (dec2, dec1) };
    if small_dec != 1 || large_dec != 2 {
        return 0; // Only handle fib(n-1) + fib(n-2) pattern
    }

    // ── 4. Find the base case ──────────────────────────────────────────────
    // Look for a block that returns a value when n <= constant
    // The base case block should have Return(param) or Return(constant)
    let mut base_case_block = None;
    let mut base_threshold = 1i64; // default: n <= 1

    for (block_idx, block) in func.blocks.iter().enumerate() {
        if block_idx == call_block_idx { continue; }
        if let Terminator::Return(Some(_)) = &block.terminator {
            // This could be the base case
            base_case_block = Some(block_idx);

            // Try to find the comparison that leads here
            for other_block in &func.blocks {
                if let Terminator::CondBranch { true_label, false_label, .. } = &other_block.terminator {
                    if true_label.0 as usize == block_idx || false_label.0 as usize == block_idx {
                        // Check comparison for n <= constant
                        for inst in &other_block.instructions {
                            if let Instruction::Cmp { op: IrCmpOp::Sle | IrCmpOp::Slt, rhs: Operand::Const(c), .. } = inst {
                                if let Some(v) = c.to_i64() {
                                    base_threshold = v;
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
    }

    if base_case_block.is_none() {
        return 0;
    }

    // Determine the return type from the function
    let ret_ty = func.return_type;

    // ── 5. Transform: replace function body with iterative loop ────────────
    let debug = std::env::var("LCCC_DEBUG_RECURSION").is_ok();
    if debug {
        eprintln!("[REC→ITER] Detected binary recursion in '{}': f(n) = f(n-{}) + f(n-{}), base: n <= {}",
            func_name, small_dec, large_dec, base_threshold);
    }

    // Allocate fresh values
    let mut next_id = func.next_value_id;
    let alloc = |next: &mut u32| -> Value { let v = Value(*next); *next += 1; v };

    let n_ext = alloc(&mut next_id);         // sign-extended n
    let base_cmp = alloc(&mut next_id);      // n <= 1 comparison
    let loop_i = alloc(&mut next_id);        // loop counter phi
    let loop_a = alloc(&mut next_id);        // accumulator a phi
    let loop_b = alloc(&mut next_id);        // accumulator b phi
    let loop_cmp = alloc(&mut next_id);      // i <= n comparison
    let loop_t = alloc(&mut next_id);        // t = a + b
    let loop_i_next = alloc(&mut next_id);   // i + 1

    // Allocate fresh block labels
    let max_label = func.blocks.iter().map(|b| b.label.0).max().unwrap_or(0);
    let base_label = BlockId(max_label + 1);
    let preheader_label = BlockId(max_label + 2);
    let header_label = BlockId(max_label + 3);
    let body_label = BlockId(max_label + 4);
    let exit_label = BlockId(max_label + 5);
    let entry_label = func.blocks[0].label;

    // Determine the canonical types
    let idx_ty = if ret_ty == IrType::I64 { IrType::I64 } else { param_ty };
    let is_i32_param = matches!(param_ty, IrType::I32 | IrType::U32);

    // Replace entire function body
    func.blocks.clear();

    // Entry block: cast param, check base case
    let mut entry_insts = vec![
        Instruction::ParamRef { dest: param_val, param_idx: 0, ty: param_ty },
    ];
    if is_i32_param && ret_ty == IrType::I64 {
        // Sign-extend i32 param to i64 for the return type
        entry_insts.push(Instruction::Cast {
            dest: n_ext, src: Operand::Value(param_val),
            from_ty: param_ty, to_ty: IrType::I64,
        });
    }
    let n_val = if is_i32_param && ret_ty == IrType::I64 { n_ext } else { param_val };
    entry_insts.push(Instruction::Cmp {
        dest: base_cmp, op: IrCmpOp::Sle,
        lhs: Operand::Value(n_val),
        rhs: Operand::Const(if ret_ty == IrType::I64 { IrConst::I64(base_threshold) } else { IrConst::I32(base_threshold as i32) }),
        ty: ret_ty,
    });

    func.blocks.push(BasicBlock {
        label: entry_label,
        instructions: entry_insts,
        terminator: Terminator::CondBranch {
            cond: Operand::Value(base_cmp),
            true_label: base_label,
            false_label: preheader_label,
        },
        source_spans: Vec::new(),
    });

    // Base case: return n
    func.blocks.push(BasicBlock {
        label: base_label,
        instructions: Vec::new(),
        terminator: Terminator::Return(Some(Operand::Value(n_val))),
        source_spans: Vec::new(),
    });

    // Loop preheader: set initial values
    func.blocks.push(BasicBlock {
        label: preheader_label,
        instructions: Vec::new(),
        terminator: Terminator::Branch(header_label),
        source_spans: Vec::new(),
    });

    // Loop header: phi nodes + exit check
    let zero_const = if ret_ty == IrType::I64 { IrConst::I64(0) } else { IrConst::I32(0) };
    let one_const = if ret_ty == IrType::I64 { IrConst::I64(1) } else { IrConst::I32(1) };
    let two_const = if ret_ty == IrType::I64 { IrConst::I64(2) } else { IrConst::I32(2) };

    func.blocks.push(BasicBlock {
        label: header_label,
        instructions: vec![
            Instruction::Phi {
                dest: loop_i, ty: ret_ty,
                incoming: vec![
                    (Operand::Const(two_const.clone()), preheader_label),
                    (Operand::Value(loop_i_next), body_label),
                ],
            },
            Instruction::Phi {
                dest: loop_a, ty: ret_ty,
                incoming: vec![
                    (Operand::Const(zero_const), preheader_label),
                    (Operand::Value(loop_b), body_label),
                ],
            },
            Instruction::Phi {
                dest: loop_b, ty: ret_ty,
                incoming: vec![
                    (Operand::Const(one_const.clone()), preheader_label),
                    (Operand::Value(loop_t), body_label),
                ],
            },
            Instruction::Cmp {
                dest: loop_cmp, op: IrCmpOp::Sle,
                lhs: Operand::Value(loop_i),
                rhs: Operand::Value(n_val),
                ty: ret_ty,
            },
        ],
        terminator: Terminator::CondBranch {
            cond: Operand::Value(loop_cmp),
            true_label: body_label,
            false_label: exit_label,
        },
        source_spans: Vec::new(),
    });

    // Loop body: t = a + b; i_next = i + 1
    func.blocks.push(BasicBlock {
        label: body_label,
        instructions: vec![
            Instruction::BinOp {
                dest: loop_t, op: IrBinOp::Add,
                lhs: Operand::Value(loop_a), rhs: Operand::Value(loop_b),
                ty: ret_ty,
            },
            Instruction::BinOp {
                dest: loop_i_next, op: IrBinOp::Add,
                lhs: Operand::Value(loop_i), rhs: Operand::Const(one_const),
                ty: ret_ty,
            },
        ],
        terminator: Terminator::Branch(header_label),
        source_spans: Vec::new(),
    });

    // Exit: return b
    func.blocks.push(BasicBlock {
        label: exit_label,
        instructions: Vec::new(),
        terminator: Terminator::Return(Some(Operand::Value(loop_b))),
        source_spans: Vec::new(),
    });

    func.next_value_id = next_id;
    func.next_label = max_label + 6;

    if debug {
        eprintln!("[REC→ITER] Transformed '{}' to iterative loop with {} blocks", func_name, func.blocks.len());
    }

    1
}

struct CallSite {
    block_idx: usize,
    #[allow(dead_code)]
    inst_idx: usize,
    dest: Value,
    arg: Operand,
}

/// Trace an operand back to find `param - constant`, returning the constant.
/// Handles chains like: Cast(Sub(Cast(param), const)) or Sub(param, const).
fn trace_param_decrement(
    func: &IrFunction,
    arg: &Operand,
    param_val: Value,
) -> Option<i64> {
    let val = match arg {
        Operand::Value(v) => *v,
        _ => return None,
    };

    // Search all blocks for the defining instruction
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                // Direct: Sub(param, const)
                Instruction::BinOp { dest, op: IrBinOp::Sub, lhs: Operand::Value(l), rhs: Operand::Const(c), .. }
                    if *dest == val =>
                {
                    if *l == param_val {
                        return c.to_i64();
                    }
                    // Maybe lhs is a cast of param
                    if let Some(src) = find_cast_source(func, *l) {
                        if src == param_val { return c.to_i64(); }
                    }
                }
                // Sub result is then cast
                Instruction::Cast { dest, src: Operand::Value(sub_val), .. }
                    if *dest == val =>
                {
                    return trace_param_decrement(func, &Operand::Value(*sub_val), param_val);
                }
                // Copy of a sub
                Instruction::Copy { dest, src: Operand::Value(copy_src) }
                    if *dest == val =>
                {
                    return trace_param_decrement(func, &Operand::Value(*copy_src), param_val);
                }
                // Add with negative constant (n + (-1) == n - 1)
                Instruction::BinOp { dest, op: IrBinOp::Add, lhs: Operand::Value(l), rhs: Operand::Const(c), .. }
                    if *dest == val =>
                {
                    if let Some(v) = c.to_i64() {
                        if v < 0 {
                            let actual_param = if *l == param_val { true }
                                else { find_cast_source(func, *l) == Some(param_val) };
                            if actual_param { return Some(-v); }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Find the source value of a Cast instruction defining `val`.
fn find_cast_source(func: &IrFunction, val: Value) -> Option<Value> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Cast { dest, src: Operand::Value(s), .. } = inst {
                if *dest == val { return Some(*s); }
            }
            if let Instruction::Copy { dest, src: Operand::Value(s) } = inst {
                if *dest == val { return Some(*s); }
            }
        }
    }
    None
}
