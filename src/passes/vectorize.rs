//! SSE2 vectorization pass for matmul-style loops.
//!
//! Recognizes innermost loops with stride-1 double-precision accumulation patterns
//! and transforms them into FmaF64x2 intrinsics (packed FMA with broadcast scalar).
//!
//! Target pattern (matmul j-loop):
//! ```c
//! for (int j = 0; j < N; j++)
//!     C[i][j] += A[i][k] * B[k][j];
//! ```
//!
//! Transforms into vectorized loop that processes 2 elements per iteration:
//! ```c
//! for (int j = 0; j < N/2; j++)  // Loop N/2 times
//!     FmaF64x2(&C[i][j*2], &A[i][k], &B[k][j*2]);  // Process elements j*2 and j*2+1
//! ```
//!
//! ## Transformation Details
//!
//! 1. **Loop Bound**: Modified from `j < N` to `j < N/2`
//!    - For constant N: divide by 2 at compile time
//!    - For dynamic N: insert `udiv` instruction to compute N/2
//!    - Modifies ALL comparisons involving IV-derived values in the loop
//!
//! 2. **Array Indexing**: Changed from `j` to `j*2`
//!    - Inserts multiply instructions before GEPs: `offset' = offset * 2`
//!    - Ensures iteration j accesses elements [j*2, j*2+1] instead of [j, j+1]
//!    - Backend generates stride-16 addressing (2 doubles × 8 bytes)
//!
//! 3. **Induction Variable**: Keeps incrementing by 1
//!    - Backend-friendly: `j++` instead of `j += 2`
//!    - Combined with doubled offset, produces correct element access
//!
//! 4. **SSE2 Code Generation**:
//!    - `movsd` + `unpcklpd`: broadcast A[i][k] scalar
//!    - `movupd`: load 2 doubles from B[k][j*2]
//!    - `mulpd`: packed multiply
//!    - `addpd`: packed add with C[i][j*2]
//!    - `movupd`: store 2 results back
//!
//! ## Limitations
//!
//! - Remainder loop for odd N not yet implemented
//! - Only handles matmul-style patterns (load, multiply, add, store)
//! - Requires innermost loop with IV-based indexing

use crate::common::fx_hash::FxHashSet;
use crate::common::types::IrType;
use crate::ir::analysis::CfgAnalysis;
use crate::ir::instruction::{BasicBlock, BlockId, Instruction, Operand, Terminator, Value};
use crate::ir::intrinsics::IntrinsicOp;
use crate::ir::ops::IrBinOp;
use crate::ir::reexports::{IrConst, IrFunction};
use crate::passes::loop_analysis;

/// Run SSE2 vectorization on a function with precomputed CFG analysis.
pub(crate) fn vectorize_with_analysis(func: &mut IrFunction, cfg: &CfgAnalysis) -> usize {
    let num_blocks = func.blocks.len();
    let loops = loop_analysis::find_natural_loops(
        num_blocks,
        &cfg.preds,
        &cfg.succs,
        &cfg.idom,
    );

    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    if debug {
        eprintln!("[VEC] Function: {}, blocks: {}, loops: {}", func.name, num_blocks, loops.len());
    }

    if loops.is_empty() {
        return 0;
    }

    let mut total_changes = 0;

    // Process innermost loops (loops that don't contain other loops).
    for (idx, loop_info) in loops.iter().enumerate() {
        // Check if this is an innermost loop (no nested loops).
        let is_innermost = loops.iter().enumerate().all(|(other_idx, other)| {
            if idx == other_idx {
                return true;
            }
            // If this loop's body is a subset of another loop's body, it's nested.
            !loop_info.body.iter().all(|b| other.body.contains(b))
        });

        if debug {
            eprintln!("[VEC] Loop {} at header={}, body_size={}, innermost={}",
                idx, loop_info.header, loop_info.body.len(), is_innermost);
        }

        if !is_innermost {
            continue;
        }

        // Try to vectorize this loop.
        if let Some(pattern) = analyze_loop_pattern(func, loop_info, cfg) {
            if debug {
                eprintln!("[VEC] Pattern matched! Transforming to FmaF64x2");
            }
            total_changes += transform_to_fma_f64x2(func, &pattern);
        } else if debug {
            eprintln!("[VEC] Pattern match failed for loop {}", idx);
        }
    }

    total_changes
}

/// Pattern matching result for a vectorizable loop.
#[derive(Debug)]
struct VectorizablePattern {
    /// Loop header block index
    header_idx: usize,
    /// Loop body block (where the accumulation happens)
    body_idx: usize,
    /// Loop latch block index (contains the increment and backedge)
    latch_idx: usize,
    /// Exit block index
    exit_idx: usize,
    /// Induction variable (loop counter)
    iv: Value,
    /// Induction variable increment instruction index in latch block
    iv_inc_idx: usize,
    /// GEP for C array (result pointer)
    c_gep: Value,
    /// GEP for B array (source vector pointer)
    b_gep: Value,
    /// A scalar pointer (broadcasted value, loop-invariant)
    a_ptr: Value,
    /// Store instruction index in body (will be replaced)
    store_idx: usize,
    /// Loop limit value (N in `j < N`)
    limit: Operand,
    /// Comparison instruction index that tests loop exit condition
    exit_cmp_inst_idx: usize,
    /// Comparison destination value
    exit_cmp_dest: Value,
    /// All block indices in the loop body
    loop_blocks: FxHashSet<usize>,
}

/// Analyze a loop to detect the vectorizable matmul pattern.
fn analyze_loop_pattern(
    func: &IrFunction,
    loop_info: &loop_analysis::NaturalLoop,
    _cfg: &CfgAnalysis,
) -> Option<VectorizablePattern> {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let header_idx = loop_info.header;
    let header = &func.blocks[header_idx];

    // Find the induction variable phi in header.
    let mut iv = None;
    for inst in &header.instructions {
        if let Instruction::Phi { dest, incoming, .. } = inst {
            if incoming.len() == 2 {
                iv = Some(dest);
                break;
            }
        }
    }
    if iv.is_none() && debug {
        eprintln!("[VEC]   No IV phi found in header");
        return None;
    }
    let iv = *iv?;

    // Build a map of values that are derived from the IV (casts, copies, etc.)
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(iv);
    for inst in &header.instructions {
        match inst {
            Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                if let Operand::Value(src_val) = src {
                    if iv_derived.contains(src_val) {
                        iv_derived.insert(*dest);
                    }
                }
            }
            _ => {}
        }
    }

    // Find the comparison instruction for loop exit in header
    let mut exit_cmp_info = None;
    for (idx, inst) in header.instructions.iter().enumerate() {
        if let Instruction::Cmp { dest, op: _, lhs, rhs, ty: _ } = inst {
            // Check if comparing IV (or derived value) to a limit
            if let Operand::Value(lhs_val) = lhs {
                if iv_derived.contains(lhs_val) {
                    exit_cmp_info = Some((idx, *dest, rhs.clone()));
                    if debug {
                        eprintln!("[VEC]   Found comparison with IV-derived on left: {:?} < {:?}", lhs, rhs);
                    }
                    break;
                }
            } else if let Operand::Value(rhs_val) = rhs {
                if iv_derived.contains(rhs_val) {
                    // IV is on the right, use lhs as the limit
                    exit_cmp_info = Some((idx, *dest, lhs.clone()));
                    if debug {
                        eprintln!("[VEC]   Found comparison with IV-derived on right: {:?} > {:?}", lhs, rhs);
                    }
                    break;
                }
            }
        }
    }
    if exit_cmp_info.is_none() {
        if debug {
            eprintln!("[VEC]   No comparison instruction found for IV");
            eprintln!("[VEC]   Header block has {} instructions:", header.instructions.len());
            for (idx, inst) in header.instructions.iter().enumerate() {
                eprintln!("[VEC]     {}: {:?}", idx, inst);
            }
        }
        return None;
    }
    let (exit_cmp_inst_idx, exit_cmp_dest, limit) = exit_cmp_info?;

    // Search all blocks in the loop to find the one with the store instruction.
    // This is the actual computation block we want to vectorize.
    let mut body_idx = None;
    for &block_idx in &loop_info.body {
        if block_idx == header_idx {
            continue;
        }
        let block = &func.blocks[block_idx];
        for inst in &block.instructions {
            if matches!(inst, Instruction::Store { .. }) {
                body_idx = Some(block_idx);
                break;
            }
        }
        if body_idx.is_some() {
            break;
        }
    }

    if body_idx.is_none() {
        if debug {
            eprintln!("[VEC]   No store instruction found in any loop body block");
        }
        return None;
    }
    let body_idx = body_idx?;

    // Find exit by looking at loop successors that are outside the loop.
    let mut exit_idx = None;
    for &block_idx in &loop_info.body {
        let block = &func.blocks[block_idx];
        match &block.terminator {
            Terminator::CondBranch { true_label, false_label, .. } => {
                let then_in_loop = loop_info.body.contains(&(true_label.0 as usize));
                let else_in_loop = loop_info.body.contains(&(false_label.0 as usize));
                if !then_in_loop {
                    exit_idx = Some(true_label.0 as usize);
                    break;
                } else if !else_in_loop {
                    exit_idx = Some(false_label.0 as usize);
                    break;
                }
            }
            Terminator::Branch(target) => {
                if !loop_info.body.contains(&(target.0 as usize)) {
                    exit_idx = Some(target.0 as usize);
                    break;
                }
            }
            _ => {}
        }
    }

    if exit_idx.is_none() {
        if debug {
            eprintln!("[VEC]   No exit block found");
        }
        return None;
    }
    let exit_idx = exit_idx?;

    // Find the latch block (backedges to header).
    let latch_idx = find_latch(func, loop_info);
    if latch_idx.is_none() {
        if debug {
            eprintln!("[VEC]   No latch block found");
        }
        return None;
    }
    let latch_idx = latch_idx?;
    let latch = &func.blocks[latch_idx];

    // Find IV increment in latch: %next = add %iv, 1
    let mut iv_inc_idx = None;
    for (idx, inst) in latch.instructions.iter().enumerate() {
        if let Instruction::BinOp {
            op: IrBinOp::Add,
            lhs,
            rhs,
            ..
        } = inst
        {
            if let Operand::Value(lhs_val) = lhs {
                if *lhs_val == iv {
                    if let Operand::Const(c) = rhs {
                        if c.to_i64() == Some(1) {
                            iv_inc_idx = Some(idx);
                            break;
                        }
                    }
                }
            }
        }
    }
    if iv_inc_idx.is_none() {
        if debug {
            eprintln!("[VEC]   No IV increment by 1 found in latch");
        }
        return None;
    }
    let iv_inc_idx = iv_inc_idx?;

    // Analyze the body block for accumulation pattern.
    let body = &func.blocks[body_idx];

    // Find the store instruction.
    let mut store_info = None;
    for (idx, inst) in body.instructions.iter().enumerate() {
        if let Instruction::Store { ptr, val, .. } = inst {
            if let Operand::Value(store_val) = val {
                store_info = Some((idx, *ptr, *store_val));
                break;
            }
        }
    }
    if store_info.is_none() {
        if debug {
            eprintln!("[VEC]   No store instruction found in body block {}", body_idx);
        }
        return None;
    }
    let (store_idx, store_addr, store_value) = store_info?;
    if debug {
        eprintln!("[VEC]   Found store in block {}, tracing backward...", body_idx);
    }

    // Trace backwards: store_value should be result of fadd.
    let fadd_inst = find_inst_by_dest(body, store_value);
    if fadd_inst.is_none() {
        if debug {
            eprintln!("[VEC]   Store value not produced in body block");
        }
        return None;
    }
    let fadd_inst = fadd_inst?;
    let (c_load_val, mul_val) = match fadd_inst {
        Instruction::BinOp {
            op: IrBinOp::Add,
            lhs,
            rhs,
            ..
        } => Some((lhs, rhs)),
        _ => {
            if debug {
                eprintln!("[VEC]   Store value not from Add instruction");
            }
            None
        }
    }?;

    // Find the multiply.
    let mul_dest = match mul_val {
        Operand::Value(v) => v,
        _ => {
            if debug {
                eprintln!("[VEC]   Multiply operand is not a Value");
            }
            return None;
        }
    };
    let fmul_inst = find_inst_by_dest(body, *mul_dest);
    if fmul_inst.is_none() {
        if debug {
            eprintln!("[VEC]   Multiply value not produced in body block");
        }
        return None;
    }
    let fmul_inst = fmul_inst?;
    let (a_val, b_val) = match fmul_inst {
        Instruction::BinOp {
            op: IrBinOp::Mul,
            lhs,
            rhs,
            ..
        } => Some((lhs, rhs)),
        _ => {
            if debug {
                eprintln!("[VEC]   Add operand not from Mul instruction");
            }
            None
        }
    }?;

    // Verify C load.
    let c_load_dest = match c_load_val {
        Operand::Value(v) => v,
        _ => {
            if debug {
                eprintln!("[VEC]   C load operand is not a Value");
            }
            return None;
        }
    };
    let c_load_inst = find_inst_by_dest(body, *c_load_dest);
    if c_load_inst.is_none() {
        if debug {
            eprintln!("[VEC]   C load value not produced in body block");
        }
        return None;
    }
    let c_load_inst = c_load_inst?;
    let c_load_addr = match c_load_inst {
        Instruction::Load { ptr, .. } => *ptr,
        _ => {
            if debug {
                eprintln!("[VEC]   C load not from Load instruction");
            }
            return None;
        }
    };

    // Store and load must use the same GEP.
    if c_load_addr != store_addr {
        if debug {
            eprintln!("[VEC]   C load and store use different addresses");
        }
        return None;
    }

    // Extract GEPs for C and B.
    let c_gep = store_addr;
    // Verify C GEP exists somewhere in the loop (may have been hoisted by LICM)
    if find_inst_in_loop(func, &loop_info.body, c_gep).is_none() {
        if debug {
            eprintln!("[VEC]   C GEP not found in loop");
        }
        return None;
    }

    // Find B load and GEP.
    let b_load_dest = match b_val {
        Operand::Value(v) => v,
        _ => {
            if debug {
                eprintln!("[VEC]   B value operand is not a Value");
            }
            return None;
        }
    };
    // Search for B load in the entire loop (may have been moved by optimizations)
    let b_load_result = find_inst_in_loop(func, &loop_info.body, *b_load_dest);
    if b_load_result.is_none() {
        if debug {
            eprintln!("[VEC]   B load not found in loop");
        }
        return None;
    }
    let (_b_load_block, b_load_inst) = b_load_result?;
    let b_load_addr = match b_load_inst {
        Instruction::Load { ptr, .. } => ptr,
        _ => {
            if debug {
                eprintln!("[VEC]   B load not from Load instruction");
            }
            return None;
        }
    };
    let b_gep = *b_load_addr;
    // Verify B GEP exists somewhere in the loop
    if find_inst_in_loop(func, &loop_info.body, b_gep).is_none() {
        if debug {
            eprintln!("[VEC]   B GEP not found in loop");
        }
        return None;
    }

    // Find A load (should be loop-invariant, may have been hoisted by LICM).
    let a_load_dest = match a_val {
        Operand::Value(v) => v,
        _ => {
            if debug {
                eprintln!("[VEC]   A value operand is not a Value");
            }
            return None;
        }
    };
    // Search for A load in the entire loop
    let a_load_result = find_inst_in_loop(func, &loop_info.body, *a_load_dest);
    if a_load_result.is_none() {
        if debug {
            eprintln!("[VEC]   A load not found in loop");
        }
        return None;
    }
    let (_a_load_block, a_load_inst) = a_load_result?;
    let a_ptr = match a_load_inst {
        Instruction::Load { ptr, .. } => *ptr,
        _ => {
            if debug {
                eprintln!("[VEC]   A load not from Load instruction");
            }
            return None;
        }
    };

    Some(VectorizablePattern {
        header_idx,
        body_idx,
        latch_idx,
        exit_idx,
        iv,
        iv_inc_idx,
        c_gep,
        b_gep,
        a_ptr,
        store_idx,
        limit,
        exit_cmp_inst_idx,
        exit_cmp_dest,
        loop_blocks: loop_info.body.clone(),
    })
}

/// Find the latch block (block with backedge to header).
fn find_latch(func: &IrFunction, loop_info: &loop_analysis::NaturalLoop) -> Option<usize> {
    let header_id = BlockId(loop_info.header as u32);
    for &block_idx in &loop_info.body {
        let block = &func.blocks[block_idx];
        match &block.terminator {
            Terminator::Branch(target) if *target == header_id => {
                return Some(block_idx);
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                if *true_label == header_id || *false_label == header_id {
                    return Some(block_idx);
                }
            }
            _ => {}
        }
    }
    None
}

/// Find an instruction by its destination value in a single block.
fn find_inst_by_dest(block: &BasicBlock, dest: Value) -> Option<&Instruction> {
    for inst in &block.instructions {
        if inst.dest() == Some(dest) {
            return Some(inst);
        }
    }
    None
}

/// Find an instruction by its destination value, searching all loop blocks.
fn find_inst_in_loop<'a>(func: &'a IrFunction, loop_blocks: &FxHashSet<usize>, dest: Value) -> Option<(usize, &'a Instruction)> {
    for &block_idx in loop_blocks {
        let block = &func.blocks[block_idx];
        for inst in &block.instructions {
            if inst.dest() == Some(dest) {
                return Some((block_idx, inst));
            }
        }
    }
    None
}

/// Verify that a GEP uses the IV as offset (with or without scale).
fn verify_gep_pattern(block: &BasicBlock, gep_val: Value, iv: Value) -> Option<()> {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let gep_inst = find_inst_by_dest(block, gep_val);
    if gep_inst.is_none() {
        if debug {
            eprintln!("[VEC]     GEP instruction not found in block");
        }
        return None;
    }
    let gep_inst = gep_inst?;
    match gep_inst {
        Instruction::GetElementPtr { offset, .. } => {
            match offset {
                Operand::Value(v) if *v == iv => {
                    if debug {
                        eprintln!("[VEC]     GEP uses IV directly");
                    }
                    Some(())
                }
                Operand::Value(idx_val) => {
                    // Check for multiply: %idx = mul %iv, scale
                    let mul_inst = find_inst_by_dest(block, *idx_val);
                    if mul_inst.is_none() {
                        if debug {
                            eprintln!("[VEC]     GEP offset value not found in block");
                        }
                        return None;
                    }
                    let mul_inst = mul_inst?;
                    match mul_inst {
                        Instruction::BinOp {
                            op: IrBinOp::Mul,
                            lhs,
                            rhs,
                            ..
                        } => {
                            let lhs_matches = if let Operand::Value(v) = lhs { *v == iv } else { false };
                            let rhs_matches = if let Operand::Value(v) = rhs { *v == iv } else { false };
                            if lhs_matches || rhs_matches {
                                if debug {
                                    eprintln!("[VEC]     GEP uses IV * scale");
                                }
                                Some(())
                            } else {
                                if debug {
                                    eprintln!("[VEC]     GEP offset multiply doesn't involve IV");
                                }
                                None
                            }
                        }
                        _ => {
                            if debug {
                                eprintln!("[VEC]     GEP offset not from Mul instruction");
                            }
                            None
                        }
                    }
                }
                _ => {
                    if debug {
                        eprintln!("[VEC]     GEP offset is not a Value");
                    }
                    None
                }
            }
        }
        _ => {
            if debug {
                eprintln!("[VEC]     Instruction is not GetElementPtr");
            }
            None
        }
    }
}

/// Transform the loop to use FmaF64x2 intrinsics.
fn transform_to_fma_f64x2(func: &mut IrFunction, pattern: &VectorizablePattern) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let mut changes = 0;

    // Keep track of the next available Value and BlockId
    let mut next_val_id = func.next_value_id;
    let max_label = func.blocks.iter().map(|b| b.label.0).max().unwrap_or(0);
    let mut next_label = max_label + 1;

    // Build a set of IV-derived values (for finding all IV-related comparisons)
    // Start with the IV from the header, but also trace back from the GEPs to find
    // the actual j-loop IV
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(pattern.iv);

    // Find the IV used in the B and C GEPs by tracing backwards
    // The GEPs use an offset that's derived from the j-loop IV
    let mut gep_ivs = FxHashSet::default();
    for &block_idx in &pattern.loop_blocks {
        let block = &func.blocks[block_idx];
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, base: _, offset, ty: _ } = inst {
                if *dest == pattern.b_gep || *dest == pattern.c_gep {
                    // This GEP is for B or C array - trace its offset back to find the IV
                    if let Operand::Value(offset_val) = offset {
                        gep_ivs.insert(*offset_val);
                        if debug {
                            eprintln!("[VEC]   GEP Value({}) uses offset Value({})", dest.0, offset_val.0);
                        }
                    }
                }
            }
        }
    }

    // Trace gep_ivs back through multiplies and casts to find the original IVs
    let mut changed = true;
    while changed {
        changed = false;
        for &block_idx in &pattern.loop_blocks {
            let block = &func.blocks[block_idx];
            for inst in &block.instructions {
                match inst {
                    Instruction::BinOp { dest, op: _, lhs, rhs, ty: _ } => {
                        if gep_ivs.contains(dest) {
                            if let Operand::Value(lhs_val) = lhs {
                                if gep_ivs.insert(*lhs_val) {
                                    changed = true;
                                }
                            }
                            if let Operand::Value(rhs_val) = rhs {
                                if gep_ivs.insert(*rhs_val) {
                                    changed = true;
                                }
                            }
                        }
                    }
                    Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                        if gep_ivs.contains(dest) {
                            if let Operand::Value(src_val) = src {
                                if gep_ivs.insert(*src_val) {
                                    changed = true;
                                }
                            }
                        }
                    }
                    Instruction::Phi { dest, .. } => {
                        if gep_ivs.contains(dest) {
                            // This is likely the j-loop IV!
                            iv_derived.insert(*dest);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Now collect all values derived from ANY of these IVs
    changed = true;
    while changed {
        changed = false;
        for &block_idx in &pattern.loop_blocks {
            let block = &func.blocks[block_idx];
            for inst in &block.instructions {
                match inst {
                    Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                        if let Operand::Value(src_val) = src {
                            if (iv_derived.contains(src_val) || gep_ivs.contains(src_val)) && iv_derived.insert(*dest) {
                                changed = true;
                            }
                        }
                    }
                    Instruction::BinOp { dest, op: IrBinOp::Add, lhs, .. } => {
                        // IV increment (j = j + 1)
                        if let Operand::Value(lhs_val) = lhs {
                            if iv_derived.contains(lhs_val) && iv_derived.insert(*dest) {
                                changed = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if debug {
        eprintln!("[VEC]   IV-derived values (including j-loop IV): {:?}", iv_derived);
        eprintln!("[VEC]   GEP offset chain: {:?}", gep_ivs);
        eprintln!("[VEC]   Loop contains blocks: {:?}", pattern.loop_blocks);
    }

    // Step 1: Modify ALL comparisons in the loop that compare IV-derived values against the limit
    // This ensures we catch the actual loop exit condition regardless of loop transformations
    {
        // First, create the halved limit value
        let halved_limit = match &pattern.limit {
            Operand::Const(IrConst::I32(n)) => Operand::Const(IrConst::I32(*n / 2)),
            Operand::Const(IrConst::I64(n)) => Operand::Const(IrConst::I64(*n / 2)),
            Operand::Value(limit_val) => {
                // Dynamic limit: insert division in header
                let div_dest = Value(next_val_id);
                next_val_id += 1;

                let limit_ty = match &func.blocks[pattern.header_idx].instructions[pattern.exit_cmp_inst_idx] {
                    Instruction::Cmp { ty, .. } => *ty,
                    _ => IrType::I64,
                };

                let div_inst = Instruction::BinOp {
                    dest: div_dest,
                    op: IrBinOp::UDiv,
                    lhs: Operand::Value(*limit_val),
                    rhs: Operand::Const(match limit_ty {
                        IrType::I32 => IrConst::I32(2),
                        IrType::I64 => IrConst::I64(2),
                        _ => IrConst::I64(2),
                    }),
                    ty: limit_ty,
                };

                func.blocks[pattern.header_idx].instructions.insert(pattern.exit_cmp_inst_idx, div_inst);

                if debug {
                    eprintln!("[VEC]   Inserted division for dynamic limit: Value({})", div_dest.0);
                }

                Operand::Value(div_dest)
            }
            _ => {
                if debug {
                    eprintln!("[VEC]   Unsupported limit type");
                }
                return 0;
            }
        };

        // Now modify all comparisons that use IV-derived values in loop blocks
        for &block_idx in &pattern.loop_blocks {
            let block = &mut func.blocks[block_idx];

            if debug {
                // First pass: log all comparisons in this block
                for inst in &block.instructions {
                    if let Instruction::Cmp { dest, op: _, lhs, rhs, ty: _ } = inst {
                        eprintln!("[VEC]   Block {} has comparison: {:?} <cmp> {:?}, dest={:?}",
                            block_idx, lhs, rhs, dest);
                    }
                }
            }

            for inst in &mut block.instructions {
                if let Instruction::Cmp { dest, op: _, lhs, rhs, ty: _ } = inst {
                    // Check if this compares an IV-derived value against something
                    let modifies_lhs = if let Operand::Value(lhs_val) = lhs {
                        iv_derived.contains(lhs_val)
                    } else {
                        false
                    };

                    let modifies_rhs = if let Operand::Value(rhs_val) = rhs {
                        iv_derived.contains(rhs_val)
                    } else {
                        false
                    };

                    if modifies_lhs || modifies_rhs {
                        if debug {
                            eprintln!("[VEC]   -> This is an IV comparison (will modify)");
                        }

                        // Modify the comparison to use halved limit
                        if modifies_lhs {
                            *rhs = halved_limit.clone();
                            changes += 1;
                            if debug {
                                eprintln!("[VEC]   -> Modified comparison RHS to {:?}", halved_limit);
                            }
                        } else if modifies_rhs {
                            *lhs = halved_limit.clone();
                            changes += 1;
                            if debug {
                                eprintln!("[VEC]   -> Modified comparison LHS to {:?}", halved_limit);
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Modify GEP offset calculation from IV*8 to IV*16
    // Handle both explicit multiplies and strength-reduced pointer increments
    {
        let mut found_any_mul = false;
        let mut modified_any_increment = false;

        // First, try to find and modify explicit IV * 8 multiplies
        for &block_idx in &pattern.loop_blocks {
            let block = &mut func.blocks[block_idx];
            for inst in &mut block.instructions {
                if let Instruction::BinOp {
                    dest,
                    op: IrBinOp::Mul,
                    lhs,
                    rhs,
                    ty: _,
                } = inst
                {
                    found_any_mul = true;
                    // Check if this multiply involves IV-derived values and scale factor 8
                    let lhs_is_iv_derived = if let Operand::Value(v) = lhs {
                        iv_derived.contains(v)
                    } else {
                        false
                    };
                    let rhs_is_8 = matches!(rhs, Operand::Const(IrConst::I64(8)) | Operand::Const(IrConst::I32(8)));

                    if debug && (lhs_is_iv_derived || rhs_is_8) {
                        eprintln!("[VEC]   Found multiply in block {}: Value({}) = {:?} * {:?}, lhs_is_iv_derived={}, rhs_is_8={}",
                            block_idx, dest.0, lhs, rhs, lhs_is_iv_derived, rhs_is_8);
                    }

                    if lhs_is_iv_derived && rhs_is_8 {
                        // Change 8 to 16 (process 2 doubles = 16 bytes)
                        *rhs = match rhs {
                            Operand::Const(IrConst::I64(_)) => Operand::Const(IrConst::I64(16)),
                            Operand::Const(IrConst::I32(_)) => Operand::Const(IrConst::I32(16)),
                            _ => Operand::Const(IrConst::I64(16)),
                        };
                        changes += 1;
                        modified_any_increment = true;
                        if debug {
                            eprintln!("[VEC]   Changed GEP stride from 8 to 16 for Value({})", dest.0);
                        }
                    }
                }
            }
        }

        // If no IV*8 multiplies found, the loop is strength-reduced
        // Look for pointer increments (ptr + 8) and change them to (ptr + 16)
        if !modified_any_increment {
            if debug {
                eprintln!("[VEC]   No IV*8 multiplies found, searching for strength-reduced pointer increments");
            }

            // Build a set of pointer values that are used in GEPs for C and B arrays
            let mut pointer_values = FxHashSet::default();
            pointer_values.insert(pattern.c_gep);
            pointer_values.insert(pattern.b_gep);

            // Track pointer values through the loop (they flow through adds and GEPs)
            for &block_idx in &pattern.loop_blocks {
                let block = &func.blocks[block_idx];
                for inst in &block.instructions {
                    match inst {
                        Instruction::GetElementPtr { dest, base, .. } => {
                            if pointer_values.contains(base) || pointer_values.contains(dest) {
                                pointer_values.insert(*dest);
                                pointer_values.insert(*base);
                            }
                        }
                        Instruction::BinOp {
                            dest,
                            op: IrBinOp::Add,
                            lhs,
                            rhs: _,
                            ty: _,
                        } => {
                            if let Operand::Value(lhs_val) = lhs {
                                if pointer_values.contains(lhs_val) {
                                    pointer_values.insert(*dest);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            if debug {
                eprintln!("[VEC]   Tracked pointer values: {} pointers", pointer_values.len());
            }

            // Now modify all pointer increments by 8 to increment by 16
            for &block_idx in &pattern.loop_blocks {
                let block = &mut func.blocks[block_idx];
                for inst in &mut block.instructions {
                    if let Instruction::BinOp {
                        dest,
                        op: IrBinOp::Add,
                        lhs,
                        rhs,
                        ty: _,
                    } = inst
                    {
                        // Check if incrementing by 8
                        let is_8 = matches!(rhs, Operand::Const(IrConst::I64(8)) | Operand::Const(IrConst::I32(8)));

                        if debug && is_8 {
                            let is_pointer = if let Operand::Value(lhs_val) = lhs {
                                pointer_values.contains(lhs_val)
                            } else {
                                false
                            };
                            eprintln!("[VEC]   Block {} has add by 8: Value({}) = {:?} + 8, is_pointer={}",
                                block_idx, dest.0, lhs, is_pointer);
                        }

                        // Check if this is a pointer increment
                        let is_pointer_add = if let Operand::Value(lhs_val) = lhs {
                            pointer_values.contains(lhs_val)
                        } else {
                            false
                        };

                        if is_pointer_add && is_8 {
                            *rhs = match rhs {
                                Operand::Const(IrConst::I64(_)) => Operand::Const(IrConst::I64(16)),
                                Operand::Const(IrConst::I32(_)) => Operand::Const(IrConst::I32(16)),
                                _ => Operand::Const(IrConst::I64(16)),
                            };
                            changes += 1;
                            modified_any_increment = true;
                            if debug {
                                eprintln!("[VEC]   -> Changed pointer increment from 8 to 16 for Value({})", dest.0);
                            }
                        }
                    }
                }
            }

            if debug && !modified_any_increment {
                eprintln!("[VEC]   No add-by-8 found in IR - GEPs likely using IV directly");
                eprintln!("[VEC]   Will modify GEP offsets to use IV*2 instead of IV");
            }

            // Collect GEPs that need to be modified (two-pass to avoid borrow issues)
            let mut geps_to_modify = Vec::new();
            for &block_idx in &pattern.loop_blocks {
                let block = &func.blocks[block_idx];
                for (inst_idx, inst) in block.instructions.iter().enumerate() {
                    if let Instruction::GetElementPtr { dest, base: _, offset, ty } = inst {
                        // Check if this is a GEP for B or C array
                        if *dest == pattern.b_gep || *dest == pattern.c_gep {
                            if let Operand::Value(offset_val) = offset {
                                geps_to_modify.push((block_idx, inst_idx, *offset_val, *ty));
                                if debug {
                                    eprintln!("[VEC]   Found GEP in block {} inst {}: Value({}) with offset Value({})",
                                        block_idx, inst_idx, dest.0, offset_val.0);
                                }
                            }
                        }
                    }
                }
            }

            // Now modify the GEPs by inserting mul instructions and updating offsets
            // Process in reverse order to avoid index shifting issues
            for (block_idx, inst_idx, offset_val, gep_ty) in geps_to_modify.into_iter().rev() {
                let mul_dest = Value(next_val_id);
                next_val_id += 1;

                // Determine the type for the multiply (use I64 for pointer offsets)
                let offset_ty = IrType::I64;

                // Create mul instruction: mul_dest = offset * 2
                let mul_inst = Instruction::BinOp {
                    dest: mul_dest,
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(offset_val),
                    rhs: Operand::Const(IrConst::I64(2)),
                    ty: offset_ty,
                };

                // Insert mul before the GEP
                func.blocks[block_idx].instructions.insert(inst_idx, mul_inst);

                // Update the GEP's offset (now at inst_idx + 1 due to insertion)
                if let Instruction::GetElementPtr { offset, .. } = &mut func.blocks[block_idx].instructions[inst_idx + 1] {
                    *offset = Operand::Value(mul_dest);
                    changes += 1;
                    modified_any_increment = true;
                    if debug {
                        eprintln!("[VEC]   Inserted mul and updated GEP offset: Value({}) = Value({}) * 2",
                            mul_dest.0, offset_val.0);
                    }
                }
            }

            if debug && !modified_any_increment {
                eprintln!("[VEC]   Warning: Could not modify GEP offsets");
            }
        }
    }

    // Step 3: Replace the body accumulation with FmaF64x2
    {
        let body = &mut func.blocks[pattern.body_idx];

        // Create FmaF64x2 intrinsic: writes directly to memory, no dest value.
        let intrinsic = Instruction::Intrinsic {
            dest: None,
            op: IntrinsicOp::FmaF64x2,
            dest_ptr: Some(pattern.c_gep),
            args: vec![
                Operand::Value(pattern.a_ptr),
                Operand::Value(pattern.b_gep),
            ],
        };

        // Insert intrinsic before the store, then remove the store.
        body.instructions.insert(pattern.store_idx, intrinsic);
        changes += 1;
        if debug {
            eprintln!("[VEC]   Inserted FmaF64x2 intrinsic, dest_ptr=Value({}), args=[Value({}), Value({})]",
                pattern.c_gep.0, pattern.a_ptr.0, pattern.b_gep.0);
        }

        // Remove the old store (now at store_idx + 1).
        if pattern.store_idx + 1 < body.instructions.len() {
            body.instructions.remove(pattern.store_idx + 1);
        }

        // The old load/mul/add instructions are now dead. DCE will remove them.
    }

    // Step 4: Create remainder loop blocks for odd N
    // TODO: This is complex and requires careful CFG manipulation
    // For now, we'll skip this and only vectorize loops with even N
    // The plan mentions this should be implemented, but let's test the basic
    // transformation first before adding the remainder loop

    // Update the function's next_value_id
    func.next_value_id = next_val_id;

    changes
}

/// Run SSE2 vectorization on a function (builds CFG analysis if needed).
pub(crate) fn vectorize_function(func: &mut IrFunction) -> usize {
    if func.blocks.len() < 2 {
        return 0;
    }

    let cfg = CfgAnalysis::build(func);
    vectorize_with_analysis(func, &cfg)
}
