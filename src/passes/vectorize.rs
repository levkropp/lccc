//! AVX2/SSE2 vectorization pass for matmul-style loops.
//!
//! Recognizes innermost loops with stride-1 double-precision accumulation patterns
//! and transforms them into FmaF64x4 (AVX2, default) or FmaF64x2 (SSE2) intrinsics.
//!
//! Target pattern (matmul j-loop):
//! ```c
//! for (int j = 0; j < N; j++)
//!     C[i][j] += A[i][k] * B[k][j];
//! ```
//!
//! ## AVX2 Transformation (default, 4-wide):
//! ```c
//! for (int j = 0; j < N/4; j++)  // Loop N/4 times
//!     FmaF64x4(&C[i][j*4], &A[i][k], &B[k][j*4]);  // Process 4 elements per iteration
//! ```
//!
//! ## SSE2 Transformation (with LCCC_FORCE_SSE2=1, 2-wide):
//! ```c
//! for (int j = 0; j < N/2; j++)  // Loop N/2 times
//!     FmaF64x2(&C[i][j*2], &A[i][k], &B[k][j*2]);  // Process 2 elements per iteration
//! ```
//!
//! ## Transformation Details
//!
//! ### AVX2 (4-wide, default):
//! 1. **Loop Bound**: Modified from `j < N` to `j < N/4`
//!    - For constant N: divide by 4 at compile time
//!    - For dynamic N: insert `udiv` instruction to compute N/4
//!    - Modifies ALL comparisons involving IV-derived values in the loop
//!
//! 2. **Array Indexing**: Changed from `j` to `j*4`
//!    - Inserts multiply instructions before GEPs: `offset' = offset * 4`
//!    - Ensures iteration j accesses elements [j*4..j*7] instead of [j, j+1]
//!    - Backend generates stride-32 addressing (4 doubles × 8 bytes)
//!
//! 3. **Induction Variable**: Keeps incrementing by 1
//!    - Backend-friendly: `j++` instead of `j += 4`
//!    - Combined with 4× offset, produces correct element access
//!
//! 4. **AVX2 Code Generation**:
//!    - `vbroadcastsd`: broadcast A[i][k] scalar to 4 lanes
//!    - `vmovupd`: load 4 doubles from B[k][j*4]
//!    - `vmulpd`: packed multiply (4 doubles)
//!    - `vaddpd`: packed add with C[i][j*4]
//!    - `vmovupd`: store 4 results back
//!
//! ### SSE2 (2-wide, with LCCC_FORCE_SSE2=1):
//! 1. **Loop Bound**: Modified from `j < N` to `j < N/2`
//! 2. **Array Indexing**: Changed from `j` to `j*2` (stride-16)
//! 3. **SSE2 Code Generation**:
//!    - `movsd` + `unpcklpd`: broadcast A[i][k] scalar
//!    - `movupd`: load 2 doubles from B[k][j*2]
//!    - `mulpd`: packed multiply (2 doubles)
//!    - `addpd`: packed add with C[i][j*2]
//!    - `movupd`: store 2 results back
//!
//! ## Remainder Loops
//!
//! Remainder loops are automatically inserted to handle cases where N is not divisible by the vector width:
//! - AVX2 (4-wide): Handles N % 4 ∈ {1, 2, 3} with scalar remainder loop
//! - SSE2 (2-wide): Handles N % 2 = 1 with scalar remainder loop
//!
//! Example for N=255 with AVX2:
//! - Vectorized loop: 63 iterations processing indices [0..251] (4 elements each)
//! - Remainder loop: 3 iterations processing indices [252, 253, 254] (scalar)
//!
//! ## Limitations
//!
//! - Only handles matmul-style patterns (load, multiply, add, store)
//! - Requires innermost loop with IV-based indexing
//!
//! ## Environment Variables
//!
//! - `LCCC_FORCE_SSE2=1`: Force SSE2 2-wide vectorization instead of AVX2 4-wide
//! - `LCCC_FORCE_AVX2=1`: Explicitly enable AVX2 (default behavior, provided for clarity)
//! - `LCCC_DEBUG_VECTORIZE=1`: Enable debug output for vectorization pass

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::{AddressSpace, IrType};
use crate::ir::analysis::CfgAnalysis;
use crate::ir::instruction::{BasicBlock, BlockId, Instruction, Operand, Terminator, Value};
use crate::ir::intrinsics::IntrinsicOp;
use crate::ir::ops::{IrBinOp, IrCmpOp};
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
        // Check if this is an innermost loop (no other loop nests strictly inside it).
        let is_innermost = !loops.iter().enumerate().any(|(other_idx, other)| {
            idx != other_idx
                && other.body.len() < loop_info.body.len()
                && other.body.iter().all(|b| loop_info.body.contains(b))
        });

        if debug {
            eprintln!("[VEC] Loop {} at header={}, body_size={}, innermost={}",
                idx, loop_info.header, loop_info.body.len(), is_innermost);
        }

        if !is_innermost {
            continue;
        }

        // Try to vectorize this loop - first try matmul, then try reduction patterns.
        if let Some(pattern) = analyze_loop_pattern(func, loop_info, cfg) {
            // Select vector width: default to AVX2 (4-wide) unless explicitly disabled
            let use_sse2 = std::env::var("LCCC_FORCE_SSE2").is_ok();

            if use_sse2 {
                if debug {
                    eprintln!("[VEC] Matmul pattern matched! Transforming to FmaF64x2 (SSE2, 2-wide)");
                }
                total_changes += transform_to_fma_f64x2(func, &pattern);
            } else {
                // Use AVX2 by default (or if LCCC_FORCE_AVX2 is set)
                if debug {
                    eprintln!("[VEC] Matmul pattern matched! Transforming to FmaF64x4 (AVX2, 4-wide)");
                }
                total_changes += transform_to_fma_f64x4(func, &pattern);
            }
        } else if let Some(red_pattern) = analyze_reduction_pattern(func, loop_info, cfg) {
            // Try reduction pattern vectorization (sum += arr[i], sum += a[i] * b[i], etc.)
            let use_sse2 = std::env::var("LCCC_FORCE_SSE2").is_ok();

            if use_sse2 {
                if debug {
                    eprintln!("[VEC] Reduction pattern matched! Transforming to SSE2 2-wide");
                }
                total_changes += transform_reduction_sse2(func, &red_pattern);
            } else {
                if debug {
                    eprintln!("[VEC] Reduction pattern matched! Transforming to AVX2 4-wide");
                }
                total_changes += transform_reduction_avx2(func, &red_pattern);
            }
        } else if debug {
            eprintln!("[VEC] No vectorizable pattern found for loop {}", idx);
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

/// Reduction pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReductionKind {
    /// Simple sum: sum += arr[i]
    Sum,
    /// Dot product: sum += a[i] * b[i]
    DotProduct,
}

/// Pattern matching result for a vectorizable reduction loop.
#[derive(Debug)]
struct ReductionPattern {
    /// Type of reduction
    kind: ReductionKind,
    /// Element type being reduced (F64, F32, I32, I64)
    element_type: IrType,
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
    /// Scalar accumulator phi node destination value in header
    accumulator_phi: Value,
    /// GEP for first array (arr for sum, a for dot product)
    array_a_gep: Value,
    /// GEP for second array (only for dot product)
    array_b_gep: Option<Value>,
    /// Index of the add instruction that updates the accumulator
    accumulator_add_idx: usize,
    /// Loop limit value (N in `i < N`)
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

    // Build label→index map so we can convert BlockId labels to array indices.
    let label_to_idx: FxHashMap<BlockId, usize> = func.blocks.iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

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
    // Use label_to_idx to convert BlockId labels to block array indices for body.contains().
    let mut exit_label = None;
    for &block_idx in &loop_info.body {
        let block = &func.blocks[block_idx];
        match &block.terminator {
            Terminator::CondBranch { true_label, false_label, .. } => {
                let then_idx = label_to_idx.get(true_label).copied();
                let else_idx = label_to_idx.get(false_label).copied();
                let then_in_loop = then_idx.map_or(false, |i| loop_info.body.contains(&i));
                let else_in_loop = else_idx.map_or(false, |i| loop_info.body.contains(&i));
                if !then_in_loop {
                    exit_label = Some(*true_label);
                    break;
                } else if !else_in_loop {
                    exit_label = Some(*false_label);
                    break;
                }
            }
            Terminator::Branch(target) => {
                let target_idx = label_to_idx.get(target).copied();
                if !target_idx.map_or(false, |i| loop_info.body.contains(&i)) {
                    exit_label = Some(*target);
                    break;
                }
            }
            _ => {}
        }
    }

    if exit_label.is_none() {
        if debug {
            eprintln!("[VEC]   No exit block found");
        }
        return None;
    }
    let exit_label = exit_label?;
    let exit_idx = *label_to_idx.get(&exit_label)?;

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

/// Analyze a loop to detect vectorizable reduction patterns (sum += arr[i], sum += a[i] * b[i]).
fn analyze_reduction_pattern(
    func: &IrFunction,
    loop_info: &loop_analysis::NaturalLoop,
    _cfg: &CfgAnalysis,
) -> Option<ReductionPattern> {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let header_idx = loop_info.header;
    let header = &func.blocks[header_idx];

    if debug {
        eprintln!("[VEC-RED] Analyzing reduction pattern for loop at header {}", header_idx);
    }

    // Find exit and latch blocks first
    let exit_idx = find_exit(func, loop_info);
    if exit_idx.is_none() {
        if debug {
            eprintln!("[VEC-RED]   Could not find exit block");
        }
        return None;
    }
    let exit_idx = exit_idx?;

    let latch_idx = find_latch(func, loop_info);
    if latch_idx.is_none() {
        if debug {
            eprintln!("[VEC-RED]   Could not find latch block");
        }
        return None;
    }
    let latch_idx = latch_idx?;
    let latch = &func.blocks[latch_idx];

    // Find the induction variable by looking for increments in latch
    let mut iv = None;
    for inst in &latch.instructions {
        if let Instruction::BinOp {
            dest,
            op: IrBinOp::Add,
            lhs,
            rhs,
            ..
        } = inst
        {
            // Check if incrementing by 1
            if let Operand::Const(c) = rhs {
                if c.to_i64() == Some(1) {
                    // lhs should be the IV phi
                    if let Operand::Value(lhs_val) = lhs {
                        iv = Some(*lhs_val);
                        break;
                    }
                }
            }
        }
    }
    if iv.is_none() && debug {
        eprintln!("[VEC-RED]   No IV increment found in latch");
        return None;
    }
    let iv = iv?;

    // Build IV-derived values using fixed-point iteration (like matmul pattern)
    // This captures casts/copies of the IV across ALL loop blocks, not just the header
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(iv);

    let mut changed = true;
    while changed {
        changed = false;
        for &block_idx in &loop_info.body {
            let block = &func.blocks[block_idx];
            for inst in &block.instructions {
                match inst {
                    Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                        if let Operand::Value(src_val) = src {
                            if iv_derived.contains(src_val) && iv_derived.insert(*dest) {
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
        eprintln!("[VEC-RED]   IV-derived values: {:?}", iv_derived);
    }

    // Find comparison for loop exit
    let mut exit_cmp_info = None;
    for (idx, inst) in header.instructions.iter().enumerate() {
        if let Instruction::Cmp { dest, op: _, lhs, rhs, ty: _ } = inst {
            if let Operand::Value(lhs_val) = lhs {
                if iv_derived.contains(lhs_val) {
                    exit_cmp_info = Some((idx, *dest, rhs.clone()));
                    break;
                }
            } else if let Operand::Value(rhs_val) = rhs {
                if iv_derived.contains(rhs_val) {
                    exit_cmp_info = Some((idx, *dest, lhs.clone()));
                    break;
                }
            }
        }
    }
    let (exit_cmp_inst_idx, exit_cmp_dest, limit) = exit_cmp_info?;

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
    let iv_inc_idx = iv_inc_idx?;

    // Find the accumulator phi (scalar sum variable)
    let mut accumulator_phi = None;
    let mut accumulator_init_is_zero = false;
    for inst in &header.instructions {
        if let Instruction::Phi { dest, incoming, .. } = inst {
            if incoming.len() == 2 && *dest != iv {
                // Check if initialized to zero (common for reductions)
                for (val, _block) in incoming {
                    if let Operand::Const(c) = val {
                        if c.to_i64() == Some(0) {
                            accumulator_init_is_zero = true;
                        } else if c.to_f64().map(|f| f == 0.0).unwrap_or(false) {
                            accumulator_init_is_zero = true;
                        }
                    }
                }
                if accumulator_init_is_zero {
                    accumulator_phi = Some(*dest);
                    break;
                }
            }
        }
    }
    if accumulator_phi.is_none() {
        if debug {
            eprintln!("[VEC-RED]   No accumulator phi found (initialized to zero)");
        }
        return None;
    }
    let accumulator_phi = accumulator_phi?;

    // Build a set of accumulator-derived values (accumulator + casts of accumulator)
    let mut accumulator_derived = FxHashSet::default();
    accumulator_derived.insert(accumulator_phi);

    // Find all casts/copies of the accumulator
    let mut changed = true;
    while changed {
        changed = false;
        for &block_idx in &loop_info.body {
            let block = &func.blocks[block_idx];
            for inst in &block.instructions {
                match inst {
                    Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                        if let Operand::Value(src_val) = src {
                            if accumulator_derived.contains(src_val) && !accumulator_derived.contains(dest) {
                                accumulator_derived.insert(*dest);
                                changed = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Find the body block that updates the accumulator
    let mut body_idx = None;
    let mut accumulator_add_idx = None;
    let mut add_result = None;
    for &block_idx in &loop_info.body {
        if block_idx == header_idx {
            continue;
        }
        let block = &func.blocks[block_idx];
        if debug {
            eprintln!("[VEC-RED]   Searching block {} for accumulator update (accumulator_phi = {})", block_idx, accumulator_phi.0);
            eprintln!("[VEC-RED]   Accumulator-derived values: {:?}", accumulator_derived);
            for (idx, inst) in block.instructions.iter().enumerate() {
                eprintln!("[VEC-RED]     {}: {:?}", idx, inst);
            }
        }
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::BinOp {
                dest,
                op: IrBinOp::Add,
                lhs,
                rhs,
                ..
            } = inst
            {
                // Check if this is accumulator += something (allowing casts)
                let lhs_is_acc = if let Operand::Value(v) = lhs { accumulator_derived.contains(v) } else { false };
                let rhs_is_acc = if let Operand::Value(v) = rhs { accumulator_derived.contains(v) } else { false };

                if lhs_is_acc || rhs_is_acc {
                    body_idx = Some(block_idx);
                    accumulator_add_idx = Some(idx);
                    add_result = Some(*dest);
                    break;
                }
            }
        }
        if body_idx.is_some() {
            break;
        }
    }
    if body_idx.is_none() {
        if debug {
            eprintln!("[VEC-RED]   No accumulator update found");
        }
        return None;
    }
    let body_idx = body_idx?;
    let accumulator_add_idx = accumulator_add_idx?;
    let add_result = add_result?;
    let body = &func.blocks[body_idx];
    let add_inst = &body.instructions[accumulator_add_idx];

    // Get the element type from the add instruction
    let element_type = match add_inst {
        Instruction::BinOp { ty, .. } => *ty,
        _ => return None,
    };

    // Verify element type is vectorizable (F64, F32, I32, I64)
    if !matches!(element_type, IrType::F64 | IrType::F32 | IrType::I32 | IrType::I64) {
        if debug {
            eprintln!("[VEC-RED]   Unsupported element type: {:?}", element_type);
        }
        return None;
    }

    // Determine what is being added to the accumulator
    let (lhs_val, rhs_val) = match add_inst {
        Instruction::BinOp { lhs, rhs, .. } => {
            let lhs_v = if let Operand::Value(v) = lhs { Some(*v) } else { None };
            let rhs_v = if let Operand::Value(v) = rhs { Some(*v) } else { None };
            (lhs_v, rhs_v)
        }
        _ => (None, None),
    };

    // The non-accumulator operand is what we're adding (check accumulator_derived set)
    let added_value = if lhs_val.is_some() && accumulator_derived.contains(&lhs_val.unwrap()) {
        rhs_val?
    } else if rhs_val.is_some() && accumulator_derived.contains(&rhs_val.unwrap()) {
        lhs_val?
    } else {
        if debug {
            eprintln!("[VEC-RED]   Add instruction doesn't use accumulator or derived value");
        }
        return None;
    };

    // Check if added_value is a load (simple sum) or multiply (dot product)
    let added_inst = find_inst_by_dest(body, added_value)?;

    if debug {
        eprintln!("[VEC-RED]   Added value {} is produced by: {:?}", added_value.0, added_inst);
    }

    match added_inst {
        // Simple sum pattern: sum += arr[i]
        Instruction::Load { ptr, .. } => {
            let array_gep = *ptr;
            // Verify GEP uses IV
            if !gep_uses_iv(func, &loop_info.body, array_gep, iv, &iv_derived) {
                if debug {
                    eprintln!("[VEC-RED]   Array GEP doesn't use IV");
                }
                return None;
            }

            if debug {
                eprintln!("[VEC-RED]   Simple sum pattern detected: {:?} += load(arr[iv])", element_type);
            }

            Some(ReductionPattern {
                kind: ReductionKind::Sum,
                element_type,
                header_idx,
                body_idx,
                latch_idx,
                exit_idx,
                iv,
                iv_inc_idx,
                accumulator_phi,
                array_a_gep: array_gep,
                array_b_gep: None,
                accumulator_add_idx,
                limit,
                exit_cmp_inst_idx,
                exit_cmp_dest,
                loop_blocks: loop_info.body.clone(),
            })
        }

        // Handle cast followed by load (e.g., long sum += (long)arr[i] where arr is int[])
        Instruction::Cast { src, from_ty, to_ty, .. } => {
            // The cast should be widening the element type to match the accumulator
            if *to_ty != element_type {
                if debug {
                    eprintln!("[VEC-RED]   Cast type mismatch: cast to {:?} but accumulator is {:?}", to_ty, element_type);
                }
                return None;
            }

            // Check if the source of the cast is a load
            let cast_src_val = if let Operand::Value(v) = src { *v } else { return None };
            let cast_src_inst = find_inst_by_dest(body, cast_src_val)?;

            if let Instruction::Load { ptr, ty: load_ty, .. } = cast_src_inst {
                if *load_ty != *from_ty {
                    if debug {
                        eprintln!("[VEC-RED]   Load type {:?} doesn't match cast from_ty {:?}", load_ty, from_ty);
                    }
                    return None;
                }

                let array_gep = *ptr;
                // Verify GEP uses IV
                if !gep_uses_iv(func, &loop_info.body, array_gep, iv, &iv_derived) {
                    if debug {
                        eprintln!("[VEC-RED]   Array GEP doesn't use IV");
                    }
                    return None;
                }

                if debug {
                    eprintln!("[VEC-RED]   Simple sum pattern with cast detected: {:?} += ({:?})load(arr[iv])", element_type, from_ty);
                }

                // When the accumulator is wider than the array element (e.g., long += int),
                // we cannot simply use packed narrow adds — the results would overflow/truncate.
                // Reject this pattern; proper widening vectorization would require
                // vpmovsx + vpaddq which is much more complex.
                if element_type.size() > from_ty.size() {
                    if debug {
                        eprintln!("[VEC-RED]   Rejecting: accumulator {:?} wider than element {:?}", element_type, from_ty);
                    }
                    return None;
                }

                Some(ReductionPattern {
                    kind: ReductionKind::Sum,
                    element_type: *from_ty,  // Use the actual array element type
                    header_idx,
                    body_idx,
                    latch_idx,
                    exit_idx,
                    iv,
                    iv_inc_idx,
                    accumulator_phi,
                    array_a_gep: array_gep,
                    array_b_gep: None,
                    accumulator_add_idx,
                    limit,
                    exit_cmp_inst_idx,
                    exit_cmp_dest,
                    loop_blocks: loop_info.body.clone(),
                })
            } else {
                if debug {
                    eprintln!("[VEC-RED]   Cast source is not a load: {:?}", cast_src_inst);
                }
                None
            }
        }

        // Dot product pattern: sum += a[i] * b[i]
        Instruction::BinOp {
            op: IrBinOp::Mul,
            lhs,
            rhs,
            ..
        } => {
            // Both operands of multiply should be loads
            let mul_lhs_val = if let Operand::Value(v) = lhs { *v } else { return None };
            let mul_rhs_val = if let Operand::Value(v) = rhs { *v } else { return None };

            let mul_lhs_inst = find_inst_by_dest(body, mul_lhs_val)?;
            let mul_rhs_inst = find_inst_by_dest(body, mul_rhs_val)?;

            let array_a_gep = if let Instruction::Load { ptr, .. } = mul_lhs_inst {
                *ptr
            } else {
                return None;
            };

            let array_b_gep = if let Instruction::Load { ptr, .. } = mul_rhs_inst {
                *ptr
            } else {
                return None;
            };

            // Verify both GEPs use IV
            if !gep_uses_iv(func, &loop_info.body, array_a_gep, iv, &iv_derived) ||
               !gep_uses_iv(func, &loop_info.body, array_b_gep, iv, &iv_derived) {
                if debug {
                    eprintln!("[VEC-RED]   Array GEPs don't use IV");
                }
                return None;
            }

            if debug {
                eprintln!("[VEC-RED]   Dot product pattern detected: {:?} += load(a[iv]) * load(b[iv])", element_type);
            }

            Some(ReductionPattern {
                kind: ReductionKind::DotProduct,
                element_type,
                header_idx,
                body_idx,
                latch_idx,
                exit_idx,
                iv,
                iv_inc_idx,
                accumulator_phi,
                array_a_gep,
                array_b_gep: Some(array_b_gep),
                accumulator_add_idx,
                limit,
                exit_cmp_inst_idx,
                exit_cmp_dest,
                loop_blocks: loop_info.body.clone(),
            })
        }

        _ => {
            if debug {
                eprintln!("[VEC-RED]   Unsupported accumulator update pattern");
            }
            None
        }
    }
}

/// Check if a GEP uses the induction variable.
fn gep_uses_iv(
    func: &IrFunction,
    loop_blocks: &FxHashSet<usize>,
    gep: Value,
    iv: Value,
    iv_derived: &FxHashSet<Value>,
) -> bool {
    // Find the GEP instruction in the loop
    for &block_idx in loop_blocks {
        let block = &func.blocks[block_idx];
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, offset, .. } = inst {
                if *dest == gep {
                    // Check if offset uses IV or IV-derived value
                    if let Operand::Value(v) = offset {
                        if v == &iv || iv_derived.contains(v) {
                            return true;
                        }
                        // Also check if offset is computed from IV (e.g., iv * 8)
                        // Trace back through multiply/add operations
                        return value_depends_on_iv(func, loop_blocks, *v, iv, iv_derived);
                    }
                }
            }
        }
    }
    false
}

/// Check if a value depends on the IV (transitively through operations)
fn value_depends_on_iv(
    func: &IrFunction,
    loop_blocks: &FxHashSet<usize>,
    val: Value,
    iv: Value,
    iv_derived: &FxHashSet<Value>,
) -> bool {
    if val == iv || iv_derived.contains(&val) {
        return true;
    }

    // Search for the instruction that produces this value
    for &block_idx in loop_blocks {
        let block = &func.blocks[block_idx];
        for inst in &block.instructions {
            if inst.dest() == Some(val) {
                // Check if this instruction uses IV-derived values
                match inst {
                    Instruction::BinOp { lhs, rhs, .. } => {
                        let lhs_uses_iv = if let Operand::Value(v) = lhs {
                            v == &iv || iv_derived.contains(v)
                        } else {
                            false
                        };
                        let rhs_uses_iv = if let Operand::Value(v) = rhs {
                            v == &iv || iv_derived.contains(v)
                        } else {
                            false
                        };
                        return lhs_uses_iv || rhs_uses_iv;
                    }
                    Instruction::Cast { src, .. } => {
                        if let Operand::Value(v) = src {
                            return v == &iv || iv_derived.contains(v);
                        }
                    }
                    Instruction::Copy { dest: _, src } => {
                        if let Operand::Value(v) = src {
                            return v == &iv || iv_derived.contains(v);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    false
}

/// Find exit block (first successor outside the loop).
fn find_exit(func: &IrFunction, loop_info: &loop_analysis::NaturalLoop) -> Option<usize> {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();

    // Build a mapping from BlockId to block index
    let mut label_to_idx = FxHashMap::default();
    for (idx, block) in func.blocks.iter().enumerate() {
        label_to_idx.insert(block.label.0, idx);
    }

    if debug {
        eprintln!("[VEC-RED]   find_exit: loop body contains block indices: {:?}", loop_info.body);
    }

    for &block_idx in &loop_info.body {
        let block = &func.blocks[block_idx];
        if debug {
            eprintln!("[VEC-RED]   Block[{}] (label={}) terminator: {:?}", block_idx, block.label.0, block.terminator);
        }
        match &block.terminator {
            Terminator::CondBranch { true_label, false_label, .. } => {
                // Convert BlockId to block index for comparison
                let true_idx = label_to_idx.get(&true_label.0).copied();
                let false_idx = label_to_idx.get(&false_label.0).copied();

                let then_in_loop = true_idx.map(|idx| loop_info.body.contains(&idx)).unwrap_or(false);
                let else_in_loop = false_idx.map(|idx| loop_info.body.contains(&idx)).unwrap_or(false);

                if debug {
                    eprintln!("[VEC-RED]   Block[{}] CondBranch: true=BlockId({}) [idx={:?}] (in_loop={}), false=BlockId({}) [idx={:?}] (in_loop={})",
                        block_idx, true_label.0, true_idx, then_in_loop, false_label.0, false_idx, else_in_loop);
                }

                if then_in_loop && !else_in_loop {
                    return false_idx;
                } else if !then_in_loop && else_in_loop {
                    return true_idx;
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the latch block (block with backedge to header).
fn find_latch(func: &IrFunction, loop_info: &loop_analysis::NaturalLoop) -> Option<usize> {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();

    // Get the actual BlockId of the header (not just the index)
    let header_label = func.blocks[loop_info.header].label;

    if debug {
        eprintln!("[VEC-RED]   find_latch: looking for backedge to header BlockId({})", header_label.0);
    }

    for &block_idx in &loop_info.body {
        let block = &func.blocks[block_idx];
        if debug {
            eprintln!("[VEC-RED]   Block[{}] (label={}) terminator: {:?}", block_idx, block.label.0, block.terminator);
        }
        match &block.terminator {
            Terminator::Branch(target) if *target == header_label => {
                if debug {
                    eprintln!("[VEC-RED]   Found latch: block[{}]", block_idx);
                }
                return Some(block_idx);
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                if *true_label == header_label || *false_label == header_label {
                    if debug {
                        eprintln!("[VEC-RED]   Found latch: block[{}]", block_idx);
                    }
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

/// Extract base pointers for C and B arrays from the pattern's GEP values.
/// Returns (c_base, a_ptr, b_base) for use in remainder loop.
fn extract_base_pointers(
    func: &IrFunction,
    pattern: &VectorizablePattern,
) -> (Value, Value, Value) {
    let mut c_base = None;
    let mut b_base = None;

    // Scan all loop blocks to find GEP instructions that define c_gep and b_gep
    for &block_idx in &pattern.loop_blocks {
        let block = &func.blocks[block_idx];
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, base, .. } = inst {
                if *dest == pattern.c_gep {
                    c_base = Some(*base);
                }
                if *dest == pattern.b_gep {
                    b_base = Some(*base);
                }
            }
        }
    }

    let c_base = c_base.expect("Could not find C array base pointer");
    let b_base = b_base.expect("Could not find B array base pointer");

    (c_base, pattern.a_ptr, b_base)
}

/// Insert remainder loop to handle N % vec_width elements after vectorized loop.
///
/// Creates new CFG structure:
/// [vec_header] ⇄ [vec_body] → [vec_latch] → [vec_exit]
///                                             ↓
///                                    [remainder_header] ⇄ [remainder_body] → [remainder_latch]
///                                             ↓
///                                          [exit]
///
/// Returns the number of changes made (typically 4 for the 4 new blocks).
fn insert_remainder_loop(
    func: &mut IrFunction,
    pattern: &VectorizablePattern,
    vec_width: usize,
    next_val_id: &mut u32,
    next_label: &mut u32,
) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();

    // Extract base pointers for arrays
    let (c_base, a_ptr, b_base) = extract_base_pointers(func, pattern);

    // Allocate new block IDs
    let vec_exit_label = BlockId(*next_label);
    *next_label += 1;
    let remainder_header_label = BlockId(*next_label);
    *next_label += 1;
    let remainder_body_label = BlockId(*next_label);
    *next_label += 1;
    let remainder_latch_label = BlockId(*next_label);
    *next_label += 1;

    // Allocate new value IDs
    let j_rem_start = Value(*next_val_id);
    *next_val_id += 1;
    let j_rem_iv = Value(*next_val_id);
    *next_val_id += 1;
    let j_rem_iv_next = Value(*next_val_id);
    *next_val_id += 1;
    let j_rem_cmp = Value(*next_val_id);
    *next_val_id += 1;
    let j_rem_cast = Value(*next_val_id);
    *next_val_id += 1;
    let j_rem_offset = Value(*next_val_id);
    *next_val_id += 1;
    let c_rem_gep = Value(*next_val_id);
    *next_val_id += 1;
    let b_rem_gep = Value(*next_val_id);
    *next_val_id += 1;
    let c_rem_load = Value(*next_val_id);
    *next_val_id += 1;
    let a_rem_load = Value(*next_val_id);
    *next_val_id += 1;
    let b_rem_load = Value(*next_val_id);
    *next_val_id += 1;
    let mul_result = Value(*next_val_id);
    *next_val_id += 1;
    let add_result = Value(*next_val_id);
    *next_val_id += 1;

    if debug {
        eprintln!("[VEC] Creating remainder loop blocks...");
        eprintln!("[VEC]   vec_exit (BlockId({}))", vec_exit_label.0);
        eprintln!("[VEC]   remainder_header (BlockId({}))", remainder_header_label.0);
        eprintln!("[VEC]   remainder_body (BlockId({}))", remainder_body_label.0);
        eprintln!("[VEC]   remainder_latch (BlockId({}))", remainder_latch_label.0);
    }

    // Step 1: Modify vectorized header to exit to vec_exit instead of original exit
    // Find the header block and change its false_label (exit branch) to vec_exit
    let header_block = &mut func.blocks[pattern.header_idx];
    if let Terminator::CondBranch { true_label: _, false_label, .. } = &mut header_block.terminator {
        if debug {
            eprintln!("[VEC]   Redirecting header exit {} → {}", false_label, vec_exit_label);
        }
        *false_label = vec_exit_label;
    }

    // Step 2: Create vec_exit block
    // Convert byte-offset IV back to element index for the scalar remainder loop:
    // j_rem_start = byte_off_final / 8 (sizeof(double))
    let vec_exit_block = BasicBlock {
        label: vec_exit_label,
        instructions: vec![
            Instruction::BinOp {
                dest: j_rem_start,
                op: IrBinOp::SDiv,
                lhs: Operand::Value(pattern.iv),  // Final byte offset
                rhs: Operand::Const(IrConst::I32(8)),  // sizeof(double)
                ty: IrType::I32,
            },
        ],
        terminator: Terminator::Branch(remainder_header_label),
        source_spans: vec![],
    };

    // Step 3: Create remainder_header block
    // Phi node for remainder IV and comparison
    let remainder_header_block = BasicBlock {
        label: remainder_header_label,
        instructions: vec![
            Instruction::Phi {
                dest: j_rem_iv,
                ty: IrType::I32,
                incoming: vec![
                    (Operand::Value(j_rem_start), vec_exit_label),
                    (Operand::Value(j_rem_iv_next), remainder_latch_label),
                ],
            },
            Instruction::Cmp {
                dest: j_rem_cmp,
                op: IrCmpOp::Slt,  // Signed less-than
                lhs: Operand::Value(j_rem_iv),
                rhs: pattern.limit,  // Original N
                ty: IrType::I32,
            },
        ],
        terminator: Terminator::CondBranch {
            cond: Operand::Value(j_rem_cmp),
            true_label: remainder_body_label,
            false_label: func.blocks[pattern.exit_idx].label,
        },
        source_spans: vec![],
    };

    // Step 4: Create remainder_body block
    // Scalar FMA: C[i][j] += A[i][k] * B[k][j]
    let remainder_body_block = BasicBlock {
        label: remainder_body_label,
        instructions: vec![
            // Cast j to i64
            Instruction::Cast {
                dest: j_rem_cast,
                src: Operand::Value(j_rem_iv),
                from_ty: IrType::I32,
                to_ty: IrType::I64,
            },
            // Compute byte offset: j * 8
            Instruction::BinOp {
                dest: j_rem_offset,
                op: IrBinOp::Mul,
                lhs: Operand::Value(j_rem_cast),
                rhs: Operand::Const(IrConst::I64(8)),
                ty: IrType::I64,
            },
            // GEP for C[i][j]
            Instruction::GetElementPtr {
                dest: c_rem_gep,
                base: c_base,
                offset: Operand::Value(j_rem_offset),
                ty: IrType::F64,  // Element type, not pointer type
            },
            // GEP for B[k][j]
            Instruction::GetElementPtr {
                dest: b_rem_gep,
                base: b_base,
                offset: Operand::Value(j_rem_offset),
                ty: IrType::F64,  // Element type, not pointer type
            },
            // Load C[i][j]
            Instruction::Load {
                dest: c_rem_load,
                ptr: c_rem_gep,
                ty: IrType::F64,
                seg_override: AddressSpace::Default,
            },
            // Load A[i][k]
            Instruction::Load {
                dest: a_rem_load,
                ptr: a_ptr,
                ty: IrType::F64,
                seg_override: AddressSpace::Default,
            },
            // Load B[k][j]
            Instruction::Load {
                dest: b_rem_load,
                ptr: b_rem_gep,
                ty: IrType::F64,
                seg_override: AddressSpace::Default,
            },
            // Multiply A * B
            Instruction::BinOp {
                dest: mul_result,
                op: IrBinOp::Mul,  // Type-generic multiply, determined by ty field
                lhs: Operand::Value(a_rem_load),
                rhs: Operand::Value(b_rem_load),
                ty: IrType::F64,
            },
            // Add C + (A * B)
            Instruction::BinOp {
                dest: add_result,
                op: IrBinOp::Add,  // Type-generic add, determined by ty field
                lhs: Operand::Value(c_rem_load),
                rhs: Operand::Value(mul_result),
                ty: IrType::F64,
            },
            // Store result back to C[i][j]
            Instruction::Store {
                val: Operand::Value(add_result),
                ptr: c_rem_gep,
                ty: IrType::F64,
                seg_override: AddressSpace::Default,
            },
        ],
        terminator: Terminator::Branch(remainder_latch_label),
        source_spans: vec![],
    };

    // Step 5: Create remainder_latch block
    // Increment j and loop back
    let remainder_latch_block = BasicBlock {
        label: remainder_latch_label,
        instructions: vec![
            Instruction::BinOp {
                dest: j_rem_iv_next,
                op: IrBinOp::Add,
                lhs: Operand::Value(j_rem_iv),
                rhs: Operand::Const(IrConst::I32(1)),
                ty: IrType::I32,
            },
        ],
        terminator: Terminator::Branch(remainder_header_label),
        source_spans: vec![],
    };

    // Step 6: Add all new blocks to the function
    func.blocks.push(vec_exit_block);
    func.blocks.push(remainder_header_block);
    func.blocks.push(remainder_body_block);
    func.blocks.push(remainder_latch_block);

    if debug {
        eprintln!("[VEC] Remainder phi: [(Value({}), BlockId({})), (Value({}), BlockId({}))]",
            j_rem_start.0, vec_exit_label.0, j_rem_iv_next.0, remainder_latch_label.0);
        eprintln!("[VEC] Transformation complete: 4 blocks added");
    }

    4  // 4 new blocks added
}

/// Transform the loop to use FmaF64x2 intrinsics.
fn transform_to_fma_f64x2(func: &mut IrFunction, pattern: &VectorizablePattern) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let mut changes = 0;

    // Keep track of the next available Value and BlockId
    let mut next_val_id = func.next_value_id;
    let mut next_label = func.next_label;

    // Restrict all IV/GEP tracing and modifications to the innermost loop blocks only.
    let innermost_blocks: FxHashSet<usize> = [
        pattern.header_idx, pattern.body_idx, pattern.latch_idx,
    ].iter().copied().collect();

    // Build a set of IV-derived values (for finding all IV-related comparisons)
    // Start with the IV from the header, but also trace back from the GEPs to find
    // the actual j-loop IV
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(pattern.iv);

    // Find the IV used in the B and C GEPs by tracing backwards
    // The GEPs use an offset that's derived from the j-loop IV
    let mut gep_ivs = FxHashSet::default();
    for &block_idx in &innermost_blocks {
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
        for &block_idx in &innermost_blocks {
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
        for &block_idx in &innermost_blocks {
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

        // Modify the exit comparison in the header block only (not outer loop comparisons)
        for block_idx in [pattern.header_idx] {
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
        for &block_idx in &innermost_blocks {
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
            for &block_idx in &innermost_blocks {
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
            for &block_idx in &innermost_blocks {
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
            for &block_idx in &innermost_blocks {
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

        // Find the store instruction by scanning (store_idx may be stale after Step 2 insertions).
        let store_pos = body.instructions.iter().position(|inst| {
            matches!(inst, Instruction::Store { ptr, .. } if *ptr == pattern.c_gep)
        });

        if let Some(pos) = store_pos {
            body.instructions.insert(pos, intrinsic);
            body.instructions.remove(pos + 1);
        } else {
            body.instructions.push(intrinsic);
        }
        changes += 1;
        if debug {
            eprintln!("[VEC]   Inserted FmaF64x2 intrinsic, dest_ptr=Value({}), args=[Value({}), Value({})]",
                pattern.c_gep.0, pattern.a_ptr.0, pattern.b_gep.0);
        }

        // The old load/mul/add instructions are now dead. DCE will clean them up.
    }

    // Step 4: Create remainder loop blocks for N % 2 != 0
    {
        let remainder_changes = insert_remainder_loop(
            func,
            pattern,
            2,  // SSE2 vector width
            &mut next_val_id,
            &mut next_label,
        );
        changes += remainder_changes;
        if debug {
            eprintln!("[VEC]   Added remainder loop: {} blocks created", remainder_changes / 4);
        }
    }

    // Update the function's next_value_id and next_label
    func.next_value_id = next_val_id;
    func.next_label = next_label;

    changes
}

/// Transform loop to use AVX2 FmaF64x4 intrinsic (4-wide, 256-bit).
/// Same pattern as SSE2 but processes 4 elements per iteration instead of 2.
fn transform_to_fma_f64x4(func: &mut IrFunction, pattern: &VectorizablePattern) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let mut changes = 0;

    // Keep track of the next available Value and BlockId
    let mut next_val_id = func.next_value_id;
    let mut next_label = func.next_label;

    // Restrict all IV/GEP tracing and modifications to the innermost loop blocks only.
    // This prevents accidentally modifying comparisons or GEPs in outer loops.
    let innermost_blocks: FxHashSet<usize> = [
        pattern.header_idx, pattern.body_idx, pattern.latch_idx,
    ].iter().copied().collect();

    // Build a set of IV-derived values (for finding all IV-related comparisons)
    // Start with the IV from the header, but also trace back from the GEPs to find
    // the actual j-loop IV
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(pattern.iv);

    // Find the IV used in the B and C GEPs by tracing backwards
    // The GEPs use an offset that's derived from the j-loop IV
    let mut gep_ivs = FxHashSet::default();
    for &block_idx in &innermost_blocks {
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
        for &block_idx in &innermost_blocks {
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
        for &block_idx in &innermost_blocks {
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

    // Step 1: Convert loop IV from element index to byte offset.
    // Instead of j=0..N/4 with GEP offset j*32, use byte_off=0..N*8 step 32.
    // This eliminates the multiply in the inner loop (the biggest single win: -8 instructions).
    //
    // Changes:
    //   - Loop limit: N/4 → N*8 (byte limit)
    //   - IV increment: +1 → +32 (bytes per AVX2 iteration = 4 doubles × 8 bytes)
    //   - GEP offset multiply: eliminated (IV IS the byte offset)
    {
        // First, create the byte limit value (N * 8 bytes per double)
        let byte_limit = match &pattern.limit {
            Operand::Const(IrConst::I32(n)) => Operand::Const(IrConst::I32(*n * 8)),
            Operand::Const(IrConst::I64(n)) => Operand::Const(IrConst::I64(*n * 8)),
            Operand::Value(limit_val) => {
                // Dynamic limit: insert multiply by 8 (sizeof(double)) in header
                let mul_dest = Value(next_val_id);
                next_val_id += 1;

                let limit_ty = match &func.blocks[pattern.header_idx].instructions[pattern.exit_cmp_inst_idx] {
                    Instruction::Cmp { ty, .. } => *ty,
                    _ => IrType::I64,
                };

                let mul_inst = Instruction::BinOp {
                    dest: mul_dest,
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(*limit_val),
                    rhs: Operand::Const(match limit_ty {
                        IrType::I32 => IrConst::I32(8),
                        IrType::I64 => IrConst::I64(8),
                        _ => IrConst::I64(8),
                    }),
                    ty: limit_ty,
                };

                func.blocks[pattern.header_idx].instructions.insert(pattern.exit_cmp_inst_idx, mul_inst);

                if debug {
                    eprintln!("[VEC]   Inserted multiply for dynamic byte limit: N * 8 => Value({})", mul_dest.0);
                }

                Operand::Value(mul_dest)
            }
            _ => {
                if debug {
                    eprintln!("[VEC]   Unsupported limit type");
                }
                return 0;
            }
        };

        // Modify the exit comparison in the header block only (not outer loop comparisons)
        for block_idx in [pattern.header_idx] {
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

                        // Modify the comparison to use byte limit (N * 8)
                        if modifies_lhs {
                            *rhs = byte_limit.clone();
                            changes += 1;
                            if debug {
                                eprintln!("[VEC]   -> Modified comparison RHS to {:?}", byte_limit);
                            }
                        } else if modifies_rhs {
                            *lhs = byte_limit.clone();
                            changes += 1;
                            if debug {
                                eprintln!("[VEC]   -> Modified comparison LHS to {:?}", byte_limit);
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Convert GEP offset to use byte-offset IV directly.
    // The IV now represents a byte offset (0, 32, 64, ...) instead of an element index.
    // Replace IV*8 multiplies with a Copy (the IV IS the byte offset).
    // Also change the IV increment from +1 to +32.
    {
        // 2a: Eliminate the IV * 8 multiply by replacing it with a Copy of the IV.
        // The multiply's source (the IV or a cast of it) already holds the byte offset.
        let mut eliminated_mul = false;
        for &block_idx in &innermost_blocks {
            let block = &mut func.blocks[block_idx];
            for inst in &mut block.instructions {
                if let Instruction::BinOp {
                    dest,
                    op: IrBinOp::Mul,
                    lhs,
                    rhs,
                    ty,
                } = inst
                {
                    let lhs_is_iv = if let Operand::Value(v) = lhs {
                        iv_derived.contains(v) || gep_ivs.contains(v)
                    } else {
                        false
                    };
                    let rhs_is_8 = matches!(rhs, Operand::Const(IrConst::I64(8)) | Operand::Const(IrConst::I32(8)));

                    if lhs_is_iv && rhs_is_8 {
                        // Replace multiply with a Copy: the IV already holds the byte offset
                        let mul_dest = *dest;
                        let src = lhs.clone();
                        *inst = Instruction::Copy {
                            dest: mul_dest,
                            src,
                        };
                        eliminated_mul = true;
                        changes += 1;
                        if debug {
                            eprintln!("[VEC]   Eliminated IV*8 multiply for Value({}) (IV is now byte offset)", mul_dest.0);
                        }
                    }
                }
                // Also handle Shl by 3 (= multiply by 8)
                if let Instruction::BinOp {
                    dest,
                    op: IrBinOp::Shl,
                    lhs,
                    rhs,
                    ty: _,
                } = inst
                {
                    let lhs_is_iv = if let Operand::Value(v) = lhs {
                        iv_derived.contains(v) || gep_ivs.contains(v)
                    } else {
                        false
                    };
                    let rhs_is_3 = matches!(rhs, Operand::Const(IrConst::I64(3)) | Operand::Const(IrConst::I32(3)));

                    if lhs_is_iv && rhs_is_3 {
                        let shl_dest = *dest;
                        let src = lhs.clone();
                        *inst = Instruction::Copy {
                            dest: shl_dest,
                            src,
                        };
                        eliminated_mul = true;
                        changes += 1;
                        if debug {
                            eprintln!("[VEC]   Eliminated IV<<3 shift for Value({}) (IV is now byte offset)", shl_dest.0);
                        }
                    }
                }
            }
        }

        // 2b: Change the IV increment from +1 to +32 (4 doubles × 8 bytes).
        // The IV increment is at pattern.iv_inc_idx in the latch block.
        {
            let latch = &mut func.blocks[pattern.latch_idx];
            if pattern.iv_inc_idx < latch.instructions.len() {
                if let Instruction::BinOp { op: IrBinOp::Add, rhs, .. } = &mut latch.instructions[pattern.iv_inc_idx] {
                    *rhs = match rhs {
                        Operand::Const(IrConst::I32(_)) => Operand::Const(IrConst::I32(32)),
                        _ => Operand::Const(IrConst::I64(32)),
                    };
                    changes += 1;
                    if debug {
                        eprintln!("[VEC]   Changed IV increment from +1 to +32");
                    }
                }
            }
        }

        if debug && !eliminated_mul {
            eprintln!("[VEC]   Warning: Could not find IV*8 multiply to eliminate");
        }
    }

    // Step 3: Replace the body with FmaF64x4.
    // TODO: Hoist A[i][k] broadcast using BroadcastLoadF64 + FmaF64x4Hoisted
    // (infrastructure ready, but the FMA body accesses wrong register for B_ptr
    // when hoisted — the header BroadcastLoadF64 instruction shifts register
    // allocation and the B_gep value ends up sharing a register with the byte
    // offset IV).
    {
        let body = &mut func.blocks[pattern.body_idx];
        let intrinsic = Instruction::Intrinsic {
            dest: None,
            op: IntrinsicOp::FmaF64x4,
            dest_ptr: Some(pattern.c_gep),
            args: vec![
                Operand::Value(pattern.a_ptr),
                Operand::Value(pattern.b_gep),
            ],
        };

        let store_pos = body.instructions.iter().position(|inst| {
            matches!(inst, Instruction::Store { ptr, .. } if *ptr == pattern.c_gep)
        });

        if let Some(pos) = store_pos {
            body.instructions.insert(pos, intrinsic);
            body.instructions.remove(pos + 1);
        } else {
            body.instructions.push(intrinsic);
        }
        changes += 1;
        if debug {
            eprintln!("[VEC]   Inserted FmaF64x4 intrinsic");
        }
    }

    // Step 4: Create remainder loop blocks for N % 4 != 0
    {
        let remainder_changes = insert_remainder_loop(
            func,
            pattern,
            4,  // AVX2 vector width
            &mut next_val_id,
            &mut next_label,
        );
        changes += remainder_changes;
        if debug {
            eprintln!("[VEC]   Added remainder loop: {} blocks created", remainder_changes / 4);
        }
    }

    // Update the function's next_value_id and next_label
    func.next_value_id = next_val_id;
    func.next_label = next_label;

    changes
}

/// Insert remainder loop for reduction patterns.
/// Creates 4 blocks:
/// - vec_exit: Performs horizontal reduction and computes remainder start index
/// - remainder_header: Loop header with IV phi and accumulator phi
/// - remainder_body: Scalar reduction operation
/// - remainder_latch: IV increment
fn insert_reduction_remainder_loop(
    func: &mut IrFunction,
    pattern: &ReductionPattern,
    vec_width: u64,
    horizontal_intrinsic: IntrinsicOp,
    vec_sum_value: Value,  // Accumulated vector SSA value
    next_val_id: &mut u32,
    next_label: &mut u32,
) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();

    // Get element size in bytes
    let element_size = match pattern.element_type {
        IrType::F64 => 8,
        IrType::F32 => 4,
        IrType::I32 => 4,
        IrType::I64 => 8,
        _ => return 0,
    };

    // Extract base pointers for arrays from the GEP instructions
    let array_a_base = {
        let body_block = &func.blocks[pattern.body_idx];
        let mut base = None;
        for inst in &body_block.instructions {
            if let Instruction::GetElementPtr { dest, base: b, .. } = inst {
                if *dest == pattern.array_a_gep {
                    base = Some(*b);
                    break;
                }
            }
        }
        let result = base.unwrap_or(pattern.array_a_gep);
        if std::env::var("LCCC_DEBUG_VECTORIZE").is_ok() {
            eprintln!("[VEC-RED] array_a_base = Value({}), array_a_gep = Value({})",
                result.0, pattern.array_a_gep.0);
        }
        result
    };

    let array_b_base = pattern.array_b_gep.and_then(|gep| {
        let body_block = &func.blocks[pattern.body_idx];
        for inst in &body_block.instructions {
            if let Instruction::GetElementPtr { dest, base, .. } = inst {
                if *dest == gep {
                    return Some(*base);
                }
            }
        }
        Some(gep)
    });

    // Allocate new block IDs
    let vec_exit_label = BlockId(*next_label);
    *next_label += 1;
    let remainder_header_label = BlockId(*next_label);
    *next_label += 1;
    let remainder_body_label = BlockId(*next_label);
    *next_label += 1;
    let remainder_latch_label = BlockId(*next_label);
    *next_label += 1;

    // Allocate new value IDs
    let scalar_sum = Value(*next_val_id);
    *next_val_id += 1;
    let i_rem_start = Value(*next_val_id);
    *next_val_id += 1;
    let i_rem_iv = Value(*next_val_id);
    *next_val_id += 1;
    let i_rem_iv_next = Value(*next_val_id);
    *next_val_id += 1;
    let i_rem_cmp = Value(*next_val_id);
    *next_val_id += 1;
    let i_rem_cast = Value(*next_val_id);
    *next_val_id += 1;
    let offset_a = Value(*next_val_id);
    *next_val_id += 1;
    let gep_rem_a = Value(*next_val_id);
    *next_val_id += 1;
    let load_rem_a = Value(*next_val_id);
    *next_val_id += 1;
    let sum_rem_phi = Value(*next_val_id);
    *next_val_id += 1;
    let sum_rem_next = Value(*next_val_id);
    *next_val_id += 1;

    // Additional values for dot product
    let (offset_b, gep_rem_b, load_rem_b, mul_rem) = if pattern.kind == ReductionKind::DotProduct {
        let vals = (
            Value(*next_val_id),
            Value(*next_val_id + 1),
            Value(*next_val_id + 2),
            Value(*next_val_id + 3),
        );
        *next_val_id += 4;
        vals
    } else {
        (Value(0), Value(0), Value(0), Value(0))
    };

    if debug {
        eprintln!("[VEC-RED] Creating remainder loop blocks...");
        eprintln!("[VEC-RED]   vec_exit (BlockId({}))", vec_exit_label.0);
        eprintln!("[VEC-RED]   remainder_header (BlockId({}))", remainder_header_label.0);
        eprintln!("[VEC-RED]   remainder_body (BlockId({}))", remainder_body_label.0);
        eprintln!("[VEC-RED]   remainder_latch (BlockId({}))", remainder_latch_label.0);
    }

    // Step 1: Redirect vectorized header to vec_exit instead of original exit
    let header_block = &mut func.blocks[pattern.header_idx];
    if let Terminator::CondBranch { false_label, .. } = &mut header_block.terminator {
        if debug {
            eprintln!("[VEC-RED]   Redirecting header exit {} → {}", false_label, vec_exit_label);
        }
        *false_label = vec_exit_label;
    }

    // Step 2: Create vec_exit block
    // Performs horizontal reduction and computes remainder start index

    // Map to register-based horizontal reduction intrinsic
    let vec_horizontal_op = match horizontal_intrinsic {
        IntrinsicOp::HorizontalAddF64x4 => IntrinsicOp::VecHorizontalAddF64x4,
        IntrinsicOp::HorizontalAddF64x2 => IntrinsicOp::VecHorizontalAddF64x2,
        IntrinsicOp::HorizontalAddI32x8 => IntrinsicOp::VecHorizontalAddI32x8,
        IntrinsicOp::HorizontalAddI32x4 => IntrinsicOp::VecHorizontalAddI32x4,
        _ => horizontal_intrinsic,  // Fallback
    };

    let vec_exit_block = BasicBlock {
        label: vec_exit_label,
        instructions: vec![
            // Horizontal reduction: scalar_sum = reduce(vec_accumulator)
            // Read from vector SSA value, reduce to scalar
            Instruction::Intrinsic {
                dest: Some(scalar_sum),
                op: vec_horizontal_op,
                dest_ptr: None,
                args: vec![Operand::Value(vec_sum_value)],
            },
            // Compute starting index for remainder: i_rem_start = iv_final * vec_width
            Instruction::BinOp {
                dest: i_rem_start,
                op: IrBinOp::Mul,
                lhs: Operand::Value(pattern.iv),
                rhs: Operand::Const(IrConst::I32(vec_width as i32)),
                ty: IrType::I32,
            },
        ],
        terminator: Terminator::Branch(remainder_header_label),
        source_spans: vec![],
    };

    // Step 3: Create remainder_header block
    let remainder_header_block = BasicBlock {
        label: remainder_header_label,
        instructions: vec![
            // IV phi
            Instruction::Phi {
                dest: i_rem_iv,
                ty: IrType::I32,
                incoming: vec![
                    (Operand::Value(i_rem_start), vec_exit_label),
                    (Operand::Value(i_rem_iv_next), remainder_latch_label),
                ],
            },
            // Accumulator phi (receives scalar_sum from horizontal reduction!)
            Instruction::Phi {
                dest: sum_rem_phi,
                ty: pattern.element_type,
                incoming: vec![
                    (Operand::Value(scalar_sum), vec_exit_label),
                    (Operand::Value(sum_rem_next), remainder_latch_label),
                ],
            },
            // Comparison
            Instruction::Cmp {
                dest: i_rem_cmp,
                op: IrCmpOp::Slt,
                lhs: Operand::Value(i_rem_iv),
                rhs: pattern.limit.clone(),  // ORIGINAL limit (not divided)
                ty: IrType::I32,
            },
        ],
        terminator: Terminator::CondBranch {
            cond: Operand::Value(i_rem_cmp),
            true_label: remainder_body_label,
            false_label: func.blocks[pattern.exit_idx].label,
        },
        source_spans: vec![],
    };

    // Step 4: Create remainder_body block
    let mut remainder_body_instructions = vec![
        // Cast i to i64
        Instruction::Cast {
            dest: i_rem_cast,
            src: Operand::Value(i_rem_iv),
            from_ty: IrType::I32,
            to_ty: IrType::I64,
        },
        // Compute offset for array A: offset = i * element_size
        Instruction::BinOp {
            dest: offset_a,
            op: IrBinOp::Mul,
            lhs: Operand::Value(i_rem_cast),
            rhs: Operand::Const(IrConst::I64(element_size)),
            ty: IrType::I64,
        },
        // GEP to array_a[i]
        Instruction::GetElementPtr {
            dest: gep_rem_a,
            base: array_a_base,
            offset: Operand::Value(offset_a),
            ty: pattern.element_type,
        },
        // Load array_a[i]
        Instruction::Load {
            dest: load_rem_a,
            ptr: gep_rem_a,
            ty: pattern.element_type,
            seg_override: AddressSpace::Default,
        },
    ];

    // Add pattern-specific operations
    match pattern.kind {
        ReductionKind::Sum => {
            // Simple sum: sum += array_a[i]
            remainder_body_instructions.push(Instruction::BinOp {
                dest: sum_rem_next,
                op: IrBinOp::Add,
                lhs: Operand::Value(sum_rem_phi),
                rhs: Operand::Value(load_rem_a),
                ty: pattern.element_type,
            });
        }
        ReductionKind::DotProduct => {
            // Dot product: sum += a[i] * b[i]
            remainder_body_instructions.extend_from_slice(&[
                // Compute offset for array B (same as A)
                Instruction::BinOp {
                    dest: offset_b,
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(i_rem_cast),
                    rhs: Operand::Const(IrConst::I64(element_size)),
                    ty: IrType::I64,
                },
                // GEP to array_b[i]
                Instruction::GetElementPtr {
                    dest: gep_rem_b,
                    base: array_b_base.unwrap(),
                    offset: Operand::Value(offset_b),
                    ty: pattern.element_type,
                },
                // Load array_b[i]
                Instruction::Load {
                    dest: load_rem_b,
                    ptr: gep_rem_b,
                    ty: pattern.element_type,
                    seg_override: AddressSpace::Default,
                },
                // Multiply a[i] * b[i]
                Instruction::BinOp {
                    dest: mul_rem,
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(load_rem_a),
                    rhs: Operand::Value(load_rem_b),
                    ty: pattern.element_type,
                },
                // Add to accumulator
                Instruction::BinOp {
                    dest: sum_rem_next,
                    op: IrBinOp::Add,
                    lhs: Operand::Value(sum_rem_phi),
                    rhs: Operand::Value(mul_rem),
                    ty: pattern.element_type,
                },
            ]);
        }
    }

    let remainder_body_block = BasicBlock {
        label: remainder_body_label,
        instructions: remainder_body_instructions,
        terminator: Terminator::Branch(remainder_latch_label),
        source_spans: vec![],
    };

    // Step 5: Create remainder_latch block
    let remainder_latch_block = BasicBlock {
        label: remainder_latch_label,
        instructions: vec![Instruction::BinOp {
            dest: i_rem_iv_next,
            op: IrBinOp::Add,
            lhs: Operand::Value(i_rem_iv),
            rhs: Operand::Const(IrConst::I32(1)),
            ty: IrType::I32,
        }],
        terminator: Terminator::Branch(remainder_header_label),
        source_spans: vec![],
    };

    // Step 6: Add all new blocks to the function
    func.blocks.push(vec_exit_block);
    func.blocks.push(remainder_header_block);
    func.blocks.push(remainder_body_block);
    func.blocks.push(remainder_latch_block);

    // Step 7: Replace uses of original scalar accumulator with remainder accumulator
    // After vectorization, pattern.accumulator_phi in the header now holds a VECTOR value.
    // The exit block and any blocks only reachable after the remainder loop should use
    // sum_rem_phi (the scalar accumulator from the remainder loop) instead.
    //
    // We update the exit block since it's now only reachable from remainder_header.
    {
        let exit_block = &mut func.blocks[pattern.exit_idx];
        let mut updates = 0;
        if debug {
            eprintln!("[VEC-RED]   Checking exit block for uses of SSA {} (vector accumulator)", pattern.accumulator_phi.0);
        }

        // Helper to replace SSA values in operands
        let replace_in_operand = |op: &mut Operand, from: u32, to: Value| -> bool {
            if let Operand::Value(v) = op {
                if v.0 == from {
                    *v = to;
                    return true;
                }
            }
            false
        };

        // Update all uses of the accumulator phi in the exit block's instructions.
        // The accumulator may be used in Copy, Store, BinOp, Call args, etc.
        let acc_id = pattern.accumulator_phi.0;
        for inst in &mut exit_block.instructions {
            match inst {
                Instruction::Copy { src, .. } => {
                    if let Operand::Value(v) = src { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                }
                Instruction::Store { val, .. } => {
                    if let Operand::Value(v) = val { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                }
                Instruction::BinOp { lhs, rhs, .. } => {
                    if let Operand::Value(v) = lhs { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                    if let Operand::Value(v) = rhs { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                }
                Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } => {
                    if let Operand::Value(v) = src { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                }
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    for a in &mut info.args {
                        if let Operand::Value(v) = a { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                    }
                }
                Instruction::Phi { incoming, .. } => {
                    for (op, _) in incoming {
                        if let Operand::Value(v) = op { if v.0 == acc_id { *v = sum_rem_phi; updates += 1; } }
                    }
                }
                _ => {}
            }
        }

        // Also update terminator
        match &mut exit_block.terminator {
            Terminator::Return(Some(op)) => {
                if replace_in_operand(op, pattern.accumulator_phi.0, sum_rem_phi) {
                    updates += 1;
                }
            }
            Terminator::CondBranch { cond, .. } => {
                if replace_in_operand(cond, pattern.accumulator_phi.0, sum_rem_phi) {
                    updates += 1;
                }
            }
            _ => {}
        }

        if debug {
            eprintln!("[VEC-RED]   Updated {} uses of accumulator in exit block", updates);
        }
    }

    if debug {
        eprintln!("[VEC-RED] Remainder loop complete: 4 blocks added");
    }

    4  // 4 new blocks added
}

/// Transform reduction loop to use AVX2 256-bit vectorization (4×F64, 8×I32, etc.).
fn transform_reduction_avx2(func: &mut IrFunction, pattern: &ReductionPattern) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let mut changes = 0;

    // Keep track of the next available Value and BlockId
    let mut next_val_id = func.next_value_id;
    let mut next_label = func.next_label;

    // Determine vector width and intrinsics based on element type
    // NOTE: These are only used for pattern matching - the actual transform uses Vec* variants
    let (vec_width, _load_intrinsic, _add_intrinsic, _mul_intrinsic, horizontal_intrinsic) = match pattern.element_type {
        IrType::F64 => (
            4u64,
            IntrinsicOp::LoadF64x4,  // Legacy - not used in register-based transform
            IntrinsicOp::AddF64x4,   // Legacy - not used in register-based transform
            Some(IntrinsicOp::MulF64x4),  // Legacy - not used in register-based transform
            IntrinsicOp::HorizontalAddF64x4,  // Used for remainder loop intrinsic selection
        ),
        IrType::I32 => {
            // I32 multiply intrinsics not yet implemented, only support Sum pattern
            if pattern.kind == ReductionKind::DotProduct {
                if debug {
                    eprintln!("[VEC-RED] I32 dot product not yet supported (missing MulI32x8 intrinsic)");
                }
                return 0;
            }
            (
                8u64,
                IntrinsicOp::LoadI32x8,
                IntrinsicOp::AddI32x8,
                None,  // No multiply for I32 yet
                IntrinsicOp::HorizontalAddI32x8,
            )
        },
        _ => {
            if debug {
                eprintln!("[VEC-RED] Unsupported type for AVX2: {:?}", pattern.element_type);
            }
            return 0;
        }
    };

    if debug {
        eprintln!("[VEC-RED] Transforming reduction to AVX2:");
        eprintln!("[VEC-RED]   Kind: {:?}", pattern.kind);
        eprintln!("[VEC-RED]   Type: {:?}", pattern.element_type);
        eprintln!("[VEC-RED]   Vec width: {}", vec_width);

        eprintln!("[VEC-RED] === LOOP STRUCTURE BEFORE TRANSFORM ===");
        eprintln!("[VEC-RED]   Header: {}, Body: {}, Latch: {}, Exit: {}",
            pattern.header_idx, pattern.body_idx, pattern.latch_idx, pattern.exit_idx);

        // Print header terminator
        eprintln!("[VEC-RED]   Header terminator: {:?}", func.blocks[pattern.header_idx].terminator);

        // Print body terminator
        eprintln!("[VEC-RED]   Body terminator: {:?}", func.blocks[pattern.body_idx].terminator);

        // Print latch terminator
        eprintln!("[VEC-RED]   Latch terminator: {:?}", func.blocks[pattern.latch_idx].terminator);

        // Print IV and limit
        eprintln!("[VEC-RED]   IV: {}, Limit: {:?}", pattern.iv.0, pattern.limit);
    }

    // Build IV-derived values using fixed-point iteration
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(pattern.iv);

    let mut changed = true;
    while changed {
        changed = false;
        for &block_idx in &pattern.loop_blocks {
            let block = &func.blocks[block_idx];
            for inst in &block.instructions {
                match inst {
                    Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                        if let Operand::Value(src_val) = src {
                            if iv_derived.contains(src_val) && iv_derived.insert(*dest) {
                                changed = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Step 1: Divide loop bound by vector width
    let divided_limit = match &pattern.limit {
        Operand::Const(IrConst::I32(n)) => Operand::Const(IrConst::I32(*n / vec_width as i32)),
        Operand::Const(IrConst::I64(n)) => Operand::Const(IrConst::I64(*n / vec_width as i64)),
        Operand::Value(limit_val) => {
            // Dynamic limit: insert division
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
                    IrType::I32 => IrConst::I32(vec_width as i32),
                    IrType::I64 => IrConst::I64(vec_width as i64),
                    _ => IrConst::I64(vec_width as i64),
                }),
                ty: limit_ty,
            };

            // Insert before comparison
            func.blocks[pattern.header_idx].instructions.insert(pattern.exit_cmp_inst_idx, div_inst);
            changes += 1;

            if debug {
                eprintln!("[VEC-RED]   Inserted division for dynamic limit: Value({})", div_dest.0);
            }

            Operand::Value(div_dest)
        }
        _ => {
            if debug {
                eprintln!("[VEC-RED]   Unsupported limit type");
            }
            return 0;
        }
    };

    // Modify all comparisons that use IV-derived values
    for &block_idx in &pattern.loop_blocks {
        let block = &mut func.blocks[block_idx];
        for inst in &mut block.instructions {
            if let Instruction::Cmp { lhs, rhs, op, .. } = inst {
                if debug {
                    eprintln!("[VEC-RED]   CMP before: {:?} {:?} {:?}", lhs, op, rhs);
                }

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

                if modifies_lhs {
                    *rhs = divided_limit.clone();
                    changes += 1;
                    if debug {
                        eprintln!("[VEC-RED]   CMP after:  {:?} {:?} {:?} (modified RHS)", lhs, op, rhs);
                    }
                } else if modifies_rhs {
                    *lhs = divided_limit.clone();
                    changes += 1;
                    if debug {
                        eprintln!("[VEC-RED]   CMP after:  {:?} {:?} {:?} (modified LHS)", lhs, op, rhs);
                    }
                }
            }
        }
    }

    // Step 2: Scale array indexing by vector width
    // Find the GEP offset and multiply it by vec_width
    for &block_idx in &pattern.loop_blocks {
        let block = &func.blocks[block_idx];

        // Find GEP instructions that match our arrays
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::GetElementPtr { dest, offset, .. } = inst {
                // Check if this is one of our array GEPs
                if *dest == pattern.array_a_gep || Some(*dest) == pattern.array_b_gep {
                    if let Operand::Value(offset_val) = offset {
                        // Insert multiply before GEP: offset' = offset * vec_width
                        let mul_dest = Value(next_val_id);
                        next_val_id += 1;

                        let mul_inst = Instruction::BinOp {
                            dest: mul_dest,
                            op: IrBinOp::Mul,
                            lhs: Operand::Value(*offset_val),
                            rhs: Operand::Const(IrConst::I64(vec_width as i64)),
                            ty: IrType::I64,
                        };

                        // Insert mul before GEP
                        let block = &mut func.blocks[block_idx];
                        block.instructions.insert(inst_idx, mul_inst);
                        changes += 1;

                        // Update GEP offset (now at inst_idx + 1)
                        if let Instruction::GetElementPtr { offset, .. } = &mut block.instructions[inst_idx + 1] {
                            *offset = Operand::Value(mul_dest);
                        }

                        if debug {
                            eprintln!("[VEC-RED]   Scaled GEP offset by {}", vec_width);
                        }

                        break; // Only process first GEP per block
                    }
                }
            }
        }
    }

    // Step 3: Transform loop body - register-based vector operations
    // Vector values are SSA values that live in stack slots (backend keeps in registers when possible)
    let vec_sum_value: Value;  // The accumulated vector value

    match pattern.kind {
        ReductionKind::Sum => {
            // Simple sum: sum += arr[i]
            // Register-based flow: vec_zero → vec_load → vec_add → horizontal_add

            // Map to register-based intrinsics based on element type
            let (vec_load_op, vec_add_op, vec_zero_op) = match pattern.element_type {
                IrType::F64 => (
                    IntrinsicOp::VecLoadF64x4,
                    IntrinsicOp::VecAddF64x4,
                    IntrinsicOp::VecZeroF64x4,
                ),
                IrType::I32 => (
                    IntrinsicOp::VecLoadI32x8,
                    IntrinsicOp::VecAddI32x8,
                    IntrinsicOp::VecZeroI32x8,
                ),
                _ => panic!("Unsupported AVX2 element type: {:?}", pattern.element_type),
            };

            if debug {
                eprintln!("[VEC-RED]   Using register-based intrinsics: {:?}, {:?}",
                    vec_load_op, vec_add_op);
            }

            // Value IDs for vector operations (SSA values)
            // IMPORTANT: Create init_zero_value FIRST because it's inserted in entry block (block 0)
            // which comes before the loop body. This ensures SSA IDs are in program order.
            let init_zero_value = Value(next_val_id);
            next_val_id += 1;
            let vec_load = Value(next_val_id);
            next_val_id += 1;
            vec_sum_value = Value(next_val_id);
            next_val_id += 1;

            // Initialize vector accumulator to zero in entry block
            let entry_block = &mut func.blocks[0];
            let zero_inst = Instruction::Intrinsic {
                dest: Some(init_zero_value),
                op: vec_zero_op,
                dest_ptr: None,
                args: vec![],
            };
            entry_block.instructions.push(zero_inst);
            changes += 1;

            // Update the accumulator PHI to use init_zero_value as the entry predecessor
            let header_block = &mut func.blocks[pattern.header_idx];
            for inst in header_block.instructions.iter_mut() {
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    if *dest == pattern.accumulator_phi {
                        for (val, _) in incoming.iter_mut() {
                            if matches!(val, Operand::Const(IrConst::F64(_)) | Operand::Const(IrConst::I32(0))) {
                                *val = Operand::Value(init_zero_value);
                            }
                        }
                    }
                }
            }

            // Get latch label before taking mutable references
            let latch_label = func.blocks[pattern.latch_idx].label;

            // Transform loop body with register-based operations
            {
                let body_block = &mut func.blocks[pattern.body_idx];

                // Vector load from array → SSA value
                let load_inst = Instruction::Intrinsic {
                    dest: Some(vec_load),
                    op: vec_load_op,
                    dest_ptr: None,
                    args: vec![
                        Operand::Value(pattern.array_a_gep),
                        Operand::Const(IrConst::I64(0)),
                    ],
                };

                // Vector add: accumulator + loaded vector → new accumulator
                let add_inst = Instruction::Intrinsic {
                    dest: Some(vec_sum_value),
                    op: vec_add_op,
                    dest_ptr: None,
                    args: vec![
                        Operand::Value(pattern.accumulator_phi),  // Current accumulator (PHI)
                        Operand::Value(vec_load),  // Loaded vector
                    ],
                };

                // Insert vector instructions and remove old scalar add
                body_block.instructions.insert(pattern.accumulator_add_idx, load_inst);
                body_block.instructions.insert(pattern.accumulator_add_idx + 1, add_inst);
                body_block.instructions.remove(pattern.accumulator_add_idx + 2);
                changes += 2;

                // Debug: Log vector accumulator flow
                if debug {
                    eprintln!("[VEC-RED-DEBUG] Vec load SSA value: {}", vec_load.0);
                    eprintln!("[VEC-RED-DEBUG] Vector accumulator SSA value: {}", vec_sum_value.0);
                    eprintln!("[VEC-RED-DEBUG] Accumulator PHI: {}", pattern.accumulator_phi.0);
                    eprintln!("[VEC-RED-DEBUG] Entry init value: {}", init_zero_value.0);
                    eprintln!("[VEC-RED-DEBUG] Vector add op: {:?}", vec_add_op);
                }
            }

            // Update the PHI's backedge to use vec_sum_value
            {
                let header_block = &mut func.blocks[pattern.header_idx];
                for inst in header_block.instructions.iter_mut() {
                    if let Instruction::Phi { dest, incoming, .. } = inst {
                        if *dest == pattern.accumulator_phi {
                            for (val, label) in incoming.iter_mut() {
                                if *label == latch_label {
                                    *val = Operand::Value(vec_sum_value);
                                }
                            }
                        }
                    }
                }
            }

            if debug {
                eprintln!("[VEC-RED]   Transformed sum body: vec_load + vec_add (register-based)");
            }
        }
        ReductionKind::DotProduct => {
            // Dot product: sum += a[i] * b[i]
            // Register-based: vec_load_a, vec_load_b, vec_mul, vec_add (all SSA)

            // Value IDs for vector operations
            let vec_load_a = Value(next_val_id);
            next_val_id += 1;
            let vec_load_b = Value(next_val_id);
            next_val_id += 1;
            let vec_mul = Value(next_val_id);
            next_val_id += 1;
            vec_sum_value = Value(next_val_id);
            next_val_id += 1;

            // Map to register-based intrinsics based on element type
            // Note: Only F64 dot products supported (no I32 multiply intrinsic yet)
            let (vec_load_op, vec_mul_op, vec_add_op, vec_zero_op) = match pattern.element_type {
                IrType::F64 => (
                    IntrinsicOp::VecLoadF64x4,
                    IntrinsicOp::VecMulF64x4,
                    IntrinsicOp::VecAddF64x4,
                    IntrinsicOp::VecZeroF64x4,
                ),
                _ => {
                    if debug {
                        eprintln!("[VEC-RED] Unsupported AVX2 dot product type: {:?}", pattern.element_type);
                    }
                    return changes;
                }
            };

            if debug {
                eprintln!("[VEC-RED]   Using register-based dot product intrinsics");
            }

            // Initialize vector accumulator to zero in entry block
            let entry_block = &mut func.blocks[0];
            let init_zero_value = Value(next_val_id);
            next_val_id += 1;
            let zero_inst = Instruction::Intrinsic {
                dest: Some(init_zero_value),
                op: vec_zero_op,
                dest_ptr: None,
                args: vec![],
            };
            entry_block.instructions.push(zero_inst);
            changes += 1;

            // Update the accumulator PHI to use init_zero_value as the entry predecessor
            let header_block = &mut func.blocks[pattern.header_idx];
            for inst in header_block.instructions.iter_mut() {
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    if *dest == pattern.accumulator_phi {
                        for (val, _) in incoming.iter_mut() {
                            if matches!(val, Operand::Const(IrConst::F64(_)) | Operand::Const(IrConst::I32(0))) {
                                *val = Operand::Value(init_zero_value);
                            }
                        }
                    }
                }
            }

            let body_block = &mut func.blocks[pattern.body_idx];

            // Load vector from array A
            let load_a_inst = Instruction::Intrinsic {
                dest: Some(vec_load_a),
                op: vec_load_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.array_a_gep),
                    Operand::Const(IrConst::I64(0)),
                ],
            };

            // Load vector from array B
            let load_b_inst = Instruction::Intrinsic {
                dest: Some(vec_load_b),
                op: vec_load_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.array_b_gep.unwrap()),
                    Operand::Const(IrConst::I64(0)),
                ],
            };

            // Multiply vectors: A * B
            let mul_inst = Instruction::Intrinsic {
                dest: Some(vec_mul),
                op: vec_mul_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(vec_load_a),
                    Operand::Value(vec_load_b),
                ],
            };

            // Add to accumulator: sum += (A * B)
            let add_inst = Instruction::Intrinsic {
                dest: Some(vec_sum_value),
                op: vec_add_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.accumulator_phi),
                    Operand::Value(vec_mul),
                ],
            };

            // Insert all instructions and remove old scalar operations
            body_block.instructions.insert(pattern.accumulator_add_idx, load_a_inst);
            body_block.instructions.insert(pattern.accumulator_add_idx + 1, load_b_inst);
            body_block.instructions.insert(pattern.accumulator_add_idx + 2, mul_inst);
            body_block.instructions.insert(pattern.accumulator_add_idx + 3, add_inst);

            // Remove old scalar add (and possibly mul) - for now, remove 2 instructions
            body_block.instructions.remove(pattern.accumulator_add_idx + 4);
            body_block.instructions.remove(pattern.accumulator_add_idx + 4);
            changes += 4;

            // Get latch label before taking a mutable reference to header
            let latch_label = func.blocks[pattern.latch_idx].label;

            // Update the PHI's backedge to use vec_sum_value
            let header_block = &mut func.blocks[pattern.header_idx];
            for inst in header_block.instructions.iter_mut() {
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    if *dest == pattern.accumulator_phi {
                        for (val, label) in incoming.iter_mut() {
                            if *label == latch_label {
                                *val = Operand::Value(vec_sum_value);
                            }
                        }
                    }
                }
            }

            if debug {
                eprintln!("[VEC-RED]   Transformed dot product body: load_a + load_b + mul + add");
            }
        }
    }

    // Step 4: Create remainder loop
    let remainder_changes = insert_reduction_remainder_loop(
        func,
        pattern,
        vec_width,
        horizontal_intrinsic,
        vec_sum_value,  // Pass the vector accumulator SSA value
        &mut next_val_id,
        &mut next_label,
    );
    changes += remainder_changes;

    if debug {
        eprintln!("[VEC-RED]   Added remainder loop: {} blocks created", remainder_changes / 4);
    }

    // Update the function's next_value_id and next_label
    func.next_value_id = next_val_id;
    func.next_label = next_label;

    changes
}

/// Transform reduction loop to use SSE2 128-bit vectorization (2×F64, 4×I32, etc.).
fn transform_reduction_sse2(func: &mut IrFunction, pattern: &ReductionPattern) -> usize {
    let debug = std::env::var("LCCC_DEBUG_VECTORIZE").is_ok();
    let mut changes = 0;

    // Keep track of the next available Value and BlockId
    let mut next_val_id = func.next_value_id;
    let mut next_label = func.next_label;

    // Determine vector width and intrinsics based on element type (SSE2 = half of AVX2)
    let (vec_width, load_intrinsic, add_intrinsic, mul_intrinsic, horizontal_intrinsic) = match pattern.element_type {
        IrType::F64 => (
            2u64,
            IntrinsicOp::LoadF64x2,
            IntrinsicOp::AddF64x2,
            Some(IntrinsicOp::MulF64x2),
            IntrinsicOp::HorizontalAddF64x2,
        ),
        IrType::I32 => {
            // I32 multiply intrinsics not yet implemented, only support Sum pattern
            if pattern.kind == ReductionKind::DotProduct {
                if debug {
                    eprintln!("[VEC-RED] I32 dot product not yet supported (missing MulI32x4 intrinsic)");
                }
                return 0;
            }
            (
                4u64,
                IntrinsicOp::LoadI32x4,
                IntrinsicOp::AddI32x4,
                None,  // No multiply for I32 yet
                IntrinsicOp::HorizontalAddI32x4,
            )
        },
        _ => {
            if debug {
                eprintln!("[VEC-RED] Unsupported type for SSE2: {:?}", pattern.element_type);
            }
            return 0;
        }
    };

    if debug {
        eprintln!("[VEC-RED] Transforming reduction to SSE2:");
        eprintln!("[VEC-RED]   Kind: {:?}", pattern.kind);
        eprintln!("[VEC-RED]   Type: {:?}", pattern.element_type);
        eprintln!("[VEC-RED]   Vec width: {}", vec_width);
    }

    // Build IV-derived values using fixed-point iteration
    let mut iv_derived = FxHashSet::default();
    iv_derived.insert(pattern.iv);

    let mut changed = true;
    while changed {
        changed = false;
        for &block_idx in &pattern.loop_blocks {
            let block = &func.blocks[block_idx];
            for inst in &block.instructions {
                match inst {
                    Instruction::Cast { dest, src, .. } | Instruction::Copy { dest, src } => {
                        if let Operand::Value(src_val) = src {
                            if iv_derived.contains(src_val) && iv_derived.insert(*dest) {
                                changed = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Step 1: Divide loop bound by vector width
    let divided_limit = match &pattern.limit {
        Operand::Const(IrConst::I32(n)) => Operand::Const(IrConst::I32(*n / vec_width as i32)),
        Operand::Const(IrConst::I64(n)) => Operand::Const(IrConst::I64(*n / vec_width as i64)),
        Operand::Value(limit_val) => {
            // Dynamic limit: insert division
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
                    IrType::I32 => IrConst::I32(vec_width as i32),
                    IrType::I64 => IrConst::I64(vec_width as i64),
                    _ => IrConst::I64(vec_width as i64),
                }),
                ty: limit_ty,
            };

            // Insert before comparison
            func.blocks[pattern.header_idx].instructions.insert(pattern.exit_cmp_inst_idx, div_inst);
            changes += 1;

            if debug {
                eprintln!("[VEC-RED]   Inserted division for dynamic limit: Value({})", div_dest.0);
            }

            Operand::Value(div_dest)
        }
        _ => {
            if debug {
                eprintln!("[VEC-RED]   Unsupported limit type");
            }
            return 0;
        }
    };

    // Modify all comparisons that use IV-derived values
    for &block_idx in &pattern.loop_blocks {
        let block = &mut func.blocks[block_idx];
        for inst in &mut block.instructions {
            if let Instruction::Cmp { lhs, rhs, op, .. } = inst {
                if debug {
                    eprintln!("[VEC-RED]   CMP before: {:?} {:?} {:?}", lhs, op, rhs);
                }

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

                if modifies_lhs {
                    *rhs = divided_limit.clone();
                    changes += 1;
                    if debug {
                        eprintln!("[VEC-RED]   CMP after:  {:?} {:?} {:?} (modified RHS)", lhs, op, rhs);
                    }
                } else if modifies_rhs {
                    *lhs = divided_limit.clone();
                    changes += 1;
                    if debug {
                        eprintln!("[VEC-RED]   CMP after:  {:?} {:?} {:?} (modified LHS)", lhs, op, rhs);
                    }
                }
            }
        }
    }

    // Step 2: Scale array indexing by vector width
    for &block_idx in &pattern.loop_blocks {
        let block = &func.blocks[block_idx];

        // Find GEP instructions that match our arrays
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::GetElementPtr { dest, offset, .. } = inst {
                // Check if this is one of our array GEPs
                if *dest == pattern.array_a_gep || Some(*dest) == pattern.array_b_gep {
                    if let Operand::Value(offset_val) = offset {
                        // Insert multiply before GEP: offset' = offset * vec_width
                        let mul_dest = Value(next_val_id);
                        next_val_id += 1;

                        let mul_inst = Instruction::BinOp {
                            dest: mul_dest,
                            op: IrBinOp::Mul,
                            lhs: Operand::Value(*offset_val),
                            rhs: Operand::Const(IrConst::I64(vec_width as i64)),
                            ty: IrType::I64,
                        };

                        // Insert mul before GEP
                        let block = &mut func.blocks[block_idx];
                        block.instructions.insert(inst_idx, mul_inst);
                        changes += 1;

                        // Update GEP offset (now at inst_idx + 1)
                        if let Instruction::GetElementPtr { offset, .. } = &mut block.instructions[inst_idx + 1] {
                            *offset = Operand::Value(mul_dest);
                        }

                        if debug {
                            eprintln!("[VEC-RED]   Scaled GEP offset by {}", vec_width);
                        }

                        break; // Only process first GEP per block
                    }
                }
            }
        }
    }

    // Step 3: Transform loop body - replace scalar operations with vector intrinsics
    // Use register-based SSA values for vector operations (no stack allocations)
    let vec_sum_value: Value;

    match pattern.kind {
        ReductionKind::Sum => {
            // Simple sum: sum += arr[i]
            // Map old intrinsics to new register-based intrinsics
            let (vec_load_op, vec_add_op, vec_zero_op) = match load_intrinsic {
                IntrinsicOp::LoadF64x2 => (
                    IntrinsicOp::VecLoadF64x2,
                    IntrinsicOp::VecAddF64x2,
                    IntrinsicOp::VecZeroF64x2,
                ),
                IntrinsicOp::LoadI32x4 => (
                    IntrinsicOp::VecLoadI32x4,
                    IntrinsicOp::VecAddI32x4,
                    IntrinsicOp::VecZeroI32x4,
                ),
                _ => panic!("Unsupported SSE2 load intrinsic: {:?}", load_intrinsic),
            };

            // Create SSA values for vector operations
            // IMPORTANT: Create init_zero_value FIRST (appears in entry block)
            let init_zero_value = Value(next_val_id);
            next_val_id += 1;
            let vec_load = Value(next_val_id);
            next_val_id += 1;
            vec_sum_value = Value(next_val_id);
            next_val_id += 1;

            // Initialize vector accumulator to zero in entry block
            let entry_block = &mut func.blocks[0];
            let zero_inst = Instruction::Intrinsic {
                dest: Some(init_zero_value),
                op: vec_zero_op,
                dest_ptr: None,
                args: vec![],
            };
            entry_block.instructions.push(zero_inst);
            changes += 1;

            if debug {
                eprintln!("[VEC-RED]   Created SSA values: vec_load={}, vec_sum={}", vec_load.0, vec_sum_value.0);
            }

            // Update the accumulator PHI to use init_zero_value as the entry predecessor
            let header_block = &mut func.blocks[pattern.header_idx];
            for inst in header_block.instructions.iter_mut() {
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    if *dest == pattern.accumulator_phi {
                        for (val, _) in incoming.iter_mut() {
                            if matches!(val, Operand::Const(IrConst::F64(_)) | Operand::Const(IrConst::I32(0))) {
                                *val = Operand::Value(init_zero_value);
                            }
                        }
                    }
                }
            }

            // Get latch label before taking mutable references
            let latch_label = func.blocks[pattern.latch_idx].label;

            // Create vector load (returns SSA value)
            let load_inst = Instruction::Intrinsic {
                dest: Some(vec_load),
                op: vec_load_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.array_a_gep),
                    Operand::Const(IrConst::I64(0)),
                ],
            };

            // Create vector add (accumulate) - reads from PHI, produces new value
            let add_inst = Instruction::Intrinsic {
                dest: Some(vec_sum_value),
                op: vec_add_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.accumulator_phi),
                    Operand::Value(vec_load),
                ],
            };

            // Insert instructions and remove old scalar add
            {
                let body_block = &mut func.blocks[pattern.body_idx];
                body_block.instructions.insert(pattern.accumulator_add_idx, load_inst);
                body_block.instructions.insert(pattern.accumulator_add_idx + 1, add_inst);
                body_block.instructions.remove(pattern.accumulator_add_idx + 2);
                changes += 2;

                // Debug: Log vector accumulator flow
                if debug {
                    eprintln!("[VEC-RED-DEBUG-SSE2] Vector accumulator SSA value: {}", vec_sum_value.0);
                    eprintln!("[VEC-RED-DEBUG-SSE2] Accumulator PHI: {}", pattern.accumulator_phi.0);
                    eprintln!("[VEC-RED-DEBUG-SSE2] Entry init value: {}", init_zero_value.0);
                    eprintln!("[VEC-RED-DEBUG-SSE2] Vector add op: {:?}", vec_add_op);
                }
            }

            // Update the PHI's backedge to use vec_sum_value
            {
                let header_block = &mut func.blocks[pattern.header_idx];
                for inst in header_block.instructions.iter_mut() {
                    if let Instruction::Phi { dest, incoming, .. } = inst {
                        if *dest == pattern.accumulator_phi {
                            for (val, label) in incoming.iter_mut() {
                                if *label == latch_label {
                                    *val = Operand::Value(vec_sum_value);
                                }
                            }
                        }
                    }
                }
            }

            if debug {
                eprintln!("[VEC-RED]   Transformed sum body: load + add (register-based)");
            }
        }
        ReductionKind::DotProduct => {
            // Dot product: sum += a[i] * b[i]
            // Note: Only F64 dot products are supported (no integer multiply intrinsic yet)
            // Create SSA values for vector operations
            let vec_load_a = Value(next_val_id);
            next_val_id += 1;
            let vec_load_b = Value(next_val_id);
            next_val_id += 1;
            let vec_mul = Value(next_val_id);
            next_val_id += 1;
            vec_sum_value = Value(next_val_id);
            next_val_id += 1;

            // Map old intrinsics to new register-based intrinsics (only F64 supported)
            let (vec_load_op, vec_mul_op, vec_add_op, vec_zero_op) = match load_intrinsic {
                IntrinsicOp::LoadF64x2 => (
                    IntrinsicOp::VecLoadF64x2,
                    IntrinsicOp::VecMulF64x2,
                    IntrinsicOp::VecAddF64x2,
                    IntrinsicOp::VecZeroF64x2,
                ),
                _ => {
                    if debug {
                        eprintln!("[VEC-RED] Unsupported SSE2 dot product type: {:?}", load_intrinsic);
                    }
                    return changes;
                }
            };

            // Initialize vector accumulator to zero in entry block
            let entry_block = &mut func.blocks[0];
            let init_zero_value = Value(next_val_id);
            next_val_id += 1;
            let zero_inst = Instruction::Intrinsic {
                dest: Some(init_zero_value),
                op: vec_zero_op,
                dest_ptr: None,
                args: vec![],
            };
            entry_block.instructions.push(zero_inst);
            changes += 1;

            if debug {
                eprintln!("[VEC-RED]   Created SSA values: vec_load_a={}, vec_load_b={}, vec_mul={}, vec_sum={}",
                    vec_load_a.0, vec_load_b.0, vec_mul.0, vec_sum_value.0);
            }

            // Update the accumulator PHI to use init_zero_value as the entry predecessor
            let header_block = &mut func.blocks[pattern.header_idx];
            for inst in header_block.instructions.iter_mut() {
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    if *dest == pattern.accumulator_phi {
                        for (val, _) in incoming.iter_mut() {
                            if matches!(val, Operand::Const(IrConst::F64(_)) | Operand::Const(IrConst::I32(0))) {
                                *val = Operand::Value(init_zero_value);
                            }
                        }
                    }
                }
            }

            // Get latch label before taking mutable references
            let latch_label = func.blocks[pattern.latch_idx].label;

            // Create vector loads (return SSA values)
            let load_a_inst = Instruction::Intrinsic {
                dest: Some(vec_load_a),
                op: vec_load_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.array_a_gep),
                    Operand::Const(IrConst::I64(0)),
                ],
            };

            let load_b_inst = Instruction::Intrinsic {
                dest: Some(vec_load_b),
                op: vec_load_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.array_b_gep.unwrap()),
                    Operand::Const(IrConst::I64(0)),
                ],
            };

            // Create vector multiply (element-wise)
            let mul_inst = Instruction::Intrinsic {
                dest: Some(vec_mul),
                op: vec_mul_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(vec_load_a),
                    Operand::Value(vec_load_b),
                ],
            };

            // Create vector add (accumulate)
            let add_inst = Instruction::Intrinsic {
                dest: Some(vec_sum_value),
                op: vec_add_op,
                dest_ptr: None,
                args: vec![
                    Operand::Value(pattern.accumulator_phi),
                    Operand::Value(vec_mul),
                ],
            };

            // Insert all instructions and remove old scalar operations
            {
                let body_block = &mut func.blocks[pattern.body_idx];
                body_block.instructions.insert(pattern.accumulator_add_idx, load_a_inst);
                body_block.instructions.insert(pattern.accumulator_add_idx + 1, load_b_inst);
                body_block.instructions.insert(pattern.accumulator_add_idx + 2, mul_inst);
                body_block.instructions.insert(pattern.accumulator_add_idx + 3, add_inst);

                // Remove old scalar add (and possibly mul) - for now, remove 2 instructions
                body_block.instructions.remove(pattern.accumulator_add_idx + 4);
                body_block.instructions.remove(pattern.accumulator_add_idx + 4);
                changes += 4;
            }

            // Update the PHI's backedge to use vec_sum_value
            {
                let header_block = &mut func.blocks[pattern.header_idx];
                for inst in header_block.instructions.iter_mut() {
                    if let Instruction::Phi { dest, incoming, .. } = inst {
                        if *dest == pattern.accumulator_phi {
                            for (val, label) in incoming.iter_mut() {
                                if *label == latch_label {
                                    *val = Operand::Value(vec_sum_value);
                                }
                            }
                        }
                    }
                }
            }

            if debug {
                eprintln!("[VEC-RED]   Transformed dot product body: load_a + load_b + mul + add (register-based)");
            }
        }
    }

    // Step 4: Create remainder loop
    let remainder_changes = insert_reduction_remainder_loop(
        func,
        pattern,
        vec_width,
        horizontal_intrinsic,
        vec_sum_value,  // Pass the vector accumulator SSA value
        &mut next_val_id,
        &mut next_label,
    );
    changes += remainder_changes;

    if debug {
        eprintln!("[VEC-RED]   Added remainder loop: {} blocks created", remainder_changes / 4);
    }

    // Update the function's next_value_id and next_label
    func.next_value_id = next_val_id;
    func.next_label = next_label;

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
