//! Linear scan register allocator.
//!
//! Assigns physical registers to IR values based on their live intervals.
//! Values with the longest live ranges and most uses get priority for register
//! assignment. Values that don't fit in available registers remain on the stack.
//!
//! Three-phase allocation:
//! 1. **Callee-saved registers** (x86: rbx, r12-r15; ARM: x20-x28; RISC-V: s1, s7-s11):
//!    Assigned to values whose live ranges span function calls. These registers
//!    are preserved across calls by the ABI, so no save/restore is needed at call
//!    sites (but prologue/epilogue must save them).
//!
//! 2. **Caller-saved registers** (x86: r11, r10, r8, r9; ARM: x13, x14):
//!    Assigned to values whose live ranges do NOT span any function call. These
//!    registers are destroyed by calls, so they can only hold values between calls.
//!    No prologue/epilogue save/restore is needed since we never assign them to
//!    values that cross call boundaries.
//!
//! 3. **Callee-saved spillover**: After phases 1 and 2, any remaining callee-saved
//!    registers are assigned to the highest-priority non-call-spanning values that
//!    didn't fit in the caller-saved pool. This is critical for call-free hot loops
//!    (e.g., hash functions, matrix multiply, sorting) where all values compete for
//!    only a few caller-saved registers. The one-time prologue/epilogue save/restore
//!    cost is amortized over many loop iterations.

use super::live_range::{self, LinearScanAllocator};
use super::liveness::{
    compute_live_intervals, for_each_operand_in_instruction, for_each_operand_in_terminator,
    LiveInterval, LivenessResult,
};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::reexports::{Instruction, IrConst, IrFunction, Operand, Terminator};

/// A physical register assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysReg(pub u8);

/// Result of register allocation for a function.
pub struct RegAllocResult {
    /// Map from value ID -> assigned physical register.
    pub assignments: FxHashMap<u32, PhysReg>,
    /// Set of physical registers actually used (for prologue/epilogue save/restore).
    pub used_regs: Vec<PhysReg>,
    /// The liveness analysis computed during register allocation, if any.
    /// Cached here so that calculate_stack_space_common can reuse it for
    /// Tier 2 liveness-based stack slot packing, avoiding a redundant
    /// O(blocks * values * iterations) dataflow computation.
    /// None when no registers were available (empty available_regs).
    pub liveness: Option<super::liveness::LivenessResult>,
}

/// Configuration for the register allocator.
pub struct RegAllocConfig {
    /// Available callee-saved registers for allocation (e.g., s1-s11 for RISC-V).
    pub available_regs: Vec<PhysReg>,
    /// Available caller-saved registers for allocation.
    /// These are assigned to values whose live ranges do NOT span any call.
    /// Since they don't cross calls, no prologue/epilogue save/restore is needed.
    /// Examples: x86 r11, r10, r8, r9.
    pub caller_saved_regs: Vec<PhysReg>,
    /// Whether to allow inline asm operands to be register-allocated.
    /// Only enable this when the backend's asm emitter checks reg_assignments
    /// before falling back to stack access. Currently only RISC-V does this.
    pub allow_inline_asm_regalloc: bool,
    /// Available XMM registers for F64 allocation (caller-saved, non-call-spanning).
    /// Examples: x86 xmm2-xmm7 (PhysReg 20-25).
    pub xmm_regs: Vec<PhysReg>,
}

/// Filter live intervals to only those eligible for register allocation,
/// using the same whitelist + ineligibility rules as the three-phase allocator.
fn filter_eligible_intervals(
    liveness: &LivenessResult,
    eligible: &FxHashSet<u32>,
) -> Vec<LiveInterval> {
    liveness
        .intervals
        .iter()
        .filter(|iv| eligible.contains(&iv.value_id))
        .filter(|iv| iv.end > iv.start)
        .copied()
        .collect()
}

/// Run the register allocator on a function.
///
/// Strategy: We assign callee-saved registers to values with the longest
/// live intervals. This is a simplified linear scan that doesn't split
/// intervals — values either get a register for their entire lifetime or
/// remain on the stack.
///
/// We avoid allocating registers to:
/// - Alloca values (they represent stack addresses)
/// - i128/float values (they need special register paths)
/// - Values used only once right after definition (no benefit from register)
pub fn allocate_registers(func: &IrFunction, config: &RegAllocConfig) -> RegAllocResult {
    if config.available_regs.is_empty() && config.caller_saved_regs.is_empty() {
        return RegAllocResult {
            assignments: FxHashMap::default(),
            used_regs: Vec::new(),
            liveness: None,
        };
    }

    // Note: Register allocation is now enabled for functions with atomics.
    // Atomic operations in all backends (x86, ARM, RISC-V) access their operands
    // exclusively through regalloc-aware helpers (operand_to_rax/x0/t0 and
    // store_rax_to/x0_to/t0_to), so register-allocated values work correctly.
    // The atomic pointer operands are individually excluded from register
    // allocation eligibility below since they need stable stack addresses
    // for the memory access instructions.

    // On 32-bit targets, I64/U64 values need two registers (eax:edx) and cannot
    // be allocated to a single callee-saved register. Exclude them from eligibility.
    let is_32bit = crate::common::types::target_is_32bit();

    // Liveness analysis now uses backward dataflow iteration to correctly
    // handle loops (values live across back-edges have their intervals extended).
    let liveness = compute_live_intervals(func);

    // Count uses per value for prioritization, weighted by loop depth.
    //
    // Uses inside loops are weighted more heavily because they execute more
    // frequently. A use inside a loop at depth D contributes 10^D to the
    // weighted use count (so a use in a singly-nested loop counts 10x, doubly-
    // nested counts 100x, etc.). This ensures inner-loop temporaries get
    // priority for register allocation over values in straight-line code,
    // which is critical for performance in compute-heavy loops like zlib's
    // deflate_slow, longest_match, and slide_hash.
    let mut use_count: FxHashMap<u32, u64> = FxHashMap::default();

    // Precompute per-block loop weight: 10^depth, capped to avoid overflow.
    let block_loop_weight: Vec<u64> = liveness
        .block_loop_depth
        .iter()
        .map(|&d| {
            match d {
                0 => 1,
                1 => 10,
                2 => 100,
                3 => 1000,
                _ => 10_000, // cap at 10K for very deep nesting
            }
        })
        .collect();

    // Collect values whose types don't fit in a single GPR.
    let non_gpr_values = collect_non_gpr_values(func, is_32bit);

    // Helper closure to check if a type is unsuitable for GPR allocation
    let is_non_gpr_type = |ty: &IrType| -> bool {
        ty.is_float()
            || ty.is_long_double()
            || matches!(ty, IrType::I128 | IrType::U128)
            || (is_32bit && matches!(ty, IrType::I64 | IrType::U64))
    };

    // Use a whitelist approach: only allocate registers for values produced
    // by simple, well-understood instructions that store results via the
    // standard accumulator path (e.g., store_rax_to on x86, store_t0_to on RISC-V).
    let mut eligible: FxHashSet<u32> = FxHashSet::default();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        // Get the loop weight for this block (default 1 if no loop info available).
        let weight: u64 = if block_idx < block_loop_weight.len() {
            block_loop_weight[block_idx]
        } else {
            1
        };

        for inst in &block.instructions {
            // Values eligible for register allocation: those stored via the
            // standard accumulator path (store_rax_to on x86, store_t0_to on RISC-V).
            // Exclude float and i128 types since they use different register paths.
            match inst {
                Instruction::BinOp { dest, ty, .. } | Instruction::UnaryOp { dest, ty, .. } => {
                    if !is_non_gpr_type(ty) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::Cmp { dest, .. } => {
                    eligible.insert(dest.0);
                }
                Instruction::Cast {
                    dest,
                    to_ty,
                    from_ty,
                    ..
                } => {
                    if !is_non_gpr_type(to_ty) && !is_non_gpr_type(from_ty) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::Load { dest, ty, .. } => {
                    if !is_non_gpr_type(ty) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::GetElementPtr { dest, .. } => {
                    eligible.insert(dest.0);
                }
                Instruction::Copy { dest, src: _ } => {
                    // Copy instructions are eligible unless the source produces a
                    // non-GPR value (float, i128, or i64 on 32-bit). We check both
                    // constant types and propagated non-GPR status from Value sources.
                    if !non_gpr_values.contains(&dest.0) {
                        eligible.insert(dest.0);
                    }
                }
                // Call results are eligible for callee-saved register allocation.
                // The result arrives in the accumulator (rax on x86, x0 on ARM, a0 on
                // RISC-V), and emit_call_store_result calls emit_store_result which
                // uses store_rax_to/store_t0_to — both of which are register-aware
                // and will emit a reg-to-reg move (e.g., movq %rax, %rbx) instead of
                // a stack spill.
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    if let Some(dest) = info.dest {
                        if !is_non_gpr_type(&info.return_type) {
                            eligible.insert(dest.0);
                        }
                    }
                }
                Instruction::Select { dest, ty, .. } => {
                    if !is_non_gpr_type(ty) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::GlobalAddr { dest, .. } | Instruction::LabelAddr { dest, .. } => {
                    eligible.insert(dest.0);
                }
                // Atomic operations store their results via store_rax_to/store_t0_to.
                Instruction::AtomicLoad { dest, ty, .. }
                | Instruction::AtomicRmw { dest, ty, .. }
                | Instruction::AtomicCmpxchg { dest, ty, .. } => {
                    if !is_non_gpr_type(ty) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::ParamRef { dest, ty, .. } => {
                    if !is_non_gpr_type(ty) {
                        eligible.insert(dest.0);
                    }
                }
                _ => {}
            }

            // Count uses of operands, weighted by loop depth of the containing block.
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    *use_count.entry(v.0).or_insert(0) += weight;
                }
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                *use_count.entry(v.0).or_insert(0) += weight;
            }
        });
    }

    // Exclude values used as pointers in instructions whose codegen paths use
    // resolve_slot_addr() directly (not register-aware).
    remove_ineligible_operands(func, &mut eligible, config);

    // --- 3-channel multiply ILP ---
    //
    // For loops with many multiply-accumulate patterns (a += b*c), we want 3
    // independent multiply chains to fully utilize the CPU's multiply port
    // (which has 3-cycle latency but 1-cycle throughput). The linear scan
    // naturally provides 2 temp registers via rotation. By excluding every
    // 3rd fusible multiply temp from allocation, it falls through to the
    // accumulator path (%eax) in the codegen, creating a 3rd channel.
    //
    // Pattern: r12, rbx, %eax, r12, rbx, %eax, ...
    exclude_every_third_mul_temp(func, &mut eligible);

    // --- Phi register coalescing ---
    //
    // For loop-carried phi variables, the backedge source value (the new value
    // computed in the loop body) should share the same register as the phi dest
    // (the value at the loop header). This eliminates the register-to-register
    // or register-to-stack copy at the backedge.
    //
    // We detect backedge Copy instructions where the dest is a multi-def value
    // (phi dest after phi elimination) and the source is a loop-local value.
    // The backedge source is removed from the eligible set so it doesn't get
    // allocated independently. After allocation, it inherits the phi dest's
    // register assignment.
    let phi_coalesce = if std::env::var("CCC_NO_PHI_COALESCE").is_ok() {
        Vec::new()
    } else {
        detect_phi_coalesce_groups(func, &liveness)
    };
    for &(_phi_dest, backedge_src) in &phi_coalesce {
        // Remove backedge source from eligibility — it will inherit the phi dest's register.
        eligible.remove(&backedge_src);
    }

    // --- Linear scan allocation (replaces three-phase greedy allocator) ---
    //
    // Phase 1: callee-saved registers for ALL eligible values.
    //   Callee-saved regs are safe across calls, so they can hold any value.
    //   Linear scan gives better coverage than the old greedy approach by
    //   considering interval overlap rather than just "does it span a call".
    //
    // Phase 2: caller-saved registers for eligible, non-call-spanning values
    //   that weren't allocated in Phase 1. Caller-saved regs are destroyed by
    //   calls so they can only hold values that don't cross call boundaries.

    let call_points = &liveness.call_points;

    // Phase 1: callee-saved linear scan.
    let phase1_intervals = filter_eligible_intervals(&liveness, &eligible);
    let phase1_ranges =
        live_range::build_live_ranges(&phase1_intervals, &liveness.block_loop_depth, func);
    let mut allocator = LinearScanAllocator::new(phase1_ranges, config.available_regs.clone());
    allocator.run();

    let mut assignments = allocator.assignments;
    let mut used_regs_set: FxHashSet<u8> = FxHashSet::default();
    for &reg in assignments.values() {
        used_regs_set.insert(reg.0);
    }

    // Phase 2: caller-saved linear scan for unallocated non-call-spanning values.
    if !config.caller_saved_regs.is_empty() {
        let phase2_intervals: Vec<LiveInterval> = liveness
            .intervals
            .iter()
            .filter(|iv| eligible.contains(&iv.value_id))
            .filter(|iv| iv.end > iv.start)
            .filter(|iv| !assignments.contains_key(&iv.value_id))
            .filter(|iv| !spans_any_call(iv, call_points))
            .copied()
            .collect();

        if !phase2_intervals.is_empty() {
            let phase2_ranges = live_range::build_live_ranges(
                &phase2_intervals,
                &liveness.block_loop_depth,
                func,
            );
            let mut caller_allocator =
                LinearScanAllocator::new(phase2_ranges, config.caller_saved_regs.clone());
            caller_allocator.run();

            for (vid, reg) in caller_allocator.assignments {
                assignments.insert(vid, reg);
                used_regs_set.insert(reg.0);
            }
        }
    }

    // Propagate phi coalesce assignments: backedge source values inherit
    // the register of their phi dest. This makes the backedge Copy a no-op
    // when both values share the same register.
    for &(phi_dest, backedge_src) in &phi_coalesce {
        if let Some(&reg) = assignments.get(&phi_dest) {
            assignments.insert(backedge_src, reg);
            // No need to add to used_regs_set — the phi dest already did.
        }
    }

    // Phase 3: XMM register allocation for F64 values that don't span calls.
    // These values were excluded from GPR allocation but can use XMM registers.
    if !config.xmm_regs.is_empty() {
        // Collect F64 values: values in non_gpr_values that are F64 typed,
        // haven't been assigned a GPR, and don't span calls.
        let f64_intervals: Vec<LiveInterval> = liveness
            .intervals
            .iter()
            .filter(|iv| non_gpr_values.contains(&iv.value_id))
            .filter(|iv| iv.end > iv.start)
            .filter(|iv| !assignments.contains_key(&iv.value_id))
            .filter(|iv| !spans_any_call(iv, call_points))
            // Only include values that are actually F64 (not i128, not f32, etc.)
            .filter(|iv| {
                // Check if this value is produced by a F64-typed instruction
                func.blocks.iter().any(|block| {
                    block.instructions.iter().any(|inst| {
                        match inst {
                            Instruction::BinOp { dest, ty, .. }
                            | Instruction::UnaryOp { dest, ty, .. } if *ty == IrType::F64 => dest.0 == iv.value_id,
                            Instruction::Load { dest, ty, .. } if *ty == IrType::F64 => dest.0 == iv.value_id,
                            Instruction::Cast { dest, to_ty, .. } if *to_ty == IrType::F64 => dest.0 == iv.value_id,
                            _ => false,
                        }
                    })
                })
            })
            .copied()
            .collect();

        if !f64_intervals.is_empty() {
            let f64_ranges = live_range::build_live_ranges(
                &f64_intervals,
                &liveness.block_loop_depth,
                func,
            );
            let mut xmm_allocator =
                LinearScanAllocator::new(f64_ranges, config.xmm_regs.clone());
            xmm_allocator.run();

            for (vid, reg) in xmm_allocator.assignments {
                assignments.insert(vid, reg);
                // XMM regs (20+) are caller-saved, no prologue save needed
            }
        }
    }

    let mut used_regs: Vec<PhysReg> = used_regs_set.iter().map(|&r| PhysReg(r)).collect();
    used_regs.sort_by_key(|r| r.0);

    RegAllocResult {
        assignments,
        used_regs,
        liveness: Some(liveness),
    }
}

/// Collect values whose types don't fit in a single GPR (floats, i128, and
/// on 32-bit targets: i64/u64). Copy instructions that chain from these
/// values must also be excluded via fixpoint propagation.
fn collect_non_gpr_values(func: &IrFunction, is_32bit: bool) -> FxHashSet<u32> {
    let is_non_gpr_type = |ty: &IrType| -> bool {
        ty.is_float()
            || ty.is_long_double()
            || matches!(ty, IrType::I128 | IrType::U128)
            || (is_32bit && matches!(ty, IrType::I64 | IrType::U64))
    };

    let mut non_gpr_values: FxHashSet<u32> = FxHashSet::default();

    // First pass: collect non-GPR values from typed instructions
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::BinOp { dest, ty, .. } | Instruction::UnaryOp { dest, ty, .. } => {
                    if is_non_gpr_type(ty) {
                        non_gpr_values.insert(dest.0);
                    }
                }
                Instruction::Cast {
                    dest,
                    to_ty,
                    from_ty,
                    ..
                } => {
                    if is_non_gpr_type(to_ty) || is_non_gpr_type(from_ty) {
                        non_gpr_values.insert(dest.0);
                    }
                }
                Instruction::Load { dest, ty, .. } => {
                    if is_non_gpr_type(ty) {
                        non_gpr_values.insert(dest.0);
                    }
                }
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    if let Some(dest) = info.dest {
                        if is_non_gpr_type(&info.return_type) {
                            non_gpr_values.insert(dest.0);
                        }
                    }
                }
                Instruction::Select { dest, ty, .. } => {
                    if is_non_gpr_type(ty) {
                        non_gpr_values.insert(dest.0);
                    }
                }
                Instruction::AtomicLoad { dest, ty, .. }
                | Instruction::AtomicRmw { dest, ty, .. }
                | Instruction::AtomicCmpxchg { dest, ty, .. } => {
                    if is_non_gpr_type(ty) {
                        non_gpr_values.insert(dest.0);
                    }
                }
                Instruction::Intrinsic { dest: Some(d), op, .. } => {
                    // Vector intrinsics produce 128/256-bit values that cannot be
                    // stored in scalar GPRs. Exclude them from register allocation.
                    use crate::ir::intrinsics::IntrinsicOp;
                    let is_vector = matches!(op,
                        IntrinsicOp::VecZeroF64x4 | IntrinsicOp::VecZeroF64x2 |
                        IntrinsicOp::VecZeroI32x8 | IntrinsicOp::VecZeroI32x4 |
                        IntrinsicOp::VecLoadF64x4 | IntrinsicOp::VecLoadF64x2 |
                        IntrinsicOp::VecLoadI32x8 | IntrinsicOp::VecLoadI32x4 |
                        IntrinsicOp::VecAddF64x4 | IntrinsicOp::VecAddF64x2 |
                        IntrinsicOp::VecAddI32x8 | IntrinsicOp::VecAddI32x4 |
                        IntrinsicOp::VecMulF64x4 | IntrinsicOp::VecMulF64x2
                    );
                    if is_vector {
                        non_gpr_values.insert(d.0);
                    }
                }
                _ => {}
            }
        }
    }

    // Propagate non-GPR status through Copy chains: if a Copy's source is a
    // non-GPR value, the dest is also non-GPR. Iterate until fixpoint since
    // Copies can chain (Copy a->b, Copy b->c).
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Copy { dest, src } = inst {
                    if non_gpr_values.contains(&dest.0) {
                        continue;
                    }
                    let src_is_non_gpr = match src {
                        Operand::Value(v) => non_gpr_values.contains(&v.0),
                        Operand::Const(IrConst::F32(_))
                        | Operand::Const(IrConst::F64(_))
                        | Operand::Const(IrConst::LongDouble(..))
                        | Operand::Const(IrConst::I128(_)) => true,
                        Operand::Const(IrConst::I64(_)) if is_32bit => true,
                        _ => false,
                    };
                    if src_is_non_gpr {
                        non_gpr_values.insert(dest.0);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    non_gpr_values
}

/// Remove values from the eligible set that are used as operands in instructions
/// whose codegen paths use resolve_slot_addr() directly (not register-aware).
/// This includes CallIndirect func pointers, Memcpy pointers, va_arg pointers,
/// atomic pointers, StackRestore, and InlineAsm operands.
fn remove_ineligible_operands(
    func: &IrFunction,
    eligible: &mut FxHashSet<u32>,
    config: &RegAllocConfig,
) {
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::CallIndirect {
                    func_ptr: Operand::Value(v),
                    ..
                } => {
                    eligible.remove(&v.0);
                }
                Instruction::Memcpy { dest, src, .. } => {
                    eligible.remove(&dest.0);
                    eligible.remove(&src.0);
                }
                Instruction::VaArg { va_list_ptr, .. } => {
                    eligible.remove(&va_list_ptr.0);
                }
                Instruction::VaStart { va_list_ptr } => {
                    eligible.remove(&va_list_ptr.0);
                }
                Instruction::VaEnd { va_list_ptr } => {
                    eligible.remove(&va_list_ptr.0);
                }
                Instruction::VaCopy { dest_ptr, src_ptr } => {
                    eligible.remove(&dest_ptr.0);
                    eligible.remove(&src_ptr.0);
                }
                Instruction::VaArgStruct {
                    dest_ptr,
                    va_list_ptr,
                    ..
                } => {
                    eligible.remove(&dest_ptr.0);
                    eligible.remove(&va_list_ptr.0);
                }
                Instruction::AtomicRmw {
                    ptr: Operand::Value(v),
                    ..
                } => {
                    eligible.remove(&v.0);
                }
                Instruction::AtomicCmpxchg {
                    ptr: Operand::Value(v),
                    ..
                } => {
                    eligible.remove(&v.0);
                }
                Instruction::AtomicLoad {
                    ptr: Operand::Value(v),
                    ..
                } => {
                    eligible.remove(&v.0);
                }
                Instruction::AtomicStore {
                    ptr: Operand::Value(v),
                    ..
                } => {
                    eligible.remove(&v.0);
                }
                Instruction::StackRestore { ptr } => {
                    eligible.remove(&ptr.0);
                }
                Instruction::InlineAsm {
                    outputs, inputs, ..
                } => {
                    if !config.allow_inline_asm_regalloc {
                        // Inline asm operands are accessed via stack slots
                        // in codegen. Exclude them from register allocation
                        // unless the backend's asm emitter checks reg_assignments.
                        for (_, val, _) in outputs {
                            eligible.remove(&val.0);
                        }
                        for (_, op, _) in inputs {
                            if let Operand::Value(v) = op {
                                eligible.remove(&v.0);
                            }
                        }
                    }
                    // When allow_inline_asm_regalloc is true (RISC-V), the
                    // asm emitter checks reg_assignments before falling back
                    // to stack slot access.
                }
                _ => {}
            }
        }
    }
}

/// Check whether a live interval spans any function call point.
/// Uses binary search since call_points is sorted by program point.
fn spans_any_call(iv: &LiveInterval, call_points: &[u32]) -> bool {
    let start_idx = call_points.partition_point(|&cp| cp < iv.start);
    start_idx < call_points.len() && call_points[start_idx] <= iv.end
}

/// Build a sorted list of allocation candidates from live intervals.
///
/// Filters by eligibility, minimum span length, and call-spanning behavior:
/// - `spans_call == Some(true)`: only intervals that span a call
/// - `spans_call == Some(false)`: only intervals that do NOT span a call
/// - `spans_call == None`: all eligible intervals
///
/// Results are sorted by weighted use count (descending), with interval length
/// as tiebreaker.
fn build_sorted_candidates<'a>(
    liveness: &'a LivenessResult,
    eligible: &FxHashSet<u32>,
    already_assigned: &FxHashMap<u32, PhysReg>,
    call_points: &[u32],
    use_count: &FxHashMap<u32, u64>,
    spans_call: Option<bool>,
) -> Vec<&'a LiveInterval> {
    let mut candidates: Vec<&LiveInterval> = liveness
        .intervals
        .iter()
        .filter(|iv| eligible.contains(&iv.value_id))
        .filter(|iv| !already_assigned.contains_key(&iv.value_id))
        .filter(|iv| iv.end > iv.start)
        .filter(|iv| match spans_call {
            Some(true) => spans_any_call(iv, call_points),
            Some(false) => !spans_any_call(iv, call_points),
            None => true,
        })
        .collect();

    candidates.sort_by(|a, b| {
        let score_a = use_count.get(&a.value_id).copied().unwrap_or(1);
        let score_b = use_count.get(&b.value_id).copied().unwrap_or(1);
        score_b.cmp(&score_a).then_with(|| {
            let len_a = (a.end - a.start) as u64;
            let len_b = (b.end - b.start) as u64;
            len_b.cmp(&len_a)
        })
    });

    candidates
}

/// Find the best callee-saved register for an interval, preferring registers
/// that are already in use (to minimize prologue/epilogue save/restore cost).
///
/// Returns the index into `available_regs` of the chosen register, or None
/// if no register is free at the interval's start point.
fn find_best_callee_reg(
    reg_free_until: &[u32],
    interval_start: u32,
    available_regs: &[PhysReg],
    used_regs_set: &FxHashSet<u8>,
) -> Option<usize> {
    let mut best_already_used: Option<usize> = None;
    let mut best_already_used_free_time: u32 = u32::MAX;
    let mut best_new: Option<usize> = None;
    let mut best_new_free_time: u32 = u32::MAX;

    for (i, &free_until) in reg_free_until.iter().enumerate() {
        if free_until <= interval_start {
            let reg_id = available_regs[i].0;
            if used_regs_set.contains(&reg_id) {
                // Already saved/restored — reusing costs nothing extra.
                if best_already_used.is_none() || free_until < best_already_used_free_time {
                    best_already_used = Some(i);
                    best_already_used_free_time = free_until;
                }
            } else {
                // Would introduce a new callee-saved register.
                if best_new.is_none() || free_until < best_new_free_time {
                    best_new = Some(i);
                    best_new_free_time = free_until;
                }
            }
        }
    }

    best_already_used.or(best_new)
}

/// Exclude every 3rd fusible multiply temp from register allocation.
///
/// This creates a 3-channel multiply ILP pattern:
/// - Channel 1: register-allocated temp (e.g., r12) via standard path
/// - Channel 2: register-allocated temp (e.g., rbx) via standard path
/// - Channel 3: unregistered temp → accumulator path (%eax) via mul-add fusion
///
/// With 3 independent multiply chains, the CPU can fully utilize the multiply
/// port's throughput (1 imul/cycle) despite its 3-cycle latency.
fn exclude_every_third_mul_temp(func: &IrFunction, eligible: &mut FxHashSet<u32>) {
    // Count uses per value
    let mut use_count: FxHashMap<u32, u32> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    *use_count.entry(v.0).or_insert(0) += 1;
                }
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                *use_count.entry(v.0).or_insert(0) += 1;
            }
        });
    }

    // Collect fusible multiply temps in program order
    let mut fusible_temps: Vec<u32> = Vec::new();
    for block in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            let (mul_dest, mul_ty) = match inst {
                Instruction::BinOp { dest, op: crate::ir::reexports::IrBinOp::Mul, ty, .. } => (dest, ty),
                _ => continue,
            };
            if mul_ty.is_float() || matches!(mul_ty, IrType::I128 | IrType::U128) {
                continue;
            }
            if use_count.get(&mul_dest.0).copied().unwrap_or(0) != 1 {
                continue;
            }
            if let Some(Instruction::BinOp { op: crate::ir::reexports::IrBinOp::Add, lhs, rhs, ty: add_ty, .. }) = block.instructions.get(idx + 1) {
                let mul_is_operand = matches!(lhs, Operand::Value(v) if v.0 == mul_dest.0)
                    || matches!(rhs, Operand::Value(v) if v.0 == mul_dest.0);
                if mul_is_operand && mul_ty == add_ty {
                    fusible_temps.push(mul_dest.0);
                }
            }
        }
    }

    // Only apply the 3-channel pattern when there are enough fusible temps
    // to benefit from ILP (at least 6 = two full rotations).
    if fusible_temps.len() < 6 {
        return;
    }

    // Exclude every 3rd temp (indices 2, 5, 8, 11, ...) from register allocation.
    // These will use the accumulator path (%eax) via multiply-add fusion.
    for (i, &temp_id) in fusible_temps.iter().enumerate() {
        if i % 3 == 2 {
            eligible.remove(&temp_id);
        }
    }
}

/// Count weighted uses per value in loop blocks.
/// Returns a map: value_id -> weighted_use_count (uses * 10^loop_depth).
fn count_value_uses_in_loop(
    func: &IrFunction,
    block_loop_depth: &[u32],
) -> FxHashMap<u32, u64> {
    let mut uses: FxHashMap<u32, u64> = FxHashMap::default();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let depth = block_loop_depth.get(block_idx).copied().unwrap_or(0);
        if depth == 0 { continue; }
        let weight = match depth {
            1 => 10u64,
            2 => 100,
            3 => 1000,
            _ => 10_000,
        };
        for inst in &block.instructions {
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    *uses.entry(v.0).or_insert(0) += weight;
                }
            });
        }
    }
    uses
}

/// Detect phi coalesce groups for loop-carried variables.
///
/// After phi elimination, loop-header phi nodes become Copy instructions in
/// predecessor blocks. For the backedge predecessor, this creates a Copy:
///   `%phi_dest = copy %backedge_src`
/// where `%phi_dest` is the multi-def phi variable and `%backedge_src` is the
/// new value computed in the loop body.
///
/// By coalescing these two values (giving them the same register), the Copy
/// becomes a no-op, eliminating a register-to-register move or stack round-trip.
///
/// Returns a list of (phi_dest, backedge_src) pairs that should share a register.
fn detect_phi_coalesce_groups(
    func: &IrFunction,
    liveness: &LivenessResult,
) -> Vec<(u32, u32)> {
    // Step 1: Find multi-def values (phi dests after phi elimination).
    // A value is multi-def if it has Copy definitions in multiple blocks.
    let mut def_block: FxHashMap<u32, usize> = FxHashMap::default();
    let mut multi_def: FxHashSet<u32> = FxHashSet::default();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Instruction::Copy { dest, .. } = inst {
                if let Some(&prev) = def_block.get(&dest.0) {
                    if prev != block_idx {
                        multi_def.insert(dest.0);
                    }
                }
                def_block.insert(dest.0, block_idx);
            }
        }
    }

    if multi_def.is_empty() {
        return Vec::new();
    }

    // Step 1b: Build use-block map for backedge source safety check.
    // If a backedge source is used in blocks OTHER than the Copy's block,
    // coalescing is unsafe: the source's register would be reused by the
    // allocator for other values in those blocks, clobbering the source
    // before its cross-block uses.
    let mut src_use_blocks: FxHashMap<u32, FxHashSet<usize>> = FxHashMap::default();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            // Skip Copy dests — we care about OPERAND uses, not definitions
            let check_operands = |inst: &Instruction| {
                let mut uses = Vec::new();
                match inst {
                    Instruction::BinOp { lhs, rhs, .. } => {
                        if let Operand::Value(v) = lhs { uses.push(v.0); }
                        if let Operand::Value(v) = rhs { uses.push(v.0); }
                    }
                    Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } => {
                        if let Operand::Value(v) = src { uses.push(v.0); }
                    }
                    Instruction::Store { val, .. } => {
                        if let Operand::Value(v) = val { uses.push(v.0); }
                    }
                    Instruction::Copy { src, .. } => {
                        if let Operand::Value(v) = src { uses.push(v.0); }
                    }
                    Instruction::Cmp { lhs, rhs, .. } => {
                        if let Operand::Value(v) = lhs { uses.push(v.0); }
                        if let Operand::Value(v) = rhs { uses.push(v.0); }
                    }
                    Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                        for a in &info.args {
                            if let Operand::Value(v) = a { uses.push(v.0); }
                        }
                    }
                    Instruction::Select { cond, true_val, false_val, .. } => {
                        if let Operand::Value(v) = cond { uses.push(v.0); }
                        if let Operand::Value(v) = true_val { uses.push(v.0); }
                        if let Operand::Value(v) = false_val { uses.push(v.0); }
                    }
                    _ => {}
                }
                uses
            };
            for vid in check_operands(inst) {
                src_use_blocks.entry(vid).or_default().insert(block_idx);
            }
        }
        // Also check terminator operands
        match &block.terminator {
            Terminator::CondBranch { cond, .. } => {
                if let Operand::Value(v) = cond {
                    src_use_blocks.entry(v.0).or_default().insert(block_idx);
                }
            }
            Terminator::Return(Some(op)) => {
                if let Operand::Value(v) = op {
                    src_use_blocks.entry(v.0).or_default().insert(block_idx);
                }
            }
            Terminator::Switch { val, .. } => {
                if let Operand::Value(v) = val {
                    src_use_blocks.entry(v.0).or_default().insert(block_idx);
                }
            }
            _ => {}
        }
    }

    // Step 2: Find backedge copies in loop blocks.
    // A backedge copy is a Copy where:
    //   - The dest is a multi-def value (phi dest)
    //   - The source is a Value (not a constant)
    //   - The copy is in a block with loop_depth > 0
    let mut groups: Vec<(u32, u32)> = Vec::new();
    let mut seen_phi_dests: FxHashSet<u32> = FxHashSet::default();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let depth = liveness.block_loop_depth.get(block_idx).copied().unwrap_or(0);
        if depth == 0 {
            continue;
        }

        for inst in &block.instructions {
            if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                if multi_def.contains(&dest.0) && !seen_phi_dests.contains(&dest.0) {
                    // Don't coalesce if src is itself a multi-def (swap cycle temporaries)
                    if !multi_def.contains(&src_val.0) {
                        // Safety: don't coalesce if the phi dest is used AFTER
                        // the backedge source's definition. This detects the
                        // "lost copy" pattern where e.g.:
                        //   v_n = Call(malloc)       ← src defined here
                        //   Store(v_head, v_n+8)     ← phi dest USED here
                        //   Copy v_head = v_n        ← coalesce candidate
                        // Coalescing v_head and v_n to the same register would
                        // clobber v_head when storing the Call result.
                        //
                        // Important: the src may be defined in a DIFFERENT block
                        // than the Copy (multi-block loop bodies). We must check
                        // the src's defining block for phi dest uses, not just
                        // the Copy's block.
                        let mut phi_dest_used_after_src = false;

                        // Find the block that defines the backedge source
                        let mut src_def_block = None;
                        for (bi, b) in func.blocks.iter().enumerate() {
                            for i in &b.instructions {
                                if let Some(d) = i.dest() {
                                    if d.0 == src_val.0 {
                                        src_def_block = Some(bi);
                                    }
                                }
                            }
                        }

                        // Check the block containing the Copy
                        {
                            let mut src_defined = false;
                            for inst2 in &block.instructions {
                                if !src_defined {
                                    if let Some(d) = inst2.dest() {
                                        if d.0 == src_val.0 {
                                            src_defined = true;
                                        }
                                    }
                                } else {
                                    if let Instruction::Copy { dest: d, .. } = inst2 {
                                        if d.0 == dest.0 {
                                            break;
                                        }
                                    }
                                    if uses_value(inst2, dest.0) {
                                        phi_dest_used_after_src = true;
                                    }
                                }
                            }
                        }

                        // If the src is defined in a DIFFERENT block, also check
                        // that block (and any other block the src's value flows
                        // through) for phi dest uses after the src definition.
                        if let Some(sdb) = src_def_block {
                            if sdb != block_idx {
                                let mut src_defined = false;
                                for inst2 in &func.blocks[sdb].instructions {
                                    if !src_defined {
                                        if let Some(d) = inst2.dest() {
                                            if d.0 == src_val.0 {
                                                src_defined = true;
                                            }
                                        }
                                    } else {
                                        if uses_value(inst2, dest.0) {
                                            phi_dest_used_after_src = true;
                                        }
                                    }
                                }
                            }
                        }
                        // Also check: the backedge source must not have uses
                        // in OTHER blocks. If it does, coalescing gives it the
                        // phi dest's register, but the allocator may reassign
                        // that register to other values in those blocks,
                        // clobbering the source before its cross-block uses.
                        let src_has_cross_block_use = src_use_blocks
                            .get(&src_val.0)
                            .map(|blocks| blocks.iter().any(|&b| b != block_idx))
                            .unwrap_or(false);

                        if !phi_dest_used_after_src && !src_has_cross_block_use {
                            if std::env::var("CCC_DEBUG_PHI_COALESCE").is_ok() {
                                eprintln!("[PHI_COALESCE] Coalescing phi_dest=Value({}) with backedge_src=Value({}) in block {}",
                                    dest.0, src_val.0, block_idx);
                            }
                            groups.push((dest.0, src_val.0));
                            seen_phi_dests.insert(dest.0);
                        } else if std::env::var("CCC_DEBUG_PHI_COALESCE").is_ok() {
                            eprintln!("[PHI_COALESCE] BLOCKED phi_dest=Value({}) with backedge_src=Value({}) in block {} (used_after={}, cross_block={})",
                                dest.0, src_val.0, block_idx, phi_dest_used_after_src, src_has_cross_block_use);
                        }
                    }
                }
            }
        }
    }

    groups
}

/// Check if an instruction uses a given value ID as an operand (not as dest).
fn uses_value(inst: &Instruction, val_id: u32) -> bool {
    let check_op = |op: &Operand| -> bool {
        matches!(op, Operand::Value(v) if v.0 == val_id)
    };
    match inst {
        Instruction::Store { val, ptr, .. } => check_op(val) || ptr.0 == val_id,
        Instruction::Load { ptr, .. } => ptr.0 == val_id,
        Instruction::BinOp { lhs, rhs, .. } => check_op(lhs) || check_op(rhs),
        Instruction::UnaryOp { src, .. } => check_op(src),
        Instruction::Cmp { lhs, rhs, .. } => check_op(lhs) || check_op(rhs),
        Instruction::Cast { src, .. } => check_op(src),
        Instruction::Copy { src, .. } => check_op(src),
        Instruction::GetElementPtr { base, offset, .. } => base.0 == val_id || check_op(offset),
        Instruction::Select { cond, true_val, false_val, .. } =>
            check_op(cond) || check_op(true_val) || check_op(false_val),
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } =>
            info.args.iter().any(|a| check_op(a)),
        Instruction::AtomicStore { val, ptr, .. } => check_op(val) || check_op(ptr),
        Instruction::AtomicLoad { ptr, .. } => check_op(ptr),
        Instruction::AtomicRmw { ptr, val, .. } => check_op(ptr) || check_op(val),
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } =>
            check_op(ptr) || check_op(expected) || check_op(desired),
        _ => false,
    }
}
