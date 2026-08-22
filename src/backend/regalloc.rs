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
    for_each_value_use_in_instruction, LiveInterval, LivenessResult,
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
    /// Caller-saved registers assigned to call-spanning values (Phase 2b).
    /// Maps PhysReg ID → list of (interval_start, interval_end) for each value
    /// assigned to that register. Used for selective save/restore at call sites.
    pub caller_save_spans: FxHashMap<u8, Vec<(u32, u32)>>,
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
    // Merge all segments of a value into one span (min start, max end). The
    // assignment map is keyed per value, so scanning segments independently
    // makes the last-processed segment's register the value's home for its
    // ENTIRE lifetime — including other segments' spans, where that register
    // may already be taken. (Start-order processing mostly hid this; priority
    // ordering exposed it as a strlen_bench segfault.)
    let mut merged: FxHashMap<u32, LiveInterval> = FxHashMap::default();
    for iv in &liveness.intervals {
        if !eligible.contains(&iv.value_id) || iv.end <= iv.start {
            continue;
        }
        merged
            .entry(iv.value_id)
            .and_modify(|m| {
                m.start = m.start.min(iv.start);
                m.end = m.end.max(iv.end);
            })
            .or_insert(*iv);
    }
    let mut out: Vec<LiveInterval> = merged.into_values().collect();
    out.sort_by_key(|iv| iv.start);
    out
}

/// Merge every interval segment of each value into one span (min start, max
/// end). See filter_eligible_intervals for why per-segment scanning is unsafe
/// with a per-value assignment map.
fn merge_intervals_by_value(intervals: Vec<LiveInterval>) -> Vec<LiveInterval> {
    let mut merged: FxHashMap<u32, LiveInterval> = FxHashMap::default();
    for iv in intervals {
        if iv.end <= iv.start {
            continue;
        }
        merged
            .entry(iv.value_id)
            .and_modify(|m| {
                m.start = m.start.min(iv.start);
                m.end = m.end.max(iv.end);
            })
            .or_insert(iv);
    }
    let mut out: Vec<LiveInterval> = merged.into_values().collect();
    out.sort_by_key(|iv| iv.start);
    out
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
            caller_save_spans: FxHashMap::default(),
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
    // The AArch64 FP pool (allocator IDs 40+) additionally keeps scalar
    // float intrinsic results out of GPRs so they can use FP registers.
    let arm_fp_pool = config.xmm_regs.first().is_some_and(|r| r.0 == 40)
        && std::env::var("CCC_NO_VECREG").is_err();
    let non_gpr_values = collect_non_gpr_values(func, is_32bit, arm_fp_pool);

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
            // Pointer/base uses (Store.ptr, Load.ptr, GEP.base, memcpy endpoints)
            // are direct Value references, not Operands. Count them too so a
            // loop-carried pointer ranks by its true temperature instead of 0.
            for_each_value_use_in_instruction(inst, |v| {
                *use_count.entry(v.0).or_insert(0) += weight;
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
    //
    // This is an x86-64-specific trick: on AArch64/RISC-V the accumulator
    // path is a single register and excluding mul temps from registers only
    // adds shuffle overhead, so it is gated to the x86-64 register pool.
    if config.xmm_regs.first().is_some_and(|r| r.0 == 20) {
        exclude_every_third_mul_temp(func, &mut eligible);
    }

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
    let all_phi_pairs = detect_phi_coalesce_groups(func, &liveness, false);
    // The FP register coalescing below verifies interval conflicts itself, so
    // it can use the relaxed pair set that allows cross-block backedge sources
    // (e.g. a value also read by the loop-exit test, like mandelbrot's y).
    let fp_phi_pairs = detect_phi_coalesce_groups(func, &liveness, true);
    let mut phi_coalesce = if std::env::var("CCC_NO_PHI_COALESCE").is_ok() {
        Vec::new()
    } else {
        all_phi_pairs.clone()
    };
    // Never propagate a register assignment onto an ineligible value.  In
    // particular an Alloca's Value denotes its address, not a scalar loaded
    // from its slot; treating it like a registered integer changes pointer
    // semantics and can corrupt loops whose phi initially receives an alloca.
    let integer_binop_defs: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|block| block.instructions.iter())
        .filter_map(|inst| match inst {
            Instruction::BinOp { dest, ty, .. } if !is_non_gpr_type(ty) => Some(dest.0),
            _ => None,
        })
        .collect();
    phi_coalesce.retain(|(phi_dest, backedge_src)| {
        eligible.contains(phi_dest)
            && eligible.contains(backedge_src)
            // Limit the relaxed live-range overlap rule to scalar arithmetic
            // recurrences. Pointer/GEP, load, call, and copy sources can carry
            // address identity or memory lifetime constraints not represented
            // by the simple interval test.
            && integer_binop_defs.contains(backedge_src)
    });
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
        live_range::build_live_ranges(&phase1_intervals, &liveness.block_loop_depth, func, true);
    let mut allocator = LinearScanAllocator::new(phase1_ranges, config.available_regs.clone());
    allocator.run();

    let mut assignments = allocator.assignments;

    // --- Post-scan rebalance: steal registers for hot loop-carried values ---
    //
    // The linear scan processes ranges in start order and never evicts, so
    // cold function-spanning values (array bases, globals, prologue pointers)
    // win every callee-saved register simply by starting first, leaving hot
    // inner-loop-carried phi values (IVs, accumulators, carried pointers)
    // stack-homed (e.g. fannkuch's flip loop, arith_loop).
    //
    // For each hot loop-carried phi dest the scan MISSED, pick the register
    // whose conflicting holders (live interval overlaps the hot value's) have
    // the coldest total use count, and fully deallocate those holders back to
    // the stack. This is safe where eviction inside the scan was not: an
    // evicted holder is deallocated for its ENTIRE interval — indistinguishable
    // from never having been assigned, so the default stack path handles all
    // its uses — and every remaining holder is provably non-overlapping with
    // the hot value. No live range is ever split, and when the scan already
    // housed every hot value (e.g. spectral_norm) the rebalance is a no-op.
    //
    // AArch64-only: relies on the wide callee-saved GPR pool.
    // CCC_NO_LOOP_PIN disables; CCC_LOOP_PIN=N caps steals per function (default 2).
    if std::env::var("CCC_NO_LOOP_PIN").is_err()
        && config.xmm_regs.first().is_some_and(|r| r.0 == 40)
        && !all_phi_pairs.is_empty()
    {
        let k: usize = std::env::var("CCC_LOOP_PIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        // Values whose register assignment the phi-coalesce propagation below
        // depends on; stealing from them would defeat coalescing.
        let phi_pair_values: FxHashSet<u32> = phi_coalesce
            .iter()
            .flat_map(|&(d, s)| [d, s])
            .collect();
        // Rank loop-carried phi dests by loop-weighted use count (uses inside
        // a loop at depth D contribute 10^D, so hot inner-loop values sort first).
        let mut candidates: Vec<(u32, u64)> = all_phi_pairs
            .iter()
            .map(|&(phi_dest, _)| phi_dest)
            .filter(|v| eligible.contains(v))
            .filter(|v| {
                liveness
                    .intervals
                    .iter()
                    .any(|iv| iv.value_id == *v && iv.end > iv.start)
            })
            .map(|v| (v, use_count.get(&v).copied().unwrap_or(0)))
            .filter(|&(_, count)| count >= 10) // at least one in-loop use
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.dedup_by_key(|&mut (v, _)| v);

        // All live-interval segments per value; conflict checks must consider
        // every segment, not just the first.
        let mut segs_of: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for iv in &liveness.intervals {
            segs_of.entry(iv.value_id).or_default().push((iv.start, iv.end));
        }
        let overlaps = |a: u32, b: u32| -> bool {
            let (Some(sa), Some(sb)) = (segs_of.get(&a), segs_of.get(&b)) else {
                return false;
            };
            sa.iter().any(|&(s1, e1)| sb.iter().any(|&(s2, e2)| s1 < e2 && s2 < e1))
        };
        // Current register holders, updated as steals happen.
        let mut holders_by_reg: FxHashMap<u8, Vec<u32>> = FxHashMap::default();
        for (&v, &r) in &assignments {
            holders_by_reg.entry(r.0).or_default().push(v);
        }

        let mut steals = 0;
        for &(vid, hot_count) in &candidates {
            if steals >= k {
                break;
            }
            if assignments.contains_key(&vid) {
                continue; // the scan already housed this hot value
            }
            if std::env::var("CCC_DEBUG_LOOP_PIN").is_ok() {
                eprintln!("[LOOP_PIN] func={} candidate v{} (use {}) MISSED by scan", func.name, vid, hot_count);
            }
            // Choose the register whose CONFLICTING holders are coldest.
            // Every holder whose live interval overlaps the hot value must be
            // fully deallocated back to the stack for the steal to be sound;
            // non-conflicting holders keep timesharing the register. Registers
            // holding phi-coalesce participants are left alone so backedge
            // coalescing stays intact. A steal is only taken when the total
            // evicted use count is strictly colder than the hot value.
            let mut best: Option<(u8, Vec<u32>, u64)> = None;
            for (&reg_id, holders) in &holders_by_reg {
                if holders.iter().any(|h| phi_pair_values.contains(h)) {
                    continue;
                }
                let evict: Vec<u32> = holders
                    .iter()
                    .copied()
                    .filter(|&h| overlaps(h, vid))
                    .collect();
                let cost: u64 = evict
                    .iter()
                    .map(|h| use_count.get(h).copied().unwrap_or(0))
                    .sum();
                if cost >= hot_count {
                    continue; // not profitable
                }
                if best.as_ref().map_or(true, |&(_, _, c)| cost < c) {
                    best = Some((reg_id, evict, cost));
                }
            }
            if let Some((reg_id, evict, cost)) = best {
                if std::env::var("CCC_DEBUG_LOOP_PIN").is_ok() {
                    eprintln!(
                        "[LOOP_PIN] func={} v{} (use {}) takes reg {}, evicting {:?} (use {})",
                        func.name, vid, hot_count, reg_id, evict, cost
                    );
                }
                for v in &evict {
                    assignments.remove(v);
                    if let Some(h) = holders_by_reg.get_mut(&reg_id) {
                        h.retain(|x| x != v);
                    }
                }
                assignments.insert(vid, PhysReg(reg_id));
                holders_by_reg.entry(reg_id).or_default().push(vid);
                steals += 1;
            }
        }
    }

    let mut used_regs_set: FxHashSet<u8> = FxHashSet::default();
    for &reg in assignments.values() {
        used_regs_set.insert(reg.0);
    }
    // Caller-saved registers allocated in Phase 2. These do NOT belong in
    // used_regs_set (no prologue save), but Phase 2b must not reallocate them.
    let mut phase2_caller_used: FxHashSet<u8> = FxHashSet::default();

    // Phase 2: caller-saved linear scan for unallocated non-call-spanning values.
    if !config.caller_saved_regs.is_empty() {
        // Merged per-value spans (same discipline as Phase 1): the assignment
        // is per value, so a caller-saved register is only safe when the
        // value's WHOLE span is call-free — checking individual segments would
        // let a value keep a caller-saved register across a call that only one
        // of its segments spans (strlen_bench segfault).
        let phase2_intervals: Vec<LiveInterval> = filter_eligible_intervals(&liveness, &eligible)
            .into_iter()
            .filter(|iv| !assignments.contains_key(&iv.value_id))
            .filter(|iv| !spans_any_call(iv, call_points))
            .collect();

        if !phase2_intervals.is_empty() {
            let phase2_ranges = live_range::build_live_ranges(
                &phase2_intervals,
                &liveness.block_loop_depth,
                func,
                true,
            );
            let mut caller_allocator =
                LinearScanAllocator::new(phase2_ranges, config.caller_saved_regs.clone());
            caller_allocator.run();

            for (vid, reg) in caller_allocator.assignments {
                assignments.insert(vid, reg);
                // Do NOT add to used_regs_set: these are caller-saved
                // registers holding values that never span a call. The ABI
                // lets this function clobber their incoming contents freely,
                // so a prologue push / epilogue pop would preserve nothing.
                // Track them separately so Phase 2b doesn't reallocate a
                // register that is already in use.
                phase2_caller_used.insert(reg.0);
            }
        }
    }

    // Debug: count overlaps BEFORE phi coalesce
    if std::env::var("CCC_VERIFY_REGALLOC").is_ok() {
        let mut pre_count = 0;
        let mut pre_reg_ivs: std::collections::HashMap<u8, Vec<(u32, u32, u32)>> = std::collections::HashMap::new();
        for iv in &liveness.intervals {
            if let Some(&reg) = assignments.get(&iv.value_id) {
                pre_reg_ivs.entry(reg.0).or_default().push((iv.start, iv.end, iv.value_id));
            }
        }
        for (_, intervals) in &pre_reg_ivs {
            for i in 0..intervals.len() {
                for j in (i+1)..intervals.len() {
                    let (s1, e1, _) = intervals[i];
                    let (s2, e2, _) = intervals[j];
                    if s1 < e2 && s2 < e1 { pre_count += 1; }
                }
            }
        }
        if pre_count > 0 {
            eprintln!("[REGALLOC-PRE-PHI] {} overlaps BEFORE phi coalesce", pre_count);
        }
    }

    // Propagate phi coalesce assignments: backedge source values inherit
    // the register of their phi dest. This makes the backedge Copy a no-op
    // when both values share the same register.
    // Safety check: only propagate if the backedge source's interval doesn't
    // conflict with other values already assigned to the same register.
    for &(phi_dest, backedge_src) in &phi_coalesce {
        if let Some(&reg) = assignments.get(&phi_dest) {
            // Check every live segment of the backedge source — not just the
            // first — against every segment of other values holding the
            // register. A multi-segment source whose later segment conflicts
            // would otherwise inherit the register and clobber/ corrupt the
            // other value (strlen_bench segfault with priority-ordered scan).
            let src_segs: Vec<(u32, u32)> = liveness.intervals.iter()
                .filter(|iv| iv.value_id == backedge_src)
                .map(|iv| (iv.start, iv.end))
                .collect();
            if !src_segs.is_empty() {
                // The source and phi-destination intervals normally overlap in
                // linearized loop liveness: the phi is conservatively live for
                // the whole loop while its replacement is computed near the
                // backedge.  That overlap is precisely what coalescing resolves.
                // detect_phi_coalesce_groups has already proved that the old phi
                // value is not used after the replacement is defined.
                // Check for conflicts with other values in the same register
                let has_conflict = src_segs.iter().any(|&(ss, se)| {
                    liveness.intervals.iter().any(|iv| {
                        if iv.value_id == backedge_src || iv.value_id == phi_dest { return false; }
                        if let Some(&other_reg) = assignments.get(&iv.value_id) {
                            other_reg.0 == reg.0 && iv.start < se && ss < iv.end
                        } else {
                            false
                        }
                    })
                });
                if !has_conflict {
                    assignments.insert(backedge_src, reg);
                }
            } else {
                // No interval info — still safe to propagate (value might be dead)
                assignments.insert(backedge_src, reg);
            }
        }
    }

    // Debug: count overlaps after phi coalesce
    if std::env::var("CCC_VERIFY_REGALLOC").is_ok() {
        let mut overlap_count = 0;
        let mut reg_ivs: std::collections::HashMap<u8, Vec<(u32, u32, u32)>> = std::collections::HashMap::new();
        for iv in &liveness.intervals {
            if let Some(&reg) = assignments.get(&iv.value_id) {
                reg_ivs.entry(reg.0).or_default().push((iv.start, iv.end, iv.value_id));
            }
        }
        for (_, intervals) in &reg_ivs {
            for i in 0..intervals.len() {
                for j in (i+1)..intervals.len() {
                    let (s1, e1, _) = intervals[i];
                    let (s2, e2, _) = intervals[j];
                    if s1 < e2 && s2 < e1 { overlap_count += 1; }
                }
            }
        }
        if overlap_count > 0 {
            eprintln!("[REGALLOC-POST-PHI] {} overlaps after phi coalesce", overlap_count);
        }
    }

    // AArch64 (allocator IDs 40..47 → v16..v23) additionally allocates
    // 128-bit vector values and copy-form F64 values (loop accumulators)
    // to FP/SIMD registers; other targets keep those stack-homed because
    // their emitters are not register-aware for them.
    let (vector_values, f64_value_set) = if arm_fp_pool {
        (collect_vector_values(func), collect_f64_values(func))
    } else {
        (FxHashSet::default(), FxHashSet::default())
    };
    // The callee-saved FP subset (allocator IDs 32..=38 → d8-d14) present in
    // the pool. Call-spanning F64 values may only draw from these; empty when
    // CCC_NO_FP_CALLEE_SAVED removed them, which also keeps the strict
    // call-span filter below (a call-spanning value in a caller-saved FP
    // register would be clobbered by the spanned call).
    let arm_callee_fp: Vec<PhysReg> = if arm_fp_pool {
        config
            .xmm_regs
            .iter()
            .copied()
            .filter(|r| (32..=38).contains(&r.0))
            .collect()
    } else {
        Vec::new()
    };

    // Phase 3: XMM register allocation for F64 values that don't span calls.
    // These values were excluded from GPR allocation but can use XMM registers.
    if !config.xmm_regs.is_empty() {
        // Values actually consumed by a real (non-Copy) instruction.  SSA copy
        // webs can carry a value across loop boundaries without it ever being
        // used in a computation; such values would otherwise win FP registers
        // with their huge intervals and starve the real accumulators.
        // A copy source feeding (transitively) a real use still qualifies —
        // e.g. an fmadd result copied into a loop accumulator.
        let mut real_use: FxHashSet<u32> = FxHashSet::default();
        if arm_fp_pool {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if matches!(inst, Instruction::Copy { .. }) {
                        continue;
                    }
                    for_each_operand_in_instruction(inst, |op| {
                        if let Operand::Value(v) = op {
                            real_use.insert(v.0);
                        }
                    });
                }
                for_each_operand_in_terminator(&block.terminator, |op| {
                    if let Operand::Value(v) = op {
                        real_use.insert(v.0);
                    }
                });
            }
            loop {
                let mut changed = false;
                for block in &func.blocks {
                    for inst in &block.instructions {
                        if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                            if real_use.contains(&dest.0) && !real_use.contains(&src_val.0) {
                                real_use.insert(src_val.0);
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        // Collect F64 values: values in non_gpr_values that are F64 typed,
        // haven't been assigned a GPR, and don't span calls.
        let f64_intervals: Vec<LiveInterval> = liveness
            .intervals
            .iter()
            .filter(|iv| non_gpr_values.contains(&iv.value_id))
            .filter(|iv| iv.end > iv.start)
            .filter(|iv| !assignments.contains_key(&iv.value_id))
            // AArch64: call-spanning F64 scalars stay eligible, but are
            // restricted below to the callee-saved FP registers (d8-d14),
            // which survive calls by the ABI. 128-bit vectors keep the
            // strict filter: only the low 64 bits of v8-v15 are callee-saved.
            // Other targets keep the strict filter (their XMM pools are
            // entirely caller-saved). CCC_NO_FP_CALLSPAN restores the old
            // behavior for A/B.
            .filter(|iv| {
                if !spans_any_call(iv, call_points) {
                    return true;
                }
                !arm_callee_fp.is_empty()
                    && std::env::var("CCC_NO_FP_CALLSPAN").is_err()
                    && f64_value_set.contains(&iv.value_id)
                    && !vector_values.contains(&iv.value_id)
            })
            // Skip values that are only ever copied (never feed a computation):
            // they don't need a register and would starve values that do.
            .filter(|iv| !arm_fp_pool || real_use.contains(&iv.value_id))
            // Only include values that are actually F64 (not i128, not f32, etc.)
            // — plus (AArch64) 128-bit vector values and copy-form F64 values.
            .filter(|iv| {
                vector_values.contains(&iv.value_id) || f64_value_set.contains(&iv.value_id) || {
                    if arm_fp_pool {
                        return false;
                    }
                    func.blocks.iter().any(|block| block.instructions.iter().any(|inst| match inst {
                        Instruction::BinOp { dest, ty, .. }
                        | Instruction::UnaryOp { dest, ty, .. } if *ty == IrType::F64 => dest.0 == iv.value_id,
                        Instruction::Load { dest, ty, .. } if *ty == IrType::F64 => dest.0 == iv.value_id,
                        Instruction::Cast { dest, to_ty, .. } if *to_ty == IrType::F64 => dest.0 == iv.value_id,
                        _ => false,
                    }))
                }
            })
            .copied()
            .collect::<Vec<_>>();
        // One range per value (the assignment map is per value — see
        // merge_intervals_by_value). Without this, a multi-segment F64 value's
        // segments are scanned independently and the last write wins.
        let f64_intervals = merge_intervals_by_value(f64_intervals);

        if std::env::var("CCC_DEBUG_VECREG").is_ok() {
            eprintln!("[VECREG] func={} vector_values={:?}", func.name, vector_values);
            for &vid in &vector_values {
                let iv = liveness.intervals.iter().find(|iv| iv.value_id == vid);
                eprintln!("[VECREG]   v{}: interval={:?} non_gpr={} assigned={} spans_call={}",
                    vid, iv.map(|i| (i.start, i.end)), non_gpr_values.contains(&vid),
                    assignments.contains_key(&vid),
                    iv.is_some_and(|i| spans_any_call(i, call_points)));
            }
        }
        if std::env::var("CCC_DEBUG_FPREG").is_ok() {
            eprintln!("[FPREG] func={} f64_count={} intervals_in={}", func.name, f64_value_set.len(), f64_intervals.len());
            for iv in &f64_intervals {
                eprintln!("[FPREG]   cand v{} [{}, {}]", iv.value_id, iv.start, iv.end);
            }
            for &vid in &f64_value_set {
                if liveness.intervals.iter().all(|iv| iv.value_id != vid) {
                    eprintln!("[FPREG]   v{}: NO INTERVAL", vid);
                }
            }
            for iv in &liveness.intervals {
                if f64_value_set.contains(&iv.value_id) && f64_intervals.iter().all(|c| c.value_id != iv.value_id) {
                    eprintln!("[FPREG]   excluded v{} [{}, {}] non_gpr={} assigned={} spans_call={}",
                        iv.value_id, iv.start, iv.end, non_gpr_values.contains(&iv.value_id),
                        assignments.contains_key(&iv.value_id), spans_any_call(iv, call_points));
                }
            }
        }

        if !f64_intervals.is_empty() {
            let f64_ranges = live_range::build_live_ranges(
                &f64_intervals,
                &liveness.block_loop_depth,
                func,
                true,
            );
            // Pool order for the general (non-call-spanning) scan: caller-saved
            // d16-d31 (ids 40+) first, callee-saved d8-d14 (ids 32-38) last.
            // Call-spanning F64 values can ONLY use d8-d14 (restricted below),
            // so general values should prefer the caller-saved pool — but may
            // overflow into d8-d14 when it is full (reserving d8-d14 entirely
            // caused FP spill storms in struct_copy's wide loops).
            let xmm_pool: Vec<PhysReg> = if arm_fp_pool {
                let mut caller: Vec<PhysReg> = config.xmm_regs.iter().copied().filter(|r| r.0 >= 40).collect();
                let mut callee: Vec<PhysReg> = config.xmm_regs.iter().copied().filter(|r| (32..=38).contains(&r.0)).collect();
                caller.append(&mut callee);
                caller
            } else {
                config.xmm_regs.clone()
            };
            let mut xmm_allocator =
                LinearScanAllocator::new(f64_ranges, xmm_pool);
            // Confine call-spanning F64 values to the callee-saved FP subset
            // (allocator IDs 32..=38 → d8-d14) present in the pool; the
            // caller-saved d16-d31 would be clobbered by the spanned call.
            if !arm_callee_fp.is_empty() {
                xmm_allocator.restricted_values = f64_intervals
                    .iter()
                    .filter(|iv| spans_any_call(iv, call_points))
                    .map(|iv| iv.value_id)
                    .collect();
                xmm_allocator.restricted_regs = arm_callee_fp.clone();
            }
            xmm_allocator.run();

            for (vid, reg) in &xmm_allocator.assignments {
                if std::env::var("CCC_DEBUG_VECREG").is_ok() && vector_values.contains(vid) {
                    eprintln!("[VECREG]   assigned v{} -> reg {}", vid, reg.0);
                }
                if std::env::var("CCC_DEBUG_FPREG").is_ok() {
                    eprintln!("[FPREG]   assigned v{} -> reg {} (pool size {})", vid, reg.0, config.xmm_regs.len());
                }
            }
            for (vid, reg) in xmm_allocator.assignments {
                assignments.insert(vid, reg);
                // XMM regs (20+) are caller-saved, no prologue save needed
            }
        }
    }

    // AArch64 reserves allocator IDs 48..55 for explicitly marked loop-carried
    // F64 values (d24..d31). They are caller-saved and disjoint from the generic
    // d16..d23 pool, so reductions do not reduce temporary-register capacity.
    if config.xmm_regs.first().is_some_and(|r| r.0 == 40) {
        for (index, value) in func.loop_promoted_f64_values.iter().take(8).enumerate() {
            assignments.insert(value.0, PhysReg(48 + index as u8));
        }
    }

    // AArch64 FP phi coalescing: give the value copied into a loop-carried
    // F64/vector accumulator at the backedge the accumulator's own register,
    // eliminating the backedge fmov/mov from the loop's serial dependency
    // chain (e.g. `fmadd d16, .., d16` instead of fmadd into a temp + fmov).
    if arm_fp_pool {
        let debug = std::env::var("CCC_DEBUG_FPCOAL").is_ok();
        for &(phi_dest, backedge_src) in &fp_phi_pairs {
            let d_reg = assignments.get(&phi_dest).copied().filter(|r| (32..=38).contains(&r.0) || (40..=55).contains(&r.0));
            let s_reg = assignments.get(&backedge_src).copied().filter(|r| (32..=38).contains(&r.0) || (40..=55).contains(&r.0));
            if debug {
                eprintln!("[FPCOAL] cand phi={} d_reg={:?} src={} s_reg={:?} f64={} vec={}",
                    phi_dest, d_reg.map(|r| r.0), backedge_src, s_reg.map(|r| r.0),
                    f64_value_set.contains(&phi_dest), vector_values.contains(&phi_dest));
            }
            if !f64_value_set.contains(&phi_dest) && !vector_values.contains(&phi_dest) {
                continue;
            }
            // Reverse coalescing: the phi dest (a copy-web accumulator that
            // only ever ferries the backedge value to the next iteration)
            // has no register of its own. Give it the backedge source's
            // register when nothing else holding that register overlaps the
            // dest's interval — the backedge copy then dies on the chain.
            if d_reg.is_none() {
                if std::env::var("CCC_NO_FP_REVERSE_COAL").is_ok() {
                    continue;
                }
                let Some(s) = s_reg else { continue };
                // Check EVERY segment of the dest — not just the first —
                // against every segment of the register's other holders
                // (same discipline as the forward path; a later-segment
                // conflict would corrupt the other value).
                let dest_segs: Vec<(u32, u32)> = liveness.intervals.iter()
                    .filter(|iv| iv.value_id == phi_dest)
                    .map(|iv| (iv.start, iv.end))
                    .collect();
                if dest_segs.is_empty() {
                    continue;
                }
                // A caller-saved FP register cannot hold a call-spanning value.
                if s.0 >= 40 && dest_segs.iter().any(|&(ds, de)| {
                    call_points.iter().any(|&cp| ds < cp && cp < de)
                }) {
                    continue;
                }
                let conflict = dest_segs.iter().any(|&(ds, de)| {
                    liveness.intervals.iter().any(|iv| {
                        if iv.value_id == phi_dest || iv.value_id == backedge_src {
                            return false;
                        }
                        assignments
                            .get(&iv.value_id)
                            .is_some_and(|&o| o.0 == s.0 && iv.start < de && ds < iv.end)
                    })
                });
                if debug {
                    eprintln!("[FPCOAL] reverse phi={} <- d{} conflict={}", phi_dest, s.0, conflict);
                }
                if !conflict {
                    assignments.insert(phi_dest, s);
                }
                continue;
            }
            let (Some(d), Some(s)) = (d_reg, s_reg) else { continue };
            if d == s {
                continue;
            }
            // Conflict check: no other value assigned d may overlap the src
            // interval (the phi dest itself is expected to overlap — that is
            // precisely what coalescing resolves).
            if let Some(src_iv) = liveness.intervals.iter().find(|iv| iv.value_id == backedge_src) {
                let conflict = liveness.intervals.iter().any(|iv| {
                    if iv.value_id == backedge_src || iv.value_id == phi_dest {
                        return false;
                    }
                    assignments
                        .get(&iv.value_id)
                        .is_some_and(|&o| o.0 == d.0 && iv.start < src_iv.end && src_iv.start < iv.end)
                });
                if std::env::var("CCC_DEBUG_FPCOAL").is_ok() {
                    eprintln!("[FPCOAL] phi={} (d{}) src={} (d{}) conflict={}", phi_dest, d.0, backedge_src, s.0, conflict);
                }
                if !conflict {
                    assignments.insert(backedge_src, d);
                }
            }
        }
    }

    // Phase 2b: Caller-saved registers for call-spanning values with
    // per-call selective save/restore. Unlike Phase 2 (non-call-spanning only),
    // Phase 2b allows call-spanning values in caller-saved registers by
    // recording their live intervals. The codegen saves/restores each register
    // only at call sites where the value is actually live.
    let mut caller_save_spans: FxHashMap<u8, Vec<(u32, u32)>> = FxHashMap::default();
    if !config.caller_saved_regs.is_empty() && std::env::var("CCC_CALLER_SAVE_SPANNING").is_ok() {
        // Use all available caller-saved registers (not just r10/r11)
        let span_regs: Vec<PhysReg> = config.caller_saved_regs.iter()
            .filter(|r| !used_regs_set.contains(&r.0)) // exclude any already used as callee-saved
            .filter(|r| !phase2_caller_used.contains(&r.0)) // exclude Phase 2 allocations
            .copied()
            .collect();

        if !span_regs.is_empty() {
            let phase2b_intervals: Vec<LiveInterval> = liveness
                .intervals
                .iter()
                .filter(|iv| eligible.contains(&iv.value_id))
                .filter(|iv| iv.end > iv.start)
                .filter(|iv| !assignments.contains_key(&iv.value_id))
                .filter(|iv| spans_any_call(iv, call_points))
                .take(500) // Limit to top 500 candidates to avoid O(n²) in linear scan
                .copied()
                .collect();

            if !phase2b_intervals.is_empty() {
                // Build interval map for live-at-call checks
                let interval_map: FxHashMap<u32, (u32, u32)> = phase2b_intervals.iter()
                    .map(|iv| (iv.value_id, (iv.start, iv.end)))
                    .collect();

                // Lightweight range builder — O(n) instead of O(n × instructions).
                // The full build_live_ranges scans ALL function instructions for
                // use-site data, which is too slow for 4000+ Phase 2b intervals
                // in large functions like sqlite3VdbeExec.
                let mut phase2b_ranges: Vec<live_range::LiveRange> = phase2b_intervals.iter()
                    .map(|iv| {
                        let mut r = live_range::LiveRange::from_interval(*iv, 0);
                        // Priority: inverse range length (shorter ranges = higher priority)
                        let len = (iv.end - iv.start).max(1) as u64;
                        r.priority = 1_000_000 / len;
                        r.calculate_spill_weight();
                        r
                    })
                    .collect();
                phase2b_ranges.sort_by(|a, b| a.start.cmp(&b.start).then(b.priority.cmp(&a.priority)));
                let mut span_allocator =
                    LinearScanAllocator::new(phase2b_ranges, span_regs);
                span_allocator.run();

                for (vid, reg) in span_allocator.assignments {
                    assignments.insert(vid, reg);
                    // Record the interval for this register for selective save/restore
                    if let Some(&(start, end)) = interval_map.get(&vid) {
                        caller_save_spans.entry(reg.0).or_default().push((start, end));
                    }
                    // Do NOT add to used_regs_set (not prologue-saved)
                }
            }
        }
    }

    // Point-precise copy-web coalescing: for copy edges where exactly one
    // side has an FP register and interval-level checks blocked the merge,
    // walk per-block live sets — the pair can share the register iff no
    // program point has both live (the copy points themselves excepted).
    // Interval liveness over-reports interference inside loops (the phi is
    // "live for the whole loop" while its replacement is computed near the
    // backedge), which is what blocks mandelbrot's middle-loop shuffles.
    // CCC_NO_WEB_POINT disables; CCC_WEB_DIAG prints the decisions.
    if arm_fp_pool && std::env::var("CCC_NO_WEB_POINT").is_err() && !liveness.block_live_in.is_empty() {
        let diag = std::env::var("CCC_WEB_DIAG").is_ok();
        let dense = &liveness.dense_of_value;
        for _round in 0..8 {
            let mut any = false;
            for (bi, block) in func.blocks.iter().enumerate() {
                for inst in &block.instructions {
                    if let Instruction::Copy { dest, src: Operand::Value(src_v) } = inst {
                        let is_fp = |v: u32| f64_value_set.contains(&v) || vector_values.contains(&v);
                        if !is_fp(dest.0) || !is_fp(src_v.0) {
                            continue;
                        }
                        let d_reg = assignments.get(&dest.0).copied().filter(|r| (32..=55).contains(&r.0));
                        let s_reg = assignments.get(&src_v.0).copied().filter(|r| (32..=55).contains(&r.0));
                        let (a, u, reg) = match (d_reg, s_reg) {
                            (Some(_), Some(_)) | (None, None) => continue,
                            (Some(r), None) => (dest.0, src_v.0, r),
                            (None, Some(r)) => (src_v.0, dest.0, r),
                        };
                        // A is the registered value, U the slot-homed one.
                        // Caller-saved FP registers cannot hold call-spanning values.
                        if reg.0 >= 40 {
                            let spans = liveness.intervals.iter().any(|iv| {
                                iv.value_id == u && call_points.iter().any(|&cp| iv.start < cp && cp < iv.end)
                            });
                            if spans {
                                continue;
                            }
                        }
                        let (Some(&da), Some(&du)) = (dense.get(&a), dense.get(&u)) else { continue };
                        let mut point_conflict = false;
                        'blocks: for (bj, b) in func.blocks.iter().enumerate() {
                            let mut live_a = liveness.block_live_out[bj].contains(da);
                            let mut live_u = liveness.block_live_out[bj].contains(du);
                            if term_uses(b, a) {
                                live_a = true;
                            }
                            if term_uses(b, u) {
                                live_u = true;
                            }
                            for inst2 in b.instructions.iter().rev() {
                                let is_pair_copy = matches!(inst2,
                                    Instruction::Copy { dest, src: Operand::Value(s) }
                                    if (dest.0 == a && s.0 == u) || (dest.0 == u && s.0 == a));
                                if live_a && live_u && !is_pair_copy {
                                    point_conflict = true;
                                    break 'blocks;
                                }
                                // A non-pair def of either value overwrites the
                                // shared register while the other is live after
                                // it — a clobber even when the def is dead.
                                if !is_pair_copy {
                                    if let Some(d) = inst2.dest() {
                                        if (d.0 == u && live_a) || (d.0 == a && live_u) {
                                            point_conflict = true;
                                            break 'blocks;
                                        }
                                    }
                                }
                                if let Some(d) = inst2.dest() {
                                    if d.0 == a {
                                        live_a = false;
                                    }
                                    if d.0 == u {
                                        live_u = false;
                                    }
                                }
                                let mut uses: Vec<u32> = Vec::new();
                                inst2.for_each_used_value(|v| uses.push(v));
                                for v in uses {
                                    if v == a {
                                        live_a = true;
                                    }
                                    if v == u {
                                        live_u = true;
                                    }
                                }
                            }
                            if !point_conflict
                                && liveness.block_live_in[bj].contains(da)
                                && liveness.block_live_in[bj].contains(du)
                            {
                                point_conflict = true;
                            }
                        }
                        if diag && !point_conflict {
                            eprintln!("[WEBPOINT] func={} merge v{} into d{}", func.name, u, reg.0);
                        }
                        if !point_conflict {
                            assignments.insert(u, reg);
                            any = true;
                        }
                    }
                }
            }
            if !any {
                break;
            }
        }
    }

    let mut used_regs: Vec<PhysReg> = used_regs_set.iter().map(|&r| PhysReg(r)).collect();
    used_regs.sort_by_key(|r| r.0);

    if std::env::var("CCC_DEBUG_REGDUMP").is_ok() {
        let mut by_reg: Vec<(u8, u32)> = assignments.iter().map(|(&v, &r)| (r.0, v)).collect();
        by_reg.sort();
        eprintln!("[REGDUMP] func={} assigned={} liveness_intervals={}", func.name, assignments.len(), liveness.intervals.len());
        for (r, v) in by_reg {
            let iv = liveness.intervals.iter().find(|iv| iv.value_id == v);
            eprintln!("[REGDUMP]   reg {:>2} -> v{} interval={:?}", r, v, iv.map(|i| (i.start, i.end)));
        }
    }

    // Verify: no two assigned values should have overlapping live intervals
    // in the same physical register.
    if std::env::var("CCC_VERIFY_REGALLOC").is_ok() {
        let mut reg_intervals: std::collections::HashMap<u8, Vec<(u32, u32, u32)>> = std::collections::HashMap::new();
        for iv in &liveness.intervals {
            if let Some(&reg) = assignments.get(&iv.value_id) {
                reg_intervals.entry(reg.0).or_default().push((iv.start, iv.end, iv.value_id));
            }
        }
        for (&reg_id, intervals) in &reg_intervals {
            for i in 0..intervals.len() {
                for j in (i+1)..intervals.len() {
                    let (s1, e1, v1) = intervals[i];
                    let (s2, e2, v2) = intervals[j];
                    if s1 < e2 && s2 < e1 {
                        eprintln!("[REGALLOC-OVERLAP] reg={} val{}[{}-{}] overlaps val{}[{}-{}]",
                            reg_id, v1, s1, e1, v2, s2, e2);
                    }
                }
            }
        }
    }

    if std::env::var("CCC_DEBUG_REGALLOC").is_ok() && eligible.len() > 50 {
        let total_eligible = eligible.len();
        let total_assigned = assignments.len();
        let total_intervals = liveness.intervals.len();
        let non_call_spanning = liveness.intervals.iter()
            .filter(|iv| eligible.contains(&iv.value_id) && !spans_any_call(iv, call_points) && iv.end > iv.start)
            .count();
        let call_spanning = liveness.intervals.iter()
            .filter(|iv| eligible.contains(&iv.value_id) && spans_any_call(iv, call_points) && iv.end > iv.start)
            .count();
        eprintln!("[REGALLOC] {} eligible, {} assigned ({:.0}%), {} call-spanning, {} non-call, {} callee, {} caller",
            total_eligible, total_assigned,
            if total_eligible > 0 { total_assigned as f64 / total_eligible as f64 * 100.0 } else { 0.0 },
            call_spanning, non_call_spanning,
            config.available_regs.len(), config.caller_saved_regs.len());
    }

    RegAllocResult {
        assignments,
        used_regs,
        caller_save_spans,
        liveness: Some(liveness),
    }
}

/// Collect values whose types don't fit in a single GPR (floats, i128, and
/// on 32-bit targets: i64/u64). Copy instructions that chain from these
/// values must also be excluded via fixpoint propagation.
fn collect_non_gpr_values(func: &IrFunction, is_32bit: bool, arm_fp_pool: bool) -> FxHashSet<u32> {
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
                    if op.produces_vector_value() {
                        non_gpr_values.insert(d.0);
                    } else if arm_fp_pool && matches!(op,
                        crate::ir::intrinsics::IntrinsicOp::SqrtF64
                        | crate::ir::intrinsics::IntrinsicOp::FabsF64
                        | crate::ir::intrinsics::IntrinsicOp::SqrtF32
                        | crate::ir::intrinsics::IntrinsicOp::FabsF32)
                    {
                        // Scalar float intrinsics produce FP values; keep them
                        // out of GPR allocation so they can use FP registers.
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

/// Collect SSA values that hold 128/256-bit vector data: destinations of
/// vector-producing intrinsics plus any Copy destinations whose source is a
/// vector value (iterated to fixpoint, mirroring the non-GPR propagation).
/// Used on AArch64 to allocate NEON registers to vector values (the Phase 3
/// FP/vector pool maps allocator IDs 40..47 to v16..v23).
fn collect_vector_values(func: &IrFunction) -> FxHashSet<u32> {
    let mut vector_values: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Intrinsic { dest: Some(d), op, .. } = inst {
                if op.produces_vector_value() {
                    vector_values.insert(d.0);
                }
            }
        }
    }
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                    if !vector_values.contains(&dest.0) && vector_values.contains(&src_val.0) {
                        vector_values.insert(dest.0);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    vector_values
}

/// Collect SSA values that hold scalar F64 data: destinations of F64-typed
/// instructions (BinOp/UnaryOp/Load/Cast), F64-returning intrinsics, and
/// F64 constants — plus, iteratively, any Copy whose source is F64.  The copy
/// propagation is what makes loop-carried F64 accumulators (lowered to Copy
/// form after phi elimination) visible to the Phase 3 FP register scan.
fn collect_f64_values(func: &IrFunction) -> FxHashSet<u32> {
    let mut f64_values: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::BinOp { dest, ty, .. }
                | Instruction::UnaryOp { dest, ty, .. }
                | Instruction::Load { dest, ty, .. } if *ty == IrType::F64 => {
                    f64_values.insert(dest.0);
                }
                Instruction::Cast { dest, to_ty, .. } if *to_ty == IrType::F64 => {
                    f64_values.insert(dest.0);
                }
                Instruction::Copy { dest, src: Operand::Const(IrConst::F64(_)) } => {
                    f64_values.insert(dest.0);
                }
                Instruction::Intrinsic { dest: Some(d), op, .. }
                    if matches!(op, crate::ir::intrinsics::IntrinsicOp::SqrtF64
                        | crate::ir::intrinsics::IntrinsicOp::FabsF64) =>
                {
                    f64_values.insert(d.0);
                }
                _ => {}
            }
        }
    }
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                    if !f64_values.contains(&dest.0) && f64_values.contains(&src_val.0) {
                        f64_values.insert(dest.0);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    f64_values
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
///
/// Also used by stack-layout copy coalescing: the same proof (phi dest not used
/// after the backedge source is defined) makes sharing a *stack slot* safe.
pub(crate) fn detect_phi_coalesce_groups(
    func: &IrFunction,
    liveness: &LivenessResult,
    permit_cross_block_src: bool,
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
    //
    // A phi dest may have several such copies (loop-entry initialization from
    // one variable plus the true backedge update, or multiple latches via
    // `continue`). All pairs are returned: the per-pair safety checks below
    // plus the consumers' own conflict checks (register assignment overlap,
    // phi-web interference) make multi-pair coalescing sound. Without this,
    // the first copy found (often the entry copy) blocked coalescing of the
    // true backedge copy — e.g. struct_copy's FP accumulator, where the entry
    // copy (183 <- 182) shadowed the backedge copy (183 <- 74).
    let mut groups: Vec<(u32, u32)> = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let depth = liveness.block_loop_depth.get(block_idx).copied().unwrap_or(0);
        if depth == 0 {
            continue;
        }

        for inst in &block.instructions {
            if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                if multi_def.contains(&dest.0) {
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
                                        if std::env::var("CCC_DEBUG_PHI_COALESCE").is_ok() {
                                            eprintln!("[PHI_COALESCE]   used_after hit: dest=v{} src=v{} block={} inst={:?}",
                                                dest.0, src_val.0, block_idx, inst2);
                                        }
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
                                            if std::env::var("CCC_DEBUG_PHI_COALESCE").is_ok() {
                                                eprintln!("[PHI_COALESCE]   used_after hit (src block {}): dest=v{} src=v{} inst={:?}",
                                                    sdb, dest.0, src_val.0, inst2);
                                            }
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
                        // Register coalescing consumers (which verify interval
                        // conflicts themselves) pass permit_cross_block_src to
                        // skip this conservative proxy.
                        let src_has_cross_block_use = !permit_cross_block_src && src_use_blocks
                            .get(&src_val.0)
                            .map(|blocks| blocks.iter().any(|&b| b != block_idx))
                            .unwrap_or(false);

                        if !phi_dest_used_after_src && !src_has_cross_block_use {
                            if std::env::var("CCC_DEBUG_PHI_COALESCE").is_ok() {
                                eprintln!("[PHI_COALESCE] Coalescing phi_dest=Value({}) with backedge_src=Value({}) in block {}",
                                    dest.0, src_val.0, block_idx);
                            }
                            groups.push((dest.0, src_val.0));
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

/// Check if a block terminator uses a given value ID.
fn term_uses(block: &crate::ir::reexports::BasicBlock, val_id: u32) -> bool {
    match &block.terminator {
        Terminator::CondBranch { cond, .. } => matches!(cond, Operand::Value(v) if v.0 == val_id),
        Terminator::Return(Some(op)) => matches!(op, Operand::Value(v) if v.0 == val_id),
        Terminator::Switch { val, .. } => matches!(val, Operand::Value(v) if v.0 == val_id),
        _ => false,
    }
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
