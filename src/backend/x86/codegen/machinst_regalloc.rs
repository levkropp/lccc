//! Register allocation for MachInst sequences.
//!
//! Builds live intervals from a MachInst buffer, runs the existing linear scan
//! allocator, and rewrites virtual registers to physical registers. Spilled
//! values get load/store instructions inserted around their use/def points.

use crate::backend::regalloc::PhysReg;
use crate::backend::live_range::{LinearScanAllocator, LiveRange};
use crate::backend::liveness::LiveInterval;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use super::machinst::*;

/// Result of MachInst register allocation.
pub struct MachAllocResult {
    /// Map from vreg ID → assigned physical register.
    pub assignments: FxHashMap<u32, PhysReg>,
    /// Map from vreg ID → spill slot offset (negative from frame pointer).
    pub spill_slots: FxHashMap<u32, i64>,
    /// Physical registers actually used (for callee-save tracking).
    pub used_regs: FxHashSet<PhysReg>,
}

/// Record a vreg use at the given program point.
fn use_reg(reg: &MachReg, point: u32, uses: &mut FxHashMap<u32, u32>) {
    if let MachReg::Vreg(id) = reg {
        uses.entry(*id).and_modify(|p| *p = (*p).max(point)).or_insert(point);
    }
}

/// Record a vreg def at the given program point.
fn def_reg(reg: &MachReg, point: u32, defs: &mut FxHashMap<u32, u32>, uses: &mut FxHashMap<u32, u32>) {
    if let MachReg::Vreg(id) = reg {
        defs.entry(*id).or_insert(point);
        uses.entry(*id).and_modify(|p| *p = (*p).max(point)).or_insert(point);
    }
}

/// Record vreg uses in a MachOperand.
fn use_operand(op: &MachOperand, point: u32, uses: &mut FxHashMap<u32, u32>) {
    match op {
        MachOperand::Reg(r) => use_reg(r, point, uses),
        MachOperand::Mem { base, .. } => use_reg(base, point, uses),
        MachOperand::MemIndex { base, index, .. } => {
            use_reg(base, point, uses);
            use_reg(index, point, uses);
        }
        _ => {}
    }
}

/// Collect all virtual register references from a MachInst, recording
/// defs and uses at the given program point.
fn collect_vreg_refs(
    inst: &MachInst,
    point: u32,
    defs: &mut FxHashMap<u32, u32>,
    uses: &mut FxHashMap<u32, u32>,
) {
    match inst {
        MachInst::Mov { src, dst, .. } => {
            use_operand(src, point, uses);
            if let MachOperand::Reg(r) = dst { def_reg(r, point, defs, uses); }
            else { use_operand(dst, point, uses); }
        }
        MachInst::Alu { src, dst, .. } => {
            use_operand(src, point, uses);
            use_reg(dst, point, uses);
            def_reg(dst, point, defs, uses);
        }
        MachInst::Imul3 { src, dst, .. } => {
            use_reg(src, point, uses);
            def_reg(dst, point, defs, uses);
        }
        MachInst::Neg { dst, .. } | MachInst::Not { dst, .. } => {
            use_reg(dst, point, uses);
            def_reg(dst, point, defs, uses);
        }
        MachInst::Shift { amount, dst, .. } => {
            use_operand(amount, point, uses);
            use_reg(dst, point, uses);
            def_reg(dst, point, defs, uses);
        }
        MachInst::Lea { base, index, dst, .. } => {
            use_reg(base, point, uses);
            if let Some((idx, _)) = index { use_reg(idx, point, uses); }
            def_reg(dst, point, defs, uses);
        }
        MachInst::Div { divisor, .. } => {
            use_operand(divisor, point, uses);
        }
        MachInst::Cmp { lhs, rhs, .. } | MachInst::Test { lhs, rhs, .. } => {
            use_operand(lhs, point, uses);
            use_operand(rhs, point, uses);
        }
        MachInst::SetCC { dst, .. } => {
            def_reg(dst, point, defs, uses);
        }
        MachInst::Movzx { src, dst, .. } | MachInst::Movsx { src, dst, .. } => {
            use_reg(src, point, uses);
            def_reg(dst, point, defs, uses);
        }
        MachInst::Cmov { src, dst, .. } => {
            use_operand(src, point, uses);
            use_reg(dst, point, uses);
            def_reg(dst, point, defs, uses);
        }
        MachInst::CallIndirect { reg, .. } => {
            use_reg(reg, point, uses);
        }
        MachInst::Cqto { .. } | MachInst::XorRdx | MachInst::Call { .. }
        | MachInst::Jcc { .. } | MachInst::Jmp { .. } | MachInst::Label(_)
        | MachInst::Ret | MachInst::Raw(_) => {}
    }
}

/// Build live intervals from a MachInst sequence.
fn build_intervals(insts: &[MachInst]) -> Vec<LiveInterval> {
    let mut defs: FxHashMap<u32, u32> = FxHashMap::default();
    let mut last_uses: FxHashMap<u32, u32> = FxHashMap::default();

    for (point, inst) in insts.iter().enumerate() {
        collect_vreg_refs(inst, point as u32, &mut defs, &mut last_uses);
    }

    let mut intervals: Vec<LiveInterval> = Vec::new();
    for (&vreg_id, &def_point) in &defs {
        let end_point = last_uses.get(&vreg_id).copied().unwrap_or(def_point);
        intervals.push(LiveInterval {
            start: def_point,
            end: end_point,
            value_id: vreg_id,
        });
    }

    // Sort by start point (required by linear scan)
    intervals.sort_by_key(|iv| iv.start);
    intervals
}

/// Record a pre-colored physreg at a program point.
fn record_fixed(reg: &MachReg, point: u32, fixed: &mut FxHashMap<PhysReg, Vec<(u32, u32)>>) {
    if let MachReg::Phys(r) = reg {
        fixed.entry(*r).or_default().push((point, point));
    }
}

/// Record pre-colored physregs in a MachOperand.
fn record_fixed_op(op: &MachOperand, point: u32, fixed: &mut FxHashMap<PhysReg, Vec<(u32, u32)>>) {
    match op {
        MachOperand::Reg(r @ MachReg::Phys(_)) => record_fixed(r, point, fixed),
        MachOperand::Mem { base: r @ MachReg::Phys(_), .. } => record_fixed(r, point, fixed),
        _ => {}
    }
}

/// Collect program points where pre-colored physical registers are used.
/// These create fixed intervals that block allocation of those registers.
fn collect_fixed_intervals(insts: &[MachInst]) -> FxHashMap<PhysReg, Vec<(u32, u32)>> {
    let mut fixed: FxHashMap<PhysReg, Vec<(u32, u32)>> = FxHashMap::default();

    for (point, inst) in insts.iter().enumerate() {
        let p = point as u32;
        match inst {
            MachInst::Mov { src, dst, .. } => {
                record_fixed_op(src, p, &mut fixed);
                if let MachOperand::Reg(r) = dst { record_fixed(r, p, &mut fixed); }
            }
            MachInst::Alu { src, dst, .. } => {
                record_fixed_op(src, p, &mut fixed);
                record_fixed(dst, p, &mut fixed);
            }
            MachInst::Shift { amount, dst, .. } => {
                record_fixed_op(amount, p, &mut fixed);
                record_fixed(dst, p, &mut fixed);
            }
            MachInst::Div { divisor, .. } => {
                record_fixed_op(divisor, p, &mut fixed);
                fixed.entry(RAX).or_default().push((p, p));
                fixed.entry(RDX).or_default().push((p, p));
            }
            MachInst::Cqto { .. } => {
                fixed.entry(RAX).or_default().push((p, p));
                fixed.entry(RDX).or_default().push((p, p));
            }
            MachInst::XorRdx => {
                fixed.entry(RDX).or_default().push((p, p));
            }
            MachInst::Call { .. } | MachInst::CallIndirect { .. } => {
                for &r in &[RAX, RCX, RDX, R8, R9, R10, R11, RDI, RSI] {
                    fixed.entry(r).or_default().push((p, p));
                }
            }
            _ => {}
        }
    }

    fixed
}

/// Collect all physical registers that appear pre-colored in the buffer.
/// These must be excluded from the allocatable pool to avoid conflicts.
fn collect_precolored_regs(insts: &[MachInst]) -> FxHashSet<PhysReg> {
    let mut used = FxHashSet::default();
    for inst in insts {
        visit_regs_shared(inst, &mut |r: &MachReg| {
            if let MachReg::Phys(p) = r { used.insert(*p); }
        });
    }
    used
}

/// Visit all MachReg references in a MachInst (shared-borrow version for internal use).
fn visit_regs_shared(inst: &MachInst, f: &mut impl FnMut(&MachReg)) {
    let mut visit_op = |op: &MachOperand, f: &mut dyn FnMut(&MachReg)| {
        match op {
            MachOperand::Reg(r) => f(r),
            MachOperand::Mem { base, .. } => f(base),
            MachOperand::MemIndex { base, index, .. } => { f(base); f(index); }
            _ => {}
        }
    };
    match inst {
        MachInst::Mov { src, dst, .. } => {
            visit_op(src, f);
            if let MachOperand::Reg(r) = dst { f(r); } else { visit_op(dst, f); }
        }
        MachInst::Alu { src, dst, .. } => { visit_op(src, f); f(dst); }
        MachInst::Imul3 { src, dst, .. } => { f(src); f(dst); }
        MachInst::Neg { dst, .. } | MachInst::Not { dst, .. } => { f(dst); }
        MachInst::Shift { amount, dst, .. } => { visit_op(amount, f); f(dst); }
        MachInst::Lea { base, index, dst, .. } => {
            f(base);
            if let Some((r, _)) = index { f(r); }
            f(dst);
        }
        MachInst::Div { divisor, .. } => { visit_op(divisor, f); }
        MachInst::Cmp { lhs, rhs, .. } | MachInst::Test { lhs, rhs, .. } => {
            visit_op(lhs, f); visit_op(rhs, f);
        }
        MachInst::SetCC { dst, .. } => { f(dst); }
        MachInst::Movzx { src, dst, .. } | MachInst::Movsx { src, dst, .. } => {
            f(src); f(dst);
        }
        MachInst::Cmov { src, dst, .. } => { visit_op(src, f); f(dst); }
        MachInst::CallIndirect { reg, .. } => { f(reg); }
        _ => {}
    }
}

/// Run register allocation on a MachInst buffer.
///
/// Uses only caller-saved registers that don't appear pre-colored in the
/// buffer. This is conservative (avoids all conflicts with Phase 1
/// allocations) but correct and sufficient for block-level allocation.
///
/// `existing_slots` maps vreg IDs to their existing stack slot offsets
/// (from the main codegen's stack layout). Spilled vregs reuse these
/// instead of creating new spill slots.
pub fn allocate_machinsts(
    insts: &[MachInst],
    available_regs: &[PhysReg],
    existing_slots: &FxHashMap<u32, i64>,
) -> MachAllocResult {
    let intervals = build_intervals(insts);

    if intervals.is_empty() {
        return MachAllocResult {
            assignments: FxHashMap::default(),
            spill_slots: FxHashMap::default(),
            used_regs: FxHashSet::default(),
        };
    }

    // Collect all pre-colored physical registers in the buffer.
    // Exclude them from the allocatable pool to avoid conflicts.
    let precolored = collect_precolored_regs(insts);
    let allocatable: Vec<PhysReg> = available_regs
        .iter()
        .copied()
        .filter(|r| !precolored.contains(r))
        .filter(|r| !matches!(r.0, 0 | 7 | 6)) // Exclude rax, rcx, rbp
        .collect();

    if allocatable.is_empty() {
        // No registers available — all vregs stay as-is (will be stack-resolved)
        return MachAllocResult {
            assignments: FxHashMap::default(),
            spill_slots: FxHashMap::default(),
            used_regs: FxHashSet::default(),
        };
    }

    // Build LiveRange entries for the linear scan
    let ranges: Vec<LiveRange> = intervals
        .iter()
        .map(|iv| LiveRange::from_interval(*iv, 0))
        .collect();

    let mut allocator = LinearScanAllocator::new(ranges.clone(), allocatable);
    allocator.init_registers();

    for range in ranges {
        allocator.allocate_range(range);
    }

    // Collect results, mapping spill slots to existing stack slots
    let mut used_regs = FxHashSet::default();
    for &reg in allocator.assignments.values() {
        used_regs.insert(reg);
    }

    let mut spill_slots: FxHashMap<u32, i64> = FxHashMap::default();
    for (&vid, _) in &allocator.spill_slots {
        if let Some(&slot) = existing_slots.get(&vid) {
            spill_slots.insert(vid, slot);
        }
        // If no existing slot, leave unmapped — resolve_stack_vregs handles it
    }

    MachAllocResult {
        assignments: allocator.assignments,
        spill_slots,
        used_regs,
    }
}

/// Rewrite a MachReg using the allocation result.
fn rewrite_reg(reg: &MachReg, result: &MachAllocResult) -> MachReg {
    match reg {
        MachReg::Vreg(id) => {
            if let Some(&phys) = result.assignments.get(id) {
                MachReg::Phys(phys)
            } else {
                // Spilled — leave as vreg (will be handled by spill insertion)
                *reg
            }
        }
        MachReg::Phys(_) => *reg,
    }
}

/// Rewrite a MachOperand using the allocation result.
fn rewrite_operand(op: &MachOperand, result: &MachAllocResult) -> MachOperand {
    match op {
        MachOperand::Reg(r) => {
            let new_r = rewrite_reg(r, result);
            if let MachReg::Vreg(id) = new_r {
                // Spilled value — replace with stack slot access
                if let Some(&slot) = result.spill_slots.get(&id) {
                    return MachOperand::StackSlot(slot);
                }
            }
            MachOperand::Reg(new_r)
        }
        MachOperand::Mem { base, offset } => MachOperand::Mem {
            base: rewrite_reg(base, result),
            offset: *offset,
        },
        MachOperand::MemIndex { base, index, scale, offset } => MachOperand::MemIndex {
            base: rewrite_reg(base, result),
            index: rewrite_reg(index, result),
            scale: *scale,
            offset: *offset,
        },
        _ => op.clone(), // Imm, StackSlot, RipRel — unchanged
    }
}

/// Check if a MachReg is a spilled vreg and return its stack slot.
fn spill_slot(reg: &MachReg, result: &MachAllocResult) -> Option<i64> {
    if let MachReg::Vreg(id) = rewrite_reg(reg, result) {
        result.spill_slots.get(&id).copied()
    } else {
        None
    }
}

/// Resolve a MachReg source: if spilled, emit a load to rax and return Phys(RAX).
/// If not spilled, return the rewritten reg directly.
fn resolve_src(reg: &MachReg, size: OpSize, result: &MachAllocResult, out: &mut Vec<MachInst>) -> MachReg {
    if let Some(slot) = spill_slot(reg, result) {
        out.push(MachInst::Mov {
            src: MachOperand::StackSlot(slot),
            dst: MachOperand::Reg(MachReg::Phys(RAX)),
            size,
        });
        MachReg::Phys(RAX)
    } else {
        rewrite_reg(reg, result)
    }
}

/// Rewrite all virtual registers in a MachInst buffer using allocation results.
/// Also inserts spill loads/stores for spilled operands that can't be represented
/// as StackSlot memory operands (e.g., when the instruction needs a register).
pub fn rewrite_machinsts(insts: &[MachInst], result: &MachAllocResult) -> Vec<MachInst> {
    let mut out = Vec::with_capacity(insts.len());

    for inst in insts {
        match inst {
            MachInst::Mov { src, dst, size } => {
                out.push(MachInst::Mov {
                    src: rewrite_operand(src, result),
                    dst: rewrite_operand(dst, result),
                    size: *size,
                });
            }
            MachInst::Alu { op, src, dst, size } => {
                if let Some(slot) = spill_slot(dst, result) {
                    out.push(MachInst::Mov {
                        src: MachOperand::StackSlot(slot),
                        dst: MachOperand::Reg(MachReg::Phys(RAX)),
                        size: *size,
                    });
                    out.push(MachInst::Alu {
                        op: *op,
                        src: rewrite_operand(src, result),
                        dst: MachReg::Phys(RAX),
                        size: *size,
                    });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(slot),
                        size: *size,
                    });
                    continue;
                }
                out.push(MachInst::Alu {
                    op: *op,
                    src: rewrite_operand(src, result),
                    dst: rewrite_reg(dst, result),
                    size: *size,
                });
            }
            MachInst::Imul3 { imm, src, dst, size } => {
                if let Some(dst_slot) = spill_slot(dst, result) {
                    // Imul3: dst = src * imm. src can be memory, dst must be reg.
                    let new_src = resolve_src(src, *size, result, &mut out);
                    out.push(MachInst::Imul3 {
                        imm: *imm,
                        src: new_src,
                        dst: MachReg::Phys(RAX),
                        size: *size,
                    });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(dst_slot),
                        size: *size,
                    });
                    continue;
                }
                let new_src = resolve_src(src, *size, result, &mut out);
                out.push(MachInst::Imul3 {
                    imm: *imm,
                    src: new_src,
                    dst: rewrite_reg(dst, result),
                    size: *size,
                });
            }
            MachInst::Neg { dst, size } => {
                if let Some(slot) = spill_slot(dst, result) {
                    out.push(MachInst::Mov {
                        src: MachOperand::StackSlot(slot),
                        dst: MachOperand::Reg(MachReg::Phys(RAX)),
                        size: *size,
                    });
                    out.push(MachInst::Neg { dst: MachReg::Phys(RAX), size: *size });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(slot),
                        size: *size,
                    });
                    continue;
                }
                out.push(MachInst::Neg { dst: rewrite_reg(dst, result), size: *size });
            }
            MachInst::Not { dst, size } => {
                if let Some(slot) = spill_slot(dst, result) {
                    out.push(MachInst::Mov {
                        src: MachOperand::StackSlot(slot),
                        dst: MachOperand::Reg(MachReg::Phys(RAX)),
                        size: *size,
                    });
                    out.push(MachInst::Not { dst: MachReg::Phys(RAX), size: *size });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(slot),
                        size: *size,
                    });
                    continue;
                }
                out.push(MachInst::Not { dst: rewrite_reg(dst, result), size: *size });
            }
            MachInst::Shift { op, amount, dst, size } => {
                if let Some(slot) = spill_slot(dst, result) {
                    out.push(MachInst::Mov {
                        src: MachOperand::StackSlot(slot),
                        dst: MachOperand::Reg(MachReg::Phys(RAX)),
                        size: *size,
                    });
                    out.push(MachInst::Shift {
                        op: *op,
                        amount: rewrite_operand(amount, result),
                        dst: MachReg::Phys(RAX),
                        size: *size,
                    });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(slot),
                        size: *size,
                    });
                    continue;
                }
                out.push(MachInst::Shift {
                    op: *op,
                    amount: rewrite_operand(amount, result),
                    dst: rewrite_reg(dst, result),
                    size: *size,
                });
            }
            MachInst::Lea { base, index, offset, dst } => {
                if let Some(dst_slot) = spill_slot(dst, result) {
                    let new_base = resolve_src(base, OpSize::S64, result, &mut out);
                    let new_index = index.as_ref().map(|(r, s)| {
                        // If index is also spilled, we can't load both to rax.
                        // For now, just rewrite (will fail if spilled).
                        (rewrite_reg(r, result), *s)
                    });
                    out.push(MachInst::Lea {
                        base: new_base, index: new_index, offset: *offset,
                        dst: MachReg::Phys(RAX),
                    });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(dst_slot),
                        size: OpSize::S64,
                    });
                    continue;
                }
                let new_base = resolve_src(base, OpSize::S64, result, &mut out);
                out.push(MachInst::Lea {
                    base: new_base,
                    index: index.as_ref().map(|(r, s)| (rewrite_reg(r, result), *s)),
                    offset: *offset,
                    dst: rewrite_reg(dst, result),
                });
            }
            MachInst::Div { divisor, signed, size } => {
                out.push(MachInst::Div {
                    divisor: rewrite_operand(divisor, result),
                    signed: *signed,
                    size: *size,
                });
            }
            MachInst::Cmp { lhs, rhs, size } => {
                out.push(MachInst::Cmp {
                    lhs: rewrite_operand(lhs, result),
                    rhs: rewrite_operand(rhs, result),
                    size: *size,
                });
            }
            MachInst::Test { lhs, rhs, size } => {
                out.push(MachInst::Test {
                    lhs: rewrite_operand(lhs, result),
                    rhs: rewrite_operand(rhs, result),
                    size: *size,
                });
            }
            MachInst::SetCC { cc, dst } => {
                out.push(MachInst::SetCC { cc: *cc, dst: rewrite_reg(dst, result) });
            }
            MachInst::Movzx { src, dst, from_size, to_size } => {
                out.push(MachInst::Movzx {
                    src: rewrite_reg(src, result),
                    dst: rewrite_reg(dst, result),
                    from_size: *from_size,
                    to_size: *to_size,
                });
            }
            MachInst::Movsx { src, dst, from_size, to_size } => {
                out.push(MachInst::Movsx {
                    src: rewrite_reg(src, result),
                    dst: rewrite_reg(dst, result),
                    from_size: *from_size,
                    to_size: *to_size,
                });
            }
            MachInst::Cmov { cc, src, dst, size } => {
                if let Some(slot) = spill_slot(dst, result) {
                    // Cmov needs register dst. Load from stack, cmov, store back.
                    out.push(MachInst::Mov {
                        src: MachOperand::StackSlot(slot),
                        dst: MachOperand::Reg(MachReg::Phys(RAX)),
                        size: *size,
                    });
                    out.push(MachInst::Cmov {
                        cc: *cc,
                        src: rewrite_operand(src, result),
                        dst: MachReg::Phys(RAX),
                        size: *size,
                    });
                    out.push(MachInst::Mov {
                        src: MachOperand::Reg(MachReg::Phys(RAX)),
                        dst: MachOperand::StackSlot(slot),
                        size: *size,
                    });
                    continue;
                }
                out.push(MachInst::Cmov {
                    cc: *cc,
                    src: rewrite_operand(src, result),
                    dst: rewrite_reg(dst, result),
                    size: *size,
                });
            }
            MachInst::CallIndirect { reg } => {
                out.push(MachInst::CallIndirect { reg: rewrite_reg(reg, result) });
            }
            // Instructions that don't reference vregs — pass through unchanged
            MachInst::Cqto { .. } | MachInst::XorRdx | MachInst::Call { .. }
            | MachInst::Jcc { .. } | MachInst::Jmp { .. } | MachInst::Label(_)
            | MachInst::Ret | MachInst::Raw(_) => {
                out.push(inst.clone());
            }
        }
    }

    out
}

/// Visit all MachReg references in a MachInst (both defs and uses).
/// Used to collect vregs that need slot lookup for the allocator.
pub fn visit_regs(inst: &MachInst, f: &mut impl FnMut(&MachReg)) {
    visit_regs_shared(inst, f);
}
