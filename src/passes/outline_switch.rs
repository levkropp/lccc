//! Switch case outlining pass.
//!
//! Reduces register pressure in large switch-based functions (like SQLite's
//! `sqlite3VdbeExec` with 170+ cases) by extracting case bodies into separate
//! functions. Each outlined case becomes a small function that takes pointers
//! to the shared local variables as arguments. The original case is replaced
//! with a call to the outlined function.
//!
//! This is effective because:
//! - The original function's register pressure drops dramatically (fewer
//!   simultaneously-live values compete for registers)
//! - Each outlined function is small enough for the existing linear scan
//!   register allocator to handle efficiently
//! - The accumulator round-trip overhead is reduced because more values
//!   fit in registers within the smaller outlined functions
//!
//! The pass only outlines cases that have simple control flow (branch to
//! switch end or return). Cases with cross-case gotos are left in place.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::instruction::{BasicBlock, BlockId, CallInfo, Instruction, Operand, Terminator, Value};
use crate::ir::module::{IrFunction, IrModule, IrParam};

/// Minimum number of switch cases to trigger outlining.
// Disabled: switch case outlining creates separate functions for each case,
// but the dispatch loop holds pOp in a caller-saved register (%r11) which
// gets clobbered by the case function call. This causes the VDBE dispatch
// to read opcodes from garbage memory after each case returns.
// TODO: fix by saving pOp to a callee-saved register or stack slot across case calls.
const MIN_CASES_FOR_OUTLINING: usize = 999999;

/// Minimum number of instructions in a case body to be worth outlining.
/// Very small cases (1-3 instructions) have more call overhead than savings.
const MIN_CASE_INSTRUCTIONS: usize = 4;

/// Maximum number of pointer arguments to pass to an outlined function.
/// x86-64 ABI passes first 6 integer args in registers; args 7+ go on the stack.
/// We allow up to 14 args since even stack-passed args are much cheaper than
/// the shuffle instructions they eliminate in the parent function.
const MAX_OUTLINE_ARGS: usize = 14;

/// Run the switch outlining pass on the entire module.
pub fn run(module: &mut IrModule) -> usize {
    // Compute global max block ID to avoid collisions when creating new blocks.
    let mut global_max_block_id: u32 = 0;
    for func in &module.functions {
        for block in &func.blocks {
            if block.label.0 > global_max_block_id {
                global_max_block_id = block.label.0;
            }
        }
    }

    let mut new_functions: Vec<IrFunction> = Vec::new();
    let mut total_outlined = 0;

    let num_funcs = module.functions.len();
    for func_idx in 0..num_funcs {
        if module.functions[func_idx].is_declaration {
            continue;
        }
        if module.functions[func_idx].blocks.len() < MIN_CASES_FOR_OUTLINING * 2 {
            continue; // Too small to bother
        }

        let outlined = outline_switch_cases(
            &mut module.functions[func_idx],
            &mut new_functions,
            &mut global_max_block_id,
        );
        total_outlined += outlined;
    }

    // Add all new outlined functions to the module.
    module.functions.extend(new_functions);

    total_outlined
}

/// Information about a switch we want to outline cases from.
struct SwitchInfo {
    /// The case entries: (case_value, target_block_id).
    cases: Vec<(i64, BlockId)>,
    /// The "end" block that break statements branch to.
    /// Detected as the most common branch target from case blocks.
    end_block: BlockId,
}

/// Information about a case body that can be outlined.
struct OutlinableCase {
    /// The case value from the switch.
    case_value: i64,
    /// The entry block for this case.
    entry_block: BlockId,
    /// All blocks belonging to this case (including entry).
    blocks: Vec<BlockId>,
    /// External values used inside the case (alloca pointers from the parent function).
    /// These become parameters to the outlined function.
    external_values: Vec<(Value, IrType)>,
    /// Whether this case contains a Return terminator (needs special handling).
    has_return: bool,
    /// External blocks (besides end_block) that the case branches to.
    /// These become additional "exit" targets: the outlined function returns an
    /// index (0 = end_block, 1..N = exit_blocks[0..N-1]) to tell the caller
    /// which block to branch to.
    exit_blocks: Vec<BlockId>,
}

/// Find the Switch terminators in a function that are worth outlining.
fn find_outline_candidates(func: &IrFunction) -> Vec<SwitchInfo> {
    let mut candidates = Vec::new();

    for block in func.blocks.iter() {
        if let Terminator::Switch { cases, default, .. } = &block.terminator {
            if cases.len() < MIN_CASES_FOR_OUTLINING {
                continue;
            }

            // Find the "end" block: the most common branch target from case blocks.
            // In a typical switch, most cases `break` which branches to the end block.
            let end_block = detect_end_block(func, cases, *default);

            candidates.push(SwitchInfo {
                cases: cases.clone(),
                end_block,
            });
        }
    }

    candidates
}

/// Detect the "end" block of a switch — the merge point that most cases branch to.
/// This is typically the block after the switch that `break` statements target.
fn detect_end_block(func: &IrFunction, cases: &[(i64, BlockId)], default: BlockId) -> BlockId {
    // Count how often each block is the target of an unconditional branch from case blocks.
    let mut target_counts: FxHashMap<BlockId, usize> = FxHashMap::default();
    let case_entries: FxHashSet<BlockId> = cases.iter().map(|(_, bid)| *bid).collect();

    for block in &func.blocks {
        if let Terminator::Branch(target) = &block.terminator {
            // Only count branches from blocks that aren't case entries themselves
            // (we want to find the external merge point, not internal case fallthrough)
            *target_counts.entry(*target).or_insert(0) += 1;
        }
    }

    // The end block is the most-targeted block that is NOT a case entry.
    target_counts.into_iter()
        .filter(|(bid, _)| !case_entries.contains(bid) && *bid != default)
        .max_by_key(|(_, count)| *count)
        .map(|(bid, _)| bid)
        .unwrap_or(default)
}

/// Collect the set of blocks belonging to a case body.
///
/// Uses a two-phase approach:
/// 1. Find the contiguous range of blocks between this case's entry and the next
///    case entry in the function's block list. This captures the sequential layout
///    from lowering (including ternary diamonds and other internal control flow).
/// 2. Verify all branches within the collected blocks either target other blocks
///    in the set, the end_block, or are returns. Reject cases with escaping branches.
fn collect_case_blocks(
    func: &IrFunction,
    case_entry: BlockId,
    all_case_entries: &FxHashSet<BlockId>,
    end_block: BlockId,
    block_map: &FxHashMap<BlockId, usize>,
) -> Option<(Vec<BlockId>, Vec<BlockId>)> {
    let entry_idx = *block_map.get(&case_entry)?;

    // Collect blocks starting from entry_idx until we hit another case entry.
    // Don't stop at end_block — it may appear within the case (e.g., when the
    // end_block detection picks a block that's inside a case body).
    let mut case_blocks = Vec::new();
    for idx in entry_idx..func.blocks.len() {
        let bid = func.blocks[idx].label;

        // Stop at another case's entry (but not our own)
        if idx != entry_idx && all_case_entries.contains(&bid) {
            break;
        }

        case_blocks.push(bid);
    }

    if case_blocks.is_empty() {
        return None;
    }

    let case_block_set: FxHashSet<BlockId> = case_blocks.iter().copied().collect();

    // Trim trailing "dead" blocks — blocks that are unreachable from the entry.
    // After a `break` or `return`, the lowerer creates a dead block that branches
    // to the next case's entry. We include these in the initial range but they
    // may branch outside the case. Remove them from the tail.
    while case_blocks.len() > 1 {
        let last = *case_blocks.last().unwrap();
        let idx = block_map[&last];
        let block = &func.blocks[idx];

        // Check if this block escapes the case.
        let escapes = match &block.terminator {
            Terminator::Branch(target) => {
                !case_block_set.contains(target) && *target != end_block
            }
            _ => false,
        };

        if escapes {
            case_blocks.pop();
        } else {
            break;
        }
    }

    // Rebuild the set after trimming.
    let case_block_set: FxHashSet<BlockId> = case_blocks.iter().copied().collect();

    // Collect exit blocks: branch targets outside the case that aren't end_block.
    let mut exit_blocks: Vec<BlockId> = Vec::new();
    let mut exit_set: FxHashSet<BlockId> = FxHashSet::default();

    let check_target = |target: BlockId, exit_set: &mut FxHashSet<BlockId>, exit_blocks: &mut Vec<BlockId>| -> bool {
        if target == end_block || case_block_set.contains(&target) {
            return true;
        }
        // Allow branches to external blocks as additional exits.
        // Limit the number of exits to keep the dispatch manageable.
        if exit_set.len() < 8 || exit_set.contains(&target) {
            if exit_set.insert(target) {
                exit_blocks.push(target);
            }
            return true;
        }
        false // Too many distinct exit targets
    };

    for &bid in &case_blocks {
        let idx = block_map[&bid];
        let block = &func.blocks[idx];

        match &block.terminator {
            Terminator::Branch(target) => {
                if !check_target(*target, &mut exit_set, &mut exit_blocks) {
                    return None;
                }
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                if !check_target(*true_label, &mut exit_set, &mut exit_blocks) {
                    return None;
                }
                if !check_target(*false_label, &mut exit_set, &mut exit_blocks) {
                    return None;
                }
            }
            Terminator::Switch { .. } | Terminator::IndirectBranch { .. } => {
                return None;
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }

    Some((case_blocks, exit_blocks))
}

/// Find all values used in a set of blocks that are defined outside those blocks.
/// These are the values that need to be passed as arguments to the outlined function.
fn find_external_values(
    func: &IrFunction,
    case_blocks: &[BlockId],
    block_map: &FxHashMap<BlockId, usize>,
) -> Vec<(Value, IrType)> {
    // First, collect all values defined inside the case blocks.
    let mut defined_inside: FxHashSet<u32> = FxHashSet::default();
    for &bid in case_blocks {
        if let Some(&idx) = block_map.get(&bid) {
            let block = &func.blocks[idx];
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    defined_inside.insert(dest.0);
                }
            }
        }
    }

    // Collect all values used inside the case blocks.
    let mut used_inside: FxHashMap<u32, IrType> = FxHashMap::default();
    for &bid in case_blocks {
        if let Some(&idx) = block_map.get(&bid) {
            let block = &func.blocks[idx];
            for inst in &block.instructions {
                visit_instruction_uses(inst, |val, ty| {
                    if !defined_inside.contains(&val.0) {
                        used_inside.entry(val.0).or_insert(ty);
                    }
                });
            }
            visit_terminator_uses(&block.terminator, |val, ty| {
                if !defined_inside.contains(&val.0) {
                    used_inside.entry(val.0).or_insert(ty);
                }
            });
        }
    }

    // Sort by value ID for deterministic ordering.
    let mut externals: Vec<(Value, IrType)> = used_inside.into_iter()
        .map(|(id, ty)| (Value(id), ty))
        .collect();
    externals.sort_by_key(|(v, _)| v.0);
    externals
}

/// Visit all Value uses in an instruction, calling f(value, type_hint).
/// The type hint is the best guess at the value's type from context.
fn visit_instruction_uses(inst: &Instruction, mut f: impl FnMut(Value, IrType)) {
    macro_rules! vop {
        ($op:expr, $ty:expr) => {
            if let Operand::Value(v) = $op { f(*v, $ty); }
        };
    }

    match inst {
        Instruction::Store { val, ptr, ty, .. } => {
            vop!(val, *ty);
            f(*ptr, IrType::Ptr);
        }
        Instruction::Load { ptr, .. } => {
            f(*ptr, IrType::Ptr);
        }
        Instruction::BinOp { lhs, rhs, ty, .. } => {
            vop!(lhs, *ty);
            vop!(rhs, *ty);
        }
        Instruction::UnaryOp { src, ty, .. } => {
            vop!(src, *ty);
        }
        Instruction::Cmp { lhs, rhs, ty, .. } => {
            vop!(lhs, *ty);
            vop!(rhs, *ty);
        }
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            if let Instruction::CallIndirect { func_ptr, .. } = inst {
                vop!(func_ptr, IrType::Ptr);
            }
            for (arg, ty) in info.args.iter().zip(info.arg_types.iter()) {
                vop!(arg, *ty);
            }
        }
        Instruction::GetElementPtr { base, offset, ty, .. } => {
            f(*base, IrType::Ptr);
            vop!(offset, *ty);
        }
        Instruction::Cast { src, from_ty, .. } => {
            vop!(src, *from_ty);
        }
        Instruction::Copy { src, .. } => {
            vop!(src, IrType::I64);
        }
        Instruction::Memcpy { dest, src, .. } => {
            f(*dest, IrType::Ptr);
            f(*src, IrType::Ptr);
        }
        Instruction::Select { cond, true_val, false_val, ty, .. } => {
            vop!(cond, IrType::I32);
            vop!(true_val, *ty);
            vop!(false_val, *ty);
        }
        Instruction::Phi { incoming, ty, .. } => {
            for (op, _) in incoming {
                vop!(op, *ty);
            }
        }
        Instruction::VaArg { va_list_ptr, .. } => {
            f(*va_list_ptr, IrType::Ptr);
        }
        Instruction::VaArgStruct { va_list_ptr, dest_ptr, .. } => {
            f(*va_list_ptr, IrType::Ptr);
            f(*dest_ptr, IrType::Ptr);
        }
        Instruction::VaStart { va_list_ptr, .. } => {
            f(*va_list_ptr, IrType::Ptr);
        }
        Instruction::VaEnd { va_list_ptr, .. } => {
            f(*va_list_ptr, IrType::Ptr);
        }
        Instruction::VaCopy { dest_ptr, src_ptr, .. } => {
            f(*dest_ptr, IrType::Ptr);
            f(*src_ptr, IrType::Ptr);
        }
        Instruction::AtomicRmw { ptr, val, ty, .. } => {
            vop!(ptr, IrType::Ptr);
            vop!(val, *ty);
        }
        Instruction::AtomicCmpxchg { ptr, expected, desired, ty, .. } => {
            vop!(ptr, IrType::Ptr);
            vop!(expected, *ty);
            vop!(desired, *ty);
        }
        Instruction::AtomicLoad { ptr, .. } => {
            vop!(ptr, IrType::Ptr);
        }
        Instruction::AtomicStore { ptr, val, ty, .. } => {
            vop!(ptr, IrType::Ptr);
            vop!(val, *ty);
        }
        Instruction::DynAlloca { size, .. } => {
            vop!(size, IrType::I64);
        }
        Instruction::StackRestore { ptr } => {
            f(*ptr, IrType::Ptr);
        }
        Instruction::InlineAsm { inputs, outputs, .. } => {
            for (_, op, _) in inputs {
                vop!(op, IrType::I64);
            }
            for (_, val, _) in outputs {
                f(*val, IrType::Ptr);
            }
        }
        Instruction::Intrinsic { dest_ptr, args, .. } => {
            if let Some(p) = dest_ptr {
                f(*p, IrType::Ptr);
            }
            for arg in args {
                vop!(arg, IrType::I64);
            }
        }
        Instruction::Alloca { .. }
        | Instruction::GlobalAddr { .. }
        | Instruction::LabelAddr { .. }
        | Instruction::Fence { .. }
        | Instruction::StackSave { .. }
        | Instruction::ParamRef { .. }
        | Instruction::GetReturnF64Second { .. }
        | Instruction::GetReturnF32Second { .. }
        | Instruction::GetReturnF128Second { .. } => {}
        Instruction::SetReturnF64Second { src }
        | Instruction::SetReturnF32Second { src }
        | Instruction::SetReturnF128Second { src } => {
            vop!(src, IrType::F64);
        }
    }
}

/// Visit all Value uses in a terminator.
fn visit_terminator_uses(term: &Terminator, mut f: impl FnMut(Value, IrType)) {
    match term {
        Terminator::Return(Some(op)) => {
            if let Operand::Value(v) = op {
                f(*v, IrType::I64);
            }
        }
        Terminator::CondBranch { cond, .. } => {
            if let Operand::Value(v) = cond {
                f(*v, IrType::I32);
            }
        }
        Terminator::Switch { val, .. } => {
            if let Operand::Value(v) = val {
                f(*v, IrType::I64);
            }
        }
        Terminator::IndirectBranch { target, .. } => {
            if let Operand::Value(v) = target {
                f(*v, IrType::Ptr);
            }
        }
        Terminator::Return(None) | Terminator::Branch(_) | Terminator::Unreachable => {}
    }
}

/// Outline eligible switch cases from a single function.
/// Returns the number of cases outlined.
fn outline_switch_cases(
    func: &mut IrFunction,
    new_functions: &mut Vec<IrFunction>,
    global_max_block_id: &mut u32,
) -> usize {
    let candidates = find_outline_candidates(func);
    if candidates.is_empty() {
        return 0;
    }

    let block_map: FxHashMap<BlockId, usize> = func.blocks.iter().enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

    let mut total_outlined = 0;

    let debug = std::env::var("CCC_DEBUG_OUTLINE").is_ok();

    for switch_info in &candidates {
        let all_case_entries: FxHashSet<BlockId> = switch_info.cases.iter()
            .map(|(_, bid)| *bid)
            .collect();

        if debug {
            eprintln!("[OUTLINE] function: {}, switch with {} cases, end_block: {:?}, global_max_block_id: {}",
                func.name, switch_info.cases.len(), switch_info.end_block, *global_max_block_id);
        }

        // Analyze each case for outlinability.
        let mut outlinable: Vec<OutlinableCase> = Vec::new();

        let mut skip_block_collect = 0usize;
        let mut skip_empty = 0usize;
        let mut skip_small = 0usize;
        let mut skip_return = 0usize;
        let mut skip_args = 0usize;

        for &(case_value, case_entry) in &switch_info.cases {
            // Collect blocks belonging to this case.
            let (case_blocks, exit_blocks) = match collect_case_blocks(
                func, case_entry, &all_case_entries, switch_info.end_block, &block_map,
            ) {
                Some(result) => result,
                None => { skip_block_collect += 1; continue; }
            };

            if case_blocks.is_empty() {
                skip_empty += 1;
                continue;
            }

            if debug {
                eprintln!("[OUTLINE]   case {}: entry={:?}, blocks={:?}, ext_count={}",
                    case_value, case_entry, case_blocks,
                    find_external_values(func, &case_blocks, &block_map).len());
            }

            // Count instructions.
            let instruction_count: usize = case_blocks.iter()
                .filter_map(|bid| block_map.get(bid))
                .map(|&idx| func.blocks[idx].instructions.len())
                .sum();

            if instruction_count < MIN_CASE_INSTRUCTIONS {
                skip_small += 1;
                continue;
            }

            // Check if any block has a Return terminator.
            let has_return = case_blocks.iter().any(|bid| {
                block_map.get(bid).map_or(false, |&idx| {
                    matches!(func.blocks[idx].terminator, Terminator::Return(_))
                })
            });
            if has_return {
                skip_return += 1;
                continue;
            }

            // Find external values (defined outside the case, used inside).
            let external_values = find_external_values(func, &case_blocks, &block_map);

            if external_values.len() > MAX_OUTLINE_ARGS {
                skip_args += 1;
                continue;
            }

            outlinable.push(OutlinableCase {
                case_value,
                entry_block: case_entry,
                blocks: case_blocks,
                external_values,
                has_return,
                exit_blocks,
            });
        }

        if debug {
            let total = switch_info.cases.len();
            let outlined = outlinable.len();
            eprintln!("[OUTLINE]   {} of {} cases outlinable (skip: {} block_collect, {} empty, {} small, {} return, {} args)",
                outlined, total, skip_block_collect, skip_empty, skip_small, skip_return, skip_args);
        }

        if outlinable.is_empty() {
            continue;
        }

        // Outline each eligible case.
        for case in &outlinable {
            let outlined_name = format!(
                "{}.__case_{}",
                func.name,
                // Use a deterministic name based on case value
                if case.case_value < 0 {
                    format!("n{}", -case.case_value)
                } else {
                    format!("{}", case.case_value)
                }
            );

            // Build the outlined function.
            let outlined_func = build_outlined_function(
                func,
                case,
                &outlined_name,
                switch_info.end_block,
                &block_map,
                global_max_block_id,
            );

            // Replace the case body in the original function with a call.
            replace_case_with_call(
                func,
                case,
                &outlined_name,
                switch_info.end_block,
                &block_map,
                global_max_block_id,
            );

            new_functions.push(outlined_func);
            total_outlined += 1;
        }
    }

    // Update the function's next_value_id since we may have added instructions.
    if total_outlined > 0 {
        func.next_value_id = 0; // Force recomputation
    }

    total_outlined
}

/// Build a new IrFunction from a case body.
///
/// The outlined function takes pointers to the external allocas as arguments
/// and returns an i32 indicating what to do after:
///   0 = branch to end_block (normal break)
///   1 = return from parent function (the return value is stored via pointer arg)
fn build_outlined_function(
    func: &IrFunction,
    case: &OutlinableCase,
    name: &str,
    end_block: BlockId,
    block_map: &FxHashMap<BlockId, usize>,
    global_max_block_id: &mut u32,
) -> IrFunction {
    // Build a mapping from external Value IDs to parameter indices.
    let value_to_param: FxHashMap<u32, usize> = case.external_values.iter().enumerate()
        .map(|(i, (v, _))| (v.0, i))
        .collect();

    // Allocate new block IDs for the outlined function.
    let block_offset = *global_max_block_id + 1;
    let old_to_new_block: FxHashMap<BlockId, BlockId> = case.blocks.iter().enumerate()
        .map(|(i, &bid)| (bid, BlockId(block_offset + i as u32)))
        .collect();
    *global_max_block_id = block_offset + case.blocks.len() as u32;

    let debug = std::env::var("CCC_DEBUG_OUTLINE").is_ok();
    if debug {
        eprintln!("[OUTLINE]   building {}: old_to_new = {:?}", name, old_to_new_block);
    }

    // Compute value offset: outlined function's values start from 0, but we need
    // to reserve IDs for the ParamRef instructions.
    let num_params = case.external_values.len();
    // ParamRef values: one per parameter, starting at Value(0)
    // The case body's values need to be remapped to avoid collision.
    // Find the max value ID used in the case blocks.
    let mut max_case_value_id: u32 = 0;
    for &bid in &case.blocks {
        if let Some(&idx) = block_map.get(&bid) {
            for inst in &func.blocks[idx].instructions {
                if let Some(dest) = inst.dest() {
                    if dest.0 > max_case_value_id {
                        max_case_value_id = dest.0;
                    }
                }
            }
        }
    }

    // We'll keep the original value IDs and add new ones for params above the max.
    let param_value_base = max_case_value_id + 1;

    // Build entry block with ParamRef instructions.
    let entry_label = BlockId(*global_max_block_id + 1);
    *global_max_block_id += 1;

    let mut entry_instructions = Vec::new();
    for (i, (_val, _ty)) in case.external_values.iter().enumerate() {
        entry_instructions.push(Instruction::ParamRef {
            dest: Value(param_value_base + i as u32),
            param_idx: i,
            ty: IrType::Ptr, // All params are pointers to allocas
        });
    }

    let first_case_block = old_to_new_block[&case.entry_block];
    let entry_block = BasicBlock {
        label: entry_label,
        instructions: entry_instructions,
        terminator: Terminator::Branch(first_case_block),
        source_spans: Vec::new(),
    };

    // Build return blocks for each exit point.
    // Return value: 0 = end_block, 1..N = exit_blocks[0..N-1]
    let has_exits = !case.exit_blocks.is_empty() || case.has_return;

    // Map: exit target → (return value, return block label)
    let mut exit_label_map: FxHashMap<BlockId, BlockId> = FxHashMap::default();

    let ret0_label = BlockId(*global_max_block_id + 1);
    *global_max_block_id += 1;

    let ret0_terminator = if has_exits {
        Terminator::Return(Some(Operand::Const(crate::ir::constants::IrConst::I32(0))))
    } else {
        Terminator::Return(None)
    };
    let ret0_block = BasicBlock {
        label: ret0_label,
        instructions: vec![],
        terminator: ret0_terminator,
        source_spans: Vec::new(),
    };

    // Create return blocks for each exit target.
    let mut exit_ret_blocks: Vec<BasicBlock> = Vec::new();
    for (i, &exit_bid) in case.exit_blocks.iter().enumerate() {
        let ret_label = BlockId(*global_max_block_id + 1);
        *global_max_block_id += 1;
        exit_label_map.insert(exit_bid, ret_label);
        exit_ret_blocks.push(BasicBlock {
            label: ret_label,
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(
                crate::ir::constants::IrConst::I32((i + 1) as i32),
            ))),
            source_spans: Vec::new(),
        });
    }

    // Clone and remap case blocks.
    let mut outlined_blocks = vec![entry_block];
    for &bid in &case.blocks {
        let idx = block_map[&bid];
        let orig_block = &func.blocks[idx];
        let new_label = old_to_new_block[&bid];

        let mut new_instructions = Vec::with_capacity(orig_block.instructions.len());
        for inst in &orig_block.instructions {
            let remapped = remap_instruction(
                inst,
                &value_to_param,
                param_value_base,
                &old_to_new_block,
            );
            new_instructions.push(remapped);
        }

        let new_terminator = remap_terminator(
            &orig_block.terminator,
            &value_to_param,
            param_value_base,
            &old_to_new_block,
            end_block,
            ret0_label,
            &exit_label_map,
        );

        outlined_blocks.push(BasicBlock {
            label: new_label,
            instructions: new_instructions,
            terminator: new_terminator,
            source_spans: orig_block.source_spans.clone(),
        });
    }

    outlined_blocks.push(ret0_block);
    outlined_blocks.extend(exit_ret_blocks);

    // Build parameter list.
    let params: Vec<IrParam> = case.external_values.iter()
        .map(|_| IrParam {
            ty: IrType::Ptr,
            struct_size: None,
            struct_align: None,
            struct_eightbyte_classes: Vec::new(),
            riscv_float_class: None,
        })
        .collect();

    let next_value_id = param_value_base + num_params as u32 + 2;

    let return_type = if has_exits { IrType::I32 } else { IrType::Void };

    IrFunction {
        name: name.to_string(),
        return_type,
        params,
        blocks: outlined_blocks,
        is_variadic: false,
        is_declaration: false,
        is_static: true,
        is_inline: false,
        is_always_inline: false,
        is_noinline: true, // Do NOT inline back — that defeats the purpose
        next_value_id,
        next_label: 0,
        section: func.section.clone(),
        visibility: None,
        is_weak: false,
        is_used: false,
        is_fastcall: false,
        is_naked: false,
        has_inlined_calls: false,
        param_alloca_values: Vec::new(),
        uses_sret: false,
        is_gnu_inline_def: false,
        global_init_label_blocks: Vec::new(),
        ret_eightbyte_classes: Vec::new(),
    }
}

/// Remap an instruction: replace external value references with loads from param pointers.
fn remap_instruction(
    inst: &Instruction,
    value_to_param: &FxHashMap<u32, usize>,
    param_value_base: u32,
    old_to_new_block: &FxHashMap<BlockId, BlockId>,
) -> Instruction {
    // Helper: remap a Value — if it's an external value, replace with the ParamRef value.
    let remap_val = |v: Value| -> Value {
        if let Some(&param_idx) = value_to_param.get(&v.0) {
            Value(param_value_base + param_idx as u32)
        } else {
            v
        }
    };

    let remap_op = |op: &Operand| -> Operand {
        match op {
            Operand::Value(v) => Operand::Value(remap_val(*v)),
            Operand::Const(c) => Operand::Const(*c),
        }
    };

    let remap_bid = |bid: BlockId| -> BlockId {
        old_to_new_block.get(&bid).copied().unwrap_or(bid) // fallback for unmapped (shouldn't happen)
    };

    match inst {
        Instruction::Alloca { dest, ty, size, align, volatile } => {
            Instruction::Alloca { dest: *dest, ty: *ty, size: *size, align: *align, volatile: *volatile }
        }
        Instruction::DynAlloca { dest, size, align } => {
            Instruction::DynAlloca { dest: *dest, size: remap_op(size), align: *align }
        }
        Instruction::Store { val, ptr, ty, seg_override } => {
            Instruction::Store { val: remap_op(val), ptr: remap_val(*ptr), ty: *ty, seg_override: *seg_override }
        }
        Instruction::Load { dest, ptr, ty, seg_override } => {
            Instruction::Load { dest: *dest, ptr: remap_val(*ptr), ty: *ty, seg_override: *seg_override }
        }
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            Instruction::BinOp { dest: *dest, op: *op, lhs: remap_op(lhs), rhs: remap_op(rhs), ty: *ty }
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            Instruction::UnaryOp { dest: *dest, op: *op, src: remap_op(src), ty: *ty }
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            Instruction::Cmp { dest: *dest, op: *op, lhs: remap_op(lhs), rhs: remap_op(rhs), ty: *ty }
        }
        Instruction::Call { func, info } => {
            Instruction::Call {
                func: func.clone(),
                info: remap_call_info(info, &remap_op),
            }
        }
        Instruction::CallIndirect { func_ptr, info } => {
            Instruction::CallIndirect {
                func_ptr: remap_op(func_ptr),
                info: remap_call_info(info, &remap_op),
            }
        }
        Instruction::GetElementPtr { dest, base, offset, ty } => {
            Instruction::GetElementPtr { dest: *dest, base: remap_val(*base), offset: remap_op(offset), ty: *ty }
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            Instruction::Cast { dest: *dest, src: remap_op(src), from_ty: *from_ty, to_ty: *to_ty }
        }
        Instruction::Copy { dest, src } => {
            Instruction::Copy { dest: *dest, src: remap_op(src) }
        }
        Instruction::GlobalAddr { dest, name } => {
            Instruction::GlobalAddr { dest: *dest, name: name.clone() }
        }
        Instruction::Memcpy { dest, src, size } => {
            Instruction::Memcpy { dest: remap_val(*dest), src: remap_val(*src), size: *size }
        }
        Instruction::Select { dest, cond, true_val, false_val, ty } => {
            Instruction::Select {
                dest: *dest,
                cond: remap_op(cond),
                true_val: remap_op(true_val),
                false_val: remap_op(false_val),
                ty: *ty,
            }
        }
        Instruction::Phi { dest, ty, incoming } => {
            Instruction::Phi {
                dest: *dest,
                ty: *ty,
                incoming: incoming.iter()
                    .map(|(op, bid)| (remap_op(op), remap_bid(*bid)))
                    .collect(),
            }
        }
        Instruction::VaArg { dest, va_list_ptr, result_ty } => {
            Instruction::VaArg { dest: *dest, va_list_ptr: remap_val(*va_list_ptr), result_ty: *result_ty }
        }
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, size, eightbyte_classes } => {
            Instruction::VaArgStruct {
                dest_ptr: remap_val(*dest_ptr),
                va_list_ptr: remap_val(*va_list_ptr),
                size: *size,
                eightbyte_classes: eightbyte_classes.clone(),
            }
        }
        Instruction::VaStart { va_list_ptr } => {
            Instruction::VaStart { va_list_ptr: remap_val(*va_list_ptr) }
        }
        Instruction::VaEnd { va_list_ptr } => {
            Instruction::VaEnd { va_list_ptr: remap_val(*va_list_ptr) }
        }
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            Instruction::VaCopy { dest_ptr: remap_val(*dest_ptr), src_ptr: remap_val(*src_ptr) }
        }
        Instruction::AtomicRmw { dest, op, ptr, val, ty, ordering } => {
            Instruction::AtomicRmw {
                dest: *dest, op: *op, ptr: remap_op(ptr), val: remap_op(val), ty: *ty, ordering: *ordering,
            }
        }
        Instruction::AtomicCmpxchg { dest, ptr, expected, desired, ty, success_ordering, failure_ordering, returns_bool } => {
            Instruction::AtomicCmpxchg {
                dest: *dest, ptr: remap_op(ptr), expected: remap_op(expected),
                desired: remap_op(desired), ty: *ty,
                success_ordering: *success_ordering, failure_ordering: *failure_ordering,
                returns_bool: *returns_bool,
            }
        }
        Instruction::AtomicLoad { dest, ptr, ty, ordering } => {
            Instruction::AtomicLoad { dest: *dest, ptr: remap_op(ptr), ty: *ty, ordering: *ordering }
        }
        Instruction::AtomicStore { ptr, val, ty, ordering } => {
            Instruction::AtomicStore { ptr: remap_op(ptr), val: remap_op(val), ty: *ty, ordering: *ordering }
        }
        Instruction::Fence { ordering } => {
            Instruction::Fence { ordering: *ordering }
        }
        Instruction::LabelAddr { dest, label } => {
            Instruction::LabelAddr { dest: *dest, label: remap_bid(*label) }
        }
        Instruction::InlineAsm { template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols, seg_overrides } => {
            Instruction::InlineAsm {
                template: template.clone(),
                outputs: outputs.iter().map(|(c, v, n)| (c.clone(), remap_val(*v), n.clone())).collect(),
                inputs: inputs.iter().map(|(c, op, n)| (c.clone(), remap_op(op), n.clone())).collect(),
                clobbers: clobbers.clone(),
                operand_types: operand_types.clone(),
                goto_labels: goto_labels.iter().map(|(n, b)| (n.clone(), remap_bid(*b))).collect(),
                input_symbols: input_symbols.clone(),
                seg_overrides: seg_overrides.clone(),
            }
        }
        Instruction::Intrinsic { dest, op, dest_ptr, args } => {
            Instruction::Intrinsic {
                dest: *dest, op: op.clone(),
                dest_ptr: dest_ptr.map(|p| remap_val(p)),
                args: args.iter().map(|a| remap_op(a)).collect(),
            }
        }
        Instruction::StackSave { dest } => {
            Instruction::StackSave { dest: *dest }
        }
        Instruction::StackRestore { ptr } => {
            Instruction::StackRestore { ptr: remap_val(*ptr) }
        }
        Instruction::ParamRef { dest, param_idx, ty } => {
            Instruction::ParamRef { dest: *dest, param_idx: *param_idx, ty: *ty }
        }
        Instruction::GetReturnF64Second { dest } => {
            Instruction::GetReturnF64Second { dest: *dest }
        }
        Instruction::SetReturnF64Second { src } => {
            Instruction::SetReturnF64Second { src: remap_op(src) }
        }
        Instruction::GetReturnF32Second { dest } => {
            Instruction::GetReturnF32Second { dest: *dest }
        }
        Instruction::SetReturnF32Second { src } => {
            Instruction::SetReturnF32Second { src: remap_op(src) }
        }
        Instruction::GetReturnF128Second { dest } => {
            Instruction::GetReturnF128Second { dest: *dest }
        }
        Instruction::SetReturnF128Second { src } => {
            Instruction::SetReturnF128Second { src: remap_op(src) }
        }
    }
}

/// Remap CallInfo operands.
fn remap_call_info(info: &CallInfo, remap_op: &dyn Fn(&Operand) -> Operand) -> CallInfo {
    CallInfo {
        dest: info.dest,
        args: info.args.iter().map(remap_op).collect(),
        arg_types: info.arg_types.clone(),
        return_type: info.return_type,
        is_variadic: info.is_variadic,
        num_fixed_args: info.num_fixed_args,
        struct_arg_sizes: info.struct_arg_sizes.clone(),
        struct_arg_aligns: info.struct_arg_aligns.clone(),
        struct_arg_classes: info.struct_arg_classes.clone(),
        struct_arg_riscv_float_classes: info.struct_arg_riscv_float_classes.clone(),
        is_sret: info.is_sret,
        is_fastcall: info.is_fastcall,
        ret_eightbyte_classes: info.ret_eightbyte_classes.clone(),
    }
}

/// Remap a terminator for the outlined function.
/// Branches to end_block become branches to ret0_label (return 0).
/// Returns become returns of 1 (signal to parent that the function returned).
fn remap_terminator(
    term: &Terminator,
    value_to_param: &FxHashMap<u32, usize>,
    param_value_base: u32,
    old_to_new_block: &FxHashMap<BlockId, BlockId>,
    end_block: BlockId,
    ret0_label: BlockId,
    exit_label_map: &FxHashMap<BlockId, BlockId>,
) -> Terminator {
    let remap_val = |v: Value| -> Value {
        if let Some(&param_idx) = value_to_param.get(&v.0) {
            Value(param_value_base + param_idx as u32)
        } else {
            v
        }
    };

    let remap_op = |op: &Operand| -> Operand {
        match op {
            Operand::Value(v) => Operand::Value(remap_val(*v)),
            Operand::Const(c) => Operand::Const(*c),
        }
    };

    let remap_bid = |bid: BlockId| -> BlockId {
        if bid == end_block {
            ret0_label // Branch to end → return 0
        } else if let Some(&new_bid) = old_to_new_block.get(&bid) {
            new_bid
        } else if let Some(&exit_ret_label) = exit_label_map.get(&bid) {
            exit_ret_label // Branch to exit block → return exit index
        } else {
            ret0_label // Fallback
        }
    };

    match term {
        Terminator::Branch(target) => {
            Terminator::Branch(remap_bid(*target))
        }
        Terminator::CondBranch { cond, true_label, false_label } => {
            Terminator::CondBranch {
                cond: remap_op(cond),
                true_label: remap_bid(*true_label),
                false_label: remap_bid(*false_label),
            }
        }
        Terminator::Return(_op) => {
            // Parent function's return — outlined function returns 1 to signal this.
            // The return value itself should have been stored to an alloca that the
            // parent can read back. For now, just return 1.
            Terminator::Return(Some(Operand::Const(
                crate::ir::constants::IrConst::I32(1),
            )))
        }
        Terminator::Switch { val, cases, default, ty } => {
            Terminator::Switch {
                val: remap_op(val),
                cases: cases.iter().map(|(v, bid)| (*v, remap_bid(*bid))).collect(),
                default: remap_bid(*default),
                ty: *ty,
            }
        }
        Terminator::IndirectBranch { target, possible_targets } => {
            Terminator::IndirectBranch {
                target: remap_op(target),
                possible_targets: possible_targets.iter().map(|bid| remap_bid(*bid)).collect(),
            }
        }
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

/// Replace a case body in the original function with a call to the outlined function.
/// The case entry block becomes: call outlined_func(args...); branch to end.
fn replace_case_with_call(
    func: &mut IrFunction,
    case: &OutlinableCase,
    outlined_name: &str,
    end_block: BlockId,
    block_map: &FxHashMap<BlockId, usize>,
    global_max_block_id: &mut u32,
) {
    // Find the entry block index.
    let entry_idx = match block_map.get(&case.entry_block) {
        Some(&idx) => idx,
        None => return,
    };

    // Build the call arguments: pass the external values directly.
    let args: Vec<Operand> = case.external_values.iter()
        .map(|(v, _)| Operand::Value(*v))
        .collect();
    let arg_types: Vec<IrType> = case.external_values.iter()
        .map(|(_, ty)| *ty)
        .collect();
    let num_args = args.len();

    let has_exits = !case.exit_blocks.is_empty() || case.has_return;

    if has_exits {
        // The outlined function returns I32: 0 = end_block, 1..N = exit_blocks.
        // We dispatch on the return value using a Switch terminator.
        let mut max_val: u32 = 0;
        for block in func.blocks.iter() {
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    if dest.0 > max_val {
                        max_val = dest.0;
                    }
                }
            }
        }
        let call_dest = Value(max_val + 1);

        let call_inst = Instruction::Call {
            func: outlined_name.to_string(),
            info: CallInfo {
                dest: Some(call_dest),
                args,
                arg_types,
                return_type: IrType::I32,
                is_variadic: false,
                num_fixed_args: num_args,
                struct_arg_sizes: vec![None; num_args],
                struct_arg_aligns: vec![None; num_args],
                struct_arg_classes: vec![Vec::new(); num_args],
                struct_arg_riscv_float_classes: vec![None; num_args],
                is_sret: false,
                is_fastcall: false,
                ret_eightbyte_classes: Vec::new(),
            },
        };

        // Build Switch cases: 0→end_block, 1..N→exit_blocks
        let mut switch_cases: Vec<(i64, BlockId)> = Vec::new();
        for (i, &exit_bid) in case.exit_blocks.iter().enumerate() {
            switch_cases.push(((i + 1) as i64, exit_bid));
        }

        func.blocks[entry_idx].instructions = vec![call_inst];
        if switch_cases.is_empty() {
            // Only has_return, no exit_blocks — just branch to end
            func.blocks[entry_idx].terminator = Terminator::Branch(end_block);
        } else {
            func.blocks[entry_idx].terminator = Terminator::Switch {
                val: Operand::Value(call_dest),
                cases: switch_cases,
                default: end_block, // 0 and any unexpected value → end_block
                ty: IrType::I32,
            };
        }
    } else {
        // Simple case: void call and branch to end.
        let call_inst = Instruction::Call {
            func: outlined_name.to_string(),
            info: CallInfo {
                dest: None,
                args,
                arg_types,
                return_type: IrType::Void,
                is_variadic: false,
                num_fixed_args: num_args,
                struct_arg_sizes: vec![None; num_args],
                struct_arg_aligns: vec![None; num_args],
                struct_arg_classes: vec![Vec::new(); num_args],
                struct_arg_riscv_float_classes: vec![None; num_args],
                is_sret: false,
                is_fastcall: false,
                ret_eightbyte_classes: Vec::new(),
            },
        };

        func.blocks[entry_idx].instructions = vec![call_inst];
        func.blocks[entry_idx].terminator = Terminator::Branch(end_block);
    }

    // Remove the other case blocks (mark them as empty with branch to end).
    // We can't remove them from the vec (would invalidate indices), so make them dead.
    for &bid in &case.blocks {
        if bid == case.entry_block {
            continue; // Already replaced
        }
        if let Some(&idx) = block_map.get(&bid) {
            func.blocks[idx].instructions.clear();
            func.blocks[idx].terminator = Terminator::Branch(end_block);
        }
    }
}
