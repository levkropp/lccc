//! Un-IVSR pass: Reverse pointer IV strength reduction when indexed addressing is beneficial.
//!
//! This pass runs after IVSR and detects pointer induction variables created by IVSR.
//! When the target architecture supports indexed addressing (like x86-64 SIB), it's often
//! more efficient to use `mov (%base,%index,scale)` than pointer arithmetic `mov (%ptr); add $stride, %ptr`.
//!
//! The pass transforms:
//!   %ptr = Phi(%initial_ptr, %ptr_next)
//!   %val = Load(%ptr)
//!   %ptr_next = GEP(%ptr, stride)
//!
//! Back to:
//!   %index = Phi(%init_index, %index_next)
//!   %offset = Mul/Shl(%index, stride)
//!   %addr = GEP(%base, %offset)
//!   %val = Load(%addr)
//!   %index_next = Add(%index, 1)
//!
//! This allows the backend's indexed addressing detection (Phase 9) to emit
//! efficient SIB-form instructions.

use crate::common::fx_hash::FxHashSet;
use crate::common::types::IrType;
use crate::ir::reexports::{
    BlockId,
    Instruction,
    IrBinOp,
    IrConst,
    IrFunction,
    Operand,
    Value,
};

/// Information about an IVSR-created pointer IV that should be un-transformed
struct IvsrPointerIV {
    /// The pointer phi's destination value
    ptr_phi_dest: Value,
    /// The original array base pointer
    base_ptr: Value,
    /// Element stride in bytes
    stride: i64,
    /// Initial offset (usually 0)
    init_offset: i64,
    /// The associated index IV (if found)
    index_iv: Option<Value>,
    /// Block where the phi resides
    header_block: BlockId,
}

/// Run the un-IVSR pass on a function.
/// Returns the number of pointer IVs that were reverted.
pub(crate) fn run_univsr(func: &mut IrFunction) -> usize {
    // Only run on x86_64 where indexed addressing is available
    // TODO: Make this target-aware when other architectures are added

    let ivsr_pointers = detect_ivsr_pointer_ivs(func);

    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();
    if debug {
        eprintln!("[Un-IVSR] Function {}: found {} pointer IVs", func.name, ivsr_pointers.len());
    }

    if ivsr_pointers.is_empty() {
        return 0;
    }

    let mut num_reverted = 0;
    for ptr_iv in &ivsr_pointers {
        if debug {
            eprintln!("[Un-IVSR]   Pointer IV: Value({}) stride={} index={:?}",
                      ptr_iv.ptr_phi_dest.0, ptr_iv.stride,
                      ptr_iv.index_iv.as_ref().map(|v| v.0));
        }

        // Only revert if stride is a valid SIB scale (1, 2, 4, or 8)
        if !is_valid_sib_scale(ptr_iv.stride) {
            if debug {
                eprintln!("[Un-IVSR]     Skipped: stride {} not valid SIB scale", ptr_iv.stride);
            }
            continue;
        }

        // Only revert if we found the index IV
        if ptr_iv.index_iv.is_none() {
            if debug {
                eprintln!("[Un-IVSR]     Skipped: no index IV found");
            }
            continue;
        }

        if revert_pointer_iv(func, ptr_iv) {
            if debug {
                eprintln!("[Un-IVSR]     ✓ Reverted successfully");
            }
            num_reverted += 1;
        } else {
            if debug {
                eprintln!("[Un-IVSR]     Failed to revert");
            }
        }
    }

    num_reverted
}

/// Detect IVSR-created pointer IVs in the function
fn detect_ivsr_pointer_ivs(func: &IrFunction) -> Vec<IvsrPointerIV> {
    let mut result = Vec::new();

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Phi { dest, ty, incoming } = inst {
                // Only interested in pointer phis with exactly 2 incoming edges
                if *ty != IrType::Ptr || incoming.len() != 2 {
                    continue;
                }

                // Check if this matches IVSR pointer pattern
                if let Some(ptr_iv) = analyze_pointer_phi(func, dest, incoming, block.label) {
                    result.push(ptr_iv);
                }
            }
        }
    }

    result
}

/// Analyze a pointer phi to see if it's an IVSR pattern
fn analyze_pointer_phi(
    func: &IrFunction,
    dest: &Value,
    incoming: &[(Operand, BlockId)],
    header_block: BlockId,
) -> Option<IvsrPointerIV> {
    // Extract the two incoming edges
    let (init_op, _init_block) = &incoming[0];
    let (backedge_op, _backedge_block) = &incoming[1];

    // Backedge value should be defined by a GEP
    let backedge_val = match backedge_op {
        Operand::Value(v) => v,
        _ => return None,
    };

    // Find the GEP instruction that defines the backedge value
    let (gep_base, stride) = find_gep_with_const_offset(func, backedge_val.0)?;

    // Verify GEP base is the phi itself (recursive pattern: %ptr_next = GEP(%ptr, stride))
    if gep_base.0 != dest.0 {
        return None;
    }

    // Extract base pointer from init value
    let (base_ptr, init_offset) = extract_base_from_init(func, init_op)?;

    // Try to find the associated index IV
    let index_iv = find_index_iv_in_header(func, header_block);

    Some(IvsrPointerIV {
        ptr_phi_dest: *dest,
        base_ptr,
        stride,
        init_offset,
        index_iv,
        header_block,
    })
}

/// Find a GEP instruction that defines a value with a constant offset
fn find_gep_with_const_offset(func: &IrFunction, val_id: u32) -> Option<(Value, i64)> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, base, offset, .. } = inst {
                if dest.0 == val_id {
                    // Extract constant offset
                    let stride = match offset {
                        Operand::Const(c) => c.to_i64()?,
                        _ => return None,
                    };
                    return Some((*base, stride));
                }
            }
        }
    }
    None
}

/// Extract the base pointer from the init value of a pointer phi
fn extract_base_from_init(func: &IrFunction, init_op: &Operand) -> Option<(Value, i64)> {
    match init_op {
        Operand::Value(v) => {
            // Check if init value is a GEP (handles cases like %init = GEP(%arr, 0))
            for block in &func.blocks {
                for inst in &block.instructions {
                    if let Instruction::GetElementPtr { dest, base, offset, .. } = inst {
                        if dest.0 == v.0 {
                            let init_offset = match offset {
                                Operand::Const(c) => c.to_i64().unwrap_or(0),
                                _ => 0,
                            };
                            return Some((*base, init_offset));
                        }
                    }
                }
            }
            // Not a GEP - just a direct pointer value
            Some((*v, 0))
        }
        _ => None,
    }
}

/// Find an integer IV phi in the same header block
fn find_index_iv_in_header(func: &IrFunction, header: BlockId) -> Option<Value> {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

    // Find the block with the matching label
    let header_block = func.blocks.iter().find(|b| b.label.0 == header.0)?;

    if debug {
        eprintln!("[Un-IVSR]     Looking for index IV in block {}", header.0);
    }

    for inst in &header_block.instructions {
        if let Instruction::Phi { dest, ty, incoming } = inst {
            if debug {
                eprintln!("[Un-IVSR]       Found phi: Value({}) ty={:?} incoming={}",
                          dest.0, ty, incoming.len());
            }

            // Look for integer phis (I32, I64, U32, U64)
            if !matches!(ty, IrType::I32 | IrType::I64 | IrType::U32 | IrType::U64) {
                continue;
            }

            // Check if it's incremented in the backedge (typical loop counter)
            // We look for any Add pattern, not just +1, because optimizations
            // might have changed the step
            if is_basic_iv(func, dest, incoming) {
                if debug {
                    eprintln!("[Un-IVSR]       ✓ Found basic IV: Value({})", dest.0);
                }
                return Some(*dest);
            } else if debug {
                eprintln!("[Un-IVSR]       Not a basic IV");
            }
        }
    }

    if debug {
        eprintln!("[Un-IVSR]     No index IV found in header");
    }

    None
}

/// Check if a phi is a basic induction variable (incremented by a constant)
fn is_basic_iv(func: &IrFunction, dest: &Value, incoming: &[(Operand, BlockId)]) -> bool {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

    if incoming.len() != 2 {
        return false;
    }

    // Get the backedge value (typically the second incoming edge)
    let backedge_val = match &incoming[1].0 {
        Operand::Value(v) => v,
        _ => {
            if debug {
                eprintln!("[Un-IVSR]         Backedge is not a Value");
            }
            return false;
        }
    };

    if debug {
        eprintln!("[Un-IVSR]         Backedge value: Value({})", backedge_val.0);
    }

    // Check if backedge value is defined by Add(%iv, const) or other BinOp
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            // Debug: print all instructions that define the backedge value
            if debug {
                if let Some(d) = inst.dest_value_id() {
                    if d == backedge_val.0 {
                        eprintln!("[Un-IVSR]         Found inst defining Value({}) in block {}: {:?}",
                                  backedge_val.0, block_idx, inst);
                    }
                }
            }

            match inst {
                Instruction::BinOp { dest: bd, op, lhs, rhs, .. } if bd.0 == backedge_val.0 => {
                    if debug {
                        eprintln!("[Un-IVSR]         Found BinOp {:?} defining backedge", op);
                    }

                    // Check if it's an Add with the phi as one operand
                    if matches!(op, IrBinOp::Add) {
                        match (lhs, rhs) {
                            (Operand::Value(v), Operand::Const(c)) | (Operand::Const(c), Operand::Value(v)) => {
                                if debug {
                                    eprintln!("[Un-IVSR]           Add operands: Value({}) + Const({:?})",
                                              v.0, c.to_i64());
                                }
                                // Accept any constant increment
                                if v.0 == dest.0 && c.to_i64().is_some() {
                                    return true;
                                }
                            }
                            _ => {
                                if debug {
                                    eprintln!("[Un-IVSR]           Add operands don't match pattern");
                                }
                            }
                        }
                    }
                }
                // Also check for Copy/Cast instructions (sometimes optimizations create these)
                Instruction::Copy { dest: cd, src } if cd.0 == backedge_val.0 => {
                    if debug {
                        eprintln!("[Un-IVSR]         Found Copy defining backedge: {:?}", src);
                    }
                    // Follow the copy chain
                    if let Operand::Value(v) = src {
                        // Check if the source is the increment
                        return is_value_from_iv_increment(func, *v, *dest);
                    }
                }
                Instruction::Cast { dest: cd, src, .. } if cd.0 == backedge_val.0 => {
                    if debug {
                        eprintln!("[Un-IVSR]         Found Cast defining backedge: {:?}", src);
                    }
                    // Follow the cast to find the actual increment
                    if let Operand::Value(v) = src {
                        return is_value_from_iv_increment(func, *v, *dest);
                    }
                }
                _ => {}
            }
        }
    }

    if debug {
        eprintln!("[Un-IVSR]         No increment pattern found");
    }

    false
}

/// Check if a value is from an IV increment (Add with constant)
fn is_value_from_iv_increment(func: &IrFunction, val: Value, phi_dest: Value) -> bool {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::BinOp { dest: bd, op: IrBinOp::Add, lhs, rhs, .. } = inst {
                if bd.0 == val.0 {
                    if debug {
                        eprintln!("[Un-IVSR]           Found Add defining Value({})", val.0);
                    }
                    match (lhs, rhs) {
                        (Operand::Value(v), Operand::Const(c)) | (Operand::Const(c), Operand::Value(v)) => {
                            if debug {
                                eprintln!("[Un-IVSR]             Add operands: Value({}) + Const({:?})",
                                          v.0, c.to_i64());
                            }
                            // Check if it's adding to the phi
                            if v.0 == phi_dest.0 && c.to_i64().is_some() {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    false
}

/// Validate that the pointer phi uses are safe to transform
fn validate_transformation_safety(
    func: &IrFunction,
    ptr_iv: &IvsrPointerIV,
    _uses: &[(usize, usize, UseKind)],
) -> bool {
    // Rule 1: No escaping uses (stored to memory, passed to calls, etc.)
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                // Unsafe: pointer stored to memory
                Instruction::Store { val: Operand::Value(v), .. }
                    if v.0 == ptr_iv.ptr_phi_dest.0 => {
                    return false;
                }

                // Unsafe: pointer passed to function call
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    for arg in &info.args {
                        if let Operand::Value(v) = arg {
                            if v.0 == ptr_iv.ptr_phi_dest.0 {
                                return false;
                            }
                        }
                    }
                }

                // Unsafe: pointer used in comparison (rare but possible)
                Instruction::Cmp { lhs, rhs, .. } => {
                    let check = |op: &Operand| {
                        if let Operand::Value(v) = op {
                            v.0 == ptr_iv.ptr_phi_dest.0
                        } else {
                            false
                        }
                    };
                    if check(lhs) || check(rhs) {
                        return false;
                    }
                }

                _ => {}
            }
        }
    }

    // Rule 2: All uses must be Load/Store (already guaranteed by find_transitive_ptr_uses)

    // Rule 3: Pointer phi must be in a loop header
    // (This is implicitly checked by IVSR pattern detection)

    true
}

/// Information about a transformation to apply
struct TransformInfo {
    block_idx: usize,
    inst_idx: usize,
    offset_inst: Instruction,
    add_inst: Option<Instruction>,
    gep_inst: Instruction,
    new_ptr: Value,
}

/// Revert a pointer IV back to indexed form
fn revert_pointer_iv(func: &mut IrFunction, ptr_iv: &IvsrPointerIV) -> bool {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

    let index_iv = match ptr_iv.index_iv {
        Some(iv) => iv,
        None => return false,
    };

    // Find all transitive uses (through GEPs)
    let ptr_uses = find_transitive_ptr_uses(func, ptr_iv.ptr_phi_dest);

    if debug {
        eprintln!("[Un-IVSR]     Found {} transitive uses", ptr_uses.len());
    }

    if ptr_uses.is_empty() {
        return false;
    }

    // Validate transformation safety
    if !validate_transformation_safety(func, ptr_iv, &ptr_uses) {
        if debug {
            eprintln!("[Un-IVSR]     Skipped: unsafe uses detected");
        }
        return false;
    }

    // Allocate fresh Value IDs for transformed instructions
    let mut next_val_id = func.next_value_id;
    if next_val_id == 0 {
        next_val_id = compute_max_value_id(func) + 1;
    }

    // Transform each use - collect all transformations first
    let mut transforms = Vec::new();
    for (block_idx, inst_idx, _use_kind) in &ptr_uses {
        // Create offset calculation: %offset = index * stride
        let offset_val = Value(next_val_id);
        next_val_id += 1;

        let offset_inst = create_offset_instruction(
            offset_val,
            index_iv,
            ptr_iv.stride,
        );

        // Create GEP: %addr = GEP(base, offset + init_offset)
        let (gep_inst, add_inst_opt) = if ptr_iv.init_offset == 0 {
            (Instruction::GetElementPtr {
                dest: Value(next_val_id),
                base: ptr_iv.base_ptr,
                offset: Operand::Value(offset_val),
                ty: IrType::Ptr,
            }, None)
        } else {
            // Need to add init_offset: %adjusted = offset + init_offset
            let adjusted_offset_val = Value(next_val_id);
            next_val_id += 1;

            let add_inst = Instruction::BinOp {
                dest: adjusted_offset_val,
                op: IrBinOp::Add,
                lhs: Operand::Value(offset_val),
                rhs: Operand::Const(IrConst::I64(ptr_iv.init_offset)),
                ty: IrType::I64,
            };

            let gep = Instruction::GetElementPtr {
                dest: Value(next_val_id),
                base: ptr_iv.base_ptr,
                offset: Operand::Value(adjusted_offset_val),
                ty: IrType::Ptr,
            };

            (gep, Some(add_inst))
        };

        let gep_val = if ptr_iv.init_offset == 0 {
            Value(next_val_id)
        } else {
            Value(next_val_id)
        };
        next_val_id += 1;

        transforms.push(TransformInfo {
            block_idx: *block_idx,
            inst_idx: *inst_idx,
            offset_inst,
            add_inst: add_inst_opt,
            gep_inst,
            new_ptr: gep_val,
        });
    }

    // Apply transforms in reverse order (to preserve instruction indices)
    // Sort by (block_idx, inst_idx) descending
    transforms.sort_by(|a, b| {
        b.block_idx.cmp(&a.block_idx)
            .then(b.inst_idx.cmp(&a.inst_idx))
    });

    for transform in transforms {
        let block = &mut func.blocks[transform.block_idx];

        // Insert new instructions before the Load/Store
        let mut insert_idx = transform.inst_idx;

        // Insert offset instruction
        block.instructions.insert(insert_idx, transform.offset_inst);
        insert_idx += 1;

        // Insert add instruction if needed
        if let Some(add_inst) = transform.add_inst {
            block.instructions.insert(insert_idx, add_inst);
            insert_idx += 1;
        }

        // Insert GEP instruction
        block.instructions.insert(insert_idx, transform.gep_inst);
        insert_idx += 1;

        // Update the Load/Store to use new GEP result
        if debug {
            eprintln!("[Un-IVSR]       Updating instruction at block {} inst {} (was inst {})",
                      transform.block_idx, insert_idx, transform.inst_idx);
            eprintln!("[Un-IVSR]         Instruction: {:?}", block.instructions[insert_idx]);
        }

        let use_inst = &mut block.instructions[insert_idx];
        match use_inst {
            Instruction::Load { ptr, .. } => {
                if debug {
                    eprintln!("[Un-IVSR]         Replacing ptr Value({}) with Value({})",
                              ptr.0, transform.new_ptr.0);
                }
                *ptr = transform.new_ptr;
            }
            Instruction::Store { ptr, .. } => {
                if debug {
                    eprintln!("[Un-IVSR]         Replacing ptr Value({}) with Value({})",
                              ptr.0, transform.new_ptr.0);
                }
                *ptr = transform.new_ptr;
            }
            _ => {
                if debug {
                    eprintln!("[Un-IVSR]     WARNING: Expected Load or Store at block {} inst {}, found {:?}",
                              transform.block_idx, insert_idx, use_inst);
                }
            }
        }
    }

    // Update next_value_id cache
    func.next_value_id = next_val_id;

    // Remove the dead pointer IV cycle: the pointer phi and its increment GEP.
    // After reverting all uses to indexed addressing, the pointer phi and its
    // backedge increment form a self-referencing cycle that DCE can't remove.
    // If the pointer IV register is reused by the indexed addressing base,
    // the increment corrupts the base address. Break the cycle by removing
    // the increment GEP (replacing with a harmless Copy of the phi).
    remove_dead_pointer_iv_cycle(func, ptr_iv);

    true
}

/// Remove the dead pointer IV cycle after reverting to indexed addressing.
///
/// After Un-IVSR transforms all Load/Store uses to use indexed GEPs, the pointer
/// phi and its increment GEP form a dead cycle:
///   %ptr = Phi(%init, %ptr_next)
///   %ptr_next = GEP(%ptr, stride)
///
/// Standard DCE can't remove cycles (both values appear "used" by each other).
/// If the register allocator assigns the pointer phi to the same register as the
/// indexed GEP's base pointer, the increment corrupts the base address.
///
/// Fix: find the increment GEP (%ptr_next = GEP(%ptr, stride)) and replace it
/// with a harmless Copy of the phi itself. This breaks the data dependency while
/// preserving SSA validity. DCE can then remove both the Copy and the phi.
fn remove_dead_pointer_iv_cycle(func: &mut IrFunction, ptr_iv: &IvsrPointerIV) {
    let phi_id = ptr_iv.ptr_phi_dest.0;
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

    // Strategy: replace the pointer phi itself with a Copy of the base pointer.
    // This breaks the phi→increment→phi cycle completely. The increment GEP
    // then reads a constant base (not the incrementing pointer), making it dead
    // code that DCE can remove.

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            // Replace the phi with a Copy of the base pointer
            if let Instruction::Phi { dest, .. } = inst {
                if dest.0 == phi_id {
                    if debug {
                        eprintln!("[Un-IVSR]     Removing phi Value({}) → Copy of base Value({})",
                                  phi_id, ptr_iv.base_ptr.0);
                    }
                    *inst = Instruction::Copy {
                        dest: ptr_iv.ptr_phi_dest,
                        src: Operand::Value(ptr_iv.base_ptr),
                    };
                    // Don't return — also need to handle the increment GEP
                }
            }
            // Replace the increment GEP with a Copy (makes it dead code)
            if let Instruction::GetElementPtr { dest, base, offset: Operand::Const(_), .. } = inst {
                if base.0 == phi_id {
                    if debug {
                        eprintln!("[Un-IVSR]     Removing GEP({}) base=Value({})",
                                  dest.0, base.0);
                    }
                    let gep_dest = *dest;
                    *inst = Instruction::Copy {
                        dest: gep_dest,
                        src: Operand::Value(ptr_iv.base_ptr),
                    };
                }
            }
        }
    }
}

/// Create an offset calculation instruction (multiply or shift)
fn create_offset_instruction(dest: Value, index: Value, stride: i64) -> Instruction {
    // Use shift for power-of-2 strides (more efficient)
    if stride > 0 && (stride & (stride - 1)) == 0 && stride <= 8 {
        // stride is a power of 2
        let shift_amount = (stride as u64).trailing_zeros() as i64;
        Instruction::BinOp {
            dest,
            op: IrBinOp::Shl,
            lhs: Operand::Value(index),
            rhs: Operand::Const(IrConst::I32(shift_amount as i32)),
            ty: IrType::I64,
        }
    } else {
        // Use multiply for non-power-of-2 strides
        Instruction::BinOp {
            dest,
            op: IrBinOp::Mul,
            lhs: Operand::Value(index),
            rhs: Operand::Const(IrConst::I64(stride)),
            ty: IrType::I64,
        }
    }
}

/// Kind of use found (for tracking purposes)
#[derive(Debug, Clone, Copy)]
enum UseKind {
    Load,
    Store,
}

/// Direct use of a value (either actual use or intermediate GEP)
#[derive(Debug)]
enum DirectUse {
    Load,
    Store,
    Gep(Value),  // GEP destination value
}

/// Find all Load/Store instructions that transitively use the pointer phi
/// This handles cases where the pointer is used through intermediate GEPs:
///   %ptr = Phi(...)
///   %gep1 = GEP(%ptr, offset)
///   %val = Load(%gep1)  <- transitive use through %gep1
fn find_transitive_ptr_uses(
    func: &IrFunction,
    ptr_val: Value,
) -> Vec<(usize, usize, UseKind)> {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();
    let mut results = Vec::new();

    // Phase 1: Find direct uses of ptr_val (Load/Store/GEP)
    let mut direct_uses: Vec<(usize, usize, DirectUse)> = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            match inst {
                Instruction::Load { ptr, .. } if ptr.0 == ptr_val.0 => {
                    direct_uses.push((block_idx, inst_idx, DirectUse::Load));
                }
                Instruction::Store { ptr, .. } if ptr.0 == ptr_val.0 => {
                    direct_uses.push((block_idx, inst_idx, DirectUse::Store));
                }
                Instruction::GetElementPtr { dest, base, offset, .. } if base.0 == ptr_val.0 => {
                    // Skip the pointer increment GEP (backedge of the phi)
                    // We only care about GEPs used for memory access
                    let is_zero_offset = matches!(offset, Operand::Const(c) if c.to_i64() == Some(0));
                    if debug && is_zero_offset {
                        eprintln!("[Un-IVSR]       Found zero-offset GEP -> Value({})", dest.0);
                    }
                    direct_uses.push((block_idx, inst_idx, DirectUse::Gep(*dest)));
                }
                _ => {}
            }
        }
    }

    if debug {
        eprintln!("[Un-IVSR]     Phase 1: found {} direct uses of Value({})",
                  direct_uses.len(), ptr_val.0);
        for (block_idx, inst_idx, use_kind) in &direct_uses {
            eprintln!("[Un-IVSR]       Block {} inst {}: {:?}",
                      block_idx, inst_idx, use_kind);
        }
    }

    // Phase 2: Build a list of values to explore (GEPs and their results, plus Copy destinations)
    // We need to follow both GEP chains and Copy chains
    let mut values_to_explore: Vec<Value> = direct_uses.iter()
        .filter_map(|(_, _, use_kind)| {
            if let DirectUse::Gep(gep_dest) = use_kind {
                Some(*gep_dest)
            } else {
                None
            }
        })
        .collect();

    // Also follow Copy instructions that copy the pointer phi
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Copy { dest, src: Operand::Value(v) } = inst {
                if v.0 == ptr_val.0 {
                    if debug {
                        eprintln!("[Un-IVSR]     Found Copy: Value({}) -> Value({})", ptr_val.0, dest.0);
                    }
                    values_to_explore.push(*dest);
                }
            }
        }
    }

    // Track which values we've already explored (avoid cycles)
    let mut explored_values = FxHashSet::default();

    if debug && !values_to_explore.is_empty() {
        eprintln!("[Un-IVSR]     Phase 2: exploring {} values (GEPs and Copies)", values_to_explore.len());
    }

    while let Some(val_to_check) = values_to_explore.pop() {
        if !explored_values.insert(val_to_check.0) {
            continue;  // Already explored
        }

        if debug {
            eprintln!("[Un-IVSR]       Exploring uses of Value({})", val_to_check.0);
        }

        // Find uses of this value
        for (block_idx, block) in func.blocks.iter().enumerate() {
            for (inst_idx, inst) in block.instructions.iter().enumerate() {

                match inst {
                    // Skip Phi uses - those are loop-carried dependencies, not memory accesses
                    Instruction::Phi { incoming, .. } => {
                        for (op, _) in incoming {
                            if let Operand::Value(v) = op {
                                if v.0 == val_to_check.0 && debug {
                                    eprintln!("[Un-IVSR]         Skipping Phi use (loop backedge)");
                                }
                            }
                        }
                    }
                    // Follow Copy chains
                    Instruction::Copy { dest, src: Operand::Value(v) } if v.0 == val_to_check.0 => {
                        if debug {
                            eprintln!("[Un-IVSR]         Found Copy chain: Value({}) -> Value({})",
                                      val_to_check.0, dest.0);
                        }
                        values_to_explore.push(*dest);
                    }
                    Instruction::Load { ptr, .. } if ptr.0 == val_to_check.0 => {
                        if debug {
                            eprintln!("[Un-IVSR]         ✓ Found Load at block {} inst {}", block_idx, inst_idx);
                        }
                        results.push((block_idx, inst_idx, UseKind::Load));
                    }
                    Instruction::Store { ptr, .. } if ptr.0 == val_to_check.0 => {
                        if debug {
                            eprintln!("[Un-IVSR]         ✓ Found Store at block {} inst {}", block_idx, inst_idx);
                        }
                        results.push((block_idx, inst_idx, UseKind::Store));
                    }
                    Instruction::GetElementPtr { dest, base, .. } if base.0 == val_to_check.0 => {
                        // Another level of GEP - continue exploring
                        if debug {
                            eprintln!("[Un-IVSR]         Found nested GEP -> Value({})", dest.0);
                        }
                        values_to_explore.push(*dest);
                    }
                    _ => {}
                }
            }
        }
    }

    // Phase 3: Add direct Load/Store uses (non-GEP path)
    for (block_idx, inst_idx, use_kind) in direct_uses {
        match use_kind {
            DirectUse::Load => results.push((block_idx, inst_idx, UseKind::Load)),
            DirectUse::Store => results.push((block_idx, inst_idx, UseKind::Store)),
            DirectUse::Gep(_) => {}, // Already handled in Phase 2
        }
    }

    if debug {
        eprintln!("[Un-IVSR]     Found {} transitive uses total", results.len());
    }

    results
}

/// Replace uses of old pointer value with new value in an instruction
fn replace_ptr_use_in_instruction(inst: &mut Instruction, old_ptr: Value, new_ptr: Value) {
    match inst {
        Instruction::Load { ptr, .. } if ptr.0 == old_ptr.0 => {
            *ptr = new_ptr;
        }
        Instruction::Store { ptr, .. } if ptr.0 == old_ptr.0 => {
            *ptr = new_ptr;
        }
        _ => {}
    }
}

/// Compute the maximum value ID in the function
fn compute_max_value_id(func: &IrFunction) -> u32 {
    let mut max_id = 0u32;

    for block in &func.blocks {
        for inst in &block.instructions {
            // Check destination value
            if let Some(dest_id) = inst.dest_value_id() {
                max_id = max_id.max(dest_id);
            }

            // Scan operands manually
            visit_instruction_values(inst, &mut |v| {
                max_id = max_id.max(v.0);
            });
        }
    }

    max_id
}

/// Visit all value uses in an instruction
fn visit_instruction_values<F>(inst: &Instruction, f: &mut F)
where
    F: FnMut(Value),
{
    match inst {
        Instruction::Load { ptr, .. } => f(*ptr),
        Instruction::Store { val, ptr, .. } => {
            if let Operand::Value(v) = val {
                f(*v);
            }
            f(*ptr);
        }
        Instruction::BinOp { lhs, rhs, .. } => {
            if let Operand::Value(v) = lhs {
                f(*v);
            }
            if let Operand::Value(v) = rhs {
                f(*v);
            }
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            f(*base);
            if let Operand::Value(v) = offset {
                f(*v);
            }
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming {
                if let Operand::Value(v) = op {
                    f(*v);
                }
            }
        }
        _ => {
            // For other instructions, we don't need to track values
            // since we're only looking for max ID
        }
    }
}

/// Helper trait to get destination value ID from instruction
trait DestValue {
    fn dest_value_id(&self) -> Option<u32>;
}

impl DestValue for Instruction {
    fn dest_value_id(&self) -> Option<u32> {
        match self {
            Instruction::BinOp { dest, .. }
            | Instruction::UnaryOp { dest, .. }
            | Instruction::Cast { dest, .. }
            | Instruction::GetElementPtr { dest, .. }
            | Instruction::Load { dest, .. }
            | Instruction::Cmp { dest, .. }
            | Instruction::Phi { dest, .. }
            | Instruction::Alloca { dest, .. }
            | Instruction::DynAlloca { dest, .. }
            | Instruction::Copy { dest, .. }
            | Instruction::GlobalAddr { dest, .. }
            | Instruction::VaArg { dest, .. }
            | Instruction::AtomicRmw { dest, .. }
            | Instruction::AtomicCmpxchg { dest, .. }
            | Instruction::AtomicLoad { dest, .. }
            | Instruction::Intrinsic { dest: Some(dest), .. }
            | Instruction::Select { dest, .. }
            | Instruction::LabelAddr { dest, .. }
            | Instruction::GetReturnF64Second { dest }
            | Instruction::GetReturnF32Second { dest }
            | Instruction::GetReturnF128Second { dest } => Some(dest.0),
            Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                info.dest.map(|v| v.0)
            }
            _ => None,
        }
    }
}

/// Check if a stride is valid for x86-64 SIB encoding
fn is_valid_sib_scale(stride: i64) -> bool {
    matches!(stride, 1 | 2 | 4 | 8)
}

/// Build use-def chains using existing infrastructure from DCE/copy-prop patterns
#[allow(dead_code)]
fn build_use_count_map(func: &IrFunction) -> Vec<u32> {
    let max_id = compute_max_value_id(func);
    let mut use_count = vec![0u32; (max_id + 1) as usize];

    for block in &func.blocks {
        for inst in &block.instructions {
            // Reuse existing for_each_used_value pattern from DCE
            if let Instruction::Phi { dest, incoming, .. } = inst {
                // Handle phi specially to avoid self-references
                for (op, _) in incoming {
                    if let Operand::Value(v) = op {
                        if v.0 != dest.0 {
                            let idx = v.0 as usize;
                            if idx < use_count.len() {
                                use_count[idx] += 1;
                            }
                        }
                    }
                }
            } else {
                inst.for_each_used_value(|id| {
                    let idx = id as usize;
                    if idx < use_count.len() {
                        use_count[idx] += 1;
                    }
                });
            }
        }
    }

    use_count
}
