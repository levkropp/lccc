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

use crate::common::fx_hash::FxHashMap;
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
    let header_block = &func.blocks[header.0 as usize];
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

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

/// Revert a pointer IV back to indexed form
fn revert_pointer_iv(func: &mut IrFunction, ptr_iv: &IvsrPointerIV) -> bool {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();

    let index_iv = match ptr_iv.index_iv {
        Some(iv) => iv,
        None => return false,
    };

    // Find all uses of the pointer phi and collect them
    let ptr_uses = find_ptr_phi_uses(func, ptr_iv.ptr_phi_dest);

    if debug {
        eprintln!("[Un-IVSR]     Found {} uses of pointer phi", ptr_uses.len());
    }

    if ptr_uses.is_empty() {
        return false;
    }

    // Get the next value ID for creating new instructions
    let mut next_val_id = func.next_value_id;
    if next_val_id == 0 {
        // Compute max value ID if not cached
        next_val_id = compute_max_value_id(func) + 1;
    }

    // For each use of the pointer phi, we need to create:
    // 1. Multiply or shift: %offset = index * stride
    // 2. GEP: %addr = GEP(base, offset)
    // 3. Replace the pointer use with the new GEP result

    for (block_idx, inst_idx) in ptr_uses {
        let block = &mut func.blocks[block_idx];

        // Create offset calculation: %offset = index * stride (or shift for powers of 2)
        let offset_val = Value(next_val_id);
        next_val_id += 1;

        let offset_inst = create_offset_instruction(
            offset_val,
            index_iv,
            ptr_iv.stride,
        );

        // Create GEP: %addr = GEP(base, offset + init_offset)
        let gep_val = Value(next_val_id);
        next_val_id += 1;

        let gep_inst = if ptr_iv.init_offset == 0 {
            Instruction::GetElementPtr {
                dest: gep_val,
                base: ptr_iv.base_ptr,
                offset: Operand::Value(offset_val),
                ty: IrType::Ptr,
            }
        } else {
            // Need to add init_offset to the computed offset
            // Create: %adjusted_offset = offset + init_offset
            let adjusted_offset_val = Value(next_val_id);
            next_val_id += 1;

            let add_inst = Instruction::BinOp {
                dest: adjusted_offset_val,
                op: IrBinOp::Add,
                lhs: Operand::Value(offset_val),
                rhs: Operand::Const(IrConst::I64(ptr_iv.init_offset)),
                ty: IrType::I64,
            };

            // Insert the add instruction
            block.instructions.insert(inst_idx, add_inst);

            Instruction::GetElementPtr {
                dest: gep_val,
                base: ptr_iv.base_ptr,
                offset: Operand::Value(adjusted_offset_val),
                ty: IrType::Ptr,
            }
        };

        // Insert offset and GEP instructions before the use
        let insert_pos = if ptr_iv.init_offset == 0 {
            inst_idx
        } else {
            inst_idx + 1  // Account for the add instruction we inserted
        };

        block.instructions.insert(insert_pos, offset_inst);
        block.instructions.insert(insert_pos + 1, gep_inst);

        // Update the instruction to use the new GEP value instead of the pointer phi
        let use_inst_idx = if ptr_iv.init_offset == 0 {
            inst_idx + 2  // offset + GEP
        } else {
            inst_idx + 3  // add + offset + GEP
        };

        replace_ptr_use_in_instruction(
            &mut block.instructions[use_inst_idx],
            ptr_iv.ptr_phi_dest,
            gep_val,
        );
    }

    // Update the function's next_value_id cache
    func.next_value_id = next_val_id;

    // Note: We don't remove the pointer phi here because:
    // 1. It might still be used in the backedge GEP (which is now dead)
    // 2. Dead code elimination will clean it up later
    // 3. Removing it requires updating the phi's uses which is complex

    true
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

/// Find all uses of the pointer phi value (Load/Store instructions)
fn find_ptr_phi_uses(func: &IrFunction, ptr_val: Value) -> Vec<(usize, usize)> {
    let debug = std::env::var("CCC_DEBUG_UNIVSR").is_ok();
    let mut uses = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            // Check all instructions for uses of the pointer value
            if debug && block_idx < 5 {
                // Check if this instruction uses our pointer value in any operand
                let uses_ptr = match inst {
                    Instruction::Load { ptr, .. } => ptr.0 == ptr_val.0,
                    Instruction::Store { ptr, .. } => ptr.0 == ptr_val.0,
                    Instruction::GetElementPtr { base, .. } => base.0 == ptr_val.0,
                    _ => false,
                };

                if uses_ptr {
                    eprintln!("[Un-IVSR]       Block {} inst {}: {:?} uses Value({})",
                              block_idx, inst_idx, inst, ptr_val.0);
                }
            }

            match inst {
                Instruction::Load { ptr, .. } if ptr.0 == ptr_val.0 => {
                    uses.push((block_idx, inst_idx));
                }
                Instruction::Store { ptr, .. } if ptr.0 == ptr_val.0 => {
                    uses.push((block_idx, inst_idx));
                }
                _ => {}
            }
        }
    }

    if debug {
        eprintln!("[Un-IVSR]     Looking for uses of Value({}) - found {} uses",
                  ptr_val.0, uses.len());
    }

    uses
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
