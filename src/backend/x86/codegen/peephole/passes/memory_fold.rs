//! Memory operand folding pass.
//!
//! Folds a stack load followed by an ALU instruction that uses the loaded register
//! as a source operand into a single instruction with a memory source operand.
//!
//! Pattern:
//!   movq  -N(%rbp), %rcx       ; LoadRbp { reg: 1(rcx), offset: -N, size: Q }
//!   addq  %rcx, %rax           ; Other: rax = rax + rcx
//!
//! Transformed to:
//!   addq  -N(%rbp), %rax       ; rax = rax + mem[rbp-N]
//!
//! Supported ALU ops: add, sub, and, or, xor, cmp, test (with q/l suffixes).
//! The loaded register must be used as the first (source) operand in AT&T syntax.
//! We only fold when the loaded register is one of the scratch registers (rax=0,
//! rcx=1, rdx=2) to avoid breaking live register values.

use super::super::types::*;
use super::helpers::is_read_modify_write;

/// Format a stack slot as an assembly memory operand string.
/// Uses (%rbp) or (%rsp) depending on the original instruction text.
fn format_stack_offset(offset: i32, original_line: &str) -> String {
    if original_line.contains("(%rsp)") {
        format!("{}(%rsp)", offset)
    } else {
        format!("{}(%rbp)", offset)
    }
}

/// Try to parse an ALU instruction of the form "OPsuffix %src, %dst"
/// where OP is add/sub/and/or/xor/cmp/test.
/// Returns (op_name_with_suffix, dst_reg_str, src_family, dst_family).
fn parse_alu_reg_reg(trimmed: &str) -> Option<(&str, &str, RegId, RegId)> {
    let b = trimmed.as_bytes();
    if b.len() < 6 { return None; }

    let op_len = if b.starts_with(b"add")
        || b.starts_with(b"sub")
        || b.starts_with(b"and")
        || b.starts_with(b"xor")
        || b.starts_with(b"cmp")
    {
        3
    } else if b.starts_with(b"test") || b.starts_with(b"imul") {
        4
    } else if b.starts_with(b"or")
        && b.len() > 2
        && (b[2] == b'q' || b[2] == b'l' || b[2] == b'w' || b[2] == b'b')
    {
        2
    } else {
        return None;
    };

    let suffix = b[op_len];
    if suffix != b'q' && suffix != b'l' && suffix != b'w' && suffix != b'b' {
        return None;
    }
    let op_with_suffix = &trimmed[..op_len + 1];

    let rest = trimmed[op_len + 1..].trim();
    let (src_str, dst_str) = rest.split_once(',')?;
    let src_str = src_str.trim();
    let dst_str = dst_str.trim();

    if !src_str.starts_with('%') || !dst_str.starts_with('%') {
        return None;
    }

    let src_fam = register_family_fast(src_str);
    let dst_fam = register_family_fast(dst_str);
    if src_fam == REG_NONE || dst_fam == REG_NONE {
        return None;
    }

    Some((op_with_suffix, dst_str, src_fam, dst_fam))
}

/// Fold movsd stack load into subsequent scalar FP binary op as memory operand.
///
/// Pattern (produced after eliminate_fp_xmm_roundtrips):
///   movsd -M(%rbp), %xmm1   ; Other{dest_reg: 25 (xmm1), rbp_offset: -M}
///   OP %xmm1, %xmm0          ; Other{dest_reg: 24 (xmm0)}, OP ∈ {mulsd, addsd, subsd, divsd}
///
/// Transformed to:
///   OP -M(%rbp), %xmm0
///
/// This reduces 4 instructions per FP binop (after roundtrip elimination) to 3.
pub(super) fn fold_fp_memory_operands(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Look for Other{dest_reg: 25} = writes to %xmm1 (family 24+1=25)
        if let LineKind::Other { dest_reg: 25 } = infos[i].kind {
            let offset = infos[i].rbp_offset;
            if offset == RBP_OFFSET_NONE { i += 1; continue; }

            let line_i = infos[i].trimmed(store.get(i));
            // Verify it is a movsd load from stack (not another xmm1-writing insn)
            if !line_i.starts_with("movsd ") || !line_i.ends_with(", %xmm1") {
                i += 1; continue;
            }

            // Find next non-NOP (skip only NOPs, not other instructions)
            let mut j = i + 1;
            while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
            if j >= len { i += 1; continue; }

            let line_j = infos[j].trimmed(store.get(j));
            let mem_op = format_stack_offset(offset, line_i);
            let replacement = match line_j {
                "mulsd %xmm1, %xmm0" => Some(format!("    mulsd {}, %xmm0", mem_op)),
                "addsd %xmm1, %xmm0" => Some(format!("    addsd {}, %xmm0", mem_op)),
                "subsd %xmm1, %xmm0" => Some(format!("    subsd {}, %xmm0", mem_op)),
                "divsd %xmm1, %xmm0" => Some(format!("    divsd {}, %xmm0", mem_op)),
                _ => None,
            };
            if let Some(new_text) = replacement {
                mark_nop(&mut infos[i]);
                replace_line(store, &mut infos[j], j, new_text);
                changed = true;
                i = j + 1;
                continue;
            }
        }

        i += 1;
    }
    changed
}

/// Fold stack-load-to-scratch relay moves: eliminate the scratch register
/// as intermediary when loading from a stack slot to another register.
///
/// Pattern:
///   movq  -N(%rbp), %rax       ; LoadRbp { reg: 0(rax), offset: -N, size: Q }
///   movq  %rax, %r12           ; Other: copy rax to callee-saved/arg register
///
/// Transformed to:
///   movq  -N(%rbp), %r12       ; direct load to destination register
///
/// Safety: The scratch register (rax) must not be read between the load and
/// the copy. We only fold loads to rax (reg 0) since codegen guarantees rax
/// is a temporary. The destination register must be a different GP register.
/// We verify rax is dead after (not read before being overwritten) to ensure
/// we don't break code that uses rax after the copy.
pub(super) fn fold_load_relay(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: Find a load from stack to %rax (scratch register).
        if let LineKind::LoadRbp { reg: 0, offset, size } = infos[i].kind {
            // Only fold Q and L loads (not sign-extending SLQ, which changes value).
            if size != MoveSize::Q && size != MoveSize::L {
                i += 1; continue;
            }

            // Step 2: Find next non-NOP instruction.
            let mut j = i + 1;
            while j < len && infos[j].is_nop() { j += 1; }
            if j >= len { i += 1; continue; }

            // Step 3: Check if it's "movq %rax, %DEST" or "movl %eax, %DESTd"
            // where DEST is a different GP register.
            let dest_reg = match infos[j].kind {
                LineKind::Other { dest_reg } if dest_reg != REG_NONE && dest_reg != 0 && dest_reg <= REG_GP_MAX => {
                    let line_j = infos[j].trimmed(store.get(j));
                    // Must be a simple register-to-register mov
                    let is_movq_rax = line_j.starts_with("movq %rax, %") && !line_j.contains('(');
                    let is_movl_eax = line_j.starts_with("movl %eax, %") && !line_j.contains('(');
                    if is_movq_rax || is_movl_eax {
                        dest_reg
                    } else {
                        i += 1; continue;
                    }
                }
                _ => { i += 1; continue; }
            };

            // Step 4: Verify no intervening store to the same offset.
            let mut intervening_store = false;
            for k in (i + 1)..j {
                if let LineKind::StoreRbp { offset: so, .. } = infos[k].kind {
                    if so == offset { intervening_store = true; break; }
                }
            }
            if intervening_store { i += 1; continue; }

            // Step 5: Check rax liveness after the copy.
            if !is_rax_dead_after(store, infos, j + 1, len) { i += 1; continue; }

            // Step 6: Transform! Replace load target and eliminate the copy.
            let load_line = infos[i].trimmed(store.get(i));
            let mem_op = format_stack_offset(offset, load_line);
            let dest_name = REG_NAMES[if size == MoveSize::L { 1 } else { 0 }][dest_reg as usize];
            let mnemonic = size.mnemonic();
            let new_load = format!("    {} {}, {}", mnemonic, mem_op, dest_name);

            replace_line(store, &mut infos[i], i, new_load);
            mark_nop(&mut infos[j]);
            changed = true;
            i = j + 1;
            continue;
        }

        i += 1;
    }

    changed
}

/// Fold load+leaq+store relay: eliminate accumulator relay for address computation.
///
/// Pattern:
///   movq  -N(%rbp), %rax       ; load base pointer from stack
///   leaq  K(%rax), %rax        ; compute base + offset
///   movq  %rax, %r12           ; store result to dest register
///
/// Transformed to:
///   movq  -N(%rbp), %r12       ; load directly to dest
///   leaq  K(%r12), %r12        ; compute offset in-place
///
/// Saves 1 instruction per occurrence. Safe when %rax is dead after the copy.
pub(super) fn fold_leaq_relay(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 2 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: Load from stack to %rax.
        if let LineKind::LoadRbp { reg: 0, offset, size: MoveSize::Q } = infos[i].kind {
            // Step 2: Next must be leaq K(%rax), %rax
            let mut j = i + 1;
            while j < len && infos[j].is_nop() { j += 1; }
            if j >= len { i += 1; continue; }

            let leaq_offset = {
                let lj = infos[j].trimmed(store.get(j));
                if !lj.starts_with("leaq ") || !lj.ends_with(", %rax") { i += 1; continue; }
                let inner = &lj[5..lj.len() - 6]; // between "leaq " and ", %rax"
                if !inner.ends_with("(%rax)") { i += 1; continue; }
                let num_str = &inner[..inner.len() - 6]; // before "(%rax)"
                match num_str.parse::<i64>() {
                    Ok(v) => v,
                    Err(_) => { i += 1; continue; }
                }
            };

            // Step 3: Next must be movq %rax, %DEST
            let mut k = j + 1;
            while k < len && infos[k].is_nop() { k += 1; }
            if k >= len { i += 1; continue; }

            let dest_reg = match infos[k].kind {
                LineKind::Other { dest_reg } if dest_reg != REG_NONE && dest_reg != 0 && dest_reg <= REG_GP_MAX => {
                    let lk = infos[k].trimmed(store.get(k));
                    if lk.starts_with("movq %rax, %") && !lk.contains('(') {
                        dest_reg
                    } else { i += 1; continue; }
                }
                _ => { i += 1; continue; }
            };

            // Step 4: Check rax is dead after k.
            let rax_dead = is_rax_dead_after(store, infos, k + 1, len);
            if !rax_dead { i += 1; continue; }

            // Step 5: Transform.
            let load_line = infos[i].trimmed(store.get(i));
            let mem_op = format_stack_offset(offset, load_line);
            let dest_64 = REG_NAMES[0][dest_reg as usize];
            let new_load = format!("    movq {}, {}", mem_op, dest_64);
            let new_leaq = format!("    leaq {}({}), {}", leaq_offset, dest_64, dest_64);

            replace_line(store, &mut infos[i], i, new_load);
            replace_line(store, &mut infos[j], j, new_leaq);
            mark_nop(&mut infos[k]);
            changed = true;
            i = k + 1;
            continue;
        }
        i += 1;
    }
    changed
}

/// Fold load+cltq+store relay: eliminate accumulator relay for sign-extend load.
///
/// Pattern:
///   movq  -N(%rbp), %rax       ; load 64-bit value (only lower 32 used)
///   cltq                       ; sign-extend %eax → %rax
///   movq  %rax, %r12           ; store sign-extended result
///
/// Transformed to:
///   movslq -N(%rbp), %r12      ; sign-extending load directly to dest
///
/// Saves 2 instructions per occurrence.
pub(super) fn fold_cltq_relay(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 2 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: Load from stack to %rax (either movq or movl).
        if let LineKind::LoadRbp { reg: 0, offset, size } = infos[i].kind {
            if size != MoveSize::Q && size != MoveSize::L { i += 1; continue; }

            // Step 2: Next must be cltq.
            let mut j = i + 1;
            while j < len && infos[j].is_nop() { j += 1; }
            if j >= len { i += 1; continue; }
            {
                let lj = infos[j].trimmed(store.get(j));
                if lj != "cltq" { i += 1; continue; }
            }

            // Step 3: Next must be movq %rax, %DEST.
            let mut k = j + 1;
            while k < len && infos[k].is_nop() { k += 1; }
            if k >= len { i += 1; continue; }

            let dest_reg = match infos[k].kind {
                LineKind::Other { dest_reg } if dest_reg != REG_NONE && dest_reg != 0 && dest_reg <= REG_GP_MAX => {
                    let lk = infos[k].trimmed(store.get(k));
                    if lk.starts_with("movq %rax, %") && !lk.contains('(') {
                        dest_reg
                    } else { i += 1; continue; }
                }
                _ => { i += 1; continue; }
            };

            // Step 4: Check rax is dead after k.
            let rax_dead = is_rax_dead_after(store, infos, k + 1, len);
            if !rax_dead { i += 1; continue; }

            // Step 5: Transform! Replace all 3 instructions with one movslq.
            let load_line = infos[i].trimmed(store.get(i));
            let mem_op = format_stack_offset(offset, load_line);
            let dest_64 = REG_NAMES[0][dest_reg as usize];
            let new_inst = format!("    movslq {}, {}", mem_op, dest_64);

            replace_line(store, &mut infos[i], i, new_inst);
            mark_nop(&mut infos[j]);
            mark_nop(&mut infos[k]);
            changed = true;
            i = k + 1;
            continue;
        }
        i += 1;
    }
    changed
}

/// Fold movzbq/movzwq/movsbq/movswq relay: eliminate rax as intermediary
/// for zero/sign-extend-then-copy patterns.
///
/// Pattern:
///   movzbq  %al, %rax          ; zero-extend byte result to 64-bit
///   movq    %rax, %r12         ; copy to dest register
///
/// Transformed to:
///   movzbl  %al, %r12d         ; zero-extend directly to dest (32-bit write, implicit 64-bit zext)
///
/// Also handles movzwq→movq, movsbq→movq, movswq→movq, and movslq→movq.
/// Saves 1 instruction per occurrence.
pub(super) fn fold_extend_relay(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: Look for extension instructions writing to %rax from %al/%ax/%eax.
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));

            // Parse the extension type and source sub-register.
            let (new_op, src_sub_idx) = if line_i == "movzbq %al, %rax" {
                // movzbq %al, %rax → movzbl %al, %DESTd
                ("movzbl", 3usize) // 3 = B (byte) index in REG_NAMES
            } else if line_i == "movzwq %ax, %rax" {
                ("movzwl", 2) // W (word)
            } else if line_i == "movsbq %al, %rax" {
                ("movsbl", 3) // B
            } else if line_i == "movswq %ax, %rax" {
                ("movswl", 2) // W
            } else {
                i += 1; continue;
            };

            // Step 2: Next must be movq %rax, %DEST.
            let mut j = i + 1;
            while j < len && infos[j].is_nop() { j += 1; }
            if j >= len { i += 1; continue; }

            let dest_reg = match infos[j].kind {
                LineKind::Other { dest_reg } if dest_reg != REG_NONE && dest_reg != 0 && dest_reg <= REG_GP_MAX => {
                    let lj = infos[j].trimmed(store.get(j));
                    if lj.starts_with("movq %rax, %") && !lj.contains('(') {
                        dest_reg
                    } else { i += 1; continue; }
                }
                _ => { i += 1; continue; }
            };

            // Step 3: Check rax is dead after.
            if !is_rax_dead_after(store, infos, j + 1, len) { i += 1; continue; }

            // Step 4: Transform.
            // Use the same sub-register for the source (al/ax from rax family=0).
            let src_name = REG_NAMES[src_sub_idx][0]; // %al or %ax
            let dest_32 = REG_NAMES[1][dest_reg as usize]; // %r12d etc.
            let new_inst = format!("    {} {}, {}", new_op, src_name, dest_32);

            replace_line(store, &mut infos[i], i, new_inst);
            mark_nop(&mut infos[j]);
            changed = true;
            i = j + 1;
            continue;
        }
        i += 1;
    }
    changed
}

/// General accumulator relay fold: retarget instructions that write to %rax
/// when the result is immediately copied to another register.
///
/// Handles:
///   leaq   X, %rax  +  movq %rax, %REG  →  leaq X, %REG
///   movslq X, %rax   +  movq %rax, %REG  →  movslq X, %REG
///   xorl   %eax, %eax + movq %rax, %REG  →  xorl %REGd, %REGd
///   addq   X, %rax  +  movq %rax, %REG  →  (not safe: flags + read-modify-write)
///
/// Only applies to instructions that purely write %rax without reading it first,
/// and where %rax is dead after the copy.
pub(super) fn fold_general_relay(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: Instruction writes to %rax (dest_reg == 0).
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));

            // Step 2: Next must be movq %rax, %DEST.
            let mut j = i + 1;
            while j < len && infos[j].is_nop() { j += 1; }
            if j >= len { i += 1; continue; }
            let dest_reg = match infos[j].kind {
                LineKind::Other { dest_reg } if dest_reg != REG_NONE && dest_reg != 0 && dest_reg <= REG_GP_MAX => {
                    let lj = infos[j].trimmed(store.get(j));
                    if lj.starts_with("movq %rax, %") && !lj.contains('(') {
                        dest_reg
                    } else { i += 1; continue; }
                }
                _ => { i += 1; continue; }
            };

            // Step 3: Check rax is dead after.
            if !is_rax_dead_after(store, infos, j + 1, len) { i += 1; continue; }

            let dest_64 = REG_NAMES[0][dest_reg as usize];
            let dest_32 = REG_NAMES[1][dest_reg as usize];

            // Step 4: Match specific retargetable patterns.
            let new_inst = if line_i.starts_with("leaq ") && line_i.ends_with(", %rax") {
                // leaq X, %rax → leaq X, %REG
                // Safe: leaq doesn't read %rax (it computes an address, doesn't deref).
                // But check the source doesn't reference rax!
                let src = &line_i[5..line_i.len() - 6]; // between "leaq " and ", %rax"
                if src.contains("%rax") || src.contains("%eax") {
                    i += 1; continue;
                }
                Some(format!("    leaq {}, {}", src, dest_64))
            } else if line_i.starts_with("movslq ") && line_i.ends_with(", %rax") {
                // movslq X, %rax → movslq X, %REG
                let src = &line_i[7..line_i.len() - 6];
                if src.contains("%rax") || src.contains("%eax") {
                    i += 1; continue;
                }
                Some(format!("    movslq {}, {}", src, dest_64))
            } else if line_i == "xorl %eax, %eax" {
                // xorl %eax, %eax → xorl %REGd, %REGd
                Some(format!("    xorl {}, {}", dest_32, dest_32))
            } else if line_i.starts_with("movq $") && line_i.ends_with(", %rax") {
                // movq $imm, %rax → movq $imm, %REG
                let imm = &line_i[5..line_i.len() - 6];
                Some(format!("    movq {}, {}", imm, dest_64))
            } else if line_i.starts_with("movl $") && line_i.ends_with(", %eax") {
                // movl $imm, %eax → movl $imm, %REGd
                let imm = &line_i[5..line_i.len() - 6];
                Some(format!("    movl {}, {}", imm, dest_32))
            } else if line_i.starts_with("movq ") && line_i.ends_with(", %rax") && line_i.contains('(') {
                // movq N(%reg), %rax → movq N(%reg), %REG (pointer dereference)
                // Safe: source is a memory operand, doesn't read %rax as a value.
                // But check the addressing mode doesn't use %rax as base/index!
                let src = &line_i[5..line_i.len() - 6]; // between "movq " and ", %rax"
                if src.contains("%rax") || src.contains("%eax") {
                    i += 1; continue;
                }
                Some(format!("    movq {}, {}", src, dest_64))
            } else if line_i.starts_with("movl ") && line_i.ends_with(", %eax") && line_i.contains('(') {
                // movl N(%reg), %eax → movl N(%reg), %REGd (32-bit pointer dereference)
                let src = &line_i[5..line_i.len() - 6];
                if src.contains("%rax") || src.contains("%eax") {
                    i += 1; continue;
                }
                Some(format!("    movl {}, {}", src, dest_32))
            } else if line_i.starts_with("movzbq ") && line_i.ends_with(", %rax") {
                // movzbq N(%reg), %rax → movzbl N(%reg), %REGd (byte load zero-extend)
                let src = &line_i[7..line_i.len() - 6];
                if src.contains("%rax") || src.contains("%eax") {
                    i += 1; continue;
                }
                Some(format!("    movzbl {}, {}", src, dest_32))
            } else if line_i.starts_with("movzwq ") && line_i.ends_with(", %rax") {
                // movzwq N(%reg), %rax → movzwl N(%reg), %REGd
                let src = &line_i[7..line_i.len() - 6];
                if src.contains("%rax") || src.contains("%eax") {
                    i += 1; continue;
                }
                Some(format!("    movzwl {}, {}", src, dest_32))
            } else {
                None
            };

            if let Some(new_text) = new_inst {
                replace_line(store, &mut infos[i], i, new_text);
                mark_nop(&mut infos[j]);
                changed = true;
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    changed
}

/// Fold store relay: `movq %reg, %rax; movq %rax, N(%rsp)` → `movq %reg, N(%rsp)`.
/// Eliminates the intermediate %rax relay for register-to-stack stores.
pub(super) fn fold_store_relay(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: movq %REG, %rax (or movl %REGd, %eax)
        let (src_reg, is_32bit) = match infos[i].kind {
            LineKind::Other { dest_reg: 0 } => {
                let line = infos[i].trimmed(store.get(i));
                if line.starts_with("movq %") && line.ends_with(", %rax") && !line.contains('(') {
                    let src = &line[6..line.len() - 6]; // between "movq %" and ", %rax"
                    if !src.contains('%') { // simple register name
                        (src.to_string(), false)
                    } else { i += 1; continue; }
                } else if line.starts_with("movl %") && line.ends_with(", %eax") && !line.contains('(') {
                    let src = &line[6..line.len() - 6];
                    if !src.contains('%') {
                        (src.to_string(), true)
                    } else { i += 1; continue; }
                } else { i += 1; continue; }
            }
            _ => { i += 1; continue; }
        };

        // Step 2: Next must be movq %rax, N(%rsp) or movl %eax, N(%rsp)
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len { i += 1; continue; }

        let stored = match infos[j].kind {
            LineKind::StoreRbp { reg: 0, offset, size } => {
                // movq/movl %rax/%eax → stack
                Some((offset, size))
            }
            _ => None,
        };

        if let Some((offset, _size)) = stored {
            // Check rax is dead after the store
            if is_rax_dead_after(store, infos, j + 1, len) {
                // Fold: movq %reg, N(%rsp) directly
                let mnem = if is_32bit { "movl" } else { "movq" };
                let line = infos[j].trimmed(store.get(j));
                // Extract the stack operand from the store instruction
                if let Some(comma) = line.rfind(',') {
                    let mem_part = line[comma + 1..].trim();
                    let new_inst = format!("    {} %{}, {}", mnem, src_reg, mem_part);
                    replace_line(store, &mut infos[j], j, new_inst);
                    mark_nop(&mut infos[i]);
                    changed = true;
                    i = j + 1;
                    continue;
                }
            }
        }

        i += 1;
    }

    changed
}

/// Check if %rax is dead starting from instruction index `start`.
/// Returns true if rax is overwritten before being read within a 16-instruction window.
fn is_rax_dead_after(store: &LineStore, infos: &[LineInfo], start: usize, len: usize) -> bool {
    let scan_limit = (start + 16).min(len);
    let mut scan = start;
    while scan < scan_limit {
        if infos[scan].is_nop() { scan += 1; continue; }
        // Control flow = rax is dead (caller-saved)
        if infos[scan].is_barrier() { return true; }
        if infos[scan].reg_refs & 1 != 0 {
            match infos[scan].kind {
                LineKind::LoadRbp { reg: 0, .. } => return true,
                LineKind::Other { dest_reg: 0 } => {
                    let t = infos[scan].trimmed(store.get(scan));
                    if (t.ends_with(", %rax") || t.ends_with(", %eax"))
                        && !is_read_modify_write(t) {
                        return true;
                    }
                    if t == "xorl %eax, %eax" { return true; }
                    return false; // rax read
                }
                _ => return false, // rax read
            }
        }
        scan += 1;
    }
    true // ran out of window = assume dead
}

/// Fold stack loads into subsequent ALU instructions as memory operands.
///
/// Safety: We only fold when the loaded register (the one being eliminated) is
/// a scratch register (rax=0, rcx=1, rdx=2) because the codegen guarantees
/// these are temporary and overwritten before the next use. We also verify
/// the loaded register is not the *destination* of the ALU instruction to avoid
/// creating a memory-destination instruction (which would write to the stack slot).
pub(super) fn fold_memory_operands(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        if let LineKind::LoadRbp { reg: load_reg, offset, size: load_size } = infos[i].kind {
            // Only fold loads into scratch registers (rax=0, rcx=1, rdx=2)
            if load_reg > 2 {
                i += 1;
                continue;
            }

            // Only fold Q and L loads (64-bit and 32-bit). SLQ (sign-extending)
            // loads have different semantics.
            if load_size != MoveSize::Q && load_size != MoveSize::L {
                i += 1;
                continue;
            }

            // Find the next non-NOP, non-empty instruction
            let mut j = i + 1;
            while j < len && (infos[j].is_nop() || infos[j].kind == LineKind::Empty) {
                j += 1;
            }
            if j >= len {
                i += 1;
                continue;
            }

            let is_foldable_target = matches!(infos[j].kind,
                LineKind::Other { .. } | LineKind::Cmp);
            if is_foldable_target {
                let trimmed_j = infos[j].trimmed(store.get(j));

                // Special case: testq/testl %REG, %REG where REG is the loaded scratch reg.
                // Fold to cmpq/cmpl $0, -N(%rbp) — both set ZF/SF/PF identically.
                let test_self = if load_reg == 0 {
                    trimmed_j == "testq %rax, %rax" || trimmed_j == "testl %eax, %eax"
                } else if load_reg == 1 {
                    trimmed_j == "testq %rcx, %rcx" || trimmed_j == "testl %ecx, %ecx"
                } else {
                    trimmed_j == "testq %rdx, %rdx" || trimmed_j == "testl %edx, %edx"
                };
                if test_self {
                    let load_line = infos[i].trimmed(store.get(i));
                    let mem_op = format_stack_offset(offset, load_line);
                    let cmp_suffix = if load_size == MoveSize::L { "cmpl" } else { "cmpq" };
                    let new_inst = format!("    {} $0, {}", cmp_suffix, mem_op);
                    mark_nop(&mut infos[i]);
                    replace_line(store, &mut infos[j], j, new_inst);
                    changed = true;
                    i = j + 1;
                    continue;
                }

                if let Some((op_suffix, dst_str, src_fam, dst_fam)) = parse_alu_reg_reg(trimmed_j) {
                    if src_fam == load_reg && dst_fam != load_reg {
                        // Check for intervening store to the same offset
                        let mut intervening_store = false;
                        for k in (i + 1)..j {
                            if let LineKind::StoreRbp { offset: so, .. } = infos[k].kind {
                                if so == offset {
                                    intervening_store = true;
                                    break;
                                }
                            }
                        }
                        if intervening_store {
                            i += 1;
                            continue;
                        }

                        let load_line = infos[i].trimmed(store.get(i));
                        let mem_op = format_stack_offset(offset, load_line);
                        let new_inst = format!("    {} {}, {}", op_suffix, mem_op, dst_str);

                        mark_nop(&mut infos[i]);
                        replace_line(store, &mut infos[j], j, new_inst);
                        changed = true;
                        i = j + 1;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }

    changed
}
