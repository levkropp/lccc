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
    } else if b.starts_with(b"test") {
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
