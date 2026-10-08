//! Local peephole pattern matching passes.
//!
//! Merges 7 simple local passes into a single linear scan (`combined_local_pass`)
//! to avoid redundant iteration over the lines array. Also includes
//! `fuse_movq_ext_truncation` which fuses movq + extension/truncation patterns.
//!
//! Merged passes in `combined_local_pass`:
//!   1. eliminate_redundant_movq_self: movq %reg, %reg (same src/dst)
//!   2. eliminate_reverse_move: movq %A,%B + movq %B,%A -> remove second
//!   3. eliminate_redundant_jumps: jmp to the immediately following label
//!   4. eliminate_cond_branch_inversion: jCC+jmp+label -> j!CC (inverted)
//!   5. eliminate_adjacent_store_load: store/load at same %rbp offset
//!   6. eliminate_redundant_zero_extend: redundant zero/sign extensions
//!   7. eliminate_redundant_xorl_zero: xorl %eax,%eax when %rax already zero

use super::super::types::*;
use super::helpers::{is_valid_gp_reg, has_implicit_reg_usage, replace_reg_family, is_callee_saved_reg, get_dest_reg, is_read_modify_write};

/// Format a stack offset string for text matching/generation.
/// Checks context to decide between (%rbp) and (%rsp).
/// Convert a 64-bit register name to its 32-bit equivalent.
/// High registers (%r8-%r15) get 'd' suffix, classic regs use 'e' prefix form.
fn reg_64_to_32(r64: &str) -> String {
    match r64 {
        "%rax" => "%eax".into(), "%rcx" => "%ecx".into(),
        "%rdx" => "%edx".into(), "%rbx" => "%ebx".into(),
        "%rsp" => "%esp".into(), "%rbp" => "%ebp".into(),
        "%rsi" => "%esi".into(), "%rdi" => "%edi".into(),
        _ if r64.starts_with("%r") => format!("{}d", r64),
        _ => String::new(),
    }
}

fn stack_offset_str(offset: i32, context: &str) -> String {
    if context.contains("(%rsp)") {
        format!("{}(%rsp)", offset)
    } else {
        format!("{}(%rbp)", offset)
    }
}

pub(super) fn combined_local_pass(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    // Track whether %rax is known to be zero for redundant xorl elimination.
    // This is set to true after `xorl %eax, %eax` and stays true across
    // StoreRbp instructions (which don't modify register values), but is
    // invalidated by anything that writes %rax, or by control flow barriers.
    let mut rax_is_zero = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // --- Pattern: redundant xorl %eax, %eax elimination ---
        // When %rax is already known to be zero (from a previous xorl %eax, %eax),
        // and only StoreRbp instructions intervene (which read but don't modify
        // registers), the repeated xorl is redundant.
        //
        // Common pattern from codegen zeroing multiple local variables:
        //   xorl %eax, %eax          # sets rax = 0
        //   movq %rax, -N(%rbp)      # stores 0, rax still 0
        //   xorl %eax, %eax          # REDUNDANT
        //   movq %rax, -M(%rbp)      # stores 0, rax still 0
        if rax_is_zero {
            if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
                let trimmed = infos[i].trimmed(store.get(i));
                if trimmed == "xorl %eax, %eax" {
                    mark_nop(&mut infos[i]);
                    changed = true;
                    i += 1;
                    continue;
                }
            }
        }

        // --- Pattern: dead xorl %eax before overwriting load ---
        // When xorl %eax, %eax is immediately followed by an instruction that
        // completely overwrites %rax (a load from stack), the xorl is dead.
        //
        //   xorl %eax, %eax          # DEAD — overwritten by next instruction
        //   movq -N(%rbp), %rax      # (or movslq, movzbq, etc.)
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let trimmed = infos[i].trimmed(store.get(i));
            if trimmed == "xorl %eax, %eax" {
                // Check if the next non-nop instruction overwrites %rax.
                let mut j = i + 1;
                while j < len && infos[j].is_nop() { j += 1; }
                if j < len {
                    let overwrites_rax = match infos[j].kind {
                        LineKind::LoadRbp { reg: 0, .. } => true, // movq/movslq/etc → %rax
                        // Do NOT treat `call` as overwriting %rax here — for variadic
                        // functions (printf, etc.), %al is an INPUT specifying the number
                        // of SSE register arguments. Removing xorl %eax before call
                        // leaves %al with garbage, causing crashes.
                        // LineKind::Call => true,
                        LineKind::Other { dest_reg: 0 } => {
                            // Check if it's a load that writes %rax (not a read-modify-write)
                            let nj = infos[j].trimmed(store.get(j));
                            nj.starts_with("movq ") || nj.starts_with("movl ")
                                || nj.starts_with("movslq ") || nj.starts_with("movzbq ")
                                || nj.starts_with("movzwq ") || nj.starts_with("movsbq ")
                                || nj.starts_with("movswq ") || nj.starts_with("leaq ")
                                || nj.starts_with("xorl %eax") // duplicate xorl
                                || nj.starts_with("movabsq ")
                        }
                        _ => false,
                    };
                    if overwrites_rax {
                        mark_nop(&mut infos[i]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Update rax_is_zero tracking based on current instruction.
        match infos[i].kind {
            LineKind::StoreRbp { .. } => {
                // Stores to stack don't modify registers, rax_is_zero unchanged.
            }
            LineKind::Other { dest_reg: 0 } => {
                // Something writes to %rax. Check if it's xorl %eax, %eax.
                let trimmed = infos[i].trimmed(store.get(i));
                rax_is_zero = trimmed == "xorl %eax, %eax";
            }
            LineKind::Other { dest_reg } if dest_reg != 0 => {
                // Writes to a non-rax register, rax_is_zero unchanged.
                // But check if it also reads/clobbers rax implicitly.
                // Most Other instructions only write their dest_reg.
                // Conservative: only keep rax_is_zero if the instruction
                // doesn't reference rax at all (via reg_refs).
                if infos[i].reg_refs & 1 != 0 {
                    // References rax - could be a read or write, invalidate
                    // But actually a read of rax is fine for rax_is_zero.
                    // Only a write to rax matters. Since dest_reg != 0,
                    // rax is not the destination, so it's a read - OK.
                    // Exception: instructions like div/idiv/mul/cqto that
                    // implicitly clobber rax through dest_reg rdx.
                    let trimmed = infos[i].trimmed(store.get(i));
                    if trimmed.starts_with("div") || trimmed.starts_with("idiv")
                        || trimmed.starts_with("mul") || trimmed.starts_with("imul")
                        || trimmed == "cqto" || trimmed == "cqo" || trimmed == "cdq"
                        || trimmed.starts_with("xchg") || trimmed.starts_with("cmpxchg") {
                        rax_is_zero = false;
                    }
                    // Otherwise rax is only read, not written - keep tracking.
                }
            }
            LineKind::LoadRbp { reg: 0, .. } => {
                // Load to rax - rax is no longer zero
                rax_is_zero = false;
            }
            LineKind::LoadRbp { .. } => {
                // Load to non-rax register, rax_is_zero unchanged.
            }
            LineKind::Label | LineKind::Jmp | LineKind::JmpIndirect
            | LineKind::CondJmp | LineKind::Ret | LineKind::Call => {
                // Control flow or label - invalidate tracking
                rax_is_zero = false;
            }
            LineKind::Pop { reg: 0 } | LineKind::SetCC { reg: 0 } => {
                rax_is_zero = false;
            }
            LineKind::Pop { .. } | LineKind::SetCC { .. }
            | LineKind::Push { .. } | LineKind::Cmp | LineKind::Directive => {
                // Don't affect rax
            }
            _ => {
                // Conservative: invalidate
                rax_is_zero = false;
            }
        }

        // --- Pattern: self-move elimination (movq %reg, %reg) ---
        // Pre-classified as SelfMove during classify_line, avoiding string parsing.
        if infos[i].kind == LineKind::SelfMove {
            mark_nop(&mut infos[i]);
            changed = true;
            i += 1;
            continue;
        }

        // --- Pattern: reverse-move elimination ---
        // Detects `movq %regA, %regB` followed by `movq %regB, %regA` and
        // eliminates the second mov (since %regA still holds the original value).
        //
        // Safety: We only skip NOPs and StoreRbp between the two instructions.
        // IMPORTANT: Don't eliminate reverse-moves where regA is caller-saved
        // and the reverse-move is needed to reload a param after a call.
        // The `pinned` check in mark_nop handles some cases, but we also skip
        // this pattern entirely when a Call appears between the two moves.
        // StoreRbp reads registers but never modifies any GP register value.
        // Any other instruction type causes the search to stop via `break`.
        if let LineKind::Other { dest_reg: dest_a } = infos[i].kind {
            if is_valid_gp_reg(dest_a) {
                let line_i = infos[i].trimmed(store.get(i));
                // Parse "movq %srcReg, %dstReg" pattern
                if let Some(rest) = line_i.strip_prefix("movq ") {
                    if let Some((src_str, dst_str)) = rest.split_once(',') {
                        let src = src_str.trim();
                        let dst = dst_str.trim();
                        let src_fam = register_family_fast(src);
                        let dst_fam = register_family_fast(dst);
                        // Both must be GP registers, different families, both register operands
                        if is_valid_gp_reg(src_fam) && is_valid_gp_reg(dst_fam)
                            && src_fam != dst_fam
                            && src.starts_with('%') && dst.starts_with('%')
                        {
                            // Find the next non-NOP, non-StoreRbp instruction.
                            // Limit search to 8 lines to avoid pathological scanning.
                            let mut j = i + 1;
                            let search_limit = (i + 8).min(len);
                            while j < search_limit {
                                if infos[j].is_nop() {
                                    j += 1;
                                    continue;
                                }
                                if matches!(infos[j].kind, LineKind::StoreRbp { .. }) {
                                    j += 1;
                                    continue;
                                }
                                break;
                            }
                            if j < search_limit {
                                // Check if line j is the reverse: movq %dstReg, %srcReg
                                if let LineKind::Other { dest_reg: dest_b } = infos[j].kind {
                                    if dest_b == src_fam {
                                        let line_j = infos[j].trimmed(store.get(j));
                                        if let Some(rest_j) = line_j.strip_prefix("movq ") {
                                            if let Some((src_j, dst_j)) = rest_j.split_once(',') {
                                                let src_j = src_j.trim();
                                                let dst_j = dst_j.trim();
                                                let src_j_fam = register_family_fast(src_j);
                                                let dst_j_fam = register_family_fast(dst_j);
                                                if src_j_fam == dst_fam && dst_j_fam == src_fam {
                                                    mark_nop(&mut infos[j]);
                                                    changed = true;
                                                    i += 1;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern: redundant jump to next label ---
        if infos[i].kind == LineKind::Jmp {
            let jmp_line = infos[i].trimmed(store.get(i));
            if let Some(target) = jmp_line.strip_prefix("jmp ") {
                let target = target.trim();
                // Find the next non-NOP, non-empty line
                let mut found_redundant = false;
                for j in (i + 1)..len {
                    if infos[j].is_nop() || infos[j].kind == LineKind::Empty {
                        continue;
                    }
                    if infos[j].kind == LineKind::Label {
                        let next = infos[j].trimmed(store.get(j));
                        if let Some(label) = next.strip_suffix(':') {
                            if label == target {
                                mark_nop(&mut infos[i]);
                                changed = true;
                                found_redundant = true;
                            }
                        }
                    }
                    break;
                }
                if found_redundant {
                    i += 1;
                    continue;
                }
            }
        }

        // --- Pattern: conditional branch inversion for fall-through ---
        // Detects:
        //   jCC .Ltrue        (conditional jump)
        //   jmp .Lfalse       (unconditional jump)
        //   .Ltrue:           (label matching the conditional target)
        //
        // Transforms to:
        //   j!CC .Lfalse      (inverted condition, jump to false target)
        //   .Ltrue:           (fall through naturally)
        if infos[i].kind == LineKind::CondJmp {
            let cond_line = infos[i].trimmed(store.get(i));
            // Parse: "jCC target" -> extract CC and target
            if let Some(space_pos) = cond_line.find(' ') {
                let cc = &cond_line[1..space_pos]; // e.g., "l", "ge", "ne"
                let cond_target = cond_line[space_pos + 1..].trim();
                // Find the next non-NOP line (should be jmp)
                let mut j = i + 1;
                while j < len && infos[j].is_nop() {
                    j += 1;
                }
                if j < len && infos[j].kind == LineKind::Jmp {
                    let jmp_line = infos[j].trimmed(store.get(j));
                    if let Some(jmp_target) = jmp_line.strip_prefix("jmp ") {
                        let jmp_target = jmp_target.trim();
                        // Find the next non-NOP/non-empty line after jmp (should be a label)
                        let mut k = j + 1;
                        while k < len && (infos[k].is_nop() || infos[k].kind == LineKind::Empty) {
                            k += 1;
                        }
                        if k < len && infos[k].kind == LineKind::Label {
                            let label_line = infos[k].trimmed(store.get(k));
                            if let Some(label_name) = label_line.strip_suffix(':') {
                                if label_name == cond_target {
                                    let inv_cc = invert_cc(cc);
                                    if inv_cc != cc {
                                        let new_line = format!("    j{} {}", inv_cc, jmp_target);
                                        replace_line(store, &mut infos[i], i, new_line);
                                        mark_nop(&mut infos[j]); // Remove the jmp
                                        changed = true;
                                        i += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern: adjacent store/load at same %rbp offset ---
        if let LineKind::StoreRbp { reg: sr, offset: so, size: ss } = infos[i].kind {
            if i + 1 < len && !infos[i + 1].is_nop() {
                if let LineKind::LoadRbp { reg: lr, offset: lo, size: ls } = infos[i + 1].kind {
                    // Different register cases are handled by global_store_forwarding
                    if so == lo && ss == ls && sr == lr && sr != REG_NONE {
                        // Same register: load is redundant
                        mark_nop(&mut infos[i + 1]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // --- Pattern: redundant zero/sign extension (including cltq) ---
        // Uses pre-classified ExtKind to avoid repeated starts_with/ends_with
        // string comparisons on every iteration.
        let mut ext_idx = i + 1;
        while ext_idx < len && ext_idx < i + 10 {
            if infos[ext_idx].is_nop() {
                ext_idx += 1;
                continue;
            }
            if matches!(infos[ext_idx].kind, LineKind::StoreRbp { .. }) {
                ext_idx += 1;
                continue;
            }
            break;
        }

        if ext_idx < len && !infos[ext_idx].is_nop() {
            let next_ext = infos[ext_idx].ext_kind;
            let prev_ext = infos[i].ext_kind;

            let is_redundant_ext = match next_ext {
                ExtKind::MovzbqAlRax => matches!(prev_ext, ExtKind::ProducerMovzbqToRax | ExtKind::MovzbqAlRax),
                ExtKind::MovzwqAxRax => matches!(prev_ext, ExtKind::ProducerMovzwqToRax | ExtKind::MovzwqAxRax),
                ExtKind::MovsbqAlRax => matches!(prev_ext, ExtKind::ProducerMovsbqToRax | ExtKind::MovsbqAlRax),
                ExtKind::MovslqEaxRax => matches!(prev_ext, ExtKind::ProducerMovslqToRax | ExtKind::MovslqEaxRax),
                ExtKind::Cltq => matches!(prev_ext,
                    ExtKind::ProducerMovslqToRax | ExtKind::ProducerMovqConstRax |
                    ExtKind::MovslqEaxRax | ExtKind::Cltq),
                ExtKind::MovlEaxEax => matches!(prev_ext,
                    ExtKind::ProducerArith32 | ExtKind::ProducerMovlToEax |
                    ExtKind::ProducerMovzbToEax | ExtKind::ProducerMovzbqToRax |
                    ExtKind::ProducerMovzwToEax | ExtKind::ProducerMovzwqToRax |
                    ExtKind::ProducerDiv32 |
                    ExtKind::MovlEaxEax),
                _ => false,
            };

            if is_redundant_ext {
                mark_nop(&mut infos[ext_idx]);
                changed = true;
                i += 1;
                continue;
            }

            // --- Extended scan: cltq past non-rax-clobbering instructions ---
            if next_ext == ExtKind::Cltq && !is_redundant_ext {
                let i_writes_rax = match infos[i].kind {
                    LineKind::Other { dest_reg } => dest_reg == 0,
                    LineKind::LoadRbp { reg, .. } => reg == 0,
                    LineKind::StoreRbp { .. } => false,
                    LineKind::Nop | LineKind::Empty => false,
                    _ => true, // conservative: barriers, calls, etc. may write rax
                };

                if !i_writes_rax && i > 0 {
                    let mut found_producer = false;
                    let scan_limit = i.saturating_sub(6);
                    let mut k = i - 1;
                    while k >= scan_limit {
                        if infos[k].is_nop() {
                            if k == 0 { break; }
                            k -= 1;
                            continue;
                        }
                        if matches!(infos[k].kind, LineKind::StoreRbp { .. }) {
                            if k == 0 { break; }
                            k -= 1;
                            continue;
                        }
                        // Stop at barriers (labels, calls, jumps, ret)
                        if infos[k].is_barrier() {
                            break;
                        }
                        // Check if this instruction is a sign-extension producer for rax
                        let k_ext = infos[k].ext_kind;
                        if matches!(k_ext,
                            ExtKind::ProducerMovslqToRax | ExtKind::ProducerMovqConstRax |
                            ExtKind::MovslqEaxRax | ExtKind::Cltq)
                        {
                            found_producer = true;
                            break;
                        }
                        // Check if this instruction writes to %rax (family 0)
                        let writes_rax = match infos[k].kind {
                            LineKind::Other { dest_reg } => dest_reg == 0,
                            LineKind::LoadRbp { reg, .. } => reg == 0,
                            _ => true, // conservative: treat unknown as writing rax
                        };
                        if writes_rax {
                            break;
                        }
                        if k == 0 { break; }
                        k -= 1;
                    }
                    if found_producer {
                        mark_nop(&mut infos[ext_idx]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── Redundant sign-extension elimination for callee-saved registers ──────────
//
// The codegen emits `movslq %REGd, %REG` after every signed i32 ALU op on
// callee-saved registers. Two patterns are eliminated:
//
// 1. Self sign-extension followed by movslq to rax:
//    movslq %r11d, %r11   ← redundant (the next movslq re-extends)
//    movslq %r11d, %rax
//
// 2. cltq followed by store: the sign-extension is unnecessary since the
//    stored value will be re-extended on the next load.

pub(super) fn eliminate_dead_sign_extensions(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        let t = infos[i].trimmed(store.get(i));

        // Pattern 1: movslq %REGd, %REG followed by movslq %REGd, %rax
        // The first movslq is dead because the second re-extends from %REGd.
        if t.starts_with("movslq %") && t.contains("d, %") {
            // Parse: "movslq %XXXd, %XXX" (self sign-extension)
            let parts: Vec<&str> = t.split(", ").collect();
            if parts.len() == 2 {
                let src = parts[0].trim_start_matches("movslq ");
                let dst = parts[1];
                // Check if it's a self sign-extension: src without 'd' suffix == dst
                let src_base = src.trim_end_matches('d');
                if src_base == dst {
                    // Look ahead for another movslq from the same 32-bit register
                    let mut j = i + 1;
                    let mut found_re_extend = false;
                    while j < len && j < i + 4 {
                        if infos[j].is_nop() { j += 1; continue; }
                        let t2 = infos[j].trimmed(store.get(j));
                        // Next instruction uses %REGd as source (re-extends)
                        if t2.starts_with("movslq ") && t2.contains(src) {
                            found_re_extend = true;
                            break;
                        }
                        // If the 64-bit form is used before re-extension, we can't remove
                        if t2.contains(dst) {
                            break;
                        }
                        j += 1;
                    }
                    if found_re_extend {
                        mark_nop(&mut infos[i]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern 2: cltq where ALL subsequent uses of %rax before its next write
        // are stores to stack (which will be re-sign-extended on load),
        // OR the next non-store consumer is a 32-bit operation that only uses %eax.
        if t == "cltq" {
            let mut j = i + 1;
            let mut can_eliminate = false;
            let mut found_store = false;
            while j < len && j < i + 8 {
                if infos[j].is_nop() { j += 1; continue; }
                if infos[j].is_barrier() { break; }
                match infos[j].kind {
                    LineKind::StoreRbp { reg: 0, .. } => {
                        found_store = true;
                        j += 1;
                        continue;
                    }
                    LineKind::Other { dest_reg: 0 } => {
                        // Something writes to %rax — check if it's a 32-bit op
                        // that only uses %eax (the cltq upper bits don't matter)
                        let tj = infos[j].trimmed(store.get(j));
                        if is_32bit_eax_consumer(tj) {
                            can_eliminate = true;
                        }
                        break;
                    }
                    _ => {
                        // Non-store, non-%rax-write — check if it reads rax
                        if infos[j].reg_refs & 1 != 0 {
                            break; // reads rax, can't eliminate
                        }
                        j += 1;
                        continue;
                    }
                }
            }
            if can_eliminate || (found_store && {
                // Original pattern: only stores found, no 64-bit consumers
                let mut safe = true;
                let mut k = j;
                while k < len && k < i + 8 {
                    if infos[k].is_nop() { k += 1; continue; }
                    if infos[k].is_barrier() { break; }
                    if infos[k].reg_refs & 1 != 0 {
                        safe = false;
                    }
                    break;
                }
                safe
            }) {
                mark_nop(&mut infos[i]);
                changed = true;
                i += 1;
                continue;
            }
        }

        // Pattern 3: movslq %REGd, %REG (self sign-extension) where the register
        // is overwritten before being read in 64-bit context. Also handles the
        // common accumulator pattern:
        //   movslq %REGd, %REG       → NOP (or kept if REG has other 64-bit readers)
        //   movq %REG, %rax          → movl %REGd, %eax (32-bit copy)
        //   addl/imull/.., %eax      → unchanged (only uses 32 bits)
        if t.starts_with("movslq %") {
            let parts: Vec<&str> = t.split(", ").collect();
            if parts.len() == 2 {
                let src = parts[0].trim_start_matches("movslq ");
                let dst = parts[1];
                // Use register family comparison to detect self sign-extension.
                // This handles both classic regs (%ebx→%rbx) and extended (%r8d→%r8).
                let src_family = super::super::types::register_family_fast(src);
                let dst_family = super::super::types::register_family_fast(dst);
                if src_family == dst_family && src_family != super::super::types::REG_NONE {
                    let reg_family = dst_family;
                    if reg_family != 0 { // rax is handled by cltq pattern
                        let reg_bit = 1u16 << reg_family;

                        // Sub-pattern 3a: next non-NOP is `movq %REG, %rax` followed
                        // by a 32-bit consumer of %eax. Replace both movslq+movq with
                        // a single `movl %REGd, %eax`.
                        let mut j = i + 1;
                        while j < len && infos[j].is_nop() { j += 1; }
                        if j < len && !infos[j].is_barrier() {
                            let reg64 = dst; // e.g., "%rbx"
                            let expected_movq = format!("movq {}, %rax", reg64);
                            let tj = infos[j].trimmed(store.get(j));
                            if tj == expected_movq {
                                // Found movq %REG, %rax. Check the next instruction
                                // is a 32-bit consumer of %eax.
                                let mut k = j + 1;
                                while k < len && infos[k].is_nop() { k += 1; }
                                let next_is_32bit = if k < len && !infos[k].is_barrier() {
                                    match infos[k].kind {
                                        LineKind::StoreRbp { reg: 0, .. } => true,
                                        LineKind::Other { dest_reg: 0 } => {
                                            let tk = infos[k].trimmed(store.get(k));
                                            is_32bit_eax_consumer(tk)
                                        }
                                        _ => false,
                                    }
                                } else { false };

                                if next_is_32bit {
                                    // Verify %REG is overwritten before next 64-bit read
                                    let mut m = j + 1;
                                    let mut reg_safe = true;
                                    while m < len && m < j + 12 {
                                        if infos[m].is_nop() { m += 1; continue; }
                                        if infos[m].is_barrier() { break; }
                                        if infos[m].reg_refs & reg_bit == 0 { m += 1; continue; }
                                        // Found reference to %REG
                                        match infos[m].kind {
                                            LineKind::LoadRbp { reg, .. } if reg == reg_family => {
                                                break; // overwritten → safe
                                            }
                                            LineKind::Other { dest_reg } if dest_reg == reg_family => {
                                                break; // overwritten → safe
                                            }
                                            _ => {
                                                reg_safe = false;
                                                break;
                                            }
                                        }
                                    }

                                    if reg_safe {
                                        // Replace: NOP the movslq, replace movq with movl
                                        mark_nop(&mut infos[i]);
                                        let new_movl = format!("    movl {}, %eax", src);
                                        replace_line(store, &mut infos[j], j, new_movl);
                                        changed = true;
                                        i += 1;
                                        continue;
                                    }
                                }
                            }
                        }

                        // Sub-pattern 3b: %REG is overwritten before any 64-bit read.
                        // Also safe if %REG is only read in 32-bit form (%REGd).
                        let reg32_suffix = src; // e.g. "%ebx"
                        let reg64_name = dst;   // e.g. "%rbx"
                        let mut j2 = i + 1;
                        let mut can_eliminate = false;
                        while j2 < len && j2 < i + 12 {
                            if infos[j2].is_nop() { j2 += 1; continue; }
                            if infos[j2].is_barrier() { break; }
                            if infos[j2].reg_refs & reg_bit == 0 { j2 += 1; continue; }
                            // This instruction references our register family.
                            match infos[j2].kind {
                                LineKind::LoadRbp { reg, .. } if reg == reg_family => {
                                    // Overwritten by load from stack → safe
                                    can_eliminate = true;
                                    break;
                                }
                                LineKind::Other { dest_reg } if dest_reg == reg_family => {
                                    // Written to by another instruction → safe
                                    can_eliminate = true;
                                    break;
                                }
                                _ => {
                                    // Check if it only reads the 32-bit form.
                                    // If the line contains %REGd but NOT %REG (64-bit),
                                    // the sign-extension upper bits don't matter.
                                    let line_text = infos[j2].trimmed(store.get(j2));
                                    if line_text.contains(reg32_suffix)
                                        && !line_text.contains(reg64_name)
                                    {
                                        // 32-bit-only read → safe, continue scanning
                                        j2 += 1;
                                        continue;
                                    }
                                    // 64-bit read → NOT safe
                                    break;
                                }
                            }
                        }
                        if can_eliminate {
                            mark_nop(&mut infos[i]);
                            changed = true;
                            i += 1;
                            continue;
                        }
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── SIB indexed addressing folding ──────────────────────────────────────────
//
// The accumulator-based codegen computes `base + index` manually:
//   movq %REG_IDX, %rax       ; copy index to accumulator
//   addq %REG_BASE, %rax      ; compute address
//   [movq %rax, %REG_TMP]     ; optional: copy to another register
//   movX SRC, (%rax|%REG_TMP) ; store through computed address
//   (or movX (%rax|%REG_TMP), DST for loads)
//
// This pass folds these into x86 SIB indexed addressing:
//   movX SRC, (%REG_BASE,%REG_IDX)
//
// Requirements: REG_IDX and REG_BASE must be callee-saved or otherwise
// guaranteed not clobbered between definition and use.

pub(super) fn fold_base_index_addressing(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Look for: movq %REG, %rax (copy a GP register to accumulator)
        let ti = infos[i].trimmed(store.get(i));
        if !ti.starts_with("movq %") || !ti.ends_with(", %rax") { i += 1; continue; }

        // Extract the source register (the index)
        let idx_reg = &ti[5..ti.len() - 6]; // strip "movq " prefix and ", %rax" suffix
        if !idx_reg.starts_with('%') || idx_reg == "%rax" || idx_reg == "%rcx" {
            i += 1; continue;
        }
        // Must be a valid GP register
        let idx_family = register_family_fast(idx_reg);
        if idx_family == REG_NONE || idx_family > REG_GP_MAX { i += 1; continue; }

        // Next non-NOP: must be addq %REG_BASE, %rax
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len || infos[j].is_barrier() { i += 1; continue; }

        let tj = infos[j].trimmed(store.get(j));
        if !tj.starts_with("addq %") || !tj.ends_with(", %rax") { i += 1; continue; }

        let base_reg = &tj[5..tj.len() - 6]; // strip "addq " and ", %rax"
        if !base_reg.starts_with('%') || base_reg == "%rax" {
            i += 1; continue;
        }
        let base_family = register_family_fast(base_reg);
        if base_family == REG_NONE || base_family > REG_GP_MAX { i += 1; continue; }

        // Next non-NOP: either a memory op using (%rax), or movq %rax, %REG_TMP
        let mut k = j + 1;
        while k < len && infos[k].is_nop() { k += 1; }
        if k >= len || infos[k].is_barrier() { i += 1; continue; }

        let tk = infos[k].trimmed(store.get(k));

        // Case 1: Direct use — the mem op uses (%rax)
        if let Some(folded) = try_fold_mem_op_with_sib(tk, "(%rax)", base_reg, idx_reg) {
            // Safety: verify rax is dead after k. The NOP'd instructions
            // leave rax without the computed address. If anything reads rax
            // after k expecting the address, the fold is unsafe.
            let rax_mask = 1u16;
            let mut rax_dead = true;
            let mut n = k + 1;
            while n < len {
                if infos[n].is_nop() { n += 1; continue; }
                if infos[n].is_barrier() { break; }
                if infos[n].reg_refs & rax_mask != 0 {
                    let dest = get_dest_reg(&infos[n]);
                    if dest == 0 { break; } // rax overwritten → dead
                    rax_dead = false;
                    break;
                }
                n += 1;
            }
            if rax_dead {
                mark_nop(&mut infos[i]); // remove movq %REG, %rax
                mark_nop(&mut infos[j]); // remove addq %REG_BASE, %rax
                replace_line(store, &mut infos[k], k, folded);
                changed = true;
                i = k + 1;
                continue;
            }
        }

        // Case 2: Intermediate copy — movq %rax, %REG_TMP, then mem op on (%REG_TMP)
        // tmp can equal base_reg (common: base loaded into %rcx, then %rcx reused
        // for the computed address). After we eliminate the addq, %base_reg still
        // holds the original base value, so SIB (%base,%idx) is correct.
        if tk.starts_with("movq %rax, %") {
            let tmp_reg = &tk[11..]; // after "movq %rax, " (includes the %)
            let tmp_family = register_family_fast(tmp_reg);
            if tmp_family != REG_NONE && tmp_family <= REG_GP_MAX
                && tmp_family != idx_family // idx must differ from tmp
            {
                let mut m = k + 1;
                while m < len && infos[m].is_nop() { m += 1; }
                if m < len && !infos[m].is_barrier() {
                    let tm = infos[m].trimmed(store.get(m));
                    let addr_pat = format!("(%{})", &tmp_reg[1..]); // e.g. "(%rcx)"
                    if let Some(folded) = try_fold_mem_op_with_sib(tm, &addr_pat, base_reg, idx_reg) {
                        // Safety: verify %TMP is dead after m. If anything reads
                        // %TMP after the folded mem op, the NOP'd movq at k means
                        // %TMP holds a stale value. Scan forward to the next
                        // barrier and check %TMP is either overwritten or not read.
                        let tmp_mask = 1u16 << tmp_family;
                        let mut tmp_dead = true;
                        let mut n = m + 1;
                        while n < len {
                            if infos[n].is_nop() { n += 1; continue; }
                            if infos[n].is_barrier() { break; }
                            if infos[n].reg_refs & tmp_mask != 0 {
                                // TMP is referenced. Check if it's a pure write (overwrite → dead).
                                let dest = get_dest_reg(&infos[n]);
                                if dest == tmp_family {
                                    break; // overwritten before read → safe
                                }
                                // TMP is read before being overwritten → not safe
                                tmp_dead = false;
                                break;
                            }
                            n += 1;
                        }

                        if tmp_dead {
                            mark_nop(&mut infos[i]); // remove movq %REG, %rax
                            mark_nop(&mut infos[j]); // remove addq
                            mark_nop(&mut infos[k]); // remove movq %rax, %TMP
                            replace_line(store, &mut infos[m], m, folded);
                            changed = true;
                            i = m + 1;
                            continue;
                        }
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

/// Try to replace a memory operand `(ADDR_PAT)` in an instruction with SIB `(%BASE,%IDX)`.
/// Returns the new instruction text if the pattern matches.
fn try_fold_mem_op_with_sib(instr: &str, addr_pat: &str, base_reg: &str, idx_reg: &str) -> Option<String> {
    // The instruction must contain the address pattern exactly once
    if !instr.contains(addr_pat) { return None; }
    // Don't fold into instructions that also reference rax/rcx in a way that conflicts
    // (the movq/addq we're removing clobber rax)
    // Build the SIB replacement
    let sib = format!("(%{}, %{})", &base_reg[1..], &idx_reg[1..]);
    let new_instr = format!("    {}", instr.replace(addr_pat, &sib));
    Some(new_instr)
}

// ── Accumulator ALU + store folding ─────────────────────────────────────────
//
// Folds the pattern:
//   movl %REGd, %eax            (copy 32-bit value to accumulator)
//   addl OFFSET(%rsp), %eax     (32-bit ALU op with memory source)
//   cltq                        (sign-extend for 64-bit store)
//   movq %rax, OFFSET2(%rsp)    (64-bit store)
//
// Into:
//   addl OFFSET(%rsp), %REGd    (ALU directly in register)
//   movl %REGd, OFFSET2(%rsp)   (32-bit store, no sign-extend needed)
//
// Saves 2 instructions per occurrence. Common in arith_loop where 32-bit
// variables are stored to 64-bit stack slots through the accumulator.

pub(super) fn fold_accumulator_alu_store(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Step 1: Look for `movl %REGd, %eax`
        let (src_reg32, src_family) = {
            let ti = infos[i].trimmed(store.get(i));
            if !ti.starts_with("movl %") || !ti.ends_with(", %eax") { i += 1; continue; }
            let sr = &ti[5..ti.len() - 6];
            if !sr.starts_with('%') || sr == "%eax" { i += 1; continue; }
            let sf = register_family_fast(sr);
            if sf == REG_NONE || sf == 0 { i += 1; continue; }
            (sr.to_string(), sf)
        };

        // Step 2: Next non-NOP must be `addl/subl/andl/orl/xorl STACK, %eax`
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len || infos[j].is_barrier() { i += 1; continue; }

        let (alu_op_s, mem_src_s) = {
            let tj = infos[j].trimmed(store.get(j));
            let ao = if let Some(pos) = tj.find(' ') { &tj[..pos] } else { i += 1; continue };
            if !matches!(ao, "addl" | "subl" | "andl" | "orl" | "xorl") { i += 1; continue; }
            if !tj.ends_with(", %eax") { i += 1; continue; }
            let ms = tj[ao.len() + 1..tj.len() - 6].trim().to_string();
            if !ms.ends_with("(%rsp)") && !ms.ends_with("(%rbp)") { i += 1; continue; }
            (ao.to_string(), ms)
        };

        // Step 3: Next must be `cltq`
        let mut k = j + 1;
        while k < len && infos[k].is_nop() { k += 1; }
        if k >= len { i += 1; continue; }
        { let tk = infos[k].trimmed(store.get(k)); if tk != "cltq" { i += 1; continue; } }

        // Step 4: Next must be `movq %rax, STACK` (store to stack)
        let mut m = k + 1;
        while m < len && infos[m].is_nop() { m += 1; }
        if m >= len || infos[m].is_barrier() { i += 1; continue; }

        let store_dst_s = {
            let is_store_rax = match infos[m].kind {
                LineKind::StoreRbp { reg: 0, size: MoveSize::Q, .. } => true,
                _ => {
                    let tm = infos[m].trimmed(store.get(m));
                    tm.starts_with("movq %rax, ") && (tm.ends_with("(%rsp)") || tm.ends_with("(%rbp)"))
                }
            };
            if !is_store_rax { i += 1; continue; }
            let tm = infos[m].trimmed(store.get(m));
            tm[11..].to_string() // after "movq %rax, "
        };

        // Step 5: Verify %SRC_REG is overwritten before its next 64-bit read
        let src_bit = 1u16 << src_family;
        let mut n = m + 1;
        let mut src_safe = true;
        while n < len && n < m + 12 {
            if infos[n].is_nop() { n += 1; continue; }
            if infos[n].is_barrier() { break; }
            if infos[n].reg_refs & src_bit == 0 { n += 1; continue; }
            match infos[n].kind {
                LineKind::LoadRbp { reg, .. } if reg == src_family => break,
                LineKind::Other { dest_reg } if dest_reg == src_family => break,
                _ => {
                    let tn = infos[n].trimmed(store.get(n));
                    if tn.contains(&*src_reg32) {
                        n += 1;
                        continue;
                    }
                    src_safe = false;
                    break;
                }
            }
        }
        if !src_safe { i += 1; continue; }

        // Transform! Replace 4 instructions with 2.
        mark_nop(&mut infos[i]);
        let new_alu = format!("    {} {}, {}", alu_op_s, mem_src_s, src_reg32);
        replace_line(store, &mut infos[j], j, new_alu);
        mark_nop(&mut infos[k]);
        let new_store = format!("    movl {}, {}", src_reg32, store_dst_s);
        replace_line(store, &mut infos[m], m, new_store);

        changed = true;
        i = m + 1;
    }
    changed
}

/// Check if an instruction is a 32-bit operation that consumes %eax,
/// meaning it only uses the lower 32 bits and then zero-extends the result.
/// Examples: addl, subl, imull, andl, orl, xorl, movl, etc.
fn is_32bit_eax_consumer(trimmed: &str) -> bool {
    // 32-bit ALU ops on %eax — these read %eax (lower 32 bits only)
    // and write back to %eax (zero-extending to 64 bits).
    if trimmed.ends_with(", %eax") || trimmed.ends_with("l %eax") {
        let op = if let Some(pos) = trimmed.find(' ') {
            &trimmed[..pos]
        } else {
            trimmed
        };
        return matches!(op,
            "addl" | "subl" | "imull" | "andl" | "orl" | "xorl"
            | "shll" | "shrl" | "sarl" | "movl" | "leal"
        );
    }
    false
}

// ── FP XMM↔GPR round-trip elimination ────────────────────────────────────────
//
// float_ops.rs emits FP binops as:
//   movq -N(%rbp), %rax    ; load lhs into GPR
//   movq %rax, %xmm0       ; shuttle to XMM ← wasteful
//   movq -M(%rbp), %rcx    ; load rhs into GPR
//   movq %rcx, %xmm1       ; shuttle to XMM ← wasteful
//   mulsd %xmm1, %xmm0     ; actual operation
//   movq %xmm0, %rax       ; result back to GPR ← wasteful
//   movq %rax, -P(%rbp)    ; store
//
// This pass eliminates the GPR intermediaries:
//   LoadRbp{rax,Q}  + "movq %rax, %xmm0"  → "movsd -N(%rbp), %xmm0"
//   LoadRbp{rcx,Q}  + "movq %rcx, %xmm1"  → "movsd -M(%rbp), %xmm1"
//   "movq %xmm0,%rax" + StoreRbp{rax,Q}   → "movsd %xmm0, -P(%rbp)"
//
// This reduces 7 instructions to 4 (then fold_fp_memory_operands reduces to 3).

pub(super) fn eliminate_fp_xmm_roundtrips(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Pattern A: LoadRbp{rax(0) or rcx(1), Q} then "movq %gpr, %xmmN"
        if let LineKind::LoadRbp { reg: load_reg, offset, size: MoveSize::Q } = infos[i].kind {
            if load_reg <= 1 {
                let mut j = i + 1;
                while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                if j < len && !infos[j].is_nop() {
                    let line_j = infos[j].trimmed(store.get(j));
                    let (expected, xmm_str) = if load_reg == 0 {
                        ("movq %rax, %xmm0", "%xmm0")
                    } else {
                        ("movq %rcx, %xmm1", "%xmm1")
                    };
                    if line_j == expected {
                        let load_text = infos[i].trimmed(store.get(i));
                        let base = if load_text.contains("(%rsp)") { "rsp" } else { "rbp" };
                        let new_text = format!("    movsd {}(%{}), {}", offset, base, xmm_str);
                        replace_line(store, &mut infos[i], i, new_text);
                        mark_nop(&mut infos[j]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern B: "movq %xmm0, %rax" then StoreRbp{rax, Q}
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));
            if line_i == "movq %xmm0, %rax" {
                let mut j = i + 1;
                while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                if j < len {
                    if let LineKind::StoreRbp { reg: 0, offset, size: MoveSize::Q } = infos[j].kind {
                        let mut k = j + 1;
                        while k < len && infos[k].is_nop() { k += 1; }
                        if !rax_is_live_at(store, infos, k, len) {
                            let store_text = infos[j].trimmed(store.get(j));
                            let base = if store_text.contains("(%rsp)") { "rsp" } else { "rbp" };
                            let new_text = format!("    movsd %xmm0, {}(%{})", offset, base);
                            mark_nop(&mut infos[i]);
                            replace_line(store, &mut infos[j], j, new_text);
                            changed = true;
                            i = j + 1;
                            continue;
                        }
                    }
                }
            }
        }

        // Pattern E: StoreRbp{rax, O} immediately followed by "movq %rax, %xmmN"
        // The stack store is dead (value used from %rax directly). NOP it so
        // Pattern D can fire on the adjacent movq-from-ptr + movq-to-xmm.
        if let LineKind::StoreRbp { reg: 0, offset, size: MoveSize::Q } = infos[i].kind {
            let mut j = i + 1;
            while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
            if j < len {
                let line_j = infos[j].trimmed(store.get(j));
                if line_j == "movq %rax, %xmm0" || line_j == "movq %rax, %xmm1" {
                    // Verify stack slot O is not read between j+1 and block end.
                    if rbp_offset_dead_after(store, infos, j + 1, len, offset as i64) {
                        mark_nop(&mut infos[i]);
                        changed = true;
                        // Don't advance i past j; let Pattern D fire next iteration.
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern D: "movq (%<ptr>), %rax" immediately followed by
        // "movq %rax, %xmmN" → "movsd (%<ptr>), %xmmN".
        // Fires after Pattern E removes the intervening dead StoreRbp.
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));
            if line_i.starts_with("movq (%") && line_i.ends_with("), %rax") {
                let mut j = i + 1;
                while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                if j < len {
                    let line_j = infos[j].trimmed(store.get(j));
                    let xmm = if line_j == "movq %rax, %xmm0" {
                        Some("%xmm0")
                    } else if line_j == "movq %rax, %xmm1" {
                        Some("%xmm1")
                    } else {
                        None
                    };
                    if let Some(xmm_str) = xmm {
                        // Extract pointer register from "movq (%<ptr>), %rax"
                        let ptr_reg = &line_i[7..line_i.len() - 7]; // strip "movq (%" and "), %rax"
                        let mut k = j + 1;
                        while k < len && infos[k].is_nop() { k += 1; }
                        if !rax_is_live_at(store, infos, k, len) {
                            let new_text = format!("    movsd (%{}), {}", ptr_reg, xmm_str);
                            replace_line(store, &mut infos[i], i, new_text);
                            mark_nop(&mut infos[j]);
                            changed = true;
                            i = j + 1;
                            continue;
                        }
                    }
                }
            }
        }

        // Pattern F: "movq %xmm0, %rax" + "movq %rax, %<gprA>" +
        //            "movq %<gprB>, %rcx" + "movq %<gprA>, (%rcx)"
        //          → "movsd %xmm0, (%<gprB>)" + NOP the rest.
        // This folds the 4-instruction store-through-pointer chain.
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));
            if line_i == "movq %xmm0, %rax" {
                // Find J: "movq %rax, %<gprA>"
                let mut j = i + 1;
                while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                if j < len {
                    let line_j = infos[j].trimmed(store.get(j));
                    if line_j.starts_with("movq %rax, %") && !line_j.ends_with("%xmm0") {
                        let gpr_a = &line_j[12..]; // "movq %rax, %" is 12 chars
                        // Find K: "movq %<gprB>, %rcx"
                        let mut k = j + 1;
                        while k < len && k < j + 4 && infos[k].is_nop() { k += 1; }
                        if k < len {
                            let line_k = infos[k].trimmed(store.get(k));
                            if line_k.starts_with("movq %") && line_k.ends_with(", %rcx") {
                                let gpr_b = &line_k[6..line_k.len() - 6]; // strip "movq %" and ", %rcx"
                                // Find L: "movq %<gprA>, (%rcx)"
                                let mut l = k + 1;
                                while l < len && l < k + 4 && infos[l].is_nop() { l += 1; }
                                if l < len {
                                    let expected_l = format!("movq %{}, (%rcx)", gpr_a);
                                    let line_l = infos[l].trimmed(store.get(l));
                                    if line_l == expected_l {
                                        // Check rax not live after J, gprA not live after L.
                                        let mut after_j = j + 1;
                                        while after_j < len && infos[after_j].is_nop() { after_j += 1; }
                                        if !rax_is_live_at(store, infos, after_j, len) {
                                            let new_text = format!("    movsd %xmm0, (%{})", gpr_b);
                                            replace_line(store, &mut infos[i], i, new_text);
                                            mark_nop(&mut infos[j]);
                                            mark_nop(&mut infos[k]);
                                            mark_nop(&mut infos[l]);
                                            changed = true;
                                            i = l + 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── Pointer-deref stack elimination (Pattern H) ─────────────────────────────
//
// Matches the common codegen idiom:
//   movq (%<ptr>), %rax          [I] load through pointer into GPR
//   movq %rax, -O(%rbp)          [J] spill to stack slot
//   ... (gap, no write to O(%rbp) or %<ptr>) ...
//   movsd/mulsd/addsd -O(%rbp)   [K] FP use of the spilled value
//
// Folds to:  NOP I, NOP J,  replace O(%rbp) in K with (%<ptr>).
// This eliminates the GPR round-trip and stack spill.
//
pub(super) fn fold_ptr_deref_through_stack(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match I: "movq (%<ptr>), %rax"
        if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));
            if line_i.starts_with("movq (%") && line_i.ends_with("), %rax") {
                let ptr_reg = &line_i[7..line_i.len() - 7]; // between "movq (%" and "), %rax"

                // Match J: immediately adjacent StoreRbp{0, O, Q}
                let mut j = i + 1;
                while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                if j >= len { i += 1; continue; }
                if let LineKind::StoreRbp { reg: 0, offset, size: MoveSize::Q } = infos[j].kind {
                    let store_text = infos[j].trimmed(store.get(j));
                    let base = if store_text.contains("(%rsp)") { "rsp" } else { "rbp" };
                    let offset_str = format!("{}(%{})", offset, base);
                    let ptr_mem = format!("(%{})", ptr_reg);

                    // Scan forward from j+1 for first FP use of O(%rbp).
                    let mut k = j + 1;
                    let mut ptr_modified = false;
                    let mut rax_overwritten = false;
                    let mut count = 0;
                    let mut found = false;

                    while k < len && count < 20 {
                        if infos[k].is_nop() { k += 1; continue; }
                        let t = infos[k].trimmed(store.get(k));

                        // Stop at control flow.
                        if t.starts_with('j') || t.starts_with("call") || t.starts_with("ret") {
                            break;
                        }

                        // Check if ptr register is modified (appears as destination).
                        let ptr_with_pct = format!("%{}", ptr_reg);
                        if t.ends_with(&format!(", {}", ptr_with_pct)) {
                            ptr_modified = true;
                        }

                        // Check if %rax is read before overwritten (safety for NOP'ing I).
                        if !rax_overwritten {
                            match infos[k].kind {
                                LineKind::LoadRbp { reg: 0, .. } |
                                LineKind::Other { dest_reg: 0 } => {
                                    rax_overwritten = true;
                                }
                                _ => {
                                    if t.contains("%rax") || t.contains("%eax") || t.contains("%al") {
                                        // %rax is read before being overwritten → unsafe.
                                        break;
                                    }
                                }
                            }
                        }

                        // Check for StoreRbp writing the same offset → our store is overwritten.
                        if let LineKind::StoreRbp { offset: o, .. } = infos[k].kind {
                            if o == offset { break; }
                        }

                        // Found an FP instruction using O(%rbp) as source operand?
                        if t.contains(&offset_str) && !ptr_modified {
                            // Verify it's a source (not dest). StoreRbp is already caught above.
                            // For movsd/mulsd/addsd/subsd/divsd: the memory operand is the source
                            // if it comes before the comma.
                            if (t.starts_with("movsd ") || t.starts_with("mulsd ") ||
                                t.starts_with("addsd ") || t.starts_with("subsd ") ||
                                t.starts_with("divsd ")) &&
                                t.contains(&offset_str)
                            {
                                // Check O(%rbp) not read again after K.
                                if rbp_offset_dead_after(store, infos, k + 1, len, i64::from(offset)) {
                                    let new_text = format!("    {}", t.replace(&offset_str, &ptr_mem));
                                    mark_nop(&mut infos[i]);
                                    mark_nop(&mut infos[j]);
                                    replace_line(store, &mut infos[k], k, new_text);
                                    changed = true;
                                    found = true;
                                }
                                break;
                            }
                            break; // Unknown instruction using the offset → bail.
                        }

                        k += 1;
                        count += 1;
                    }

                    if found {
                        i = k + 1;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── FP spill elimination around load (Pattern I) ─────────────────────────────
//
// Matches:
//   movsd %xmm0, O(%rbp)       [I] spill product to stack
//   ... (gap: address calc, no xmm usage) ...
//   movsd (%ptr), %xmm0        [K] load C (overwrites xmm0)
//   addsd O(%rbp), %xmm0       [L] C += spilled product
//
// Rewrites to:
//   (NOP)                       [I] eliminated
//   ... (gap) ...
//   movsd (%ptr), %xmm1        [K] load C into xmm1
//   addsd %xmm1, %xmm0         [L] product + C (register-only)
//
// This avoids the stack spill+reload by keeping the product in xmm0 and
// routing the C load through xmm1 instead.
//
pub(super) fn eliminate_fp_spill_around_load(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }
        let line_i = infos[i].trimmed(store.get(i));

        // Match I: "movsd %xmm0, <offset>(%rbp)"
        if line_i.starts_with("movsd %xmm0, ") && (line_i.ends_with("(%rbp)") || line_i.ends_with("(%rsp)")) {
            let mem_operand = &line_i[13..]; // after "movsd %xmm0, " (13 chars)
            // Extract numeric offset from e.g. "-48(%rbp)"
            let paren_pos = mem_operand.find('(');
            if let Some(pp) = paren_pos {
                let offset_num: Result<i64, _> = mem_operand[..pp].parse();
                if let Ok(offset) = offset_num {
                    // Scan forward for K (load into xmm0) then L (addsd from same slot).
                    let mut k_pos = 0usize;
                    let mut l_pos = 0usize;
                    let mut xmm1_clear = true;
                    let mut j = i + 1;
                    let mut count = 0;

                    while j < len && count < 16 {
                        if infos[j].is_nop() { j += 1; continue; }
                        let t = infos[j].trimmed(store.get(j));
                        if t.starts_with('j') || t.starts_with("call") || t.starts_with("ret") { break; }
                        if t.contains("%xmm1") { xmm1_clear = false; break; }

                        // Find K: "movsd <something>, %xmm0" (load overwriting xmm0)
                        if k_pos == 0 && t.starts_with("movsd ") && t.ends_with(", %xmm0") {
                            k_pos = j;
                        }

                        // Find L: "addsd <offset>(%rbp), %xmm0" (reload from spill slot)
                        if k_pos > 0 {
                            let expected_l = format!("addsd {}, %xmm0", mem_operand);
                            if t == expected_l {
                                l_pos = j;
                                break;
                            }
                        }

                        j += 1;
                        count += 1;
                    }

                    if k_pos > 0 && l_pos > 0 && xmm1_clear {
                        // Verify the spill slot is dead after L.
                        if rbp_offset_dead_after(store, infos, l_pos + 1, len, offset) {
                            // Also verify xmm1 is dead after L.
                            let mut after_l = l_pos + 1;
                            while after_l < len && infos[after_l].is_nop() { after_l += 1; }
                            let xmm1_dead_after = if after_l >= len { true }
                            else {
                                let t = infos[after_l].trimmed(store.get(after_l));
                                !t.contains("%xmm1")
                            };

                            if xmm1_dead_after {
                                // NOP I (the spill store).
                                mark_nop(&mut infos[i]);
                                // Change K: "movsd ..., %xmm0" → "movsd ..., %xmm1"
                                let line_k = infos[k_pos].trimmed(store.get(k_pos));
                                let new_k = format!("    {}", line_k.replace(", %xmm0", ", %xmm1"));
                                replace_line(store, &mut infos[k_pos], k_pos, new_k);
                                // Change L: "addsd O(%rbp), %xmm0" → "addsd %xmm1, %xmm0"
                                let new_l = "    addsd %xmm1, %xmm0".to_string();
                                replace_line(store, &mut infos[l_pos], l_pos, new_l);
                                changed = true;
                                i = l_pos + 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── Loop-invariant FP stack promotion ────────────────────────────────────────
//
// Promotes a loop-invariant `movsd -O(%rbp), %xmm0` to a register:
//   Preheader: movq %rax, -O(%rbp) → movq %rax, %xmm2
//   Loop body: movsd -O(%rbp), %xmm0 → movapd %xmm2, %xmm0
//
// Conditions:
//   - The inner loop has a back-edge jmp to a label above
//   - -O(%rbp) is not written inside the loop body
//   - xmm2 is not used inside the loop body
//   - The preheader (block before loop header) stores to -O(%rbp)
//
pub(super) fn promote_loop_invariant_fp_load(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;

    // Find loop back-edges: jmp to a label that appears before the jmp.
    let mut i = 0;
    while i < len {
        if infos[i].is_nop() { i += 1; continue; }
        if infos[i].kind != LineKind::Jmp { i += 1; continue; }
        let jmp_text = infos[i].trimmed(store.get(i));
        if !jmp_text.starts_with("jmp ") { i += 1; continue; }
        let target = &jmp_text[4..];
        let target_label = format!("{}:", target);

        // Find the target label (must be before the jmp = back-edge).
        let mut header_pos = None;
        for lbl in 0..i {
            if infos[lbl].kind == LineKind::Label {
                if infos[lbl].trimmed(store.get(lbl)) == target_label {
                    header_pos = Some(lbl);
                    break;
                }
            }
        }
        let header = match header_pos {
            Some(h) => h,
            None => { i += 1; continue; }
        };

        // Loop body is [header..=i]. Find movsd -O(%rbp), %xmm0 in body.
        let mut body_start = header;
        // Skip past header's condition check to the body label.
        for pos in header + 1..i {
            if infos[pos].kind == LineKind::Label {
                body_start = pos;
                break;
            }
        }

        // Scan body for "movsd -O(%rbp), %xmm0" candidates.
        for pos in body_start + 1..i {
            if infos[pos].is_nop() { continue; }
            let t = infos[pos].trimmed(store.get(pos));
            if !t.starts_with("movsd ") || !t.ends_with(", %xmm0") { continue; }
            // Check it's a stack load: "movsd -N(%rbp), %xmm0" or "movsd N(%rsp), %xmm0"
            let src = &t[6..t.len() - 7]; // between "movsd " and ", %xmm0"
            if !src.ends_with("(%rbp)") && !src.ends_with("(%rsp)") { continue; }
            let offset_str = src.to_string();

            // Check -O(%rbp) is NOT written in the loop body [body_start..=i].
            // Extract numeric offset from the source string (e.g., "-24(%rbp)" → -24)
            let numeric_offset_end = offset_str.find('(').unwrap_or(offset_str.len());
            let numeric_offset: i32 = offset_str[..numeric_offset_end].parse().unwrap_or(0);
            let mut written_in_body = false;
            for chk in body_start + 1..i {
                if infos[chk].is_nop() { continue; }
                if let LineKind::StoreRbp { offset: o, .. } = infos[chk].kind {
                    if o == numeric_offset { written_in_body = true; break; }
                }
                // Also check text for movsd stores to this offset.
                let ct = infos[chk].trimmed(store.get(chk));
                if ct.ends_with(&offset_str) && ct.starts_with("movsd ") {
                    written_in_body = true; break;
                }
            }
            if written_in_body { continue; }

            // Check xmm2 is not used anywhere in [header..=i].
            let mut xmm2_used = false;
            for chk in header..=i {
                if infos[chk].is_nop() { continue; }
                let ct = infos[chk].trimmed(store.get(chk));
                if ct.contains("%xmm2") { xmm2_used = true; break; }
            }
            if xmm2_used { continue; }

            // Find preheader store: "movq %rax, -O(%rbp)" before the header.
            // Scan backward, crossing labels but stopping at function boundaries (.size).
            let mut preheader_store = None;
            let mut ph = if header > 0 { header - 1 } else { 0 };
            let mut ph_count = 0;
            while ph_count < 60 {
                if infos[ph].is_nop() {
                    if ph == 0 { break; }
                    ph -= 1;
                    continue;
                }
                // Stop at function boundaries, not labels
                if infos[ph].kind == LineKind::Directive {
                    let dt = infos[ph].trimmed(store.get(ph));
                    if dt.starts_with(".size ") || dt.starts_with(".globl ") || dt.starts_with(".type ") {
                        break;
                    }
                }
                // Labels are OK to cross — we're looking for ANY store to this offset
                // that dominates the loop header.
                if let LineKind::StoreRbp { reg: 0, offset: o, .. } = infos[ph].kind {
                    if o == numeric_offset {
                        preheader_store = Some(ph);
                        break;
                    }
                }
                if ph == 0 { break; }
                ph -= 1;
                ph_count += 1;
            }

            if let Some(ph_pos) = preheader_store {
                // Replace preheader store: "movq %rax, -O(%rbp)" → "movq %rax, %xmm2"
                let new_ph = "    movq %rax, %xmm2".to_string();
                replace_line(store, &mut infos[ph_pos], ph_pos, new_ph);
                // Replace body load: "movsd -O(%rbp), %xmm0" → "movapd %xmm2, %xmm0"
                let new_body = "    movapd %xmm2, %xmm0".to_string();
                replace_line(store, &mut infos[pos], pos, new_body);
                changed = true;
                break; // Only promote one per loop for now.
            }
        }

        i += 1;
    }
    changed
}

// ── Copy + operation fusion ──────────────────────────────────────────────────
//
// Fuses a register copy with the following operation that reads and writes
// the destination, when the copy is the sole producer:
//
//   movq %A, %B  +  leaq disp(%B), %B   →  leaq disp(%A), %B    (copy+lea)
//   movq %A, %B  +  addq $imm, %B        →  leaq imm(%A), %B     (copy+add)
//   movq %A, %B  +  shlq $N, %B          →  leaq (,%A,2^N), %B   (copy+shl, N=1..3)
//
// The addq→leaq rewrite drops flags. We verify the next non-NOP instruction
// sets its own flags (doesn't consume ours) before applying.
//
pub(super) fn fuse_copy_and_operation(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: "movq %<src>, %<dst>" where src != dst, both are GP regs.
        if let LineKind::Other { dest_reg } = infos[i].kind {
            if dest_reg > 15 { i += 1; continue; }
            let line_i = infos[i].trimmed(store.get(i));
            if !line_i.starts_with("movq %") { i += 1; continue; }

            // Parse "movq %<src>, %<dst>"
            if let Some(comma) = line_i.find(", %") {
                let src_reg = &line_i[6..comma]; // after "movq %"
                let dst_reg_str = &line_i[comma + 3..]; // after ", %"
                if src_reg == dst_reg_str || src_reg.contains('(') { i += 1; continue; }

                let mut j = i + 1;
                while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                if j >= len { i += 1; continue; }
                let line_j = infos[j].trimmed(store.get(j));

                // Sub-pattern: leaq disp(%<dst>), %<dst> → leaq disp(%<src>), %<dst>
                let lea_prefix = format!("leaq ");
                let lea_base = format!("(%{}), %{}", dst_reg_str, dst_reg_str);
                if line_j.starts_with(&lea_prefix) && line_j.ends_with(&format!("), %{}", dst_reg_str)) &&
                   line_j.contains(&format!("(%{})", dst_reg_str))
                {
                    let new_text = format!("    {}", line_j.replace(
                        &format!("(%{})", dst_reg_str),
                        &format!("(%{})", src_reg),
                    ));
                    mark_nop(&mut infos[i]);
                    replace_line(store, &mut infos[j], j, new_text);
                    changed = true;
                    i = j + 1;
                    continue;
                }

                // Sub-pattern: addq $imm, %<dst> → leaq imm(%<src>), %<dst>
                // Only if flags are not consumed by the next instruction.
                let add_suffix = format!(", %{}", dst_reg_str);
                if line_j.starts_with("addq $") && line_j.ends_with(&add_suffix) {
                    let imm_str = &line_j[6..line_j.len() - add_suffix.len()]; // between "addq $" and ", %dst"
                    // Check flags safety: next instruction must not read flags.
                    let mut k = j + 1;
                    while k < len && infos[k].is_nop() { k += 1; }
                    let flags_safe = if k >= len { true } else {
                        let next = infos[k].trimmed(store.get(k));
                        // Instructions that consume flags: jCC, setCC, adcq, sbbq, cmovc, etc.
                        // Conservative: if next instruction starts with any of these, bail.
                        !(next.starts_with("ja") || next.starts_with("jb") ||
                          next.starts_with("je") || next.starts_with("jn") ||
                          next.starts_with("jg") || next.starts_with("jl") ||
                          next.starts_with("js") || next.starts_with("jo") ||
                          next.starts_with("jc") || next.starts_with("jp") ||
                          next.starts_with("set") || next.starts_with("adc") ||
                          next.starts_with("sbb") || next.starts_with("cmov"))
                    };
                    if flags_safe {
                        let new_text = format!("    leaq {}(%{}), %{}", imm_str, src_reg, dst_reg_str);
                        mark_nop(&mut infos[i]);
                        replace_line(store, &mut infos[j], j, new_text);
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

/// Returns true if %rax is live (potentially read) at instruction index `at`.
/// Conservative: treats "movq %<non-rax>, %rax" as a pure write (not live).
fn rax_is_live_at(store: &LineStore, infos: &[LineInfo], at: usize, len: usize) -> bool {
    if at >= len {
        return false;
    }
    match infos[at].kind {
        // LoadRbp{reg:0} is "movq -N(%rbp), %rax" — unconditional write.
        LineKind::LoadRbp { reg: 0, .. } => false,
        // Other{dest_reg:0}: %rax is the destination. If text is "movq <src>, %rax"
        // and <src> doesn't mention %rax, it's a pure write → not live.
        LineKind::Other { dest_reg: 0 } => {
            let t = infos[at].trimmed(store.get(at));
            if t.starts_with("movq ") && t.ends_with(", %rax") {
                // Check if source (between "movq " and ", %rax") contains %rax.
                let src = &t[5..t.len() - 6];
                src.contains("%rax")
            } else {
                t.contains("%rax")
            }
        }
        _ => infos[at].trimmed(store.get(at)).contains("%rax"),
    }
}

/// Returns true if the rbp offset `offset` is dead (not read before being
/// written or before a control-flow boundary) starting at instruction `start`.
/// Scans up to 32 instructions forward; stops at any jump/call.
fn rbp_offset_dead_after(
    store: &LineStore,
    infos: &[LineInfo],
    start: usize,
    len: usize,
    offset: i64,
) -> bool {
    let offset_str_rbp = format!("{}(%rbp)", offset);
    let offset_str_rsp = format!("{}(%rsp)", offset);
    let mut i = start;
    let mut count = 0;
    while i < len && count < 32 {
        if infos[i].is_nop() { i += 1; continue; }
        let t = infos[i].trimmed(store.get(i));
        // Stop at control-flow instructions.
        if t.starts_with('j') || t.starts_with("call") || t.starts_with("ret") {
            return true; // Reached block boundary without a read → dead.
        }
        // If this is a write to the same offset (StoreRbp), it's dead from here.
        if let LineKind::StoreRbp { offset: o, .. } = infos[i].kind {
            if i64::from(o) == offset { return true; }
        }
        // If text contains the offset string (potential read), not dead.
        if t.contains(&offset_str_rbp) || t.contains(&offset_str_rsp) {
            return false;
        }
        i += 1;
        count += 1;
    }
    true // Ran out of instructions without a read → dead.
}

// ── rcx address-register copy elimination (Pattern G) ────────────────────────
//
// LCCC's codegen always copies the pointer into %rcx before a memory op:
//   movq %<ptr>, %rcx
//   movq (%rcx), %rax   OR   movsd (%rcx), %xmmN
//
// When %rcx is dead after the dereference we can fold to:
//   movq (%<ptr>), %rax   OR   movsd (%<ptr>), %xmmN
//
pub(super) fn eliminate_rcx_address_copy(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: "movq %<gpr>, %rcx" (any GPR → rcx, rcx = family 1)
        if let LineKind::Other { dest_reg: 1 } = infos[i].kind {
            let line_i = infos[i].trimmed(store.get(i));
            if line_i.starts_with("movq %") && line_i.ends_with(", %rcx") {
                let src_reg = &line_i[6..line_i.len() - 6]; // between "movq %" and ", %rcx"
                // src_reg must not be "rcx" itself (no-op move) and must be a plain GPR
                if src_reg != "rcx" && !src_reg.contains('(') && !src_reg.contains('$') {
                    let mut j = i + 1;
                    while j < len && j < i + 4 && infos[j].is_nop() { j += 1; }
                    if j < len {
                        let line_j = infos[j].trimmed(store.get(j));

                        // Sub-pattern G1: movq (%rcx), %rax → movq (%<src>), %rax
                        if line_j == "movq (%rcx), %rax" {
                            let mut k = j + 1;
                            while k < len && infos[k].is_nop() { k += 1; }
                            if !rcx_is_live_at(store, infos, k, len) {
                                let new = format!("    movq (%{}), %rax", src_reg);
                                mark_nop(&mut infos[i]);
                                replace_line(store, &mut infos[j], j, new);
                                changed = true;
                                i = j + 1;
                                continue;
                            }
                        }

                        // Sub-pattern G2: movsd (%rcx), %xmmN → movsd (%<src>), %xmmN
                        if line_j.starts_with("movsd (%rcx), %xmm") {
                            let xmm_dest = &line_j[14..]; // after "movsd (%rcx), " → "%xmmN"
                            let mut k = j + 1;
                            while k < len && infos[k].is_nop() { k += 1; }
                            if !rcx_is_live_at(store, infos, k, len) {
                                let new = format!("    movsd (%{}), {}", src_reg, xmm_dest);
                                mark_nop(&mut infos[i]);
                                replace_line(store, &mut infos[j], j, new);
                                changed = true;
                                i = j + 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

fn rcx_is_live_at(store: &LineStore, infos: &[LineInfo], at: usize, len: usize) -> bool {
    if at >= len { return false; }
    match infos[at].kind {
        // LoadRbp loads into a GP reg — doesn't use %rcx as address
        LineKind::LoadRbp { .. } => false,
        LineKind::Other { dest_reg: 1 } => {
            // rcx is the destination. "movq <src>, %rcx" is a pure write if src ≠ %rcx.
            let t = infos[at].trimmed(store.get(at));
            if t.starts_with("movq ") && t.ends_with(", %rcx") {
                let src = &t[5..t.len() - 6];
                src.contains("%rcx")
            } else {
                t.contains("%rcx")
            }
        }
        _ => infos[at].trimmed(store.get(at)).contains("%rcx"),
    }
}

// ── Movq + extension/truncation fusion ───────────────────────────────────────
//
// Fuses `movq %REG, %rax` followed by a cast instruction into a single
// instruction. The two-instruction pattern arises from the accumulator-based
// codegen model: emit_load_operand loads a 64-bit value into %rax, then
// emit_cast_instrs emits an extension/truncation on %rax/%eax/%ax/%al.
//
// Fused patterns (all require REG != rax, no intervening non-NOP instructions):
//   movq %REG, %rax + movl %eax, %eax   -> movl %REGd, %eax    (truncate to u32)
//   movq %REG, %rax + movslq %eax, %rax -> movslq %REGd, %rax  (sign-extend i32->i64)
//   movq %REG, %rax + cltq              -> movslq %REGd, %rax   (sign-extend i32->i64)
//   movq %REG, %rax + movzbq %al, %rax  -> movzbl %REGb, %eax  (zero-extend u8->i64)
//   movq %REG, %rax + movzwq %ax, %rax  -> movzwl %REGw, %eax  (zero-extend u16->i64)
//   movq %REG, %rax + movsbq %al, %rax  -> movsbq %REGb, %rax  (sign-extend i8->i64)

pub(super) fn fuse_movq_ext_truncation(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i + 1 < len {
        // Look for ProducerMovqRegToRax
        if infos[i].ext_kind != ExtKind::ProducerMovqRegToRax {
            i += 1;
            continue;
        }

        // Find next non-NOP instruction (skip only NOPs, not stores)
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        // Check if next instruction is a fusable extension/truncation on %rax
        let next_ext = infos[j].ext_kind;
        let fusable = matches!(next_ext,
            ExtKind::MovlEaxEax | ExtKind::MovslqEaxRax | ExtKind::Cltq |
            ExtKind::MovzbqAlRax | ExtKind::MovzwqAxRax |
            ExtKind::MovsbqAlRax);
        if !fusable {
            i += 1;
            continue;
        }

        // Extract source register family from the movq instruction
        let movq_line = infos[i].trimmed(store.get(i));
        let src_family = if let Some(rest) = movq_line.strip_prefix("movq ") {
            if let Some((src, _dst)) = rest.split_once(',') {
                let src = src.trim();
                let fam = register_family_fast(src);
                if fam != REG_NONE && fam != 0 { fam } else { REG_NONE }
            } else { REG_NONE }
        } else { REG_NONE };

        if src_family == REG_NONE {
            i += 1;
            continue;
        }

        // Build the fused instruction based on the extension type
        let new_text = match next_ext {
            ExtKind::MovlEaxEax => {
                let src_32 = REG_NAMES[1][src_family as usize];
                format!("    movl {}, %eax", src_32)
            }
            ExtKind::MovslqEaxRax | ExtKind::Cltq => {
                let src_32 = REG_NAMES[1][src_family as usize];
                format!("    movslq {}, %rax", src_32)
            }
            ExtKind::MovzbqAlRax => {
                let src_8 = REG_NAMES[3][src_family as usize];
                format!("    movzbl {}, %eax", src_8)
            }
            ExtKind::MovzwqAxRax => {
                let src_16 = REG_NAMES[2][src_family as usize];
                format!("    movzwl {}, %eax", src_16)
            }
            ExtKind::MovsbqAlRax => {
                let src_8 = REG_NAMES[3][src_family as usize];
                format!("    movsbq {}, %rax", src_8)
            }
            _ => unreachable!("mov+ext fusion matched unexpected ExtKind"),
        };

        replace_line(store, &mut infos[i], i, new_text);
        mark_nop(&mut infos[j]);
        changed = true;
        i = j + 1;
        continue;
    }
    changed
}

// ── Sign-extend + move fusion ────────────────────────────────────────────────
//
// Fuses a sign-extend to %rax followed by a move from %rax to another register,
// when %rax is not needed afterward (or only used for a 32-bit compare that can
// be redirected to the original source register).
//
// Pattern A (rax dead):
//   movslq %Xd, %rax       →  movslq %Xd, %Y
//   movq %rax, %Y
//
// Pattern B (rax used only in cmpl):
//   movslq %Xd, %rax       →  movslq %Xd, %Y
//   movq %rax, %Y              cmpl $imm, %Xd
//   cmpl $imm, %eax
//
// Also handles cltq (= movslq %eax, %rax) as the sign-extend source.

pub(super) fn fuse_signext_and_move(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: movslq %Xd, %rax (or cltq)
        let ti = infos[i].trimmed(store.get(i));
        let src_family = if ti.starts_with("movslq %") && ti.ends_with(", %rax") {
            let src_32 = &ti[7..ti.len() - 6]; // between "movslq " and ", %rax"
            if !src_32.starts_with('%') { i += 1; continue; }
            let fam = register_family_fast(src_32);
            if fam == REG_NONE || fam == 0 { i += 1; continue; }
            fam
        } else if ti == "cltq" {
            0u8 // cltq sign-extends %eax → %rax, src is rax family itself
        } else {
            i += 1; continue;
        };

        // For cltq, we can't retarget (src == dest == rax family), skip
        if src_family == 0 { i += 1; continue; }

        // Next non-NOP: must be movq %rax, %Y
        let j = next_non_nop(infos, i + 1, len);
        if j >= len || infos[j].is_barrier() { i += 1; continue; }

        let tj = infos[j].trimmed(store.get(j));
        if !tj.starts_with("movq %rax, %") { i += 1; continue; }
        let dest_reg = &tj[11..]; // after "movq %rax, " (includes %)
        let dest_family = register_family_fast(dest_reg);
        if dest_family == REG_NONE || dest_family == 0 || dest_family == src_family {
            i += 1; continue;
        }

        // Check if %rax is dead after j, or only used in a redirectable cmpl.
        // Scan forward from j+1 until the next basic block boundary (barrier).
        // No line limit — large functions like SQLite's 10K-line VDBE interpreter
        // can have long stretches of non-barrier instructions.
        let src_32_name = REG_NAMES[1][src_family as usize];
        let mut rax_dead = false; // conservative default: assume alive
        let mut cmpl_line: Option<usize> = None;
        let mut n = j + 1;
        while n < len {
            if infos[n].is_nop() { n += 1; continue; }
            if infos[n].is_barrier() {
                // At ANY barrier, conservatively assume rax is alive.
                // Even jmp/call could have successors that read rax
                // (e.g., fall-through after conditional, or rax used
                // by a different predecessor to the jmp target).
                // Only explicit overwrite (LoadRbp/Other with dest=0)
                // proves rax is dead.
                break;
            }
            // Check if this instruction references rax
            if infos[n].reg_refs & 1 != 0 {
                // rax is referenced. Check if it's a PURE write (rax overwritten → dead).
                // Instructions that both read and write rax (like movl %eax, %eax
                // for zero-extension) are NOT pure writes — they depend on the
                // current rax value.
                match infos[n].kind {
                    LineKind::LoadRbp { reg: 0, .. } => { rax_dead = true; break; }
                    LineKind::Other { dest_reg: 0 } => {
                        // Verify it's a pure write, not a read-modify-write.
                        let tn = infos[n].trimmed(store.get(n));
                        if !is_read_modify_write(tn) {
                            // Also check: source operand must not reference any
                            // rax-family register (eax, ax, al, rax).
                            let is_rax_in_src = if let Some(comma) = tn.rfind(',') {
                                let src = &tn[..comma];
                                src.contains("%rax") || src.contains("%eax")
                                    || src.contains("%ax") || src.contains("%al")
                            } else {
                                true // single-operand → assumes reads rax
                            };
                            if !is_rax_in_src {
                                rax_dead = true;
                                break;
                            }
                        }
                        // Falls through: rax is read, not dead
                    }
                    _ => {}
                }
                // Check if it's a cmpl $imm, %eax
                let tn = infos[n].trimmed(store.get(n));
                if tn.starts_with("cmpl $") && tn.ends_with(", %eax") && cmpl_line.is_none() {
                    // Check src_family is not modified between i and n.
                    // Must check ALL instruction kinds that can write to a register,
                    // including calls (clobber caller-saved) and pop.
                    let src_bit = 1u16 << src_family;
                    let mut src_modified = false;
                    for chk in (i + 1)..n {
                        if infos[chk].is_nop() { continue; }
                        // Calls clobber all caller-saved registers. If src is
                        // caller-saved (rdi=6, rsi=7, r8-r11=8-11), it's modified.
                        if infos[chk].kind == LineKind::Call {
                            if src_family >= 6 { // caller-saved families
                                src_modified = true; break;
                            }
                        }
                        // Any barrier could modify the register
                        if infos[chk].is_barrier() && infos[chk].kind == LineKind::Call {
                            // already handled above
                        }
                        if infos[chk].reg_refs & src_bit == 0 { continue; }
                        match infos[chk].kind {
                            LineKind::Other { dest_reg: d } if d == src_family => { src_modified = true; break; }
                            LineKind::LoadRbp { reg: r, .. } if r == src_family => { src_modified = true; break; }
                            LineKind::Pop { reg: r } if r == src_family => { src_modified = true; break; }
                            LineKind::StoreRbp { reg: r, .. } if r == src_family => {} // store reads, doesn't modify
                            _ => {} // read-only reference is fine
                        }
                    }
                    if !src_modified {
                        cmpl_line = Some(n);
                        n += 1;
                        continue;
                    }
                }
                // rax is used in a non-redirectable way → not dead
                rax_dead = false;
                break;
            }
            n += 1;
        }

        if !rax_dead && cmpl_line.is_none() { i += 1; continue; }

        // Apply the transformation
        let dest_reg_stripped = &dest_reg[1..]; // without leading %
        let new_signext = format!("    movslq {}, %{}", src_32_name, dest_reg_stripped);
        replace_line(store, &mut infos[i], i, new_signext);
        mark_nop(&mut infos[j]); // remove movq %rax, %Y

        // If there's a cmpl to redirect, do it
        if let Some(cmp_idx) = cmpl_line {
            let tc = infos[cmp_idx].trimmed(store.get(cmp_idx));
            let new_cmp = format!("    {}", tc.replace(", %eax", &format!(", {}", src_32_name)));
            replace_line(store, &mut infos[cmp_idx], cmp_idx, new_cmp);
        }

        changed = true;
        i = j + 1;
    }
    changed
}

// ── Phi-copy register coalescing ─────────────────────────────────────────────
//
// Eliminates temporary register copies used for SSA phi resolution:
//
//   movq %SRC, %TMP        →  <ops directly on SRC>
//   ... ops on %TMP ...
//   movq %TMP, %SRC
//
// Conditions:
//   - SRC is not read or written between the copy-out and copy-back
//   - TMP is dead after the copy-back (next barrier or TMP overwritten)
//   - No implicit register hazards (div, mul, etc.)
//   - Window limited to 6 instructions to keep analysis local

pub(super) fn coalesce_phi_register_copies(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: movq %SRC, %TMP where both are GP regs, SRC != TMP
        let ti = infos[i].trimmed(store.get(i));
        if !ti.starts_with("movq %") { i += 1; continue; }
        if let Some(comma) = ti.find(", %") {
            let src_reg = &ti[5..comma]; // includes leading %
            let tmp_reg = &ti[comma + 2..]; // includes leading %
            if src_reg == tmp_reg || src_reg.contains('(') { i += 1; continue; }

            let src_family = register_family_fast(src_reg);
            let tmp_family = register_family_fast(tmp_reg);
            if src_family == REG_NONE || src_family > REG_GP_MAX { i += 1; continue; }
            if tmp_family == REG_NONE || tmp_family > REG_GP_MAX { i += 1; continue; }
            // Don't coalesce rax(0) or rcx(1) as SRC — they're accumulator regs
            // with heavy implicit use
            if src_family <= 1 || tmp_family <= 1 { i += 1; continue; }
            // Don't coalesce when SRC is a caller-saved register and TMP is callee-saved.
            // Caller-saved registers (rax=0, rcx=1, rdx=2, rsi=6, rdi=7, r8-r11=8-11)
            // get clobbered by function calls. Coalescing away the copy to a callee-saved
            // register loses the value across calls. This is critical for parameter
            // pre-stores (movq %rdi, %r12) that save params before calls.
            let src_is_caller_saved = matches!(src_family, 0 | 1 | 2 | 6 | 7 | 8 | 9 | 10 | 11);
            let tmp_is_callee_saved = matches!(tmp_family, 3 | 5 | 12 | 13 | 14 | 15);
            if src_is_caller_saved && tmp_is_callee_saved { i += 1; continue; }

            let src_bit = 1u16 << src_family;
            let tmp_bit = 1u16 << tmp_family;

            // Scan forward for the copy-back: movq %TMP, %SRC
            // Also track chains: if TMP is sign-extended to TMP2 (movslq %TMPd, %TMP2),
            // accept movq %TMP2, %SRC as the copy-back too.
            // Within a window of 8 non-NOP instructions.
            let copy_back_pat = format!("movq {}, {}", tmp_reg, src_reg);
            let mut j = i + 1;
            let mut instr_count = 0;
            let mut src_referenced = false;
            let mut has_implicit_hazard = false;
            let mut copy_back_pos = None;
            // Chain tracking: if TMP gets sign-extended to a different register
            let mut chain_family: RegId = REG_NONE;
            let mut chain_bit: u16 = 0;

            while j < len && instr_count < 8 {
                if infos[j].is_nop() { j += 1; continue; }
                if infos[j].is_barrier() { break; }

                let tj = infos[j].trimmed(store.get(j));

                // Check for direct copy-back: movq %TMP, %SRC
                if tj == copy_back_pat {
                    copy_back_pos = Some(j);
                    break;
                }

                // Check for chain copy-back: movq %TMP2, %SRC (where TMP2 came from movslq %TMPd, %TMP2)
                if chain_family != REG_NONE {
                    let chain_wb = format!("movq {}, {}", REG_NAMES[0][chain_family as usize], src_reg);
                    if tj == chain_wb {
                        copy_back_pos = Some(j);
                        break;
                    }
                }

                // Check SRC is not referenced between copy-out and copy-back
                if infos[j].reg_refs & src_bit != 0 {
                    src_referenced = true;
                    break;
                }

                // Track sign-extend chain: movslq %TMPd, %OTHER
                if chain_family == REG_NONE && tj.starts_with("movslq ") {
                    let tmp_32 = REG_NAMES[1][tmp_family as usize];
                    let chain_prefix = format!("movslq {}, %", tmp_32);
                    if tj.starts_with(&chain_prefix) {
                        let other_reg = &tj[chain_prefix.len() - 1..];
                        let other_fam = register_family_fast(other_reg);
                        if other_fam != REG_NONE && other_fam != tmp_family && other_fam != src_family && other_fam > 1 {
                            chain_family = other_fam;
                            chain_bit = 1u16 << chain_family;
                        }
                    }
                }

                // Check for implicit register hazards
                if has_implicit_reg_usage(tj) {
                    has_implicit_hazard = true;
                    break;
                }

                instr_count += 1;
                j += 1;
            }

            if src_referenced || has_implicit_hazard || copy_back_pos.is_none() {
                i += 1; continue;
            }

            let cb = copy_back_pos.unwrap();

            // Verify TMP (and chain TMP2 if present) are dead after the copy-back.
            let check_bits = tmp_bit | chain_bit;
            let mut tmp_dead = false;
            let mut n = cb + 1;
            let mut chk_count = 0;
            while n < len && chk_count < 8 {
                if infos[n].is_nop() { n += 1; continue; }
                if infos[n].is_barrier() {
                    tmp_dead = true;
                    break;
                }
                if infos[n].reg_refs & check_bits != 0 {
                    // Check each referenced temp
                    let mut is_write = false;
                    match infos[n].kind {
                        LineKind::Other { dest_reg } if dest_reg == tmp_family || dest_reg == chain_family => { is_write = true; }
                        LineKind::LoadRbp { reg, .. } if reg == tmp_family || reg == chain_family => { is_write = true; }
                        _ => {}
                    }
                    if !is_write {
                        break; // TMP or chain reg read after copy-back → can't coalesce
                    }
                }
                chk_count += 1;
                n += 1;
            }
            if chk_count >= 8 { tmp_dead = true; }

            if !tmp_dead { i += 1; continue; }

            // Apply: remove copy-out, replace TMP with SRC in intermediate ops, remove copy-back
            // For chain coalescing, also replace TMP2 with SRC
            mark_nop(&mut infos[i]);
            for mid in (i + 1)..cb {
                if infos[mid].is_nop() { continue; }
                let needs_replace = infos[mid].reg_refs & (tmp_bit | chain_bit) != 0;
                if needs_replace {
                    let mut new_text = store.get(mid).to_string();
                    if infos[mid].reg_refs & tmp_bit != 0 {
                        new_text = replace_reg_family(&new_text, tmp_family, src_family);
                    }
                    if chain_family != REG_NONE && infos[mid].reg_refs & chain_bit != 0 {
                        new_text = replace_reg_family(&new_text, chain_family, src_family);
                    }
                    if new_text != store.get(mid) {
                        replace_line(store, &mut infos[mid], mid, new_text);
                    }
                }
            }
            mark_nop(&mut infos[cb]);

            changed = true;
            i = cb + 1;
            continue;
        }
        i += 1;
    }
    changed
}

// ── Loop-invariant GPR load hoisting ─────────────────────────────────────────
//
// Hoists stack loads that are invariant across a loop body to just before the
// loop header. Generalizes promote_loop_invariant_fp_load to GPR loads.
//
// Pattern:
//   .LBB_header:
//     ...
//   .LBB_body:
//     movq OFFSET(%rsp), %REG   ← invariant load (OFFSET not written in loop)
//     ...use %REG...
//     jmp .LBB_header            ← back-edge
//
// Transformed to:
//   movq OFFSET(%rsp), %REG     ← hoisted before header
//   .LBB_header:
//     ...
//   .LBB_body:
//     ...use %REG...             ← load removed
//     jmp .LBB_header

pub(super) fn hoist_loop_invariant_gpr_load(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() { i += 1; continue; }
        // Match both unconditional (jmp) and conditional (jle, jl, jne, etc.) back-edges
        let jmp_text = infos[i].trimmed(store.get(i));
        let target = if jmp_text.starts_with("jmp ") { &jmp_text[4..] }
            else if jmp_text.starts_with("jl ") { &jmp_text[3..] }
            else if jmp_text.starts_with("jle ") { &jmp_text[4..] }
            else if jmp_text.starts_with("jne ") { &jmp_text[4..] }
            else if jmp_text.starts_with("jge ") { &jmp_text[4..] }
            else if jmp_text.starts_with("jg ") { &jmp_text[3..] }
            else if jmp_text.starts_with("jb ") { &jmp_text[3..] }
            else if jmp_text.starts_with("ja ") { &jmp_text[3..] }
            else { i += 1; continue; };
        if !target.starts_with(".L") { i += 1; continue; }
        let target_label = format!("{}:", target);

        // Find the target label (must be before the branch = back-edge)
        let mut header_pos = None;
        for lbl in 0..i {
            if infos[lbl].kind == LineKind::Label {
                if infos[lbl].trimmed(store.get(lbl)) == target_label {
                    header_pos = Some(lbl);
                    break;
                }
            }
        }
        let header = match header_pos {
            Some(h) => h,
            None => { i += 1; continue; }
        };

        // Validate this is a real loop: the range [header..=i] must not contain
        // a ret instruction (which would indicate the range spans the epilogue
        // and is not a natural loop body).
        let mut has_ret = false;
        let mut has_call = false;
        for chk in header..=i {
            if infos[chk].is_nop() { continue; }
            match infos[chk].kind {
                LineKind::Ret => { has_ret = true; break; }
                LineKind::Call => { has_call = true; }
                _ => {}
            }
        }
        if has_ret { i += 1; continue; }

        // Loop body is [header..=i]. Find the body start (first label after header).
        let mut body_start = header;
        for pos in header + 1..i {
            if infos[pos].kind == LineKind::Label {
                body_start = pos;
                break;
            }
        }

        // Scan body for movq OFFSET(%rsp), %REG (or %rbp) candidates.
        // Only hoist one load per loop per pass (to avoid interactions).
        // Skip loops with function calls — caller-saved regs could be clobbered.
        if has_call { i += 1; continue; }
        let mut hoisted_one = false;
        for pos in body_start + 1..i {
            if hoisted_one { break; }
            if infos[pos].is_nop() { continue; }

            // Match: movq OFFSET(%rsp), %REG or movq OFFSET(%rbp), %REG
            let t = infos[pos].trimmed(store.get(pos));
            if !t.starts_with("movq ") { continue; }
            // Parse: "movq SRC, %DST"
            let after_movq = &t[5..];
            let comma = match after_movq.find(", %") {
                Some(c) => c,
                None => continue,
            };
            let src_part = &after_movq[..comma];
            let dst_part = &after_movq[comma + 2..]; // includes %

            // Source must be a stack slot
            if !src_part.ends_with("(%rsp)") && !src_part.ends_with("(%rbp)") { continue; }
            // Destination must be a GP register
            let dst_family = register_family_fast(dst_part);
            if dst_family == REG_NONE || dst_family > REG_GP_MAX { continue; }
            // Don't hoist into rax/rcx (accumulator) or rsp/rbp
            // Skip rax (primary accumulator), rsp, rbp (frame registers)
            if dst_family == 0 || dst_family == 4 || dst_family == 5 { continue; }

            // Parse the numeric offset
            let offset_end = src_part.find('(').unwrap_or(src_part.len());
            let numeric_offset: i32 = src_part[..offset_end].parse().unwrap_or(i32::MIN);
            if numeric_offset == i32::MIN { continue; }

            // Check: the stack slot is NOT written anywhere in [header..=i]
            let mut slot_written = false;
            for chk in header..=i {
                if infos[chk].is_nop() { continue; }
                if let LineKind::StoreRbp { offset: o, .. } = infos[chk].kind {
                    if o == numeric_offset { slot_written = true; break; }
                }
                // Also check Other instructions that store to this offset
                let ct = infos[chk].trimmed(store.get(chk));
                if ct.ends_with(src_part) && (ct.starts_with("movq ") || ct.starts_with("movl ")
                    || ct.starts_with("movb ") || ct.starts_with("movw ")) {
                    // This could be a store TO this slot
                    if let Some(c) = ct.find(", ") {
                        if ct[c + 2..] == *src_part {
                            slot_written = true;
                            break;
                        }
                    }
                }
            }
            if slot_written { continue; }

            // Check: the destination register is NOT written by any other instruction
            // in [header..=i] besides this load. Also check it's not used as a
            // destination in any other instruction.
            let dst_bit = 1u16 << dst_family;
            let mut reg_written_elsewhere = false;
            for chk in header..=i {
                if chk == pos { continue; } // skip the load itself
                if infos[chk].is_nop() { continue; }
                match infos[chk].kind {
                    LineKind::Other { dest_reg } if dest_reg == dst_family => {
                        reg_written_elsewhere = true;
                        break;
                    }
                    LineKind::LoadRbp { reg, .. } if reg == dst_family => {
                        reg_written_elsewhere = true;
                        break;
                    }
                    LineKind::Call => {
                        // Calls clobber caller-saved regs. If dst is caller-saved, bail.
                        if !is_callee_saved_reg(dst_family) {
                            reg_written_elsewhere = true;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if reg_written_elsewhere { continue; }

            // All checks passed. Hoist the load before the header.
            let load_text = store.get(pos).to_string();
            // Insert by replacing the header label line with: load + \n + label
            let header_text = store.get(header).to_string();
            let new_header = format!("{}\n{}", load_text.trim_end(), header_text.trim_end());
            store.replace(header, new_header);
            // Re-classify header (it's now multi-line but LineStore handles this)
            // Actually, LineStore doesn't handle multi-line. We need a different approach.
            // Instead: find an insertion point before the header and use it.

            // Better approach: replace the header label with load + label on same logical slot
            // by using the slot just before header if it exists.
            // Revert: just mark the body load as NOP and insert before header.
            store.replace(header, header_text); // undo

            // Find a NOP slot before header, or use the line just before header
            // by prepending the load instruction.
            // Simplest: convert the load at `pos` to NOP, and insert the load
            // at a position just before the header label.
            // Since LineStore doesn't support insertion, we'll use the approach
            // from promote_loop_invariant_fp_load: find a preheader instruction
            // to append after.
            // Actually, let's find if there's a preceding NOP or unused line.

            // Alternative: find the store that wrote this value before the loop.
            // Replace that store's destination from stack to register directly.
            // But that's complex. Let's just look for any NOP before header.
            let mut insert_pos = None;
            if header > 0 {
                for p in (0..header).rev() {
                    if infos[p].is_nop() {
                        insert_pos = Some(p);
                        break;
                    }
                    // Don't go past function boundaries
                    if infos[p].kind == LineKind::Directive {
                        let dt = infos[p].trimmed(store.get(p));
                        if dt.starts_with(".size ") || dt.starts_with(".globl ") || dt.starts_with(".type ") {
                            break;
                        }
                    }
                    // Stop if we hit a non-trivial instruction without finding a NOP
                    if p < header.saturating_sub(20) { break; }
                }
            }

            if let Some(ins) = insert_pos {
                // Use the NOP slot to place our hoisted load
                replace_line(store, &mut infos[ins], ins, load_text.trim_end().to_string());
                mark_nop(&mut infos[pos]); // remove original load
                changed = true;
                hoisted_one = true;
            } else {
                // No NOP slot found: prepend the load to the header label line.
                // This places the load just before the loop entry.
                let header_text = store.get(header).to_string();
                let combined = format!("{}\n{}", load_text.trim_end(), header_text.trim_end());
                store.replace(header, combined);
                mark_nop(&mut infos[pos]);
                changed = true;
                hoisted_one = true;
            }
        }

        i += 1;
    }
    changed
}

// ── Loop-invariant broadcast hoisting ────────────────────────────────────────
//
// Hoists loop-invariant movsd+vbroadcastsd pairs out of the inner loop:
//
//   .loop:
//     movsd (%REG), %xmm1          →  (NOPed, hoisted before loop)
//     vbroadcastsd %xmm1, %ymm1    →  (NOPed, hoisted before loop)
//     vmovupd ...                      vmovupd ...
//     vfmadd231pd ...                  vfmadd231pd ...
//
// The hoist is safe when %REG is not modified within the loop body.

pub(super) fn hoist_loop_invariant_fp_broadcast(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }
        // Find back-edge: jl/jle/jne/jb/ja/jmp to a label before this instruction
        let jmp_text = infos[i].trimmed(store.get(i));
        let target = if jmp_text.starts_with("jl ") { &jmp_text[3..] }
            else if jmp_text.starts_with("jle ") { &jmp_text[4..] }
            else if jmp_text.starts_with("jne ") { &jmp_text[4..] }
            else if jmp_text.starts_with("jmp ") { &jmp_text[4..] }
            else { i += 1; continue; };
        if !target.starts_with(".L") { i += 1; continue; }

        let target_label = format!("{}:", target);
        let mut header_pos = None;
        for lbl in 0..i {
            if infos[lbl].kind == LineKind::Label &&
               infos[lbl].trimmed(store.get(lbl)) == target_label {
                header_pos = Some(lbl);
                break;
            }
        }
        let header = match header_pos {
            Some(h) => h,
            None => { i += 1; continue; }
        };

        // Validate: no ret or call in loop
        let mut has_ret = false;
        let mut has_call = false;
        for chk in header..=i {
            if infos[chk].is_nop() { continue; }
            match infos[chk].kind {
                LineKind::Ret => { has_ret = true; break; }
                LineKind::Call => { has_call = true; }
                _ => {}
            }
        }
        if has_ret || has_call { i += 1; continue; }

        // Scan loop body for: movsd (%REG), %xmm1 followed by vbroadcastsd %xmm1, %ymm1
        let mut hoisted_one = false;
        for pos in header + 1..i {
            if hoisted_one { break; }
            if infos[pos].is_nop() { continue; }
            let t1 = infos[pos].trimmed(store.get(pos));
            if !t1.starts_with("movsd (%") || !t1.ends_with("), %xmm1") { continue; }

            // Extract the source register
            let reg_start = 7; // after "movsd (%"
            let reg_end = t1.find("), %xmm1").unwrap_or(0);
            if reg_end <= reg_start { continue; }
            let src_reg = &t1[reg_start..reg_end];

            // Next non-NOP must be vbroadcastsd
            let mut pos2 = pos + 1;
            while pos2 < i && infos[pos2].is_nop() { pos2 += 1; }
            if pos2 >= i { continue; }
            let t2 = infos[pos2].trimmed(store.get(pos2));
            if t2 != "vbroadcastsd %xmm1, %ymm1" { continue; }

            // Check that src_reg is NOT modified within the loop
            let write_pattern = format!(", %{}", src_reg);
            let mut reg_modified = false;
            for chk in header..=i {
                if chk == pos || chk == pos2 { continue; }
                if infos[chk].is_nop() { continue; }
                let ct = infos[chk].trimmed(store.get(chk));
                if ct.contains(&write_pattern) || ct.ends_with(&format!("%{}", src_reg)) {
                    // Check if it's a destination (after last comma)
                    if let Some(last_comma) = ct.rfind(", ") {
                        let dest_part = &ct[last_comma + 2..];
                        if dest_part.contains(src_reg) {
                            reg_modified = true;
                            break;
                        }
                    }
                }
            }
            if reg_modified { continue; }

            // Find a NOP slot JUST before the header (within 10 lines) to place
            // both hoisted instructions as a combined two-line string.
            let mut slot = None;
            for p in (0..header).rev() {
                if infos[p].is_nop() {
                    slot = Some(p);
                    break;
                }
                // Only search in the immediate preheader
                if p < header.saturating_sub(10) { break; }
                if infos[p].kind == LineKind::Label { break; }
            }

            if let Some(s) = slot {
                let movsd_text = store.get(pos).trim_end().to_string();
                let bcast_text = store.get(pos2).trim_end().to_string();
                // Combine both instructions into one slot
                let combined = format!("{}\n{}", movsd_text, bcast_text);
                replace_line(store, &mut infos[s], s, combined);
                mark_nop(&mut infos[pos]);
                mark_nop(&mut infos[pos2]);
                changed = true;
                hoisted_one = true;
            }
        }

        i += 1;
    }
    changed
}

// ── Add + sign-extend fusion ─────────────────────────────────────────────────
//
// Fuses addl + movslq when the intermediate 32-bit register is only used
// as a temporary:
//
//   addl %SRC, %TMP            →  addl %SRC, %DSTd
//   movslq %TMPd, %DST        →  (NOP)
//
// The 32-bit addl into %DSTd automatically zero-extends to 64-bit on x86-64.
// This is safe when the value is non-negative (array indices, loop counters).

pub(super) fn fuse_add_sign_extend(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    // Simple adjacent-pair scan: find addl %X, %TMP immediately followed by
    // movslq %TMP, %DST. Check that %TMP is not used elsewhere in the
    // surrounding block.
    //
    // SAFETY: addl %X, %TMP is a read-modify-write: it reads TMP, adds X,
    // writes back to TMP. We can only redirect the destination to DST when
    // TMP == X (self-add, e.g. addl %ebx, %ebx → doubling), because then
    // changing to addl %DSTd, %DSTd is correct if DST was initialized with
    // the same value. For X != TMP, the redirect loses the TMP read.
    while i + 1 < len {
        let t1 = store.get(i).trim().to_string();
        if !t1.starts_with("addl %") { i += 1; continue; }
        let parts1: Vec<&str> = t1.split(", %").collect();
        if parts1.len() != 2 { i += 1; continue; }
        let src_reg = parts1[0][6..].to_string(); // after "addl %"
        let tmp_reg = parts1[1].to_string();

        // addl %X, %TMP is read-modify-write on TMP. Redirecting the
        // destination to DST is only safe when X == TMP (self-add / doubling).
        // For X != TMP, the old TMP value is part of the computation and
        // DST wouldn't have it.
        let src_base = src_reg.trim_end_matches('d').to_string();
        if src_base != tmp_reg && src_reg != tmp_reg {
            i += 1; continue;
        }

        // Next line must be movslq %TMP, %DST (check all lines in case of multi-line)
        let t2_raw = store.get(i + 1);
        let t2 = t2_raw.trim().to_string();
        // Handle multi-line entries: check first line
        let t2_first = t2.split('\n').next().unwrap_or(&t2).trim().to_string();
        if !t2_first.starts_with("movslq %") { i += 1; continue; }
        let parts2: Vec<&str> = t2_first.split(", %").collect();
        if parts2.len() != 2 { i += 1; continue; }
        let sext_src = parts2[0][8..].to_string();
        let dst_reg = parts2[1].to_string();

        let sext_base = sext_src.trim_end_matches('d').to_string();
        if sext_base != tmp_reg && sext_src != tmp_reg {
            i += 1; continue;
        }

        // Find enclosing block bounds for safety check
        let mut block_start = 0;
        for k in (0..i).rev() {
            let tk = store.get(k);
            if tk.trim().ends_with(':') || tk.contains(":\n") {
                block_start = k;
                break;
            }
        }
        let mut block_end = len - 1;
        for k in (i + 2)..len.min(i + 30) {
            let tk = store.get(k).trim().to_string();
            if tk.starts_with("j") || tk.ends_with(':') {
                block_end = k;
                break;
            }
        }

        // Safety: %TMP must not appear elsewhere in the block
        let tmp_pattern = format!("%{}", tmp_reg);
        let mut tmp_used_elsewhere = false;
        for k in block_start..=block_end {
            if k == i || k == i + 1 { continue; }
            let tk = store.get(k).to_string();
            if tk.contains(&tmp_pattern) {
                tmp_used_elsewhere = true;
                break;
            }
        }
        if tmp_used_elsewhere { i += 1; continue; }

        // Transform!
        let dst_reg32 = format!("{}d", dst_reg);
        let new_add = format!("    addl %{}, %{}", src_reg, dst_reg32);
        let t2_raw_owned = store.get(i + 1).to_string();
        store.replace(i, new_add);
        // If movslq was part of a multi-line entry, keep the rest
        if t2_raw_owned.contains('\n') {
            let rest = t2_raw_owned.split_once('\n').map(|(_, r)| r).unwrap_or("");
            store.replace(i + 1, rest.to_string());
        } else {
            store.replace(i + 1, String::new());
        }
        changed = true;
        i += 2;
        continue;
    }
    changed
}

// ── Increment chain collapse ─────────────────────────────────────────────────
//
// Collapses a common SSA phi-resolution pattern for loop counter increments:
//
//   leaq 1(%SRC), %TMP1       →  addl $1, %SRCd
//   movslq %TMP1d, %TMP2      →  movslq %SRCd, %SRC
//   movq %TMP2, %SRC          →  (removed)
//
// Saves 1 instruction per loop iteration. Common in loop counter increments.

pub(super) fn collapse_increment_chain(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: leaq DISP(%SRC), %TMP1
        let ti = infos[i].trimmed(store.get(i));
        if !ti.starts_with("leaq ") { i += 1; continue; }

        let after_leaq = &ti[5..];
        let paren_open = match after_leaq.find('(') {
            Some(p) => p,
            None => { i += 1; continue; }
        };
        let disp_str = &after_leaq[..paren_open];
        let paren_close = match after_leaq.find(')') {
            Some(p) => p,
            None => { i += 1; continue; }
        };
        let src_reg = &after_leaq[paren_open + 1..paren_close];
        let after_paren = &after_leaq[paren_close + 1..];
        if !after_paren.starts_with(", %") { i += 1; continue; }
        let tmp1_reg_name = &after_paren[2..];

        let src_family = register_family_fast(src_reg);
        let tmp1_family = register_family_fast(tmp1_reg_name);
        if src_family == REG_NONE || src_family > REG_GP_MAX { i += 1; continue; }
        if tmp1_family == REG_NONE || tmp1_family > REG_GP_MAX { i += 1; continue; }
        if src_family == tmp1_family || src_family <= 1 { i += 1; continue; }

        let disp: i32 = match disp_str.parse() {
            Ok(d) => d,
            Err(_) => { i += 1; continue; }
        };

        // Next non-NOP: movslq %TMP1d, %TMP2
        let j = next_non_nop(infos, i + 1, len);
        if j >= len || infos[j].is_barrier() { i += 1; continue; }

        let tj = infos[j].trimmed(store.get(j));
        let tmp1_32 = REG_NAMES[1][tmp1_family as usize];
        let expected_prefix = format!("movslq {}, %", tmp1_32);
        if !tj.starts_with(&expected_prefix) { i += 1; continue; }
        let tmp2_reg_str = &tj[expected_prefix.len() - 1..];
        let tmp2_family = register_family_fast(tmp2_reg_str);
        if tmp2_family == REG_NONE || tmp2_family > REG_GP_MAX { i += 1; continue; }

        // Search for movq %TMP2, %SRC within a window of 4 non-NOP instructions
        let src_64 = REG_NAMES[0][src_family as usize];
        let expected_wb = format!("movq {}, {}", tmp2_reg_str, src_64);
        let mut k = j + 1;
        let mut wb_found = false;
        let mut wb_count = 0;
        while k < len && wb_count < 4 {
            if infos[k].is_nop() { k += 1; continue; }
            if infos[k].is_barrier() { break; }
            let tk = infos[k].trimmed(store.get(k));
            if tk == expected_wb { wb_found = true; break; }
            wb_count += 1;
            k += 1;
        }
        if !wb_found { i += 1; continue; }

        // Check SRC not referenced between i+1 and k
        let src_bit = 1u16 << src_family;
        let mut src_ref = false;
        for mid in (i + 1)..k {
            if infos[mid].is_nop() { continue; }
            if infos[mid].reg_refs & src_bit != 0 { src_ref = true; break; }
        }
        if src_ref { i += 1; continue; }

        // Check TMPs dead after k
        let tmp1_bit = 1u16 << tmp1_family;
        let tmp2_bit = 1u16 << tmp2_family;
        let mut tmps_dead = false;
        let mut n = k + 1;
        let mut chk = 0;
        while n < len && chk < 8 {
            if infos[n].is_nop() { n += 1; continue; }
            if infos[n].is_barrier() { tmps_dead = true; break; }
            if infos[n].reg_refs & tmp1_bit != 0 {
                match infos[n].kind {
                    LineKind::Other { dest_reg } if dest_reg == tmp1_family => {}
                    LineKind::LoadRbp { reg, .. } if reg == tmp1_family => {}
                    _ => break,
                }
            }
            if tmp2_family != tmp1_family && infos[n].reg_refs & tmp2_bit != 0 {
                match infos[n].kind {
                    LineKind::Other { dest_reg } if dest_reg == tmp2_family => {}
                    LineKind::LoadRbp { reg, .. } if reg == tmp2_family => {}
                    _ => break,
                }
            }
            chk += 1;
            n += 1;
        }
        if chk >= 8 { tmps_dead = true; }
        if !tmps_dead { i += 1; continue; }

        // Flags safety check
        let post = next_non_nop(infos, k + 1, len);
        if post < len && !infos[post].is_barrier() {
            let tp = infos[post].trimmed(store.get(post));
            if tp.starts_with("ja") || tp.starts_with("jb") ||
               tp.starts_with("je") || tp.starts_with("jn") ||
               tp.starts_with("jg") || tp.starts_with("jl") ||
               tp.starts_with("js") || tp.starts_with("jo") ||
               tp.starts_with("set") || tp.starts_with("cmov") ||
               tp.starts_with("adc") || tp.starts_with("sbb") {
                i += 1; continue;
            }
        }

        // Apply
        let src_32 = REG_NAMES[1][src_family as usize];
        let new_add = format!("    addl ${}, {}", disp, src_32);
        let new_ext = format!("    movslq {}, {}", src_32, src_64);
        replace_line(store, &mut infos[i], i, new_add);
        replace_line(store, &mut infos[j], j, new_ext);
        mark_nop(&mut infos[k]);

        changed = true;
        i = k + 1;
    }
    changed
}

// ── Cascaded shift folding ───────────────────────────────────────────────────
//
// Folds a cascaded shift pattern where a value is shifted, copied, then shifted again:
//
//   movq %SRC, %TMP            →  movq %SRC, %DST
//   shlq $A, %TMP              →  shlq $(A+B), %DST
//   movq %TMP, %DST
//   shlq $B, %DST
//
// Also matches when %TMP == %rax (accumulator pattern):
//   shlq $A, %rax
//   movq %rax, %DST
//   shlq $B, %DST
//
// Saves 2 instructions per occurrence. Common in array address computation
// where stride = element_size * vector_width (e.g., 8 * 4 = 32 for AVX2 doubles).

pub(super) fn fold_cascaded_shifts(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        let ti = infos[i].trimmed(store.get(i));

        // Pattern 1: movq %SRC, %TMP; shlq $A, %TMP; movq %TMP, %DST; shlq $B, %DST
        if ti.starts_with("movq %") && ti.contains(", %") && !ti.contains("(%") {
            let comma = match ti.find(", %") { Some(c) => c, None => { i += 1; continue; } };
            let src_name = &ti[5..comma];
            let tmp_name = &ti[comma + 2..];
            if src_name == tmp_name || !src_name.starts_with('%') { i += 1; continue; }
            let src_fam = register_family_fast(src_name);
            let tmp_fam = register_family_fast(tmp_name);
            if src_fam == REG_NONE || tmp_fam == REG_NONE { i += 1; continue; }

            // Next: shlq $A, %TMP
            let j = next_non_nop(infos, i + 1, len);
            if j >= len || infos[j].is_barrier() { i += 1; continue; }
            let tj = infos[j].trimmed(store.get(j));
            let shl_suffix = format!(", {}", tmp_name);
            if !tj.starts_with("shlq $") || !tj.ends_with(&shl_suffix) { i += 1; continue; }
            let shift_a: u32 = match tj[6..tj.len() - shl_suffix.len()].parse() {
                Ok(s) => s, Err(_) => { i += 1; continue; }
            };

            // Next: movq %TMP, %DST
            let k = next_non_nop(infos, j + 1, len);
            if k >= len || infos[k].is_barrier() { i += 1; continue; }
            let tk = infos[k].trimmed(store.get(k));
            let mov_prefix = format!("movq {}, %", tmp_name);
            if !tk.starts_with(&mov_prefix) { i += 1; continue; }
            let dst_name = &tk[mov_prefix.len() - 1..]; // includes %
            let dst_fam = register_family_fast(dst_name);
            if dst_fam == REG_NONE || dst_fam == tmp_fam { i += 1; continue; }

            // Next: shlq $B, %DST
            let m = next_non_nop(infos, k + 1, len);
            if m >= len || infos[m].is_barrier() { i += 1; continue; }
            let tm = infos[m].trimmed(store.get(m));
            let shl_dst = format!(", {}", dst_name);
            if !tm.starts_with("shlq $") || !tm.ends_with(&shl_dst) { i += 1; continue; }
            let shift_b: u32 = match tm[6..tm.len() - shl_dst.len()].parse() {
                Ok(s) => s, Err(_) => { i += 1; continue; }
            };

            let total_shift = shift_a + shift_b;
            if total_shift > 63 { i += 1; continue; }

            // Check TMP is dead after this sequence
            let tmp_bit = 1u16 << tmp_fam;
            let mut tmp_dead = false;
            let mut n = m + 1;
            let mut chk = 0;
            while n < len && chk < 8 {
                if infos[n].is_nop() { n += 1; continue; }
                if infos[n].is_barrier() { tmp_dead = true; break; }
                if infos[n].reg_refs & tmp_bit != 0 {
                    match infos[n].kind {
                        LineKind::Other { dest_reg } if dest_reg == tmp_fam => { tmp_dead = true; break; }
                        LineKind::LoadRbp { reg, .. } if reg == tmp_fam => { tmp_dead = true; break; }
                        _ => break, // TMP read → not dead
                    }
                }
                chk += 1;
                n += 1;
            }
            if chk >= 8 { tmp_dead = true; }
            if !tmp_dead { i += 1; continue; }

            // Apply: movq %SRC, %DST + shlq $total, %DST
            let new_mov = format!("    movq {}, {}", src_name, dst_name);
            let new_shl = format!("    shlq ${}, {}", total_shift, dst_name);
            replace_line(store, &mut infos[i], i, new_mov);
            mark_nop(&mut infos[j]);
            replace_line(store, &mut infos[k], k, new_shl);
            mark_nop(&mut infos[m]);

            changed = true;
            i = m + 1;
            continue;
        }

        i += 1;
    }
    changed
}

// ── Loop rotation ────────────────────────────────────────────────────────────
//
// Moves the loop condition from the header to the latch, eliminating one
// unconditional branch per iteration:
//
//   .header:                    →  .header:            (preheader)
//       <setup>                 →      <setup>
//       <compare>               →      <compare>
//       jCC .exit               →      jCC .exit
//   .body:                      →  .body:              (rotated loop)
//       <body>                  →      <body>
//       jmp .header             →      <setup_copy>
//                               →      <compare_copy>
//                               →      j!CC .body
//
// Only applied when the header setup is 0-3 instructions (simple cases).
// The jmp is replaced with a multi-line string (setup + cmp + inverted branch).

pub(super) fn rotate_loops(
    store: &mut LineStore,
    infos: &mut [LineInfo],
) -> bool {
    let len = store.len();
    let mut changed = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() { i += 1; continue; }
        if infos[i].kind != LineKind::Jmp { i += 1; continue; }
        let jmp_text = infos[i].trimmed(store.get(i));
        if !jmp_text.starts_with("jmp ") { i += 1; continue; }
        let target = &jmp_text[4..];
        let target_label = format!("{}:", target);

        // Find the target label (must be before the jmp = back-edge)
        let mut header_pos = None;
        for lbl in 0..i {
            if infos[lbl].kind == LineKind::Label {
                if infos[lbl].trimmed(store.get(lbl)) == target_label {
                    header_pos = Some(lbl);
                    break;
                }
            }
        }
        let header = match header_pos {
            Some(h) => h,
            None => { i += 1; continue; }
        };

        // Validate: no ret/call in the loop body
        let mut has_ret = false;
        let mut has_call = false;
        for chk in header..=i {
            if infos[chk].is_nop() { continue; }
            match infos[chk].kind {
                LineKind::Ret => { has_ret = true; break; }
                LineKind::Call => { has_call = true; }
                _ => {}
            }
        }
        if has_ret || has_call { i += 1; continue; }

        // Collect non-NOP instructions between header label and the first conditional jump.
        let mut header_instrs: Vec<usize> = Vec::new();
        let mut cond_jmp_pos = None;
        let mut body_label_pos = None;
        let mut pos = header + 1;
        while pos < i {
            if infos[pos].is_nop() { pos += 1; continue; }
            if infos[pos].kind == LineKind::CondJmp {
                cond_jmp_pos = Some(pos);
                // Find body label right after conditional jump
                let mut bl = pos + 1;
                while bl < i {
                    if infos[bl].is_nop() { bl += 1; continue; }
                    if infos[bl].kind == LineKind::Label {
                        body_label_pos = Some(bl);
                    }
                    break;
                }
                break;
            }
            if infos[pos].kind == LineKind::Label { break; } // complex header
            if infos[pos].is_barrier() { break; }
            header_instrs.push(pos);
            pos += 1;
        }

        let cjmp = match cond_jmp_pos {
            Some(c) => c,
            None => { i += 1; continue; }
        };

        // Only handle simple headers (0-3 setup instructions before the cond jmp)
        if header_instrs.len() > 3 { i += 1; continue; }

        // The instruction(s) before the cond jmp must include a compare/test.
        // Find the compare in the header setup.
        let has_cmp = header_instrs.iter().any(|&idx| {
            let t = infos[idx].trimmed(store.get(idx));
            t.starts_with("cmpl ") || t.starts_with("cmpq ") ||
            t.starts_with("cmpb ") || t.starts_with("cmpw ") ||
            t.starts_with("testl ") || t.starts_with("testq ") ||
            matches!(infos[idx].kind, LineKind::Cmp)
        });
        if !has_cmp { i += 1; continue; }

        // Get the conditional jump and invert it
        let cjmp_text = infos[cjmp].trimmed(store.get(cjmp));
        let (cond, exit_label) = match cjmp_text.find(' ') {
            Some(space) => (&cjmp_text[..space], &cjmp_text[space + 1..]),
            None => { i += 1; continue; }
        };

        let inv_cond = match cond {
            "je" => "jne", "jne" => "je",
            "jl" => "jge", "jge" => "jl",
            "jle" => "jg", "jg" => "jle",
            "jb" => "jae", "jae" => "jb",
            "jbe" => "ja", "ja" => "jbe",
            "js" => "jns", "jns" => "js",
            _ => { i += 1; continue; }
        };

        // Find the body label for the rotated backedge
        let body_label = match body_label_pos {
            Some(bl) => {
                let bt = infos[bl].trimmed(store.get(bl));
                bt.trim_end_matches(':').to_string()
            }
            None => { i += 1; continue; }
        };

        // Optimize the latch: detect redundant sign-extend pattern.
        // If the header setup is `movslq %Xd, %Y; cmpl $imm, %Yd` and the body
        // (just before the jmp) has `movslq %Xd, %X`, we can:
        // - NOP the body's `movslq %Xd, %X` (redundant self sign-extend)
        // - Retarget the latch's movslq to `movslq %Xd, %Y` (for next iter's index)
        // - Use `cmpl $imm, %Xd` directly instead of `cmpl $imm, %Yd`
        let mut nop_body_signext = None;
        let mut optimized_latch = false;

        if header_instrs.len() >= 2 {
            // Check: first header instr is `movslq %Xd, %Y`
            let first_setup = infos[header_instrs[0]].trimmed(store.get(header_instrs[0]));
            if first_setup.starts_with("movslq %") {
                if let Some(comma) = first_setup.find(", %") {
                    let src_32 = &first_setup[7..comma]; // e.g., "%r12d"
                    let dst_64 = &first_setup[comma + 2..]; // e.g., "%r14"
                    // src_32 should be a 32-bit register like "%r12d"
                    // Derive the 64-bit version by removing the 'd' suffix
                    let src_64 = if src_32.ends_with('d') {
                        let base = &src_32[..src_32.len() - 1];
                        if base.starts_with("%r") && base.len() >= 3 {
                            Some(base.to_string())
                        } else if base == "%eax" { Some("%rax".to_string()) }
                        else if base == "%ecx" { Some("%rcx".to_string()) }
                        else if base == "%edx" { Some("%rdx".to_string()) }
                        else if base == "%ebx" { Some("%rbx".to_string()) }
                        else if base == "%esi" { Some("%rsi".to_string()) }
                        else if base == "%edi" { Some("%rdi".to_string()) }
                        else { None }
                    } else { None };

                    if let Some(ref src_64_name) = src_64 {
                        // Look for `movslq %Xd, %X` in the body just before the jmp
                        let self_signext = format!("movslq {}, {}", src_32, src_64_name);
                        // Search backwards from the jmp for this pattern
                        let mut search = if i > 0 { i - 1 } else { 0 };
                        let mut search_count = 0;
                        while search > header && search_count < 4 {
                            if infos[search].is_nop() {
                                if search == 0 { break; }
                                search -= 1;
                                continue;
                            }
                            let st = infos[search].trimmed(store.get(search));
                            if st == self_signext {
                                nop_body_signext = Some(search);
                                break;
                            }
                            search_count += 1;
                            if search == 0 { break; }
                            search -= 1;
                        }

                        // Check: the compare uses %Yd (the destination of the movslq)
                        if nop_body_signext.is_some() {
                            let last_setup = header_instrs.last().unwrap();
                            let cmp_text = infos[*last_setup].trimmed(store.get(*last_setup));
                            let dst_32 = reg_64_to_32(dst_64);
                            if !dst_32.is_empty() && cmp_text.contains(&dst_32) {
                                // Can optimize: replace cmpl's operand from %Yd to %Xd
                                optimized_latch = true;
                            }
                        }
                    }
                }
            }
        }

        // Build the replacement: duplicate setup + compare + inverted branch to body
        let mut replacement_lines = Vec::new();
        if optimized_latch {
            // Optimized: skip the self sign-extend in the body, keep movslq to index reg,
            // use source register directly in compare
            if let Some(nop_idx) = nop_body_signext {
                mark_nop(&mut infos[nop_idx]); // remove redundant movslq %Xd, %X
            }
            // Emit: movslq %Xd, %Y (for next iter's index)
            let first_setup = store.get(header_instrs[0]).to_string();
            replacement_lines.push(first_setup.trim_end().to_string());
            // Emit remaining setup except the compare, then emit optimized compare
            for &setup_idx in &header_instrs[1..header_instrs.len() - 1] {
                let text = store.get(setup_idx).to_string();
                replacement_lines.push(text.trim_end().to_string());
            }
            // Emit compare with source register instead of destination register
            let first_text = infos[header_instrs[0]].trimmed(store.get(header_instrs[0]));
            if let Some(comma) = first_text.find(", %") {
                let src_32 = &first_text[7..comma]; // e.g., "%r12d"
                let dst_64 = &first_text[comma + 2..]; // e.g., "%r14"
                let dst_32 = if dst_64.starts_with("%r") && dst_64.len() >= 3 {
                    format!("{}d", dst_64)
                } else {
                    match dst_64 {
                        "%rax" => "%eax".to_string(), "%rcx" => "%ecx".to_string(),
                        "%rdx" => "%edx".to_string(), "%rbx" => "%ebx".to_string(),
                        "%rsi" => "%esi".to_string(), "%rdi" => "%edi".to_string(),
                        _ => String::new(),
                    }
                };
                let last_setup_idx = *header_instrs.last().unwrap();
                let cmp_text = store.get(last_setup_idx).to_string();
                let optimized_cmp = cmp_text.trim_end().replace(&dst_32, src_32);
                replacement_lines.push(optimized_cmp);
            }
        } else {
            // Standard: duplicate all setup instructions verbatim
            for &setup_idx in &header_instrs {
                let text = store.get(setup_idx).to_string();
                replacement_lines.push(text.trim_end().to_string());
            }
        }
        replacement_lines.push(format!("    {} {}", inv_cond, body_label));
        // When the loop exits (inverted condition not taken), fall-through goes
        // to the next instruction. But the original loop exit target might be a
        // trampoline block elsewhere (created by phi elimination for critical
        // edges). We must emit a jmp to the original exit target so the fall-
        // through reaches the correct destination with phi initialization copies.
        replacement_lines.push(format!("    jmp {}", exit_label));

        let replacement = replacement_lines.join("\n");

        // Replace the jmp with the duplicated latch + exit jump
        store.replace(i, replacement);
        infos[i] = classify_line(store.get(i));

        changed = true;
        i += 1;
    }
    changed
}

/// Eliminate redundant `leaq src, %rax` when %rax already holds `src` from
/// a previous leaq in the same basic block.
///
/// Pattern: `leaq X, %rax` ... `leaq X, %rax` where %rax wasn't clobbered.
/// The second leaq is eliminated (marked NOP).
///
/// This handles the common accumulator pattern where alloca addresses are
/// recomputed multiple times for successive Load/Store operations.
pub(super) fn eliminate_redundant_leaq(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    let mut changed = false;
    // Track: the last leaq source that produced %rax, and its position
    let mut rax_leaq_src: Option<String> = None;

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        let line = infos[i].trimmed(store.get(i));

        // Block boundary resets tracking
        if line.ends_with(':') || line.starts_with(".LBB") || line == "ret"
            || line.starts_with("jmp ") || line.starts_with("call ") {
            rax_leaq_src = None;
            continue;
        }

        // Check if this is `leaq X, %rax`
        if line.starts_with("leaq ") && line.ends_with(", %rax") {
            let src = &line[5..line.len() - 6]; // between "leaq " and ", %rax"
            if let Some(ref prev_src) = rax_leaq_src {
                if src == prev_src.as_str() {
                    // Redundant — rax already holds this address
                    super::super::types::mark_nop(&mut infos[i]);
                    changed = true;
                    continue;
                }
            }
            rax_leaq_src = Some(src.to_string());
            continue;
        }

        // Check if %rax is written (clobbered). Conservative: any instruction
        // that writes to rax/eax as destination invalidates the leaq cache.
        // Reads of rax (movq %rax, X / cmpq %rax / testq %rax) are OK.
        let writes_rax = line.ends_with(", %rax") || line.ends_with(", %eax")
            || line.starts_with("call ") || line.starts_with("xorl %eax, %eax")
            || line == "cltq" || line == "cqto"
            || line.starts_with("popq %rax")
            || (line.starts_with("addq ") && line.ends_with(", %rax"))
            || (line.starts_with("subq ") && line.ends_with(", %rax"))
            || (line.starts_with("imulq ") && line.ends_with(", %rax"))
            || (line.starts_with("andq ") && line.ends_with(", %rax"))
            || (line.starts_with("orq ") && line.ends_with(", %rax"))
            || (line.starts_with("xorq ") && line.ends_with(", %rax"));
        if writes_rax {
            rax_leaq_src = None;
        }
    }
    changed
}
