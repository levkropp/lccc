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
use super::helpers::is_valid_gp_reg;

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
                        let new_text = format!("    movsd {}(%rbp), {}", offset, xmm_str);
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
                            let new_text = format!("    movsd %xmm0, {}(%rbp)", offset);
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
                    let offset_str = format!("{}(%rbp)", offset);
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
        if line_i.starts_with("movsd %xmm0, ") && line_i.ends_with("(%rbp)") {
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
    let offset_str = format!("{}(%rbp)", offset);
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
        if t.contains(&offset_str) {
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
                            let xmm = &line_j[18..]; // after "movsd (%rcx), "
                            let mut k = j + 1;
                            while k < len && infos[k].is_nop() { k += 1; }
                            if !rcx_is_live_at(store, infos, k, len) {
                                let new = format!("    movsd (%{}), {}", src_reg, xmm);
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
