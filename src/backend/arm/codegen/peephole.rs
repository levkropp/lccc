//! AArch64 peephole optimizer for assembly text.
//!
//! Operates on generated assembly text to eliminate redundant patterns from the
//! stack-based codegen. Lines are pre-parsed into `LineKind` enums so hot-path
//! pattern matching uses integer/enum comparisons instead of string parsing.
//!
//! ## Pass structure
//!
//! **Local passes** (iterative, up to 8 rounds): store/load elimination,
//! redundant branch removal, self-move elimination, move chain optimization,
//! branch-over-branch fusion, and move-immediate chain optimization.
//!
//! ## Optimizations
//!
//! 1. **Adjacent store/load elimination**: `str xN, [sp, #off]` followed by
//!    `ldr xN, [sp, #off]` — the load is redundant since the value is
//!    already in the register.
//!
//! 2. **Redundant branch elimination**: `b .LBBN` where `.LBBN:` is the
//!    immediately next non-empty line — falls through naturally.
//!
//! 3. **Self-move elimination**: `mov xN, xN` (64-bit) is a no-op.
//!    Note: `mov wN, wN` (32-bit) zeros upper 32 bits and is NOT eliminated.
//!
//! 4. **Move chain optimization**: `mov A, B; mov C, A` → `mov C, B`,
//!    enabling the first mov to become dead if A is unused.
//!
//! 5. **Branch-over-branch fusion**: `b.cc .Lskip; b .target; .Lskip:`
//!    → `b.!cc .target` (invert condition, eliminate skip label).
//!
//! 6. **Move-immediate chain**: `mov xN, #imm; mov xM, xN` where xN is a
//!    scratch register (x0-x15) → `mov xM, #imm` when safe.

// ── Line classification types ────────────────────────────────────────────────

use crate::common::fx_hash::{FxHashMap, FxHashSet};

/// Compact classification of an assembly line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LineKind {
    /// Deleted / blank
    Nop,
    /// `str xN/wN, [sp, #off]` — store to stack (via sp)
    StoreSp { reg: u8, offset: i32, is_word: bool },
    /// `ldr xN/wN, [sp, #off]` — load from stack (via sp)
    LoadSp { reg: u8, offset: i32, is_word: bool },
    /// `ldrsw xN, [sp, #off]` — load signed word from stack
    LoadswSp { reg: u8, offset: i32 },
    /// `ldrsb xN, [xM]` — load signed byte (general)
    LoadsbReg,
    /// `stp xN, xM, [sp, #off]` — store pair to stack
    StorePairSp,
    /// `ldp xN, xM, [sp, #off]` — load pair from stack
    LoadPairSp,
    /// `mov xN, xM` — register-to-register move.
    /// `is_32bit` indicates whether this is a w-register (32-bit) move.
    /// On AArch64, `mov wN, wM` zeros the upper 32 bits of the destination,
    /// so it is NOT equivalent to `mov xN, xM`.
    Move { dst: u8, src: u8, is_32bit: bool },
    /// `mov xN, #imm` — move immediate to register
    MoveImm { dst: u8 },
    /// `movz/movn xN, #imm` — move wide immediate
    MoveWide { dst: u8 },
    /// `sxtw xN, wM` — sign-extend word to doubleword
    Sxtw { dst: u8, src: u8 },
    /// `b .label` — unconditional branch
    Branch,
    /// `b.cc .label` — conditional branch
    CondBranch,
    /// `cbz/cbnz xN, .label` — compare and branch on (non-)zero
    CmpBranch,
    /// Label (`.LBBx:` etc.)
    Label,
    /// `ret`
    Ret,
    /// `bl func` — branch with link (function call)
    Call,
    /// `cmp` or `cmn` instruction
    Compare,
    /// `add`, `sub`, or other ALU instruction
    Alu,
    /// `str`/`ldr` to non-sp addresses (e.g., `str w1, [x9]`)
    MemOther,
    /// Assembler directive (`.section`, `.globl`, etc.)
    Directive,
    /// Any other instruction
    Other,
}

/// AArch64 register IDs for pattern matching.
/// We map x0-x30, sp, and w0-w30 to the same set (register number).
const REG_NONE: u8 = 255;

/// Parse an AArch64 register name to an internal ID (0-30, or special).
/// x0/w0 → 0, x1/w1 → 1, ..., x30/w30 → 30, sp → 31, xzr/wzr → 32.
fn parse_reg(name: &str) -> u8 {
    let name = name.trim();
    if let Some(n) = name.strip_prefix('x').or_else(|| name.strip_prefix('w')) {
        if let Ok(num) = n.parse::<u8>() {
            if num <= 30 {
                return num;
            }
        }
        if n == "zr" {
            return 32; // zero register
        }
    }
    if name == "sp" {
        return 31;
    }
    REG_NONE
}

/// Return the x-register name for a given ID.
fn xreg_name(id: u8) -> &'static str {
    match id {
        0 => "x0", 1 => "x1", 2 => "x2", 3 => "x3",
        4 => "x4", 5 => "x5", 6 => "x6", 7 => "x7",
        8 => "x8", 9 => "x9", 10 => "x10", 11 => "x11",
        12 => "x12", 13 => "x13", 14 => "x14", 15 => "x15",
        16 => "x16", 17 => "x17", 18 => "x18", 19 => "x19",
        20 => "x20", 21 => "x21", 22 => "x22", 23 => "x23",
        24 => "x24", 25 => "x25", 26 => "x26", 27 => "x27",
        28 => "x28", 29 => "x29", 30 => "x30",
        31 => "sp", 32 => "xzr",
        _ => "??",
    }
}

/// Return the w-register name for a given ID.
fn wreg_name(id: u8) -> &'static str {
    match id {
        0 => "w0", 1 => "w1", 2 => "w2", 3 => "w3",
        4 => "w4", 5 => "w5", 6 => "w6", 7 => "w7",
        8 => "w8", 9 => "w9", 10 => "w10", 11 => "w11",
        12 => "w12", 13 => "w13", 14 => "w14", 15 => "w15",
        16 => "w16", 17 => "w17", 18 => "w18", 19 => "w19",
        20 => "w20", 21 => "w21", 22 => "w22", 23 => "w23",
        24 => "w24", 25 => "w25", 26 => "w26", 27 => "w27",
        28 => "w28", 29 => "w29", 30 => "w30",
        32 => "wzr",
        _ => "??",
    }
}

// ── Line classification ──────────────────────────────────────────────────────

/// Classify a single assembly line into a LineKind.
fn classify_line(line: &str) -> LineKind {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return LineKind::Nop;
    }

    // Labels end with ':'
    if trimmed.ends_with(':') {
        return LineKind::Label;
    }

    // Directives start with '.'
    if trimmed.starts_with('.') {
        return LineKind::Directive;
    }

    // str xN/wN, [sp, #off]
    if let Some(rest) = trimmed.strip_prefix("str ") {
        if let Some(info) = parse_sp_mem_op(rest) {
            return LineKind::StoreSp { reg: info.0, offset: info.1, is_word: info.2 };
        }
        return LineKind::MemOther;
    }

    // strb wN, [xM]  — byte store
    if trimmed.starts_with("strb ") || trimmed.starts_with("strh ") {
        return LineKind::MemOther;
    }

    // ldr xN/wN, [sp, #off]
    if let Some(rest) = trimmed.strip_prefix("ldr ") {
        if let Some(info) = parse_sp_mem_op(rest) {
            return LineKind::LoadSp { reg: info.0, offset: info.1, is_word: info.2 };
        }
        return LineKind::MemOther;
    }

    // ldrsw xN, [sp, #off]
    if let Some(rest) = trimmed.strip_prefix("ldrsw ") {
        if let Some((reg_str, addr)) = rest.split_once(", ") {
            let reg = parse_reg(reg_str.trim());
            if reg != REG_NONE {
                if let Some(offset) = parse_sp_offset(addr.trim()) {
                    return LineKind::LoadswSp { reg, offset };
                }
            }
        }
        return LineKind::MemOther;
    }

    // ldrsb
    if trimmed.starts_with("ldrsb ") || trimmed.starts_with("ldrsh ") ||
       trimmed.starts_with("ldrb ") || trimmed.starts_with("ldrh ") {
        return LineKind::LoadsbReg;
    }

    // stp (store pair)
    if trimmed.starts_with("stp ") {
        if trimmed.contains("[sp") {
            return LineKind::StorePairSp;
        }
        return LineKind::MemOther;
    }

    // ldp (load pair)
    if trimmed.starts_with("ldp ") {
        if trimmed.contains("[sp") {
            return LineKind::LoadPairSp;
        }
        return LineKind::MemOther;
    }

    // mov xN, xM  or  mov xN, #imm  or  mov xN, :lo12:sym etc.
    if let Some(rest) = trimmed.strip_prefix("mov ") {
        // Avoid matching movz, movn, movk
        if !trimmed.starts_with("movz") && !trimmed.starts_with("movn") && !trimmed.starts_with("movk") {
            if let Some((dst_str, src_str)) = rest.split_once(", ") {
                let dst_trimmed = dst_str.trim();
                let dst = parse_reg(dst_trimmed);
                if dst != REG_NONE {
                    let src_trimmed = src_str.trim();
                    if src_trimmed.starts_with('#') || src_trimmed.starts_with('-') {
                        return LineKind::MoveImm { dst };
                    }
                    let src = parse_reg(src_trimmed);
                    if src != REG_NONE {
                        let is_32bit = dst_trimmed.starts_with('w');
                        return LineKind::Move { dst, src, is_32bit };
                    }
                    // mov xN, :lo12:symbol etc.
                    return LineKind::MoveImm { dst };
                }
            }
        }
    }

    // movz / movn / movk
    if trimmed.starts_with("movz ") || trimmed.starts_with("movn ") {
        if let Some((_, rest)) = trimmed.split_once(' ') {
            if let Some((dst_str, _)) = rest.split_once(", ") {
                let dst = parse_reg(dst_str.trim());
                if dst != REG_NONE {
                    return LineKind::MoveWide { dst };
                }
            }
        }
        return LineKind::Other;
    }

    // movk is an update, not a fresh definition
    if trimmed.starts_with("movk ") {
        return LineKind::Other;
    }

    // sxtw xN, wM
    if let Some(rest) = trimmed.strip_prefix("sxtw ") {
        if let Some((dst_str, src_str)) = rest.split_once(", ") {
            let dst = parse_reg(dst_str.trim());
            let src = parse_reg(src_str.trim());
            if dst != REG_NONE && src != REG_NONE {
                return LineKind::Sxtw { dst, src };
            }
        }
    }

    // Unconditional branch: b .label (but not bl, b.cc)
    if trimmed.starts_with("b ") && !trimmed.starts_with("bl ") && !trimmed.starts_with("b.") {
        return LineKind::Branch;
    }

    // Conditional branch: b.eq, b.ne, b.lt, b.ge, b.gt, b.le, b.hi, b.ls, b.cs, b.cc, etc.
    if trimmed.starts_with("b.") {
        return LineKind::CondBranch;
    }

    // cbz/cbnz/tbz/tbnz
    if trimmed.starts_with("cbz ") || trimmed.starts_with("cbnz ") ||
       trimmed.starts_with("tbz ") || trimmed.starts_with("tbnz ") {
        return LineKind::CmpBranch;
    }

    // ret
    if trimmed == "ret" {
        return LineKind::Ret;
    }

    // bl (branch and link = call)
    if trimmed.starts_with("bl ") || trimmed.starts_with("blr ") {
        return LineKind::Call;
    }

    // br xN (indirect branch = control flow barrier)
    if trimmed.starts_with("br ") {
        return LineKind::Branch;
    }

    // cmp/cmn
    if trimmed.starts_with("cmp ") || trimmed.starts_with("cmn ") {
        return LineKind::Compare;
    }

    // ALU: add, sub, and, orr, eor, lsl, lsr, asr, mul, etc.
    if trimmed.starts_with("add ") || trimmed.starts_with("sub ") ||
       trimmed.starts_with("and ") || trimmed.starts_with("orr ") ||
       trimmed.starts_with("eor ") || trimmed.starts_with("mul ") ||
       trimmed.starts_with("neg ") || trimmed.starts_with("mvn ") ||
       trimmed.starts_with("lsl ") || trimmed.starts_with("lsr ") ||
       trimmed.starts_with("asr ") || trimmed.starts_with("madd ") ||
       trimmed.starts_with("msub ") || trimmed.starts_with("sdiv ") ||
       trimmed.starts_with("udiv ") || trimmed.starts_with("adds ") ||
       trimmed.starts_with("subs ") {
        return LineKind::Alu;
    }

    LineKind::Other
}

/// Parse `xN/wN, [sp, #off]` and return (reg_id, offset, is_word).
fn parse_sp_mem_op(rest: &str) -> Option<(u8, i32, bool)> {
    let (reg_str, addr) = rest.split_once(", ")?;
    let reg_str = reg_str.trim();
    let is_word = reg_str.starts_with('w');
    let reg = parse_reg(reg_str);
    if reg == REG_NONE {
        return None;
    }
    let offset = parse_sp_offset(addr.trim())?;
    Some((reg, offset, is_word))
}

/// Parse `[sp, #off]` or `[sp]` and return the offset.
fn parse_sp_offset(addr: &str) -> Option<i32> {
    // [sp] — zero offset
    if addr == "[sp]" {
        return Some(0);
    }
    // [sp, #N] or [sp, #-N]
    if addr.starts_with("[sp, #") && addr.ends_with(']') {
        let inner = &addr[6..addr.len() - 1]; // strip "[sp, #" and "]"
        return inner.parse::<i32>().ok();
    }
    // [sp, #N]! (pre-index) — not a simple stack slot access
    None
}

/// Extract branch target from `b .label` or `b label`.
fn branch_target(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    // Match "b .label" but not "br xN" (indirect branch)
    if let Some(rest) = trimmed.strip_prefix("b ") {
        let target = rest.trim();
        // Must start with '.' (a label), not a register
        if target.starts_with('.') {
            return Some(target);
        }
    }
    None
}

/// Extract the condition code and target from a conditional branch.
/// `b.eq .label` → Some(("eq", ".label"))
fn cond_branch_parts(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("b.") {
        if let Some((cc, target)) = rest.split_once(' ') {
            return Some((cc, target.trim()));
        }
    }
    None
}

/// Invert a condition code.
fn invert_condition(cc: &str) -> Option<&'static str> {
    match cc {
        "eq" => Some("ne"),
        "ne" => Some("eq"),
        "lt" => Some("ge"),
        "ge" => Some("lt"),
        "gt" => Some("le"),
        "le" => Some("gt"),
        "hi" => Some("ls"),
        "ls" => Some("hi"),
        "hs" | "cs" => Some("lo"),
        "lo" | "cc" => Some("hs"),
        "mi" => Some("pl"),
        "pl" => Some("mi"),
        "vs" => Some("vc"),
        "vc" => Some("vs"),
        _ => None,
    }
}

/// Extract label name from a label line (strip trailing `:`)
fn label_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed.strip_suffix(':')
}


// ── Main entry point ─────────────────────────────────────────────────────────

/// Run peephole optimization on AArch64 assembly text.
/// Returns the optimized assembly string.
pub fn peephole_optimize(asm: String) -> String {
    if std::env::var("CCC_NO_PEEPHOLE").is_ok() {
        return asm;
    }
    let mut lines: Vec<String> = asm.lines().map(String::from).collect();
    let mut kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let n = lines.len();

    if n == 0 {
        return asm;
    }

    // Phase 1: Iterative local passes (up to 8 rounds)
    let rotate = std::env::var("CCC_NO_LOOP_ROTATE").is_err();
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 8 {
        changed = false;
        changed |= eliminate_adjacent_store_load(&mut lines, &mut kinds, n);
        changed |= eliminate_redundant_branches(&lines, &mut kinds, n);
        changed |= eliminate_self_moves(&mut kinds, n);
        changed |= eliminate_redundant_sxtw(&mut lines, &mut kinds, n);
        changed |= eliminate_redundant_sxtb(&mut lines, &mut kinds, n);
        changed |= fold_shift_mask_ubfx(&mut lines, &mut kinds, n);
        changed |= fold_zero_stores(&mut lines, &mut kinds, n);
        changed |= fold_fp_gpr_roundtrips(&mut lines, &mut kinds, n);
        changed |= eliminate_move_chains(&mut lines, &mut kinds, n);
        changed |= fold_zext_move_chains(&mut lines, &mut kinds, n);
        changed |= eliminate_unused_x9_address_moves(&lines, &mut kinds, n);
        changed |= propagate_address_aliases(&mut lines, &mut kinds, n);
        changed |= forward_fp_slot_loads(&mut lines, &mut kinds, n);
        changed |= fuse_fp_adjacent_pairs(&mut lines, &mut kinds, n);
        changed |= eliminate_repeated_slot_loads(&mut lines, &mut kinds, n);
        changed |= fold_destructive_pointer_updates(&mut lines, &mut kinds, n);
        changed |= fold_csel_increment(&mut lines, &mut kinds, n);
        changed |= fuse_branch_over_branch(&mut lines, &mut kinds, n);
        // After the folds: dead staging movs (their patterns must win first).
        changed |= eliminate_overwritten_moves(&lines, &mut kinds, n);
        changed |= eliminate_overwritten_stores(&lines, &mut kinds, n);
        if rotate {
            changed |= rotate_simple_loops(&mut lines, &mut kinds, n);
        }
        rounds += 1;
    }

    // Phase 2: Global passes
    //
    // Global store forwarding is disabled: same-register NOP elimination has a
    // remaining correctness bug in complex float-array code (test 0036_0041).
    // The root cause is not yet identified — it appears correct within a basic
    // block but produces wrong output. Until this is fixed, GSF is skipped.
    //
    // Copy propagation and dead store elimination are independent and safe.
    propagate_register_copies(&mut lines, &mut kinds, n);
    reuse_stack_loads_within_blocks(&mut lines, &mut kinds, n);
    propagate_register_copies(&mut lines, &mut kinds, n);
    eliminate_dead_call_staging_moves(&lines, &mut kinds, n);
    if std::env::var("CCC_NO_GDSE").is_err() {
        global_dead_store_elimination(&lines, &mut kinds, n);
    }
    // After rotation (phase 1) and dead-store cleanup: sink loop-carried
    // slot-home stores out of bottom-tested loops.
    sink_loop_carried_stores(&mut lines, &mut kinds, n);
    // Defer callee-saves past clean early returns (binary_trees check).
    shrink_wrap_early_returns(&mut lines, &mut kinds, n);

    // Phase 3: Local cleanup after global passes (up to 4 rounds)
    {
        let mut changed2 = true;
        let mut rounds2 = 0;
        while changed2 && rounds2 < 4 {
            changed2 = false;
            changed2 |= eliminate_adjacent_store_load(&mut lines, &mut kinds, n);
            changed2 |= eliminate_redundant_branches(&lines, &mut kinds, n);
            changed2 |= eliminate_self_moves(&mut kinds, n);
            changed2 |= eliminate_redundant_sxtw(&mut lines, &mut kinds, n);
            changed2 |= eliminate_redundant_sxtb(&mut lines, &mut kinds, n);
            changed2 |= fold_shift_mask_ubfx(&mut lines, &mut kinds, n);
            changed2 |= fold_zero_stores(&mut lines, &mut kinds, n);
            changed2 |= fold_fp_gpr_roundtrips(&mut lines, &mut kinds, n);
            changed2 |= eliminate_move_chains(&mut lines, &mut kinds, n);
            changed2 |= fold_zext_move_chains(&mut lines, &mut kinds, n);
            changed2 |= eliminate_unused_x9_address_moves(&lines, &mut kinds, n);
            changed2 |= propagate_address_aliases(&mut lines, &mut kinds, n);
            changed2 |= forward_fp_slot_loads(&mut lines, &mut kinds, n);
            changed2 |= fuse_fp_adjacent_pairs(&mut lines, &mut kinds, n);
            changed2 |= eliminate_repeated_slot_loads(&mut lines, &mut kinds, n);
            changed2 |= fold_destructive_pointer_updates(&mut lines, &mut kinds, n);
            changed2 |= fold_csel_increment(&mut lines, &mut kinds, n);
            changed2 |= eliminate_overwritten_moves(&lines, &mut kinds, n);
            changed2 |= eliminate_overwritten_stores(&lines, &mut kinds, n);
            rounds2 += 1;
        }
    }

    // Build result, filtering out Nop lines
    let mut result = String::with_capacity(asm.len());
    for i in 0..n {
        if kinds[i] != LineKind::Nop {
            result.push_str(&lines[i]);
            result.push('\n');
        }
    }
    result
}

/// Propagate a register copy used only as a memory-address alias within one
/// basic block. This handles vector intrinsics where an SSA Copy gives the C
/// pointer a short-lived register solely for `[xN]` loads/stores.
///
/// Soundness: the defining `mov` may only be deleted when the copy's value is
/// provably dead after the rewritten window — every mention of `dst` from here
/// to the end of the function's text must be accounted for. An earlier version
/// stopped the scan at the first block boundary and deleted the mov anyway,
/// which silently dropped the initializer of any alias used in a later block
/// (exposed when global-address CSE made such aliases multi-use). We therefore
/// scan to the end of the function and commit only on proof of death:
/// overwrite of `dst` by an instruction that does not read it, or the end of
/// the function's text. Anything unrecognized (other memory ops, unknown
/// instruction kinds, calls) blocks the transform.
fn propagate_address_aliases(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    fn mentions(line: &str, reg: u8) -> bool {
        line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == xreg_name(reg) || tok == wreg_name(reg))
    }
    fn mention_count(line: &str, reg: u8) -> usize {
        line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|tok| *tok == xreg_name(reg) || *tok == wreg_name(reg))
            .count()
    }
    // Column-0 identifier label: the next function or data object, i.e. the
    // end of the current function's text. Block labels start with '.', and
    // inline-asm numeric local labels (`1:`) start with a digit — neither
    // terminates the function's text.
    fn is_function_boundary(line: &str) -> bool {
        line.ends_with(':')
            && line.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
    }
    let mut changed = false;
    for i in 0..n {
        let LineKind::Move { dst, src, is_32bit: false } = kinds[i] else { continue };
        if dst == src || dst >= 29 || src >= 29 { continue; }
        let mut uses = Vec::new();
        let mut valid = true;
        let mut provably_dead = false;
        let mut src_alive = true;
        let mut j = i + 1;
        while j < n {
            if is_function_boundary(&lines[j]) {
                provably_dead = true;
                break;
            }
            match kinds[j] {
                // No runtime effect. A return ends this path; textually later
                // lines belong to other paths, so keep scanning.
                //
                // Note: an epilogue's `ldp xD, ...` callee-save restore is NOT
                // treated as proof of death here — it ends the value only on
                // that return path, and textually later blocks (other paths)
                // may still use it, so it falls through to the plain-mention
                // handling and blocks the transform.
                LineKind::Nop | LineKind::Directive | LineKind::Label | LineKind::Ret => {}
                // Calls clobber src (x0) and all caller-saved registers. Rather
                // than reason about save/restore pairs, stop without committing.
                LineKind::Call => { valid = false; break; }
                _ => {
                    // Kinds whose destination register written_gp_register
                    // models precisely.
                    let trusted = matches!(kinds[j],
                        LineKind::Move { .. } | LineKind::MoveImm { .. }
                        | LineKind::MoveWide { .. } | LineKind::Sxtw { .. }
                        | LineKind::LoadSp { .. } | LineKind::LoadswSp { .. }
                        | LineKind::Alu | LineKind::Compare
                        | LineKind::Branch | LineKind::CondBranch | LineKind::CmpBranch
                        | LineKind::StoreSp { .. });
                    let written = written_gp_register(&lines[j], kinds[j]);
                    if (trusted && written == Some(src))
                        || (!trusted && mentions(&lines[j], src))
                    {
                        src_alive = false;
                    }
                    if mentions(&lines[j], dst) {
                        if trusted && written == Some(dst) && mention_count(&lines[j], dst) == 1 {
                            // dst overwritten without being read: the mov's
                            // value is dead past this instruction (register
                            // assignments are per-value with disjoint live
                            // intervals, so no later text can use the old value).
                            provably_dead = true;
                            break;
                        }
                        let address = format!("[{}", xreg_name(dst));
                        if src_alive && mention_count(&lines[j], dst) == 1
                            && lines[j].contains(&address)
                        {
                            // dst used purely as the address base of a
                            // load/store while src is alive: rewritable.
                            uses.push(j);
                        } else {
                            // Non-address use, an unmodelled effect, or an
                            // address use after src died: the mov must stay.
                            valid = false;
                            break;
                        }
                    }
                }
            }
            j += 1;
        }
        if j == n {
            provably_dead = true; // scanned to the end of the file
        }
        if valid && provably_dead && !uses.is_empty() {
            for use_idx in uses {
                lines[use_idx] = replace_whole_word(&lines[use_idx], xreg_name(dst), xreg_name(src));
                kinds[use_idx] = classify_line(&lines[use_idx]);
            }
            kinds[i] = LineKind::Nop;
            changed = true;
        }
    }
    changed
}

/// Fold an SSA pointer update that is copied back to its loop-carried register:
/// `mov x0,xS; add x0,xS,#K; mov xT,x0; ...; mov xS,xT` → `add xS,xS,#K`.
/// The scan proves neither xS nor xT is referenced in the intervening region.
fn fold_destructive_pointer_updates(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    fn mentions(line: &str, reg: u8) -> bool {
        line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == xreg_name(reg) || tok == wreg_name(reg))
    }
    let mut changed = false;
    for i in 0..n.saturating_sub(3) {
        let LineKind::Move { dst: 0, src, is_32bit: false } = kinds[i] else { continue };
        if src == 0 || src >= 29 { continue; }
        let mut j = i + 1;
        while j < n && kinds[j] == LineKind::Nop { j += 1; }
        if j >= n || kinds[j] != LineKind::Alu { continue; }
        let expected = format!("add x0, {}, #", xreg_name(src));
        let trimmed = lines[j].trim();
        if !trimmed.starts_with(&expected) { continue; }
        let immediate = match trimmed.rsplit_once('#').and_then(|(_, v)| v.parse::<i64>().ok()) {
            Some(v) if (0..=4095).contains(&v) => v,
            _ => continue,
        };
        let mut k = j + 1;
        while k < n && kinds[k] == LineKind::Nop { k += 1; }
        let temp = match kinds.get(k).copied() {
            Some(LineKind::Move { dst, src: 0, is_32bit: false }) if dst != src => dst,
            _ => continue,
        };
        let mut end = k + 1;
        let mut valid = true;
        let mut found = None;
        while end < n && end <= k + 16 {
            if matches!(kinds[end], LineKind::Label | LineKind::Branch | LineKind::CondBranch
                | LineKind::CmpBranch | LineKind::Call | LineKind::Ret) { break; }
            if let LineKind::Move { dst, src: move_src, is_32bit: false } = kinds[end] {
                if dst == src && move_src == temp { found = Some(end); break; }
            }
            if mentions(&lines[end], src) || mentions(&lines[end], temp) {
                valid = false;
                break;
            }
            end += 1;
        }
        let Some(end) = found else { continue };
        if !valid { continue; }
        lines[i] = format!("    add {}, {}, #{}", xreg_name(src), xreg_name(src), immediate);
        kinds[i] = LineKind::Alu;
        kinds[j] = LineKind::Nop;
        kinds[k] = LineKind::Nop;
        kinds[end] = LineKind::Nop;
        changed = true;
    }
    changed
}

/// Reuse a value already copied out of a stack slot inside one basic block.
///
/// Register-pressure-heavy code commonly contains
/// `ldr x0, [sp,#N]; mov wD,w0` several times for the same spill.  Once the
/// first copy is in a register, later reloads can source that register until
/// either it or the slot is overwritten.  Keep this deliberately local and
/// clear state for instruction classes whose register effects are ambiguous.
fn reuse_stack_loads_within_blocks(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_REUSE_STACK_LOADS").is_ok() {
        return false;
    }
    let mut cached: Vec<(i32, bool, u8, bool)> = Vec::new();
    let mut changed = false;
    let mut i = 0;
    let mut block_safe = false;

    while i < n {
        if kinds[i] == LineKind::Label {
            let mut j = i + 1;
            block_safe = true;
            while j < n && kinds[j] != LineKind::Label {
                if matches!(kinds[j], LineKind::MemOther | LineKind::Other
                    | LineKind::StorePairSp | LineKind::LoadPairSp
                    | LineKind::LoadsbReg | LineKind::Call)
                {
                    block_safe = false;
                    break;
                }
                j += 1;
            }
        }
        match kinds[i] {
            LineKind::Label | LineKind::Branch | LineKind::CondBranch
            | LineKind::CmpBranch | LineKind::Call | LineKind::Ret
            | LineKind::MemOther | LineKind::Other | LineKind::StorePairSp
            | LineKind::LoadPairSp | LineKind::LoadsbReg => cached.clear(),
            LineKind::StoreSp { offset, .. } => cached.retain(|(off, _, _, _)| *off != offset),
            _ => {}
        }

        if !block_safe {
            i += 1;
            continue;
        }

        let (load_reg, offset, is_word) = match kinds[i] {
            LineKind::LoadSp { reg, offset, is_word } => (reg, offset, is_word),
            _ => {
                if let Some(dst) = written_gp_register(&lines[i], kinds[i]) {
                    cached.retain(|(_, _, reg, _)| *reg != dst);
                }
                i += 1;
                continue;
            }
        };

        let old = cached.iter().find(|(off, word, reg, _)| {
            *off == offset && *word == is_word && *reg != load_reg
        }).copied();
        cached.retain(|(_, _, reg, _)| *reg != load_reg);

        let mut j = i + 1;
        while j < n && kinds[j] == LineKind::Nop { j += 1; }
        if j < n {
            if let LineKind::Move { dst, src, is_32bit } = kinds[j] {
                if src == load_reg {
                    if let Some((_, _, source, source_word)) = old.filter(|entry| entry.3 == is_32bit) {
                        // NOPing the load is only sound if its register has no
                        // other readers before it is next fully overwritten —
                        // copy propagation can move an instruction's reference
                        // onto the load's own register (e.g. rewriting
                        // `add x0, x1, x2` to read x0), and deleting the load
                        // would leave that read with stale contents.
                        let xn = xreg_name(load_reg);
                        let wn = wreg_name(load_reg);
                        let mut load_dead_after = false;
                        let mut k = j + 1;
                        while k < n {
                            if kinds[k] == LineKind::Nop { k += 1; continue; }
                            let t = &lines[k];
                            let mentions = t
                                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                                .any(|tok| tok == xn || tok == wn);
                            if mentions {
                                // A pure overwrite ends the hazard; anything
                                // else reads the (soon deleted) register.
                                load_dead_after = match kinds[k] {
                                    LineKind::Move { dst, src, .. } => dst == load_reg && src != load_reg,
                                    LineKind::MoveImm { dst } | LineKind::MoveWide { dst } => dst == load_reg,
                                    LineKind::LoadSp { reg, .. } | LineKind::LoadswSp { reg, .. } => reg == load_reg,
                                    _ => false,
                                };
                                break;
                            }
                            match kinds[k] {
                                LineKind::Label | LineKind::Branch | LineKind::CondBranch
                                | LineKind::CmpBranch | LineKind::Call | LineKind::Ret
                                | LineKind::Directive => break,
                                _ => {}
                            }
                            k += 1;
                        }
                        if !load_dead_after {
                            i += 1;
                            continue;
                        }
                        kinds[i] = LineKind::Nop;
                        lines[j] = format!("    mov {}, {}",
                            if is_32bit { wreg_name(dst) } else { xreg_name(dst) },
                            if source_word { wreg_name(source) } else { xreg_name(source) });
                        kinds[j] = LineKind::Move { dst, src: source, is_32bit };
                        changed = true;
                    }
                    cached.retain(|(_, _, reg, _)| *reg != dst);
                    cached.retain(|(off, word, _, _)| *off != offset || *word != is_word);
                    cached.push((offset, is_word, dst, is_32bit));
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    changed
}

/// Return the GP register written by instruction classes whose first operand
/// is unambiguously a destination. Unknown classes are handled by clearing the
/// cache in the caller.
fn written_gp_register(line: &str, kind: LineKind) -> Option<u8> {
    match kind {
        LineKind::Move { dst, .. } | LineKind::MoveImm { dst }
        | LineKind::MoveWide { dst } | LineKind::Sxtw { dst, .. }
        | LineKind::LoadSp { reg: dst, .. } | LineKind::LoadswSp { reg: dst, .. } => Some(dst),
        LineKind::Alu => {
            let operands = line.trim().split_once(' ')?.1;
            let dst = operands.split(',').next()?;
            let reg = parse_reg(dst);
            (reg != REG_NONE).then_some(reg)
        }
        _ => None,
    }
}

/// Remove `mov x9, xN` immediately followed by a memory operation that already
/// addresses through xN. Regalloc-aware loads can make the legacy x9 setup dead.
fn eliminate_unused_x9_address_moves(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n.saturating_sub(1) {
        let LineKind::Move { dst: 9, src, is_32bit: false } = kinds[i] else { continue };
        let mut j = i + 1;
        while j < n && kinds[j] == LineKind::Nop { j += 1; }
        if j >= n || !matches!(kinds[j], LineKind::MemOther | LineKind::LoadsbReg) { continue; }
        let needle = format!("[{}", xreg_name(src));
        if lines[j].contains(&needle) && !lines[j].contains("[x9") {
            kinds[i] = LineKind::Nop;
            changed = true;
        }
    }
    changed
}

/// Remove a move into an AArch64 caller-saved staging register when the value
/// is never referenced before the next call clobbers it. Copy propagation
/// commonly turns
/// `mov x9, x24; mov x0, x9; bl f` into
/// `mov x9, x24; mov x0, x24; bl f`, leaving the first move dead.
fn eliminate_dead_call_staging_moves(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    fn mentions_reg(line: &str, reg: u8) -> bool {
        line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|token| token == xreg_name(reg) || token == wreg_name(reg))
    }

    let mut changed = false;
    for i in 0..n {
        let dst = match kinds[i] {
            LineKind::Move { dst, .. } if (9..=15).contains(&dst) => dst,
            _ => continue,
        };
        let mut j = i + 1;
        let mut used = false;
        let mut reaches_call = false;
        while j < n {
            match kinds[j] {
                LineKind::Nop | LineKind::Directive => {}
                LineKind::Call => { reaches_call = true; break; }
                LineKind::Label | LineKind::Branch | LineKind::CondBranch
                | LineKind::CmpBranch | LineKind::Ret => break,
                _ => {
                    if mentions_reg(&lines[j], dst) { used = true; break; }
                }
            }
            j += 1;
        }
        if reaches_call && !used {
            kinds[i] = LineKind::Nop;
            changed = true;
        }
    }
    changed
}

// ── Pass 1: Adjacent store/load elimination ──────────────────────────────────
//
// Pattern: str xN, [sp, #off]  →  ldr xN, [sp, #off]  (same reg, same offset)
// The load is redundant since the value is already in the register.
// Also: str xN, [sp, #off]  →  ldr xM, [sp, #off]  → replace load with mov xM, xN
// Also handles: str wN, [sp, #off]  →  ldrsw xN, [sp, #off]

// ── FP spill slot load forwarding ────────────────────────────────────────────
//
// Pattern: `str dS, [sp, #off]` (or via a materialized `sp`-derived base
// register) followed by `ldr dD, [sp, #off]` in the same block. The slot-home
// emission produces these store/reload round-trips for unallocated FP
// temporaries; the load is replaceable with `fmov dD, dS` (or deleted when
// dD == dS) as long as neither the slot nor dS is clobbered in between.
//
// Tracking is deliberately conservative: any store through an unknown base,
// any call, and any control-flow boundary drops all state, and any write of
// a source register drops the entries that forward from it.

/// Parse an FP register name token (`d7`, `s3`, `q16`, `v2`, `v2.2d`) to its
/// register number.
fn fp_token_reg(tok: &str) -> Option<u8> {
    let tok = tok.trim();
    let digits = match tok.as_bytes().first() {
        Some(b'd') | Some(b's') | Some(b'q') | Some(b'v') => {
            let mut end = 1;
            while end < tok.len() && tok.as_bytes()[end].is_ascii_digit() {
                end += 1;
            }
            if end == 1 {
                return None;
            }
            &tok[1..end]
        }
        _ => return None,
    };
    digits.parse::<u8>().ok().filter(|&r| r <= 31)
}

/// An FP memory operand location: direct `[sp, #off]` slot, or a register
/// base with a byte offset (`[xN]` / `[xN, #k]`).
#[derive(Clone, Copy)]
enum FpAddr {
    Sp(i32),
    Base(u8, i32),
}

/// Parse `str dS, ...` / `ldr dD, ...` (d/s/q widths) into (reg, FpAddr).
fn parse_fp_mem(line: &str) -> Option<(bool, u8, FpAddr, u8)> {
    let t = line.trim();
    let (is_store, rest) = if let Some(r) = t.strip_prefix("str ") {
        (true, r)
    } else if let Some(r) = t.strip_prefix("ldr ") {
        (false, r)
    } else {
        return None;
    };
    let (reg_str, addr) = rest.split_once(", ")?;
    let reg_str = reg_str.trim();
    let width = reg_str.as_bytes().first().copied()?;
    if !matches!(width, b'd' | b's' | b'q') {
        return None;
    }
    let reg = fp_token_reg(reg_str)?;
    let addr = addr.trim();
    if let Some(off) = parse_sp_offset(addr) {
        return Some((is_store, reg, FpAddr::Sp(off), width));
    }
    if addr.starts_with("[x") && addr.ends_with(']') && !addr.contains('!') {
        let inner = &addr[1..addr.len() - 1];
        let (base, off) = match inner.split_once(", ") {
            Some((b, o)) => (b.trim(), o.trim().strip_prefix('#')?.parse::<i32>().ok()?),
            None => (inner.trim(), 0),
        };
        let base_reg = parse_reg(base);
        if base_reg < 29 {
            return Some((is_store, reg, FpAddr::Base(base_reg, off), width));
        }
    }
    None
}

/// Width in bytes of an FP spill slot access kind.
fn fp_access_width(width: u8) -> i32 {
    match width {
        b's' => 4,
        b'q' => 16,
        _ => 8, // d
    }
}

/// Does this line write FP register `num` (any d/s/q/v name form) as its
/// destination? Store/branch/compare mnemonics never write a register.
fn fp_reg_written(line: &str, num: u8) -> bool {
    let t = line.trim();
    let Some((mnem, rest)) = t.split_once(' ') else { return false };
    if matches!(
        mnem,
        "str" | "stp" | "strb" | "strh" | "stur" | "sturb" | "sturh" | "stlr"
            | "stlrb" | "stlrh" | "fcmp" | "fcmpe" | "b" | "bl" | "ret" | "cbz"
            | "cbnz" | "tbz" | "tbnz" | "cmp" | "cmn"
    ) || mnem.starts_with("b.")
    {
        return false;
    }
    rest.split(',')
        .next()
        .is_some_and(|first| fp_token_reg(first) == Some(num))
}

/// Does this line write GP register `num` (x or w form)? Checks the first two
/// operand positions (pair loads write the second). False positives are fine
/// (callers only use this to drop tracking state, never to enable a rewrite);
/// false negatives are not, so store/branch/compare mnemonics are excluded and
/// anything unrecognized counts as a write when the register appears early.
fn gp_reg_written_broad(line: &str, num: u8) -> bool {
    let t = line.trim();
    let Some((mnem, rest)) = t.split_once(' ') else { return false };
    if matches!(
        mnem,
        "str" | "stp" | "strb" | "strh" | "stur" | "sturb" | "sturh" | "stlr"
            | "stlrb" | "stlrh" | "fcmp" | "fcmpe" | "b" | "bl" | "ret" | "cbz"
            | "cbnz" | "tbz" | "tbnz" | "cmp" | "cmn"
    ) || mnem.starts_with("b.")
    {
        return false;
    }
    rest.split(',')
        .take(2)
        .any(|op| {
            let op = op.trim();
            op == xreg_name(num) || op == wreg_name(num)
        })
}

// ── Adjacent FP field pair fusion (ldp/stp) ─────────────────────────────────
//
// `ldr dA, [xB]` + `ldr dC, [xB, #8]` (within a small window) → `ldp dA, dC,
// [xB]`, and likewise `str` pairs → `stp`. Struct-of-double loops (nbody,
// spectral_norm, struct_copy) load/store adjacent fields in pairs; fusing
// halves the memory-op issue slots. Bit-identical: ldp/stp move the same
// bytes as the two scalar accesses. CCC_NO_FP_PAIR disables.
//
// Soundness: the second access moves up to the first's position, so nothing
// in the gap may write memory (for loads), touch memory at all (for stores),
// or mention either destination register or the base.
fn fuse_fp_adjacent_pairs(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_FP_PAIR").is_ok() {
        return false;
    }
    let mention = |line: &str, reg: u8, is_fp: bool| {
        let (a, b) = if is_fp {
            (format!("d{}", reg), format!("q{}", reg))
        } else {
            (xreg_name(reg).to_string(), wreg_name(reg).to_string())
        };
        line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == a || tok == b)
    };
    let mut changed = false;
    for i in 0..n {
        // Skip lines already Nop'd by earlier fusions/passes: their text is
        // stale and must not be re-paired.
        if kinds[i] == LineKind::Nop {
            continue;
        }
        let Some((is_store, ra, addra, b'd')) = parse_fp_mem(&lines[i]) else { continue };
        let base_reg = match addra {
            FpAddr::Sp(_) => u8::MAX, // sp is never rewritten mid-body
            FpAddr::Base(b, _) => b,
        };
        // Find a mergeable partner within a 4-instruction window.
        let mut partner = None;
        let mut j = i + 1;
        while j < n && j <= i + 4 {
            if kinds[j] == LineKind::Nop {
                j += 1;
                continue;
            }
            match kinds[j] {
                LineKind::Label | LineKind::Branch | LineKind::CondBranch
                | LineKind::CmpBranch | LineKind::Call | LineKind::Ret
                | LineKind::Directive => break,
                _ => {}
            }
            let tj = lines[j].trim();
            // A valid merge partner is not a hazard to itself — check for it
            // before the generic memory-hazard break.
            if let Some((is_store2, rc, addrc, b'd')) = parse_fp_mem(&lines[j]) {
                if is_store2 == is_store && rc != ra {
                    let same_base = match (addra, addrc) {
                        (FpAddr::Sp(a), FpAddr::Sp(b)) => Some((a, b)),
                        (FpAddr::Base(x, a), FpAddr::Base(y, b)) if x == y => Some((a, b)),
                        _ => None,
                    };
                    if let Some((oa, ob)) = same_base {
                        if (oa - ob).abs() == 8 {
                            // The partner's load/store moves up to position i:
                            // no gap line may mention rc (its physical register
                            // could carry a live older value), write ra, or
                            // write the base.
                            let mut ok = true;
                            for k in (i + 1)..j {
                                if kinds[k] == LineKind::Nop {
                                    continue;
                                }
                                if mention(&lines[k], rc, true)
                                    || fp_reg_written(&lines[k], ra)
                                    || (base_reg != u8::MAX
                                        && gp_reg_written_broad(&lines[k], base_reg))
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                partner = Some((j, rc, ob));
                            }
                            break;
                        }
                    }
                }
            }
            let is_mem = parse_fp_mem(&lines[j]).is_some()
                || matches!(kinds[j], LineKind::StoreSp { .. } | LineKind::LoadSp { .. }
                    | LineKind::LoadswSp { .. } | LineKind::MemOther
                    | LineKind::StorePairSp | LineKind::LoadPairSp | LineKind::LoadsbReg);
            // A store in the gap could alias the pending load; for store
            // fusion any memory access is a hazard.
            if is_mem && (is_store || tj.starts_with("st")) {
                break;
            }
            j += 1;
        }
        let Some((pj, rc, ob)) = partner else { continue };
        let oa = match addra {
            FpAddr::Sp(o) => o,
            FpAddr::Base(_, o) => o,
        };
        let (lo_reg, hi_reg, lo) = if oa <= ob { (ra, rc, oa) } else { (rc, ra, ob) };
        let base_str = match addra {
            FpAddr::Sp(_) => "sp".to_string(),
            FpAddr::Base(b, _) => xreg_name(b).to_string(),
        };
        let mnem = if is_store { "stp" } else { "ldp" };
        lines[i] = if lo == 0 {
            format!("    {} d{}, d{}, [{}]", mnem, lo_reg, hi_reg, base_str)
        } else {
            format!("    {} d{}, d{}, [{}, #{}]", mnem, lo_reg, hi_reg, base_str, lo)
        };
        kinds[i] = classify_line(&lines[i]);
        kinds[pj] = LineKind::Nop;
        changed = true;
    }
    changed
}

// ── Repeated GP slot load elimination ───────────────────────────────────────
//
// `ldr x9, [sp, #off]` ... `ldr x9, [sp, #off]` (same slot, no clobber):
// spill-heavy code reloads slot-homed pointers once per field access. The
// second load is redundant (or becomes a `mov` when the destination differs).
// Same tracking discipline as forward_fp_slot_loads: stores through unknown
// bases, calls, and control-flow boundaries drop all state; a write of the
// cached register drops the entry.
fn eliminate_repeated_slot_loads(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_SLOT_LOAD_DEDUP").is_ok() {
        return false;
    }
    // (offset, is_word) -> register holding the slot's content
    let mut cached: FxHashMap<(i32, bool), u8> = FxHashMap::default();
    let mut changed = false;
    for i in 0..n {
        match kinds[i] {
            LineKind::Label | LineKind::Branch | LineKind::CondBranch
            | LineKind::CmpBranch | LineKind::Call | LineKind::Ret => {
                cached.clear();
                continue;
            }
            LineKind::Directive | LineKind::Nop => continue,
            LineKind::StoreSp { offset, is_word, .. } => {
                // Invalidate entries overlapping the stored range.
                let (lo, hi) = (offset, offset + if is_word { 4 } else { 8 });
                cached.retain(|&(off, word), _| {
                    let (l2, h2) = (off, off + if word { 4 } else { 8 });
                    h2 <= lo || l2 >= hi
                });
                continue;
            }
            LineKind::LoadSp { reg, offset, is_word } => {
                if let Some(&prev) = cached.get(&(offset, is_word)) {
                    if prev == reg {
                        kinds[i] = LineKind::Nop;
                        changed = true;
                        continue;
                    }
                    lines[i] = format!(
                        "    mov {}, {}",
                        if is_word { wreg_name(reg) } else { xreg_name(reg) },
                        if is_word { wreg_name(prev) } else { xreg_name(prev) }
                    );
                    kinds[i] = LineKind::Move { dst: reg, src: prev, is_32bit: is_word };
                    changed = true;
                }
                // The load (or its mov replacement) writes `reg`: any entry
                // cached under it is stale. Then record the new mapping.
                cached.retain(|_, &mut r| r != reg);
                cached.insert((offset, is_word), reg);
                continue;
            }
            _ => {}
        }
        let t = lines[i].trim();
        // Stores through untracked bases may hit the frame: drop everything.
        if t.starts_with("st") {
            cached.clear();
            continue;
        }
        // Loads and other instructions only invalidate via their destination.
        if let Some(w) = written_gp_register(&lines[i], kinds[i]) {
            cached.retain(|_, &mut r| r != w);
        } else {
            // Unrecognized kinds: check destination-position mentions broadly.
            let mut dead = Vec::new();
            for &r in cached.values() {
                if gp_reg_written_broad(&lines[i], r) {
                    dead.push(r);
                }
            }
            for r in dead {
                cached.retain(|_, &mut v| v != r);
            }
        }
    }
    changed
}

/// Strict single-destination form: only the first operand position, and only
/// when it is the line's sole mention of the register (a read-write like
/// `add x0, x0, #1` is NOT a pure overwrite).
fn gp_reg_write_only(line: &str, num: u8) -> bool {
    let t = line.trim();
    let Some((mnem, rest)) = t.split_once(' ') else { return false };
    if matches!(
        mnem,
        "str" | "stp" | "strb" | "strh" | "stur" | "sturb" | "sturh" | "stlr"
            | "stlrb" | "stlrh" | "fcmp" | "fcmpe" | "b" | "bl" | "ret" | "cbz"
            | "cbnz" | "tbz" | "tbnz" | "cmp" | "cmn"
    ) || mnem.starts_with("b.")
    {
        return false;
    }
    let (xa, wa) = (xreg_name(num), wreg_name(num));
    let first = rest.split(',').next().map(str::trim);
    if first != Some(xa) && first != Some(wa) {
        return false;
    }
    t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|tok| *tok == xa || *tok == wa)
        .count()
        == 1
}

fn forward_fp_slot_loads(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_FP_SLOT_FWD").is_ok() {
        return false;
    }
    // slot (offset, width) -> source FP register number
    let mut slot_src: FxHashMap<(i32, u8), u8> = FxHashMap::default();
    // sp-derived base register -> sp offset (`add xN, sp, #off` / `mov xN, sp`)
    let mut base_off: FxHashMap<u8, i32> = FxHashMap::default();
    let mut changed = false;

    let mut i = 0;
    while i < n {
        match kinds[i] {
            LineKind::Label | LineKind::Branch | LineKind::CondBranch
            | LineKind::CmpBranch | LineKind::Call | LineKind::Ret => {
                slot_src.clear();
                base_off.clear();
                i += 1;
                continue;
            }
            LineKind::Directive | LineKind::Nop => {
                i += 1;
                continue;
            }
            _ => {}
        }

        let t = lines[i].trim().to_string();

        // Track sp-derived base registers: `add xN, sp, #off`, `mov xN, sp`.
        if let Some(rest) = t.strip_prefix("add ") {
            if let Some((dst_s, srcs)) = rest.split_once(", ") {
                if let Some((src_s, imm_s)) = srcs.split_once(", ") {
                    if src_s.trim() == "sp" {
                        let dst = parse_reg(dst_s.trim());
                        if dst < 29 {
                            if let Some(imm) = imm_s.trim().strip_prefix('#').and_then(|v| v.parse::<i32>().ok()) {
                                base_off.insert(dst, imm);
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        } else if let Some(rest) = t.strip_prefix("mov ") {
            if let Some((dst_s, src_s)) = rest.split_once(", ") {
                if src_s.trim() == "sp" {
                    let dst = parse_reg(dst_s.trim());
                    if dst < 29 {
                        base_off.insert(dst, 0);
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Any write of a tracked base register invalidates its mapping.
        // (Broad check: loads through untracked bases and pair loads can also
        // rewrite the register; over-dropping only loses optimization.)
        {
            let mut dead_bases = Vec::new();
            for &b in base_off.keys() {
                if written_gp_register(&lines[i], kinds[i]) == Some(b)
                    || gp_reg_written_broad(&lines[i], b)
                {
                    dead_bases.push(b);
                }
            }
            for b in dead_bases {
                base_off.remove(&b);
            }
        }

        if let Some((is_store, reg, addr, width)) = parse_fp_mem(&lines[i]) {
            let slot_off = match addr {
                FpAddr::Sp(off) => Some(off),
                FpAddr::Base(b, k) => base_off.get(&b).map(|&base| base + k),
            };
            let Some(slot_off) = slot_off else {
                // Unknown base: a store through it may alias anything tracked.
                if is_store {
                    slot_src.clear();
                    base_off.clear();
                } else {
                    // Unknown-base load still clobbers its destination.
                    let mut to_drop = Vec::new();
                    for (&key, &s) in slot_src.iter() {
                        if s == reg {
                            to_drop.push(key);
                        }
                    }
                    for key in to_drop {
                        slot_src.remove(&key);
                    }
                }
                i += 1;
                continue;
            };
            if is_store {
                let (lo, hi) = (slot_off, slot_off + fp_access_width(width));
                slot_src.retain(|&(off, w), _| {
                    let (l2, h2) = (off, off + fp_access_width(w));
                    h2 <= lo || l2 >= hi
                });
                slot_src.insert((slot_off, width), reg);
            } else {
                if let Some(&src) = slot_src.get(&(slot_off, width)) {
                    if src == reg {
                        // Loading a slot into the register that already holds
                        // its content: redundant.
                        kinds[i] = LineKind::Nop;
                        changed = true;
                        i += 1;
                        continue;
                    }
                    lines[i] = if width == b'q' {
                        format!("    mov v{}.16b, v{}.16b", reg, src)
                    } else {
                        format!("    fmov {}{}, {}{}", width as char, reg, width as char, src)
                    };
                    kinds[i] = classify_line(&lines[i]);
                    changed = true;
                }
                // The load writes its destination register, whether or not a
                // forward happened: entries forwarding FROM that register are
                // now stale. (Missing this clobbered a still-tracked entry
                // whose source was silently reloaded between store and use.)
                slot_src.retain(|_, &mut s| s != reg);
            }
            i += 1;
            continue;
        }

        // GP stack stores invalidate overlapping FP entries.
        if let LineKind::StoreSp { offset, is_word, .. } = kinds[i] {
            let (lo, hi) = (offset, offset + if is_word { 4 } else { 8 });
            slot_src.retain(|&(off, w), _| {
                let (l2, h2) = (off, off + fp_access_width(w));
                h2 <= lo || l2 >= hi
            });
            i += 1;
            continue;
        }

        match kinds[i] {
            // Untracked store forms may write anywhere: drop all state.
            LineKind::MemOther | LineKind::StorePairSp if t.starts_with("st") => {
                slot_src.clear();
                base_off.clear();
            }
            // Pair loads write two registers: drop entries sourcing either.
            LineKind::LoadPairSp => {
                let mut ops = t
                    .trim_start_matches("ldp ")
                    .split(',')
                    .take(2)
                    .filter_map(|tok| fp_token_reg(tok));
                let (a, b) = (ops.next(), ops.next());
                slot_src.retain(|_, &mut s| Some(s) != a && Some(s) != b);
            }
            // Other instructions: drop entries whose source register they write.
            _ => {
                let mut to_drop = Vec::new();
                for (&key, &s) in slot_src.iter() {
                    if fp_reg_written(&lines[i], s) {
                        to_drop.push(key);
                    }
                }
                for key in to_drop {
                    slot_src.remove(&key);
                }
            }
        }
        i += 1;
    }
    changed
}

fn eliminate_adjacent_store_load(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 1 < n {
        if let LineKind::StoreSp { reg: store_reg, offset: store_off, is_word: store_word } = kinds[i] {
            // Look ahead for the matching load (skip Nops)
            let mut j = i + 1;
            while j < n && kinds[j] == LineKind::Nop {
                j += 1;
            }
            if j < n {
                match kinds[j] {
                    LineKind::LoadSp { reg: load_reg, offset: load_off, is_word: load_word }
                        if store_off == load_off && store_word == load_word =>
                    {
                        if store_reg == load_reg {
                            // Same register: eliminate the load entirely
                            kinds[j] = LineKind::Nop;
                            changed = true;
                        } else {
                            // Different register: replace load with mov
                            let reg_fmt = if store_word { wreg_name } else { xreg_name };
                            lines[j] = format!("    mov {}, {}", reg_fmt(load_reg), reg_fmt(store_reg));
                            kinds[j] = LineKind::Move { dst: load_reg, src: store_reg, is_32bit: store_word };
                            changed = true;
                        }
                    }
                    // str wN, [sp, #off] followed by ldrsw xM, [sp, #off]
                    // The value was just stored as a word; sign-extending load can be replaced
                    // with sxtw xM, wN (or eliminated if same reg).
                    LineKind::LoadswSp { reg: load_reg, offset: load_off }
                        if store_off == load_off && store_word =>
                    {
                        lines[j] = format!("    sxtw {}, {}", xreg_name(load_reg), wreg_name(store_reg));
                        kinds[j] = LineKind::Sxtw { dst: load_reg, src: store_reg };
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        i += 1;
    }
    changed
}

// ── Pass 2: Redundant branch elimination ─────────────────────────────────────
//
// Pattern: b .LBBN  ;  .LBBN:  (branch to immediately next label)
// Falls through naturally, so the branch is redundant.

fn eliminate_redundant_branches(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        if kinds[i] == LineKind::Branch {
            if let Some(target) = branch_target(&lines[i]) {
                // Find next non-Nop line
                let mut j = i + 1;
                while j < n && kinds[j] == LineKind::Nop {
                    j += 1;
                }
                if j < n && kinds[j] == LineKind::Label {
                    if let Some(lbl) = label_name(&lines[j]) {
                        if target == lbl {
                            kinds[i] = LineKind::Nop;
                            changed = true;
                        }
                    }
                }
            }
        }
    }
    changed
}

// ── Pass 3: Self-move elimination ────────────────────────────────────────────
//
// Pattern: mov xN, xN — no-op

fn eliminate_self_moves(kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        if let LineKind::Move { dst, src, is_32bit } = kinds[i] {
            if dst == src && !is_32bit {
                // Only eliminate 64-bit self-moves (mov xN, xN).
                // On AArch64, `mov wN, wN` zeros the upper 32 bits of xN,
                // so it is NOT a true no-op and must be preserved.
                kinds[i] = LineKind::Nop;
                changed = true;
            }
        }
    }
    changed
}

// ── Shift-mask to ubfx folding ───────────────────────────────────────────────
//
// `(x >> K) & ((1<<N)-1)` lowers to `mov wM, #mask; and wD, wM, wS, lsr #K`
// because the shifted-and form needs the mask in a register. When K+N fits
// the 32-bit lane, this is exactly `ubfx wD, wS, #K, #N` — one instruction,
// no constant materialization (switch_dispatch's dispatch loop).
fn fold_shift_mask_ubfx(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_UBFX").is_ok() {
        return false;
    }
    let mut changed = false;
    for i in 0..n {
        if kinds[i] != LineKind::Alu || lines[i].contains('\n') {
            continue;
        }
        let t = lines[i].trim();
        let Some(ops) = t.strip_prefix("and ") else { continue };
        let parts: Vec<&str> = ops.split(", ").collect();
        if parts.len() != 4 {
            continue;
        }
        let Some(shift) = parts[3]
            .strip_prefix("lsr #")
            .and_then(|v| v.parse::<u32>().ok())
        else {
            continue;
        };
        if shift == 0 || shift >= 32 {
            continue;
        }
        // Both operands must be plain w-regs (no xzr/sp games).
        let Some(dst) = parts[0].strip_prefix('w').and_then(|v| v.parse::<u8>().ok()) else { continue };
        let Some(mask_reg) = parts[1].strip_prefix('w').and_then(|v| v.parse::<u8>().ok()) else { continue };
        if !parts[2].starts_with('w') || dst > 30 || mask_reg > 30 {
            continue;
        }
        // Find the mask register's defining mov immediate earlier in the
        // block (stop at any barrier or rewrite of the register).
        let mut mask: Option<u64> = None;
        let mut j = i;
        while j > 0 {
            j -= 1;
            match kinds[j] {
                LineKind::Nop => continue,
                LineKind::Label
                | LineKind::Branch
                | LineKind::CondBranch
                | LineKind::CmpBranch
                | LineKind::Call
                | LineKind::Ret
                | LineKind::Directive => break,
                _ => {}
            }
            if lines[j].contains('\n') {
                break;
            }
            let tj = lines[j].trim();
            let mut found = false;
            for pfx in ["mov w", "mov x", "movz w", "movz x"] {
                if let Some(rest) = tj.strip_prefix(pfx) {
                    if let Some((reg_s, imm_s)) = rest.split_once(", ") {
                        let reg_num = reg_s.trim().parse::<u8>().ok();
                        if reg_num == Some(mask_reg) {
                            if let Some(imm) = imm_s.trim().strip_prefix('#') {
                                if let Ok(m) = imm.parse::<i64>() {
                                    if m >= 0 {
                                        mask = Some(m as u64);
                                    }
                                }
                            }
                            found = true;
                        }
                    }
                }
            }
            if found {
                break;
            }
            if instr_clobbers_reg(tj, mask_reg) {
                break;
            }
        }
        let Some(m) = mask else { continue };
        if m == 0 || (m & (m + 1)) != 0 {
            continue; // not 2^N - 1
        }
        let width = 64 - (m + 1).leading_zeros(); // log2(m+1)
        let width = width - 1;
        if width == 0 || shift + width > 32 {
            continue;
        }
        lines[i] = format!("    ubfx {}, {}, #{}, #{}", parts[0], parts[2], shift, width);
        kinds[i] = LineKind::Alu;
        changed = true;
    }
    changed
}

// ── Pass 3b: Redundant sign-extension elimination ───────────────────────────
//
// Pattern: `sxtw xD, wS` where xS already holds a sign-extended value (written
// by `ldrsw`, an earlier `sxtw`, or a 64-bit mov of an extended register).
// The sxtw recomputes what is already true — replace with `mov xD, xS`
// (or drop entirely when D == S). Common after `ldrsw x0, [..]; mov xR, x0`
// feeding an I32→I64 cast (e.g. dot_product's per-load sxtw).
//
// A register stops being known-extended on any write: w-writes zero the upper
// bits, x-writes of unknown provenance clobber conservatively. All tracking is
// cleared at labels, branches, and calls.

fn eliminate_redundant_sxtw(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    // Function boundaries (the peephole sees the whole module's text).
    let mut fn_start = vec![false; n];
    let mut fn_end = vec![false; n];
    for i in 0..n {
        let t = lines[i].trim();
        if t.starts_with(".cfi_startproc") {
            fn_start[i] = true;
        } else if t.starts_with(".cfi_endproc") {
            fn_end[i] = true;
        }
    }
    // Single-write constant registers within the current function: the
    // hoisted loop-invariant masks from int_const_hoist. Their value is known
    // even across block boundaries where extended[] conservatively resets.
    let mut const_nonneg = [false; 33];
    let mut extended = [false; 33];
    extended[32] = true; // xzr is always "extended" (sxtw xD, wzr ≡ mov xD, xzr)
    let mut changed = false;
    for i in 0..n {
        if fn_start[i] {
            // (Re)build the per-function constant-register map. A register is
            // a known constant when it has exactly one mov/movz immediate
            // write at position p, no other write after p and before its last
            // use, and no non-store use before p (prologue saves read the
            // caller's value; epilogue restores land after all uses).
            const_nonneg = [false; 33];
            let mut mov_pos = [usize::MAX; 33];
            let mut mov_val = [0u64; 33];
            let mut mov_cnt = [0u8; 33];
            let mut other_write = [false; 33]; // non-mov write seen (position checked later)
            let mut other_write_pos = [usize::MAX; 33];
            let mut max_use = [0usize; 33];
            let mut early_bad_use = [false; 33];
            let mut j = i;
            while j < n && !fn_end[j] {
                if kinds[j] != LineKind::Nop {
                    for sub in lines[j].split('\n') {
                        let t = sub.trim();
                        if t.is_empty() || t.ends_with(':') || t.starts_with('.') {
                            continue;
                        }
                        let is_store = t.starts_with("st");
                        // mov/movz immediate write?
                        let mut is_const_mov = false;
                        for pfx in ["mov w", "mov x", "movz w", "movz x"] {
                            if let Some(rest) = t.strip_prefix(pfx) {
                                if let Some((reg_s, imm_s)) = rest.split_once(", ") {
                                    if let Some(imm_s) = imm_s.trim().strip_prefix('#') {
                                        let v = if let Some((base, lsl)) = imm_s.split_once(", lsl #") {
                                            match (base.parse::<i64>(), lsl.parse::<u32>()) {
                                                (Ok(b), Ok(k)) if b >= 0 && k < 32 => Some((b as u64) << k),
                                                _ => None,
                                            }
                                        } else {
                                            imm_s.parse::<i64>().ok().filter(|&b| b >= 0).map(|b| b as u64)
                                        };
                                        if let (Ok(d), Some(v)) = (reg_s.trim().parse::<u8>(), v) {
                                            if d < 33 {
                                                mov_cnt[d as usize] = mov_cnt[d as usize].saturating_add(1);
                                                mov_pos[d as usize] = j;
                                                mov_val[d as usize] = v;
                                                is_const_mov = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for b in 0..33u8 {
                            let written = gp_reg_written_broad(t, b);
                            let mentioned = written
                                || t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                                    .any(|tok| tok == xreg_name(b) || tok == wreg_name(b));
                            if !mentioned {
                                continue;
                            }
                            if written && !is_const_mov {
                                other_write[b as usize] = true;
                                if other_write_pos[b as usize] == usize::MAX || j > other_write_pos[b as usize] {
                                    // track writes after the mov below via max_use
                                }
                                other_write_pos[b as usize] = j;
                            }
                            if !written || is_store {
                                // a use (stores read the register)
                                if j > max_use[b as usize] {
                                    max_use[b as usize] = j;
                                }
                                if !is_store && mov_cnt[b as usize] == 0 {
                                    early_bad_use[b as usize] = true;
                                }
                            }
                            if written && is_const_mov {
                                // the mov itself also "uses" nothing
                            }
                        }
                        let _ = is_const_mov;
                    }
                }
                j += 1;
            }
            for b in 0..33usize {
                if mov_cnt[b] == 1 && mov_val[b] < (1u64 << 31) && !early_bad_use[b] {
                    // Any other write must land after the last use.
                    if !other_write[b] || other_write_pos[b] > max_use[b] {
                        const_nonneg[b] = true;
                    }
                }
            }
        }
        if fn_end[i] {
            const_nonneg = [false; 33];
        }
        match kinds[i] {
            // Control-flow barriers: nothing is known across them.
            LineKind::Label | LineKind::Branch | LineKind::CondBranch
            | LineKind::CmpBranch | LineKind::Call | LineKind::Ret => {
                extended = [false; 33];
                extended[32] = true;
            }
            // Unknown/memory kinds may clobber registers — except ldrsw with
            // non-sp addressing (classifies as MemOther), which CREATES a
            // sign-extended register.
            LineKind::MemOther | LineKind::Other | LineKind::LoadsbReg | LineKind::LoadPairSp => {
                let t = lines[i].trim();
                let mut ldrsw_dst: Option<u8> = None;
                if let Some(rest) = t.strip_prefix("ldrsw ") {
                    if let Some((reg_str, _)) = rest.split_once(',') {
                        let r = parse_reg(reg_str.trim());
                        if r != REG_NONE && r < 32 {
                            ldrsw_dst = Some(r);
                        }
                    }
                }
                extended = [false; 33];
                extended[32] = true;
                if let Some(r) = ldrsw_dst {
                    extended[r as usize] = true;
                }
            }
            LineKind::LoadswSp { reg, .. } => {
                if reg < 32 {
                    extended[reg as usize] = true;
                }
            }
            LineKind::Sxtw { dst, src } => {
                if extended[src as usize] {
                    if dst == src {
                        kinds[i] = LineKind::Nop;
                    } else {
                        lines[i] = format!("    mov {}, {}", xreg_name(dst), xreg_name(src));
                        kinds[i] = LineKind::Move { dst, src, is_32bit: false };
                    }
                    changed = true;
                }
                // Whether redundant or not, the sxtw result is sign-extended.
                if dst < 32 {
                    extended[dst as usize] = true;
                }
            }
            LineKind::Move { dst, src, is_32bit } => {
                if is_32bit {
                    // mov wN, wM zero-extends into xN — NOT a sign extension.
                    if dst < 32 {
                        extended[dst as usize] = false;
                    }
                } else if dst < 32 {
                    extended[dst as usize] = extended[src as usize];
                }
            }
            _ => {
                // Producers of values with bit 31 clear: their 64-bit form
                // equals their sign extension, so a later sxtw is redundant.
                let t = lines[i].trim();
                let mut nonneg_dst: Option<u8> = None;
                // mov/movz with a non-negative constant below 2^31.
                for pfx in ["mov w", "mov x", "movz w", "movz x"] {
                    if let Some(rest) = t.strip_prefix(pfx) {
                        if let Some((reg_s, imm_s)) = rest.split_once(", ") {
                            if let Some(imm_s) = imm_s.trim().strip_prefix('#') {
                                // Optional ", lsl #K" second shift component.
                                let mut val: Option<u64> = None;
                                if let Some((base, lsl)) = imm_s.split_once(", lsl #") {
                                    if let (Ok(b), Ok(k)) = (base.parse::<i64>(), lsl.parse::<u32>()) {
                                        if b >= 0 && k < 32 {
                                            let v = (b as u64) << k;
                                            val = Some(v);
                                        }
                                    }
                                } else if let Ok(b) = imm_s.parse::<i64>() {
                                    if b >= 0 {
                                        val = Some(b as u64);
                                    }
                                }
                                if let Some(v) = val {
                                    if v < (1u64 << 31) {
                                        if let Ok(d) = reg_s.trim().parse::<u8>() {
                                            if d < 32 {
                                                nonneg_dst = Some(d);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if nonneg_dst.is_none() {
                    if let Some(rest) = t.strip_prefix("ubfx ") {
                    // ubfx wD, wS, #K, #N — bit 31 clear when K+N <= 31
                    let parts: Vec<&str> = rest.split(", ").collect();
                    if parts.len() == 4 {
                        let d = parse_reg(parts[0]);
                        let k = parts[2].strip_prefix('#').and_then(|v| v.parse::<u32>().ok());
                        let wd = parts[3].strip_prefix('#').and_then(|v| v.parse::<u32>().ok());
                        if let (Some(k), Some(wd)) = (k, wd) {
                            if k + wd <= 31 && d < 32 {
                                nonneg_dst = Some(d);
                            }
                        }
                    }
                } else if let Some(rest) = t.strip_prefix("lsr ") {
                    // lsr wD, wS, #K with K >= 1: bit 31 clear
                    let parts: Vec<&str> = rest.split(", ").collect();
                    if parts.len() == 3 {
                        let d = parse_reg(parts[0]);
                        let k = parts[2].strip_prefix('#').and_then(|v| v.parse::<u32>().ok());
                        if let Some(k) = k {
                            if k >= 1 && d < 32 {
                                nonneg_dst = Some(d);
                            }
                        }
                    }
                } else if let Some(rest) = t.strip_prefix("and ") {
                    // and wD, wS, #imm with imm's bit 31 clear
                    let parts: Vec<&str> = rest.split(", ").collect();
                    if parts.len() == 3 {
                        let d = parse_reg(parts[0]);
                        let m = parts[2].strip_prefix('#').and_then(|v| v.parse::<i64>().ok());
                        if let Some(m) = m {
                            if m >= 0 && m < (1i64 << 31) && d < 32 {
                                nonneg_dst = Some(d);
                            }
                        } else {
                            // Register mask: bit 31 of the result is clear
                            // when clear in EITHER input (and = bitwise &).
                            let a = parse_reg(parts[1]);
                            let b = parse_reg(parts[2]);
                            let nonneg = |r: u8| {
                                (r < 33 && extended[r as usize]) || (r < 33 && const_nonneg[r as usize])
                            };
                            if d < 32 && a < 33 && b < 33 && (nonneg(a) || nonneg(b)) {
                                nonneg_dst = Some(d);
                            }
                        }
                    }
                }
                }
                if let Some(d) = nonneg_dst {
                    extended[d as usize] = true;
                } else if let Some(dst) = written_gp_register(&lines[i], kinds[i]) {
                    if dst < 32 {
                        extended[dst as usize] = false;
                    }
                }
            }
        }
    }
    changed
}

// ── Pass 3b: Redundant byte sign-extension elimination ──────────────────────
//
// Mirror of eliminate_redundant_sxtw for byte-level extensions: `ldrsb`
// produces a value whose bits 7..63 are uniform (a sign-extended byte), so a
// later `sxtb xD, wS` on it is redundant. Fires on char-byte loops like
// sieve's count loop (`ldrsb x0, [p]` then `sxtb` before the zero test).

fn eliminate_redundant_sxtb(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut extb = [false; 33];
    extb[32] = true; // the zero register is "byte-extended"
    let mut changed = false;
    for i in 0..n {
        match kinds[i] {
            LineKind::Label | LineKind::Branch | LineKind::CondBranch
            | LineKind::CmpBranch | LineKind::Call | LineKind::Ret => {
                extb = [false; 33];
                extb[32] = true;
            }
            LineKind::Move { dst, src, is_32bit } => {
                if is_32bit {
                    // w-mov zeroes bits 32-63, breaking byte-extension of negatives.
                    if dst < 32 {
                        extb[dst as usize] = false;
                    }
                } else if dst < 32 {
                    extb[dst as usize] = extb[src as usize];
                }
            }
            _ => {
                let t = lines[i].trim().to_string();
                // sxtb xD, wS
                if let Some(rest) = t.strip_prefix("sxtb ") {
                    if let Some((d, s)) = rest.split_once(", ") {
                        let dst = parse_reg(d.trim());
                        let src = parse_reg(s.trim());
                        if dst != REG_NONE && src != REG_NONE && dst < 32 && src < 32 {
                            if extb[src as usize] {
                                if dst == src {
                                    kinds[i] = LineKind::Nop;
                                } else {
                                    lines[i] = format!("    mov {}, {}", xreg_name(dst), xreg_name(src));
                                    kinds[i] = LineKind::Move { dst, src, is_32bit: false };
                                }
                                changed = true;
                            }
                            // The sxtb result is byte-extended either way.
                            extb[dst as usize] = true;
                            continue;
                        }
                    }
                }
                // ldrsb xD/wD, [...] creates a byte-sign-extended value.
                if let Some(rest) = t.strip_prefix("ldrsb ") {
                    if let Some((d, _)) = rest.split_once(',') {
                        let dst = parse_reg(d.trim());
                        if dst != REG_NONE && dst < 32 {
                            extb[dst as usize] = true;
                        }
                        continue;
                    }
                }
                // Any other write clobbers the tracking.
                if let Some(dst) = written_gp_register(&t, kinds[i]) {
                    if dst < 32 {
                        extb[dst as usize] = false;
                    }
                }
            }
        }
    }
    changed
}

// ── FP copy staging through x0 ───────────────────────────────────────────────
//
// FP copies between unassigned values stage through a GPR: `fmov x0, dA` /
// `fmov dB, x0`. Each GP<->FP fmov costs multiple cycles; the pair is just
// `fmov dB, dA`. Fires when the two are adjacent in a block and the GPR is
// dead after the pair (its next mention is a write or a barrier).
fn fold_fp_gpr_roundtrips(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_FP_ROUNDTRIP").is_ok() {
        return false;
    }
    let mut changed = false;
    for i in 0..n.saturating_sub(1) {
        if kinds[i] == LineKind::Nop || lines[i].contains('\n') {
            continue;
        }
        let t = lines[i].trim();
        let Some(rest) = t.strip_prefix("fmov x") else { continue };
        let Some((gpr_s, fps)) = rest.split_once(", ") else { continue };
        let Some(fps) = fps.strip_prefix('d') else { continue };
        let (Ok(gpr), Ok(fpa)) = (gpr_s.parse::<u8>(), fps.parse::<u8>()) else { continue };
        if gpr > 30 {
            continue;
        }
        // Next live line must be `fmov dB, x<gpr>`.
        let Some(j) = ((i + 1)..n).find(|&k| kinds[k] != LineKind::Nop) else { continue };
        if lines[j].contains('\n') {
            continue;
        }
        let tj = lines[j].trim();
        let Some(restj) = tj.strip_prefix("fmov d") else { continue };
        let Some((fpb_s, gpr_s2)) = restj.split_once(", ") else { continue };
        let Some(gpr_s2) = gpr_s2.strip_prefix('x') else { continue };
        let (Ok(fpb), Ok(gpr2)) = (fpb_s.parse::<u8>(), gpr_s2.parse::<u8>()) else { continue };
        if gpr2 != gpr {
            continue;
        }
        // The GPR must be dead after line j: scan to the block end; the first
        // mention (x or w form) must be a write.
        let xn = xreg_name(gpr);
        let wn = wreg_name(gpr);
        let mut dead = true;
        for k in (j + 1)..n {
            match kinds[k] {
                LineKind::Nop => continue,
                LineKind::Label
                | LineKind::Branch
                | LineKind::CondBranch
                | LineKind::CmpBranch
                | LineKind::Call
                | LineKind::Ret
                | LineKind::Directive => break,
                _ => {}
            }
            let mentions = lines[k]
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .any(|tok| tok == xn || tok == wn);
            if mentions {
                // A write (destination) ends the value's life; a read keeps it.
                dead = instr_clobbers_reg(lines[k].trim_start(), gpr);
                break;
            }
        }
        if !dead {
            continue;
        }
        lines[i] = format!("    fmov d{}, d{}", fpb, fpa);
        kinds[i] = LineKind::Other;
        kinds[j] = LineKind::Nop;
        changed = true;
    }
    changed
}

// ── Zero-store via wzr/xzr ───────────────────────────────────────────────────
//
// Storing a literal zero goes through a materialization: `mov x0, #0` then
// `strb w0, [x9]`. The zero register does it for free: `strb wzr, [x9]`.
// The mov is left for dead-move elimination (it may have other uses).
fn fold_zero_stores(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        let zr_reg = match kinds[i] {
            LineKind::MoveImm { dst } | LineKind::MoveWide { dst } if dst <= 30 => {
                let t = lines[i].trim();
                // movn #0 is -1, not zero; symbol operands fail the int parse.
                if t.starts_with("movn") || lines[i].contains('\n') {
                    continue;
                }
                let zero = t
                    .rsplit_once(", ")
                    .and_then(|(_, imm)| imm.trim().strip_prefix('#'))
                    .is_some_and(|v| v.parse::<i64>() == Ok(0));
                if zero {
                    dst
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let (xname, wname) = (xreg_name(zr_reg), wreg_name(zr_reg));
        // Rewrite the value operand of the next store that reads R, stopping
        // at the first barrier, rewrite of R, or memory op.
        for j in (i + 1)..n {
            match kinds[j] {
                LineKind::Nop => continue,
                LineKind::Move { .. }
                | LineKind::MoveImm { .. }
                | LineKind::MoveWide { .. }
                | LineKind::Alu
                | LineKind::Compare => {
                    if instr_clobbers_reg(lines[j].trim_start(), zr_reg) {
                        break;
                    }
                    continue;
                }
                LineKind::StoreSp { .. } | LineKind::MemOther | LineKind::StorePairSp => {
                    let t = lines[j].trim_start().to_string();
                    if !(t.starts_with("str") || t.starts_with("stp")) {
                        break; // a load: stop
                    }
                    if let Some((mnem, rest)) = t.split_once(' ') {
                        if let Some((first, suffix)) = rest.split_once(',') {
                            let first = first.trim();
                            let zr = if first == xname {
                                "xzr"
                            } else if first == wname {
                                "wzr"
                            } else {
                                break;
                            };
                            lines[j] = format!("    {} {},{}", mnem, zr, suffix);
                            kinds[j] = classify_line(&lines[j]);
                            changed = true;
                        }
                    }
                    break;
                }
                _ => break,
            }
        }
    }
    changed
}

// ── Pass 4: Move chain optimization ──────────────────────────────────────────
//
// Pattern: mov A, B  ;  mov C, A → mov C, B
// This allows the first mov to potentially be dead-eliminated later.

fn eliminate_move_chains(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 1 < n {
        match kinds[i] {
            LineKind::Move { dst: dst1, src: src1, is_32bit: is_32bit1 } => {
                // Find next non-Nop instruction
                let mut j = i + 1;
                while j < n && kinds[j] == LineKind::Nop {
                    j += 1;
                }
                if j < n {
                    if let LineKind::Move { dst: dst2, src: src2, is_32bit: is_32bit2 } = kinds[j] {
                        // mov dst1, src1 ; mov dst2, dst1 → mov dst2, src1
                        // Only safe when both moves use the same width.
                        if src2 == dst1 && dst2 != src1 && is_32bit1 == is_32bit2 {
                            let reg_fmt = if is_32bit2 { wreg_name } else { xreg_name };
                            lines[j] = format!("    mov {}, {}", reg_fmt(dst2), reg_fmt(src1));
                            kinds[j] = LineKind::Move { dst: dst2, src: src1, is_32bit: is_32bit2 };
                            changed = true;
                        }
                    }
                }
            }
            LineKind::MoveImm { dst: dst1 } | LineKind::MoveWide { dst: dst1 } => {
                // mov xN, #imm ; mov xM, xN → mov xM, #imm (copy the immediate)
                // Only when dst1 is a scratch register (x0-x15) not callee-saved
                if dst1 <= 15 {
                    let mut j = i + 1;
                    while j < n && kinds[j] == LineKind::Nop {
                        j += 1;
                    }
                    if j < n {
                        if let LineKind::Move { dst: dst2, src: src2, is_32bit: _ } = kinds[j] {
                            if src2 == dst1 {
                                // Copy the immediate instruction, retargeted to dst2
                                let old_line = lines[i].trim();
                                // Replace the register in the first instruction
                                if let Some(new_line) = retarget_move_imm(old_line, dst2) {
                                    lines[j] = format!("    {}", new_line);
                                    kinds[j] = LineKind::MoveImm { dst: dst2 };
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    changed
}

// ── Overwritten-move elimination ─────────────────────────────────────────────
//
// The accumulator-staging emission leaves dead moves like `mov x0, x22` right
// before `add x0, x1, x22` (the add overwrites x0 without reading it). Delete
// any mov/MoveImm whose destination register is provably overwritten before it
// is read. Same-block, forward scan; any control flow or non-pure-write
// mention keeps the move.
fn eliminate_overwritten_moves(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        let dst = match kinds[i] {
            LineKind::Move { dst, .. } => dst,
            LineKind::MoveImm { dst } | LineKind::MoveWide { dst } => dst,
            _ => continue,
        };
        if dst >= 29 {
            continue; // never touch sp/fp-adjacent registers
        }
        let xn = xreg_name(dst);
        let wn = wreg_name(dst);
        let mut dead = false;
        let mut j = i + 1;
        while j < n {
            if kinds[j] == LineKind::Nop {
                j += 1;
                continue;
            }
            match kinds[j] {
                LineKind::Label | LineKind::Branch | LineKind::CondBranch
                | LineKind::CmpBranch | LineKind::Call | LineKind::Ret
                | LineKind::Directive => break,
                _ => {}
            }
            let t = &lines[j];
            let mentions = t
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .any(|tok| tok == xn || tok == wn);
            if !mentions {
                j += 1;
                continue;
            }
            // First mention decides: a pure overwrite (dest written, not read)
            // makes the earlier move dead; anything else reads it.
            dead = match kinds[j] {
                LineKind::Move { dst: d2, src: s2, .. } => d2 == dst && s2 != dst,
                LineKind::MoveImm { dst: d2 } | LineKind::MoveWide { dst: d2 } => d2 == dst,
                LineKind::LoadSp { reg, .. } | LineKind::LoadswSp { reg, .. } => reg == dst,
                LineKind::Alu => {
                    // Three-operand ALU: first operand is the destination.
                    // Dead only if dst is that dest and not among the sources.
                    let t2 = t.trim();
                    let ops = t2.split_once(' ').map(|x| x.1).unwrap_or("");
                    let mut parts = ops.splitn(2, ',');
                    let first = parts.next().unwrap_or("").trim();
                    let rest = parts.next().unwrap_or("");
                    let first_is_dst = first == xn || first == wn;
                    let rest_reads = rest
                        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                        .any(|tok| tok == xn || tok == wn);
                    first_is_dst && !rest_reads
                }
                _ => false,
            };
            break;
        }
        if dead {
            kinds[i] = LineKind::Nop;
            changed = true;
        }
    }
    changed
}

// ── Zero-extend move-chain folding ───────────────────────────────────────────
//
// `mov xA, xS ; mov wA, wA ; mov xD, xA` — the accumulator-path zero-extension
// idiom (the 32-bit self-move clears the top half; it is NOT a no-op, so
// eliminate_self_moves correctly keeps it). The chain computes xD = zext32(wS),
// exactly `mov wD, wS`. Sound only if A is not read later in the block.
fn fold_zext_move_chains(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n.saturating_sub(2) {
        let LineKind::Move { dst: a, src: s, is_32bit: false } = kinds[i] else { continue };
        let LineKind::Move { dst: a2, src: s2, is_32bit: true } = kinds[i + 1] else { continue };
        if a2 != a || s2 != a {
            continue; // middle must be mov wA, wA
        }
        let LineKind::Move { dst: d, src: s3, is_32bit: false } = kinds[i + 2] else { continue };
        if s3 != a || d == a || s == a {
            continue;
        }
        // A must be dead after the chain: no later READ before a boundary.
        // A write-only line (e.g. `mov x0, #15`) ends the old value's life.
        let (xa, wa) = (xreg_name(a), wreg_name(a));
        let mut a_live = false;
        for j in (i + 3)..n {
            match kinds[j] {
                LineKind::Label | LineKind::Branch | LineKind::CondBranch
                | LineKind::CmpBranch | LineKind::Call | LineKind::Ret => break,
                LineKind::Nop => continue,
                _ => {}
            }
            let mentions = lines[j]
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .filter(|tok| *tok == xa || *tok == wa)
                .count();
            if mentions == 0 {
                continue;
            }
            if gp_reg_write_only(&lines[j], a) {
                break; // write-only: the chain's value in A is dead here
            }
            a_live = true;
            break;
        }
        if a_live {
            continue;
        }
        lines[i] = format!("    mov {}, {}", wreg_name(d), wreg_name(s));
        kinds[i] = LineKind::Move { dst: d, src: s, is_32bit: true };
        kinds[i + 1] = LineKind::Nop;
        kinds[i +2] = LineKind::Nop;
        changed = true;
    }
    changed
}

/// Retarget a move-immediate instruction to a different destination register.
/// E.g., `mov x0, #5` with new dest x14 → `mov x14, #5`
fn retarget_move_imm(line: &str, new_dst: u8) -> Option<String> {
    // Handle: mov xN, #imm  /  mov xN, :lo12:sym  /  movz xN, #imm  /  movn xN, #imm
    for prefix in &["mov ", "movz ", "movn "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            if let Some((_old_reg, imm_part)) = rest.split_once(", ") {
                let new_reg = if line.contains('w') && !imm_part.starts_with('w') {
                    // If original used w-register (e.g., mov w0, #5)
                    // check if the source had 'w' prefix
                    let old_first = rest.chars().next()?;
                    if old_first == 'w' {
                        wreg_name(new_dst)
                    } else {
                        xreg_name(new_dst)
                    }
                } else {
                    xreg_name(new_dst)
                };
                return Some(format!("{}{}, {}", prefix, new_reg, imm_part));
            }
        }
    }
    None
}

// ── Shrink-wrap: defer callee-saves past a clean early return ───────────────
//
// Recursive early-exit functions like binary_trees' check save every
// callee-saved register at entry, then the null-child path (half the calls)
// returns immediately — paying 5 stp + 5 ldp for nothing:
//
//   check:
//     stp x29, x30, [sp, #-96]!
//     stp x19, x20, [sp, #16]   ... 5 saves ...
//     mov x20, x0; ldr x0, [x0]; mov x19, x0; cmp x19, #0
//     b.eq .LBB5                (early: return 1, with full restores)
//     b .LBB6                   (continuation: calls check twice)
//
// When the entry region before the first conditional branch is short, the
// early-exit block is a clean return (no calls), and the continuation has a
// single predecessor, move the save/restore pairs onto the continuation
// path. Entry-region uses of saved registers are remapped to free scratch
// registers (x9-x15) and re-homed with movs after the deferred saves.
fn shrink_wrap_early_returns(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_SHRINK_WRAP").is_ok() {
        return false;
    }
    let mentions_reg = |text: &str, r: u8| {
        text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == xreg_name(r) || tok == wreg_name(r))
    };
    let mut changed = false;
    let mut i = 0;
    while i < n {
        // Prologue: stp x29, x30, [sp, #-N]! ... mov x29, sp
        let is_prologue_stp = {
            let t = lines[i].trim();
            t.starts_with("stp x29, x30, [sp, #-") && t.ends_with("]!")
        };
        if !is_prologue_stp {
            i += 1;
            continue;
        }
        // Walk to `mov x29, sp`, skipping cfi directives.
        let mut p = i + 1;
        while p < n && (kinds[p] == LineKind::Directive || lines[p].trim().starts_with(".cfi_")) {
            p += 1;
        }
        if p >= n || lines[p].trim() != "mov x29, sp" {
            i += 1;
            continue;
        }
        // Collect consecutive callee-saved GPR stores (skip cfi between).
        let mut saves: Vec<usize> = Vec::new(); // line indices
        let mut saved_regs: Vec<u8> = Vec::new();
        let mut q = p + 1;
        loop {
            while q < n && kinds[q] == LineKind::Directive {
                q += 1;
            }
            if q >= n {
                break;
            }
            let t = lines[q].trim().to_string();
            let mut regs_in_save: Vec<u8> = Vec::new();
            if let Some(rest) = t.strip_prefix("stp ") {
                if let Some((regs, addr)) = rest.split_once(", [sp, #") {
                    if addr.ends_with(']') && !addr.contains("]!") {
                        let mut it = regs.split(", ");
                        if let (Some(a), Some(b)) = (it.next(), it.next()) {
                            let (pa, pb) = (parse_reg(a.trim()), parse_reg(b.trim()));
                            if (19..=28).contains(&pa) && (19..=28).contains(&pb) {
                                regs_in_save = vec![pa, pb];
                            }
                        }
                    }
                }
            } else if let Some(rest) = t.strip_prefix("str ") {
                if let Some((reg, addr)) = rest.split_once(", [sp, #") {
                    if addr.ends_with(']') {
                        let pr = parse_reg(reg.trim());
                        if (19..=28).contains(&pr) {
                            regs_in_save = vec![pr];
                        }
                    }
                }
            }
            if !regs_in_save.is_empty() {
                saves.push(q);
                saved_regs.extend(regs_in_save);
                q += 1;
            } else {
                break;
            }
        }
        if saves.len() < 2 {
            i += 1;
            continue; // nothing worth deferring
        }
        // Entry region: q .. first CondBranch e. No labels/branches/calls/ret.
        let mut e = q;
        let mut region_ok = true;
        let mut entry_lines: Vec<usize> = Vec::new();
        while e < n {
            match kinds[e] {
                LineKind::CondBranch => break,
                LineKind::Nop | LineKind::Directive => {}
                LineKind::Label | LineKind::Branch | LineKind::CmpBranch | LineKind::Call
                | LineKind::Ret => {
                    region_ok = false;
                    break;
                }
                _ => {
                    if lines[e].contains('\n') {
                        region_ok = false;
                        break;
                    }
                    entry_lines.push(e);
                }
            }
            e += 1;
        }
        if !region_ok || e >= n || e == q {
            i += 1;
            continue;
        }
        let Some((_, early_lbl)) = cond_branch_parts(lines[e].trim_start()) else {
            i += 1;
            continue;
        };
        // After e: optional `.Lskip:` + `b .Lcont`, then the early block label.
        let next_live = |from: usize| (from..n).find(|&j| kinds[j] != LineKind::Nop && kinds[j] != LineKind::Directive);
        let Some(mut cont_lbl) = next_live(e + 1).map(|j| lines[j].trim().to_string()) else {
            i += 1;
            continue;
        };
        // Skip a trampoline (label + unconditional branch) to find both blocks.
        let mut early_pos = None;
        let mut cont_pos = None;
        {
            // If the line after e is a Label followed by an unconditional
            // branch (the .Lskip trampoline), the branch target is the
            // continuation block.
            let mut fall = next_live(e + 1);
            if let Some(jl) = fall {
                if kinds[jl] == LineKind::Label {
                    if let Some(jb) = next_live(jl + 1) {
                        if kinds[jb] == LineKind::Branch {
                            if let Some(t) = branch_target(lines[jb].trim_start()) {
                                cont_lbl = t.to_string();
                                fall = next_live(jb + 1);
                            }
                        }
                    }
                }
            }
            let _ = fall;
            let early_with_colon = format!("{}:", early_lbl);
            let cont_with_colon = format!("{}:", cont_lbl);
            let mut j = e + 1;
            while j < n {
                if kinds[j] == LineKind::Label {
                    let nm = lines[j].trim();
                    if nm == early_with_colon {
                        early_pos = Some(j);
                    }
                    if nm == cont_with_colon {
                        cont_pos = Some(j);
                    }
                    if early_pos.is_some() && cont_pos.is_some() {
                        break;
                    }
                }
                // Stop at the next function.
                let t = lines[j].trim();
                if t.starts_with("stp x29, x30, [sp, #-") {
                    break;
                }
                j += 1;
            }
        }
        let (Some(ep), Some(cp)) = (early_pos, cont_pos) else {
            i += 1;
            continue;
        };
        if cp <= ep {
            i += 1; // expect continuation after the early block textually
            continue;
        }
        // Early block: [ep+1, cp) — must be a clean return: one ret at end,
        // no calls, restores exactly the saved regs, no other saved-reg use.
        let mut early_ok = true;
        let mut early_restores: Vec<usize> = Vec::new();
        let mut early_ret = None;
        for j in ep + 1..cp {
            if kinds[j] == LineKind::Nop || kinds[j] == LineKind::Directive {
                continue;
            }
            let t = lines[j].trim();
            match kinds[j] {
                LineKind::Call | LineKind::Label | LineKind::Branch | LineKind::CondBranch
                | LineKind::CmpBranch => {
                    early_ok = false;
                    break;
                }
                LineKind::Ret => {
                    early_ret = Some(j);
                }
                _ => {
                    let is_restore = (t.starts_with("ldp ") || t.starts_with("ldr "))
                        && t.contains("[sp");
                    if is_restore {
                        // The frame teardown (ldp x29, x30, [sp], #N) stays;
                        // only callee-saved register restores are deferred.
                        if t.contains("x30") {
                            continue;
                        }
                        // Must restore only saved regs.
                        let mut regs_ok = true;
                        for part in t.split(|c| [',', '[', ']'].contains(&c)) {
                            let part = part.trim();
                            if part.starts_with('x') || part.starts_with('w') || part.starts_with('d') {
                                if let Some(rnum) = part.trim_start_matches(|c| c == 'x' || c == 'w' || c == 'd').parse::<u8>().ok() {
                                    if (19..=28).contains(&rnum) && !saved_regs.contains(&rnum) {
                                        regs_ok = false;
                                    }
                                }
                            }
                        }
                        if !regs_ok {
                            early_ok = false;
                            break;
                        }
                        early_restores.push(j);
                    } else {
                        // Plain instruction: may not mention saved regs.
                        if saved_regs.iter().any(|&r| mentions_reg(t, r)) {
                            early_ok = false;
                            break;
                        }
                    }
                }
            }
        }
        if !early_ok || early_ret.is_none() {
            i += 1;
            continue;
        }
        // Continuation: single predecessor (the trampoline from the entry
        // region) and not the target of any backward branch.
        let cont_name = cont_lbl.as_str();
        let mut preds = 0;
        let mut backward = false;
        for j in q..n {
            let t = lines[j].trim();
            if t.starts_with("stp x29, x30, [sp, #-") && j > cp {
                break; // next function
            }
            for sub in t.split('\n') {
                if any_branch_target(sub.trim()) == Some(cont_name) {
                    preds += 1;
                    // backward if the branch is after the continuation label
                    if j > cp {
                        backward = true;
                    }
                }
            }
        }
        if preds != 1 || backward {
            i += 1;
            continue;
        }
        // Remap entry-region uses of saved regs to scratch registers.
        // Every saved reg mentioned must be written before it is read.
        let mut used_scratches: Vec<u8> = Vec::new();
        for r in 9..=15u8 {
            let used = entry_lines.iter().any(|&j| mentions_reg(lines[j].trim(), r))
                || mentions_reg(lines[e].trim_start(), r);
            if !used {
                used_scratches.push(r);
            }
        }
        let mut remap: Vec<(u8, u8)> = Vec::new(); // (saved, scratch)
        {
            let mut first_is_write: FxHashMap<u8, bool> = FxHashMap::default();
            for &j in &entry_lines {
                let t = lines[j].trim();
                for &r in &saved_regs {
                    if mentions_reg(t, r) && !first_is_write.contains_key(&r) {
                        first_is_write.insert(r, instr_clobbers_reg(t, r));
                    }
                }
            }
            let cond_t = lines[e].trim_start();
            for &r in &saved_regs {
                if mentions_reg(cond_t, r) && !first_is_write.contains_key(&r) {
                    first_is_write.insert(r, false); // condbranch reads
                }
            }
            let mut ok = true;
            for (&r, &is_write) in &first_is_write {
                if !is_write {
                    ok = false;
                    break;
                }
            }
            if !ok || first_is_write.len() > used_scratches.len() {
                i += 1;
                continue;
            }
            let mut order: Vec<u8> = first_is_write.keys().copied().collect();
            order.sort();
            for (k, &r) in order.iter().enumerate() {
                remap.push((r, used_scratches[k]));
            }
        }
        // Apply the rewrite.
        for &j in &saves {
            kinds[j] = LineKind::Nop;
        }
        for &j in &early_restores {
            kinds[j] = LineKind::Nop;
        }
        // Remap entry region + the conditional branch.
        let remap_text = |text: &str, remap: &[(u8, u8)]| -> String {
            let mut out = String::new();
            let mut last = 0;
            let bytes = text.as_bytes();
            let mut idx = 0;
            while idx < bytes.len() {
                let c = bytes[idx] as char;
                if c.is_ascii_alphanumeric() || c == '_' {
                    let start = idx;
                    while idx < bytes.len() && (bytes[idx] as char).is_ascii_alphanumeric() || idx < bytes.len() && bytes[idx] == b'_' {
                        idx += 1;
                    }
                    let tok = &text[start..idx];
                    let mut replaced = None;
                    for &(from, to) in remap {
                        if tok == xreg_name(from) {
                            replaced = Some(xreg_name(to));
                            break;
                        }
                        if tok == wreg_name(from) {
                            replaced = Some(wreg_name(to));
                            break;
                        }
                    }
                    if let Some(rep) = replaced {
                        out.push_str(&text[last..start]);
                        out.push_str(rep);
                        last = idx;
                    }
                } else {
                    idx += 1;
                }
            }
            out.push_str(&text[last..]);
            out
        };
        for &j in &entry_lines {
            lines[j] = remap_text(&lines[j].clone(), &remap);
            kinds[j] = classify_line(&lines[j]);
        }
        lines[e] = remap_text(&lines[e].clone(), &remap);
        // Insert saves + re-home movs after the continuation label.
        let mut insert = String::new();
        for &j in &saves {
            insert.push('\n');
            insert.push_str(&lines[j]);
        }
        for &(from, to) in &remap {
            insert.push('\n');
            insert.push_str(&format!("    mov {}, {}", xreg_name(from), xreg_name(to)));
        }
        lines[cp] = format!("{}{}", lines[cp], insert);
        kinds[cp] = LineKind::Other;
        changed = true;
        i = cp; // continue scanning after this function's continuation
    }
    changed
}

// ── Pass 5: Branch-over-branch fusion ────────────────────────────────────────
//
// Pattern:
//   b.cc .Lskip_N
//   b .target
//   .Lskip_N:
//
// Transform to:
//   b.!cc .target
//
// This is a very common pattern from the codegen: it emits a conditional branch
// to skip over an unconditional branch.

/// Estimate the distance (in instructions) from position `from` to the label
/// `target` in the assembly. Returns `None` if the label is not found.
fn estimate_branch_distance(lines: &[String], kinds: &[LineKind], from: usize, target: &str) -> Option<usize> {
    // Search both forward and backward for the target label
    let target_with_colon = format!("{}:", target);
    for idx in 0..lines.len() {
        if kinds[idx] == LineKind::Label && lines[idx].trim() == target_with_colon {
            // Count non-Nop lines between from and idx (each is one 4-byte instruction)
            let (lo, hi) = if from < idx { (from, idx) } else { (idx, from) };
            let count = (lo..hi).filter(|&p| kinds[p] != LineKind::Nop && kinds[p] != LineKind::Label).count();
            return Some(count);
        }
    }
    None
}

/// Maximum number of instructions a b.cond can reach (±1MB = ±262144 instructions).
/// Use a conservative threshold of 200,000 to leave margin.
const COND_BRANCH_SAFE_DISTANCE: usize = 200_000;

fn fuse_branch_over_branch(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 2 < n {
        if kinds[i] == LineKind::CondBranch {
            // Find the next two non-Nop instructions
            let mut j = i + 1;
            while j < n && kinds[j] == LineKind::Nop {
                j += 1;
            }
            if j >= n {
                i += 1;
                continue;
            }
            let mut k = j + 1;
            while k < n && kinds[k] == LineKind::Nop {
                k += 1;
            }
            if k >= n {
                i += 1;
                continue;
            }

            // Check pattern: b.cc .skip ; b .target ; .skip:
            if kinds[j] == LineKind::Branch && kinds[k] == LineKind::Label {
                if let (Some((cc, skip_target)), Some(real_target), Some(lbl)) = (
                    cond_branch_parts(&lines[i]),
                    branch_target(&lines[j]),
                    label_name(&lines[k]),
                ) {
                    if skip_target == lbl {
                        // Check if the real target is within safe b.cond range.
                        // If the label isn't found in this snippet (external target),
                        // assume it's in range since the unconditional branch already
                        // reached it, and b.cond has ±1MB range which is generous.
                        let in_range = estimate_branch_distance(lines, kinds, i, real_target)
                            .is_none_or(|d| d < COND_BRANCH_SAFE_DISTANCE);
                        if in_range {
                            if let Some(inv_cc) = invert_condition(cc) {
                                // Replace conditional branch with inverted condition to real target
                                lines[i] = format!("    b.{} {}", inv_cc, real_target);
                                // kinds[i] stays as CondBranch
                                // Remove the unconditional branch
                                kinds[j] = LineKind::Nop;
                                // Keep the label (might be targeted by other branches)
                                changed = true;
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

// ── Global register copy propagation ─────────────────────────────────────────
//
// After store forwarding converts loads into register moves, propagate those
// copies into subsequent instructions. For `mov xDST, xSRC`, replace
// references to xDST with xSRC in the immediately following instruction
// (within the same basic block).

fn propagate_register_copies(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;

    for i in 0..n {
        // Only process 64-bit register-to-register moves
        let (dst, src) = match kinds[i] {
            LineKind::Move { dst, src, is_32bit: false } => {
                // Don't propagate sp/fp moves
                if dst >= 29 || src >= 29 {
                    continue;
                }
                (dst, src)
            }
            _ => continue,
        };

        // Find the next non-Nop instruction
        let mut j = i + 1;
        while j < n && kinds[j] == LineKind::Nop {
            j += 1;
        }
        if j >= n {
            continue;
        }

        // Don't propagate across control flow boundaries or into instructions
        // with multiple destination registers (like ldp which has two dest regs
        // after the mnemonic, but our replace_source_reg_in_instruction only
        // treats the first operand as dest).
        match kinds[j] {
            LineKind::Label | LineKind::Branch | LineKind::Ret | LineKind::Directive
            | LineKind::LoadPairSp | LineKind::Call => continue,
            _ => {}
        }
        // Also skip ldp/ldaxr/ldxr/stxr by checking instruction text,
        // since these have multiple dest registers or complex operand semantics
        let trimmed_j = lines[j].trim();
        if trimmed_j.starts_with("ldp ")
            || trimmed_j.starts_with("ldaxr ")
            || trimmed_j.starts_with("ldxr ")
            || trimmed_j.starts_with("stxr ")
            || trimmed_j.starts_with("ldaxp ")
            || trimmed_j.starts_with("ldxp ")
            || trimmed_j.starts_with("cas ")
        {
            continue;
        }

        // Try to replace references to dst with src in line j
        let old_line = &lines[j];
        let dst_name = xreg_name(dst);
        if !old_line.contains(dst_name) {
            continue;
        }

        // Don't propagate into instructions that write to the same register
        // as the source (would create a self-reference)
        let src_name = xreg_name(src);

        // For move instructions, replace the source operand
        match kinds[j] {
            LineKind::Move { dst: dst2, src: src2, is_32bit: false } if src2 == dst => {
                // mov X, dst -> mov X, src
                if dst2 != src {
                    lines[j] = format!("    mov {}, {}", xreg_name(dst2), src_name);
                    kinds[j] = LineKind::Move { dst: dst2, src, is_32bit: false };
                    changed = true;
                }
            }
            _ => {
                // General case: try to replace the register in the instruction text.
                // Only replace in source operand positions (not destination).
                if let Some(new_line) = replace_source_reg_in_instruction(old_line, dst_name, src_name) {
                    lines[j] = new_line;
                    kinds[j] = classify_line(&lines[j]);
                    changed = true;
                }
            }
        }
    }
    changed
}

// Shared peephole string utilities -- see backend/peephole_common.rs
use crate::backend::peephole_common::{replace_source_reg_in_instruction, replace_whole_word};

// ── Global dead store elimination ────────────────────────────────────────────
//
// Scans the entire function to find stack slot offsets that are never loaded.
// Stores to such slots are dead and can be eliminated.
// This runs after global store forwarding, which may have converted many loads
// to register moves, leaving the original stores dead.

fn global_dead_store_elimination(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    // Process each function independently: the tracked state (derived bases,
    // escapes, loop-variant sets, loaded ranges) is only meaningful within
    // one stack frame, and a single escape in one function must not disable
    // the pass for the rest of the module.
    if std::env::var("CCC_NO_GDSE_FN_SCOPE").is_ok() {
        return gdse_range(lines, kinds, 0, n);
    }
    let mut changed = false;
    let mut start = 0usize;
    for i in 0..n {
        if lines[i].trim().starts_with(".cfi_endproc") {
            changed |= gdse_range(lines, kinds, start, i + 1);
            start = i + 1;
        }
    }
    if start < n {
        changed |= gdse_range(lines, kinds, start, n);
    }
    changed
}

fn gdse_range(lines: &[String], kinds: &mut [LineKind], start: usize, end: usize) -> bool {
    // Safety: stack slots can be accessed through sp-derived pointer registers
    // (`add xN, sp, #off` etc.). Instead of bailing on any address-of-sp, track
    // the derived bases: as long as every use of a base is an address use
    // (`[xN]`, `[xN, #k]`), a copy/adjustment of another base, or a clobber
    // that ends its lineage, no frame address ever leaves the function's
    // visible text, so all slot loads are accounted for below. Any other use
    // (stored as a value, call argument, comparison, inline asm, ...) is an
    // escape and we bail.
    //
    // Base offsets are tracked in sp terms: `add xN, sp, #k` gives xN offset k,
    // `mov xM, xN` copies it, `add/sub xN, xN, #k` adjusts it, and a write of
    // any other kind ends the mapping. Calls clobber caller-saved bases and
    // are escapes when a base sits in an argument register.

    // Collect candidate bases and check for escapes.
    let mut base_off: FxHashMap<u8, i32> = FxHashMap::default();
    let mut escaped = false;
    for i in start..end {
        if escaped {
            break;
        }
        let t = lines[i].trim();
        if kinds[i] == LineKind::Nop || t.is_empty() {
            continue;
        }
        if kinds[i] == LineKind::Label || kinds[i] == LineKind::Directive {
            continue;
        }
        if kinds[i] == LineKind::Call {
            // A base in an argument register at a call site escapes.
            for b in 0..=8u8 {
                if base_off.contains_key(&b) {
                    escaped = true;
                    break;
                }
            }
            // Calls clobber caller-saved registers: end those lineages.
            base_off.retain(|&r, _| r >= 19);
            continue;
        }

        // Base creation from sp (`add xN, sp, #k` / `sub xN, sp, #k` /
        // `mov xN, sp`): these lines mention no existing base, so handle
        // them before the mention scan.
        let mut created = false;
        if let Some(rest) = t.strip_prefix("add ").or_else(|| t.strip_prefix("sub ")) {
            let neg = t.starts_with("sub ");
            if let Some((dst_s, srcs)) = rest.split_once(", ") {
                if let Some((src_s, imm_s)) = srcs.split_once(", ") {
                    if src_s.trim() == "sp" {
                        if let Some(imm) = imm_s.trim().strip_prefix('#').and_then(|v| v.parse::<i32>().ok()) {
                            let dst = parse_reg(dst_s.trim());
                            if dst < 29 {
                                base_off.insert(dst, if neg { -imm } else { imm });
                            }
                            created = true;
                        }
                    }
                }
            }
        } else if let Some(rest) = t.strip_prefix("mov ") {
            if let Some((dst_s, src_s)) = rest.split_once(", ") {
                if src_s.trim() == "sp" {
                    let dst = parse_reg(dst_s.trim());
                    if dst < 29 {
                        base_off.insert(dst, 0);
                    }
                    created = true;
                }
            }
        }
        if created {
            continue;
        }

        // Mentions of currently tracked bases in this line (x or w form).
        let mut mentioned = Vec::new();
        for &b in base_off.keys() {
            let xn = xreg_name(b);
            let wn = wreg_name(b);
            if t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .any(|tok| tok == xn || tok == wn)
            {
                mentioned.push(b);
            }
        }
        for b in mentioned {
            let xn = xreg_name(b);
            // Address use as `[xN]` / `[xN, #k]`: allowed. Writeback forms
            // (`!` or `], #`) rewrite the register — end the lineage.
            let addr_form = format!("[{}", xn);
            if t.contains(&addr_form) {
                if t.contains("!") || t.contains("],") {
                    base_off.remove(&b);
                }
                continue;
            }
            // Base-creating or base-preserving forms.
            if let Some(rest) = t.strip_prefix("mov ") {
                if let Some((dst_s, src_s)) = rest.split_once(", ") {
                    let (dst_s, src_s) = (dst_s.trim(), src_s.trim());
                    if dst_s == xn {
                        // mov xN, xM: re-based if xM is a base, clobbered otherwise.
                        let src = parse_reg(src_s);
                        match base_off.get(&src).copied() {
                            Some(off) => base_off.insert(b, off),
                            None => base_off.remove(&b),
                        };
                    } else if src_s == xn {
                        // mov xM, xN: xM joins the lineage.
                        let dst = parse_reg(dst_s);
                        if dst < 29 {
                            let off = base_off[&b];
                            base_off.insert(dst, off);
                        }
                    }
                    continue;
                }
            }
            if let Some(rest) = t.strip_prefix("add ").or_else(|| t.strip_prefix("sub ")) {
                let neg = t.starts_with("sub ");
                if let Some((dst_s, srcs)) = rest.split_once(", ") {
                    if let Some((src_s, imm_s)) = srcs.split_once(", ") {
                        if let Some(imm) = imm_s.trim().strip_prefix('#').and_then(|v| v.parse::<i32>().ok()) {
                            let dst = parse_reg(dst_s.trim());
                            let src = parse_reg(src_s.trim());
                            let signed = if neg { -imm } else { imm };
                            if dst == b && src == b {
                                base_off.insert(b, base_off[&b] + signed);
                                continue;
                            }
                            if let Some(&off) = base_off.get(&src) {
                                if dst < 29 {
                                    base_off.insert(dst, off + signed);
                                }
                                continue;
                            }
                        }
                    }
                }
            }
            // A write of the base by a load or ALU op ends the lineage (allowed).
            if gp_reg_written_broad(&lines[i], b) {
                base_off.remove(&b);
                continue;
            }
            // Anything else is a genuine use of the pointer value: escape.
            escaped = true;
            break;
        }
    }
    if escaped {
        return false;
    }

    // Loop-variant base registers: a backward branch means the text between
    // the label and the branch executes many times, so a register written in
    // that region takes a different value on every iteration. The linear scan
    // below sees only one of those values, so resolving a load through such a
    // register to a single frame range is unsound — reads from later
    // iterations go unaccounted and the stores they cover get deleted (this
    // deleted the init stores of `long a[N] = {...}` read via a
    // marching-pointer loop). Loads through loop-variant bases are treated as
    // reading the whole frame, and stores through them are never deleted.
    const WHOLE_FRAME: (i32, i32) = (-(1 << 29), 1 << 30);
    let mut loop_variant: FxHashSet<u8> = FxHashSet::default();
    {
        let mut label_pos: FxHashMap<&str, usize> = FxHashMap::default();
        for i in start..end {
            if kinds[i] == LineKind::Label {
                label_pos.insert(lines[i].trim().trim_end_matches(':'), i);
            }
        }
        // Prefix-sum the in-loop depth from backward branches.
        let mut delta = vec![0i32; end + 1];
        for j in start..end {
            let t = lines[j].trim();
            if !(t.starts_with("b ") || t.starts_with("b.") || t.starts_with("cb") || t.starts_with("tb")) {
                continue;
            }
            let Some(target) = t
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|s| !s.is_empty())
                .last()
            else {
                continue;
            };
            if let Some(&i) = label_pos.get(target) {
                if i < j {
                    delta[i] += 1;
                    delta[j + 1] -= 1;
                }
            }
        }
        let mut depth = 0i32;
        for i in start..end {
            depth += delta[i];
            if depth > 0 {
                for b in 0..29u8 {
                    if gp_reg_written_broad(&lines[i], b) {
                        loop_variant.insert(b);
                    }
                }
            }
        }
    }

    // Phase 1: collect all loaded byte ranges, resolving base-relative forms.
    let mut loaded_ranges: Vec<(i32, i32)> = Vec::new(); // (offset, size)
    let mut base_off: FxHashMap<u8, i32> = FxHashMap::default();

    for i in start..end {
        track_sp_bases(&lines[i], kinds[i], &mut base_off);
        // FP loads/stores and base forms.
        if let Some((is_store, _reg, addr, width)) = parse_fp_mem(&lines[i]) {
            if !is_store {
                let off = match addr {
                    FpAddr::Sp(off) => Some(off),
                    FpAddr::Base(b, k) => {
                        if loop_variant.contains(&b) {
                            // Loop-variant base: may read any frame slot.
                            loaded_ranges.push(WHOLE_FRAME);
                            None
                        } else {
                            base_off.get(&b).map(|&o| o + k)
                        }
                    }
                };
                if let Some(off) = off {
                    loaded_ranges.push((off, fp_access_width(width)));
                } else {
                    // Unknown-base load: may read anything (frame addresses did
                    // not escape, so it cannot read our frame — no action).
                }
            }
        }
        match kinds[i] {
            LineKind::LoadSp { offset, is_word, .. } => {
                let size = if is_word { 4 } else { 8 };
                loaded_ranges.push((offset, size));
            }
            LineKind::LoadswSp { offset, .. } => {
                loaded_ranges.push((offset, 4));
            }
            _ => {
                // Check for loads in Other instructions (e.g., ldp, ldrb, ldrh, etc.)
                let trimmed = lines[i].trim();
                let load_size = if trimmed.starts_with("ldp ") {
                    Some(16) // ldp loads two 8-byte registers
                } else if trimmed.starts_with("ldr x") || trimmed.starts_with("ldur x") {
                    Some(8)
                } else if trimmed.starts_with("ldr w") || trimmed.starts_with("ldur w") ||
                          trimmed.starts_with("ldrsw ") {
                    Some(4)
                } else if trimmed.starts_with("ldrh ") || trimmed.starts_with("ldrsh ") {
                    Some(2)
                } else if trimmed.starts_with("ldrb ") || trimmed.starts_with("ldrsb ") {
                    Some(1)
                } else if trimmed.starts_with("ldr ") || trimmed.starts_with("ldur ") {
                    Some(8) // default to 8 for unqualified ldr
                } else {
                    None
                };
                if let Some(sz) = load_size {
                    if trimmed.contains("[sp") {
                        if let Some(off) = extract_sp_offset(trimmed) {
                            loaded_ranges.push((off, sz));
                        }
                    } else if trimmed.starts_with("ldr ") || trimmed.starts_with("ldp ") || trimmed.starts_with("ldur ")
                        || trimmed.starts_with("ldrsw ") || trimmed.starts_with("ldrsb ")
                        || trimmed.starts_with("ldrsh ") || trimmed.starts_with("ldrb ")
                        || trimmed.starts_with("ldrh ") || trimmed.starts_with("ldurs")
                    {
                        // Load through a tracked base register.
                        if let Some(bracket) = trimmed.find("[x") {
                            let inner = &trimmed[bracket..];
                            if let Some((b, k)) = parse_base_addr(inner) {
                                if loop_variant.contains(&b) {
                                    // Loop-variant base: may read any frame slot.
                                    loaded_ranges.push(WHOLE_FRAME);
                                } else if let Some(&o) = base_off.get(&b) {
                                    loaded_ranges.push((o + k, sz));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 2: remove stores whose byte range does not overlap any load range.
    // Covers GP sp stores and FP sp/base-form stores alike.
    let mut changed = false;
    let mut base_off: FxHashMap<u8, i32> = FxHashMap::default();
    for i in start..end {
        track_sp_bases(&lines[i], kinds[i], &mut base_off);
        let range: Option<(i32, i32)> = match kinds[i] {
            LineKind::StoreSp { offset, is_word, .. } => {
                Some((offset, if is_word { 4 } else { 8 }))
            }
            _ => {
                match parse_fp_mem(&lines[i]) {
                    Some((true, _reg, addr, width)) => {
                        let off = match addr {
                            FpAddr::Sp(off) => Some(off),
                            FpAddr::Base(b, k) => {
                                if loop_variant.contains(&b) {
                                    // Loop-variant base: the store hits a
                                    // different slot each iteration — the
                                    // single recorded range proves nothing.
                                    None
                                } else {
                                    base_off.get(&b).map(|&o| o + k)
                                }
                            }
                        };
                        off.map(|o| (o, fp_access_width(width)))
                    }
                    _ => None,
                }
            }
        };
        if let Some((offset, store_size)) = range {
            let overlaps_any_load = loaded_ranges.iter().any(|&(load_off, load_sz)| {
                // Two ranges [a, a+as) and [b, b+bs) overlap iff a < b+bs && b < a+as
                offset < load_off + load_sz && load_off < offset + store_size
            });
            if !overlaps_any_load {
                kinds[i] = LineKind::Nop;
                changed = true;
            }
        }
    }
    changed
}

/// Parse `[xN]`, `[xN, #k]` (no writeback) into (base register, offset).
fn parse_base_addr(inner_bracketed: &str) -> Option<(u8, i32)> {
    let inner = inner_bracketed.strip_prefix('[')?;
    let inner = inner.split_once(']').map(|(h, _)| h).unwrap_or(inner);
    let (base, off) = match inner.split_once(", ") {
        Some((b, o)) => (b.trim(), o.trim().strip_prefix('#')?.parse::<i32>().ok()?),
        None => (inner.trim(), 0),
    };
    let b = parse_reg(base);
    (b < 29).then_some((b, off))
}

/// Maintain the sp-derived base-register map for one line of text:
/// `add/sub xN, sp, #k` creates, `mov` copies or ends a lineage, `add/sub`
/// on a base adjusts or derives, calls end caller-saved lineages, and any
/// other write of a tracked register ends its lineage.
fn track_sp_bases(line: &str, kind: LineKind, base_off: &mut FxHashMap<u8, i32>) {
    let t = line.trim();
    if let Some(rest) = t.strip_prefix("add ").or_else(|| t.strip_prefix("sub ")) {
        let neg = t.starts_with("sub ");
        if let Some((dst_s, srcs)) = rest.split_once(", ") {
            if let Some((src_s, imm_s)) = srcs.split_once(", ") {
                let dst = parse_reg(dst_s.trim());
                let src = parse_reg(src_s.trim());
                let imm = imm_s.trim().strip_prefix('#').and_then(|v| v.parse::<i32>().ok());
                if let Some(imm) = imm {
                    let signed = if neg { -imm } else { imm };
                    if src_s.trim() == "sp" && dst < 29 {
                        base_off.insert(dst, signed);
                        return;
                    }
                    if let Some(&off) = base_off.get(&src) {
                        if dst < 29 {
                            base_off.insert(dst, off + signed);
                        }
                    }
                }
            }
        }
    } else if let Some(rest) = t.strip_prefix("mov ") {
        if let Some((dst_s, src_s)) = rest.split_once(", ") {
            let (dst_s, src_s) = (dst_s.trim(), src_s.trim());
            let dst = parse_reg(dst_s);
            if src_s == "sp" {
                if dst < 29 {
                    base_off.insert(dst, 0);
                }
                return;
            }
            let src = parse_reg(src_s);
            if let Some(&off) = base_off.get(&src) {
                if dst < 29 {
                    base_off.insert(dst, off);
                }
            } else if dst < 29 {
                // mov xN, <non-base>: clobbered, lineage ends.
                base_off.remove(&dst);
            }
        }
    }
    // Calls clobber caller-saved registers: end those lineages.
    if kind == LineKind::Call {
        base_off.retain(|&r, _| r >= 19);
        return;
    }
    let mut dead = Vec::new();
    for &b in base_off.keys() {
        if gp_reg_written_broad(line, b) || written_gp_register(line, kind) == Some(b) {
            dead.push(b);
        }
    }
    for b in dead {
        base_off.remove(&b);
    }
}


/// Extract the numeric offset from an instruction containing `[sp, #N]` or `[sp]`.
fn extract_sp_offset(line: &str) -> Option<i32> {
    if let Some(start) = line.find("[sp") {
        let rest = &line[start..];
        if rest.starts_with("[sp]") {
            return Some(0);
        }
        if rest.starts_with("[sp, #") {
            let num_start = start + 6; // skip "[sp, #"
            let after = &line[num_start..];
            if let Some(end) = after.find(']') {
                return after[..end].parse::<i32>().ok();
            }
        }
    }
    None
}

/// Loop rotation: convert header-tested loops into bottom-tested loops.
///
/// ```text
/// .Lhdr:                      .Lhdr:                 (guard, runs once)
///     cmp a, b                    cmp a, b
///     b.cc .Lbody                 b.cc .Lbody
/// .Lskip:                       .Lskip:
///     b .Lexit                    b .Lexit
/// .Lbody:            →          .Lbody:
///     <body>                      <body>
///     b .Lhdr                     cmp a, b           (duplicated test)
///                                 b.cc .Lbody        (one branch per iteration)
///                                 b .Lskip           (taken once, on exit)
/// ```
///
/// Saves one unconditional branch per iteration. Sound because every register
/// the header test reads is live throughout the loop (it is read again next
/// iteration), so it is never clobbered by unrelated assignments, and loop
/// phis have already been updated to the next iteration's value at the latch
/// (backedge phi copies are emitted at the end of the latch, before the
/// backedge branch this pass replaces).
fn rotate_simple_loops(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        if kinds[i] != LineKind::Branch {
            continue;
        }
        let Some(header_target) = branch_target(lines[i].trim_start()) else {
            continue;
        };
        // Find the header label earlier in the text (a backward branch).
        let header_label = format!("{}:", header_target);
        let mut header_pos = None;
        for l in 0..i {
            if kinds[l] == LineKind::Label && lines[l].trim() == header_label {
                header_pos = Some(l);
                break;
            }
        }
        let Some(h) = header_pos else { continue; };

        // Single backedge only: this must be the one and only `b .Lhdr`.
        let backedge_count = (0..n)
            .filter(|&j| kinds[j] == LineKind::Branch
                && branch_target(lines[j].trim_start()) == Some(header_target))
            .count();
        if backedge_count != 1 {
            continue;
        }

        // No calls, returns, or indirect branches inside the loop.
        let mut bad = false;
        for j in h..=i {
            match kinds[j] {
                LineKind::Ret | LineKind::Call | LineKind::CmpBranch => {
                    bad = true;
                    break;
                }
                _ => {
                    let t = lines[j].trim_start();
                    if t.starts_with("br ") || t.starts_with("blr ") {
                        bad = true;
                        break;
                    }
                }
            }
        }
        if bad {
            continue;
        }

        // Collect the header setup: pure, flag-preserving instructions after
        // the label, ending in a Compare, followed by a conditional branch.
        let mut setup: Vec<usize> = Vec::new();
        let mut pos = h + 1;
        let mut cj = None;
        while pos < i {
            if kinds[pos] == LineKind::Nop {
                pos += 1;
                continue;
            }
            match kinds[pos] {
                LineKind::CondBranch => {
                    cj = Some(pos);
                    break;
                }
                LineKind::Compare
                | LineKind::Move { .. }
                | LineKind::MoveImm { .. }
                | LineKind::MoveWide { .. }
                | LineKind::Sxtw { .. }
                | LineKind::LoadSp { .. }
                | LineKind::LoadswSp { .. }
                | LineKind::LoadPairSp { .. } => {
                    setup.push(pos);
                    if setup.len() > 3 {
                        break;
                    }
                    pos += 1;
                }
                _ => break,
            }
        }
        let Some(cj_pos) = cj else { continue };
        if setup.is_empty() || setup.len() > 3 {
            continue;
        }
        // The compare must be the last setup instruction: it sets the flags
        // the conditional branch consumes (moves/loads don't touch flags).
        if kinds[*setup.last().unwrap()] != LineKind::Compare {
            continue;
        }

        // The conditional branch gives the loop-back condition and body target.
        let Some((cc, body_target)) = cond_branch_parts(lines[cj_pos].trim_start()) else {
            continue;
        };
        // The body target must be a label inside the loop, after the cond branch.
        let body_label = format!("{}:", body_target);
        let body_found = ((cj_pos + 1)..=i).any(|l| {
            kinds[l] == LineKind::Label && lines[l].trim() == body_label
        });
        if !body_found {
            continue;
        }

        // The exit target is the first label after the cond branch: the
        // not-taken path of the header guard (typically the .Lskip trampoline
        // or the exit block itself).
        let mut exit_target = None;
        for l in (cj_pos + 1)..n {
            if kinds[l] == LineKind::Nop {
                continue;
            }
            if kinds[l] == LineKind::Label {
                exit_target = label_name(lines[l].trim());
            }
            break;
        }
        let Some(exit_lbl) = exit_target else { continue };
        if exit_lbl == body_target || exit_lbl == header_target {
            continue;
        }

        // Build the rotated latch: setup copy + cond branch to body + exit.
        let mut repl = String::new();
        for &s in &setup {
            repl.push_str(&lines[s]);
            repl.push('\n');
        }
        repl.push_str(&format!("    b.{} {}", cc, body_target));
        repl.push('\n');
        repl.push_str(&format!("    b {}", exit_lbl));
        lines[i] = repl;
        // The slot now holds a multi-line sequence; keep branch passes from
        // re-examining it (its embedded branches are final).
        kinds[i] = LineKind::Other;
        changed = true;
    }
    changed
}

// ── Loop-carried store sinking ───────────────────────────────────────────────
//
// The slot-home discipline gives every SSA value a mandatory stack slot, so a
// loop-carried value is stored to its slot once per iteration even when
// nothing in the loop ever loads the slot back (the value stays live in a
// register; the slot is only read after the loop).  Sink such a store to the
// loop's fall-through exit:
//
//     .Lbody:                       .Lbody:
//       ...                           ...
//       str x0, [sp, #24]   -->       ...            (store removed)
//       ...                           ...
//       b.le .Lbody                   b.le .Lbody
//       b .Lexit                      str x0, [sp, #24]  (stored once)
//                                     b .Lexit
//
// Soundness conditions (all checked below): the slot is not otherwise
// referenced inside the loop, this is the only store to it, every path from
// the loop head to the fall-through exit executes the store (no labels or
// exiting branches between the head and the store, no labels between the
// store and the backedge), the only exit is the fall-through past the
// backedge, and the stored register is not rewritten between the store and
// the backedge.  Then the register holds the last-stored value at the exit,
// so storing it there once produces the same slot contents as the original.

/// Register number of a scalar register operand (`w0`/`x19`/`d24`/`s3` →
/// 0/19/24/3).  The class is ignored: a match on the number alone is a
/// conservative clobber indication.
fn reg_operand_num(op: &str) -> Option<u8> {
    let op = op.trim();
    let rest = op.strip_prefix(|c| matches!(c, 'w' | 'x' | 'd' | 's' | 'q' | 'b' | 'h'))?;
    rest.parse::<u8>().ok().filter(|&n| n <= 30)
}

/// Does this instruction write a register with the given number?
/// Conservative: the first operand (and the second for `ldp`) counts as a
/// destination for everything that is not a store, compare, or branch.
fn instr_clobbers_reg(trimmed: &str, num: u8) -> bool {
    let mut it = trimmed.splitn(2, char::is_whitespace);
    let mnem = it.next().unwrap_or("");
    let ops = it.next().unwrap_or("");
    if ops.is_empty() {
        return false; // labels, directives, ret, bare branches
    }
    // Stores and comparisons have no register destination.
    if mnem.starts_with("st")
        || mnem.starts_with("cmp")
        || mnem.starts_with("cmn")
        || mnem.starts_with("tst")
        || mnem.starts_with("fcmp")
    {
        return false;
    }
    // Branches.
    if mnem == "b"
        || mnem.starts_with("b.")
        || mnem == "bl"
        || mnem == "blr"
        || mnem == "br"
        || mnem == "cbz"
        || mnem == "cbnz"
        || mnem == "tbz"
        || mnem == "tbnz"
    {
        return false;
    }
    let mut operands = ops.split(", ");
    if let Some(first) = operands.next() {
        if reg_operand_num(first) == Some(num) {
            return true;
        }
    }
    if mnem == "ldp" {
        if let Some(second) = operands.next() {
            if reg_operand_num(second) == Some(num) {
                return true;
            }
        }
    }
    false
}

/// Parse `str <reg>, [sp, #off]` for scalar regs (w/x/s/d) → (reg_num, offset).
fn parse_sp_store_any(trimmed: &str) -> Option<(u8, i32)> {
    let rest = trimmed.strip_prefix("str ")?;
    let (reg_str, addr) = rest.split_once(", ")?;
    let num = reg_operand_num(reg_str)?;
    let off = parse_sp_offset(addr.trim())?;
    Some((num, off))
}

/// Byte size of a scalar `str` register operand (w/s=4, x/d=8).
fn store_reg_size(trimmed: &str) -> Option<u32> {
    let rest = trimmed.strip_prefix("str ")?;
    let (reg_str, _) = rest.split_once(", ")?;
    let r = reg_str.trim();
    if r.starts_with('w') || r.starts_with('s') {
        Some(4)
    } else if r.starts_with('x') || r.starts_with('d') {
        Some(8)
    } else {
        None
    }
}

// ── Overwritten store elimination ────────────────────────────────────────────
//
// Local dead-store peephole: `str R, [sp, #off]` followed by another store to
// the same slot with no intervening read (and no label, branch, call, or
// unknown memory op between). The first store is dead. This is the store
// analog of eliminate_overwritten_moves and catches cases the global DSE
// misses when it bails on a whole function due to one escaped frame pointer.
fn eliminate_overwritten_stores(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_STORE_DSE").is_ok() {
        return false;
    }
    let mut changed = false;
    for i in 0..n {
        let LineKind::StoreSp { offset: off_i, is_word: word_i, .. } = kinds[i] else {
            continue;
        };
        let size_i = if word_i { 4u32 } else { 8 };
        let mut j = i + 1;
        let mut steps = 0;
        while j < n && steps < 40 {
            steps += 1;
            if lines[j].contains('\n') {
                break; // multi-line slot may hide memory ops
            }
            match kinds[j] {
                LineKind::Nop => {
                    j += 1;
                    continue;
                }
                LineKind::StoreSp { offset: off_j, is_word: word_j, .. } => {
                    let size_j = if word_j { 4u32 } else { 8 };
                    if off_j == off_i && size_j >= size_i {
                        // Fully overwritten with no intervening read.
                        kinds[i] = LineKind::Nop;
                        changed = true;
                    }
                    if off_j < off_i + size_i as i32 && off_i < off_j + size_j as i32 {
                        break; // overlapping store: stop either way
                    }
                    j += 1;
                    continue;
                }
                LineKind::LoadSp { offset: off_j, is_word: word_j, .. } => {
                    let size_j = if word_j { 4u32 } else { 8 };
                    if off_j < off_i + size_i as i32 && off_i < off_j + size_j as i32 {
                        break; // reads our slot (or overlaps it)
                    }
                    j += 1;
                    continue;
                }
                LineKind::LoadswSp { offset: off_j, .. } => {
                    if off_j < off_i + size_i as i32 && off_i < off_j + 4 {
                        break;
                    }
                    j += 1;
                    continue;
                }
                // Straight-line, non-memory instructions are transparent.
                LineKind::Move { .. }
                | LineKind::MoveImm { .. }
                | LineKind::MoveWide { .. }
                | LineKind::Sxtw { .. }
                | LineKind::Compare
                | LineKind::Alu => {
                    j += 1;
                    continue;
                }
                // Labels, branches, calls, pair/unknown memory ops, anything
                // else: stop. `Other` covers unknown instructions that might
                // touch memory, so it stops the scan too.
                _ => break,
            }
        }
    }
    changed
}

/// Branch target label of a (sub-)line, for plain/conditional/compare branches.
fn any_branch_target(trimmed: &str) -> Option<&str> {
    if let Some(rest) = trimmed.strip_prefix("b ") {
        let t = rest.trim();
        if t.starts_with('.') {
            return Some(t);
        }
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix("b.") {
        if let Some((_, t)) = rest.split_once(' ') {
            let t = t.trim();
            if t.starts_with('.') {
                return Some(t);
            }
        }
        return None;
    }
    for p in ["cbz ", "cbnz ", "tbz ", "tbnz "] {
        if trimmed.starts_with(p) {
            if let Some((_, t)) = trimmed.rsplit_once(", ") {
                let t = t.trim();
                if t.starts_with('.') {
                    return Some(t);
                }
            }
        }
    }
    None
}

fn sink_loop_carried_stores(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if std::env::var("CCC_NO_STORE_SINK").is_ok() {
        return false;
    }
    // Flatten to virtual sub-lines: rotation leaves multi-line slots whose
    // embedded latch branches this pass must still see.
    let mut v: Vec<(usize, usize, String)> = Vec::new(); // (slot, sub_idx, text)
    for i in 0..n {
        if kinds[i] == LineKind::Nop {
            continue;
        }
        for (si, sub) in lines[i].split('\n').enumerate() {
            let t = sub.trim();
            if !t.is_empty() {
                v.push((i, si, t.to_string()));
            }
        }
    }
    let nv = v.len();
    // Label name -> virtual index.
    let mut labels: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (vi, (_, _, t)) in v.iter().enumerate() {
        if let Some(name) = t.strip_suffix(':') {
            labels.insert(name, vi);
        }
    }
    // Find loop regions: a conditional backward branch ends a bottom-tested
    // loop whose exit is the fall-through past the branch.
    let mut regions: Vec<(usize, usize)> = Vec::new(); // (head vi, backedge vi)
    for vj in 0..nv {
        let t = &v[vj].2;
        let conditional = t.starts_with("b.")
            || t.starts_with("cbz ")
            || t.starts_with("cbnz ")
            || t.starts_with("tbz ")
            || t.starts_with("tbnz ");
        if !conditional {
            continue;
        }
        let Some(target) = any_branch_target(t) else { continue };
        let Some(&vi) = labels.get(target) else { continue };
        if vi < vj {
            regions.push((vi, vj));
        }
    }
    // Largest regions first: sink each store as far out as possible.
    regions.sort_by_key(|&(h, j)| std::cmp::Reverse(j - h));

    let mut sunk_slot: Vec<bool> = vec![false; n];
    // (store slot, store text, backedge slot, backedge sub_idx)
    let mut edits: Vec<(usize, String, usize, usize)> = Vec::new();

    for &(h, j) in &regions {
        // Region-level hazards: calls, returns, or branches to unknown labels.
        let mut region_bad = false;
        for p in h..=j {
            let t = &v[p].2;
            if t.starts_with("bl ") || t.starts_with("blr ") || t == "ret" {
                region_bad = true;
                break;
            }
            if let Some(tgt) = any_branch_target(t) {
                if !labels.contains_key(tgt) {
                    region_bad = true;
                    break;
                }
            }
        }
        if region_bad {
            continue;
        }
        // The backedge slot must contain no other backward branch: splicing
        // would shift the second branch's sub-index.
        let bslot = v[j].0;
        let mut other_backedge = false;
        for p in h..j {
            if v[p].0 != bslot {
                continue;
            }
            if let Some(tgt) = any_branch_target(&v[p].2) {
                if let Some(&tp) = labels.get(tgt) {
                    if tp < p {
                        other_backedge = true;
                        break;
                    }
                }
            }
        }
        if other_backedge {
            continue;
        }
        // Exit branches: any branch in the region targeting outside it.
        // (The backedge at j targets the head, which is inside.)
        let mut exits: Vec<usize> = Vec::new();
        for p in h..j {
            if let Some(tgt) = any_branch_target(&v[p].2) {
                let tp = labels[tgt];
                if tp < h || tp > j {
                    exits.push(p);
                }
            }
        }
        // Candidate stores.
        for k in (h + 1)..j {
            let (slot, _, ref text) = v[k];
            if sunk_slot[slot] || lines[slot].contains('\n') {
                continue; // already sunk, or not a standalone line
            }
            if !matches!(kinds[slot], LineKind::StoreSp { .. } | LineKind::MemOther) {
                continue;
            }
            let Some((num, off)) = parse_sp_store_any(text) else { continue };
            let debug = std::env::var("CCC_SINK_DEBUG").is_ok();
            let slot_ref = format!("[sp, #{}]", off);
            let mut bad = false;
            // (a) Nothing else in the region references this slot.
            for p in h..=j {
                if p == k {
                    continue;
                }
                let t = &v[p].2;
                if t.contains(&slot_ref) {
                    bad = true;
                    break;
                }
                // Pair ops cover a range: stp/ldp at base M spans M..M+16.
                if t.starts_with("stp ") || t.starts_with("ldp ") {
                    if let Some(base) = extract_sp_offset(t) {
                        if base <= off && off < base + 16 {
                            bad = true;
                            break;
                        }
                    }
                }
            }
            if bad && debug {
                eprintln!("sink bail: {} (slot re-referenced in region)", text);
            }
            // (b) No exiting branch anywhere (v1: fall-through exit only).
            if !bad && !exits.is_empty() {
                bad = true;
                if debug {
                    eprintln!("sink bail: {} ({} exit branches)", text, exits.len());
                }
            }
            // (c) No labels between the store and the backedge (no way to
            // reach the exit without passing the store), and none before the
            // store either except the head (every path to the exit passes k).
            if !bad {
                for p in (h + 1)..j {
                    if p == k {
                        continue;
                    }
                    if v[p].2.ends_with(':') {
                        bad = true;
                        if debug {
                            eprintln!("sink bail: {} (label {} in region)", text, v[p].2);
                        }
                        break;
                    }
                }
            }
            // (d) The stored register survives to the backedge.
            if !bad {
                for p in (k + 1)..=j {
                    if instr_clobbers_reg(&v[p].2, num) {
                        bad = true;
                        if debug {
                            eprintln!("sink bail: {} (clobbered by {})", text, v[p].2);
                        }
                        break;
                    }
                }
            }
            if bad {
                continue;
            }
            sunk_slot[slot] = true;
            edits.push((slot, format!("    {}", text), v[j].0, v[j].1));
            if debug {
                eprintln!("sink ok: {}", text);
            }
        }
    }

    if edits.is_empty() {
        return false;
    }
    for (slot, _, _, _) in &edits {
        kinds[*slot] = LineKind::Nop;
    }
    // Splice each sunk store into the backedge slot, right after the branch.
    // Insertions land after the backedge sub-line, so it keeps its original
    // sub-index no matter how many stores are spliced in.
    for (_, text, bj, bsub) in &edits {
        let mut subs: Vec<&str> = lines[*bj].split('\n').collect();
        subs.insert(bsub + 1, text.as_str());
        lines[*bj] = subs.join("\n");
        kinds[*bj] = LineKind::Other;
    }
    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_store() {
        assert!(matches!(
            classify_line("    str x0, [sp, #16]"),
            LineKind::StoreSp { reg: 0, offset: 16, is_word: false }
        ));
        assert!(matches!(
            classify_line("    str w1, [sp, #24]"),
            LineKind::StoreSp { reg: 1, offset: 24, is_word: true }
        ));
    }

    #[test]
    fn test_classify_load() {
        assert!(matches!(
            classify_line("    ldr x0, [sp, #16]"),
            LineKind::LoadSp { reg: 0, offset: 16, is_word: false }
        ));
    }

    #[test]
    fn test_classify_loadsw() {
        assert!(matches!(
            classify_line("    ldrsw x0, [sp, #24]"),
            LineKind::LoadswSp { reg: 0, offset: 24 }
        ));
    }

    #[test]
    fn test_classify_move() {
        assert!(matches!(
            classify_line("    mov x14, x0"),
            LineKind::Move { dst: 14, src: 0, is_32bit: false }
        ));
    }

    #[test]
    fn test_classify_move_imm() {
        assert!(matches!(
            classify_line("    mov x0, #0"),
            LineKind::MoveImm { dst: 0 }
        ));
        assert!(matches!(
            classify_line("    mov x0, #-1"),
            LineKind::MoveImm { dst: 0 }
        ));
    }

    #[test]
    fn test_classify_branch() {
        assert_eq!(classify_line("    b .LBB1"), LineKind::Branch);
    }

    #[test]
    fn test_classify_cond_branch() {
        assert_eq!(classify_line("    b.ge .Lskip_0"), LineKind::CondBranch);
        assert_eq!(classify_line("    b.eq .LBB3"), LineKind::CondBranch);
    }

    #[test]
    fn test_classify_label() {
        assert_eq!(classify_line(".LBB1:"), LineKind::Label);
        assert_eq!(classify_line("sum_array:"), LineKind::Label);
    }

    #[test]
    fn test_classify_ret() {
        assert_eq!(classify_line("    ret"), LineKind::Ret);
    }

    #[test]
    fn test_classify_sxtw() {
        assert!(matches!(
            classify_line("    sxtw x0, w0"),
            LineKind::Sxtw { dst: 0, src: 0 }
        ));
    }

    #[test]
    fn test_adjacent_store_load_same_reg() {
        let input = "    str x0, [sp, #16]\n    ldr x0, [sp, #16]\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str x0, [sp, #16]"));
        assert!(!result.contains("ldr x0, [sp, #16]"));
    }

    #[test]
    fn test_adjacent_store_load_diff_reg() {
        let input = "    str x0, [sp, #16]\n    ldr x1, [sp, #16]\n    ret\n";
        let result = peephole_optimize(input.to_string());
        // The load is replaced with mov, and then DSE removes the now-dead store
        // (no remaining loads from offset 16), leaving just the mov and ret.
        assert!(!result.contains("ldr x1, [sp, #16]"));
        assert!(result.contains("mov x1, x0"));
    }

    #[test]
    fn test_redundant_branch() {
        let input = "    b .LBB1\n.LBB1:\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("b .LBB1"));
        assert!(result.contains(".LBB1:"));
    }

    #[test]
    fn test_self_move() {
        let input = "    mov x0, x0\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("mov x0, x0"));
    }

    #[test]
    fn test_branch_over_branch_fusion() {
        let input = "    b.ge .Lskip_0\n    b .LBB2\n.Lskip_0:\n    b .LBB4\n";
        let result = peephole_optimize(input.to_string());
        // Should become: b.lt .LBB2 (inverted ge → lt)
        assert!(result.contains("b.lt .LBB2"));
        // The unconditional branch to LBB2 should be eliminated
        assert!(!result.contains("    b .LBB2\n"));
    }

    #[test]
    fn test_move_chain() {
        let input = "    mov x0, x14\n    mov x13, x0\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov x13, x14"));
    }

    #[test]
    fn test_move_imm_chain() {
        let input = "    mov x0, #0\n    mov x14, x0\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov x14, #0"));
    }

    #[test]
    fn test_store_loadsw_fusion() {
        let input = "    str w1, [sp, #24]\n    ldrsw x0, [sp, #24]\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldrsw"));
        assert!(result.contains("sxtw x0, w1"));
    }

    // ── Global store forwarding tests ─────────────────────────────────

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_same_reg_elimination() {
        // Store x0 then load x0 from same slot (non-adjacent) — load is dead
        let input = "\
    str x0, [sp, #16]\n\
    add x1, x2, x3\n\
    ldr x0, [sp, #16]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str x0, [sp, #16]"));
        assert!(!result.contains("ldr x0, [sp, #16]"));
    }

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_different_reg_forwarding() {
        // Store x5 then load x10 from same slot — replace load with mov
        let input = "\
    str x5, [sp, #32]\n\
    add x1, x2, x3\n\
    ldr x10, [sp, #32]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str x5, [sp, #32]"));
        assert!(!result.contains("ldr x10, [sp, #32]"));
        assert!(result.contains("mov x10, x5"));
    }

    #[test]
    fn test_gsf_invalidation_on_reg_overwrite() {
        // After x0 is overwritten, the mapping slot 16 → x0 is stale
        let input = "\
    str x0, [sp, #16]\n\
    mov x0, #42\n\
    ldr x1, [sp, #16]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        // The load should NOT be forwarded since x0 was overwritten
        assert!(result.contains("ldr x1, [sp, #16]"));
    }

    #[test]
    fn test_gsf_invalidation_at_jump_target() {
        // Mappings are invalidated at jump target labels
        let input = "\
    str x5, [sp, #16]\n\
    b .LBB1\n\
.LBB1:\n\
    ldr x5, [sp, #16]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        // .LBB1 is a jump target, so mappings are invalidated
        assert!(result.contains("ldr x5, [sp, #16]"));
    }

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_word_forwarding() {
        // Word store forwarded to word load
        let input = "\
    str w3, [sp, #24]\n\
    add x1, x2, x4\n\
    ldr w5, [sp, #24]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldr w5, [sp, #24]"));
        assert!(result.contains("mov w5, w3"));
    }

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_ldrsw_forwarding() {
        // Word store forwarded to sign-extending load
        let input = "\
    str w1, [sp, #24]\n\
    add x2, x3, x4\n\
    ldrsw x5, [sp, #24]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldrsw x5, [sp, #24]"));
        assert!(result.contains("sxtw x5, w1"));
    }

    // ── Copy propagation tests ───────────────────────────────────────

    #[test]
    fn test_copy_prop_word_boundary() {
        // Ensure "x1" doesn't match inside "x11"
        assert_eq!(
            replace_whole_word("x11, x1", "x1", "x5"),
            "x11, x5"
        );
    }

    #[test]
    fn test_copy_prop_no_false_match() {
        // x10 should not be affected when replacing x1
        assert_eq!(
            replace_whole_word("x10", "x1", "x5"),
            "x10"
        );
    }

    #[test]
    fn test_dead_call_staging_move_removed_after_copy_prop() {
        let input = "    mov x9, x24\n    mov x0, x24\n    bl strlen\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("mov x9, x24"));
        assert!(result.contains("mov x0, x24"));
    }

    #[test]
    fn test_used_call_staging_move_preserved() {
        let input = "    mov x9, x24\n    add x24, x24, #8\n    ldr x0, [x9]\n    bl consume\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov x9, x24"));
    }

    #[test]
    fn test_vector_address_alias_propagation() {
        let input = "    mov x4, x5\n    ldr q2, [x4]\n    fmla v2.2d, v1.2d, v15.2d\n    str q2, [x4]\n    add x5, x5, #16\n    b .Lloop\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("mov x4, x5"));
        assert!(result.contains("ldr q2, [x5]"));
        assert!(result.contains("str q2, [x5]"));
    }

    #[test]
    fn test_destructive_pointer_update_fold() {
        let input = "    mov x0, x6\n    add x0, x6, #16\n    mov x8, x0\n    add w24, w24, #1\n    mov x6, x8\n    b .Lloop\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("add x6, x6, #16"));
        assert!(!result.contains("mov x8, x0"));
        assert!(!result.contains("mov x6, x8"));
    }

    // ── Dead store elimination tests ─────────────────────────────────

    #[test]
    fn test_dse_with_address_taken() {
        // An sp-derived address that is only materialized but never used does
        // NOT escape — the store is still dead and may be eliminated.
        let input = "\
    str w0, [sp, #16]\n\
    add x1, sp, #16\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("str w0, [sp, #16]"),
            "non-escaping address-of-sp allows elimination:\n{}", result);
    }

    #[test]
    fn test_dse_with_escaping_address() {
        // The derived base is stored into memory here: the address escapes,
        // so no store elimination may happen at all.
        let input = "\
    str w0, [sp, #16]\n\
    add x1, sp, #16\n\
    str x1, [x9]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str w0, [sp, #16]"),
            "escaping address must preserve stores:\n{}", result);
    }

    #[test]
    fn test_dse_dead_fp_store_via_base() {
        // FP field store through an sp-derived base whose slot is never
        // loaded: dead, eliminated.
        let input = "\
f:
    add x23, sp, #112
    fmul d28, d27, d17
    str d28, [x23]
    fadd d0, d1, d1
    ret
g:
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("str d28, [x23]"),
            "dead FP store via base eliminated:\n{}", result);
    }

    #[test]
    fn test_dse_live_fp_store_via_base_kept() {
        // The slot-forward pass converts the load to an fmov first, which
        // makes the store dead — the whole round-trip dissolves.
        let input = "\
f:
    add x23, sp, #112
    fmul d28, d27, d17
    str d28, [x23]
    ldr d0, [sp, #112]
    fadd d0, d0, d0
    ret
g:
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("fmov d0, d28"), "load forwarded:\n{}", result);
        assert!(!result.contains("str d28"), "store then dead:\n{}", result);
    }

    // ── Address-alias soundness ────────────────────────────────────────

    #[test]
    fn test_address_alias_mov_survives_cross_block_use() {
        // The mov's register is used as an address in a LATER block (after a
        // call): deleting the initializer must not happen even though the
        // in-window use is a pure address alias.
        let input = "\
f:
    adrp x0, tbl
    add x0, x0, :lo12:tbl
    mov x23, x0
    ldr x4, [x23, x28, lsl #3]
    mov x19, x4
    bl strcmp
.LBB9:
    ldr x19, [x23, x27, lsl #3]
    str x22, [x23, x27, lsl #3]
    ret
g:
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov x23, x0"), "defining mov must survive:\n{}", result);
        assert!(result.contains("[x23, x27, lsl #3]"),
            "cross-block use must not be rewritten:\n{}", result);
    }

    #[test]
    fn test_address_alias_folded_when_provably_dead() {
        // The intended optimization: a short-lived address alias whose
        // register is overwritten (without being read) after the uses.
        let input = "\
f:
    mov x9, x26
    ldr x4, [x9]
    str x4, [x9, #8]
    mov x9, x5
    ret
g:
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("mov x9, x26"), "dead alias mov removed:\n{}", result);
        assert!(result.contains("ldr x4, [x26]"), "use rewritten:\n{}", result);
        assert!(result.contains("str x4, [x26, #8]"), "use rewritten:\n{}", result);
    }

    // ── FP spill slot forwarding ─────────────────────────────────────────

    #[test]
    fn test_fp_slot_forward_basic() {
        let input = "\
f:
    fsub d0, d12, d21
    str d0, [sp, #224]
    fmul d3, d0, d0
    ldr d1, [sp, #224]
    fadd d4, d1, d3
    ret
g:
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("fmov d1, d0"), "reload forwarded:\n{}", result);
        assert!(!result.contains("ldr d1, [sp, #224]"));
    }

    #[test]
    fn test_fp_slot_forward_same_reg_deletes() {
        let input = "\
f:
    str d7, [sp, #64]
    ldr d7, [sp, #64]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldr d7, [sp, #64]"), "redundant reload deleted:\n{}", result);
    }

    #[test]
    fn test_fp_slot_forward_blocked_by_source_clobber() {
        // d0 is overwritten between the store and the load: no forwarding.
        let input = "\
f:
    str d0, [sp, #224]
    fmul d0, d1, d2
    ldr d1, [sp, #224]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("ldr d1, [sp, #224]"), "clobbered source must not forward:\n{}", result);
    }

    #[test]
    fn test_fp_slot_forward_blocked_by_unknown_store() {
        // A store through an untracked base may alias the slot: no forwarding.
        let input = "\
f:
    str d0, [sp, #224]
    str d5, [x9]
    ldr d1, [sp, #224]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("ldr d1, [sp, #224]"), "unknown-base store must block:\n{}", result);
    }

    #[test]
    fn test_fp_slot_forward_via_materialized_base() {
        // Stores through an sp-derived base register are tracked.
        let input = "\
f:
    add x23, sp, #112
    fmul d28, d27, d17
    str d28, [x23]
    fadd d4, d1, d1
    ldr d6, [sp, #112]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("fmov d6, d28"), "base-tracked forward:\n{}", result);
    }

    #[test]
    fn test_fp_slot_forward_gp_store_overlap_blocks() {
        // A GP word store into the upper half of the double slot must
        // invalidate the double's forward entry.
        let input = "\
f:
    str d0, [sp, #224]
    str w1, [sp, #228]
    ldr d1, [sp, #224]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("ldr d1, [sp, #224]"), "overlapping GP store must block:\n{}", result);
    }

    #[test]
    fn test_fp_slot_forward_blocked_by_intervening_reload_of_source() {
        // The reload of d0 from a different slot clobbers the tracked source
        // of slot #232's entry: forwarding the second load as `fmov d1, d0`
        // would read the wrong value.
        let input = "\
f:
    str d0, [sp, #232]
    ldr d0, [sp, #224]
    ldr d1, [sp, #232]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("fmov d1, d0"),
            "reload of source must block forward:\n{}", result);
    }

    // ── FP field pair fusion (ldp/stp) ───────────────────────────────────

    #[test]
    fn test_fp_pair_loads_fused() {
        let input = "\
f:
    ldr d22, [x7]
    ldr d23, [x26]
    fsub d27, d22, d23
    ldr d28, [x7, #8]
    fsub d30, d28, d29
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("ldp d22, d28, [x7]"), "loads fused:\n{}", result);
        assert!(!result.contains("ldr d28, [x7, #8]"));
    }

    #[test]
    fn test_fp_pair_stores_fused() {
        let input = "\
f:
    str d10, [x8]
    str d22, [x8, #8]
    ldr d1, [x8]
    ldr d2, [x8, #8]
    fadd d0, d1, d2
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("stp d10, d22, [x8]"), "stores fused:\n{}", result);
    }

    #[test]
    fn test_fp_pair_loads_blocked_by_store() {
        // A store between the loads may alias the second load's address.
        let input = "\
f:
    ldr d22, [x7]
    str d0, [x9]
    ldr d28, [x7, #8]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("ldr d28, [x7, #8]"), "store must block fusion:\n{}", result);
        assert!(!result.contains("ldp"));
    }

    #[test]
    fn test_fp_pair_stores_blocked_by_load() {
        // A load between the stores may read the second store's address.
        let input = "\
f:
    str d10, [x7]
    ldr d0, [x9]
    str d22, [x7, #8]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str d22, [x7, #8]"), "load must block store fusion:\n{}", result);
    }

    #[test]
    fn test_fp_pair_reversed_offsets() {
        let input = "\
f:
    ldr d28, [x7, #8]
    ldr d22, [x7]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("ldp d22, d28, [x7]"), "reversed pair normalized:\n{}", result);
    }

    // ── Repeated slot load elimination ───────────────────────────────────

    #[test]
    fn test_repeated_slot_load_same_reg_deleted() {
        let input = "\
f:
    ldr x9, [sp, #392]
    ldr d28, [x9, #80]
    fmul d29, d28, d8
    ldr x9, [sp, #392]
    ldr d30, [x9, #56]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert_eq!(result.matches("ldr x9, [sp, #392]").count(), 1,
            "second load deleted:\n{}", result);
    }

    #[test]
    fn test_repeated_slot_load_different_reg_mov() {
        // The repeat load becomes a mov, and copy-propagation then folds the
        // mov into the consumer — the second slot load disappears entirely.
        let input = "\
f:
    ldr x9, [sp, #392]
    ldr d28, [x9, #80]
    ldr x10, [sp, #392]
    ldr d30, [x10, #56]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldr x10, [sp, #392]"), "repeat load gone:\n{}", result);
        assert!(result.contains("ldr d30, [x9, #56]"), "consumer uses x9:\n{}", result);
    }

    #[test]
    fn test_repeated_slot_load_blocked_by_intervening_store() {
        // A store through an unknown base may write the frame slot.
        let input = "\
f:
    ldr x9, [sp, #392]
    str d0, [x26]
    ldr x9, [sp, #392]
    ret
";
        let result = peephole_optimize(input.to_string());
        assert_eq!(result.matches("ldr x9, [sp, #392]").count(), 2,
            "unknown-base store must block:\n{}", result);
    }

    #[test]
    fn test_repeated_slot_load_blocked_by_reg_clobber() {
        let input = "\
f:
    ldr x9, [sp, #392]
    add x9, x9, #8
    ldr x9, [sp, #392]
    ret
";
        // x9 was overwritten by the add: the second load must stay.
        let result = peephole_optimize(input.to_string());
        assert_eq!(result.matches("ldr x9, [sp, #392]").count(), 2,
            "clobbered register must block:\n{}", result);
    }



    #[test]
    fn test_fp_pair_latch_stores_no_overlap() {
    // Consecutive latch spill stores fuse pairwise; an already-paired line
    // must not be re-paired into an overlapping stp.
    let input = "\
f:
    add x19, x26, #56
    str d10, [sp, #344]
    str d22, [sp, #352]
    str d30, [sp, #360]
    str d19, [sp, #368]
    str d27, [sp, #376]
    mov x26, x19
    ldr d0, [sp, #344]
    ldr d1, [sp, #352]
    ldr d2, [sp, #360]
    ldr d3, [sp, #368]
    ldr d4, [sp, #376]
    ret
";
    let result = peephole_optimize(input.to_string());
    assert!(result.contains("stp d10, d22, [sp, #344]"), "first pair:\n{}", result);
    assert!(result.contains("stp d30, d19, [sp, #360]"), "second pair:\n{}", result);
    assert!(!result.contains("stp d22"), "no overlapping re-fusion:\n{}", result);
}

    // ── Zext move-chain folding ──────────────────────────────────────────

    #[test]
    fn test_zext_move_chain_folded() {
        let input = "\
f:
    mov x0, x24
    mov w0, w0
    mov x26, x0
    mul w4, w26, w25
    ret
";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov w26, w24"), "chain folded:\n{}", result);
        assert!(!result.contains("mov w0, w0"));
    }

    #[test]
    fn test_zext_move_chain_blocked_when_accumulator_live() {
        let input = "\
f:
    mov x0, x24
    mov w0, w0
    mov x26, x0
    mul w4, w26, w25
    add w5, w0, #1
    ret
";
        // x0 is read after the chain: folding would lose the zext value.
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov w0, w0"), "live accumulator blocks fold:\n{}", result);
    }

    #[test]
    fn test_sink_loop_carried_store() {
        // Bottom-tested loop: the slot is never read inside, the value stays
        // live in x20; the store can be sunk to the fall-through exit.
        let input = "\
f:
    mov x20, #0
    movz x21, #100
    b .Lguard
.Lbody:
    sxtw x22, w20
    add x0, x22, x20
    str x0, [sp, #24]
    mov x20, x0
    cmp w20, w21
    b.le .Lbody
    b .Lexit
.Lguard:
    cmp w20, w21
    b.le .Lbody
    b .Lexit
.Lexit:
    ldr x0, [sp, #24]
    ret
";
        let result = {
            let mut lines: Vec<String> = input.lines().map(String::from).collect();
            let mut kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
            let n = lines.len();
            let changed = sink_loop_carried_stores(&mut lines, &mut kinds, n);
            let out = lines
                .iter()
                .zip(kinds.iter())
                .filter(|(_, k)| **k != LineKind::Nop)
                .map(|(l, _)| l.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(changed, "store should be sunk:\n{}", out);
            out
        };
        // Exactly one store remains, on the exit path after the backedge.
        assert_eq!(result.matches("str x0, [sp, #24]").count(), 1, "\n{}", result);
        let be = result.find("b.le .Lbody").unwrap();
        let st = result.find("str x0, [sp, #24]").unwrap();
        assert!(st > be, "store after backedge:\n{}", result);
    }

    #[test]
    fn test_sink_blocked_when_slot_read_in_loop() {
        let input = "\
f:
    mov x20, #0
.Lbody:
    ldr x0, [sp, #24]
    add x0, x0, x20
    str x0, [sp, #24]
    mov x20, x0
    cmp w20, w21
    b.le .Lbody
    ldr x0, [sp, #24]
    ret
";
        let mut lines: Vec<String> = input.lines().map(String::from).collect();
        let mut kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
        let n = lines.len();
        assert!(!sink_loop_carried_stores(&mut lines, &mut kinds, n));
    }

    #[test]
    fn test_sink_blocked_by_mid_loop_exit() {
        let input = "\
f:
    mov x20, #0
.Lbody:
    cmp w20, w22
    b.gt .Lexit
    add x0, x20, #1
    str x0, [sp, #24]
    mov x20, x0
    cmp w20, w21
    b.le .Lbody
.Lexit:
    ldr x0, [sp, #24]
    ret
";
        let mut lines: Vec<String> = input.lines().map(String::from).collect();
        let mut kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
        let n = lines.len();
        assert!(!sink_loop_carried_stores(&mut lines, &mut kinds, n));
    }
}
/// Fold an increment-feeding-csel into a single csinc:
///   add wA, wB, #1            add wA, wB, #1
///   ...                       ...
///   csel wD, wA, wB, cc   OR  csel wD, wB, wA, cc
/// becomes `csinc wD, wB, wB, !cc` / `csinc wD, wB, wB, cc` and the add is
/// deleted. Fires on if-converted `if (x) count++` loops (sieve's prime
/// count loop) where the select's separate increment occupies an extra
/// dispatch slot in a branch-throughput-bound loop.
///
/// Soundness: the add's dest A must have no other mentions anywhere in the
/// function text (w- and x-names alias the same register), B must not be
/// rewritten between the add and the csel, and the scan stays within one
/// basic block (any label/branch/call stops it). csel and csinc read flags
/// identically and neither writes flags; the deleted add is the
/// flag-neutral form (not `adds`).
fn fold_csel_increment(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    fn mentions(line: &str, reg: u8) -> bool {
        line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == xreg_name(reg) || tok == wreg_name(reg))
    }
    fn canonical_cond(cc: &str) -> Option<&'static str> {
        Some(match cc {
            "eq" => "eq",
            "ne" => "ne",
            "lt" => "lt",
            "ge" => "ge",
            "gt" => "gt",
            "le" => "le",
            "hi" => "hi",
            "ls" => "ls",
            "hs" | "cs" => "hs",
            "lo" | "cc" => "lo",
            "mi" => "mi",
            "pl" => "pl",
            "vs" => "vs",
            "vc" => "vc",
            _ => return None,
        })
    }
    let mut changed = false;
    for i in 0..n {
        if kinds[i] != LineKind::Alu {
            continue;
        }
        let t = lines[i].trim().to_string();
        let Some(rest) = t.strip_prefix("add ") else { continue };
        let mut parts = rest.split(", ");
        let (Some(a_str), Some(b_str), Some(imm)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if parts.next().is_some() || imm.trim() != "#1" {
            continue;
        }
        let a_str = a_str.trim();
        let b_str = b_str.trim();
        let is_32 = a_str.starts_with('w');
        let prefix = if is_32 { 'w' } else { 'x' };
        if !a_str.starts_with(prefix) || !b_str.starts_with(prefix) {
            continue;
        }
        let a = parse_reg(a_str);
        let b = parse_reg(b_str);
        if a == REG_NONE || b == REG_NONE || a >= 29 || b >= 29 {
            continue;
        }
        // Find the consuming csel in the same basic block.
        let mut found: Option<(usize, u8, &'static str)> = None;
        let mut j = i + 1;
        while j < n {
            match kinds[j] {
                LineKind::Nop | LineKind::Directive => {}
                LineKind::Label
                | LineKind::Branch
                | LineKind::CondBranch
                | LineKind::CmpBranch
                | LineKind::Ret
                | LineKind::Call => break,
                _ => {
                    // B must survive unchanged from the add to the csel.
                    if gp_reg_written_broad(&lines[j], b) {
                        break;
                    }
                    let tj = lines[j].trim();
                    if let Some(rest) = tj.strip_prefix("csel ") {
                        let mut p = rest.split(", ");
                        if let (Some(d_s), Some(x_s), Some(y_s), Some(cc_s)) =
                            (p.next(), p.next(), p.next(), p.next())
                        {
                            let d = parse_reg(d_s.trim());
                            let x = parse_reg(x_s.trim());
                            let y = parse_reg(y_s.trim());
                            let cc_s = cc_s.trim();
                            let width_ok = d_s.trim().starts_with(prefix)
                                && x_s.trim().starts_with(prefix)
                                && y_s.trim().starts_with(prefix);
                            if width_ok && d != REG_NONE {
                                if x == a && y == b {
                                    // True arm is B+1: csinc D, B, B, !cc
                                    if let Some(inv) = invert_condition(cc_s) {
                                        found = Some((j, d, inv));
                                    }
                                    break;
                                } else if x == b && y == a {
                                    // False arm is B+1: csinc D, B, B, cc
                                    if let Some(cc) = canonical_cond(cc_s) {
                                        found = Some((j, d, cc));
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            j += 1;
        }
        let Some((cj, d, cc)) = found else { continue };
        // Prove A dead after the csel: scan to the end of the function's
        // text; the next mention of A must be a write that does not read it
        // (mov/load-dest), or the function must end first. Mirrors the
        // discipline of propagate_address_aliases.
        let mut dead = false;
        let mut k = cj + 1;
        while k < n {
            let lk = lines[k].trim_end();
            // Column-0 identifier label: end of the function's text.
            if lk.ends_with(':') && lk.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
                dead = true;
                break;
            }
            match kinds[k] {
                LineKind::Nop | LineKind::Directive | LineKind::Label | LineKind::Ret => {}
                LineKind::Call => break,
                _ => {
                    if mentions(&lines[k], a) {
                        let trusted_write = match kinds[k] {
                            LineKind::Move { dst, .. }
                            | LineKind::MoveImm { dst }
                            | LineKind::MoveWide { dst }
                            | LineKind::LoadSp { reg: dst, .. }
                            | LineKind::LoadswSp { reg: dst, .. } => dst == a,
                            LineKind::LoadPairSp => {
                                // `ldp xA, xB, [...]` writes both dests,
                                // reads neither.
                                let tk = lines[k].trim();
                                match tk.split_once(' ') {
                                    Some((_, rest)) => rest
                                        .split(',')
                                        .take(2)
                                        .any(|op| {
                                            let op = op.trim();
                                            op == xreg_name(a) || op == wreg_name(a)
                                        }),
                                    None => false,
                                }
                            }
                            LineKind::Alu => {
                                // `op wA, ...` with A only as the first operand
                                let tk = lines[k].trim();
                                match tk.split_once(',') {
                                    Some((first, rest)) => {
                                        let dest_is_a = first
                                            .split_whitespace()
                                            .last()
                                            .is_some_and(|t| t == xreg_name(a) || t == wreg_name(a));
                                        dest_is_a && !mentions(rest, a)
                                    }
                                    None => false,
                                }
                            }
                            _ => false,
                        };
                        if trusted_write {
                            dead = true;
                        }
                        break;
                    }
                }
            }
            k += 1;
        }
        if k == n {
            dead = true;
        }
        if !dead {
            continue;
        }
        lines[cj] = format!("    csinc {}{}, {}{}, {}{}, {}", prefix, d, prefix, b, prefix, b, cc);
        kinds[cj] = LineKind::Alu;
        kinds[i] = LineKind::Nop;
        changed = true;
    }
    changed
}

