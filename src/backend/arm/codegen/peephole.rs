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
        changed |= eliminate_move_chains(&mut lines, &mut kinds, n);
        changed |= eliminate_unused_x9_address_moves(&lines, &mut kinds, n);
        changed |= propagate_address_aliases(&mut lines, &mut kinds, n);
        changed |= fold_destructive_pointer_updates(&mut lines, &mut kinds, n);
        changed |= fuse_branch_over_branch(&mut lines, &mut kinds, n);
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
    global_dead_store_elimination(&lines, &mut kinds, n);

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
            changed2 |= eliminate_move_chains(&mut lines, &mut kinds, n);
            changed2 |= eliminate_unused_x9_address_moves(&lines, &mut kinds, n);
            changed2 |= propagate_address_aliases(&mut lines, &mut kinds, n);
            changed2 |= fold_destructive_pointer_updates(&mut lines, &mut kinds, n);
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
    let mut extended = [false; 33];
    extended[32] = true; // xzr is always "extended" (sxtw xD, wzr ≡ mov xD, xzr)
    let mut changed = false;
    for i in 0..n {
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
                if let Some(dst) = written_gp_register(&lines[i], kinds[i]) {
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
    // Safety check: if any instruction takes the address of sp (e.g., `add xN, sp, #off`),
    // stack slots could be accessed through pointers, so we must not eliminate any stores.
    // This is conservative but sound — it prevents miscompilation when arrays or structs
    // are allocated on the stack and passed by pointer to callees.
    for i in 0..n {
        if kinds[i] == LineKind::Nop {
            continue;
        }
        let trimmed = lines[i].trim();
        // Check for address-of-sp patterns:
        // - `add xN, sp, #offset` (common for array/struct address)
        // - `mov xN, sp` (copying stack pointer)
        // - `sub xN, sp, #N` (stack address computation)
        if (trimmed.starts_with("add ") || trimmed.starts_with("sub ")) && trimmed.contains(", sp,") {
            return false;
        }
        if trimmed.starts_with("mov ") && trimmed.contains(", sp") {
            // Check it's actually "mov xN, sp" not "mov sp, xN"
            if let Some(rest) = trimmed.strip_prefix("mov ") {
                if let Some((_, src)) = rest.split_once(", ") {
                    if src.trim() == "sp" {
                        return false;
                    }
                }
            }
        }
    }

    // Phase 1: Collect all (offset, size) byte ranges that are loaded from.
    // We must use byte-range overlap (not exact offset match) because a wide
    // store (e.g. `str x` at offset 16, 8 bytes) can be partially read by a
    // narrower load at a different offset (e.g. `ldr w` at offset 20, 4 bytes).
    let mut loaded_ranges: Vec<(i32, i32)> = Vec::new(); // (offset, size)
    for i in 0..n {
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
                    }
                }
            }
        }
    }

    // Phase 2: Remove stores whose byte range does not overlap any load range
    let mut changed = false;
    for i in 0..n {
        if let LineKind::StoreSp { offset, is_word, .. } = kinds[i] {
            let store_size = if is_word { 4 } else { 8 };
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
        // When sp address is taken, no stores should be eliminated
        let input = "\
    str w0, [sp, #16]\n\
    add x1, sp, #16\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        // Store must be preserved because address of stack slot is taken
        assert!(result.contains("str w0, [sp, #16]"));
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
}
