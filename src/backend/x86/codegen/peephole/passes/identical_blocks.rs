//! Identical block merging pass.
//!
//! Detects basic blocks with identical instruction sequences and the same
//! jump target, then merges them by redirecting all branches to duplicates
//! to a single canonical copy. The duplicate blocks are eliminated.
//!
//! This primarily targets phi elimination trampoline blocks in large switch
//! statements (e.g., sqlite3VdbeExec), where many case blocks produce
//! identical phi copy sequences.

use super::super::types::*;
use crate::common::fx_hash::FxHashMap;

/// Merge identical basic blocks.
/// Returns true if any blocks were merged.
pub(super) fn merge_identical_blocks(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    if len < 10 { return false; }

    // Phase 1: Find all labels and their block boundaries.
    // A block starts at a label and ends at the next label, ret, or end.
    let mut blocks: Vec<(usize, usize, String)> = Vec::new(); // (start_line, end_line, label_name)
    let mut i = 0;
    while i < len {
        if infos[i].kind == LineKind::Label {
            let label = infos[i].trimmed(store.get(i));
            if let Some(label_name) = label.strip_suffix(':') {
                // Only process .LBB labels (compiler-generated, not user symbols)
                if label_name.starts_with(".LBB") {
                    let start = i;
                    let mut end = i + 1;
                    while end < len {
                        if infos[end].is_nop() { end += 1; continue; }
                        // Block ends at next label, ret, or function boundary
                        if infos[end].kind == LineKind::Label { break; }
                        if infos[end].kind == LineKind::Directive { break; }
                        end += 1;
                    }
                    blocks.push((start, end, label_name.to_string()));
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }

    if blocks.len() < 2 { return false; }

    // Phase 2: Hash each block's content (excluding the label itself).
    // Two blocks are identical if they have the same instruction sequence.
    let mut block_hashes: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    for (idx, &(start, end, _)) in blocks.iter().enumerate() {
        // Build content string for hashing (skip NOPs and the label line itself)
        let mut hasher = 0u64;
        let mut instr_count = 0u32;
        for j in (start + 1)..end {
            if infos[j].is_nop() { continue; }
            let line = store.get(j);
            // Simple hash: FNV-1a
            for byte in line.bytes() {
                hasher ^= byte as u64;
                hasher = hasher.wrapping_mul(0x100000001b3);
            }
            instr_count += 1;
        }
        // Only consider blocks with >= 4 instructions (worth merging)
        if instr_count >= 4 {
            hasher ^= instr_count as u64;
            block_hashes.entry(hasher).or_default().push(idx);
        }
    }

    // Phase 3: For each group of blocks with the same hash, verify they're truly
    // identical (not just hash collisions), then pick a canonical representative.
    let mut changed = false;
    let mut redirects: FxHashMap<String, String> = FxHashMap::default(); // old_label → canonical_label

    for (_hash, group) in &block_hashes {
        if group.len() < 2 { continue; }

        // Verify all blocks in the group are truly identical.
        // Compare instruction-by-instruction against the first block.
        let canonical_idx = group[0];
        let (can_start, can_end, _) = &blocks[canonical_idx];

        // Collect canonical block's instruction texts
        let canonical_instrs: Vec<String> = ((*can_start + 1)..*can_end)
            .filter(|&j| !infos[j].is_nop())
            .map(|j| store.get(j).to_string())
            .collect();

        for &other_idx in &group[1..] {
            let (other_start, other_end, ref other_label) = blocks[other_idx];

            let other_instrs: Vec<String> = ((other_start + 1)..other_end)
                .filter(|&j| !infos[j].is_nop())
                .map(|j| store.get(j).to_string())
                .collect();

            if canonical_instrs == other_instrs {
                // Truly identical! Redirect branches to other_label → canonical_label.
                let (_, _, ref canonical_label) = blocks[canonical_idx];
                redirects.insert(other_label.clone(), canonical_label.clone());

                // NOP out the duplicate block
                for j in other_start..other_end {
                    if !infos[j].is_nop() {
                        mark_nop(&mut infos[j]);
                        changed = true;
                    }
                }
            }
        }
    }

    if redirects.is_empty() {
        return false;
    }

    // Phase 4: Rewrite all branch targets that reference redirected labels.
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        match infos[i].kind {
            LineKind::Jmp | LineKind::CondJmp => {
                let line = store.get(i).to_string();
                let trimmed = infos[i].trimmed(&line);
                // Extract the target label from jmp/jCC instructions
                if let Some(space_pos) = trimmed.find(' ') {
                    let target = trimmed[space_pos + 1..].trim();
                    if let Some(canonical) = redirects.get(target) {
                        let prefix = &trimmed[..space_pos + 1];
                        let new_line = format!("    {}{}", prefix, canonical);
                        replace_line(store, &mut infos[i], i, new_line);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }

    changed
}
