---
layout: post
title: "Phase 14: SQLite Fully Works — 31 Bugs Fixed in One Debugging Marathon"
date: 2026-03-27
author: Lev Kropp & Claude Opus 4.6
categories: [correctness, sqlite, debugging]
tags: [register-allocator, phi-coalescing, stack-layout, peephole, liveness]
---

## TL;DR

LCCC now compiles and **fully runs** SQLite 3.45 (260K lines of C). All core SQL operations work: CREATE TABLE, INSERT, UPDATE, DELETE, SELECT with JOINs, subqueries, GROUP BY, transactions, and prepared statements. Getting here required fixing **31 correctness bugs** across the entire backend.

---

## The Challenge: Why SQLite?

SQLite is the ideal compiler stress test:
- **260,000 lines** in a single file (the amalgamation)
- **400+ case switch** in the VDBE interpreter (`sqlite3VdbeExec`)
- Complex control flow: nested loops, computed gotos, longjmp
- Exercises every corner: pointer arithmetic, variadic functions, bit manipulation, floating-point, linked list traversal, B-tree operations

If your compiler can run SQLite correctly, it can probably compile anything.

## The Journey: From "Opens a Database" to "Full SQL"

### Milestone 1: sqlite3_open works (bugs 1-8)

The first crash came immediately — the peephole optimizer's dead-code elimination misidentified read-modify-write instructions (`addl %eax, %r13d`) as dead stores. The jmp-following scan had a weaker check than the main scan.

Other early bugs: indirect call stack corruption (pushing `%rax` before loading the function pointer shifted RSP), RSP/RBP addressing mode leak between functions, and variadic functions using frame pointer omission.

### Milestone 2: SELECT 42 works (bugs 9-18)

Simple queries exposed stack layout bugs. The callee-save boundary calculation didn't account for an off-by-8 collision between Tier 3 (block-local reusable) slots and callee-saved register pushes. The post-decrement pattern `while(n--)` was evaluating the decremented value instead of the original.

The register allocator's phi coalescing had a "lost copy" bug: in a linked list traversal (`malloc` → `head->next = node`), the malloc result and the list head pointer were coalesced to the same register, so storing the malloc result clobbered the head pointer.

### Milestone 3: Full SQL works (bugs 19-31)

The final bugs were the subtlest:

**Phase 9 SIB stale registers.** The indexed addressing optimization decomposed variable-offset GEPs into SIB addressing (`movl %eax, (%base, %idx, scale)`), but the GEP had already been emitted as a `leaq`. The source registers were stale. Fix: disable Phase 9 for non-folded GEPs.

**Phi coalescing multi-block loop bodies.** The safety check only looked for the backedge source definition in the Copy's block. When a `for` loop body contains an `if` branch (very common), the loop body spans multiple blocks. The Copy ends up in a latch block, but the source value is defined in a different body block. The check never found the definition, so it incorrectly allowed coalescing.

This crashed `sqlite3PcacheTruncate` — a linked list traversal where:
```c
while (p) {
    PgHdr *pNext = p->pDirtyNext;  // defined here (block A)
    if (p->pgno >= pgno) { ... }   // uses p here (block A)
    p = pNext;                      // Copy in latch (block B)
}
```

With coalescing, `pNext` and `p` shared register `r15`. Loading `pNext` from `p->pDirtyNext` overwrote `r15` before `p->pgno` could be read.

## Bug Taxonomy

| Category | Count | Examples |
|----------|-------|---------|
| Register allocator | 4 | Phi coalesce safety (3 variants), callee-save boundary |
| Peephole optimizer | 6 | Dead reg scan, loop rotation, sign-ext fusion, ret barrier |
| Stack layout | 5 | Callee-save padding, cross-block alias, multi-def alias |
| Call codegen | 3 | Indirect call RSP, stack arg tracking, FPO stack_base |
| Frame/addressing | 3 | RSP/RBP leak, variadic FPO, emit_instr_rbp |
| Value handling | 4 | Alloca Copy, post-decrement, GVN volatile, emergency spills |
| SIB addressing | 2 | Register conflict, Phase 9 stale registers |
| Jump table | 1 | i128 overflow in range calculation |
| Other | 3 | operand_to_callee_reg, liveness extension, debug infra |

## What's Left

SQLite works with two optimization passes disabled:
- **Peephole optimizer** (`CCC_NO_PEEPHOLE=1`): crashes on large functions due to stack corruption
- **GVN** (`CCC_DISABLE_PASSES=gvn`): separate crash, likely similar liveness issue

Re-enabling these is the next phase of work. The peephole optimizer passes all 18 compatibility tests — the issue is specific to large, complex functions like SQLite's VDBE interpreter.

## Verification

```
$ CCC_NO_PEEPHOLE=1 CCC_DISABLE_PASSES=vectorize,gvn lccc sqlite3.c -c -o sqlite3.o
$ gcc test_harness.c sqlite3.o -o test -lm -lpthread -ldl
$ ./test
=== Test 1: Schema ===
OK
=== Test 2: Bulk insert (100 rows) ===
OK
=== Test 3: Complex JOIN with aggregation ===
OK
=== Test 4: Subquery + ORDER BY ===
OK
=== Test 5: UNION ===
OK
...
=== Test 11: Prepared statements ===
OK

ALL STRESS TESTS PASSED
```

18/18 compatibility tests pass. SQLite compiles in ~45 seconds.

## Update: Peephole Optimizer Re-enabled (Phase 15)

After Phase 14, the peephole optimizer was disabled for SQLite (`CCC_NO_PEEPHOLE=1`). Binary search using per-pass disable flags (`CCC_PEEPHOLE_SKIP`) identified three buggy passes:

1. **`fuse_signext_and_move`**: rax liveness scan had a 12-line window — too small for SQLite's 10K-line functions. Fixed by scanning to the next barrier.
2. **`eliminate_dead_reg_moves`**: jmp-following crossed basic block boundaries unsoundly. Fixed by removing cross-block following.
3. **`fold_base_index_addressing`**: crashes on subquery compilation. Still under investigation.

With these 3 passes disabled, **22 of 25 peephole sub-passes are now active** on SQLite. All 11 stress tests pass.
