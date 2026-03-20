# Phase 9b Implementation Notes

## Summary

Phase 9b was planned to extend indexed addressing optimization to handle IVSR (IV Strength Reduction) transformed loops. However, implementation revealed a fundamental architectural limitation: **phi nodes are eliminated before codegen**, making it impossible to detect IVSR patterns at the code generation stage.

## Current Status

**Phase 9 (complete):** ✅ Successfully detects and emits indexed addressing for explicit multiply/shift patterns like:
```c
double foo(double *arr, int i) {
    return arr[i];  // Emits: movsd (%base,%index,8), %xmm0
}
```

**Phase 9b (architectural limitation):** ❌ Cannot detect IVSR patterns at codegen time due to phi elimination.

## The Problem

### What IVSR Does

The IVSR pass (`src/passes/iv_strength_reduce.rs`) transforms array access patterns:

**Before IVSR:**
```
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
```

**IR Before IVSR:**
```
%i = Phi(0, %i_next)
%offset = Shl(%i, 3)          // i * 8
%addr = GEP(%arr, %offset)
%val = Load(%addr)
%i_next = Add(%i, 1)
```

**IR After IVSR:**
```
%ptr = Phi(%arr, %ptr_next)    // Pointer IV created!
%val = Load(%ptr)
%ptr_next = GEP(%ptr, 8)       // Constant stride
```

This eliminates the multiply/shift per iteration, which is generally beneficial.

### The Phi Elimination Problem

The optimization pipeline (src/driver/pipeline.rs) runs:

1. **Optimization passes (3 iterations)**
   - GVN, LICM, IVSR run here
   - Phi nodes exist and are used by IVSR

2. **Phi Elimination** (`eliminate_phis()`)
   - Converts all phi nodes to stack slot loads/stores
   - After this point, no phi nodes exist in the IR

3. **Code Generation**
   - Phase 9b detection runs here
   - **Problem:** Phi nodes are gone! Cannot detect IVSR pattern.

### Assembly Output Comparison

**sum_array (IVSR pattern - pointer arithmetic):**
```asm
.Lloop:
    movsd (%r12), %xmm0     # Load from pointer
    addsd %xmm0, ...
    leaq 8(%r12), %rax       # Increment pointer
    movq %rax, %r12
    jmp .Lloop
```

**add_arrays (Phase 9 working - multiply pattern preserved):**
```asm
.Lloop:
    movsd (%r11,%r13,8), %xmm0   # ✓ Indexed addressing!
    addsd (%r10,%r13,8), %xmm0
    ...
```

Why does `add_arrays` work? Likely because with multiple arrays, IVSR chooses to keep the index-based form to avoid creating multiple pointer IVs.

## Attempted Solutions

### Attempt 1: Codegen-Time Detection (Failed)

**Approach:** Detect IVSR patterns during code generation by looking for pointer phi nodes.

**Implementation:** Added infrastructure to `src/backend/x86/codegen/emit.rs`:
- `IvsrPointerInfo` struct
- `analyze_ivsr_pointers()` function
- `detect_ivsr_pattern()` logic

**Result:** Analysis finds 0 phi nodes because `eliminate_phis()` ran first.

**Files modified:**
- `src/backend/x86/codegen/emit.rs` (detection infrastructure)
- `src/backend/x86/codegen/memory.rs` (try_emit_ivsr_indexed_load/store)
- `src/backend/x86/codegen/prologue.rs` (call to analyze_ivsr_pointers)

### Attempt 2: IR Transformation Pass (Partial)

**Approach:** Create a new optimization pass that runs AFTER IVSR but BEFORE phi elimination to revert pointer IVs back to indexed form.

**File created:** `src/passes/univsr.rs` (Un-IVSR pass skeleton)

**Status:** Skeleton only - full implementation would require:
1. Finding all Load/Store uses of pointer phi
2. Creating new Mul/Shl and GEP instructions
3. Updating all uses while maintaining SSA form
4. Removing dead phi nodes
5. Careful handling of dominance and liveness

**Complexity:** High - requires deep IR manipulation and could introduce subtle bugs.

## Correct Implementation Path

For Phase 9b to work properly, it should be implemented as:

### Option A: IR Transformation Pass (Recommended)

**When:** After IVSR, before phi elimination
**Where:** `src/passes/univsr.rs` (new pass)
**Pipeline integration:** Add to `run_passes()` in iteration loop

**Steps:**
1. Detect IVSR pointer phis (while they still exist)
2. Check if stride is valid SIB scale (1, 2, 4, 8)
3. Find associated index IV
4. Replace pointer loads with indexed GEP+load
5. Update backedge to increment index instead of pointer
6. Remove pointer phi (will be DCE'd)

**Advantages:**
- Clean separation of concerns
- Works at IR level (target-independent detection)
- Backend indexed addressing (Phase 9) handles emission

**Disadvantages:**
- Complex IR manipulation
- Need to handle SSA form updates correctly
- Additional pass adds compile time

### Option B: Smarter IVSR (Alternative)

**When:** During IVSR pass itself
**Where:** Modify `src/passes/iv_strength_reduce.rs`

**Approach:** Make IVSR target-aware - don't create pointer IVs when indexed addressing is available and beneficial.

**Steps:**
1. Add target information to IVSR
2. Check if stride is valid for indexed addressing
3. Skip pointer IV creation for eligible patterns
4. Let original multiply+GEP pattern remain

**Advantages:**
- No additional pass needed
- Simpler than un-IVSR
- Works naturally with existing infrastructure

**Disadvantages:**
- Makes IVSR target-dependent
- Might miss IVSR opportunities on non-indexed architectures

### Option C: Post-Phi-Elimination Pattern Matching (Complex)

**When:** At codegen time (current approach location)
**Where:** Backend codegen

**Approach:** Detect pointer increment patterns in register-allocated code.

**Challenges:**
- After phi elimination, pointer is a stack slot
- Register allocator may allocate to register
- Pattern is obscured by stack loads/stores
- Very fragile and target-specific

**Verdict:** Not recommended - too fragile and complex.

## Performance Impact Analysis

### Benchmarks That Would Benefit

Based on assembly analysis of test cases:

| Function | Current | With Phase 9b | Reason |
|----------|---------|---------------|---------|
| `sum_array` | Pointer arith | Indexed addr | Simple array sum loop |
| `sum_longs` | Pointer arith | Indexed addr | Integer array |
| `sum_ints` | Pointer arith | Indexed addr | 4-byte elements |
| `add_arrays` | **Already optimal** | No change | Phase 9 working |

**Estimated improvement:** 3-5% on array-heavy code with simple loops.

**Why the improvement:**
- Indexed addressing uses dedicated AGU (Address Generation Unit)
- Frees ALU from pointer arithmetic
- Simpler dependency chains
- Better instruction-level parallelism

### Current Phase 9 Coverage

Phase 9 already handles many cases well:
- Non-loop array access: `arr[i]` ✓
- Multi-array loops: `a[i] + b[i]` ✓ (IVSR doesn't transform these)
- 2D arrays: `matrix[i][j]` ✓ (partially)

## Recommendation

**Short term:** Document the limitation. Phase 9 provides good coverage for many cases.

**Medium term:** Implement Option B (Smarter IVSR) - make IVSR skip pointer IV creation when:
- Target has indexed addressing
- Stride is 1, 2, 4, or 8 (valid SIB scale)
- Pattern is simple array access

**Long term:** If needed, implement Option A (Un-IVSR pass) as a separate optimization.

## Files to Review/Modify for Full Implementation

### Core IVSR Detection (Option A - Un-IVSR Pass)
- `src/passes/univsr.rs` - Complete the skeleton implementation
- `src/passes/mod.rs` - Add to pipeline after IVSR, before phi elimination
- `src/driver/pipeline.rs` - Integrate into optimization iterations

### Smarter IVSR (Option B - Modify IVSR)
- `src/passes/iv_strength_reduce.rs` - Add target-awareness
- Add target capabilities query (has_indexed_addressing, valid_sib_scales)
- Skip pointer IV creation for eligible patterns

### Testing
- `tests/test_ivsr_indexed.c` - Comprehensive test suite already created
- Add more edge cases (multiple IVs, non-power-of-2 strides, etc.)

## Test Coverage

Test file created: `tests/test_ivsr_indexed.c`

Covers:
- ✓ Simple array sum (double)
- ✓ Multiple arrays
- ✓ Integer arrays (various sizes)
- ✓ Non-power-of-2 stride (struct Point with 24 bytes)
- ✓ Array write patterns
- ✓ Read-modify-write patterns

## Lessons Learned

1. **Optimization pipeline ordering matters** - Phi elimination before codegen prevents certain pattern detection.

2. **IR transformations are powerful but complex** - Modifying SSA form requires careful handling of dominance, liveness, and use-def chains.

3. **Target-aware optimizations are tricky** - Either make passes target-aware (complex) or use IR metadata/hints (better).

4. **Phase 9 already provides good coverage** - The multiply+GEP pattern detection works well for many cases.

5. **IVSR is not always beneficial** - On architectures with indexed addressing, the original multiply form can be superior.

## Conclusion

Phase 9b infrastructure has been implemented but is disabled because phi elimination prevents pattern detection at codegen time. The proper implementation requires either:
- A new IR transformation pass (Un-IVSR) that runs before phi elimination
- Making IVSR target-aware to skip pointer IV creation when indexed addressing is available

The current Phase 9 implementation already provides good indexed addressing coverage for cases where IVSR doesn't run or preserves the multiply pattern.

---

*Documentation Date: 2026-03-20*
*Phase 9 Status: Complete and Working*
*Phase 9b Status: Architecturally Limited - Requires IR-Level Implementation*
