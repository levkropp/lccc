# Next Steps Assessment — Post Phase 7b

**Date:** 2026-03-20
**Current Status:** Phase 7b (Remainder Loops) complete
**Performance vs GCC:** ~2× slower on matmul, 1.5× on arith_loop, 3.68× on fib

---

## Current State Summary

### ✅ Completed Phases (Phases 1–7b)

| Phase | Feature | Impact | Status |
|-------|---------|--------|--------|
| 2 | Linear-scan register allocator | +20–25% on reg-pressure code | ✅ Complete |
| 3a | Tail-call elimination | 139× on accumulator recursion | ✅ Complete |
| 3b | Phi-copy stack coalescing | +20% on loop-heavy code | ✅ Complete |
| 4 | Loop unrolling + FP intrinsics | +45% on matmul vs CCC | ✅ Complete |
| 5 | FP peephole optimization | +41% additional on matmul | ✅ Complete |
| 6 | SSE2 vectorization (2-wide) | ~2× on matmul-style loops | ✅ Complete |
| 7a | AVX2 vectorization (4-wide) | ~2× additional on matmul | ✅ Complete |
| 7b | Remainder loops | Production-ready for any N | ✅ Complete |

**Current benchmarks:**

| Benchmark | LCCC | GCC -O2 | vs GCC | Key bottleneck |
|-----------|-----:|--------:|:------:|----------------|
| `arith_loop` | 0.103 s | 0.068 s | 1.50× slower | Address calculations (Phase 9 target) |
| `sieve` | 0.036 s | 0.024 s | 1.50× slower | Address calculations |
| `qsort` | 0.096 s | 0.087 s | 1.10× slower | Branch mispredictions |
| `fib(40)` | 0.352 s | 0.096 s | **3.68× slower** | Poor inlining (Phase 8 target) |
| `matmul` | 0.008 s | 0.004 s | **2.0× slower** | Strength reduction + unroll-and-jam |
| `tce_sum` | 0.008 s | 0.008 s | ≈ equal | ✅ TCE solved this |

---

## Phase 9: Loop Strength Reduction (Highest Impact)

### Problem Analysis

**Current inefficiency (observed in arith_loop assembly):**
```asm
# LCCC generates (4 instructions per array access):
movslq %r13d, %rax      # IV sign-extend (1 cycle)
shlq $3, %rax           # IV * 8 (1 cycle)
addq %rbx, %rax         # base + offset (1 cycle)
movsd (%rax), %xmm0     # Load (3-4 cycles)
# Total: 4 instructions, 6-7 cycles

# GCC generates (1 instruction):
movsd (%rbx,%r13,8), %xmm0  # Single indexed load (3-4 cycles)
# Total: 1 instruction, 3-4 cycles
```

**Impact:**
- arith_loop: 32 variables × 10M iterations = **320 million redundant instructions**
- Estimated 5–10% performance gain across all array-heavy code
- Matmul: ~5% speedup (addressing overhead in innermost loop)
- Sieve: ~5% speedup (bool array indexing)

### Implementation Plan

**Root cause:** Backend doesn't recognize `base + IV*scale` patterns for x86-64 SIB (Scale-Index-Base) addressing modes.

**Solution:** Three-layer backend enhancement

**Layer 1: IV Tracking**
```rust
pub struct X86Codegen {
    // Existing fields...

    /// Maps value ID to (base_phi_id, scale_factor)
    iv_derived: FxHashMap<u32, (u32, i64)>,

    /// Set of phi nodes that are basic IVs (loop counters)
    basic_ivs: FxHashSet<u32>,
}
```

- Mark phi nodes in loop headers as potential IVs
- Track Cast/Shl/Mul operations on IV-derived values
- Maintain map: `Value → (base_iv, scale_factor)`

**Layer 2: Pattern Recognition**
```rust
fn try_emit_indexed_load(&mut self, dest: Value, ptr: Value, ty: IrType) -> bool {
    // 1. Check if ptr is defined by GEP
    let gep_info = self.find_gep_definition(ptr)?;

    // 2. Check if offset is IV-derived
    let (base_iv, scale) = self.iv_derived.get(&gep_info.offset.0)?;

    // 3. Check register assignments
    let base_reg = self.get_register_for_value(gep_info.base)?;
    let index_reg = self.get_register_for_value(base_iv)?;

    // 4. Emit indexed load
    self.state.emit(&format!(
        "    movsd ({},%{},{}), %xmm0",
        base_reg, index_reg, scale
    ));
    true
}
```

**Layer 3: Code Generation**
- Detect address computation patterns in load/store instructions
- Emit indexed addressing directly: `movsd (%base,%index,scale), %xmm0`
- Fallback to current sequential instructions if pattern doesn't match

### Files to Modify

1. `src/backend/x86/codegen/emit.rs` (~50 new lines)
   - Add `iv_derived` and `basic_ivs` fields
   - Detect IVs in phi processing
   - Track derived values in Shl/Mul/Cast

2. `src/backend/x86/codegen/memory.rs` (~150 new lines)
   - `try_emit_indexed_load/store()` functions
   - Pattern analysis helpers
   - Integration in `emit_load_impl/emit_store_impl`

3. `src/backend/x86/codegen/mod.rs` (~50 lines)
   - `is_valid_scale()` — check if scale is 1/2/4/8
   - `find_gep_definition()` — trace ptr back to GEP
   - `get_register_for_value()` — lookup in reg_assignments

### Estimated Impact

- **arith_loop:** 0.103s → ~0.093s (**10% faster**)
- **matmul:** 0.008s → ~0.0076s (**5% faster**)
- **sieve:** 0.036s → ~0.034s (**5% faster**)

**Risk level:** Medium (backend pattern matching can be fragile)
**Implementation time:** 5–7 days (1 week)
**Value:** High — helps all loops with array indexing

---

## Phase 8: Better Inlining (High Impact on fib)

### Problem Analysis

**Current issue:**
- `fib(40)`: LCCC 0.352s vs GCC 0.096s (3.68× slower)
- GCC inlines the entire recursion tree into unrolled loops
- LCCC's inliner has poor cost model and doesn't inline aggressively enough

**Evidence from assembly:**
```asm
# LCCC generates: Full function call overhead on every level
fib:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    # ... full call overhead ...
    call fib
    # ... full call overhead ...

# GCC generates: Inlined and unrolled
fib:
    # No calls at all — entire tree computed inline
```

### Implementation Plan

**1. Better Cost Model**
```rust
fn should_inline(callee: &IrFunction, call_site: &CallSite) -> bool {
    let callee_size = estimate_size(callee);
    let call_depth = get_call_depth(call_site);

    // Current (too conservative):
    // callee_size < 50 || (callee_size < 100 && call_depth == 0)

    // Proposed (more aggressive):
    if callee_size < 20 { return true; }  // Always inline tiny functions
    if call_depth < 2 && callee_size < 100 { return true; }  // Inline small functions twice
    if is_recursive_tail_call(callee, call_site) { return true; }  // TCE handles this
    false
}
```

**2. Recursive Inlining Budget**
- Allow recursive functions to be inlined N times (N=2 or 3)
- Track inlining depth per function
- Stop when depth exceeds budget (prevents explosion)

**3. Hot Call Site Priority**
- Inline call sites in loops first (higher impact)
- Use loop depth as priority metric
- Already have loop depth analysis from vectorization

### Files to Modify

1. `src/passes/inline.rs` (~100 lines)
   - Update `should_inline()` cost model
   - Add recursive inlining depth tracking
   - Prioritize hot call sites

### Estimated Impact

- **fib(40):** 0.352s → ~0.176s (**2× faster**, but still 1.8× slower than GCC)
- **Other benchmarks:** Minimal change (already well-inlined or don't benefit)

**Risk level:** Low (conservative limits prevent code explosion)
**Implementation time:** 3–5 days
**Value:** High for recursive code, minimal for others

---

## Phase 10: Profile-Guided Optimization (Long-term)

### Overview

Use runtime profiling to guide optimization decisions:
- Which functions to inline (profile shows hot paths)
- Which loops to unroll (profile shows trip counts)
- Which branches to optimize (profile shows taken frequency)

### Implementation Phases

**10a: Instrumentation (2 weeks)**
- Insert counters at function entry, loop headers, branches
- Write profile data to file on program exit
- Minimal overhead (1–2% slowdown)

**10b: Profile-Guided Inlining (1 week)**
- Parse profile data during compilation
- Use hot path info to prioritize inlining
- Already have inliner infrastructure from Phase 8

**10c: Profile-Guided Unrolling (1 week)**
- Use trip count profiles to choose unroll factors
- Specialize hot loops with constant trip counts

### Estimated Impact

- **General speedup:** 1.2–1.5× on real workloads
- **Synthetic benchmarks:** May see less benefit (simple profiles)

**Risk level:** Medium (profiling overhead, data stability)
**Implementation time:** 4 weeks total
**Value:** Moderate — better payoff for complex real-world code

---

## Recommended Priority Order

### Immediate (1–2 weeks)

**✅ Phase 9: Loop Strength Reduction**
- **Why first:** Broad impact (helps arith_loop, sieve, matmul)
- **Estimated gain:** 5–10% across array-heavy code
- **Risk:** Medium (backend changes, but well-understood pattern)

### Near-term (2–3 weeks)

**✅ Phase 8: Better Inlining**
- **Why second:** Closes fib gap, minimal risk
- **Estimated gain:** 2× on fib, minimal elsewhere
- **Risk:** Low (conservative limits)

### Long-term (4+ weeks)

**Phase 10: PGO** — Future work after infrastructure stabilizes

---

## Alternative: Low-Hanging Fruit

If we want **quick wins** before tackling Phase 9, consider:

### Loop Invariant Code Motion (LICM) Enhancements

**Current LICM:** Basic (hoists simple operations)
**GCC LICM:** More aggressive (hoists address calculations out of loops)

**Example:**
```c
for (int i = 0; i < N; i++) {
    A[i] = B[i] + C[i];  // LCCC computes &B, &C inside loop
}
```

**LCCC:**
```asm
loop:
    lea B(%rip), %rax    # Inside loop ❌
    lea C(%rip), %rbx    # Inside loop ❌
    # ... rest of loop ...
```

**GCC:**
```asm
    lea B(%rip), %rax    # Outside loop ✅
    lea C(%rip), %rbx    # Outside loop ✅
loop:
    # ... rest of loop ...
```

**Estimated impact:** 2–3% on loops with global/static arrays
**Implementation time:** 2–3 days
**Risk:** Low

---

## Summary Table

| Phase | Estimated Gain | Implementation Time | Risk | Priority |
|-------|----------------|---------------------|------|----------|
| **9: Strength Reduction** | **5–10% broad** | 5–7 days | Medium | **✅ High** |
| **8: Better Inlining** | **2× on fib** | 3–5 days | Low | **✅ High** |
| LICM enhancement | 2–3% on array loops | 2–3 days | Low | Medium |
| 10: PGO | 1.2–1.5× general | 4 weeks | Medium | Low (future) |

---

## Recommendation

**Start with Phase 9 (Loop Strength Reduction)**

**Rationale:**
1. **Broad impact** — helps 3 out of 6 benchmarks (arith_loop, sieve, matmul)
2. **Well-understood** — x86-64 indexed addressing is a solved problem
3. **Foundation** — sets up backend infrastructure for future optimizations
4. **Measurable** — easy to verify in assembly and benchmarks

**After Phase 9:** Move to Phase 8 (Better Inlining) to close the fib gap.

**Combined impact estimate:**
- Phase 9 + Phase 8: Brings overall geometric mean from ~1.9× slower than GCC to ~1.5× slower
- Keeps us on track for the goal of <1.5× slower on typical workloads

---

## Testing Strategy

Before starting Phase 9, we should:

1. ✅ **Verify Phase 7b correctness**
   - Run existing test suite (514 tests)
   - Create remainder-specific tests (N % 4 ∈ {1, 2, 3})
   - Benchmark overhead on non-aligned N

2. ✅ **Document Phase 7b**
   - Update README ✅
   - Write blog post ✅
   - Update roadmap ✅

3. ✅ **Establish baseline**
   - Run benchmarks 5× and record best-of-5
   - Document current assembly patterns for arith_loop
   - Save assembly for before/after comparison

4. **Plan Phase 9 verification**
   - Create test cases for indexed addressing detection
   - Verify assembly shows `movsd (%base,%index,scale)` patterns
   - Benchmark before/after on arith_loop, sieve, matmul

---

*Last updated: 2026-03-20 after Phase 7b completion*
