# LCCC Optimization Roadmap — Next Steps to Close the Gap with GCC

## Current Status (After Phase 7a)

| Benchmark | LCCC | GCC | Gap | Notes |
|-----------|------|-----|-----|-------|
| `arith_loop` | 0.103s | 0.068s | **1.5×** | Register pressure optimized ✓ |
| `sieve` | 0.036s | 0.024s | **1.5×** | Loop unrolling helps ✓ |
| `matmul` | ~0.004s (est.) | 0.004s | **~1×** | AVX2 vectorization ✓ |
| `fib` | 0.352s | 0.096s | **3.7×** | Recursive overhead |
| `qsort` | 0.096s | 0.087s | **1.1×** | Nearly optimal |
| `tce_sum` | 0.008s | 0.008s | **1.0×** | Tail-call elim = perfect |

**Overall:** Within 1.0–3.7× of GCC across benchmarks. The goal is to get everything to ~1.5× or better.

**Phase 7a Complete!** AVX2 4-wide vectorization is now implemented and working:
- ✅ Added `FmaF64x4` intrinsic to `src/ir/intrinsics.rs`
- ✅ Modified vectorization pass to divide loop bound by 4 and multiply offsets by 4
- ✅ Backend generates AVX2 instructions (vbroadcastsd, vmovupd, vmulpd, vaddpd)
- ✅ Environment variable control: `LCCC_FORCE_SSE2=1` to use 2-wide instead of 4-wide
- ✅ Assembly verification shows correct N/4 loop iterations with 32-byte stride

---

## Priority 1: Finalize matmul optimizations

### Phase 7b: Remainder Loop (Est: 3-5 days)

**Status:** Planned (not yet started)

**Goal:** Handle N not divisible by 4 correctly

---

### Phase 7b: Remainder Loop (Est: 3-5 days)

**Goal:** Handle odd N correctly

**Current issue:** If N=257, element 256 is skipped.

**Implementation:**
1. After vectorized loop, add remainder check:
   ```rust
   let remainder_check = BasicBlock {
       label: BlockId(next_label),
       instructions: vec![
           // %is_odd = and %N, 1
           Instruction::BinOp { dest: is_odd, op: And, lhs: N, rhs: Const(1), ... },
           // %cond = cmp ne %is_odd, 0
           Instruction::Cmp { dest: cond, op: Ne, lhs: is_odd, rhs: Const(0), ... },
       ],
       terminator: CondBranch { cond, true_label: remainder_body, false_label: exit },
   };
   ```

2. Remainder body: scalar version of the loop body for index `N-1`

3. Redirect vectorized loop exit to remainder_check instead of original exit

**Expected gain:** Correctness (no performance change on even N)

**Risk:** Low (plan already written, just needs implementation)

---

## Priority 2: Reduce fib Overhead (3.7× → 2×)

### Phase 8: Better Inlining Heuristics (Est: 1 week)

**Goal:** Inline hot recursive calls more aggressively

**Current issue:** `fib(40)` has massive call overhead. GCC inlines aggressively.

**Analysis:**
```c
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);  // Two recursive calls
}
```

**LCCC:** Keeps both calls as function calls
**GCC:** Inlines one or both calls (reduces call overhead by 50–100%)

**Implementation:**
1. Track call frequency in hot paths (via loop depth heuristic)
2. Inline if:
   - Function is small (<20 IR instructions)
   - Called from inside a loop OR recursively
   - Total inline budget not exceeded
3. Re-run inlining multiple times (current: once)

**Expected gain:** 1.5–2× on fib (bring 3.7× down to ~2×)

**Risk:** Medium (inlining is complex, can cause code bloat)

---

## Priority 3: General Speedups (1.5× → 1.2×)

### Phase 9: Loop Strength Reduction Improvements (Est: 1 week)

**Goal:** Eliminate more address calculations in inner loops

**Current issue:** arith_loop and sieve still have some redundant address math.

**Example (from arith_loop):**
```asm
# Current:
movslq %r13d, %rax      # IV sign-extend
shlq $3, %rax           # IV * 8
addq %rbx, %rax         # base + offset
movsd (%rax), %xmm0     # Load

# Better:
movsd (%rbx,%r13,8), %xmm0  # Single indexed load
```

**Implementation:**
1. IVSR pass: detect `base + IV*scale` patterns
2. Track base/IV/scale separately
3. Backend: emit indexed addressing modes (`[base + index*scale]`)

**Expected gain:** 5–10% on loop-heavy code (arith_loop, sieve)

**Risk:** Low (well-understood transformation)

---

### Phase 10: Profile-Guided Optimization (Est: 2-3 weeks)

**Goal:** Use runtime profiling to guide optimizations

**Approach:**
1. **Instrumentation mode:** Insert edge counters in CFG
2. **Profile collection:** Run benchmark suite, collect counts
3. **Profile-guided compilation:**
   - Inline based on actual call frequency
   - Unroll based on actual trip counts
   - Vectorize based on actual loop structure
   - Layout basic blocks by execution frequency (reduce branch mispredicts)

**Expected gain:** 10–20% across the board (conservative estimate)

**Risk:** Medium-high (requires two-pass compilation, profile format design)

---

## Priority 4: Broader Vectorization (Beyond matmul)

### Phase 11a: Integer Vectorization (Est: 1-2 weeks)

**Goal:** Vectorize integer loops (sieve counting, arith_loop if it uses ints)

**Example (sieve counting):**
```c
for (int i = 0; i < N; i++)
    count += is_prime[i];  // Sum of 0/1 values
```

**Current:** Scalar (1 element/iteration)
**AVX2:** 8 elements/iteration (256-bit SIMD for 32-bit ints)

**Implementation:**
1. Extend pattern matcher to handle integer accumulation
2. Add `VaddEpi32x8` intrinsic (packed 8-way int32 add)
3. Backend: emit `vpaddd` instructions

**Expected gain:** 2–4× on sieve counting pass

---

### Phase 11b: Reduction Patterns (Est: 1 week)

**Goal:** Recognize sum/max/min reductions

**Example:**
```c
int sum = 0;
for (int i = 0; i < N; i++)
    sum += arr[i];
```

**Current:** Scalar (dependency chain prevents vectorization)
**Better:** Horizontal reduction (4-way parallel accumulation, then combine)

**Implementation:**
1. Detect reduction pattern (accumulator with associative op)
2. Split into N/4 partial sums
3. Combine at end with horizontal add

**Expected gain:** 2–4× on reduction-heavy code

---

## Priority 5: Backend Improvements

### Phase 12: Better Register Allocation for Vectors (Est: 1 week)

**Goal:** Use more XMM/YMM registers efficiently

**Current issue:** Linear scan allocator treats vectors like scalars. Not taking advantage of all 16 XMM registers.

**Implementation:**
1. Separate register classes: GPR vs XMM vs YMM
2. Track vector live ranges independently
3. Prefer XMM registers for FP values, even if they're not vectorized

**Expected gain:** 5–10% on FP-heavy code (fewer spills)

---

### Phase 13: Instruction Scheduling (Est: 2 weeks)

**Goal:** Reorder instructions to hide latency

**Current issue:** movsd has 3-cycle latency. If we immediately use the result, we stall.

**Example:**
```asm
# Current (stalls):
movsd  (%rax), %xmm0   # 3-cycle latency
mulsd  %xmm1, %xmm0    # Stalls waiting for load

# Better (pipelined):
movsd  (%rax), %xmm0
movsd  (%rbx), %xmm2   # Load next value while waiting
mulsd  %xmm1, %xmm0    # Now ready
```

**Implementation:**
1. Build dependency graph for basic blocks
2. Schedule instructions to maximize ILP (instruction-level parallelism)
3. Heuristic: prioritize loads, then ALU ops, then stores

**Expected gain:** 10–15% on latency-bound code

**Risk:** Medium (complex, can break if done wrong)

---

## Recommended Execution Order

### Quarter 1 (Next 3 months):

1. **Phase 7a: AVX2 Vectorization** (2 weeks) — Biggest single gain (~2× matmul)
2. **Phase 7b: Remainder Loop** (3 days) — Correctness fix
3. **Phase 9: Loop Strength Reduction** (1 week) — Broad 5–10% gain
4. **Phase 8: Better Inlining** (1 week) — Close fib gap

**Expected result:** matmul at ~1× of GCC, fib at ~2× of GCC, arith_loop/sieve at ~1.3× of GCC

### Quarter 2:

5. **Phase 11a: Integer Vectorization** (2 weeks) — Help sieve
6. **Phase 12: Vector Register Allocation** (1 week) — 5–10% FP boost
7. **Phase 10: Profile-Guided Optimization** (3 weeks) — 10–20% across board

**Expected result:** All benchmarks within 1.0–1.5× of GCC

### Quarter 3+:

8. **Phase 11b: Reduction Patterns**
9. **Phase 13: Instruction Scheduling**
10. Additional patterns, more aggressive optimizations

**Target:** Competitive with GCC -O2 on most workloads

---

## Alternative: Target Specific Workloads

Instead of chasing GCC on all benchmarks, focus on **specific high-value domains:**

### Option A: Scientific Computing
- Prioritize matmul, stencils, reductions
- Phases 7a, 11b, 13 (AVX2, reductions, scheduling)
- Target: **Matrix ops at GCC speed**

### Option B: Systems Programming
- Prioritize sieve, qsort, string ops
- Phases 9, 12 (strength reduction, better regalloc)
- Target: **Kernel/database code within 1.2× of GCC**

### Option C: Mixed (Recommended)
- Do AVX2 first (universal FP benefit)
- Then inlining (helps recursive code)
- Then PGO (helps everything)
- Target: **Competitive on typical workloads**

---

## Low-Hanging Fruit (Quick Wins)

### Week 1: Fix Remainder Loop
- Already planned, just needs implementation
- Makes vectorization correct for odd N

### Week 2-3: AVX2 Vectorization
- Incremental upgrade from SSE2 (code is ready)
- Immediate 2× matmul gain

### Week 4: Indexed Addressing
- Backend improvement, no IR changes needed
- 5–10% on loop-heavy code

**3-week sprint → LCCC competitive with GCC on matmul** ✓

---

## Metrics to Track

| Metric | Current | Phase 7a | Phase 8 | Phase 10 |
|--------|---------|----------|---------|----------|
| matmul gap | 2.0× | **1.0×** | 1.0× | 0.9× |
| fib gap | 3.7× | 3.7× | **2.0×** | 1.5× |
| arith_loop gap | 1.5× | 1.5× | 1.4× | **1.2×** |
| sieve gap | 1.5× | 1.4× | 1.4× | **1.2×** |
| qsort gap | 1.1× | 1.1× | 1.1× | **1.0×** |

**Bold** = expected significant improvement

---

## Conclusion

**Next immediate priority:** Phase 7a (AVX2 vectorization) — closes matmul gap from 2× to 1×.

**Long-term goal:** All benchmarks within 1.2–1.5× of GCC -O2 by end of 2026.

**Philosophy:** We're not trying to beat GCC. We're making a C compiler that's **fast enough
for real systems work** while being fully self-contained (no external toolchain dependencies).
The 1.5× target is practical and achievable.

**Current status:** 🎯 **6 phases complete, matmul at 2× of GCC, heading toward 1×**
