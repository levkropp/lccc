# Current Register Allocator Analysis

## Overview

The current register allocator in `ccc/src/backend/regalloc.rs` (574 lines) is a simplified **three-phase linear scan** that conservatively allocates registers to IR values. This analysis documents the current behavior, limitations, and opportunities for improvement.

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `ccc/src/backend/regalloc.rs` | 574 | Main allocator: 3-phase algorithm |
| `ccc/src/backend/liveness.rs` | ~400 | Liveness analysis: live interval computation |
| `ccc/src/backend/prologue.rs` | ~300 | Integration: calls allocator, uses results |

## Three-Phase Algorithm

### Phase 1: Callee-Saved Registers (Lines 248-264)

**Purpose:** Allocate values whose live ranges **span function calls** to callee-saved registers.

**Logic:**
```rust
for interval in &candidates {
    if let Some(reg_idx) = find_best_callee_reg(&reg_free_until, interval.start, ...) {
        reg_free_until[reg_idx] = interval.end + 1;
        assignments.insert(interval.value_id, config.available_regs[reg_idx]);
    }
}
```

**Key Properties:**
- Values assigned here must survive function calls
- Callee-saved registers are preserved by the ABI (rbx, r12-r15 on x86; s1, s7-s11 on RISC-V)
- Prologue/epilogue must save/restore these registers
- Cost: one-time prologue/epilogue overhead, amortized across function calls

**Register Preference:**
- Prefers registers already marked as used (to minimize prologue/epilogue cost)
- Falls back to new registers if needed

### Phase 2: Caller-Saved Registers (Lines 269-295)

**Purpose:** Allocate values whose live ranges **do NOT span any function call** to caller-saved registers.

**Logic:**
```rust
for interval in &caller_candidates {
    let mut best: Option<usize> = None;
    for (i, &free_until) in caller_free_until.iter().enumerate() {
        if free_until <= interval.start && (best.is_none() || free_until < best_free_time) {
            best = Some(i);
            best_free_time = free_until;
        }
    }
    if let Some(reg_idx) = best {
        assignments.insert(interval.value_id, config.caller_saved_regs[reg_idx]);
    }
}
```

**Key Properties:**
- Caller-saved registers are destroyed by calls (r11, r10, r8, r9 on x86)
- Since these values don't cross calls, no prologue/epilogue save/restore needed
- If no caller-saved register is free at the interval's start, allocation fails (no spill)

**Register Preference:**
- Greedily takes the register that freed up earliest
- No preference for already-used registers (they're caller-saved, so no prologue overhead)

### Phase 3: Callee-Saved Spillover (Lines 297-317)

**Purpose:** Assign remaining callee-saved registers to high-priority values that didn't fit in Phase 2.

**Motivation:**
- Phase 2 might run out of caller-saved registers
- Some functions are call-free loops (hash functions, matrix multiply, sorting)
- In these hot loops, all values compete for the small caller-saved pool
- Better to use one prologue/epilogue save/restore than to spill to stack in the loop

**Logic:**
```rust
for interval in &spillover_candidates {
    if let Some(reg_idx) = find_best_callee_reg(&reg_free_until, interval.start, ...) {
        reg_free_until[reg_idx] = interval.end + 1;
        assignments.insert(interval.value_id, config.available_regs[reg_idx]);
    }
}
```

## Eligibility Filtering

### Values That Get Registers

**Lines 155-226:** Whitelist approach — only simple instructions whose results go through standard accumulator paths:

```
✓ BinOp, UnaryOp        (binary, unary arithmetic)
✓ Cmp                   (integer comparisons)
✓ Cast                  (integer casts)
✓ Load                  (integer loads)
✓ GetElementPtr         (pointer arithmetic)
✓ Copy                  (register-to-register copies)
✓ Call / CallIndirect   (call results arriving in rax/x0/a0)
✓ Select                (conditional selection)
✓ GlobalAddr, LabelAddr (address constants)
✓ AtomicLoad, AtomicRmw, AtomicCmpxchg (atomic operations)
✓ ParamRef              (function parameters)
```

### Values That Are Excluded

**Explicitly excluded (lines 326-486):**

1. **Non-GPR types:**
   - Float types (f32, f64, f128)
   - 128-bit integers (i128, u128)
   - i64/u64 on 32-bit targets (need two registers: edx:eax)

2. **Used in non-register-aware instructions:**
   - CallIndirect function pointers (must go through stack slot resolution)
   - Memcpy pointers (dest and src)
   - va_arg/va_start/va_end/va_copy pointers
   - Atomic operation pointers
   - InlineAsm operands (unless `allow_inline_asm_regalloc` is true)
   - StackRestore pointers

3. **Alloca values:**
   - Never eligible (they represent stack addresses, must stay in memory)

**Copy propagation chains (lines 386-412):**
- If a Copy sources a non-GPR value, the Copy dest is also marked non-GPR
- Iterates to fixpoint to handle chains

## Prioritization (Lines 108-130, 523-535)

### Use Counting with Loop Depth Weighting

```rust
let weight: u64 = match block_loop_depth {
    0 => 1,
    1 => 10,
    2 => 100,
    3 => 1_000,
    _ => 10_000,
};
*use_count.entry(v.0).or_insert(0) += weight;
```

**Rationale:** Uses inside loops matter more because they execute more frequently.

**Sorting (lines 523-535):**
```rust
candidates.sort_by(|a, b| {
    let score_a = use_count.get(&a.value_id).copied().unwrap_or(1);
    let score_b = use_count.get(&b.value_id).copied().unwrap_or(1);
    score_b.cmp(&score_a)
        .then_with(|| {
            let len_a = (a.end - a.start) as u64;
            let len_b = (b.end - b.start) as u64;
            len_b.cmp(&len_a)
        })
});
```

1. Primary: Higher use count (weighted by loop depth) → earlier in list
2. Tiebreaker: Longer intervals → earlier in list

## Liveness Analysis (liveness.rs)

### Live Interval Computation

```rust
pub struct LiveInterval {
    pub start: u32,        // Program point where value is defined
    pub end: u32,          // Last program point where value is used
    pub value_id: u32,
}
```

**How it works:**
1. Assign sequential program points to all instructions/terminators
2. Run backward dataflow to compute live-in/live-out sets per block
3. Build intervals from def/use points and live-through blocks
4. Handles loops correctly via backward dataflow iteration

### Call Point Tracking

```rust
pub struct LivenessResult {
    pub intervals: Vec<LiveInterval>,
    pub call_points: Vec<u32>,      // Program points of Call/CallIndirect
    pub block_loop_depth: Vec<u32>, // Loop nesting depth per block
}
```

Used by allocator to determine `spans_any_call()`:
```rust
fn spans_any_call(iv: &LiveInterval, call_points: &[u32]) -> bool {
    let start_idx = call_points.partition_point(|&cp| cp < iv.start);
    start_idx < call_points.len() && call_points[start_idx] <= iv.end
}
```

## Current Behavior: Why It's Conservative

### Problem 1: High Eligibility Filter

Only ~5% of values get registers because:
1. Float operations excluded entirely
2. i128/u64 on 32-bit excluded
3. Many pointer values excluded (atomics, memcpy, etc.)
4. Copy chains propagate non-GPR status aggressively

### Problem 2: Simple Best-Fit Algorithm

- Phase 1: Finds first free register (not optimal packing)
- Phase 2: Greedy earliest-freed (doesn't consider future allocation patterns)
- No interval splitting (value either gets whole interval or none)
- No coalescing (doesn't eliminate redundant copies)

### Problem 3: Limited Live Range Analysis

- Intervals are [definition, last use]
- No dead code elimination before allocation
- No value numbering or copy removal

### Problem 4: Register Ordering

Current implementation depends on register list order. Improvement opportunity: smarter register selection based on ABI constraints and usage patterns.

## Performance Impact

### Stack Spilling

Current: 32-variable function uses **11KB stack** instead of registers.

Why:
1. Only ~5% of values eligible
2. Even fewer fit in available registers (typically 4-8 per phase)
3. Remaining values spill to stack
4. Each stack access = 3 memory operations (address computation, load/store, possibly cache miss)

### "Shuttle Pattern"

Values repeatedly spill/reload between registers and stack:
```asm
mov    %rax, -0x20(%rbp)    # spill
...
mov    -0x20(%rbp), %rax    # reload
mov    %rax, %rbx
```

Should be:
```asm
mov    %rax, %rbx           # register-to-register
```

### Cache Efficiency

Larger stack frame = more memory pressure = more cache misses.

## Improvement Opportunities

### Opportunity 1: Expand Eligibility

- Handle float operations (via soft-float or float-specific register paths)
- Better copy propagation (eliminate redundant copies before allocation)
- Smarter handling of atomic/memcpy pointers (track which can be promoted)

### Opportunity 2: Better Allocation Algorithm

- Linear scan with interval splitting (allow values to move between registers/stack)
- Register coalescing (merge values that are assigned copies)
- Graph coloring (optimal but slower)

### Opportunity 3: Improve Register Selection

- Consider use frequencies and block execution counts
- Prefer registers with good ABI placement
- Avoid clobbering high-value registers

### Opportunity 4: Post-Allocation Optimization

- Dead store elimination
- Copy propagation in generated code
- Peephole optimization tuned for allocation output

## Test Case: 32-Variable Function

To measure current behavior, create a function:

```c
int compute(int a, int b, int c, int d, int e, int f, int g, int h,
            int i, int j, int k, int l, int m, int n, int o, int p,
            int q, int r, int s, int t, int u, int v, int w, int x,
            int y, int z, int a2, int b2, int c2, int d2, int e2, int f2) {
    int sum = 0;
    sum += a + b + c + d + e + f + g + h;
    sum += i + j + k + l + m + n + o + p;
    sum += q + r + s + t + u + v + w + x;
    sum += y + z + a2 + b2 + c2 + d2 + e2 + f2;
    return sum;
}
```

Expected baseline:
- Stack frame: ~11KB
- Register assignments: ~0 (all parameters on stack)
- Generated code: many mem→reg→ALU→reg→mem sequences

Expected after improvement:
- Stack frame: ~0 (all in registers)
- Register assignments: 32 (with splitting across phases)
- Generated code: direct reg→reg operations, minimal stack access

## Next Steps

1. **Add instrumentation** to log allocator decisions
2. **Study LLVM LinearScan** for better algorithms
3. **Design replacement allocator** with interval splitting and coalescing
4. **Implement new allocator** (Week 2)
5. **Benchmark improvements** (Week 3)
