# Phase 1: Register Allocation Implementation Checklist

## Overview
Implement linear scan register allocator to improve register allocation beyond current 3-phase algorithm. Current allocator is conservative; we'll expand it to allocate more variables to registers, reducing stack pressure and improving performance.

## Timeline: 2-3 weeks

## Detailed Implementation Steps

### Week 1: Infrastructure & Analysis

#### 1.1: Deep Study of Current Allocator
**Goal**: Fully understand regalloc.rs implementation

**Tasks**:
- [ ] Read entire regalloc.rs (573 lines)
- [ ] Understand 3-phase algorithm (Phase 1: callee-saved, Phase 2: caller-saved, Phase 3: spillover)
- [ ] Identify which values qualify for allocation
- [ ] Map register names to PhysReg IDs (1-5: callee, 10-15: caller)
- [ ] Understand RegAllocConfig input structure
- [ ] Understand RegAllocResult output structure
- [ ] Trace execution with specific test case

**Key Files**:
- `ccc/src/backend/regalloc.rs:248-317` (allocation phases)
- `ccc/src/backend/regalloc.rs:155-226` (value eligibility)
- `ccc/src/backend/liveness.rs:42-52` (LivenessResult)

**Success Criteria**: 
- Can explain each phase in detail
- Understand why only ~5% of values get allocated
- Know how to read LivenessResult output

#### 1.2: Create Diagnostic Instrumentation
**Goal**: Add debug output to measure current allocator behavior

**Tasks**:
- [ ] Add logging to regalloc.rs:248-317 to show:
  - Which values were considered for allocation
  - Which values got registers
  - Which values got spilled
  - Register assignment statistics
- [ ] Add logging to liveness.rs to show:
  - Live range calculations
  - Loop depth for each value
- [ ] Create test program to capture this output

**Output**: Analysis of allocator behavior on test cases

**Success Criteria**: 
- Can run test and see allocator decisions
- Understand performance impact of allocation choices

#### 1.3: Design Linear Scan Algorithm
**Goal**: Design the replacement allocator

**Tasks**:
- [ ] Study LLVM LinearScan paper/code (reference: RegAllocLinearScan.cpp)
- [ ] Design algorithm for x86-64 (16 general-purpose registers)
- [ ] Design spill cost heuristic
- [ ] Design register preference/hint system
- [ ] Design split point strategy for long intervals
- [ ] Decide on coalescing strategy

**Document**: `lccc-improvements/register-allocation/LINEAR_SCAN_DESIGN.md`

**Success Criteria**:
- Clear design document
- Algorithm handles all x86-64 register constraints
- Plan for incremental implementation

#### 1.4: Plan Integration Points
**Goal**: Understand where linear scan plugs in

**Tasks**:
- [ ] Trace prologue.rs:81-84 call
- [ ] Understand RegAllocConfig creation (prologue.rs)
- [ ] Understand RegAllocResult usage (prologue.rs:87-92)
- [ ] Plan to replace allocate_registers() while keeping interface
- [ ] Identify any dependencies on 3-phase structure

**Success Criteria**:
- Know exactly where to plug in new allocator
- No interface changes needed

### Week 2: Core Implementation

#### 2.1: Create Live Range Module
**Goal**: Implement LiveRange and LiveInterval data structures

**Tasks**:
- [ ] Create `ccc/src/backend/live_range.rs` (or extend liveness.rs)
- [ ] Define LiveInterval struct:
  - variable: u32
  - start: usize (program point)
  - end: usize (program point)
  - uses: Vec<usize> (use points)
  - loop_depth: u32 (from liveness)
- [ ] Implement comparison/sorting traits
- [ ] Implement methods:
  - `overlaps_with(other: &LiveInterval) -> bool`
  - `conflicts_at_use_point(use_point: usize) -> bool`
  - `split_at(point: usize) -> (LiveInterval, LiveInterval)`
- [ ] Write unit tests for LiveInterval

**Success Criteria**:
- Live ranges compute correctly
- All unit tests pass

#### 2.2: Create Register Pool Module
**Goal**: Manage available registers during allocation

**Tasks**:
- [ ] Create register pool with x86-64 registers
- [ ] Implement register allocation tracking:
  - `fn allocate_register(hint: Option<PhysReg>) -> Option<PhysReg>`
  - `fn free_register(reg: PhysReg)`
  - `fn is_available(reg: PhysReg) -> bool`
- [ ] Implement register classes (ABI-aware):
  - Parameter passing (rdi, rsi, rdx, rcx, r8, r9)
  - Return value (rax, rdx)
  - Callee-saved (rbx, r12-r15)
  - Caller-saved (r8-r11, etc)
- [ ] Implement register hints (prefer certain registers)

**Success Criteria**:
- Register pool tracks allocation state correctly
- All register constraints honored

#### 2.3: Create Linear Scan Allocator
**Goal**: Implement the main allocation algorithm

**Tasks**:
- [ ] Create `ccc/src/backend/linear_scan.rs`
- [ ] Implement LinearScanAllocator struct:
  ```rust
  pub struct LinearScanAllocator {
      intervals: Vec<LiveInterval>,
      registers: RegisterPool,
      assignments: FxHashMap<u32, PhysReg>,
      spilled: Vec<u32>,
  }
  ```
- [ ] Implement allocation algorithm:
  1. Build live intervals from LivenessResult
  2. Sort intervals by start point
  3. For each interval:
     - Try to allocate register
     - If no register available, mark for spilling
     - Coalesce moves where possible
  4. Return assignments

**Key Implementation Details**:
- Handle register hints from function ABI
- Use loop depth for prioritization in spill decisions
- Implement register splitting for long intervals
- Handle call-clobbered registers

**Success Criteria**:
- Allocator compiles and links
- Passes unit tests on simple functions

#### 2.4: Integration & Testing
**Goal**: Hook allocator into compilation pipeline

**Tasks**:
- [ ] Create wrapper function:
  ```rust
  pub fn allocate_with_linear_scan(
      func: &Function,
      config: &RegAllocConfig,
  ) -> RegAllocResult
  ```
- [ ] Modify regalloc.rs to use new allocator (keep old as fallback)
- [ ] Add feature flag `linear-scan-allocator` for A/B testing
- [ ] Create test cases:
  - Simple function (5 variables)
  - Medium function (20 variables)
  - Complex function (50+ variables)
  - Stress test (100+ variables)
- [ ] Test compilation of test suite

**Success Criteria**:
- All 499 tests still pass
- New allocator active when feature enabled
- Can compare old vs new allocation decisions

### Week 3: Validation & Benchmarking

#### 3.1: Correctness Testing
**Goal**: Verify generated code is correct

**Tasks**:
- [ ] Build with linear scan enabled
- [ ] Run entire test suite: `cargo test --release`
- [ ] Compile micro-benchmarks
- [ ] Run binaries and compare output with GCC
- [ ] Test edge cases:
  - Functions with no allocatable values
  - Functions with more variables than registers
  - Functions with complex control flow
  - Functions with loops at various depths

**Success Criteria**:
- All tests pass
- Binary outputs match expected results
- No crashes or errors

#### 3.2: Performance Analysis
**Goal**: Measure improvement from new allocator

**Tasks**:
- [ ] Compile test programs with old allocator
  - `stress_ccc_old_O0`
  - `stress_ccc_old_O2`
- [ ] Compile with linear scan allocator
  - `stress_ccc_new_O0`
  - `stress_ccc_new_O2`
- [ ] Compare:
  - Binary sizes
  - Stack space usage
  - Runtime performance
  - Compilation time
- [ ] Run assembly analysis:
  - Count register vs stack operations
  - Measure instruction count
  - Measure memory access patterns
- [ ] Document results in performance report

**Metrics to Track**:
- Stack frame size reduction
- Register allocation percentage increase
- Binary size change
- Runtime speedup
- Compilation time impact

**Success Criteria**:
- At least 3-4x speedup on register stress test
- Stack frame size reduced by 50%+
- No compilation time regression

#### 3.3: Edge Case Handling
**Goal**: Fix issues discovered during testing

**Tasks**:
- [ ] Document any failures or unexpected behavior
- [ ] Fix register allocation bugs
- [ ] Handle special cases:
  - Inline assembly register constraints
  - Function parameters and returns
  - Call-clobbered registers
  - Setjmp/longjmp handling
- [ ] Re-run tests after fixes

**Success Criteria**:
- All edge cases handled correctly
- Full test suite passes

#### 3.4: Optimization & Fine-tuning
**Goal**: Improve allocator performance and quality

**Tasks**:
- [ ] Profile allocator runtime (measure time spent in allocation)
- [ ] Optimize hot paths (sorting, searching)
- [ ] Improve heuristics:
  - Better register hints
  - Smarter spill decisions
  - Coalescing improvements
- [ ] Measure impact on real programs (SQLite when ready)

**Success Criteria**:
- Allocator is reasonably fast (<1% compile time overhead)
- Further improvements identified for future work

## Fallback Plan

If linear scan implementation hits unexpected issues:
1. Keep both allocators (feature flag)
2. Default to current allocator for stability
3. Use linear scan in Phase 2 after Phase 1 is stable
4. Consider hybrid approach (linear scan + 3-phase)

## Definition of "Done" for Phase 1

1. **Correctness**: All 499 tests pass ✅
2. **Performance**: 3-4x speedup on register stress test ✅
3. **Code Quality**: Clean, documented, tested code ✅
4. **Integration**: Seamless integration with existing pipeline ✅
5. **Benchmarking**: Performance improvements documented ✅

## File Structure (To Create)

```
ccc/src/backend/
├── linear_scan.rs          (NEW - main allocator)
├── live_range.rs           (NEW or extend liveness.rs)
├── register_pool.rs        (NEW - register tracking)
└── regalloc.rs             (MODIFY - keep old as fallback)

Tests:
ccc/tests/
├── linear_scan_basic.rs    (NEW)
└── linear_scan_stress.rs   (NEW)

Benchmarks:
lccc-improvements/
└── register-allocation/
    └── benches/
        └── allocator_comparison.rs
```

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Stack frame | 256 bytes | < 100 bytes |
| Registers allocated | ~5% | 30-50% |
| Register-to-stack ratio | Mostly stack | 2:1 to 3:1 |
| 32-var function runtime | 8.6s | < 2s |
| Binary size | 4.27 MB (SQLite) | < 3 MB |

## Risk Assessment

**Low Risk**:
- Clean interface separation (RegAllocConfig → RegAllocResult)
- Can use feature flag to enable/disable
- Existing allocator remains as fallback

**Medium Risk**:
- Implementation complexity of linear scan algorithm
- May need to handle edge cases not in initial design

**Mitigation**:
- Careful testing after each step
- Incremental integration (one architecture at a time)
- Reference implementations available (LLVM, GCC)

