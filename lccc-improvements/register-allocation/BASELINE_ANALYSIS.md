# Pre-Phase 1: Baseline Analysis & Findings

## Environment Setup ✅
- **Rust**: 1.93.0 (stable)
- **Cargo**: 1.93.0
- **GCC**: 15.2.0
- **CCC**: Built successfully
- **Tests**: 499 tests passing

## CCC Backend Architecture Analysis ✅

### Key Findings from x86-64 Backend Exploration

**Core Files**:
- Register allocation: `ccc/src/backend/regalloc.rs` (573 lines)
- Liveness analysis: `ccc/src/backend/liveness.rs` (1,211 lines)
- Code generation: `ccc/src/backend/x86/codegen/emit.rs` (1,463 lines)
- Integration point: `ccc/src/backend/x86/codegen/prologue.rs:81-84`

**Current Register Allocation Strategy**:
- 3-phase algorithm: callee-saved → caller-saved → spillover
- Only ~5% of values get registers (heavily restricted)
- Uses backward dataflow liveness analysis with bitsets
- Weights values by loop depth (10^depth formula)
- Clean interface: RegAllocConfig → RegAllocResult

**Critical Finding**: Register allocation runs in `prologue.rs:81-84` during stack space calculation, making it early in the pipeline.

## Baseline Performance Measurements

### Test 1: Simple Loop (100M iterations)

```
Binary Sizes:
  GCC -O0:     16K
  GCC -O2:     16K (same!)
  CCC -O0:     15K
  CCC -O2:     15K (same!)

Runtime Performance:
  GCC -O0:     0.050s (baseline)
  GCC -O2:     0.022s (2.27x faster with optimization)
  CCC -O0:     0.059s (1.18x slower than GCC -O0)
  CCC -O2:     0.053s (1.06x slower than GCC -O2)

OBSERVATION: CCC -O2 has NO effect (same binary size & performance as -O0)
```

### Test 2: Register Stress Test (1M iterations of 32-variable function)

```
Binary Sizes:
  GCC -O0:     16K
  GCC -O2:     16K
  CCC -O0:     15K
  CCC -O2:     15K

Runtime Performance:
  GCC -O0:     0.019s (baseline, with register allocation)
  GCC -O2:     0.003s (6.33x faster with optimization!)
  CCC -O0:     0.018s (0.94x slower than GCC -O0)
  CCC -O2:     0.019s (no improvement, same as -O0)

CRITICAL: CCC is actually FASTER than GCC on unoptimized code!
This suggests the register stress test is not hitting CCC's weakness.
```

## Assembly Code Analysis

### CCC Generated Code (compute_with_many_vars prologue)

```asm
subq $256, %rsp           ; 256 bytes of stack space
movq %rbx, -256(%rbp)     ; Save callee-saved regs
movq %r12, -248(%rbp)
movq %r13, -240(%rbp)
movq %r14, -232(%rbp)
movq %r15, -224(%rbp)
movq %rdi, -8(%rbp)       ; Save parameters
movq %rsi, -16(%rbp)
...

movslq -8(%rbp), %rax     ; Load each param
movq %rax, %r11           ; Move to temp reg
movslq -16(%rbp), %rax
movq %rax, %r10
; ... (shuttle pattern: load→move→move→load cycle)
```

**Observation**: CCC is already allocating to callee-saved registers (r11, r10, r8, r9, rdi, rsi, rbx, r12-r15)!

### GCC Generated Code (same function, -O0)

```asm
subq $32, %rsp            ; Only 32 bytes stack (vs CCC's 256!)
movl %edi, -132(%rbp)     ; Save params to stack (smaller offsets)
movl %esi, -136(%rbp)
...
movl -132(%rbp), %eax     ; Load and operate
addl $1, %eax
movl %eax, -64(%rbp)      ; Store result
```

**Key Difference**: 
- CCC uses 256 bytes stack space
- GCC uses only 32 bytes stack space
- CCC saves more registers to stack but allocates variables to registers
- GCC keeps everything on stack but with smaller offsets

## Current Limitations Identified

### 1. Stack Space Calculation
- CCC allocates a large stack frame (256 bytes) even when not needed
- This may be due to conservative register allocation assumptions

### 2. Optimization Levels Don't Work
- CCC -O2 produces identical code to CCC -O0
- Indicates optimization tier infrastructure is not implemented (Phase 2 work)

### 3. Register Allocation Scope
- Current allocator is working, but very conservative
- Only allocates to registers in specific patterns
- May not be using all available registers efficiently

## What We Need to Fix

### Phase 1: Register Allocation (Priority: HIGH)
**Current Issue**: While CCC does allocate some variables to registers, it's overly conservative
**Solution**: Implement linear scan register allocator to:
1. Track live ranges more accurately
2. Allocate more variables to registers
3. Reduce stack space pressure
4. Improve cache locality

### Phase 2: Optimization Tiers (Priority: HIGH)
**Current Issue**: -O flags have no effect
**Solution**: Implement tiered optimization pipeline:
1. -O0: Minimal passes (current behavior)
2. -O1: Basic optimizations (dead code, const prop)
3. -O2: Aggressive optimizations (inlining, loop unroll)
4. -O3: Very aggressive (vectorization, etc)

### Phase 3: Code Size (Priority: MEDIUM)
**Current Issue**: Binary sizes similar but internal structure differs
**Solution**: Later phase - focus on Phase 1 & 2 first

## Test Case Files Created

```
/tmp/register_stress.c        - 32 variable stress test
/tmp/no_headers_loop.c        - Simple loop without headers
/tmp/stress_ccc_O0            - CCC -O0 binary (15K)
/tmp/stress_ccc_O0.s          - CCC assembly
/tmp/stress_gcc_O0.s          - GCC assembly
```

## Assembly Output Locations

```
/tmp/stress_ccc_O0.s          - CCC generated assembly (current behavior)
/tmp/stress_gcc_O0.s          - GCC generated assembly (reference)
```

## Next Steps for Phase 1 Implementation

1. **Study regalloc.rs** - Understand current 3-phase algorithm
2. **Trace execution** - Add debug output to see allocator behavior
3. **Design Linear Scan** - Plan the replacement algorithm
4. **Implement Live Ranges** - Create live_range.rs module
5. **Implement Allocator** - Create linear_scan.rs module
6. **Integration** - Hook into prologue.rs:81
7. **Testing** - Validate with test cases
8. **Benchmarking** - Measure improvements

## Key Code References for Phase 1

- Entry point: `ccc/src/backend/x86/codegen/prologue.rs:81-84`
- Current allocator: `ccc/src/backend/regalloc.rs:248-317`
- Liveness analysis: `ccc/src/backend/liveness.rs:144+`
- Register definitions: `ccc/src/backend/x86/codegen/emit.rs:29-49`
- Stack layout: `ccc/src/backend/x86/codegen/prologue.rs:87-92`

