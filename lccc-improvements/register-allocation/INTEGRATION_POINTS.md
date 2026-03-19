# Register Allocator Integration Points

## Summary

The register allocator is called from `prologue.rs` (architecture-specific) and must integrate seamlessly with:
1. Stack space calculation
2. Prologue/epilogue generation
3. Code generation

This document maps exact integration points and defines the interface for replacing the allocator.

## Current Integration Flow

### 1. Call Site: `prologue.rs:81-85` (x86 example)

```rust
let (reg_assigned, cached_liveness) = crate::backend::generation::run_regalloc_and_merge_clobbers(
    func, available_regs, caller_saved_regs, &asm_clobbered_regs,
    &mut self.reg_assignments, &mut self.used_callee_saved,
    false,
);
```

**Location:** `ccc/src/backend/x86/codegen/prologue.rs:81-85`

**Similar locations:**
- `ccc/src/backend/i686/codegen/prologue.rs` (32-bit x86)
- `ccc/src/backend/arm/codegen/prologue.rs` (AArch64)
- `ccc/src/backend/riscv/codegen/prologue.rs` (RISC-V 64)

### 2. Wrapper Function: `generation.rs:run_regalloc_and_merge_clobbers()`

**Location:** `ccc/src/backend/generation.rs` (to find exact line, search for this function)

**Purpose:** Calls the core allocator and merges inline asm clobbered registers.

### 3. Core Allocator Entry: `regalloc.rs:80-324`

**Function:** `pub fn allocate_registers(func: &IrFunction, config: &RegAllocConfig) -> RegAllocResult`

**Inputs:**
```rust
pub struct RegAllocConfig {
    pub available_regs: Vec<PhysReg>,              // Callee-saved (x86: rbx, r12-r15)
    pub caller_saved_regs: Vec<PhysReg>,          // Caller-saved (x86: r11, r10, r8, r9)
    pub allow_inline_asm_regalloc: bool,          // Only RISC-V = true
}
```

**Output:**
```rust
pub struct RegAllocResult {
    pub assignments: FxHashMap<u32, PhysReg>,     // Value ID → physical register
    pub used_regs: Vec<PhysReg>,                  // Registers actually used (for prologue/epilogue)
    pub liveness: Option<super::liveness::LivenessResult>,  // Cached for stack layout
}
```

### 4. Stack Space Calculation: `prologue.rs:87-92`

```rust
let mut space = calculate_stack_space_common(&mut self.state, func, 0, |space, alloc_size, align| {
    let effective_align = if align > 0 { align.max(8) } else { 8 };
    let alloc = (alloc_size + 7) & !7;
    let new_space = ((space + alloc + effective_align - 1) / effective_align) * effective_align;
    (-new_space, new_space)
}, &reg_assigned, &X86_CALLEE_SAVED, cached_liveness, false);
```

**Key parameter:** `&reg_assigned` (values that got registers) — these skip stack allocation.

### 5. Prologue/Epilogue Emission: `prologue.rs:111-149` and beyond

```rust
pub(super) fn emit_prologue_impl(&mut self, func: &IrFunction, frame_size: i64) {
    // ...
    let used_regs = self.used_callee_saved.clone();
    for (i, &reg) in used_regs.iter().enumerate() {
        let offset = -frame_size + (i as i64 * 8);
        let reg_name = phys_reg_name(reg);
        self.state.out.emit_instr_reg_rbp("    movq", reg_name, offset);
    }
    // ...
}
```

**Key field:** `self.used_callee_saved` (populated from `RegAllocResult::used_regs`)

### 6. Code Generation: References `self.reg_assignments`

During instruction emission, codegen checks `self.reg_assignments` to see if a value is assigned a register.

**Example (pseudo-code):**
```rust
if let Some(reg) = self.reg_assignments.get(&value_id) {
    emit_register_access(reg);  // Use register instead of stack
} else {
    emit_stack_access(value_id);  // Fall back to stack slot
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ allocate_registers(func, config) → RegAllocResult            │
│   Input: IrFunction, available_regs, caller_saved_regs      │
│   Output: assignments (Value→PhysReg), used_regs, liveness  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  Stack layout   Prologue/Epilogue  Codegen
  (skip stack    (save/restore      (emit reg
   slots for     regs for           accesses)
   allocated     allocated vars)
   values)
```

## Register Numbering (x86-64 Example)

**PhysReg IDs:**

| PhysReg(n) | Register | Type | ABI |
|-----------|----------|------|-----|
| 0 | rax | Caller-saved (implicit) | Accumulator |
| 1 | rbx | Callee-saved | General |
| 2 | rcx | Caller-saved | arg3 |
| 3 | rdx | Caller-saved | arg2 |
| 4 | rsi | Caller-saved | arg1 |
| 5 | rdi | Caller-saved | arg0 |
| 6 | r8 | Caller-saved | arg4 |
| 7 | r9 | Caller-saved | arg5 |
| 8 | r10 | Caller-saved | Scratch |
| 9 | r11 | Caller-saved | Scratch |
| 10 | r12 | Callee-saved | General |
| 11 | r13 | Callee-saved | General |
| 12 | r14 | Callee-saved | General |
| 13 | r15 | Callee-saved | General |

**Configured in:**
- `X86_CALLEE_SAVED` = [rbx, r12, r13, r14, r15] = [PhysReg(1), PhysReg(10), PhysReg(11), PhysReg(12), PhysReg(13)]
- `X86_CALLER_SAVED` = [r11, r10, r8, r9] = [PhysReg(9), PhysReg(8), PhysReg(6), PhysReg(7)] (filtered based on function characteristics)

## Configuration Pre-Processing

Before calling allocator, prologue.rs filters caller-saved registers based on function characteristics:

```rust
if has_indirect_call {
    caller_saved_regs.retain(|r| r.0 != 11);  // Remove r10
}
if has_i128_ops {
    caller_saved_regs.retain(|r| r.0 != 12 && r.0 != 13 && r.0 != 14 && r.0 != 15);
    // Remove r8, r9, rdi, rsi (used by i128 operations)
}
if has_atomic_rmw {
    caller_saved_regs.retain(|r| r.0 != 12);  // Remove r8
}
```

## Interface Stability

**The interface that replacement allocator must maintain:**

```rust
pub fn allocate_registers(
    func: &IrFunction,
    config: &RegAllocConfig,
) -> RegAllocResult
```

**No changes needed to:**
- `PhysReg` struct
- `RegAllocConfig` struct
- `RegAllocResult` struct
- Call sites in prologue.rs
- Code generation sites

## Performance Expectations

### Current Behavior
- Returns ~5% of values with registers
- Typical function: 0-4 values allocated
- Stack frame: varies, but often 1KB+

### Expected After Improvement
- Returns ~50% of eligible values with registers
- Better register packing (especially in loops)
- Stack frame: only for truly uncovered values + alloca'd variables

## Testing Approach

1. **Compile test function:**
   ```c
   int compute_32(int a, int b, ..., int f2) {
       return a + b + ... + f2;
   }
   ```

2. **Generate assembly:**
   ```bash
   ./ccc -S test.c -o test.s
   ```

3. **Compare register assignments:**
   - Before: stack only
   - After: registers for all parameters

4. **Measure stack frame:**
   - Before: ~11KB
   - After: minimal (only alignment padding)

## Next Steps

1. **Design new allocator** with interval splitting and coalescing (WEEK 1)
2. **Implement LiveInterval tracking** with split points (WEEK 2)
3. **Test on 32-variable function** (WEEK 2-3)
4. **Validate on SQLite benchmark** (WEEK 3)
5. **Compare register assignments** to baseline

## Architecture-Specific Notes

The allocator is **architecture-agnostic** because:
- `RegAllocConfig` provides available registers per-arch
- `LiveInterval` is arch-independent
- `PhysReg` is a simple ID number

Each architecture passes its own register lists:
- **x86-64:** 16 GPR (0-15)
- **i686:** 8 GPR (0-7)
- **AArch64:** 32 GPR (0-31)
- **RISC-V 64:** 32 GPR (0-31)

New allocator must respect these boundaries and not make arch-specific assumptions.
