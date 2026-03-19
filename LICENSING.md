# LCCC Licensing Guide

This document explains the dual licensing structure used in LCCC (Lev's Claude's C Compiler) and how to handle licensing for contributions.

## Quick Summary

| Code Type | License | Details |
|-----------|---------|---------|
| **Original CCC Code** | CC0 1.0 (Public Domain) | No copyright restrictions, attribution not required |
| **LCCC Improvements** | MIT OR Apache-2.0 OR BSD-2-Clause | Your choice, per contribution |
| **Mixed Files** | Both CC0 + your choice | Original sections CC0, new sections your license |

## Licensing Structure

### CCC Code (Original)
**Licensed under**: CC0 1.0 Universal (Public Domain)

All code from the original Claude's C Compiler project:
- CCC frontend (lexer, parser, semantic analysis)
- CCC IR and SSA optimizer
- CCC backends (x86-64, ARM, RISC-V, i686)
- CCC assembler and linker

**What this means**: 
- Use freely for any purpose
- No attribution required
- No copyright restrictions
- More permissive than any open source license

See [`LICENSE-CC0-CCC`](./LICENSE-CC0-CCC) for full text.

### LCCC Code (New & Improvements)
**Licensed under**: MIT OR Apache-2.0 OR BSD-2-Clause (your choice)

All code authored as part of LCCC optimization efforts:
- Register allocation improvements (Phase 1)
- Optimization tier implementation (Phase 2)
- Code size reduction passes (Phase 3)
- DWARF debug information improvements (Phase 4)
- Linker relocation fixes (Phase 5)
- Performance benchmarking tools
- Any new modules or major rewrites

**What this means**:
- You pick ONE of three licenses for each contribution
- Standard open source protections
- Users can use your code under their chosen license
- Each license file available: [`LICENSE-MIT`](./LICENSE-MIT), [`LICENSE-APACHE`](./LICENSE-APACHE), [`LICENSE-BSD`](./LICENSE-BSD)

### Hybrid Files (CCC Base + LCCC Improvements)
**Licensed under**: Both CC0 + your chosen license

When you modify existing CCC files to add improvements:
- Original CCC code sections remain CC0
- Your new/modified sections are under your license choice
- Both licenses apply to the file
- Users can use either license for their needs

**Example structure**:
```rust
// File: ccc/src/backend/regalloc.rs
//
// Original CCC code: Lines 1-247, 318-573
// Licensed under: CC0 1.0 Universal
//
// LCCC improvements: Lines 248-317 (linear scan integration)
// Licensed under: MIT (or your choice)

// ... code ...
```

## How to Determine Which License Applies

For any code in LCCC, ask these questions in order:

### Question 1: Is this code new or substantially rewritten?
**YES** → Use MIT/Apache/BSD (your choice)
- You are the author of new optimization work
- Pick the license that best fits your preference

**NO** → Go to Question 2

### Question 2: Is this code from the original CCC project?
**YES** → CC0 applies (no copyright restrictions)
- Original CCC frontend, IR, backends, assembler, linker
- No license action needed - already public domain
- Respect the original author's CC0 dedication

**NO** → Go to Question 3

### Question 3: Is this a hybrid (CCC base + LCCC improvements)?
**YES** → Both licenses apply
- Identify which sections are CCC (CC0)
- Identify which sections are new (your choice)
- Document clearly in git commits and code headers

**NO** → You found something else
- Document appropriately and ask maintainers

## Practical Examples

### Example 1: Creating a New Module (Pure LCCC Code)

File: `ccc/src/backend/linear_scan.rs`

```rust
// NEW MODULE: Linear Scan Register Allocator
// Licensed under: MIT
// 
// Part of LCCC Phase 1: Register Allocation Improvements

pub struct LinearScanAllocator {
    // ...
}
```

**Git commit**:
```
[LCCC] Phase 1: Implement linear scan register allocator

New module implementing linear scan allocation algorithm.
Replaces overly conservative 3-phase allocator.

License: MIT
```

### Example 2: Modifying Existing CCC File

File: `ccc/src/backend/regalloc.rs` (originally CC0, now with LCCC modifications)

```rust
// ORIGINAL CCC CODE (CC0 1.0 Universal)
// Lines 1-247: Value eligibility checks
// Lines 318-573: Support functions

// LCCC ADDITION (MIT)
// Lines 248-317: Linear scan integration
// 
// New code to hook linear scan allocator while keeping
// compatibility with existing 3-phase algorithm.
// Can be disabled via feature flag.

fn run_linear_scan_allocator(...) {
    // ...
}
```

**Git commit**:
```
[LCCC] Phase 1: Integrate linear scan allocator into regalloc.rs

Added linear scan integration point in existing regalloc.rs.
Original CCC code remains CC0, new integration code MIT.
Feature flag allows fallback to original 3-phase algorithm.

License: MIT (for new code only)
```

### Example 3: Major Rewrite of Existing File

If you substantially rewrite a CCC file (>80% new code):

Option A - Keep both licenses:
```rust
// File mostly rewritten for LCCC improvements
// Original CCC portions: CC0 1.0 Universal
// LCCC rewrite: MIT
// See git log for exact attribution
```

Option B - Replace entirely with new module:
```rust
// This module was deprecated and replaced by:
// ccc/src/backend/new_improved_module.rs
// 
// If you need original CCC code, see git history
```

## License Choice Guide

### MIT License
**Choose if you**:
- Want broad industry compatibility
- Are used to MIT-licensed projects
- Want simple, concise language
- Prefer permissive without restriction

### Apache-2.0 License
**Choose if you**:
- Want explicit patent grant protection
- Are contributing to an Apache project later
- Want more legal specificity
- Need broader liability/warranty clauses

### BSD-2-Clause License
**Choose if you**:
- Want BSD heritage/compatibility
- Prefer concise non-attribution variant
- Are in BSD/academic communities

**Recommendation**: Default to MIT if unsure. All three are compatible with CC0 code and each other.

## Contributing to LCCC

### Your Responsibilities

1. **Clearly mark new code as LCCC** with license header
2. **Preserve CCC code** as CC0 - don't try to relicense it
3. **Document mixed files** with git commit messages
4. **Choose a license** for each contribution (or use default)
5. **Check git history** if unsure about code origin

### In Your Commits

Always include license information:

```
[LCCC] Phase X: Brief description

Details about what was changed and why.

Files modified:
- new_file.rs (NEW, MIT)
- existing_file.rs (CCC code + MIT additions)

License: MIT (or Apache-2.0 or BSD-2-Clause)
```

### In Your Code

Add headers to new files:

```rust
// Linear Scan Register Allocator
// Licensed under: MIT
// Part of LCCC Phase 1

// This module implements a linear scan register allocation algorithm
// to improve upon CCC's overly conservative 3-phase allocator.

pub struct LinearScanAllocator { ... }
```

For modifications to existing files:

```rust
// LCCC ADDITION (MIT):
// New integration point for linear scan allocator
// Original CCC code above/below this section remains CC0

fn integrate_linear_scan(...) { ... }
```

## FAQ

### Q: What if I don't specify a license for my code?

**A**: The default in `Cargo.toml` is `MIT OR Apache-2.0 OR BSD-2-Clause`. Your code uses that by default.

```toml
[workspace.package]
license = "MIT OR Apache-2.0 OR BSD-2-Clause"
```

### Q: Can I relicense CCC code to MIT?

**A**: No. CCC code must remain CC0. However, you don't need to! 
- CC0 is MORE permissive than MIT
- Any code using CC0 parts is already unrestricted
- Relicensing would reduce freedom, not increase it

### Q: Can I use LCCC code in a proprietary project?

**A**: Yes! All LCCC code is under MIT/Apache/BSD, which permit proprietary use. The CC0 parts have even fewer restrictions.

### Q: What if multiple people contribute to one file?

**A**: Document in commit messages and file headers:

```rust
// File: ccc/src/backend/regalloc.rs
//
// Original CCC code: CC0 1.0 Universal
// LCCC additions by Alice (2026): MIT
// LCCC improvements by Bob (2026): MIT
// See git log for detailed attribution
```

### Q: Can I contribute code under a different license?

**A**: Better to stick with MIT/Apache/BSD for consistency. If you really need GPL/AGPL/etc:
- Discuss with maintainers first
- Creates incompatibility issues
- Makes distribution more complex
- Generally not recommended for library code

### Q: What about documentation and non-code files?

**A**: These typically follow the project default (MIT OR Apache-2.0 OR BSD-2-Clause). If unclear, add a header:

```markdown
# Documentation
Licensed under: MIT OR Apache-2.0 OR BSD-2-Clause
```

### Q: If I use LCCC in my project, what do I need to do?

**A**: 
1. Include a copy of the licenses you use (CC0 and/or MIT/Apache/BSD)
2. For MIT/Apache/BSD, include copyright notices (optional for CC0)
3. Follow the license terms you choose

### Q: Is this licensing setup unusual?

**A**: Not really! It's similar to:
- LLVM (has both permissive and restricted code)
- GCC (combines FSF code with other licenses)
- Chromium (combines MIT, Apache, BSD, etc)

Layered licensing is common for large compiler projects.

## File Locations

- **CCC License**: [`LICENSE-CC0-CCC`](./LICENSE-CC0-CCC)
- **MIT License**: [`LICENSE-MIT`](./LICENSE-MIT)
- **Apache License**: [`LICENSE-APACHE`](./LICENSE-APACHE)
- **BSD License**: [`LICENSE-BSD`](./LICENSE-BSD)
- **Project README**: [`README.md`](./README.md) - Overview with licensing structure
- **Contributing Guide**: [`CONTRIBUTING.md`](./CONTRIBUTING.md) - How to contribute

## Summary Table

| Scenario | Action | License |
|----------|--------|---------|
| New optimization module | Create new file | MIT (or your choice) |
| Modify existing CCC file | Add clear comments | Both CC0 + your choice |
| Major rewrite (>80% new) | Consider new module | Your choice |
| Question about origin | Check git history | See `git log --follow` |
| Not sure which license | Use default | MIT OR Apache-2.0 OR BSD-2-Clause |
| Contributing | Mark clearly in commit | Your choice of MIT/Apache/BSD |

---

## Questions?

If you're unsure about licensing for a specific contribution:
1. Check similar files in the codebase
2. Review git history: `git log --follow filename`
3. Ask in issue or PR discussion
4. Default to MIT if still uncertain

The goal is to **maximize freedom** for everyone while **respecting original authors**.

## Related Documents

- **README.md** - Project overview including licensing summary
- **CONTRIBUTING.md** - How to contribute while respecting licenses
- **LICENSE-MIT** - Full MIT License text
- **LICENSE-APACHE** - Full Apache 2.0 License text
- **LICENSE-BSD** - Full BSD 2-Clause License text
- **LICENSE-CC0-CCC** - Full CC0 1.0 License text (CCC code)

---

**Last updated**: 2026-03-19  
**Version**: 1.0

This guide ensures everyone understands the LCCC dual licensing model while respecting the original CCC project's CC0 dedication.
