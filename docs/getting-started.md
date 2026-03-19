---
layout: doc
title: Getting Started
description: Build LCCC from source and compile your first C program.
prev_page:
next_page:
  title: Architecture
  url: /docs/architecture
---

# Getting Started
{:.doc-subtitle}
Build LCCC, compile a C program, and run the benchmark suite in under five minutes.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Rust stable** (2021 edition) | Install via [rustup](https://rustup.rs/) |
| **Linux x86-64 host** | Compiler targets Linux ELF; macOS/Windows are untested |
| **GCC installed** | Needed for built-in headers (`stddef.h`, `stdarg.h`) |
| **Python 3.9+** | For the benchmark runner (optional) |

LCCC uses a completely standalone assembler and linker — no external toolchain is needed at compile time, only for the GCC built-in headers at C preprocessing time.

## Clone and Build

```bash
git clone --recurse-submodules https://github.com/levkropp/lccc.git
cd lccc
cargo build --release
```

This produces five binaries in `target/release/`:

| Binary | Target architecture |
|--------|---------------------|
| `ccc` | x86-64 (default) |
| `ccc-x86` | x86-64 (explicit) |
| `ccc-arm` | AArch64 |
| `ccc-riscv` | RISC-V 64 |
| `ccc-i686` | i686 (32-bit x86) |

> **Note:** The binary is named `ccc` for drop-in GCC compatibility. It reports `gcc (GCC) 14.2.0` to build systems.

## Compile Your First Program

```bash
# LCCC needs GCC's built-in headers for stddef.h, stdarg.h, etc.
GCC_INC="-I/usr/lib/gcc/x86_64-linux-gnu/$(gcc -dumpversion)/include"

cat > hello.c <<'EOF'
#include <stdio.h>
int main(void) {
    printf("Hello from LCCC!\n");
    return 0;
}
EOF

./target/release/ccc $GCC_INC -O2 -o hello hello.c
./hello
# Hello from LCCC!
```

## GCC-Compatible Flags

LCCC accepts the standard GCC command-line interface:

```bash
# Compile and link
ccc -O2 -o output input.c

# Emit assembly
ccc -S -O2 input.c

# Compile to object file only
ccc -c input.c

# Preprocessor only
ccc -E input.c

# Debug info
ccc -g -O2 -o output input.c

# Macros and include paths
ccc -DFOO=1 -Iinclude/ input.c

# Cross-compile
ccc-arm   -O2 -o output-arm   input.c   # AArch64
ccc-riscv -O2 -o output-riscv input.c   # RISC-V 64
```

Unrecognized flags (architecture-specific `-m` flags, unknown `-f` flags) are silently ignored, so LCCC works as a drop-in in most build systems.

## Use as a Make `CC`

```bash
make CC=/path/to/lccc/target/release/ccc CFLAGS="$GCC_INC -O2"
```

## Run the Benchmark Suite

```bash
python3 lccc-improvements/benchmarks/bench.py --reps 5 --md results.md
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--reps N` | 5 | Repetitions per benchmark |
| `--bench ID` | all | Run only one benchmark (e.g. `01_arith_loop`) |
| `--md FILE` | — | Write Markdown report |
| `--json FILE` | — | Write JSON data |
| `--verbose` | off | Show compile errors |

## Run the Unit Tests

```bash
cargo test --lib   # 500 tests, ~0.04s
```

The full test suite runs optimizer passes, IR lowering, and register allocation on synthetic IR functions. Doctests in `if_convert.rs` contain a known Rust syntax quirk — use `--lib` to skip them.
