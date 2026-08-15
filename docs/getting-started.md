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
| **POSIX host** | Compiler emits Linux ELF; developed on Linux and macOS |
| **GCC installed** | Needed for built-in headers (`stddef.h`, `stdarg.h`) |
| **Python 3.9+** | For the test/benchmark runners (optional) |

LCCC uses a completely standalone assembler and linker — no external toolchain is needed at
compile time, only for the GCC built-in headers at C preprocessing time.

## Clone and Build

```bash
git clone https://github.com/levkropp/lccc.git
cd lccc
cargo build --release
```

This produces five binaries in `target/release/`:

| Binary | Target architecture |
|--------|---------------------|
| `lccc` | picked by host / argv[0] name |
| `lccc-x86` | x86-64 |
| `lccc-arm` | AArch64 |
| `lccc-riscv` | RISC-V 64 |
| `lccc-i686` | i686 (32-bit x86) |

> **Note:** The compiler also picks its target from the argv[0] name — symlinking the binary
> to `aarch64-linux-gnu-cc` (or similar) selects that target, which is how the benchmark suite
> drives cross-compilation.

## Compile Your First Program

```bash
# LCCC needs GCC's built-in headers for stddef.h, stdarg.h, etc.
GCC_INC="-I$(gcc -print-file-name=include)"

cat > hello.c <<'EOF'
#include <stdio.h>
int main(void) {
    printf("Hello from LCCC!\n");
    return 0;
}
EOF

./target/release/lccc $GCC_INC -O2 -o hello hello.c
./hello
# Hello from LCCC!
```

## GCC-Compatible Flags

LCCC accepts the standard GCC command-line interface:

```bash
# Compile and link
lccc -O2 -o output input.c

# Emit assembly
lccc -S -O2 input.c

# Compile to object file only
lccc -c input.c

# Preprocessor only
lccc -E input.c

# Debug info
lccc -g -O2 -o output input.c

# Macros and include paths
lccc -DFOO=1 -Iinclude/ input.c

# Cross-compile
lccc-arm   -O2 -o output-arm   input.c   # AArch64
lccc-riscv -O2 -o output-riscv input.c   # RISC-V 64
```

Unrecognized flags (architecture-specific `-m` flags, unknown `-f` flags) are silently
ignored, so LCCC works as a drop-in in most build systems.

## Use as a Make `CC`

```bash
make CC=/path/to/lccc/target/release/lccc CFLAGS="$GCC_INC -O2"
```

## Run the Test Suites

```bash
# Unit tests (536 pass)
cargo test --release

# 18-benchmark suite (performance + correctness + size)
python3 tests/benchmark/run_benchmarks.py --reps 5

# 50-program correctness suite (vs GCC output)
python3 tests/correctness/run_correctness.py

# Progressive feature tests (levels 1–8)
python3 tests/integration/test_progressive.py
```

The unit suite runs optimizer passes, IR lowering, register allocation, and assemblers on
synthetic inputs. The Python suites compile and execute real C programs and compare output
against GCC.
