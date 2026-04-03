//! Machine instruction IR with virtual registers for x86-64.
//!
//! This is a thin intermediate representation between the SSA IR and text
//! assembly. IR instructions are lowered to MachInst sequences that use
//! virtual registers (MachReg::Vreg). A register allocator maps virtual
//! registers to physical registers, inserting spill/reload as needed.
//! The allocated MachInst sequence is then emitted as AT&T assembly text.
//!
//! The MachInst enum only models patterns the codegen actually emits —
//! it is NOT a general x86 instruction set. The `Raw(String)` variant
//! provides an escape hatch for instruction types not yet migrated.

use crate::backend::regalloc::PhysReg;

/// A register operand: either a physical register (pre-colored or allocated)
/// or a virtual register waiting for allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MachReg {
    /// Pre-colored physical register (e.g., rax for division, rcx for shifts).
    /// The register allocator must not assign any vreg to this physreg at
    /// program points where it's pre-colored.
    Phys(PhysReg),
    /// Virtual register identified by IR Value ID. Will be replaced with
    /// Phys after allocation, or spilled to a stack slot.
    Vreg(u32),
}

/// Operand size for instruction encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpSize {
    S8,   // byte (al, bl, r12b, ...)
    S16,  // word (ax, bx, r12w, ...)
    S32,  // dword (eax, ebx, r12d, ...)
    S64,  // qword (rax, rbx, r12, ...)
}

/// ALU operation (two-address form: dst = dst OP src).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Imul,  // 2-operand signed multiply (dst *= src)
}

/// Shift operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftOp {
    Shl,
    Shr,   // logical shift right
    Sar,   // arithmetic shift right
}

/// Condition code for Jcc, SetCC, CMov.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    E,   // equal (ZF=1)
    Ne,  // not equal (ZF=0)
    L,   // less (signed) (SF≠OF)
    Le,  // less or equal (signed) (ZF=1 or SF≠OF)
    G,   // greater (signed) (ZF=0 and SF=OF)
    Ge,  // greater or equal (signed) (SF=OF)
    B,   // below (unsigned) (CF=1)
    Be,  // below or equal (unsigned) (CF=1 or ZF=1)
    A,   // above (unsigned) (CF=0 and ZF=0)
    Ae,  // above or equal (unsigned) (CF=0)
}

/// An operand in a machine instruction.
#[derive(Debug, Clone)]
pub enum MachOperand {
    /// Register operand.
    Reg(MachReg),
    /// Immediate constant.
    Imm(i64),
    /// Memory: offset(%base). Base is a register.
    Mem { base: MachReg, offset: i64 },
    /// Indexed memory: offset(%base, %index, scale). Scale is 1/2/4/8.
    MemIndex { base: MachReg, index: MachReg, scale: u8, offset: i64 },
    /// Stack slot (assigned after register allocation for spills).
    /// The i64 is the offset from the frame pointer (%rbp or %rsp-relative).
    StackSlot(i64),
    /// RIP-relative symbol reference: symbol(%rip).
    RipRel(String),
}

/// A machine instruction with virtual registers.
///
/// These model the x86-64 instruction forms that the codegen actually emits.
/// Not all x86 instructions are represented — only the patterns needed for
/// lowering the LCCC IR. The `Raw` variant provides an escape hatch.
#[derive(Debug, Clone)]
pub enum MachInst {
    // ── Data movement ────────────────────────────────────────────────
    /// mov src, dst (covers reg-reg, imm-reg, mem-reg, reg-mem, etc.)
    Mov { src: MachOperand, dst: MachOperand, size: OpSize },

    /// movzx src, dst (zero-extend: movzbl, movzwl, movzbq, movzwq)
    Movzx { src: MachReg, dst: MachReg, from_size: OpSize, to_size: OpSize },

    /// movsx src, dst (sign-extend: movsbq, movswq, movslq)
    Movsx { src: MachReg, dst: MachReg, from_size: OpSize, to_size: OpSize },

    // ── Arithmetic ───────────────────────────────────────────────────
    /// Two-address ALU: dst = dst OP src.
    /// x86 form: `{add,sub,and,or,xor,imul} src, dst`
    Alu { op: AluOp, src: MachOperand, dst: MachReg, size: OpSize },

    /// Three-operand multiply: dst = src * imm.
    /// x86 form: `imul $imm, src, dst`
    Imul3 { imm: i64, src: MachReg, dst: MachReg, size: OpSize },

    /// Unary ALU: dst = OP(dst). Covers neg, not.
    Neg { dst: MachReg, size: OpSize },
    Not { dst: MachReg, size: OpSize },

    /// Shift: dst = dst SHIFT amount.
    /// Amount must be Imm or Reg(Phys(RCX)) — x86 constraint.
    Shift { op: ShiftOp, amount: MachOperand, dst: MachReg, size: OpSize },

    /// LEA: dst = base + index*scale + offset (3-address add).
    /// x86 form: `leaq offset(%base, %index, scale), %dst`
    Lea { base: MachReg, index: Option<(MachReg, u8)>, offset: i64, dst: MachReg },

    // ── Division (implicit rax:rdx) ──────────────────────────────────
    /// Sign-extend rax → rdx:rax. x86: `cqto` (64-bit) or `cltd` (32-bit).
    Cqto { size: OpSize },

    /// Zero rdx for unsigned division. x86: `xorl %edx, %edx`.
    XorRdx,

    /// Integer division: rdx:rax / divisor.
    /// Quotient → rax, remainder → rdx.
    /// x86: `idivq divisor` (signed) or `divq divisor` (unsigned).
    Div { divisor: MachOperand, signed: bool, size: OpSize },

    // ── Comparison & flags ───────────────────────────────────────────
    /// Compare: sets flags based on (lhs - rhs).
    /// x86 form: `cmp rhs, lhs` (note: AT&T operand order is reversed).
    Cmp { lhs: MachOperand, rhs: MachOperand, size: OpSize },

    /// Test: sets flags based on (lhs & rhs).
    /// x86 form: `test rhs, lhs`.
    Test { lhs: MachOperand, rhs: MachOperand, size: OpSize },

    /// Set byte from condition code: `setCC %dst_8bit`.
    /// Always writes to the 8-bit sub-register.
    SetCC { cc: CondCode, dst: MachReg },

    /// Conditional move: dst = cc ? src : dst.
    /// x86 form: `cmovCC src, dst`.
    Cmov { cc: CondCode, src: MachOperand, dst: MachReg, size: OpSize },

    // ── Control flow ─────────────────────────────────────────────────
    /// Conditional jump: `jCC target`.
    Jcc { cc: CondCode, target: String },

    /// Unconditional jump: `jmp target`.
    Jmp { target: String },

    /// Block label (pseudo-instruction).
    Label(String),

    /// Direct function call: `call target`. Clobbers caller-saved regs.
    Call { target: String },

    /// Indirect function call: `call *%reg`. Clobbers caller-saved regs.
    CallIndirect { reg: MachReg },

    /// Return: `ret`.
    Ret,

    // ── Passthrough ──────────────────────────────────────────────────
    /// Raw assembly text for instructions not yet migrated to MachInst.
    /// The register allocator treats this as a barrier (invalidates all vregs).
    Raw(String),
}

// ── Well-known physical register IDs ─────────────────────────────────────
//
// PhysReg IDs match the existing codegen convention (emit.rs:79-88).
// IDs 0 and 7 are NEW — rax/rcx are scratch in the existing accumulator
// model but need pre-coloring in the MachInst model for division/shifts.
// The MachInst emitter (machinst_emit.rs) handles all IDs including 0/7.

/// rax — accumulator, division result, function return value.
/// NOT in the existing allocatable pool (used as scratch).
pub const RAX: PhysReg = PhysReg(0);
/// rcx — shift count (%cl), 4th SysV argument register.
/// NOT in the existing allocatable pool (used as scratch).
pub const RCX: PhysReg = PhysReg(7);
/// rdx — division high half, 3rd SysV argument register.
pub const RDX: PhysReg = PhysReg(16);

// Callee-saved registers (available for allocation, preserved across calls):
pub const RBX: PhysReg = PhysReg(1);
pub const R12: PhysReg = PhysReg(2);
pub const R13: PhysReg = PhysReg(3);
pub const R14: PhysReg = PhysReg(4);
pub const R15: PhysReg = PhysReg(5);
pub const RBP: PhysReg = PhysReg(6);

// Caller-saved registers (destroyed by calls, available between calls):
pub const R11: PhysReg = PhysReg(10);
pub const R10: PhysReg = PhysReg(11);
pub const R8:  PhysReg = PhysReg(12);
pub const R9:  PhysReg = PhysReg(13);
pub const RDI: PhysReg = PhysReg(14);
pub const RSI: PhysReg = PhysReg(15);

/// All GPR registers that the MachInst allocator can assign to virtual registers.
/// Excludes rax (0) and rcx (7) which are reserved as scratch.
pub const MACHINST_ALLOCATABLE_GPRS: &[PhysReg] = &[
    // Callee-saved (survive calls):
    RBX, R12, R13, R14, R15, RBP,
    // Caller-saved (destroyed by calls, only for non-call-spanning values):
    R11, R10, R8, R9, RDI, RSI, RDX,
];

impl MachReg {
    /// Returns true if this is a virtual register (needs allocation).
    pub fn is_vreg(&self) -> bool {
        matches!(self, MachReg::Vreg(_))
    }

    /// Returns true if this is a pre-colored physical register.
    pub fn is_phys(&self) -> bool {
        matches!(self, MachReg::Phys(_))
    }

    /// Get the virtual register ID, if this is a vreg.
    pub fn vreg_id(&self) -> Option<u32> {
        match self {
            MachReg::Vreg(id) => Some(*id),
            _ => None,
        }
    }

    /// Get the physical register, if this is pre-colored.
    pub fn phys(&self) -> Option<PhysReg> {
        match self {
            MachReg::Phys(r) => Some(*r),
            _ => None,
        }
    }
}

impl OpSize {
    /// Convert an IrType to an OpSize.
    pub fn from_ir_type(ty: crate::common::types::IrType) -> Self {
        use crate::common::types::IrType;
        match ty {
            IrType::I8 | IrType::U8 => OpSize::S8,
            IrType::I16 | IrType::U16 => OpSize::S16,
            IrType::I32 | IrType::U32 | IrType::F32 => OpSize::S32,
            _ => OpSize::S64, // I64, U64, F64, Ptr, etc.
        }
    }

    /// AT&T instruction suffix for this size.
    pub fn suffix(&self) -> &'static str {
        match self {
            OpSize::S8 => "b",
            OpSize::S16 => "w",
            OpSize::S32 => "l",
            OpSize::S64 => "q",
        }
    }
}
