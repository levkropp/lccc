# AArch64 codegen quality: findings + register-allocator roadmap

Session outcome: geomean vs GCC -O2 improved 0.61x -> 0.581x with 9 verified
wins (all 18/18 byte-identical, suites green). The remaining gap to GCC is
architectural, not pass-level. This doc records the evidence and the plan.

## Verified wins this session (all committed)

- Direct FP stack reloads: `ldr dN, [slot]` instead of `ldr x0` + `fmov dN,x0`.
- FP loop-phi coalescing (relaxed cross-block filter): backedge fmov removed.
- Plain-copy loop vectorization (dst[i]=src[i]) via the map pattern.
- Late redundant-load dedup keyed on alias linear-forms (nbody mass loads).
- Register-direct Select emission (no x0/x1/x2 staging) + redundant sxtb elim.
- Opened unused loop-promotion FP regs (d27-d31) to the general FP scan.
- Quadratic strength reduction (triangular indices) for spectral_norm.

## Reverted / documented dead-ends (measured, do not retry)

- Post-indexed addressing (ldr xN,[xM],#4): neutral-to-slightly-negative on
  Apple Silicon (writeback uop offsets the saved add).
- ldp/stp pairing of adjacent stack slots: fires too rarely to matter.
- Same-block overwritten-store DSE: the address-of-sp guard disables it where
  it would help (functions with stack arrays).
- GVN alias-selective invalidation: geomean-neutral, possible regressions.
- Affine LICM load hoisting: 60% nbody REGRESSION (FP pool exhaustion).
- IV widening i32->i64: new i64 values get scratch+spill codegen, worse.
- GEP phi coalescing: regressed nbody +17%.
- Extended register steal to non-phi hot values: MISCOMPILE (strlen_bench).
- d15/v15 pool opening: regressed nbody +4%.
- Full unroll for struct_copy: 30ms vs 18.6ms baseline (body bloat).

## Root cause of the remaining gap (fannkuch 1.95x, struct_copy 1.67x, nbody 1.35x)

The backend's slot-home discipline: every SSA value has a mandatory stack
slot; register assignment is a steal on top. Per-iteration costs that persist:
- Loop-carried values that lose the allocation get str/ldr per iteration.
- Inner-loop temporaries in nested loops (fannkuch) spill because ~15
  outer-loop values pin all 17 GPRs across the whole body.
- The linear scan NEVER evicts (a past eviction miscompiled loop accumulators);
  the post-scan steal is capped and phi-dest-only.

## The roadmap to 0.55x (register-allocator rewrite)

Replace slot-home with on-demand spilling. Concretely, in priority order:
1. Values that are register-resident for their whole live range get NO slot
   and no slot stores (needs whole-function slot-read analysis, not the
   text-level global_dead_store_elimination which bails on address-of-sp).
2. Eviction in the linear scan that is slot-discipline-safe: when a register
   is reassigned, ensure the evicted value's slot is written on every def
   (the strlen miscompile came from missing this).
3. Live-range splitting around inner loops so outer-loop invariants don't pin
   registers through hot inner loops (160472ec was loop-transparent splitting;
   the need is loop-NESTED splitting).
4. SROA for non-escaping struct arrays (struct_copy): split allocas into
   per-field scalars; handles the marching-pointer case without full unroll.
5. SLP pairing of adjacent F64 struct fields (nbody/struct_copy dx,dy).

Verification at every step: 536 unit tests, correctness suite 45/50,
progressive 19/22, 18/18 benchmarks byte-identical, CCC_VERIFY_REGALLOC=1.
