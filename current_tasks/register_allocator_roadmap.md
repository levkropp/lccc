# AArch64 codegen quality: findings + register-allocator roadmap

Session outcome: geomean vs GCC -O2 improved 0.61x -> ~0.56x with 15 verified
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
- Call-spanning F64 values allocate to the callee-saved FP pool (d8-d14)
  instead of never being allocated (restricted-pool support in the scan).
- GlobalAddr CSE (substitution-based pass): one SSA value per symbol per
  dominance scope instead of one per access (fannkuch -3-5%).
- fmsub/fnmsub fusion of float Mul;Sub, gated off loop-carried accumulators
  (fusing acc-=a*b lengthens the serial chain: ungated nbody +13%).
- Aggregate memcpy-temp forwarding hoist: alloca/global terminals end the
  def chain instead of rejecting it (struct_copy temps forward into place).
- Full unroll of small constant-trip loops enabled by default (trip<=16,
  <=512 expanded insts) — a win now that the forwarding chain works
  (struct_copy 1.80x -> 1.31x).
- FP spill-slot store->load forwarding peephole (str dS,[slot] / ldr dD,[slot]
  -> fmov), tracking sp-derived base registers (struct_copy -2.7%).
- Dead-store elimination escape analysis: sp-derived bases that only feed
  address uses no longer bail the pass; dead FP/base-form stores removed.
- Conditional-sum vectorization (sum_positive): late Select-clamped reduction
  pass, NEON smax-against-zero + sadalp, 4-wide i32->i64.
- Max-reduction vectorization (find_max shape `mx = max(mx, arr[i])`, IV init
  c=1): late pass; broadcast scalar init (`dup`), lane-wise `smax`, `smaxv`
  horizontal reduce. Legality requires the marching pointer's preheader GEP
  offset == c*elem; the remainder start/limit are shifted by c so coverage is
  exactly [c, n) (loop_patterns -7.2% interleaved A/B).

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
- CCC_LOOP_PIN cap sweep on fannkuch: 0->4066, 1->3715, 2->3458, 4->3474,
  6->3417 ms. The default cap of 2 is the sweet spot; more steals evict
  warm holders. Do not raise.
- GVN-style (Copy-inserting) CSE for pointer-valued expressions: hits the
  same stale-base-register landmine that disabled GEP CSE. GlobalAddr CSE
  must be substitution-based (see global_addr_cse.rs).
- fmsub/fnmsub fusion of `a - b*c` on accumulator update chains lengthens
  the serial dependency (fmsub latency > fsub): nbody +13%. Shipped GATED
  (e3b21b8f): accumulator-feeding Subs stay split (mandelbrot -1.4%).
- Read-only loop-invariant F64 load hoisting (nbody bodies[i].x/y/z,
  struct_copy particle fields): needs phi-derived-pointer alias proof and
  pool-aware gating; the 60% affine-LICM regression came from hoisted values
  outbidding accumulators for FP registers. Do not retry without a
  register-pressure-aware gate.
- Widening the const-GEP fold to slot-homed (Indirect) bases: MISCOMPILE.
  The base's live interval ends at the GEP (its last IR use); the folded
  load reads the base's slot one step later, after Tier-2 slot packing may
  have reused it. The same interval-edge hazard applies to register reuse
  after a value's last IR use — remember it for the allocator rewrite.
- GPR pool extension via x9-x15: the emitter's scratch discipline
  (address staging, memcpy, call staging, intrinsics) is load-bearing;
  x16/x17 are also used as scratch in 24+ sites. Not a bounded change.
- Trampoline branch inversion (b.cc exit / skip-trampoline / latch):
  spectral_norm +5% consistently (hot path became the taken branch),
  nbody -1.7%. Net negative; dropped.
- Second-chance FP scan for loop-carried copy webs (mandebrot zr/zi web):
  web members' intervals coexist, so each takes its OWN register and the
  web copies persist as fmovs while pool pressure rises — mandelbrot +11%.
  The web needs coalescing (one register per logical accumulator), not
  more registers. Do not retry without true web coalescing.

## Latent backend hazards found (fixed)

- peephole propagate_address_aliases deleted the defining `mov xD, x0` of an
  address alias based on a same-block window, ignoring cross-block uses
  (segfault when GlobalAddr CSE made aliases multi-use). Fixed with a
  full-function scan that only deletes on proven death of the alias.
- tail_call_elim::replace_values_in_inst missed Intrinsic::dest_ptr and
  InlineAsm::outputs — dangling uses after substitution passes. Fixed in the
  shared helper.
- Lesson: making any pointer-valued SSA value multi-use exercises backend
  fast paths (accumulator reg_cache, text peepholes, fold analyses) that were
  written assuming single-use/immediate consumption. Verify with the
  progressive suite (hash_table_mini catches this class).

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
