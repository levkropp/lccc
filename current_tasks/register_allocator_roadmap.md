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
- Overwritten-store peephole (eliminate_overwritten_stores): local,
  block-scoped analog of GDSE for `str R,[sp,#off]` overwritten by a later
  store with no intervening read. Catches what GDSE misses when it bails on
  a whole function over one escaped frame base (strlen_bench itoa loop:
  3 stores -> 1 per iteration).
- Loop-carried store sinking (sink_loop_carried_stores): a `str R,[sp,#off]`
  in a bottom-tested loop whose slot is never referenced elsewhere in the
  loop, with no mid-loop exits, no interior labels, and R surviving to the
  backedge, moves to the fall-through exit (stored once per loop instead of
  once per iteration). Fires on sieve's marking loop; most other candidates
  across the suite fail the strict conditions (mid-loop exits, slot reloads,
  or interior labels) — extending it needs dedicated-exit-label analysis.

## Latent bug fixed this session (was committed as a0791ef9)

- a0791ef9's direct call-arg staging (skip x9-x16 temps when no int arg
  source is in x0-x7) loaded int args into x0-x7 FIRST, then emitted FP args
  — but emit_call_fp_reg_args routes every FP arg through the x0 scratch
  (`emit_load_arg_to_reg(arg, "x0", ...)` + `fmov dN, x0`). The FP material-
  ization clobbered the staged format pointer: float_cast/union_type_punning/
  float_special/varargs_mixed_types segfaulted or miscompiled (correctness
  44/50 on the real binary). The a0791ef9 validation had run against a STALE
  binary (virtiofs mtime skip) — lesson: touch changed files and confirm the
  binary hash changes before trusting suite results. Fix: emit FP args before
  int-arg staging in the fast path (x0 still free), matching the slow path's
  ordering guarantee. Correctness back to 48/50.


## Reverted / documented dead-ends (measured, do not retry)

- Skip-trampoline straightening (b.cc TA / .Lskip: b TB layouts in loops):
  both condition inversion (b.inv TB; b TA) and hoisting the single-
  predecessor target block into the fall-through slot are noise-neutral on
  the suite — A/B control runs (same binary vs itself) swing +/-2.5% on this
  VM, and neither variant beat that band on mandelbrot/fannkuch/
  loop_patterns/strlen_bench. The trampoline jump is effectively free on
  Apple Silicon. Reverted; code deleted.
- GPR reverse phi coalescing (give an unassigned phi dest the backedge
  source's register, or allocate both into a proven-free register post-scan):
  implemented both ways; fires NOWHERE in the suite — hot phi dests already
  get registers via the loop-pin steal, cold ones don't matter. Removed.
  Keeping the sources eligible instead (so they get scanned) cost hash_table
  a reproducible +3% (more scan pressure on x19-x28).
- Copy-propagating staging movs into cmp/cbz/str first operands (plus
  treating non-sp ldr as a dest-write in eliminate_overwritten_moves):
  folds correctly (cbz x19 directly, dead staging movs removed) but
  hash_table +2.2% reproducibly, binary_trees +0.7%, everything else
  noise-neutral. Net negative; reverted. Lesson: on Apple Silicon,
  register movs are rename-eliminated (free), and instruction-count
  peepholes that do not remove loads, stores, branches, or chain latency
  cannot win — they only disturb layout.
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
  warm holders. Do not raise. (The k>2 miscompile behind this was later
  root-caused to the indexed-load invalidate_acc bug below and fixed;
  the priority-first scan then made the steal largely moot.)
- GVN-style (Copy-inserting) CSE for pointer-valued expressions: hits the
  same stale-base-register landmine that disabled GEP CSE. GlobalAddr CSE
  must be substitution-based (see global_addr_cse.rs).
- fmsub/fnmsub fusion of `a - b*c` on accumulator update chains lengthens
  the serial dependency (fmsub latency > fsub): nbody +13%. Shipped GATED
  (e3b21b8f): accumulator-feeding Subs stay split (mandelbrot -1.4%).
- 32-bit move propagation in propagate_register_copies: arith_loop -4.5%
  but qsort +4.5% / hash_table +3.4% (rewriting uses onto the source
  register extends its live range in tight loops). Net negative; reverted.
- F64 values in split_call_spanning_ranges (to free the d8-d14 restricted
  pool from call-spanning constants): segfaults nbody/struct_copy/
  spectral_norm — the split's store/reload + phi insertion is not F64-safe.
- Bidirectional real_use propagation for FP copy webs (even restricted to
  multi-def/loop-carried dests): makes loop-accumulator webs FP-register
  candidates but crashes matmul/nbody/struct_copy — the phi-coalesce
  inheritance path is not sound for these webs. The mandelbrot per-iteration
  accumulator store/fmov remains UNFIXED; it needs either point-level
  coalesce-conflict validation or the whole-function slot-read analysis
  (roadmap item 1) so the post-loop read uses the register directly.
- Exact per-register occupancy lists in the linear scan (replacing the
  free_until approximation so priority-ordered scans reuse registers across
  disjoint windows): initially HUNG fannkuch. Root cause found: the indexed
  addressing fold reads the pre-scale index of a Shl/Mul-peeled GEP offset
  (`[base, index, lsl #k]`), but extend_gep_base_liveness only extended the
  offset value's liveness — the peeled index died at the Shl and its
  register was reused for the load's own destination. Fixed by extending
  liveness through the peel chain (Shl/Mul/widening-Cast) to the Load/Store
  points. Shipped: occupancy checks + priority-first GPR scan.
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
- aggregate_copy_forward::eliminate_dead_aggregate_field_stores collapsed
  unknown pointer offsets (variable GEP, multi-suffix phi) to suffix 0, so a
  loop reading `long a[N] = {...}` via a marching pointer "covered" only field
  0 — the 12 remaining init stores were deleted (fib_rec2iter CI failure,
  correctness 45→48, progressive 19→21). Fixed with an absorbing
  SUFFIX_UNKNOWN plus periodic coverage analysis: a marching-pointer phi
  (start s0, constant period p) records loads as the residue classes
  {s0 + p*j + r}, so stores at unread residues (struct_copy's id/name fields)
  stay eliminable while same-residue init stores are kept.
- ARM global_dead_store_elimination resolved loads through sp-derived base
  registers with a LINEAR text scan: a loop-variant base (marching pointer)
  contributed only its iteration-0 offset, so init stores covered only by
  later iterations were deleted. Fixed by marking registers written inside
  backward-branch regions loop-variant: loads through them count as
  whole-frame reads and stores through them are never deleted.
- Lesson: making any pointer-valued SSA value multi-use exercises backend
  fast paths (accumulator reg_cache, text peepholes, fold analyses) that were
  written assuming single-use/immediate consumption. Verify with the
  progressive suite (hash_table_mini catches this class).
- Lesson: any analysis that maps a pointer register to ONE frame offset is
  unsound across loop back-edges unless it proves the register loop-invariant.
- emit_load_indexed_impl (unassigned int dest) ran store_x0_to(dest) and then
  invalidate_acc(), wiping the only record of a slot-less value; the next read
  silently materialized `mov x0, #0` (fannkuch miscompile at CCC_LOOP_PIN>2).
- Per-segment decisions where assignment is per-value: Phase-1 scanned
  multi-segment values segment-by-segment (last write wins the assignments
  map) and Phase-2 filtered call-spanning per segment — both now merge to
  whole-value spans. Phi-coalesce inheritance checked only the backedge
  source's FIRST segment for conflicts; now checks all segments.
- reuse_stack_loads_within_blocks NOPed a stack reload whose register copy
  propagation had made the sole reader (add x0,x1,x2 -> add x0,x1,x0) —
  deleting the load left the add reading a stale x0 (strlen_bench segfault
  under priority-first scan). Now requires the load's register to be
  provably dead (no reads until its next full overwrite) before NOPing.
- Lesson: the whole-lifetime assignment model means ANY per-segment or
  per-site reasoning (span checks, conflict checks, slot packing, text
  peepholes) must be validated against the value's full segment set.

## Root cause of the remaining gap (fannkuch 1.17x after priority scan)

The backend's slot-home discipline: every SSA value has a mandatory stack
slot; register assignment is a steal on top. Per-iteration costs that persist:
- Loop-carried values that lose the allocation get str/ldr per iteration.
- ~~Inner-loop temporaries in nested loops (fannkuch) spill because ~15
  outer-loop values pin all 17 GPRs across the whole body~~ ADDRESSED by the
  priority-first Phase-1 scan (hottest ranges claim registers first; the
  free-until discipline is order-independent so cold ranges just spill).
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
