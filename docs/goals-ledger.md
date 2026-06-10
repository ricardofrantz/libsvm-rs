## libsvm-rs-966 — Migration guide (2026-06-10)
- AC1 six sections: PASS (docs/MIGRATION.md, 157 lines)
- AC2 all CLI flags in table: PASS (grep sweep clean, -wi included)
- AC3 API rows verified vs svm.h + lib.rs/types.rs: PASS (supervisor re-grepped
  check_parameter, svm_get_*, accessors, LoadOptions paths — all exist;
  check_parameter reachable at root via `pub use types::*`)
- AC4 README link: PASS (README.md:213)
- AC5 parity wording: PASS (numerical equivalence framing, "not bitwise")
- Follow-ups: none
## libsvm-rs-zrt — SvmParameter builder (2026-06-10)
- AC1 builder module, one method/param, docs w/ defaults: PASS (builder.rs, 243 lines)
- AC2 build() delegates to existing validate(), no duplicated rules: PASS (builder.rs:141-144)
- AC3 no-method build == default (test): PASS
- AC4 lib.rs export + Quick-Start doctest: PASS (module doc example)
- AC5 unit tests, 4+ invalid cases: PASS (7 tests; gamma/eps/cache_size/degree)
- AC6 no semantic changes: PASS (diff touches builder.rs + 2-line lib.rs only)
- Gates re-run by supervisor: 10/10 test targets ok, clippy -D warnings clean, fmt clean
- Follow-ups: none (README builder example was optional, skipped — doctest covers it)
## libsvm-rs-jeh — SVR probability perf gap (2026-06-10) — BLOCKED
- Coder stopped correctly on two stop conditions; no code changes made.
- Profile (callgrind; perf blocked by perf_event_paranoid=4): 42.6% of
  instructions in Kernel::evaluate — hotspot is kernel.rs (Ask-first scope).
  Evidence: reference/perf_svr_probability_notes.md (uncommitted).
- Differential gate: brief omitted DIFF_SCOPE=full (default=quick, 45 cfgs) —
  supervisor brief error. Full-scope re-run on THIS box: 139/78/23/10 vs
  committed Mac baseline 236/4/0/10, unmodified tree.
- ROOT CAUSE (supervisor triage): probability.rs c_rand() replicates BSD
  rand() on macOS but uses a non-glibc LCG fallback elsewhere → on Linux the
  internal probability-CV fold shuffle differs from the C reference's glibc
  rand() → different Platt fits → 23 probability label-flip fails. Parity
  baseline is platform-scoped (Mac/clang) — README claim needs that caveat.
- Follow-up: parity-fix bead to be filed (glibc rand replication on Linux +
  re-baseline + document platform scope). jeh parked pending that gate.
