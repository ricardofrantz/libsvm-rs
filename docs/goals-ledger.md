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
