# Goal: SvmParameter builder with construction-time validation   (bead: libsvm-rs-zrt)

## 1. Objective
Add `SvmParameterBuilder`: a fluent, documented, additive way to construct a
validated `SvmParameter`. Ergonomics layer only — zero semantic change
(VISION.md point 5: idiomatic to hold, identical in behavior).

## 2. Acceptance Criteria
- [ ] New module `crates/libsvm/src/builder.rs` with `SvmParameterBuilder`:
      one method per parameter (svm_type, kernel_type, degree, gamma, coef0,
      c, nu, p, cache_size, eps, shrinking, probability, weight/weights),
      LIBSVM-matching names and docs stating each default.
- [ ] `build(self) -> Result<SvmParameter, SvmError>` constructs the struct
      and calls the EXISTING `SvmParameter::validate()`
      (crates/libsvm/src/types.rs:136) — do NOT duplicate validation rules.
      Docs state the split: data-dependent checks stay in `check_parameter`.
- [ ] Defaults identical to `SvmParameter::default()`: a no-method
      `build()` equals `SvmParameter::default()` (unit test asserts equality).
- [ ] Exported from lib.rs; doctest on the builder showing the README
      Quick Start configuration (CSvc, Rbf, gamma 1/13, c 1.0) via builder.
- [ ] Unit tests: happy path equality vs field assignment; at least 4
      invalid cases rejected by build() (e.g. negative gamma, eps <= 0,
      cache_size <= 0, negative degree) asserting SvmError is returned.
- [ ] No changes to SvmParameter fields, Default, validate(), training,
      solver, or I/O.

## 3. Verification
- Quick: `cargo test --locked -p libsvm-rs builder`
- Full:  `cargo test --locked --all-features`
         `cargo clippy --locked --all-targets --all-features -- -D warnings`
         `cargo fmt --all -- --check`

## 4. Scope
✅ ALWAYS:    crates/libsvm/src/builder.rs (new), crates/libsvm/src/lib.rs
              (module decl + re-export only), README.md (optional: one short
              builder example beside the existing Quick Start)
⚠️ ASK FIRST: any change to types.rs or error.rs
🚫 NEVER:     solver.rs, train.rs, io.rs, kernel.rs, probability.rs,
              cross_validation.rs, vendor/, reference/, .github/

## 5. Non-Goals / Constraints
- NOT this cycle: problem-dependent validation in the builder; deprecating
  field assignment; new dependencies.
- `weight` builder method appends `(label, weight)` pairs matching the
  existing `weight: Vec<(i32, f64)>` representation.

## 6. Context Pointers
- `br show libsvm-rs-zrt` — full background/reasoning/considerations.
- `VISION.md` — additive-API rule.
- `crates/libsvm/src/types.rs:130-200` — SvmParameter, Default, validate(),
  check_parameter (already implement the validation split; reuse, don't move).
- Module style reference: small focused modules like metrics.rs.
- Skills: rust.

## 7. Task Breakdown
1. Write builder struct + methods + docs → compiles.
2. build() + tests (equality, invalid cases, doctest) → quick gate green.
3. lib.rs export + optional README snippet → full gate green.

## 8. Stop Conditions
- DONE when all Acceptance Criteria pass and the full gate is green.
- STOP and report if validate() proves insufficient for a documented C
  svm_check_parameter rule (would need types.rs changes — Ask-first).
