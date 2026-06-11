# Goal: libsvm-rs-cww — Fresh security audit; rewrite SECURITY_AUDIT.md

Supervisor-driven cycle (user requested Fable 5 authorship of the audit report).
Method mirrors the 2026-06-02 audit: 4 parallel read-only reviewer lanes +
coordinator (supervisor) reproduction of every finding before it enters the report.

## Reviewer lanes (read-only)
1. serde surface: derive/impls on SvmModel/SvmParameter vs text-parser invariants
   (validate_model_header parity, caps bypass via serde_json::from_str).
2. rayon paths: CV folds + probability fold trainings — panic safety, determinism,
   shared state.
3. Regression: F1–F6 fixes intact; SvmParameterBuilder validation parity with
   svm_check_parameter.
4. Supply chain: deny.toml, workflow SHA pinning, toolchain pin, fuzz coverage of
   serde surface.

## Acceptance criteria
- SECURITY_AUDIT.md rewritten: current date + audited commit on main, covers
  serde/rayon/builder/LCG surfaces, findings marked [Verified]/[Reviewer].
- Actionable findings fixed-with-regression-test here or filed as bug beads
  blocking libsvm-rs-2wg.
- ≥1 adversarial serde input test passing by REJECTION (not panic).
- `cargo test --workspace --all-features` green.

## Verification
`cargo test --workspace --all-features 2>&1 | tail -5` + audit diff read.

## Scope
✅ SECURITY_AUDIT.md; if fixes needed: crates/libsvm/src/{io.rs,probability.rs,
cross_validation.rs,train.rs,parameter,serde sites}, fuzz/, tests/.
🚫 tolerance files, reference/ artifacts, CI workflows.
