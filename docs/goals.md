# Goal: Replicate glibc rand() for probability CV shuffle on Linux   (bead: libsvm-rs-4r5)

## 1. Objective
On Linux, `c_rand()` in probability.rs falls back to a generic LCG that does NOT
match glibc rand(), so the internal probability-CV fold shuffle diverges from the
C reference → 23 differential fails on this box (139/78/23/10 vs the Mac baseline
236/4/0/10). Implement glibc rand() semantics under `cfg(target_os = "linux")`,
restore 0 fails, and make the parity claim's platform scope explicit in docs.
Full background + algorithm spec: `br show libsvm-rs-4r5`.

## 2. Acceptance Criteria
- [ ] Linux `c_rand()` implements glibc's TYPE_3 additive-feedback generator
      (31-entry table, r[i] = r[i-3] + r[i-31] mod 2^32, output (r[i] >> 1) &
      0x7fffffff; documented seeding: LCG expansion of seed 1, discard first 310
      outputs). Reference: glibc stdlib/random_r.c.
- [ ] Hermetic unit test: first ≥20 outputs match hardcoded expected constants.
      Derive the constants ONCE via a tiny throwaway C program on this box
      (gcc available); commit only the constants, not the C program.
- [ ] macOS path byte-identical to current code; non-mac/non-linux keeps the
      existing LCG fallback with a doc comment scoping the parity claim.
- [ ] Check cross_validation.rs / util.rs `shuffle_range`: if public k-fold CV
      consumes rand() in C (vendor/libsvm/svm.cpp ~1367-1460), apply the same
      source of randomness there; if not, note why in the report.
- [ ] `DIFF_SCOPE=full python3 scripts/run_differential_suite.py` on this box:
      0 fails (record actual counts in the CODER REPORT).
- [ ] README parity section + reference/tolerance_policy.md: one sentence each
      stating the differential baseline is per-platform (rand() replication
      exists for macOS and Linux; baselines recorded per platform).
- [ ] `cargo test --locked --all-features` green; clippy `-D warnings`; fmt clean.

## 3. Verification
- Quick: `cargo test --locked -p libsvm-rs c_rand` and
  `cargo test --locked -p libsvm-rs probability`
- Full (logged, run once):
  `DIFF_SCOPE=full python3 scripts/run_differential_suite.py 2>&1 | tee .sc/4r5-differential.log | tail -3`

## 4. Scope
✅ ALWAYS:    crates/libsvm/src/probability.rs, crates/libsvm/src/cross_validation.rs,
              crates/libsvm/src/util.rs (shuffle only), README.md (one sentence),
              reference/tolerance_policy.md (one sentence)
⚠️ ASK FIRST: any change that alters macOS-path behavior; solver.rs/train.rs
🚫 NEVER:     vendor/, scripts/, .github/; do NOT regenerate or modify committed
              reference/ artifacts (differential_report.md, results.json,
              benchmark_report.md) — if the suite rewrites them, `git checkout`
              those files back and report counts from the log only (supervisor
              decides re-baselining at review).

## 5. Non-Goals / Constraints
- No perf work (that is libsvm-rs-jeh, gated on this bead). No algorithm changes
  beyond the RNG. Identical call order to C at every shuffle site.

## 6. Context Pointers
- `br show libsvm-rs-4r5` — full spec, evidence, considerations.
- `VISION.md` — parity is the product; trust requires honest platform scope.
- C call sites: vendor/libsvm/svm.cpp svm_binary_svc_probability,
  svm_svr_probability (~1130-1250), svm_cross_validation (~1367-1460).

## 7. Stop Conditions
- DONE when criteria pass. STOP and report if differential fails do not reach 0
  after the RNG matches the unit-test constants, or an ⚠️ item is required.
