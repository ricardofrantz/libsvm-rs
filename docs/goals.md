# Goal: Parallelize binary-SVC probability internal fold trainings   (bead: libsvm-rs-aud)

## 1. Objective
Finish the rayon story from eo9. SUPERVISOR SCOPE NARROWING vs the bead text:
`svm_svr_probability` already calls `svm_cross_validation` with
`probability=false`, so it already takes eo9's parallel branch — DO NOT touch
it. The only remaining serial CV is `svm_binary_svc_probability`
(probability.rs:218): Fisher-Yates shuffle (serial, c_rand) then 5 folds, each
training rand-free (`subparam.probability = false`) and writing to disjoint
`dec_values[perm[j]]` indices. Parallelize that fold loop under
`cfg(feature = "rayon")`. Full bead: `br show libsvm-rs-aud`.

## 2. Acceptance Criteria
- [ ] Shuffle stays serial, before any parallelism. Under rayon, the 5 fold
      bodies run via par_iter; each fold returns its (begin..end) predictions
      and they are scattered to dec_values afterward (or disjoint-slice
      pattern in permuted order, as eo9 did in cross_validation.rs). The
      degenerate one-class-count branches (0.0 / 1.0 / -1.0 fills) are part of
      the fold body and must move with it.
- [ ] Fold-internal training output suppressed under rayon via the existing
      `with_suppressed_info` (lib.rs) — same rationale as eo9.
- [ ] Serial path (no feature) byte-identical to today — same code or pure
      refactor extraction, mirroring eo9's evaluate_fold pattern.
- [ ] BITWISE REGRESSION GATE: the four pinned digests in
      tests/rayon_parity.rs must NOT change — run that test with and without
      `--features rayon`; both green proves parallel == serial bitwise.
      Do not regenerate or edit the pinned digest constants.
- [ ] lib.rs rayon feature doc: one sentence extension (probability CV folds
      also parallel). No README change needed.
- [ ] Informational: wall-clock before/after for probability CV on heart_scale
      (e.g. time the rayon_parity probability cases or a small bench command);
      report numbers in the CODER REPORT.
- [ ] Gates: bash .sc/aud.gate.sh green.

## 3. Verification
- Quick: `cargo test --locked --features rayon --test rayon_parity`
          and `cargo test --locked --test rayon_parity`
- Full: `bash .sc/aud.gate.sh` (fmt, clippy -D warnings, rayon/all/no-default
  test matrix, rayon-absent-from-default-tree, quick differential suite —
  must stay 45/0/0/0; restore reference/ + data/generated/ if the suite
  rewrites them: `git checkout -- reference/ data/generated/`).

## 4. Scope
✅ ALWAYS:    crates/libsvm/src/probability.rs (svm_binary_svc_probability fold
              loop only), crates/libsvm/src/lib.rs (one doc sentence)
⚠️ ASK FIRST: any change to shuffle logic, sigmoid_train, predict_values
              call shape, or tests/rayon_parity.rs beyond running it
🚫 NEVER:     svm_svr_probability, svm_one_class_probability, solver/kernel/
              qmatrix/train internals, cross_validation.rs, util.rs, vendor/,
              scripts/, reference/, data/, Cargo.toml, ci.yml, README.md

## 5. Non-Goals / Constraints
- No new tests required — rayon_parity's pinned digests are the regression net.
- No parallelism anywhere else; nested rayon (outer serial, inner parallel) is
  the intended shape when called from cross_validation's probability branch.

## 6. Context Pointers
- eo9 diff in cross_validation.rs — the evaluate_fold + disjoint-slice pattern.
- `br show libsvm-rs-aud`, lib.rs `with_suppressed_info`.

## 7. Stop Conditions
- DONE when criteria pass. STOP if any pinned digest changes (that means a
  determinism break — do not re-pin) or an ⚠️ item is needed.
