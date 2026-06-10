# Goal: Wire the rayon feature — parallel cross-validation folds   (bead: libsvm-rs-eo9)

## 1. Objective
Cargo.toml declares an optional `rayon` dep but zero code uses it. Under
`cfg(feature = "rayon")`, parallelize the fold loop in `svm_cross_validation`
(cross_validation.rs) and the internal 5-fold CV in SVR probability training
(probability.rs). Serial path stays the default and byte-identical to today.
Full spec: `br show libsvm-rs-eo9`.

## 2. Acceptance Criteria
- [ ] Fold ASSIGNMENT (shuffle/PRNG) stays on the serial path, computed before
      any parallelism — fold membership identical with feature on/off. Folds
      write to disjoint slices (indexed collection / par_chunks; no locks).
- [ ] Bitwise-equality test (new tests/rayon_parity.rs or similar): CV results
      identical via f64::to_bits, feature on vs off — heart_scale and
      housing_scale; classification + SVR + probability. NOTE: the on/off halves
      cannot run in one cargo invocation; pin expected to_bits snapshots that
      both `--features rayon` and default test runs assert against, OR have the
      gate script run the test under both feature sets and diff outputs.
- [ ] No rayon in default tree: `cargo tree --no-default-features -e normal
      --prefix none | grep -c rayon` == 0 (note `-e normal` — dev-deps don't count).
- [ ] Print/quiet decision: parallel folds must not interleave per-fold training
      output — suppress or buffer fold-internal prints under the rayon path and
      document the choice in lib.rs docs. No interleaved garbage.
- [ ] Memory note documented (README/lib.rs): k parallel folds = up to
      min(k, threads) kernel caches of cache_size each; never silently divide it.
- [ ] Docs: Cargo.toml [features] comment, lib.rs feature docs, README one
      paragraph (opt-in, preserves zero-default-deps story).
- [ ] CI: minimal addition to .github/workflows/ci.yml —
      `cargo test --locked --features rayon` in the existing test job.
- [ ] Informational: measured wall-clock speedup on 5-fold CV on this box
      (any >1.5× on 4+ cores fine); record numbers in the CODER REPORT.
- [ ] Gates: bash .sc/eo9.gate.sh green.

## 3. Verification
- Quick: `cargo test --locked --features rayon -p libsvm-rs cross_validation`
- Full: `bash .sc/eo9.gate.sh` (fmt, clippy all-features -D warnings, test
  matrix incl. no-default-features, dep-tree check, differential suite —
  serial default path, counts must stay 240/0/0/10).

## 4. Scope
✅ ALWAYS:    crates/libsvm/src/cross_validation.rs, probability.rs (fold-loop
              parallelization only), lib.rs (cfg + docs), crates/libsvm/Cargo.toml
              (feature comment), README.md (one paragraph),
              .github/workflows/ci.yml (one test step), new parity test file
⚠️ ASK FIRST: any change to fold-assignment logic or util.rs RNG, any pub API
              signature change, parallelism anywhere besides the two fold loops
🚫 NEVER:     solver.rs, kernel.rs, qmatrix.rs, train.rs internals, vendor/,
              scripts/, reference/, io.rs, workspace root Cargo.toml deps

## 5. Non-Goals / Constraints
- Parallelism inside svm_train/solver is OUT — parity risk. Folds only.
- Default (no-feature) build must compile the identical serial code.

## 6. Context Pointers
- `br show libsvm-rs-eo9` — full spec incl. determinism reasoning.
- `VISION.md` point 4 (performance) — rayon stays opt-in.
- util.rs c_rand/shuffle (4r5 work) — do not touch; assignment uses it serially.

## 7. Stop Conditions
- DONE when criteria pass. STOP if bitwise on/off parity cannot be achieved or
  an ⚠️ item is needed.
