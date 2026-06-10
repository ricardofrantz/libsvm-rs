# Goal: Close the SVR probability-training perf gap — round 2   (bead: libsvm-rs-jeh)

## 1. Objective
Rust loses to C only on train_probability (median 1.025, worst ~1.39× on
housing_scale ε-SVR). Profile evidence already captured (callgrind, committed:
reference/perf_svr_probability_notes.md): 42.6% of instructions in
`Kernel::evaluate`. Ask-first was GRANTED for kernel.rs (see bead comment):
optimize the hot paths under a bitwise guard — results must not change at all.

## 2. Acceptance Criteria
- [ ] BEFORE any edit: build clean-tree release binary and save baseline model
      files to `.sc/jeh-models-before/` for: housing_scale (-s 3 -t 2 -b 1),
      heart_scale (-s 0 -t 2 -b 1), iris.scale (-s 0 -t 2 -b 1).
- [ ] Optimize the justified hot paths — kernel.rs allowed (e.g. layout,
      inlining, redundant-work removal in dot/powi paths), plus probability.rs,
      cross_validation.rs, cache.rs, train.rs subset construction. Identical
      arithmetic, operation order, and iteration order — bitwise-same results.
- [ ] Bitwise guard: retrain the three cases post-change; `cmp` each model file
      byte-identical to `.sc/jeh-models-before/`. Record the cmp results in
      reference/perf_svr_probability_notes.md.
- [ ] Benchmark before/after: `BENCH_WARMUP=3 BENCH_RUNS=30 python3
      scripts/benchmark_compare.py 2>&1 | tee .sc/jeh-bench.log | tail -20`;
      append train_probability ratios to the notes file. Target worst ≤1.15 OR
      documented evidence the residual is environmental/noise.
- [ ] `DIFF_SCOPE=full python3 scripts/run_differential_suite.py` on this box:
      240 pass / 0 warn / 0 fail / 10 skip (the new Linux baseline — any drift
      from these exact counts is a STOP).
- [ ] `cargo test --locked --all-features` green; clippy -D warnings; fmt clean.

## 3. Verification
- Quick: `cargo test --locked -p libsvm-rs kernel probability`
- Full (logged, run once each):
  `DIFF_SCOPE=full python3 scripts/run_differential_suite.py 2>&1 | tee .sc/jeh-differential2.log | tail -3`
  plus the benchmark command above.

## 4. Scope
✅ ALWAYS:    crates/libsvm/src/kernel.rs, probability.rs, cross_validation.rs,
              cache.rs, train.rs (subset construction only),
              reference/perf_svr_probability_notes.md (append)
⚠️ ASK FIRST: solver.rs, qmatrix.rs, any unsafe code, SIMD intrinsics,
              changing Qfloat/f32 vs f64 anywhere
🚫 NEVER:     vendor/, scripts/ (the benchmark is the measuring stick),
              .github/, io.rs formatting; do NOT regenerate committed
              reference/ artifacts (differential_report.md, results.json,
              benchmark_report.md) — revert if the suite rewrites them.

## 5. Non-Goals / Constraints
- NOT this cycle: rayon/parallelism (libsvm-rs-eo9 depends on this bead),
  algorithmic changes (different kernel math, caching strategy redesigns).
- A NEGATIVE result remains acceptable: if no safe win exists, write the
  evidence into the notes file and stop — do not force changes.

## 6. Context Pointers
- `br show libsvm-rs-jeh` — background + the Ask-first grant in comments.
- reference/perf_svr_probability_notes.md — existing callgrind evidence.
- `VISION.md` point 4 — slower-than-C is not a replacement; parity is sacred.
- C hot path: vendor/libsvm/svm.cpp Kernel::dot / kernel_function (~195-360).

## 7. Stop Conditions
- DONE when criteria pass (or documented-negative-result path).
- STOP if: any model byte differs, differential counts drift from 240/0/0/10,
  a fix needs an ⚠️ item, or perf tooling regresses (callgrind is the fallback).
