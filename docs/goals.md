# Goal: Close the SVR probability-training perf gap   (bead: libsvm-rs-jeh)

## 1. Objective
Rust train_probability is the one benchmark where we lose to C (median 1.025,
worst ~1.39× on housing_scale ε-SVR). Profile it, remove the serial waste
(allocations/copies/cache behavior — NOT algorithm changes), and prove parity
is untouched. VISION.md point 4: a replacement that is slower is not a
replacement.

## 2. Acceptance Criteria
- [ ] Profile evidence captured BEFORE any code change: flamegraph or
      perf report of `svm-train-rs -s 3 -t 2 -b 1 data/housing_scale`
      (release build w/ debug symbols), summary written to
      `reference/perf_svr_probability_notes.md` (top functions + % time).
- [ ] Minimal fixes implemented in the hot paths the profile justifies
      (candidates: per-fold problem subset cloning, fold-loop allocations in
      probability.rs/cross_validation.rs, cache sizing). Identical arithmetic
      and iteration order — no result value may change.
- [ ] Benchmark re-run with `BENCH_WARMUP=3 BENCH_RUNS=30
      python3 scripts/benchmark_compare.py`; before/after train_probability
      ratios recorded in the notes file. Target worst case ≤1.15, OR
      documented evidence the residual gap is environmental/benchmark noise.
- [ ] Differential suite unchanged: `python3 scripts/run_differential_suite.py`
      reports 236 pass / 4 warn / 0 fail / 10 skip, log saved to
      `.sc/jeh-differential.log` (do not commit the log).
- [ ] `cargo test --locked --all-features` green.

## 3. Verification
- Quick: `cargo test --locked -p libsvm-rs probability cross_validation`
- Full (logged, expensive — run once, keep logs in .sc/):
  `python3 scripts/run_differential_suite.py 2>&1 | tee .sc/jeh-differential.log | tail -5`
  `BENCH_WARMUP=3 BENCH_RUNS=30 python3 scripts/benchmark_compare.py 2>&1 | tee .sc/jeh-bench.log | tail -20`

## 4. Scope
✅ ALWAYS:    crates/libsvm/src/probability.rs, crates/libsvm/src/cross_validation.rs,
              crates/libsvm/src/cache.rs, crates/libsvm/src/train.rs (subset
              construction only), reference/perf_svr_probability_notes.md (new)
⚠️ ASK FIRST: solver.rs, kernel.rs, qmatrix.rs, anything changing numeric results
🚫 NEVER:     io.rs formatting, vendor/, .github/, scripts/ (benchmark code is
              the measuring stick — do not "fix" the benchmark),
              reference/benchmark_report.md regeneration (supervisor decides)

## 5. Non-Goals / Constraints
- NOT this cycle: rayon/parallelism (separate bead libsvm-rs-eo9 depends on
  this one), algorithmic changes, solver tuning.
- A NEGATIVE result is acceptable: if the profile shows no Rust-side waste,
  write the evidence into the notes file and stop — do not force changes.

## 6. Context Pointers
- `br show libsvm-rs-jeh` — full background, candidate hypotheses, reasoning.
- `VISION.md` — parity is the product; speed without parity is regression.
- `reference/benchmark_report.md` — current ratios (the baseline).
- svm_svr_probability upstream: vendor/libsvm/svm.cpp ~lines 1130–1250.
- Skills: rust, systematic-debugging, profiling-software-performance.

## 7. Task Breakdown
1. Build + profile the worst case → notes file with top hotspots.
2. Fix the top justified waste → quick gate green after each change.
3. Re-benchmark + differential suite → record before/after, logs in .sc/.

## 8. Stop Conditions
- DONE when criteria pass (including the documented-negative-result path).
- STOP and report if: a fix would require touching ⚠️ files, differential
  counts change at all, or profiling tools are unavailable on this box.
