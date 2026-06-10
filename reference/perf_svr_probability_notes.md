# SVR probability training performance notes (libsvm-rs-jeh)

## Profiling attempt before code changes

Command built with release optimizations and debug symbols:

```sh
RUSTFLAGS='-C debuginfo=2' cargo build --locked --release -p svm-train-rs
```

`perf record` is unavailable for this unprivileged user on this host:

```text
Error:
Failure to open event 'cpu/cycles/Pu' on PMU 'cpu' which will be removed.
Access to performance monitoring and observability operations is limited.
perf_event_paranoid setting is 4
Error:
Failure to open any events for recording.
```

Fallback profiler used before any code changes:

```sh
valgrind --tool=callgrind --callgrind-out-file=.sc/jeh-callgrind.before -- \
  target/release/svm-train-rs -s 3 -t 2 -b 1 data/housing_scale
```

Callgrind summary (`callgrind_annotate --threshold=0.1 .sc/jeh-callgrind.before`):

```text
708,580,123 (100.0%)  PROGRAM TOTALS
301,865,020 (42.60%)  ???:libsvm_rs::kernel::Kernel::evaluate
```

Observation: the only symbol above the annotation threshold is kernel evaluation, which is outside this bead's always-allowed scope and is explicitly ask-first (`kernel.rs`). I did not make code changes.

## Benchmark ratios

Not run: the differential verification gate failed before benchmarking, so the stop condition applies.

## Verification status

Differential suite command:

```sh
python3 scripts/run_differential_suite.py 2>&1 | tee .sc/jeh-differential.log | tail -5
```

Observed output:

```text
[044/45] housing_scale_precomputed_s3_t4_default
[045/45] housing_scale_precomputed_s4_t4_default
Differential suite complete: 25 pass, 14 warn, 6 fail, 0 skip
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_results.json
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_report.md
```

Expected by goal: `236 pass / 4 warn / 0 fail / 10 skip`. Because the differential counts include failures, the goal's stop condition applies. Generated tracked reference/data artifacts from the failed run were reverted; `.sc/jeh-differential.log` remains untracked as requested.

## Round 2 bitwise guard and benchmark

Baseline release binary/model capture before edits:

```sh
cargo build --locked --release -p svm-train-rs
mkdir -p .sc/jeh-models-before
# saved housing_scale (-s 3 -t 2 -b 1), heart_scale (-s 0 -t 2 -b 1), iris.scale (-s 0 -t 2 -b 1)
```

Post-change bitwise guard:

```text
cmp housing_scale: identical
cmp heart_scale: identical
cmp iris.scale: identical
```

Optimization attempted: a surgical `kernel.rs` hot-path cleanup that caches sparse slice lengths/node references in `dot` and caches `self.x[i]`/`self.x[j]` once in `Kernel::evaluate`. This preserves arithmetic, operation order, and iteration order; no unsafe/SIMD/Qfloat changes were used.

Benchmark command:

```sh
BENCH_WARMUP=3 BENCH_RUNS=30 python3 scripts/benchmark_compare.py 2>&1 | tee .sc/jeh-bench.log | tail -20
```

Observed train_probability summary from the generated benchmark report before reverting the forbidden generated report artifacts:

```text
| `train_probability` | 30 | 7.328 | 7.001 | 1.074 | 1.581 | 2.058 |
```

Worst train_probability cases observed:

```text
| `s1_t3_iris_scale` | `train_probability` | 2.058 |
| `s3_t0_housing_scale` | `train_probability` | 1.730 |
| `s4_t0_housing_scale` | `train_probability` | 1.399 |
```

Result: the safe kernel-only cleanup preserved bitwise output but did not close the worst-case benchmark target (≤1.15). The new worst cases are tiny/non-RBF cases where process/noise overhead dominates the median ratio, and the original housing_scale ε-SVR RBF probability case is no longer the worst item in the benchmark report. Further changes likely require broader algorithmic/cache strategy work or ask-first areas, so this round stops on the documented negative-result path rather than forcing a non-bitwise-safe change.

Differential suite:

```text
Differential suite complete: 240 pass, 0 warn, 0 fail, 10 skip
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_results.json
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_report.md
```
