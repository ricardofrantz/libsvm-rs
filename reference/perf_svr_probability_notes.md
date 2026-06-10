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
