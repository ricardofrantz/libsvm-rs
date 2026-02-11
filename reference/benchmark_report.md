# Benchmark Report

Date: 2026-02-11 06:47:30Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `0`
- Measured runs per command: `1`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 4.790 | 4.439 | 0.981 | 1.253 | 1.752 |
| `predict_probability` | 30 | 5.643 | 5.397 | 0.992 | 1.422 | 1.891 |
| `train` | 40 | 5.961 | 5.773 | 1.008 | 1.427 | 1.646 |
| `train_probability` | 30 | 10.732 | 9.167 | 1.165 | 1.329 | 1.435 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s0_t2_heart_scale` | `predict_probability` | 1.891 |
| `s1_t0_iris_scale` | `predict` | 1.752 |
| `s2_t3_iris_scale` | `train` | 1.646 |
| `s0_t1_iris_scale` | `predict_probability` | 1.549 |
| `s2_t0_heart_scale` | `train` | 1.533 |
| `s4_t0_housing_scale` | `train_probability` | 1.435 |
| `s0_t1_iris_scale` | `train` | 1.422 |
| `s1_t2_iris_scale` | `predict` | 1.395 |
| `s1_t3_heart_scale` | `train` | 1.381 |
| `s0_t0_heart_scale` | `train_probability` | 1.360 |

Raw data: `reference/benchmark_results.json`

