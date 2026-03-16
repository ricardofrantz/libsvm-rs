# Benchmark Report

Date: 2026-03-16 06:22:15Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.654 | 3.067 | 0.808 | 0.972 | 1.152 |
| `predict_probability` | 30 | 3.130 | 3.401 | 0.838 | 1.034 | 1.167 |
| `train` | 40 | 3.609 | 3.875 | 0.909 | 1.267 | 1.298 |
| `train_probability` | 30 | 10.246 | 9.753 | 1.052 | 1.343 | 1.380 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.380 |
| `s4_t3_housing_scale` | `train_probability` | 1.352 |
| `s3_t3_housing_scale` | `train_probability` | 1.332 |
| `s4_t0_housing_scale` | `train` | 1.298 |
| `s4_t2_housing_scale` | `train_probability` | 1.296 |
| `s4_t1_housing_scale` | `train_probability` | 1.279 |
| `s4_t3_housing_scale` | `train` | 1.268 |
| `s3_t3_housing_scale` | `train` | 1.267 |
| `s3_t2_housing_scale` | `train_probability` | 1.260 |
| `s3_t1_housing_scale` | `train_probability` | 1.238 |

Raw data: `reference/benchmark_results.json`

