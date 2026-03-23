# Benchmark Report

Date: 2026-03-23 06:13:31Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.636 | 3.021 | 0.815 | 0.988 | 1.162 |
| `predict_probability` | 30 | 3.113 | 3.371 | 0.839 | 1.031 | 1.166 |
| `train` | 40 | 3.595 | 3.816 | 0.913 | 1.260 | 1.314 |
| `train_probability` | 30 | 10.222 | 9.644 | 1.047 | 1.348 | 1.374 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.374 |
| `s4_t3_housing_scale` | `train_probability` | 1.355 |
| `s3_t3_housing_scale` | `train_probability` | 1.338 |
| `s4_t0_housing_scale` | `train` | 1.314 |
| `s4_t2_housing_scale` | `train_probability` | 1.293 |
| `s4_t1_housing_scale` | `train_probability` | 1.283 |
| `s3_t3_housing_scale` | `train` | 1.265 |
| `s3_t2_housing_scale` | `train_probability` | 1.263 |
| `s4_t3_housing_scale` | `train` | 1.259 |
| `s3_t1_housing_scale` | `train_probability` | 1.236 |

Raw data: `reference/benchmark_results.json`

