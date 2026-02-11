# Benchmark Report

Date: 2026-02-11 19:54:00Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 5.598 | 5.287 | 1.008 | 1.224 | 1.368 |
| `predict_probability` | 30 | 5.822 | 6.213 | 0.992 | 1.137 | 1.218 |
| `train` | 40 | 6.239 | 6.248 | 1.020 | 1.171 | 1.282 |
| `train_probability` | 30 | 13.066 | 11.439 | 1.138 | 1.344 | 1.442 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.442 |
| `s2_t0_heart_scale` | `predict` | 1.368 |
| `s3_t0_housing_scale` | `train_probability` | 1.356 |
| `s1_t0_iris_scale` | `train_probability` | 1.329 |
| `s4_t0_housing_scale` | `train` | 1.282 |
| `s3_t1_housing_scale` | `train_probability` | 1.279 |
| `s3_t0_housing_scale` | `predict` | 1.273 |
| `s0_t0_heart_scale` | `train_probability` | 1.252 |
| `s3_t2_housing_scale` | `train_probability` | 1.245 |
| `s4_t1_housing_scale` | `train_probability` | 1.242 |

Raw data: `reference/benchmark_results.json`

