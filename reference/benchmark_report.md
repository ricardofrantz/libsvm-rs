# Benchmark Report

Date: 2026-04-13 07:17:43Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.538 | 2.952 | 0.812 | 0.958 | 0.972 |
| `predict_probability` | 30 | 3.012 | 3.426 | 0.862 | 0.970 | 0.982 |
| `train` | 40 | 3.583 | 3.730 | 0.913 | 1.193 | 1.332 |
| `train_probability` | 30 | 9.845 | 9.048 | 1.067 | 1.304 | 1.377 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.377 |
| `s4_t0_housing_scale` | `train` | 1.332 |
| `s4_t1_housing_scale` | `train_probability` | 1.318 |
| `s4_t2_housing_scale` | `train_probability` | 1.288 |
| `s3_t1_housing_scale` | `train_probability` | 1.286 |
| `s3_t2_housing_scale` | `train_probability` | 1.249 |
| `s3_t0_housing_scale` | `train_probability` | 1.236 |
| `s3_t0_housing_scale` | `train` | 1.216 |
| `s4_t3_housing_scale` | `train_probability` | 1.204 |
| `s4_t1_housing_scale` | `train` | 1.192 |

Raw data: `reference/benchmark_results.json`

