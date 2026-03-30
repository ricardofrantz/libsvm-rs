# Benchmark Report

Date: 2026-03-30 06:56:00Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.522 | 2.917 | 0.807 | 0.975 | 1.170 |
| `predict_probability` | 30 | 3.018 | 3.284 | 0.831 | 1.034 | 1.166 |
| `train` | 40 | 3.427 | 3.714 | 0.924 | 1.269 | 1.295 |
| `train_probability` | 30 | 10.137 | 9.546 | 1.044 | 1.345 | 1.381 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.381 |
| `s4_t3_housing_scale` | `train_probability` | 1.354 |
| `s3_t3_housing_scale` | `train_probability` | 1.334 |
| `s4_t2_housing_scale` | `train_probability` | 1.303 |
| `s4_t0_housing_scale` | `train` | 1.295 |
| `s4_t1_housing_scale` | `train_probability` | 1.284 |
| `s4_t3_housing_scale` | `train` | 1.275 |
| `s3_t3_housing_scale` | `train` | 1.269 |
| `s3_t2_housing_scale` | `train_probability` | 1.255 |
| `s3_t1_housing_scale` | `train_probability` | 1.226 |

Raw data: `reference/benchmark_results.json`

