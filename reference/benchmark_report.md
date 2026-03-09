# Benchmark Report

Date: 2026-03-09 06:03:10Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.344 | 2.728 | 0.792 | 0.915 | 0.981 |
| `predict_probability` | 30 | 2.763 | 3.191 | 0.835 | 0.945 | 0.985 |
| `train` | 40 | 3.318 | 3.665 | 0.913 | 1.248 | 1.431 |
| `train_probability` | 30 | 11.151 | 9.695 | 1.131 | 1.405 | 1.542 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.542 |
| `s4_t0_housing_scale` | `train` | 1.431 |
| `s3_t0_housing_scale` | `train_probability` | 1.409 |
| `s4_t1_housing_scale` | `train_probability` | 1.400 |
| `s0_t0_heart_scale` | `train_probability` | 1.365 |
| `s4_t2_housing_scale` | `train_probability` | 1.365 |
| `s3_t2_housing_scale` | `train_probability` | 1.337 |
| `s3_t1_housing_scale` | `train_probability` | 1.310 |
| `s4_t3_housing_scale` | `train_probability` | 1.310 |
| `s3_t0_housing_scale` | `train` | 1.309 |

Raw data: `reference/benchmark_results.json`

