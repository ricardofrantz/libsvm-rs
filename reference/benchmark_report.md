# Benchmark Report

Date: 2026-06-15 10:58:31Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.574 | 2.928 | 0.836 | 0.958 | 0.987 |
| `predict_probability` | 30 | 3.036 | 3.429 | 0.869 | 0.963 | 1.003 |
| `train` | 40 | 3.586 | 3.703 | 0.931 | 1.216 | 1.356 |
| `train_probability` | 30 | 9.909 | 9.087 | 1.083 | 1.326 | 1.403 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.403 |
| `s4_t0_housing_scale` | `train` | 1.356 |
| `s4_t1_housing_scale` | `train_probability` | 1.339 |
| `s3_t0_housing_scale` | `train_probability` | 1.309 |
| `s3_t1_housing_scale` | `train_probability` | 1.299 |
| `s4_t2_housing_scale` | `train_probability` | 1.293 |
| `s3_t2_housing_scale` | `train_probability` | 1.258 |
| `s3_t0_housing_scale` | `train` | 1.241 |
| `s4_t1_housing_scale` | `train` | 1.214 |
| `s3_t1_housing_scale` | `train` | 1.211 |

Raw data: `reference/benchmark_results.json`

