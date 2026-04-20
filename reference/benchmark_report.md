# Benchmark Report

Date: 2026-04-20 07:20:13Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.533 | 2.942 | 0.804 | 0.981 | 1.158 |
| `predict_probability` | 30 | 3.026 | 3.287 | 0.834 | 1.030 | 1.157 |
| `train` | 40 | 3.427 | 3.738 | 0.916 | 1.260 | 1.308 |
| `train_probability` | 30 | 10.047 | 9.576 | 1.025 | 1.346 | 1.388 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.388 |
| `s4_t3_housing_scale` | `train_probability` | 1.353 |
| `s3_t3_housing_scale` | `train_probability` | 1.337 |
| `s4_t0_housing_scale` | `train` | 1.308 |
| `s4_t2_housing_scale` | `train_probability` | 1.307 |
| `s4_t1_housing_scale` | `train_probability` | 1.282 |
| `s4_t3_housing_scale` | `train` | 1.273 |
| `s3_t3_housing_scale` | `train` | 1.259 |
| `s3_t2_housing_scale` | `train_probability` | 1.254 |
| `s3_t0_housing_scale` | `train_probability` | 1.219 |

Raw data: `reference/benchmark_results.json`

