# Benchmark Report

Date: 2026-02-16 06:09:34Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.593 | 3.029 | 0.808 | 1.000 | 1.145 |
| `predict_probability` | 30 | 3.069 | 3.381 | 0.829 | 1.034 | 1.150 |
| `train` | 40 | 3.472 | 3.829 | 0.919 | 1.264 | 1.298 |
| `train_probability` | 30 | 10.134 | 9.857 | 1.038 | 1.352 | 1.382 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.382 |
| `s4_t3_housing_scale` | `train_probability` | 1.359 |
| `s3_t3_housing_scale` | `train_probability` | 1.343 |
| `s4_t2_housing_scale` | `train_probability` | 1.304 |
| `s4_t0_housing_scale` | `train` | 1.298 |
| `s3_t3_housing_scale` | `train` | 1.272 |
| `s3_t2_housing_scale` | `train_probability` | 1.270 |
| `s4_t1_housing_scale` | `train_probability` | 1.268 |
| `s4_t3_housing_scale` | `train` | 1.264 |
| `s3_t0_housing_scale` | `train_probability` | 1.234 |

Raw data: `reference/benchmark_results.json`

