# Benchmark Report

Date: 2026-02-23 06:09:09Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.349 | 2.679 | 0.781 | 0.947 | 1.058 |
| `predict_probability` | 30 | 2.776 | 3.224 | 0.838 | 0.947 | 0.988 |
| `train` | 40 | 3.317 | 3.564 | 0.916 | 1.230 | 1.421 |
| `train_probability` | 30 | 10.531 | 9.680 | 1.099 | 1.364 | 1.525 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.525 |
| `s4_t0_housing_scale` | `train` | 1.421 |
| `s3_t0_housing_scale` | `train_probability` | 1.371 |
| `s4_t2_housing_scale` | `train_probability` | 1.356 |
| `s4_t1_housing_scale` | `train_probability` | 1.327 |
| `s3_t2_housing_scale` | `train_probability` | 1.327 |
| `s3_t0_housing_scale` | `train` | 1.318 |
| `s3_t1_housing_scale` | `train_probability` | 1.296 |
| `s4_t3_housing_scale` | `train_probability` | 1.277 |
| `s0_t0_heart_scale` | `train_probability` | 1.272 |

Raw data: `reference/benchmark_results.json`

