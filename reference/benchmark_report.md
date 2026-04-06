# Benchmark Report

Date: 2026-04-06 06:57:42Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.524 | 2.927 | 0.808 | 0.974 | 1.152 |
| `predict_probability` | 30 | 3.012 | 3.266 | 0.832 | 1.031 | 1.135 |
| `train` | 40 | 3.517 | 3.714 | 0.915 | 1.225 | 1.312 |
| `train_probability` | 30 | 10.064 | 9.518 | 1.043 | 1.335 | 1.387 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s4_t0_housing_scale` | `train_probability` | 1.387 |
| `s4_t3_housing_scale` | `train_probability` | 1.344 |
| `s3_t3_housing_scale` | `train_probability` | 1.324 |
| `s4_t0_housing_scale` | `train` | 1.312 |
| `s4_t2_housing_scale` | `train_probability` | 1.304 |
| `s4_t1_housing_scale` | `train_probability` | 1.283 |
| `s3_t3_housing_scale` | `train` | 1.275 |
| `s3_t2_housing_scale` | `train_probability` | 1.258 |
| `s3_t1_housing_scale` | `train_probability` | 1.229 |
| `s4_t3_housing_scale` | `train` | 1.223 |

Raw data: `reference/benchmark_results.json`

