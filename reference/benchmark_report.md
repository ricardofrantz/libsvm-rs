# Benchmark Report

Date: 2026-06-08 09:27:51Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `20`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.556 | 2.977 | 0.812 | 0.968 | 1.183 |
| `predict_probability` | 30 | 3.283 | 3.307 | 0.835 | 1.149 | 13.576 |
| `train` | 40 | 3.635 | 3.726 | 0.938 | 1.264 | 1.317 |
| `train_probability` | 30 | 10.332 | 9.596 | 1.055 | 1.349 | 1.411 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s1_t3_heart_scale` | `predict_probability` | 13.576 |
| `s4_t0_housing_scale` | `train_probability` | 1.411 |
| `s4_t3_housing_scale` | `train_probability` | 1.358 |
| `s3_t3_housing_scale` | `train_probability` | 1.338 |
| `s4_t0_housing_scale` | `train` | 1.317 |
| `s4_t2_housing_scale` | `train_probability` | 1.312 |
| `s4_t1_housing_scale` | `train_probability` | 1.287 |
| `s3_t3_housing_scale` | `train` | 1.276 |
| `s3_t0_housing_scale` | `train_probability` | 1.270 |
| `s3_t2_housing_scale` | `train_probability` | 1.266 |

Raw data: `reference/benchmark_results.json`

