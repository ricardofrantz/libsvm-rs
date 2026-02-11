# libsvm-rs

A pure Rust port of [LIBSVM](https://github.com/cjlin1/libsvm) with matching model format, matching CLI behavior targets, and a reproducible Rust-vs-C verification pipeline.

[![Crates.io](https://img.shields.io/crates/v/libsvm-rs.svg)](https://crates.io/crates/libsvm-rs)
[![Documentation](https://docs.rs/libsvm-rs/badge.svg)](https://docs.rs/libsvm-rs)
[![CI](https://github.com/ricardofrantz/libsvm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardofrantz/libsvm-rs/actions)
[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](LICENSE)

## Project Status (2026-02-11)

- Upstream target is pinned to LIBSVM `v337` (`LIBSVM_VERSION=337`) via `reference/libsvm_upstream_lock.json`.
- Differential verification (`quick` scope): `45` total, `45 pass`, `0 warn`, `0 fail`, `0 skip`.
- Differential verification (`full` scope, generated `2026-02-11T13:56:06Z`): `250` total, `236 pass`, `4 warn`, `0 fail`, `10 skip`.
- Active tolerance policy is `differential-v3` (`reference/tolerance_policy.md`).
- All `10` current skips are `nu_svc` on the synthetic `gen_binary_imbalanced.scale` family, where both C and Rust fail identically.
- Coverage gate currently passes:
  - `libsvm-rs` line: `93.19%`
  - `libsvm-rs` function: `92.86%`
  - workspace line: `88.77%`
- Benchmark infrastructure exists, but current report was generated with low sample count and should be rerun with higher repetitions before drawing performance conclusions.
- Revised security audit reports strong baseline posture and zero RustSec findings (`SECURITY_AUDIT.md`).

Current reports:
- `reference/differential_report.md`
- `reference/tolerance_policy.md`
- `reference/coverage_report.md`
- `reference/benchmark_report.md`
- `SECURITY_AUDIT.md`

## What This Repository Is

This repository contains:

- A Rust library crate implementing LIBSVM-compatible training and prediction.
- Rust CLI binaries matching the `svm-train`, `svm-predict`, and `svm-scale` workflows.
- Verification scripts that run Rust and upstream C on the same data/parameters and produce machine-readable artifacts under `reference/`.

Goal: trust-grade parity evidence against upstream LIBSVM, not marketing claims.

## What Was Implemented in This Revision

1. Upstream lock and CI validation:
   - `reference/libsvm_upstream_lock.json`
   - `scripts/check_libsvm_reference_lock.sh`
   - CI gate in `.github/workflows/ci.yml`
2. Pinned upstream build and provenance:
   - `scripts/setup_reference_libsvm.sh`
   - `reference/reference_provenance.json`
   - `reference/reference_build_report.md`
3. Deterministic synthetic differential datasets:
   - `scripts/generate_differential_datasets.py`
   - `data/generated/*`
   - `reference/dataset_manifest.json`
4. Differential suite harness with JSON + markdown outputs:
   - `scripts/run_differential_suite.py`
   - `reference/differential_results.json`
   - `reference/differential_report.md`
5. Differential tolerance policy and warning governance:
   - `reference/tolerance_policy.md`
   - policy id `differential-v3`
   - targeted warning guardrail for `housing_scale_s3_t2_tuned` (epsilon-SVR, RBF, tuned)
6. Security hardening and revised audit evidence:
   - `SECURITY_AUDIT.md`
   - negative feature index hardening in `svm-scale-rs`
   - RustSec dependency audit clean

## Upstream Compatibility Target and Versioning Policy

Parity target is locked to an upstream C release independent of crate semver.

- Crate version (`libsvm-rs`): Rust semver for API/package lifecycle.
- Parity target (`reference/libsvm_upstream_lock.json`): exact upstream URL/tag/commit/version used for verification.

This allows stable Rust release management while keeping parity claims auditable against a specific upstream commit.

## How To Verify This Port End-to-End

Run from repository root.

1. Validate lock consistency:

```bash
bash scripts/check_libsvm_reference_lock.sh
```

2. Build pinned upstream reference and generate provenance:

```bash
bash scripts/setup_reference_libsvm.sh
```

3. Run differential verification:

```bash
# canonical matrix
python3 scripts/run_differential_suite.py

# expanded matrix (canonical + generated + tuned)
DIFF_SCOPE=full python3 scripts/run_differential_suite.py

# strict-only (disable targeted SVR warning downgrade)
DIFF_ENABLE_TARGETED_SVR_WARN=0 DIFF_SCOPE=full python3 scripts/run_differential_suite.py

# sensitivity study (global non-prob scalar relative tolerance override)
DIFF_NONPROB_REL_TOL=2e-5 DIFF_SCOPE=full python3 scripts/run_differential_suite.py
```

4. Run coverage gate:

```bash
bash scripts/check_coverage_thresholds.sh
```

5. Run Rust-vs-C benchmarks:

```bash
# Example stronger sampling than the default
BENCH_WARMUP=3 BENCH_RUNS=30 python3 scripts/benchmark_compare.py
```

## Verification Artifact Map

- Lock and provenance:
  - `reference/libsvm_upstream_lock.json`
  - `reference/reference_provenance.json`
  - `reference/reference_build_report.md`
- Differential results:
  - `reference/differential_results.json`
  - `reference/differential_report.md`
  - `reference/tolerance_policy.md`
- Coverage:
  - `reference/coverage_report.md`
- Performance:
  - `reference/benchmark_results.json`
  - `reference/benchmark_report.md`
- Security:
  - `SECURITY_AUDIT.md`

## How To Read Differential Results

- `pass`: no parity issues detected for the case under configured tolerances.
- `warn`: non-fatal differences detected under explicit policy rules. In current runs these are limited to:
  - one targeted epsilon-SVR near-parity drift case (`housing_scale_s3_t2_tuned`) with extra cross-predict checks
  - probability metadata drift
  - rho-only near-equivalence drift
  - one-class near-boundary label drift
- `fail`: deterministic parity break (for example label mismatch or model header mismatch outside thresholds).
- `skip`: combo not executed, usually because training failed in both implementations for that combo.

## Current Caveats Before Any "Full Parity" Claim

As of 2026-02-11 full-scope run:

- `0` hard failures under default `differential-v3` policy.
- `4` warnings remain:
  - targeted epsilon-SVR near-parity drift (`housing_scale_s3_t2_tuned`)
  - one probability header drift case (`probA`)
  - one rho-only near-equivalence drift case
  - one one-class near-boundary label drift case
- `10` skips need classification as invalid-combo vs genuine missing coverage.
- Benchmarks should be rerun with stronger statistical settings before claiming outperformance.

Current honest claim: no hard differential failures under the documented default policy, with a small set of explicitly justified warnings. This is strong parity evidence, but not bitwise identity across all modes.

## Developer Notes: What Works Well

These areas are currently in good shape and can be treated as stable unless new evidence appears:

- End-to-end Rust-vs-C differential harness is reproducible and version-locked.
- Canonical matrix (`quick`) is fully clean (`45/45` pass).
- Full matrix (`full`) has no hard failures under default policy.
- Predictor path parity is strong; residual drift currently comes from training-side numerics, not prediction command behavior.
- Parser and CLI hardening include explicit feature-index bounds checks, with security audit evidence in `SECURITY_AUDIT.md`.

## Developer Notes: What Is Not Perfect Yet

These are known non-perfect areas and should not be hidden in release notes or parity claims:

1. `housing_scale_s3_t2_tuned` (epsilon-SVR, RBF, tuned) is a targeted warning, not a pass.
2. One generated regression case has probability header `probA` drift.
3. One generated extreme-scale case has rho-only near-equivalence drift.
4. One generated one-class case has near-boundary label drift.
5. Ten generated `nu_svc` imbalanced cases are skipped because both implementations fail training.

Current warning case IDs (from latest full run):

- `housing_scale_s3_t2_tuned`
- `gen_regression_sparse_scale_s4_t3_tuned`
- `gen_extreme_scale_scale_s0_t1_default`
- `gen_extreme_scale_scale_s2_t1_default`

Current skip case family:

- `gen_binary_imbalanced_scale_s1_*` and `gen_binary_imbalanced_scale_precomputed_s1_t4_*` (default+tuned), all with reason: both C and Rust training failed.

## Targeted Warning Policy (Why It Exists)

The targeted epsilon-SVR warning for `housing_scale_s3_t2_tuned` is intentionally narrow and guarded:

- applies to one case ID only
- max non-prob drift bounds must hold (`max_rel <= 6e-5`, `max_abs <= 6e-4`)
- model drift bounds must hold (`rho_rel <= 1e-5`, `max sv_coef abs diff <= 4e-3`)
- cross-predict parity must hold in both directions:
  - Rust predictor on C model matches C predictor on C model
  - C predictor on Rust model matches Rust predictor on Rust model

This can be disabled for strict-only runs:

```bash
DIFF_ENABLE_TARGETED_SVR_WARN=0 DIFF_SCOPE=full python3 scripts/run_differential_suite.py
```

## Release Claim Checklist For Developers

Before stating parity/security status publicly, run and verify:

1. `bash scripts/check_libsvm_reference_lock.sh`
2. `bash scripts/setup_reference_libsvm.sh`
3. `DIFF_SCOPE=quick python3 scripts/run_differential_suite.py`
4. `DIFF_SCOPE=full python3 scripts/run_differential_suite.py`
5. `bash scripts/check_coverage_thresholds.sh`
6. `cargo audit`

And then confirm these artifacts are current:

- `reference/differential_results.json`
- `reference/differential_report.md`
- `reference/tolerance_policy.md`
- `reference/coverage_report.md`
- `SECURITY_AUDIT.md`

## Features

- All 5 SVM types (`-s 0..4`)
- All kernels including precomputed (`-t 0..4`)
- Model I/O compatible with LIBSVM text format
- Probability mode (`-b 1`) implemented
- Cross-validation support
- CLI tools:
  - `svm-train-rs`
  - `svm-predict-rs`
  - `svm-scale-rs`

## Installation

```toml
[dependencies]
libsvm-rs = "0.5"
```

## Quick Start

```rust
use libsvm_rs::io::{load_problem, save_model};
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{KernelType, SvmParameter};
use std::path::Path;

let problem = load_problem(Path::new("data/heart_scale")).unwrap();

let mut param = SvmParameter::default();
param.kernel_type = KernelType::Rbf;
param.gamma = 1.0 / 13.0;

let model = svm_train(&problem, &param);
let label = predict(&model, &problem.instances[0]);

println!("predicted label: {label}");
save_model(Path::new("heart_scale.model"), &model).unwrap();
```

## CLI Usage

```bash
# train
svm-train-rs data/heart_scale
svm-train-rs -s 1 -t 0 -v 5 data/heart_scale

# predict
svm-predict-rs data/heart_scale heart_scale.model output.txt
svm-predict-rs -b 1 data/heart_scale heart_scale.model output_prob.txt

# scale
svm-scale-rs -l 0 -u 1 data/heart_scale > scaled.txt
```

## License

BSD-3-Clause, same family as original LIBSVM. See `LICENSE`.
