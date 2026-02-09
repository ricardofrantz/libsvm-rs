# libsvm-rs

A **pure Rust** reimplementation of the classic [LIBSVM](https://github.com/cjlin1/libsvm) library, targeting numerical equivalence and model-file compatibility.

[![Crates.io](https://img.shields.io/crates/v/libsvm-rs.svg)](https://crates.io/crates/libsvm-rs)
[![Documentation](https://docs.rs/libsvm-rs/badge.svg)](https://docs.rs/libsvm-rs)
[![CI](https://github.com/ricardofrantz/libsvm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardofrantz/libsvm-rs/actions)
[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](LICENSE)

**Status**: Complete (February 2026). Full training for all 5 SVM types, probability estimates, cross-validation, CLI tools, and 65+ tests verified against C LIBSVM.

## What is LIBSVM?

LIBSVM is one of the most widely cited machine learning libraries ever created:

- **Authors**: Chih-Chung Chang and Chih-Jen Lin (National Taiwan University).
- **First release**: ~2000, still actively maintained (v3.37, December 2025).
- **Citations**: >53,000 (Google Scholar) for the [original paper](https://dl.acm.org/doi/10.1145/1961189.1961199).
- **Core functionality**: Efficient training and inference for **Support Vector Machines (SVMs)**.
  - Classification: C-SVC, nu-SVC
  - Regression: epsilon-SVR, nu-SVR
  - Distribution estimation / novelty detection: one-class SVM
- **Strengths**: Battle-tested SMO solver, excellent performance on sparse/high-dimensional data, compact codebase (~3,300 LOC core).

## Why a Pure Rust Port?

| Crate | Type | Training | Prediction | LIBSVM Compatible | Active |
|-------|------|----------|------------|-------------------|--------|
| **libsvm-rs** | Pure Rust | Yes | Yes | Yes | Yes (2026) |
| libsvm (FFI) | C++ bindings | Yes | Yes | Yes | No (2022) |
| linfa-svm | Pure Rust | Yes | Yes | No | Yes |
| smartcore | Pure Rust | Yes | Yes | No | Yes |
| ffsvm | Pure Rust | No | Yes | Partial | No |

This project provides:

- **Numerical equivalence** with LIBSVM (same predictions on benchmark datasets, within floating-point tolerance).
- **Zero C/C++ dependencies** at runtime (pure Rust, no native linkage).
- **Full memory/thread safety** via Rust's ownership model.
- **Model file interoperability** — files loadable by both implementations.
- **Easy deployment**: single binary, WebAssembly-compatible.

## Features

- [x] All 5 SVM types (C-SVC, nu-SVC, one-class, epsilon-SVR, nu-SVR)
- [x] All kernels (linear, polynomial, RBF, sigmoid, precomputed)
- [x] Full SMO solver with WSS3 working-set selection and shrinking
- [x] LRU kernel cache (Qfloat = f32)
- [x] Model I/O (LIBSVM text format, byte-exact roundtrip)
- [x] Prediction (verified zero mismatches against C `svm-predict`)
- [x] Probability estimates (Platt scaling)
- [x] Cross-validation (stratified for classification)
- [x] Quiet mode (`set_quiet`)
- [x] CLI tools: `svm-train-rs`, `svm-predict-rs`, `svm-scale-rs`

## Installation

```toml
[dependencies]
libsvm-rs = "0.5"
```

## Quick Start

```rust
use libsvm_rs::io::{load_problem, save_model};
use libsvm_rs::train::svm_train;
use libsvm_rs::predict::predict;
use libsvm_rs::{SvmParameter, KernelType};
use std::path::Path;

let problem = load_problem(Path::new("heart_scale")).unwrap();
let mut param = SvmParameter::default(); // C-SVC + RBF
param.gamma = 1.0 / 13.0;

let model = svm_train(&problem, &param);
let label = predict(&model, &problem.instances[0]);
save_model(Path::new("heart_scale.model"), &model).unwrap();
```

See `examples/` for grid search, probability estimates, and model loading demos.

## CLI Tools

```bash
# Train (default: C-SVC with RBF kernel)
svm-train-rs data/heart_scale

# Train with options
svm-train-rs -s 1 -t 0 -v 5 data/heart_scale    # nu-SVC, linear, 5-fold CV
svm-train-rs -b 1 -w1 2 data/heart_scale          # probability, class 1 weight=2

# Predict
svm-predict-rs data/heart_scale heart_scale.model output.txt
svm-predict-rs -b 1 data/heart_scale heart_scale.model output_prob.txt

# Scale features
svm-scale-rs -l 0 -u 1 data/heart_scale > scaled.txt
svm-scale-rs -s params.txt data/heart_scale > scaled.txt   # save params
svm-scale-rs -r params.txt new_data > scaled_new.txt       # restore params
```

## Numerical Equivalence

We target **numerical equivalence**, not bitwise identity. Floating-point results across different compilers and languages are [not guaranteed to be identical](https://gafferongames.com/post/floating_point_determinism/) due to operation reordering, FMA instructions, and intermediate precision differences.

In practice:

- Identical predicted labels on benchmark datasets.
- Regression predictions within ~1e-6 relative tolerance.
- Model files interoperable with original LIBSVM.
- Same support vectors selected (barring degenerate tie-breaking).

## Design

- **Workspace layout**: `crates/libsvm/` (library) + `bins/` (CLI tools)
- **Zero runtime deps**: Only `thiserror`; Rayon is feature-gated
- **Ownership replaces manual memory**: No `free_sv` flag, no `unsafe`
- **Float formatting**: Matches C's `%.17g` for model file compatibility
- **Solver variants**: Enum-based (Standard/Nu) with 95% shared code

## Development

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo setup, CI | Done |
| 1 | Types, I/O | Done |
| 2 | Kernels, cache, prediction | Done |
| 3 | SMO solver (all 5 types) | Done |
| 4 | Probability, cross-validation | Done |
| 5 | CLI tools | Done |
| 6 | Testing, validation pipeline | Done |
| 7 | Docs, polish, publish | Done |

## Contributing

Contributions welcome. Open an issue first for major changes.

## License

BSD-3-Clause (same as original LIBSVM). See [LICENSE](LICENSE).

## Acknowledgments

- Original LIBSVM by Chih-Chung Chang and Chih-Jen Lin.
- Rust ML ecosystem ([linfa](https://github.com/rust-ml/linfa), [smartcore](https://github.com/smartcorelib/smartcore), [ffsvm](https://github.com/ralfbiedert/ffsvm)) for prior art.
