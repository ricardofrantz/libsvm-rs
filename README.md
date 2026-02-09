# libsvm-rs

A **pure Rust** reimplementation of the classic [LIBSVM](https://github.com/cjlin1/libsvm) library, targeting numerical equivalence and model-file compatibility.

[![Crates.io](https://img.shields.io/crates/v/libsvm-rs.svg)](https://crates.io/crates/libsvm-rs)
[![Documentation](https://docs.rs/libsvm-rs/badge.svg)](https://docs.rs/libsvm-rs)
[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](LICENSE)

**Status**: Active development (February 2026). Phases 0–3 complete (types, I/O, kernels, cache, prediction, **full SMO solver**). Training works for all 5 SVM types.

## What is LIBSVM?

LIBSVM is one of the most widely cited machine learning libraries ever created:

- **Authors**: Chih-Chung Chang and Chih-Jen Lin (National Taiwan University).
- **First release**: ~2000, still actively maintained (v3.37, December 2025).
- **Citations**: >53,000 (Google Scholar) for the [original paper](https://dl.acm.org/doi/10.1145/1961189.1961199).
- **Core functionality**: Efficient training and inference for **Support Vector Machines (SVMs)**.
  - Classification: C-SVC, ν-SVC
  - Regression: ε-SVR, ν-SVR
  - Distribution estimation / novelty detection: one-class SVM
- **Key features**:
  - Multiple kernels: linear, polynomial, RBF (Gaussian), sigmoid, precomputed.
  - Probability estimates (via Platt scaling).
  - Cross-validation and parameter selection helpers.
  - Simple text-based model format for interoperability.
  - CLI tools: `svm-train`, `svm-predict`, `svm-scale`.
- **Strengths**: Battle-tested SMO (Sequential Minimal Optimization) solver, excellent performance on sparse/high-dimensional data (text classification, bioinformatics, sensor data), compact codebase (~3,300 LOC core).

## Why a Pure Rust Port?

Existing Rust options for SVMs don't provide full LIBSVM-compatible training:

| Option | Type | Pros | Cons |
|---|---|---|---|
| **[libsvm](https://crates.io/crates/libsvm)** | FFI bindings to C++ | Full feature parity | Stale (last updated 2022), requires native build |
| **[linfa-svm](https://crates.io/crates/linfa-svm)** | Pure Rust (linfa) | Modern API, active | Different algorithms/heuristics, not compatible |
| **[smartcore](https://crates.io/crates/smartcore)** | Pure Rust | Good coverage, active | Approximate solver, not LIBSVM-equivalent |
| **[ffsvm](https://crates.io/crates/ffsvm)** | Pure Rust | LIBSVM model loading, fast inference | **Prediction only** — no training |

**This project aims to fill the gap** by providing:

- **Numerical equivalence** with LIBSVM (same predictions and model files on benchmark datasets, within floating-point tolerance).
- **Full memory/thread safety** via Rust's ownership model — no undefined behavior in sparse data handling.
- **Zero C/C++ dependencies** at runtime (pure Rust, no native linkage).
- **Fearless concurrency** (e.g., parallel cross-validation with Rayon).
- **Easy deployment**: single binary, WebAssembly support for browser inference.
- **Modern ergonomics** while preserving compatibility (builders, iterators, `Result`-based error handling).

Ideal for:

- Reproducible research needing LIBSVM-compatible results.
- Embedded/lightweight ML (WASM, edge devices).
- Rust data/ML pipelines without native build headaches.

### A Note on Numerical Equivalence

We target **numerical equivalence**, not bitwise identity. Floating-point results across different compilers (GCC vs LLVM) and languages are [not guaranteed to be identical](https://gafferongames.com/post/floating_point_determinism/) due to operation reordering, FMA instructions, and intermediate precision differences. This is an [open problem](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p3375r3.html) even within C++ itself.

In practice, this means:

- Identical predicted labels on benchmark datasets.
- Probabilities within ~1e-8 tolerance.
- Model files interoperable with original LIBSVM (loadable by either implementation).
- Same support vectors selected (barring degenerate tie-breaking cases).

## Goals

1. **Compatibility**
   - Pass all official LIBSVM test scenarios.
   - Equivalent output (predictions, probabilities, model files) on standard datasets (`heart_scale`, `a9a`, etc.).
   - Model files readable by both this library and original LIBSVM.

2. **Safety**
   - 100% safe Rust where possible (no `unsafe` unless heavily justified and tested).
   - Comprehensive error handling (`thiserror`).
   - Graceful handling of malformed input.

3. **Performance**
   - Target: match original C++ speed after optimization (initial port may be 10–20% slower).
   - Optional Rayon parallelism for cross-validation and grid search.

4. **Extras (Post-MVP)**
   - PyO3 bindings for Python drop-in replacement.
   - WASM examples.
   - Optional dense matrix support via `ndarray`.

## Features Roadmap

- [x] Core data structures (`SvmNode`, `SvmProblem`, `SvmParameter`, `SvmModel`)
- [x] All kernels (linear, polynomial, RBF, sigmoid, precomputed)
- [x] Kernel cache (O(1) LRU)
- [x] Model save/load (exact LIBSVM text format, byte-exact roundtrip)
- [x] Prediction (verified zero mismatches against C `svm-predict`)
- [x] Full SMO solver (C-SVC, ν-SVC, ε-SVR, ν-SVR, one-class)
- [x] Shrinking heuristic
- [x] WSS3 working-set selection (Fan et al., JMLR 2005)
- [x] QMatrix implementations (SvcQ, OneClassQ, SvrQ)
- [ ] Probability estimates (Platt scaling)
- [ ] Cross-validation (parallel optional)
- [ ] CLI tools: `svm-train-rs`, `svm-predict-rs`, `svm-scale-rs`
- [ ] Comprehensive test suite with reference outputs

## Installation

```toml
# Cargo.toml — when published
[dependencies]
libsvm-rs = "0.3.0"
```

Until published:

```bash
cargo add libsvm-rs --git https://github.com/ricardofrantz/libsvm-rs
```

## Usage Example

```rust
use libsvm_rs::io::{load_problem, save_model};
use libsvm_rs::train::svm_train;
use libsvm_rs::predict::predict;
use libsvm_rs::{SvmParameter, SvmType, KernelType};

// Load training data
let problem = load_problem("heart_scale").unwrap();

// Set parameters
let param = SvmParameter {
    svm_type: SvmType::CSvc,
    kernel_type: KernelType::Rbf,
    gamma: 1.0 / 13.0,  // 1/num_features
    c: 1.0,
    ..Default::default()
};

// Train
let model = svm_train(&problem, &param);

// Predict
let label = predict(&model, &problem.instances[0]);
println!("Predicted label: {}", label);

// Save model (loadable by original LIBSVM)
save_model("heart_scale.model", &model).unwrap();
```

See `examples/` for full demos (once implemented).

## Development Plan

### Project Structure

```
src/
  lib.rs
  types.rs      # SvmNode, SvmProblem, SvmParameter, SvmModel
  kernel.rs     # kernel functions + cache
  solver.rs     # core SMO
  cache.rs      # LRU kernel cache
  io.rs         # model/problem parsing (LIBSVM text format)
  bin/
    train.rs
    predict.rs
    scale.rs
tests/
  integration/
examples/
benches/
```

### Phases

| Phase | Description | Estimated Effort |
|---|---|---|
| **0** | Repository setup, CI, dependencies | 1–2 days |
| **1** | Data structures & I/O (parsing, model format) | 1–2 weeks |
| **2** | Kernels, cache & prediction (load pre-trained models, verify) | 1–2 weeks |
| **3** | Core SMO solver (all SVM types) | 6–12 weeks |
| **4** | Probability estimates, shrinking, cross-validation | 2–4 weeks |
| **5** | CLI tools (`svm-train-rs`, `svm-predict-rs`, `svm-scale-rs`) | 1–2 weeks |
| **6** | Testing & validation (reference outputs, fuzzing, benchmarks) | Ongoing |
| **7** | Documentation, polish, publish to crates.io | 1–2 weeks |

**Total estimated effort**: 3–6 months.

Phase 3 is the bulk of the work — the SMO solver in `svm.cpp` is ~1,000 lines of subtle numerical code with heuristics (working set selection, shrinking, cache management). Translating C++ manual memory management to Rust ownership patterns, plus verifying numerical correctness across all SVM types, is the primary challenge.

### Key References

- [svm.h](https://github.com/cjlin1/libsvm/blob/master/svm.h) — API and struct definitions
- [svm.cpp](https://github.com/cjlin1/libsvm/blob/master/svm.cpp) — Core implementation (~3,300 LOC)
- [LIBSVM datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) — Benchmark data

### Testing Strategy

1. Run original LIBSVM on benchmark datasets → save all outputs as reference.
2. Integration tests compare against reference:
   - Exact label matches.
   - Probabilities within tolerance (`float-cmp` with ε ≈ 1e-8).
   - Model file compatibility (load in both directions).
3. Include regression suite from official LIBSVM `tools/` subdirectory.
4. Fuzz with `cargo-fuzz` on input parsing.
5. Benchmark with `criterion` against original C++ implementation.

## Contributing

Contributions welcome! Especially:

- Translating specific solver components.
- Adding dataset-based tests.
- Performance improvements (preserving numerical behavior).

Open an issue first for major changes.

## Changelog

### v0.3.0 (February 2026) — SMO Solver

- **Full SMO solver** for all 5 SVM types: C-SVC, ν-SVC, one-class, ε-SVR, ν-SVR
- **WSS3 working-set selection** (second-order heuristic from Fan et al., JMLR 2005)
- **Shrinking heuristic** with gradient reconstruction
- **QMatrix trait** with three implementations: `SvcQ`, `OneClassQ`, `SvrQ`
- **`svm_train`** function producing `SvmModel` compatible with C LIBSVM
- **Multiclass support** via one-vs-one with class grouping and sv_coef assembly
- **Cache::swap_index** bug fix — added column swap loop (critical for shrinking correctness)
- **Kernel refactor** — `Vec<&[SvmNode]>` for swappable data point references
- 50 tests (12 new), verified against C LIBSVM reference outputs

### v0.2.0 (February 2026) — Prediction & I/O

- Core types: `SvmNode`, `SvmProblem`, `SvmParameter`, `SvmModel`
- All 5 kernel functions (linear, polynomial, RBF, sigmoid, precomputed)
- LRU kernel cache
- Model and problem I/O (LIBSVM text format, byte-exact roundtrip)
- Prediction (zero mismatches against C `svm-predict` on heart_scale)
- Parameter validation with ν-SVC feasibility check
- 38 tests

### v0.1.0 (February 2026) — Initial Release

- Repository setup, workspace layout, CI

## License

BSD-3-Clause (same as original LIBSVM) for maximum compatibility.

## Acknowledgments

- Original LIBSVM by Chih-Chung Chang and Chih-Jen Lin.
- Existing Rust ML ecosystem ([linfa](https://github.com/rust-ml/linfa), [smartcore](https://github.com/smartcorelib/smartcore), [ffsvm](https://github.com/ralfbiedert/ffsvm)) for prior art.
