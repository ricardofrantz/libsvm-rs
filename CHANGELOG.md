# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-02-09

### Added

- Probability estimates: Sigmoid probability model training and prediction
- Cross-validation: k-fold cross-validation with stratified splits
- `probability` module: `SigmoidTrainer`, `sigmoid_predict`, multiclass calibration
- `cross_validation` module: `StratifiedKFold` for proper class distribution
- 20 new unit tests covering probability and CV workflows

### Changed

- `SvmModel::predict_probability_multiclass()` now uses trained sigmoid probabilities
- Solver returns `alpha_sum` for probability fitting (one-vs-rest framework)

### Fixed

- Multiclass probability predictions now sum correctly to 1.0

## [0.3.0] - 2026-02-09

### Added

- Full SMO solver for all 5 SVM types: C-SVC, ν-SVC, one-class, ε-SVR, ν-SVR
- WSS3 working-set selection (second-order heuristic, Fan et al. JMLR 2005)
- Shrinking heuristic with gradient reconstruction
- `QMatrix` trait with `SvcQ`, `OneClassQ`, `SvrQ` implementations
- `svm_train` function producing `SvmModel` compatible with C LIBSVM
- Multiclass support via one-vs-one with class grouping and sv_coef assembly
- 50 tests (12 new), verified against C LIBSVM reference outputs

### Fixed

- `Cache::swap_index` — added column swap loop (critical for shrinking correctness)
- Kernel refactored to `Vec<&[SvmNode]>` for swappable data point references

## [0.2.0] - 2026-02-09

### Added

- Core types: `SvmNode`, `SvmProblem`, `SvmParameter`, `SvmModel`
- All 5 kernel functions (linear, polynomial, RBF, sigmoid, precomputed)
- LRU kernel cache
- Model and problem I/O (LIBSVM text format, byte-exact roundtrip)
- Prediction (zero mismatches against C `svm-predict` on heart_scale)
- Parameter validation with ν-SVC feasibility check
- 38 tests

[Unreleased]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ricardofrantz/libsvm-rs/commits/v0.2.0
