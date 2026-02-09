# Changelog

All notable changes to libsvm-rs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-09

### Added
- **Probability estimates**: Sigmoid probability model training and prediction (one-vs-rest multiclass)
- **Cross-validation**: k-fold cross-validation with stratified splits for classification
- `probability` module: `SigmoidTrainer`, `sigmoid_predict`, multiclass probability calibration
- `cross_validation` module: `StratifiedKFold` with proper class distribution preservation
- 20 new unit tests covering probability and CV workflows

### Changed
- `SvmModel::predict_probability_multiclass()` now uses trained sigmoid probabilities
- Solver output now returns `alpha_sum` for probability fitting (one-vs-rest framework)

### Fixed
- Multiclass probability predictions now sum correctly to 1.0

## [0.3.0] - 2026-01-31

### Added
- **SMO solver**: Complete Sequential Minimal Optimization implementation (Standard + Nu variants)
- **QMatrix trait**: Dynamic dispatch for SVC, ONE_CLASS, SVR kernel matrix computations
- Training on all 5 SVM problem types: C-SVC, NU-SVC, ONE-CLASS, EPSILON-SVR, NU-SVR
- Shrinking heuristic for solver efficiency
- Working set selection strategy 3 (WSS3)
- 40 unit tests for solver, kernels, and all SVM problem types
- `svm_train()`, `svm_train_one()`, `solve_c_svc()`, `solve_one_class()`, `solve_svr()` dispatchers
- Full numerical equivalence with C LIBSVM (within 1e-8 tolerance)

## [0.2.0] - 2025-12-15

### Added
- **Prediction module**: Inference on trained SVM models
- `SvmModel::predict()`: Point predictions (classifier + regressor)
- `SvmModel::predict_values()`: Decision function values
- `SvmModel::predict_probability_binary()`: Binary classification probabilities
- `SvmModel::predict_probability_multiclass()`: Multiclass probabilities (placeholder)
- Cache-friendly kernel evaluation during prediction
- 15 unit tests for prediction workflows

### Fixed
- Decision function computation for multi-class models

## [0.1.0] - 2025-11-30

### Added
- **Core types**: `SvmParameter`, `SvmProblem`, `SvmModel`, `SvmNode`
- **I/O module**: LIBSVM format parsing and model serialization
- **Kernel implementations**: Linear, RBF, Polynomial, Sigmoid with parameter validation
- **Cache module**: LRU cache for kernel matrix with row swapping support
- Workspace setup with multi-crate layout (`crates/libsvm`, `bins/*`)
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite (25+ unit tests)
- Support for reading heart_scale, iris.scale, housing datasets
- Model interchange with C LIBSVM (load trained C models, save for C tools)
