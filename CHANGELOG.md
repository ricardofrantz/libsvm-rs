# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.7.0] - 2026-02-12

### Added

- `metrics` module: `accuracy_percentage()`, `regression_metrics()` (public API)
- `util` module: `parse_feature_index()`, `MAX_FEATURE_INDEX` (public API)
- CLI integration tests — flag permutation and edge-case coverage
- Shared `cli_flag_helpers.rs` for property-based CLI testing

### Changed

- Deduplicated `group_classes` from train.rs + cross_validation.rs into util module
- Collapsed `parse_multiple_f64`/`parse_multiple_i32` into generic `parse_multiple<T>` in io.rs
- CLI arg parsing simplified via `parse_flag_arg()` helper in all 3 binaries
- svm-predict collects predictions into Vec before computing metrics (cleaner flow)

### Fixed

- `.tmp/` added to `.gitignore`

## [0.6.0] - 2026-02-11

### Added

- Testing and validation infrastructure:
  - Differential verification suite against upstream LIBSVM (250 test configurations)
  - Upstream lock file and CI validation (`reference/libsvm_upstream_lock.json`)
  - Deterministic synthetic dataset generation (6 families: binary, multiclass, one-class, regression, sparse, extreme scale)
  - Reference build pipeline with provenance tracking (`scripts/setup_reference_libsvm.sh`)
  - Coverage threshold checking with CI enforcement (93.19% line coverage, 92.86% function coverage)
  - Benchmark comparison framework (`scripts/benchmark_compare.py`)
  - Tolerance policy documentation (`reference/tolerance_policy.md`)
- Precomputed kernel support:
  - Full training and prediction support for precomputed kernels (kernel_type=4)
  - Validation and reference data for heart_scale, iris.scale, housing_scale
- CLI integration tests:
  - `svm-train-rs`: model file output, cross-validation, quiet mode
  - `svm-predict-rs`: prediction output, probability mode rejection for non-prob models, quiet mode
  - `svm-scale-rs`: scaling output, save/restore parameters, negative index hardening, inconsistent bounds checking
- Library enhancements:
  - Precomputed kernel evaluation path in `Kernel`
  - Extended probability module with NaN/Inf guards
  - Parameter validation for precomputed kernels
  - Helper functions for querying model properties by SVM type

### Changed

- README: comprehensive Phase 6 status update with verification metrics
- CI workflow: added upstream lock validation, coverage gates
- Solver output: additional stability checks for edge cases
- Probability estimation: improved numerical stability for one-class and SVR

### Fixed

- Model I/O: robust header parsing with oversized count guards (continued from v0.5.1)
- Kernel evaluation: correct precomputed kernel access and bounds checking
- Probability estimation: handle edge cases with insufficient or degenerate samples
- Scale CLI: prevent panic on negative feature indices

### Security

- Added SECURITY_AUDIT.md with RustSec audit results (zero findings)
- Hardened parsing against malicious inputs (oversized headers, negative indices)

## [0.5.1] - 2026-02-09

### Fixed

- Model loading: harden against oversized header counts that could cause memory allocation failures
- Cross-validation: fix probability prediction output and zero-fold edge case (now clamps to leave-one-out)
- Code quality: resolve 43 clippy warnings (collapsible_else_if, needless_range_loop, excessive_precision, field_reassign_with_default, manual_memcpy, etc.)

## [0.5.0] - 2026-02-09

### Added

- CLI tools: `svm-train-rs`, `svm-predict-rs`, `svm-scale-rs` matching C LIBSVM interface
- Quiet mode: `set_quiet(true)` suppresses all solver output; `-q` flag in all CLIs
- Reference comparison pipeline: scripts to generate and compare outputs against C LIBSVM
- Criterion benchmarks for training and prediction
- Property tests for determinism and label-range validation
- Fuzz targets for problem and model file parsing
- 4 examples: basic_train, predict_from_model, cross_validation, probability
- CI: build matrix (3 OS × 2 toolchains), MSRV check, Miri, security audit, coverage, benchmarks
- Release workflow with prebuilt binaries for Linux, macOS (x86+arm), Windows

### Changed

- Solver output formatting: `obj`, `rho`, `mae` now use `{:.6}` (matches C's `%f`)
- All internal `eprintln!` calls replaced with `info()` respecting quiet mode

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

[Unreleased]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ricardofrantz/libsvm-rs/commits/v0.2.0
