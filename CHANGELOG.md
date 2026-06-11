# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `SvmParameterBuilder` provides idiomatic construction-time validation for
  parameters, using the same checks as `svm_check_parameter`.
- Optional `serde` support for `SvmModel` and `SvmParameter`, with LIBSVM's
  integer enum codes preserved and the text model format remaining canonical.
- Optional `rayon` support for parallel cross-validation folds and parallel
  probability-estimation fold training.
- CI now verifies that the core library builds for `wasm32-unknown-unknown`.

### Changed

- MSRV raised from 1.75 to 1.80, required by rayon-core 1.13; previously the
  1.75 claim was unsatisfiable with the `rayon` feature.
- Linux probability estimation and probability cross-validation now replicate
  glibc `rand()` for the stratified shuffle, restoring C LIBSVM parity; Linux
  users of probability estimation or cross-validation should expect outputs to
  differ from 0.8.1 by design.
- The default repository branch is now `main`; downstream links, badges, clone
  commands, and automation that referenced `master` should be updated.
- Differential reference artifacts were re-baselined on macOS, and benchmark and
  integration demo artifacts were refreshed.
- Development tooling and CI pins were refreshed, including the Node 24 GitHub
  Actions runtime and current benchmark/dependency updates.

### Fixed

- The WebAssembly core-library CI job now installs the `wasm32-unknown-unknown`
  target for the pinned Rust toolchain before building.
- Benchmark CI now uses static, least-privilege permissions and the Criterion
  benchmark harness uses `std::hint::black_box` with current Criterion releases.

### Security

- Reject negative `gamma` for polynomial/RBF/sigmoid kernels at model load
  (text and serde paths); refreshed security audit covering the serde,
  rayon, builder, and PRNG surfaces (`SECURITY_AUDIT.md`).

### Documentation

- Added a migration guide for moving C LIBSVM and `libsvm-sys2` workflows to
  `libsvm-rs`.
- Added `VISION.md` and refreshed README, reference, and performance notes for
  the 0.9.0 release line.

## [0.8.1] - 2026-06-02

### Security

- Reject malformed classification models that omit the `nr_sv`/`label` lines.
  Such files previously loaded with empty vectors and then panicked (index out
  of bounds) during prediction; `validate_model_header` now requires
  `label.len() == nr_class` and `n_sv.len() == nr_class` for `c_svc`/`nu_svc`
  and returns a `ModelFormatError` instead.
- Guard the command-line tools against a bare `-` argument, which previously
  panicked (`svm-train-rs`, `svm-predict-rs`, `svm-scale-rs`); they now print
  usage and exit cleanly.
- Reject non-finite (`NaN`/`Inf`) values in model `gamma`, `coef0`, `rho`,
  support-vector coefficients, and feature values, and reject a negative
  `degree`, at load time.
- `cargo-deny` advisory policy now fails CI on yanked (`yanked = "deny"`) and
  unmaintained (`unmaintained = "all"`) crates.

### Changed

- Retain the upstream LIBSVM copyright (Chih-Chung Chang and Chih-Jen Lin)
  alongside the Rust port's. Restore the upstream BSD-3-Clause `COPYRIGHT` file
  for the vendored source and add a project `NOTICE` documenting provenance.

## [0.8.0] - 2026-04-22

### Added

- `LoadOptions` and explicit `load_*_from_reader_with_options` entrypoints for
  bounded problem/model loading.
- Fuzz corpus seeds and property tests for parser and model serialization
  stability.
- Supply-chain policy checks with `cargo-deny` and a pinned release
  reproduction guide.

### Changed

- Default problem/model loaders now treat input as untrusted and apply byte,
  line-length, support-vector, class-count, and feature-index caps.
- Root Rust toolchain is pinned to `1.93.1`; MSRV remains `1.75.0`.

### Security

- Hardened model parsing against header-driven allocation and inconsistent
  model metadata before support-vector storage is allocated.
- Model files now validate `rho`, `label`, `nr_sv`, probability metadata,
  precomputed-kernel rows, and support-vector feature ordering.
- Problem and model loaders reject over-limit input, oversized lines, embedded
  NUL bytes, malformed `index:value` tokens, non-ascending feature indices, and
  over-limit feature indices.
- Production library code now denies `unwrap`, `expect`, `panic!`, and
  `unreachable!` outside tests.
- No breaking API changes are expected; `LoadOptions` is additive and existing
  loader entrypoints delegate to `LoadOptions::default()`.

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

[Unreleased]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ricardofrantz/libsvm-rs/commits/v0.2.0
