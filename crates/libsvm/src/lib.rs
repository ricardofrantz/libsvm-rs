//! # libsvm-rs
//!
//! A pure Rust reimplementation of [LIBSVM](https://github.com/cjlin1/libsvm),
//! targeting numerical equivalence and model-file compatibility with the
//! original C++ library.
//!
//! ## Status
//!
//! **Phases 0–4 complete**: types, I/O, kernels, cache, prediction, full
//! SMO solver, probability estimates (Platt scaling), and cross-validation.
//! Training works for all 5 SVM types (C-SVC, ν-SVC, one-class, ε-SVR,
//! ν-SVR). See [`train::svm_train`] for training, [`predict::predict`]
//! for inference, and [`predict::predict_probability`] for probabilistic
//! outputs.
//!
//! ## Trust Boundary
//!
//! Problem and model files are treated as untrusted text input by default.
//! The [`io`] loaders apply [`LoadOptions`] caps, reject malformed sparse
//! feature rows, and validate model-header consistency before allocating
//! support-vector storage. These checks bound parsing work and memory use; they
//! do not authenticate a model or prove that it is appropriate for a particular
//! deployment.
//!
//! ## Feature Flags
//!
//! - `rayon` — Enable parallel cross-validation (off by default). Fold
//!   assignment remains serial and deterministic, then each fold trains on an
//!   independent worker. Per-fold training diagnostics are suppressed while the
//!   parallel workers run so output cannot interleave; use the default serial
//!   path if you need fold-internal progress text. With `k` parallel folds, peak
//!   memory can include up to `min(k, rayon_threads)` simultaneous kernel caches
//!   of `SvmParameter::cache_size` each; the cache size is never divided
//!   implicitly.
//! - `serde` — Enable `Serialize`/`Deserialize` for model and parameter
//!   types. `SvmType` and `KernelType` serialize as pinned LIBSVM integer
//!   codes (`0..4`). Deserializing `SvmModel` runs the same structural
//!   validation as the text model loader; LIBSVM text model files remain the
//!   C-compatible interchange format.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static QUIET_MODE: AtomicBool = AtomicBool::new(false);
static SUPPRESS_INFO_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Enable or disable quiet mode. When quiet, solver diagnostic messages
/// are suppressed (equivalent to LIBSVM's `-q` flag).
pub fn set_quiet(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::Relaxed);
}

/// Print an info message to stderr (suppressed in quiet mode).
pub(crate) fn info(msg: &str) {
    if !QUIET_MODE.load(Ordering::Relaxed) && SUPPRESS_INFO_DEPTH.load(Ordering::Relaxed) == 0 {
        eprint!("{}", msg);
    }
}

#[cfg(feature = "rayon")]
pub(crate) fn with_suppressed_info<T>(f: impl FnOnce() -> T) -> T {
    struct SuppressInfoGuard;

    impl Drop for SuppressInfoGuard {
        fn drop(&mut self) {
            SUPPRESS_INFO_DEPTH.fetch_sub(1, Ordering::Relaxed);
        }
    }

    SUPPRESS_INFO_DEPTH.fetch_add(1, Ordering::Relaxed);
    let _guard = SuppressInfoGuard;
    f()
}

pub mod builder;
pub mod cache;
pub mod error;
pub mod io;
pub mod kernel;
pub mod metrics;
pub mod qmatrix;
pub mod solver;
pub mod train;
pub mod types;
pub mod util;

pub mod cross_validation;
pub mod predict;
pub mod probability;

pub use builder::SvmParameterBuilder;
pub use error::SvmError;
pub use io::LoadOptions;
pub use metrics::{accuracy_percentage, regression_metrics};
pub use types::*;
