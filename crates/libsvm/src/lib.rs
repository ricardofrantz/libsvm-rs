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
//! ## Feature Flags
//!
//! - `rayon` — Enable parallel cross-validation (off by default).

use std::sync::atomic::{AtomicBool, Ordering};

static QUIET_MODE: AtomicBool = AtomicBool::new(false);

/// Enable or disable quiet mode. When quiet, solver diagnostic messages
/// are suppressed (equivalent to LIBSVM's `-q` flag).
pub fn set_quiet(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::Relaxed);
}

/// Print an info message to stderr (suppressed in quiet mode).
pub(crate) fn info(msg: &str) {
    if !QUIET_MODE.load(Ordering::Relaxed) {
        eprint!("{}", msg);
    }
}

pub mod cache;
pub mod error;
pub mod io;
pub mod kernel;
pub mod qmatrix;
pub mod solver;
pub mod train;
pub mod types;

pub mod cross_validation;
pub mod predict;
pub mod probability;

pub use error::SvmError;
pub use types::*;
