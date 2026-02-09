//! # libsvm-rs
//!
//! A pure Rust reimplementation of [LIBSVM](https://github.com/cjlin1/libsvm),
//! targeting numerical equivalence and model-file compatibility with the
//! original C++ library.
//!
//! ## Status
//!
//! **Phases 0–3 complete**: types, I/O, kernels, cache, prediction, and full
//! SMO solver. Training works for all 5 SVM types (C-SVC, ν-SVC, one-class,
//! ε-SVR, ν-SVR). See [`train::svm_train`] for training and
//! [`predict::predict`] for inference.
//!
//! ## Feature Flags
//!
//! - `rayon` — Enable parallel cross-validation (off by default).

pub mod types;
pub mod error;
pub mod io;
pub mod kernel;
pub mod cache;
pub mod qmatrix;
pub mod solver;
pub mod train;

pub mod predict;

pub use error::SvmError;
pub use types::*;
