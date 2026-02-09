//! # libsvm-rs
//!
//! A pure Rust reimplementation of [LIBSVM](https://github.com/cjlin1/libsvm),
//! targeting numerical equivalence and model-file compatibility with the
//! original C++ library.
//!
//! ## Status
//!
//! **Early development** — core types and I/O are being implemented.
//! Training and prediction are not yet available.
//!
//! ## Feature Flags
//!
//! - `rayon` — Enable parallel cross-validation (off by default).

pub mod types;
pub mod error;
pub mod io;
pub mod kernel;
pub mod cache;

pub mod predict;

// Planned modules (not yet implemented):
// pub mod solver;

pub use error::SvmError;
pub use types::*;
