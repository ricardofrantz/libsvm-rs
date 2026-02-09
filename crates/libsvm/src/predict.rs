//! Prediction functions matching the original LIBSVM.
//!
//! Provides `predict` and `predict_values` for all SVM types:
//! - Classification (C-SVC, ν-SVC): one-vs-one voting
//! - One-class SVM: sign of decision value
//! - Regression (ε-SVR, ν-SVR): continuous output

use crate::kernel::k_function;
use crate::types::{SvmModel, SvmNode, SvmType};

/// Compute decision values and return the predicted label/value.
///
/// For classification, `dec_values` receives `nr_class * (nr_class - 1) / 2`
/// pairwise decision values. For regression/one-class, a single value.
///
/// Returns the predicted label (classification) or function value (regression).
///
/// Matches LIBSVM's `svm_predict_values`.
pub fn predict_values(model: &SvmModel, x: &[SvmNode], dec_values: &mut [f64]) -> f64 {
    match model.param.svm_type {
        SvmType::OneClass | SvmType::EpsilonSvr | SvmType::NuSvr => {
            let sv_coef = &model.sv_coef[0];
            let mut sum = 0.0;
            for (i, sv) in model.sv.iter().enumerate() {
                sum += sv_coef[i] * k_function(x, sv, &model.param);
            }
            sum -= model.rho[0];
            dec_values[0] = sum;

            if model.param.svm_type == SvmType::OneClass {
                if sum > 0.0 { 1.0 } else { -1.0 }
            } else {
                sum
            }
        }
        SvmType::CSvc | SvmType::NuSvc => {
            let nr_class = model.nr_class;
            let l = model.sv.len();

            // Compute kernel values for all SVs
            let kvalue: Vec<f64> = model
                .sv
                .iter()
                .map(|sv| k_function(x, sv, &model.param))
                .collect();

            // Compute start indices for each class's SVs
            let mut start = vec![0usize; nr_class];
            for i in 1..nr_class {
                start[i] = start[i - 1] + model.n_sv[i - 1];
            }

            // One-vs-one voting
            let mut vote = vec![0usize; nr_class];
            let mut p = 0;
            for i in 0..nr_class {
                for j in (i + 1)..nr_class {
                    let mut sum = 0.0;
                    let si = start[i];
                    let sj = start[j];
                    let ci = model.n_sv[i];
                    let cj = model.n_sv[j];

                    let coef1 = &model.sv_coef[j - 1];
                    let coef2 = &model.sv_coef[i];

                    for k in 0..ci {
                        sum += coef1[si + k] * kvalue[si + k];
                    }
                    for k in 0..cj {
                        sum += coef2[sj + k] * kvalue[sj + k];
                    }
                    sum -= model.rho[p];
                    dec_values[p] = sum;

                    if sum > 0.0 {
                        vote[i] += 1;
                    } else {
                        vote[j] += 1;
                    }
                    p += 1;
                }
            }

            // Find class with most votes
            let vote_max_idx = vote
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap_or(0);

            let _ = l; // suppress unused warning
            model.label[vote_max_idx] as f64
        }
    }
}

/// Predict the label/value for a single instance.
///
/// Convenience wrapper around `predict_values` that allocates the
/// decision values buffer internally. Matches LIBSVM's `svm_predict`.
pub fn predict(model: &SvmModel, x: &[SvmNode]) -> f64 {
    let n = match model.param.svm_type {
        SvmType::OneClass | SvmType::EpsilonSvr | SvmType::NuSvr => 1,
        SvmType::CSvc | SvmType::NuSvc => {
            model.nr_class * (model.nr_class - 1) / 2
        }
    };
    let mut dec_values = vec![0.0; n];
    predict_values(model, x, &mut dec_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_model;
    use crate::io::load_problem;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn predict_heart_scale() {
        // Load model trained by C LIBSVM and predict on training data
        let model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();

        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == problem.labels[i] {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / problem.labels.len() as f64;
        // C LIBSVM gets ~86.67% accuracy on training set with default params
        assert!(
            accuracy > 0.85,
            "accuracy {:.2}% too low (expected >85%)",
            accuracy * 100.0
        );
    }

    #[test]
    fn predict_values_binary() {
        let model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();

        // For binary classification, there's exactly 1 decision value
        let mut dec_values = vec![0.0; 1];
        let label = predict_values(&model, &problem.instances[0], &mut dec_values);

        // Decision value should be non-zero
        assert!(dec_values[0].abs() > 1e-10);
        // Label should match what predict returns
        assert_eq!(label, predict(&model, &problem.instances[0]));
    }

    #[test]
    fn predict_matches_c_svm_predict() {
        // Run C svm-predict and compare outputs
        // First, let's verify our predictions match by checking a few specific instances
        let model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();

        // Run C svm-predict to get reference predictions
        let c_predict = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("vendor")
            .join("libsvm")
            .join("svm-predict");

        if !c_predict.exists() {
            // Skip if C binary not compiled
            return;
        }

        // Write predictions to a temp file
        let output_path = data_dir().join("heart_scale.predict_test");
        let status = std::process::Command::new(&c_predict)
            .args([
                data_dir().join("heart_scale").to_str().unwrap(),
                data_dir().join("heart_scale.model").to_str().unwrap(),
                output_path.to_str().unwrap(),
            ])
            .output();

        if let Ok(output) = status {
            if output.status.success() {
                let c_preds: Vec<f64> = std::fs::read_to_string(&output_path)
                    .unwrap()
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(|l| l.trim().parse().unwrap())
                    .collect();

                assert_eq!(c_preds.len(), problem.labels.len());

                let mut mismatches = 0;
                for (i, instance) in problem.instances.iter().enumerate() {
                    let rust_pred = predict(&model, instance);
                    if rust_pred != c_preds[i] {
                        mismatches += 1;
                    }
                }

                assert_eq!(
                    mismatches, 0,
                    "{} predictions differ between Rust and C",
                    mismatches
                );

                // Clean up
                let _ = std::fs::remove_file(&output_path);
            }
        }
    }
}
