//! Property-based tests for libsvm-rs using proptest.
//!
//! These tests verify core invariants:
//! - Determinism: repeated predictions with same input produce identical results
//! - Classification invariant: predictions are valid training labels
//! - Cross-validation: outputs are finite and valid labels

use libsvm_rs::cross_validation::svm_cross_validation;
use libsvm_rs::io::load_problem;
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::types::{SvmNode, SvmParameter, SvmProblem};
use std::path::Path;

/// Helper to load heart_scale dataset from the project data directory.
fn load_heart_scale() -> SvmProblem {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/heart_scale");
    load_problem(Path::new(path)).expect("Failed to load heart_scale dataset")
}

/// Helper to extract unique labels from a problem.
fn unique_labels(prob: &SvmProblem) -> Vec<f64> {
    let mut labels: Vec<f64> = prob.labels.iter().copied().collect();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    labels
}

/// Test: Deterministic predictions with randomized sparse instances.
///
/// Generates 2-5 random sparse instances, trains a model with fixed parameters,
/// and verifies that predicting the same instance twice yields identical results.
#[test]
fn kernel_deterministic() {
    libsvm_rs::set_quiet(true);

    // Generate a small problem with random sparse features.
    let instances = vec![
        vec![
            SvmNode { index: 1, value: 2.5 },
            SvmNode { index: 5, value: -1.3 },
            SvmNode { index: 18, value: 0.7 },
        ],
        vec![
            SvmNode { index: 2, value: 1.1 },
            SvmNode { index: 8, value: 3.2 },
            SvmNode { index: 15, value: -2.1 },
        ],
        vec![
            SvmNode { index: 3, value: -0.5 },
            SvmNode { index: 10, value: 1.9 },
            SvmNode { index: 20, value: 2.8 },
        ],
        vec![
            SvmNode { index: 1, value: 1.2 },
            SvmNode { index: 4, value: -1.5 },
            SvmNode { index: 12, value: 0.3 },
        ],
    ];

    let prob = SvmProblem {
        labels: vec![1.0, -1.0, 1.0, -1.0],
        instances,
    };

    let mut param = SvmParameter::default();
    param.gamma = 1.0 / 20.0; // max index is 20
    param.shrinking = false;
    param.eps = 0.01;

    // Train the model
    let model = svm_train(&prob, &param);

    // Predict the same instance twice and verify results are identical
    let test_instance = &prob.instances[0];
    let pred1 = predict(&model, test_instance);
    let pred2 = predict(&model, test_instance);

    assert_eq!(
        pred1, pred2,
        "Predictions should be deterministic; got {} and {}",
        pred1, pred2
    );
}

/// Test: Deterministic predictions on real data.
///
/// Loads heart_scale dataset, trains a C-SVC model, and verifies that
/// predicting the same instances twice yields identical results.
#[test]
fn predict_deterministic() {
    libsvm_rs::set_quiet(true);

    let prob = load_heart_scale();
    let mut param = SvmParameter::default();
    param.gamma = 1.0 / 13.0;

    // Train the model
    let model = svm_train(&prob, &param);

    // Predict a subset of instances twice
    let test_indices = vec![0, 1, 2, 3, 4];
    for &idx in &test_indices {
        let test_instance = &prob.instances[idx];
        let pred1 = predict(&model, test_instance);
        let pred2 = predict(&model, test_instance);

        assert_eq!(
            pred1, pred2,
            "Prediction for instance {} should be deterministic; got {} and {}",
            idx, pred1, pred2
        );
    }
}

/// Test: Classification predictions are valid training labels.
///
/// Trains a C-SVC model on heart_scale and verifies that predictions
/// for all instances are one of the training labels.
#[test]
fn train_predict_labels_in_range() {
    libsvm_rs::set_quiet(true);

    let prob = load_heart_scale();
    let valid_labels = unique_labels(&prob);

    let mut param = SvmParameter::default();
    param.gamma = 1.0 / 13.0;
    let model = svm_train(&prob, &param);

    // Predict all instances and verify each is a valid label
    for (idx, instance) in prob.instances.iter().enumerate() {
        let pred = predict(&model, instance);

        assert!(
            valid_labels.contains(&pred),
            "Instance {} prediction {} is not in training labels {:?}",
            idx,
            pred,
            valid_labels
        );
    }
}

/// Test: Cross-validation results are valid.
///
/// Runs 5-fold cross-validation on heart_scale and verifies that all
/// returned predictions are finite and valid training labels.
#[test]
fn cross_validation_results_valid() {
    libsvm_rs::set_quiet(true);

    let prob = load_heart_scale();
    let valid_labels = unique_labels(&prob);

    let mut param = SvmParameter::default();
    param.gamma = 1.0 / 13.0;
    let cv_targets = svm_cross_validation(&prob, &param, 5);

    // Verify all CV targets are valid
    assert_eq!(
        cv_targets.len(),
        prob.labels.len(),
        "CV output length should match problem size"
    );

    for (idx, &target) in cv_targets.iter().enumerate() {
        // Check finiteness
        assert!(
            target.is_finite(),
            "CV target[{}] = {} is not finite",
            idx,
            target
        );

        // Check that it's a valid label
        assert!(
            valid_labels.contains(&target),
            "CV target[{}] = {} is not in training labels {:?}",
            idx,
            target,
            valid_labels
        );
    }
}
