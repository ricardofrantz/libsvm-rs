//! Cross-validation for SVM models.
//!
//! Provides stratified cross-validation for classification and simple
//! random-split cross-validation for regression/one-class problems.
//! Matches LIBSVM's `svm_cross_validation` (svm.cpp:2437–2556).

use crate::predict::{predict, predict_probability};
use crate::train::svm_train;
use crate::types::{SvmModel, SvmNode, SvmParameter, SvmProblem, SvmType};

// ─── RNG helper ──────────────────────────────────────────────────────

fn rng_next(state: &mut u64) -> usize {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as usize
}

fn predict_cv_target(model: &SvmModel, param: &SvmParameter, x: &[SvmNode]) -> f64 {
    if param.probability && matches!(param.svm_type, SvmType::CSvc | SvmType::NuSvc) {
        predict_probability(model, x)
            .map(|(label, _)| label)
            .unwrap_or_else(|| predict(model, x))
    } else {
        predict(model, x)
    }
}

// ─── Public API ──────────────────────────────────────────────────────

/// Perform k-fold cross-validation on an SVM problem.
///
/// Returns a `Vec<f64>` of length `prob.labels.len()` where `target[i]`
/// is the prediction for instance `i` when it was held out.
///
/// - **Classification** (C-SVC, ν-SVC) with `nr_fold < l`: stratified
///   splitting that preserves class ratios across folds.
/// - **Regression / one-class** or `nr_fold == l`: simple random split.
///
/// If `nr_fold > l`, clamps to `l` (leave-one-out).
pub fn svm_cross_validation(
    prob: &SvmProblem,
    param: &SvmParameter,
    mut nr_fold: usize,
) -> Vec<f64> {
    let l = prob.labels.len();

    if l == 0 {
        return Vec::new();
    }

    if nr_fold == 0 {
        crate::info(
            "WARNING: # folds (0) <= 0. Will use # folds = # data instead \
             (i.e., leave-one-out cross validation)\n",
        );
        nr_fold = l;
    }

    if nr_fold > l {
        crate::info(&format!(
            "WARNING: # folds ({}) > # data ({}). Will use # folds = # data instead \
             (i.e., leave-one-out cross validation)\n",
            nr_fold, l
        ));
        nr_fold = l;
    }

    let mut rng: u64 = 1;
    let mut perm: Vec<usize> = (0..l).collect();
    let mut fold_start = vec![0usize; nr_fold + 1];

    // ── Fold assignment ──────────────────────────────────────────
    if matches!(param.svm_type, SvmType::CSvc | SvmType::NuSvc) && nr_fold < l {
        // Stratified: group by class, shuffle within class, distribute
        let (label_list, start, count, group_perm) = group_classes(&prob.labels);
        let nr_class = label_list.len();

        let mut index = group_perm;

        // Shuffle within each class
        for c in 0..nr_class {
            let s = start[c];
            let n = count[c];
            for i in 0..n {
                let j = i + rng_next(&mut rng) % (n - i);
                index.swap(s + i, s + j);
            }
        }

        // Compute fold sizes
        let mut fold_count = vec![0usize; nr_fold];
        for i in 0..nr_fold {
            for c in 0..nr_class {
                fold_count[i] += ((i + 1) * count[c]) / nr_fold - (i * count[c]) / nr_fold;
            }
        }

        fold_start[0] = 0;
        for i in 1..=nr_fold {
            fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
        }

        // Distribute samples to folds, preserving class balance
        // (C++ increments fold_start[i] as a running pointer; we
        // use a separate offset array to keep fold_start immutable.)
        let mut offset = vec![0usize; nr_fold];
        for c in 0..nr_class {
            for i in 0..nr_fold {
                let begin = start[c] + (i * count[c]) / nr_fold;
                let end = start[c] + ((i + 1) * count[c]) / nr_fold;
                for j in begin..end {
                    perm[fold_start[i] + offset[i]] = index[j];
                    offset[i] += 1;
                }
            }
        }

        // Rebuild fold_start from fold_count
        fold_start[0] = 0;
        for i in 1..=nr_fold {
            fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
        }
    } else {
        // Simple random shuffle
        for i in 0..l {
            let j = i + rng_next(&mut rng) % (l - i);
            perm.swap(i, j);
        }
        for i in 0..=nr_fold {
            fold_start[i] = i * l / nr_fold;
        }
    }

    // ── Evaluate each fold ───────────────────────────────────────
    let mut target = vec![0.0; l];

    for i in 0..nr_fold {
        let begin = fold_start[i];
        let end = fold_start[i + 1];

        // Build sub-problem excluding held-out [begin..end)
        let sub_l = l - (end - begin);
        let mut sub_labels = Vec::with_capacity(sub_l);
        let mut sub_instances = Vec::with_capacity(sub_l);

        for j in 0..begin {
            sub_labels.push(prob.labels[perm[j]]);
            sub_instances.push(prob.instances[perm[j]].clone());
        }
        for j in end..l {
            sub_labels.push(prob.labels[perm[j]]);
            sub_instances.push(prob.instances[perm[j]].clone());
        }

        let subprob = SvmProblem {
            labels: sub_labels,
            instances: sub_instances,
        };
        let submodel = svm_train(&subprob, param);

        // Predict held-out
        for j in begin..end {
            target[perm[j]] = predict_cv_target(&submodel, param, &prob.instances[perm[j]]);
        }
    }

    target
}

// ─── Internal helpers ────────────────────────────────────────────────

/// Group samples by class label (same logic as `train::svm_group_classes`).
fn group_classes(labels: &[f64]) -> (Vec<i32>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let l = labels.len();
    let mut label_list: Vec<i32> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    let mut data_label = vec![0usize; l];

    for i in 0..l {
        let this_label = labels[i] as i32;
        if let Some(pos) = label_list.iter().position(|&lab| lab == this_label) {
            count[pos] += 1;
            data_label[i] = pos;
        } else {
            data_label[i] = label_list.len();
            label_list.push(this_label);
            count.push(1);
        }
    }

    let nr_class = label_list.len();

    // For binary with -1/+1 where -1 appears first, swap to put +1 first
    if nr_class == 2 && label_list[0] == -1 && label_list[1] == 1 {
        label_list.swap(0, 1);
        count.swap(0, 1);
        for dl in data_label.iter_mut() {
            *dl = if *dl == 0 { 1 } else { 0 };
        }
    }

    let mut start = vec![0usize; nr_class];
    for i in 1..nr_class {
        start[i] = start[i - 1] + count[i - 1];
    }

    let mut perm = vec![0usize; l];
    let mut start_copy = start.clone();
    for i in 0..l {
        let cls = data_label[i];
        perm[start_copy[cls]] = i;
        start_copy[cls] += 1;
    }

    (label_list, start, count, perm)
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_problem;
    use crate::types::{KernelType, SvmModel, SvmNode};
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn cross_validation_basic() {
        let labels = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let instances: Vec<Vec<SvmNode>> = (0..10)
            .map(|i| vec![SvmNode { index: 1, value: i as f64 * 0.1 }])
            .collect();

        let prob = SvmProblem { labels, instances };
        let param = SvmParameter {
            kernel_type: KernelType::Linear,
            ..Default::default()
        };

        let target = svm_cross_validation(&prob, &param, 5);
        assert_eq!(target.len(), 10);
        for &pred in &target {
            assert!(pred == 1.0 || pred == -1.0);
        }
    }

    #[test]
    fn cross_validation_classification() {
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let target = svm_cross_validation(&problem, &param, 5);
        assert_eq!(target.len(), problem.labels.len());

        let correct = target
            .iter()
            .zip(problem.labels.iter())
            .filter(|(&pred, &label)| pred == label)
            .count();
        let accuracy = correct as f64 / problem.labels.len() as f64;
        assert!(
            accuracy > 0.70,
            "5-fold CV accuracy {:.1}% too low (expected >70%)",
            accuracy * 100.0
        );
    }

    #[test]
    fn cross_validation_regression() {
        let problem = load_problem(&data_dir().join("housing_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            p: 0.1,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let target = svm_cross_validation(&problem, &param, 5);
        assert_eq!(target.len(), problem.labels.len());

        let mse: f64 = target
            .iter()
            .zip(problem.labels.iter())
            .map(|(&pred, &label)| (pred - label).powi(2))
            .sum::<f64>()
            / problem.labels.len() as f64;
        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 500.0, "MSE {} too high", mse);
    }

    #[test]
    fn cross_validation_zero_folds_clamps_to_leave_one_out() {
        let labels = vec![1.0, -1.0, 1.0, -1.0];
        let instances: Vec<Vec<SvmNode>> = vec![
            vec![SvmNode {
                index: 1,
                value: 1.0,
            }],
            vec![SvmNode {
                index: 1,
                value: -1.0,
            }],
            vec![SvmNode {
                index: 1,
                value: 0.8,
            }],
            vec![SvmNode {
                index: 1,
                value: -0.9,
            }],
        ];
        let prob = SvmProblem { labels, instances };
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Linear,
            c: 1.0,
            eps: 0.001,
            ..Default::default()
        };

        let target = svm_cross_validation(&prob, &param, 0);
        assert_eq!(target.len(), prob.labels.len());
        for &pred in &target {
            assert!(pred == 1.0 || pred == -1.0);
        }
    }

    #[test]
    fn cross_validation_empty_problem_returns_empty() {
        let prob = SvmProblem {
            labels: Vec::new(),
            instances: Vec::new(),
        };
        let target = svm_cross_validation(&prob, &SvmParameter::default(), 5);
        assert!(target.is_empty());
    }

    #[test]
    fn predict_cv_target_uses_probability_label_for_classification() {
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Linear,
            probability: true,
            ..Default::default()
        };
        let model = SvmModel {
            param: param.clone(),
            nr_class: 2,
            sv: vec![
                vec![SvmNode {
                    index: 1,
                    value: 1.0,
                }],
                vec![SvmNode {
                    index: 1,
                    value: 1.0,
                }],
            ],
            sv_coef: vec![vec![1.0, -1.0]],
            rho: vec![-1.0],
            prob_a: vec![1.0],
            prob_b: vec![0.0],
            prob_density_marks: Vec::new(),
            sv_indices: vec![1, 2],
            label: vec![1, -1],
            n_sv: vec![1, 1],
        };
        let x = vec![SvmNode {
            index: 1,
            value: 1.0,
        }];

        let vote_label = predict(&model, &x);
        let (prob_label, _) = predict_probability(&model, &x).unwrap();

        assert_eq!(vote_label, 1.0);
        assert_eq!(prob_label, -1.0);
        assert_eq!(predict_cv_target(&model, &param, &x), prob_label);
    }
}
