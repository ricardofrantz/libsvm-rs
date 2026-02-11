/// Type of SVM formulation.
///
/// Matches the integer constants in the original LIBSVM (`svm.h`):
/// `C_SVC=0, NU_SVC=1, ONE_CLASS=2, EPSILON_SVR=3, NU_SVR=4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SvmType {
    /// C-Support Vector Classification.
    CSvc = 0,
    /// ν-Support Vector Classification.
    NuSvc = 1,
    /// One-class SVM (distribution estimation / novelty detection).
    OneClass = 2,
    /// ε-Support Vector Regression.
    EpsilonSvr = 3,
    /// ν-Support Vector Regression.
    NuSvr = 4,
}

/// Type of kernel function.
///
/// Matches the integer constants in the original LIBSVM (`svm.h`):
/// `LINEAR=0, POLY=1, RBF=2, SIGMOID=3, PRECOMPUTED=4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum KernelType {
    /// `K(x,y) = x·y`
    Linear = 0,
    /// `K(x,y) = (γ·x·y + coef0)^degree`
    Polynomial = 1,
    /// `K(x,y) = exp(-γ·‖x-y‖²)`
    Rbf = 2,
    /// `K(x,y) = tanh(γ·x·y + coef0)`
    Sigmoid = 3,
    /// Kernel values supplied as a precomputed matrix.
    Precomputed = 4,
}

/// A single sparse feature: `index:value`.
///
/// In the original LIBSVM, a sentinel node with `index = -1` marks the end
/// of each instance. In this Rust port, instance length is tracked by
/// `Vec::len()` instead, so no sentinel is needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvmNode {
    /// 1-based feature index. Uses `i32` to match the original C `int` and
    /// preserve file-format compatibility.
    pub index: i32,
    /// Feature value.
    pub value: f64,
}

/// A training/test problem: a collection of labelled sparse instances.
#[derive(Debug, Clone, PartialEq)]
pub struct SvmProblem {
    /// Label (class for classification, target for regression) per instance.
    pub labels: Vec<f64>,
    /// Sparse feature vectors, one per instance.
    pub instances: Vec<Vec<SvmNode>>,
}

/// SVM parameters controlling the formulation, kernel, and solver.
///
/// Default values match the original LIBSVM defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SvmParameter {
    /// SVM formulation type.
    pub svm_type: SvmType,
    /// Kernel function type.
    pub kernel_type: KernelType,
    /// Degree for polynomial kernel.
    pub degree: i32,
    /// γ parameter for RBF, polynomial, and sigmoid kernels.
    /// Set to `1/num_features` when 0.
    pub gamma: f64,
    /// Independent term in polynomial and sigmoid kernels.
    pub coef0: f64,
    /// Cache memory size in MB.
    pub cache_size: f64,
    /// Stopping tolerance for the solver.
    pub eps: f64,
    /// Cost parameter C (for C-SVC, ε-SVR, ν-SVR).
    pub c: f64,
    /// Per-class weight overrides: `(class_label, weight)` pairs.
    pub weight: Vec<(i32, f64)>,
    /// ν parameter (for ν-SVC, one-class SVM, ν-SVR).
    pub nu: f64,
    /// ε in the ε-insensitive loss function (ε-SVR).
    pub p: f64,
    /// Whether to use the shrinking heuristic.
    pub shrinking: bool,
    /// Whether to train for probability estimates.
    pub probability: bool,
}

impl Default for SvmParameter {
    fn default() -> Self {
        Self {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            degree: 3,
            gamma: 0.0, // means 1/num_features
            coef0: 0.0,
            cache_size: 100.0,
            eps: 0.001,
            c: 1.0,
            weight: Vec::new(),
            nu: 0.5,
            p: 0.1,
            shrinking: true,
            probability: false,
        }
    }
}

impl SvmParameter {
    /// Validate parameter values (independent of training data).
    ///
    /// This checks the same constraints as the original LIBSVM's
    /// `svm_check_parameter`, except for the ν-SVC feasibility check
    /// which requires the problem. Use [`check_parameter`] for the full check.
    pub fn validate(&self) -> Result<(), crate::error::SvmError> {
        use crate::error::SvmError;

        // gamma must be non-negative for kernels that use it
        if matches!(
            self.kernel_type,
            KernelType::Polynomial | KernelType::Rbf | KernelType::Sigmoid
        ) && self.gamma < 0.0
        {
            return Err(SvmError::InvalidParameter("gamma < 0".into()));
        }

        // polynomial degree must be non-negative
        if self.kernel_type == KernelType::Polynomial && self.degree < 0 {
            return Err(SvmError::InvalidParameter(
                "degree of polynomial kernel < 0".into(),
            ));
        }

        if self.cache_size <= 0.0 {
            return Err(SvmError::InvalidParameter("cache_size <= 0".into()));
        }

        if self.eps <= 0.0 {
            return Err(SvmError::InvalidParameter("eps <= 0".into()));
        }

        // C > 0 for formulations that use it
        if matches!(
            self.svm_type,
            SvmType::CSvc | SvmType::EpsilonSvr | SvmType::NuSvr
        ) && self.c <= 0.0
        {
            return Err(SvmError::InvalidParameter("C <= 0".into()));
        }

        // nu ∈ (0, 1] for formulations that use it
        if matches!(
            self.svm_type,
            SvmType::NuSvc | SvmType::OneClass | SvmType::NuSvr
        ) && (self.nu <= 0.0 || self.nu > 1.0)
        {
            return Err(SvmError::InvalidParameter("nu <= 0 or nu > 1".into()));
        }

        // p >= 0 for epsilon-SVR
        if self.svm_type == SvmType::EpsilonSvr && self.p < 0.0 {
            return Err(SvmError::InvalidParameter("p < 0".into()));
        }

        Ok(())
    }
}

/// Full parameter check including ν-SVC feasibility against training data.
///
/// Matches the original LIBSVM `svm_check_parameter()`.
pub fn check_parameter(
    problem: &SvmProblem,
    param: &SvmParameter,
) -> Result<(), crate::error::SvmError> {
    use crate::error::SvmError;

    // Run the data-independent checks first
    param.validate()?;

    // ν-SVC feasibility: for every pair of classes (i, j),
    // nu * (count_i + count_j) / 2 must be <= min(count_i, count_j)
    //
    // Note: LIBSVM casts labels to int for class grouping. We match this
    // behavior. Classification labels must be integers (non-integer labels
    // will be truncated, matching `(int)prob->y[i]` in the C code).
    if param.svm_type == SvmType::NuSvc {
        let mut class_counts: Vec<(i32, usize)> = Vec::new();
        for &y in &problem.labels {
            let label = y as i32;
            if let Some(entry) = class_counts.iter_mut().find(|(l, _)| *l == label) {
                entry.1 += 1;
            } else {
                class_counts.push((label, 1));
            }
        }

        for (i, &(_, n1)) in class_counts.iter().enumerate() {
            for &(_, n2) in &class_counts[i + 1..] {
                if param.nu * (n1 + n2) as f64 / 2.0 > n1.min(n2) as f64 {
                    return Err(SvmError::InvalidParameter(
                        "specified nu is infeasible".into(),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// A trained SVM model.
///
/// Produced by training, or loaded from a LIBSVM model file.
#[derive(Debug, Clone, PartialEq)]
pub struct SvmModel {
    /// Parameters used during training.
    pub param: SvmParameter,
    /// Number of classes (2 for binary, >2 for multiclass, 2 for regression).
    pub nr_class: usize,
    /// Support vectors (sparse feature vectors).
    pub sv: Vec<Vec<SvmNode>>,
    /// Support vector coefficients. For k classes, this is a
    /// `(k-1) × num_sv` matrix stored as `Vec<Vec<f64>>`.
    pub sv_coef: Vec<Vec<f64>>,
    /// Bias terms (rho). One per class pair: `k*(k-1)/2` values.
    pub rho: Vec<f64>,
    /// Pairwise probability parameter A (Platt scaling). Empty if not trained
    /// with probability estimates.
    pub prob_a: Vec<f64>,
    /// Pairwise probability parameter B (Platt scaling). Empty if not trained
    /// with probability estimates.
    pub prob_b: Vec<f64>,
    /// Probability density marks (for one-class SVM).
    pub prob_density_marks: Vec<f64>,
    /// Original indices of support vectors in the training set (1-based).
    pub sv_indices: Vec<usize>,
    /// Class labels (in the order used internally).
    pub label: Vec<i32>,
    /// Number of support vectors per class.
    pub n_sv: Vec<usize>,
}

impl SvmModel {
    /// Return the SVM type used by the model.
    pub fn svm_type(&self) -> SvmType {
        self.param.svm_type
    }

    /// Return number of classes.
    pub fn class_count(&self) -> usize {
        self.nr_class
    }

    /// Return class labels in internal one-vs-one order.
    pub fn labels(&self) -> &[i32] {
        &self.label
    }

    /// Return original 1-based support-vector indices.
    pub fn support_vector_indices(&self) -> &[usize] {
        &self.sv_indices
    }

    /// Return total number of support vectors.
    pub fn support_vector_count(&self) -> usize {
        self.sv.len()
    }

    /// Return SVR sigma when a probability-capable SVR model is available.
    pub fn svr_probability(&self) -> Option<f64> {
        match self.param.svm_type {
            SvmType::EpsilonSvr | SvmType::NuSvr => self.prob_a.first().copied(),
            _ => None,
        }
    }

    /// Check whether the model contains probability metadata.
    pub fn has_probability_model(&self) -> bool {
        match self.param.svm_type {
            SvmType::CSvc | SvmType::NuSvc => !self.prob_a.is_empty() && !self.prob_b.is_empty(),
            SvmType::EpsilonSvr | SvmType::NuSvr => !self.prob_a.is_empty(),
            SvmType::OneClass => !self.prob_density_marks.is_empty(),
        }
    }
}

/// C-API style helper matching LIBSVM's `svm_get_svm_type`.
pub fn svm_get_svm_type(model: &SvmModel) -> SvmType {
    model.svm_type()
}

/// C-API style helper matching LIBSVM's `svm_get_nr_class`.
pub fn svm_get_nr_class(model: &SvmModel) -> usize {
    model.class_count()
}

/// C-API style helper matching LIBSVM's `svm_get_labels`.
pub fn svm_get_labels(model: &SvmModel) -> &[i32] {
    model.labels()
}

/// C-API style helper matching LIBSVM's `svm_get_sv_indices`.
pub fn svm_get_sv_indices(model: &SvmModel) -> &[usize] {
    model.support_vector_indices()
}

/// C-API style helper matching LIBSVM's `svm_get_nr_sv`.
pub fn svm_get_nr_sv(model: &SvmModel) -> usize {
    model.support_vector_count()
}

/// C-API style helper matching LIBSVM's `svm_get_svr_probability`.
pub fn svm_get_svr_probability(model: &SvmModel) -> Option<f64> {
    model.svr_probability()
}

/// C-API style helper matching LIBSVM's `svm_check_probability_model`.
pub fn svm_check_probability_model(model: &SvmModel) -> bool {
    model.has_probability_model()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::svm_train;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn default_params_are_valid() {
        SvmParameter::default().validate().unwrap();
    }

    #[test]
    fn negative_gamma_rejected() {
        let p = SvmParameter {
            gamma: -1.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn zero_cache_rejected() {
        let p = SvmParameter {
            cache_size: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn zero_c_rejected() {
        let p = SvmParameter {
            c: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn nu_out_of_range_rejected() {
        let p = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 1.5,
            ..Default::default()
        };
        assert!(p.validate().is_err());

        let p2 = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.0,
            ..Default::default()
        };
        assert!(p2.validate().is_err());
    }

    #[test]
    fn negative_p_rejected_for_svr() {
        let p = SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            p: -0.1,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn negative_poly_degree_rejected() {
        let p = SvmParameter {
            kernel_type: KernelType::Polynomial,
            degree: -1,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn nu_svc_feasibility_check() {
        // 2 classes with 3 samples each: nu * (3+3)/2 <= 3  →  nu <= 1
        let problem = SvmProblem {
            labels: vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            instances: vec![vec![]; 6],
        };
        let ok_param = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.5,
            ..Default::default()
        };
        check_parameter(&problem, &ok_param).unwrap();

        // nu = 0.9: 0.9 * 6/2 = 2.7 <= 3 → feasible
        let borderline = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.9,
            ..Default::default()
        };
        check_parameter(&problem, &borderline).unwrap();
    }

    #[test]
    fn nu_svc_infeasible() {
        // 5 class-A, 1 class-B: nu*(5+1)/2 > min(5,1)=1  →  nu > 1/3
        let problem = SvmProblem {
            labels: vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            instances: vec![vec![]; 6],
        };
        let param = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.5, // 0.5 * 6/2 = 1.5 > 1
            ..Default::default()
        };
        let err = check_parameter(&problem, &param);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("infeasible"));
    }

    #[test]
    fn c_api_style_model_helpers() {
        let problem = crate::io::load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            gamma: 1.0 / 13.0,
            ..Default::default()
        };
        let model = svm_train(&problem, &param);

        assert_eq!(svm_get_svm_type(&model), SvmType::CSvc);
        assert_eq!(svm_get_nr_class(&model), 2);
        assert_eq!(svm_get_nr_sv(&model), model.sv.len());
        assert_eq!(svm_get_labels(&model), model.label.as_slice());
        assert_eq!(svm_get_sv_indices(&model), model.sv_indices.as_slice());
        assert!(!svm_check_probability_model(&model));
        assert_eq!(svm_get_svr_probability(&model), None);
    }

    #[test]
    fn probability_helpers_by_svm_type() {
        let svm = vec![SvmNode {
            index: 1,
            value: 1.0,
        }];

        let csvc_model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::CSvc,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![svm.clone()],
            sv_coef: vec![vec![1.0]],
            rho: vec![0.0],
            prob_a: vec![1.0],
            prob_b: vec![-0.5],
            prob_density_marks: vec![],
            sv_indices: vec![1],
            label: vec![1, -1],
            n_sv: vec![1, 0],
        };
        assert!(csvc_model.has_probability_model());
        assert!(svm_check_probability_model(&csvc_model));
        assert_eq!(svm_get_svr_probability(&csvc_model), None);

        let eps_svr_model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::EpsilonSvr,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![svm.clone()],
            sv_coef: vec![vec![0.8]],
            rho: vec![0.0],
            prob_a: vec![0.123],
            prob_b: vec![],
            prob_density_marks: vec![],
            sv_indices: vec![1],
            label: vec![],
            n_sv: vec![],
        };
        assert!(eps_svr_model.has_probability_model());
        assert_eq!(svm_get_svr_probability(&eps_svr_model), Some(0.123));

        let one_class_model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::OneClass,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![svm],
            sv_coef: vec![vec![1.0]],
            rho: vec![0.0],
            prob_a: vec![],
            prob_b: vec![],
            prob_density_marks: vec![0.1; 10],
            sv_indices: vec![1],
            label: vec![],
            n_sv: vec![],
        };
        assert!(one_class_model.has_probability_model());
        assert_eq!(svm_get_svr_probability(&one_class_model), None);
    }
}
