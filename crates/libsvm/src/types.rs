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
