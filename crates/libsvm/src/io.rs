//! I/O routines for LIBSVM problem and model files.
//!
//! File formats match the original LIBSVM exactly, ensuring cross-tool
//! interoperability.

use std::io::{BufRead, Write};
use std::path::Path;

use crate::error::SvmError;
use crate::types::*;

// ─── C-compatible %g formatting ─────────────────────────────────────
//
// C's printf `%.Pg` format strips trailing zeros and picks fixed vs.
// scientific notation based on the exponent. Rust has no built-in
// equivalent, so we replicate the POSIX specification:
//   - Use scientific if exponent < -4 or exponent >= precision
//   - Otherwise use fixed notation
//   - Strip trailing zeros (and trailing decimal point)

use std::fmt;

/// Formats `f64` like C's `%.17g` (or any precision).
struct Gfmt {
    value: f64,
    precision: usize,
}

impl Gfmt {
    fn new(value: f64, precision: usize) -> Self {
        Self { value, precision }
    }
}

impl fmt::Display for Gfmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.value;
        let p = self.precision;

        if !v.is_finite() {
            return write!(f, "{}", v); // inf, -inf, NaN
        }

        if v == 0.0 {
            // Preserve sign of -0.0
            if v.is_sign_negative() {
                return write!(f, "-0");
            }
            return write!(f, "0");
        }

        // Compute the exponent (floor of log10(|v|))
        let abs_v = v.abs();
        let exp = abs_v.log10().floor() as i32;

        if exp < -4 || exp >= p as i32 {
            // Use scientific notation
            let s = format!("{:.prec$e}", v, prec = p.saturating_sub(1));
            // Rust uses 'e', C uses 'e'. Strip trailing zeros in mantissa.
            // C zero-pads exponent to at least 2 digits (e-05 not e-5).
            if let Some((mantissa, exponent)) = s.split_once('e') {
                let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
                // Parse exponent, reformat with at least 2 digits
                let exp_val: i32 = exponent.parse().unwrap_or(0);
                let exp_str = if exp_val < 0 {
                    format!("-{:02}", -exp_val)
                } else {
                    format!("+{:02}", exp_val)
                };
                write!(f, "{}e{}", mantissa, exp_str)
            } else {
                write!(f, "{}", s)
            }
        } else {
            // Use fixed notation. Number of decimal places = precision - (exp + 1)
            let decimal_places = if exp >= 0 {
                p.saturating_sub((exp + 1) as usize)
            } else {
                p + (-1 - exp) as usize
            };
            let s = format!("{:.prec$}", v, prec = decimal_places);
            let s = s.trim_end_matches('0').trim_end_matches('.');
            write!(f, "{}", s)
        }
    }
}

/// Format like C's `%.17g`
fn fmt_17g(v: f64) -> Gfmt {
    Gfmt::new(v, 17)
}

/// Format like C's `%.8g`
fn fmt_8g(v: f64) -> Gfmt {
    Gfmt::new(v, 8)
}

/// Format a float like C's `%g` (6 significant digits).
pub fn format_g(v: f64) -> String {
    format!("{}", Gfmt::new(v, 6))
}

/// Format a float like C's `%.17g` (17 significant digits).
pub fn format_17g(v: f64) -> String {
    format!("{}", Gfmt::new(v, 17))
}

// ─── String tables matching original LIBSVM ──────────────────────────

const SVM_TYPE_TABLE: &[&str] = &["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"];
const KERNEL_TYPE_TABLE: &[&str] = &["linear", "polynomial", "rbf", "sigmoid", "precomputed"];

fn svm_type_to_str(t: SvmType) -> &'static str {
    SVM_TYPE_TABLE[t as usize]
}

fn kernel_type_to_str(t: KernelType) -> &'static str {
    KERNEL_TYPE_TABLE[t as usize]
}

fn str_to_svm_type(s: &str) -> Option<SvmType> {
    match s {
        "c_svc" => Some(SvmType::CSvc),
        "nu_svc" => Some(SvmType::NuSvc),
        "one_class" => Some(SvmType::OneClass),
        "epsilon_svr" => Some(SvmType::EpsilonSvr),
        "nu_svr" => Some(SvmType::NuSvr),
        _ => None,
    }
}

fn str_to_kernel_type(s: &str) -> Option<KernelType> {
    match s {
        "linear" => Some(KernelType::Linear),
        "polynomial" => Some(KernelType::Polynomial),
        "rbf" => Some(KernelType::Rbf),
        "sigmoid" => Some(KernelType::Sigmoid),
        "precomputed" => Some(KernelType::Precomputed),
        _ => None,
    }
}

// ─── Problem file I/O ────────────────────────────────────────────────

/// Load an SVM problem from a file in LIBSVM sparse format.
///
/// Format: `<label> <index1>:<value1> <index2>:<value2> ...`
pub fn load_problem(path: &Path) -> Result<SvmProblem, SvmError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_problem_from_reader(reader)
}

/// Load an SVM problem from any buffered reader.
pub fn load_problem_from_reader(reader: impl BufRead) -> Result<SvmProblem, SvmError> {
    let mut labels = Vec::new();
    let mut instances = Vec::new();

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let line_num = line_idx + 1;
        let mut parts = line.split_whitespace();

        // Parse label
        let label_str = parts.next().ok_or_else(|| SvmError::ParseError {
            line: line_num,
            message: "missing label".into(),
        })?;
        let label: f64 = label_str.parse().map_err(|_| SvmError::ParseError {
            line: line_num,
            message: format!("invalid label: {}", label_str),
        })?;

        // Parse features (must be in ascending index order)
        let mut nodes = Vec::new();
        let mut prev_index: i32 = 0;
        for token in parts {
            let (idx_str, val_str) = token.split_once(':').ok_or_else(|| SvmError::ParseError {
                line: line_num,
                message: format!("expected index:value, got: {}", token),
            })?;
            let index: i32 = idx_str.parse().map_err(|_| SvmError::ParseError {
                line: line_num,
                message: format!("invalid index: {}", idx_str),
            })?;
            if !nodes.is_empty() && index <= prev_index {
                return Err(SvmError::ParseError {
                    line: line_num,
                    message: format!(
                        "feature indices must be ascending: {} follows {}",
                        index, prev_index
                    ),
                });
            }
            let value: f64 = val_str.parse().map_err(|_| SvmError::ParseError {
                line: line_num,
                message: format!("invalid value: {}", val_str),
            })?;
            prev_index = index;
            nodes.push(SvmNode { index, value });
        }

        labels.push(label);
        instances.push(nodes);
    }

    Ok(SvmProblem { labels, instances })
}

// ─── Model file I/O ──────────────────────────────────────────────────

/// Save an SVM model to a file in the original LIBSVM format.
pub fn save_model(path: &Path, model: &SvmModel) -> Result<(), SvmError> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    save_model_to_writer(writer, model)
}

/// Save an SVM model to any writer.
pub fn save_model_to_writer(mut w: impl Write, model: &SvmModel) -> Result<(), SvmError> {
    let param = &model.param;

    writeln!(w, "svm_type {}", svm_type_to_str(param.svm_type))?;
    writeln!(w, "kernel_type {}", kernel_type_to_str(param.kernel_type))?;

    if param.kernel_type == KernelType::Polynomial {
        writeln!(w, "degree {}", param.degree)?;
    }
    if matches!(
        param.kernel_type,
        KernelType::Polynomial | KernelType::Rbf | KernelType::Sigmoid
    ) {
        writeln!(w, "gamma {}", fmt_17g(param.gamma))?;
    }
    if matches!(
        param.kernel_type,
        KernelType::Polynomial | KernelType::Sigmoid
    ) {
        writeln!(w, "coef0 {}", fmt_17g(param.coef0))?;
    }

    let nr_class = model.nr_class;
    writeln!(w, "nr_class {}", nr_class)?;
    writeln!(w, "total_sv {}", model.sv.len())?;

    // rho
    write!(w, "rho")?;
    for r in &model.rho {
        write!(w, " {}", fmt_17g(*r))?;
    }
    writeln!(w)?;

    // label (classification only)
    if !model.label.is_empty() {
        write!(w, "label")?;
        for l in &model.label {
            write!(w, " {}", l)?;
        }
        writeln!(w)?;
    }

    // probA
    if !model.prob_a.is_empty() {
        write!(w, "probA")?;
        for v in &model.prob_a {
            write!(w, " {}", fmt_17g(*v))?;
        }
        writeln!(w)?;
    }

    // probB
    if !model.prob_b.is_empty() {
        write!(w, "probB")?;
        for v in &model.prob_b {
            write!(w, " {}", fmt_17g(*v))?;
        }
        writeln!(w)?;
    }

    // prob_density_marks (one-class)
    if !model.prob_density_marks.is_empty() {
        write!(w, "prob_density_marks")?;
        for v in &model.prob_density_marks {
            write!(w, " {}", fmt_17g(*v))?;
        }
        writeln!(w)?;
    }

    // nr_sv
    if !model.n_sv.is_empty() {
        write!(w, "nr_sv")?;
        for n in &model.n_sv {
            write!(w, " {}", n)?;
        }
        writeln!(w)?;
    }

    // SV section
    writeln!(w, "SV")?;
    let num_sv = model.sv.len();
    let num_coef_rows = model.sv_coef.len(); // nr_class - 1

    for i in 0..num_sv {
        // sv_coef columns for this SV: %.17g
        for j in 0..num_coef_rows {
            write!(w, "{} ", fmt_17g(model.sv_coef[j][i]))?;
        }
        // sparse features: %.8g
        if model.param.kernel_type == KernelType::Precomputed {
            if let Some(node) = model.sv[i].first() {
                write!(w, "0:{} ", node.value as i32)?;
            }
        } else {
            for node in &model.sv[i] {
                write!(w, "{}:{} ", node.index, fmt_8g(node.value))?;
            }
        }
        writeln!(w)?;
    }

    Ok(())
}

/// Load an SVM model from a file in the original LIBSVM format.
pub fn load_model(path: &Path) -> Result<SvmModel, SvmError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_model_from_reader(reader)
}

/// Load an SVM model from any buffered reader.
pub fn load_model_from_reader(reader: impl BufRead) -> Result<SvmModel, SvmError> {
    let mut lines = reader.lines();

    // Defaults
    let mut param = SvmParameter::default();
    let mut nr_class: usize = 0;
    let mut total_sv: usize = 0;
    let mut rho = Vec::new();
    let mut label = Vec::new();
    let mut prob_a = Vec::new();
    let mut prob_b = Vec::new();
    let mut prob_density_marks = Vec::new();
    let mut n_sv = Vec::new();

    // Read header
    let mut line_num: usize = 0;
    loop {
        let line = lines
            .next()
            .ok_or_else(|| SvmError::ModelFormatError("unexpected end of file in header".into()))??;
        line_num += 1;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap();

        match cmd {
            "svm_type" => {
                let val = parts.next().ok_or_else(|| SvmError::ModelFormatError(
                    format!("line {}: missing svm_type value", line_num),
                ))?;
                param.svm_type = str_to_svm_type(val).ok_or_else(|| {
                    SvmError::ModelFormatError(format!("line {}: unknown svm_type: {}", line_num, val))
                })?;
            }
            "kernel_type" => {
                let val = parts.next().ok_or_else(|| SvmError::ModelFormatError(
                    format!("line {}: missing kernel_type value", line_num),
                ))?;
                param.kernel_type = str_to_kernel_type(val).ok_or_else(|| {
                    SvmError::ModelFormatError(format!("line {}: unknown kernel_type: {}", line_num, val))
                })?;
            }
            "degree" => {
                param.degree = parse_single(&mut parts, line_num, "degree")?;
            }
            "gamma" => {
                param.gamma = parse_single(&mut parts, line_num, "gamma")?;
            }
            "coef0" => {
                param.coef0 = parse_single(&mut parts, line_num, "coef0")?;
            }
            "nr_class" => {
                nr_class = parse_single(&mut parts, line_num, "nr_class")?;
            }
            "total_sv" => {
                total_sv = parse_single(&mut parts, line_num, "total_sv")?;
            }
            "rho" => {
                rho = parse_multiple_f64(&mut parts, line_num, "rho")?;
            }
            "label" => {
                label = parse_multiple_i32(&mut parts, line_num, "label")?;
            }
            "probA" => {
                prob_a = parse_multiple_f64(&mut parts, line_num, "probA")?;
            }
            "probB" => {
                prob_b = parse_multiple_f64(&mut parts, line_num, "probB")?;
            }
            "prob_density_marks" => {
                prob_density_marks = parse_multiple_f64(&mut parts, line_num, "prob_density_marks")?;
            }
            "nr_sv" => {
                n_sv = parts
                    .map(|s| {
                        s.parse::<usize>().map_err(|_| {
                            SvmError::ModelFormatError(format!(
                                "line {}: invalid nr_sv value: {}",
                                line_num, s
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "SV" => break,
            _ => {
                return Err(SvmError::ModelFormatError(format!(
                    "line {}: unknown keyword: {}",
                    line_num, cmd
                )));
            }
        }
    }

    // Read SV section
    let m = if nr_class > 1 { nr_class - 1 } else { 1 };
    let mut sv_coef: Vec<Vec<f64>> = (0..m).map(|_| Vec::with_capacity(total_sv)).collect();
    let mut sv: Vec<Vec<SvmNode>> = Vec::with_capacity(total_sv);

    for _ in 0..total_sv {
        let line = lines
            .next()
            .ok_or_else(|| SvmError::ModelFormatError("unexpected end of file in SV section".into()))??;
        line_num += 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();

        // First m tokens are sv_coef values
        for (k, coef_row) in sv_coef.iter_mut().enumerate() {
            let val_str = parts.next().ok_or_else(|| SvmError::ModelFormatError(
                format!("line {}: missing sv_coef[{}]", line_num, k),
            ))?;
            let val: f64 = val_str.parse().map_err(|_| SvmError::ModelFormatError(
                format!("line {}: invalid sv_coef: {}", line_num, val_str),
            ))?;
            coef_row.push(val);
        }

        // Remaining tokens are index:value pairs
        let mut nodes = Vec::new();
        for token in parts {
            let (idx_str, val_str) = token.split_once(':').ok_or_else(|| {
                SvmError::ModelFormatError(format!(
                    "line {}: expected index:value, got: {}",
                    line_num, token
                ))
            })?;
            let index: i32 = idx_str.parse().map_err(|_| {
                SvmError::ModelFormatError(format!("line {}: invalid index: {}", line_num, idx_str))
            })?;
            let value: f64 = val_str.parse().map_err(|_| {
                SvmError::ModelFormatError(format!("line {}: invalid value: {}", line_num, val_str))
            })?;
            nodes.push(SvmNode { index, value });
        }
        sv.push(nodes);
    }

    Ok(SvmModel {
        param,
        nr_class,
        sv,
        sv_coef,
        rho,
        prob_a,
        prob_b,
        prob_density_marks,
        sv_indices: Vec::new(), // not stored in model file
        label,
        n_sv,
    })
}

// ─── Helper parsers ──────────────────────────────────────────────────

fn parse_single<T: std::str::FromStr>(
    parts: &mut std::str::SplitWhitespace<'_>,
    line_num: usize,
    field: &str,
) -> Result<T, SvmError> {
    let val_str = parts.next().ok_or_else(|| {
        SvmError::ModelFormatError(format!("line {}: missing {} value", line_num, field))
    })?;
    val_str.parse().map_err(|_| {
        SvmError::ModelFormatError(format!("line {}: invalid {} value: {}", line_num, field, val_str))
    })
}

fn parse_multiple_f64(
    parts: &mut std::str::SplitWhitespace<'_>,
    line_num: usize,
    field: &str,
) -> Result<Vec<f64>, SvmError> {
    parts
        .map(|s| {
            s.parse::<f64>().map_err(|_| {
                SvmError::ModelFormatError(format!(
                    "line {}: invalid {} value: {}",
                    line_num, field, s
                ))
            })
        })
        .collect()
}

fn parse_multiple_i32(
    parts: &mut std::str::SplitWhitespace<'_>,
    line_num: usize,
    field: &str,
) -> Result<Vec<i32>, SvmError> {
    parts
        .map(|s| {
            s.parse::<i32>().map_err(|_| {
                SvmError::ModelFormatError(format!(
                    "line {}: invalid {} value: {}",
                    line_num, field, s
                ))
            })
        })
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn parse_heart_scale() {
        let path = data_dir().join("heart_scale");
        let problem = load_problem(&path).unwrap();
        assert_eq!(problem.labels.len(), 270);
        assert_eq!(problem.instances.len(), 270);
        // First instance: +1 label, 12 features (index 11 is missing/sparse)
        assert_eq!(problem.labels[0], 1.0);
        assert_eq!(problem.instances[0][0], SvmNode { index: 1, value: 0.708333 });
        assert_eq!(problem.instances[0].len(), 12);
    }

    #[test]
    fn parse_iris() {
        let path = data_dir().join("iris.scale");
        let problem = load_problem(&path).unwrap();
        assert_eq!(problem.labels.len(), 150);
        // 3 classes: 1, 2, 3
        let classes: std::collections::HashSet<i64> =
            problem.labels.iter().map(|&l| l as i64).collect();
        assert_eq!(classes.len(), 3);
    }

    #[test]
    fn parse_housing() {
        let path = data_dir().join("housing_scale");
        let problem = load_problem(&path).unwrap();
        assert_eq!(problem.labels.len(), 506);
        // Regression: labels are continuous
        assert!((problem.labels[0] - 24.0).abs() < 1e-10);
    }

    #[test]
    fn parse_empty_lines() {
        let input = b"+1 1:0.5\n\n-1 2:0.3\n";
        let problem = load_problem_from_reader(&input[..]).unwrap();
        assert_eq!(problem.labels.len(), 2);
    }

    #[test]
    fn parse_error_unsorted_indices() {
        let input = b"+1 3:0.5 1:0.3\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("ascending"), "error: {}", msg);
    }

    #[test]
    fn parse_error_duplicate_indices() {
        let input = b"+1 1:0.5 1:0.3\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_error_missing_colon() {
        let input = b"+1 1:0.5 bad_token\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
    }

    #[test]
    fn load_c_trained_model() {
        // Load a model produced by the original C LIBSVM svm-train
        let path = data_dir().join("heart_scale.model");
        let model = load_model(&path).unwrap();
        assert_eq!(model.nr_class, 2);
        assert_eq!(model.param.svm_type, SvmType::CSvc);
        assert_eq!(model.param.kernel_type, KernelType::Rbf);
        assert!((model.param.gamma - 0.076923076923076927).abs() < 1e-15);
        assert_eq!(model.sv.len(), 132);
        assert_eq!(model.label, vec![1, -1]);
        assert_eq!(model.n_sv, vec![64, 68]);
        assert!((model.rho[0] - 0.42446205176771573).abs() < 1e-15);
        // sv_coef should have 1 row (nr_class - 1) with 132 entries
        assert_eq!(model.sv_coef.len(), 1);
        assert_eq!(model.sv_coef[0].len(), 132);
    }

    #[test]
    fn roundtrip_c_model() {
        // Load C model, save it back, and verify byte-exact match
        let path = data_dir().join("heart_scale.model");
        let original_bytes = std::fs::read_to_string(&path).unwrap();
        let model = load_model(&path).unwrap();

        let mut buf = Vec::new();
        save_model_to_writer(&mut buf, &model).unwrap();
        let rust_output = String::from_utf8(buf).unwrap();

        // Compare line by line for better diagnostics
        let orig_lines: Vec<&str> = original_bytes.lines().collect();
        let rust_lines: Vec<&str> = rust_output.lines().collect();
        assert_eq!(
            orig_lines.len(),
            rust_lines.len(),
            "line count mismatch: C={} Rust={}",
            orig_lines.len(),
            rust_lines.len()
        );
        for (i, (o, r)) in orig_lines.iter().zip(rust_lines.iter()).enumerate() {
            assert_eq!(o, r, "line {} differs:\n  C:    {:?}\n  Rust: {:?}", i + 1, o, r);
        }
    }

    #[test]
    fn gfmt_matches_c_printf() {
        // Reference values from C's printf("%.17g|%.8g\n", v, v)
        let cases: &[(f64, &str, &str)] = &[
            (0.5,                    "0.5",                      "0.5"),
            (-1.0,                   "-1",                       "-1"),
            (0.123456789012345,      "0.123456789012345",        "0.12345679"),
            (-0.987654321098765,     "-0.98765432109876505",     "-0.98765432"),
            (0.42446200000000001,    "0.42446200000000001",      "0.424462"),
            (0.0,                    "0",                        "0"),
            (1e-5,                   "1.0000000000000001e-05",   "1e-05"),
            (1e-4,                   "0.0001",                   "0.0001"),
            (1e20,                   "1e+20",                    "1e+20"),
            (-0.25,                  "-0.25",                    "-0.25"),
            (0.75,                   "0.75",                     "0.75"),
            (0.708333,               "0.70833299999999999",      "0.708333"),
            (1.0,                    "1",                        "1"),
        ];
        for &(v, expected_17g, expected_8g) in cases {
            let got_17 = format!("{}", fmt_17g(v));
            let got_8 = format!("{}", fmt_8g(v));
            assert_eq!(got_17, expected_17g, "%.17g mismatch for {}", v);
            assert_eq!(got_8, expected_8g, "%.8g mismatch for {}", v);
        }
    }

    #[test]
    fn model_roundtrip() {
        // Create a minimal model and verify save → load roundtrip
        let model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::CSvc,
                kernel_type: KernelType::Rbf,
                gamma: 0.5,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![
                vec![SvmNode { index: 1, value: 0.5 }, SvmNode { index: 3, value: -1.0 }],
                vec![SvmNode { index: 1, value: -0.25 }, SvmNode { index: 2, value: 0.75 }],
            ],
            sv_coef: vec![vec![0.123456789012345, -0.987654321098765]],
            rho: vec![0.42446200000000001],
            prob_a: vec![],
            prob_b: vec![],
            prob_density_marks: vec![],
            sv_indices: vec![],
            label: vec![1, -1],
            n_sv: vec![1, 1],
        };

        let mut buf = Vec::new();
        save_model_to_writer(&mut buf, &model).unwrap();

        let loaded = load_model_from_reader(&buf[..]).unwrap();

        assert_eq!(loaded.nr_class, model.nr_class);
        assert_eq!(loaded.param.svm_type, model.param.svm_type);
        assert_eq!(loaded.param.kernel_type, model.param.kernel_type);
        assert_eq!(loaded.sv.len(), model.sv.len());
        assert_eq!(loaded.label, model.label);
        assert_eq!(loaded.n_sv, model.n_sv);
        assert_eq!(loaded.rho.len(), model.rho.len());
        // Check rho within tolerance (roundtrip through text)
        for (a, b) in loaded.rho.iter().zip(model.rho.iter()) {
            assert!((a - b).abs() < 1e-10, "rho mismatch: {} vs {}", a, b);
        }
        // Check sv_coef within tolerance
        for (row_a, row_b) in loaded.sv_coef.iter().zip(model.sv_coef.iter()) {
            for (a, b) in row_a.iter().zip(row_b.iter()) {
                assert!((a - b).abs() < 1e-10, "sv_coef mismatch: {} vs {}", a, b);
            }
        }
    }
}
