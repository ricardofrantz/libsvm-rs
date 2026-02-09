/// Errors returned by libsvm-rs operations.
#[derive(Debug, thiserror::Error)]
pub enum SvmError {
    /// An SVM parameter failed validation.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A parse error occurred while reading a problem or model file.
    #[error("parse error at line {line}: {message}")]
    ParseError {
        /// 1-based line number where the error occurred.
        line: usize,
        /// Description of the parse failure.
        message: String,
    },

    /// A model file could not be loaded due to format issues.
    #[error("model format error: {0}")]
    ModelFormatError(String),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
