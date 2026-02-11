use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn data_file(name: &str) -> PathBuf {
    workspace_root().join("data").join(name)
}

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let dir = workspace_root()
        .join(".tmp")
        .join("cli-tests")
        .join(format!("{}-{}-{}", tag, std::process::id(), id));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn bin_path() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_svm-train-rs"))
}

#[test]
fn no_args_prints_help_and_exits_nonzero() {
    let output = Command::new(bin_path()).output().unwrap();
    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-train"));
}

#[test]
fn cross_validation_requires_at_least_two_folds() {
    let output = Command::new(bin_path())
        .arg("-v")
        .arg("1")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();
    assert!(!output.status.success());

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("n-fold cross validation: n must >= 2"));
}

#[test]
fn training_writes_model_file() {
    let dir = unique_tmp_dir("svm-train");
    let model_path = dir.join("heart.model");

    let output = Command::new(bin_path())
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(&model_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    assert!(model_path.exists());

    let content = fs::read_to_string(&model_path).unwrap();
    assert!(content.contains("svm_type"));
    assert!(content.contains("SV"));
}

#[test]
fn cross_validation_prints_accuracy() {
    let output = Command::new(bin_path())
        .arg("-q")
        .arg("-v")
        .arg("3")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Cross Validation Accuracy"));
}
