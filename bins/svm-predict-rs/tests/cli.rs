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
    Path::new(env!("CARGO_BIN_EXE_svm-predict-rs"))
}

#[test]
fn no_args_prints_help_and_exits_nonzero() {
    let output = Command::new(bin_path()).output().unwrap();
    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-predict"));
}

#[test]
fn prediction_writes_output_file() {
    let dir = unique_tmp_dir("svm-predict");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    assert!(output_path.exists());

    let lines = fs::read_to_string(&output_path).unwrap();
    assert_eq!(lines.lines().count(), 270);
}

#[test]
fn probability_flag_rejected_for_non_probability_model() {
    let dir = unique_tmp_dir("svm-predict-prob");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-b")
        .arg("1")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Model does not support probability estimates"));
}

#[test]
fn quiet_mode_suppresses_accuracy_summary() {
    let dir = unique_tmp_dir("svm-predict-quiet");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!stderr.contains("Accuracy ="));
}
