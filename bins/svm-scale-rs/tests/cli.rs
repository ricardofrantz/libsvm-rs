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
    Path::new(env!("CARGO_BIN_EXE_svm-scale-rs"))
}

#[test]
fn no_args_prints_help_and_exits_nonzero() {
    let output = Command::new(bin_path()).output().unwrap();
    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-scale"));
}

#[test]
fn inconsistent_bounds_return_error() {
    let output = Command::new(bin_path())
        .arg("-l")
        .arg("1")
        .arg("-u")
        .arg("0")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("inconsistent lower/upper specification"));
}

#[test]
fn save_then_restore_produces_same_scaled_output() {
    let dir = unique_tmp_dir("svm-scale");
    let data_path = dir.join("tiny.scale");
    let scale_path = dir.join("tiny.params");

    fs::write(&data_path, "1 1:1.0 3:2.0\n-1 1:0.0 2:4.0 3:1.0\n1 2:2.0\n").unwrap();

    let out_save = Command::new(bin_path())
        .arg("-s")
        .arg(&scale_path)
        .arg(&data_path)
        .output()
        .unwrap();
    assert!(
        out_save.status.success(),
        "stderr: {}",
        String::from_utf8(out_save.stderr).unwrap()
    );
    assert!(scale_path.exists());

    let out_restore = Command::new(bin_path())
        .arg("-r")
        .arg(&scale_path)
        .arg(&data_path)
        .output()
        .unwrap();
    assert!(
        out_restore.status.success(),
        "stderr: {}",
        String::from_utf8(out_restore.stderr).unwrap()
    );

    let save_stdout = String::from_utf8(out_save.stdout).unwrap();
    let restore_stdout = String::from_utf8(out_restore.stdout).unwrap();
    assert_eq!(save_stdout, restore_stdout);
}

#[test]
fn scaling_emits_non_empty_output() {
    let output = Command::new(bin_path())
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.trim().is_empty());
}

#[test]
fn negative_feature_index_returns_error_instead_of_panicking() {
    let dir = unique_tmp_dir("svm-scale-neg-index");
    let data_path = dir.join("negative_index.scale");
    fs::write(&data_path, "1 -1:0.5 1:1.0\n").unwrap();

    let output = Command::new(bin_path()).arg(&data_path).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("feature index -1 out of valid range [1, 10000000]"),
        "stderr: {stderr}"
    );
}
