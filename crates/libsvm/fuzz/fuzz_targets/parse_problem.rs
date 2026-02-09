#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    let mut temp = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if temp.write_all(data).is_err() {
        return;
    }
    let _ = libsvm_rs::io::load_problem(temp.path());
});
