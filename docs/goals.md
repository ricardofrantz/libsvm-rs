# Goal: libsvm-rs-246 — docs.rs full pass: document all public items, missing_docs lint, doctests, feature metadata

Context pointers (read first): `br show libsvm-rs-246` (it carries the full
task spec — follow its 5 numbered task items exactly) · VISION.md ·
docs/MIGRATION.md and reference/function_parity_matrix.md (reuse the LIBSVM
parity mapping in doc comments; do not invent).

## Task (summary — bead is authoritative)
1. `missing_docs` lint in crates/libsvm/src/lib.rs (deny if fully clean at the
   end, else warn — state the choice in your report) + document every flagged
   public item, with LIBSVM correspondence where relevant.
2. Doctests on train, predict, cross-validation, model save/load, builder —
   data-light examples should actually run; `no_run` only where training time
   matters.
3. `#[cfg_attr(docsrs, doc(cfg(feature = "...")))]` on serde/rayon items +
   `[package.metadata.docs.rs] all-features = true, rustdoc-args = ["--cfg", "docsrs"]`
   in crates/libsvm/Cargo.toml.
4. Crate-level `//!` overview: what it is, parity statement, features, quick
   example, link to MIGRATION guide + repo.
DOCS ONLY — no behavior/API changes; API warts go in your report, not fixes.

## Acceptance criteria (from the bead)
- `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc --all-features --no-deps -p libsvm-rs`
  clean (fallback if no nightly: stable doc clean + cfg_attr by inspection;
  say which you ran).
- `cargo test --doc --all-features` passes with >0 doctests for
  train/predict/save/load/builder — name them in your report.
- Zero missing_docs warnings with the lint enabled.
- `cargo fmt --check` + `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.

## Verification
`cargo doc --all-features --no-deps -p libsvm-rs 2>&1 | tail -5` and
`cargo test --doc --all-features 2>&1 | tail -5`.

## Scope
✅ crates/libsvm/src/**/*.rs (doc comments + lint attr only),
   crates/libsvm/Cargo.toml (docs.rs metadata only).
🚫 Everything else. README.md and assets/ have uncommitted changes from
another session — do not touch or stage them.
