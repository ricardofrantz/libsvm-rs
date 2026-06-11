# Goal: libsvm-rs-wa5 — README accuracy pass for 0.9.0

Audit the ENTIRE README.md (814 lines) against the current tree and fix drift.
Accuracy pass, not a redesign — do NOT restructure sections wholesale.

## Tasks
1. Features section: document every flag in crates/libsvm/Cargo.toml [features]
   (`serde`, `rayon`) — what each enables; default = [] keeps the single
   runtime dep (thiserror only).
2. Quick Start: show the SvmParameterBuilder API alongside (or instead of) raw
   SvmParameter struct literals; keep one direct-struct example (still public API).
3. Link docs/MIGRATION.md prominently in the "How this differs"/"When to Use It"
   area — a migrating libsvm-sys2 user must find it in one hop.
4. Parity Status + Performance numbers: cross-check EVERY number against the
   CURRENT committed artifacts: reference/compare_summary.json (now
   65 pass / 0 fail / 29 warn / 5 skip), reference/differential_report.md,
   reference/benchmark_report.md. Never carry stale figures or invent numbers.
5. Security Considerations: reflect the refreshed SECURITY_AUDIT.md (2026-06-11,
   serde/rayon/builder/LCG surfaces; negative-gamma load fix).
6. Installation snippet: keep current exact published version (0.8.1) and leave
   a note line for the release bead to bump (no TODO markers in shipped prose).
7. MSRV caveat: rust-version is 1.75, but the `rayon` feature currently needs
   rustc 1.80 (rayon-core 1.13) — if README states MSRV, say this honestly
   (open bug libsvm-rs-9dt; do not resolve it here).
8. Humanize changed prose: plain, factual, no AI-slop (no "blazingly fast",
   no triads, no marketing flourish). Match the existing README voice.

## Acceptance criteria
- Every number in README traces to a committed artifact or the code.
- Features, builder, MIGRATION.md link present as above.
- No section restructuring; diff confined to README.md.
- Gate passes.

## Verification (gate)
- `cargo fmt --all -- --check` (no-op expected, README only)
- Extract every Rust code block from README and confirm it matches current API
  by compiling the crate's doctests: `cargo test --doc --all-features` must stay
  green, and any README snippet you change must use only public API that exists
  (spot-check against crates/libsvm/src/lib.rs exports).
- `grep -n 'MIGRATION.md' README.md` shows the link.
- `git diff --stat` shows only README.md changed.

## Scope
- ✅ Always: README.md
- ⚠️ Ask-first: any change outside README.md; any restructuring of sections
- 🚫 Never: version bumps, CHANGELOG, reference/ artifacts, code, tests, CI
