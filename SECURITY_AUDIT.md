# Security Audit — libsvm-rs

Date: 2026-06-02 · Scope: untrusted-input parsers, arithmetic safety, CLI robustness,
supply chain, CI gates · Method: 3 parallel read-only reviewers + coordinator
reproduction. Findings marked **[Verified]** were reproduced or code-confirmed by the
coordinator; **[Reviewer]** are reviewer-reported and code-plausible but not separately
reproduced.

Commit audited: `0bfe9f5` (master).

## Resolution status (2026-06-02)

All actionable findings fixed and verified in the working tree:

| ID | Sev | Status |
|---|---|---|
| F1 | High | **Fixed** — `validate_model_header` now requires `label.len()==nr_class` and `n_sv.len()==nr_class` for c_svc/nu_svc. Repro returns clean `ModelFormatError`, no panic. |
| F2 | Med×3 | **Fixed** — all three bins guard `flag.len() < 2` → usage/help. Repros no longer panic. |
| F3 | Low | **Fixed** — non-finite `gamma`/`coef0`/`rho`/`sv_coef`/feature values and negative `degree` rejected at load. |
| F4 | Low | **Documented** — comment added at `kernel.rs` precomputed cast (intentional, no panic). |
| F5 | Low | **Documented** — comment added at `qmatrix.rs` cache-size cast (clamped, no panic). |
| F6 | Low | **Fixed** — `deny.toml [advisories]` now `yanked = "deny"`, `unmaintained = "all"` (CI-verified). |

Verification: 4/4 panic repros now return clean errors (exit 1, no `panicked`); full
workspace suite passes under `RUSTFLAGS=-Coverflow-checks=on` (174 tests, 0 failed);
`cargo clippy --workspace --all-targets -D warnings` clean. New regression tests: 5 in
`io.rs`, 3 CLI tests in the bins; 2 new fuzz corpus seeds.

## Summary (original audit)

The crate is already well-hardened: zero `unsafe` (only a comment mentions it), a real
`LoadOptions` cap system with saturating arithmetic, SHA-pinned GitHub Actions, CI gates
(`cargo deny`/`audit`/`clippy`/tests/fuzz), and a minimal runtime surface (`thiserror`,
optional `rayon`). Full workspace test suite passes under `RUSTFLAGS=-Coverflow-checks=on`
(exit 0).

The audit found **4 reachable panics on untrusted/malformed input** (1 High via model file,
3 Medium via CLI args) plus minor trust/policy hardening items. No memory-safety (UB) issues
— all panics are safe Rust bounds checks, i.e. DoS/availability, not RCE.

## Findings

### F1 — High — Malformed classification model panics on predict (OOB) · [Verified]
- **Where:** sink `crates/libsvm/src/predict.rs:56` (`model.n_sv[i-1]`), also `:67-68`, `:100`; root cause `crates/libsvm/src/io.rs:953,971` (`validate_model_header` treats `label`/`n_sv` as optional — validated only `if !is_empty()`).
- **What:** A `c_svc`/`nu_svc` model with valid `nr_class`/`total_sv`/`rho`/SV rows but **no `nr_sv` and no `label` lines** loads successfully (empty vecs), then `predict` indexes `n_sv[]`/`label[]` out of bounds.
- **Verified repro:** model `svm_type c_svc / kernel_type linear / nr_class 2 / total_sv 1 / rho 0 / SV / 1 1:1` → `svm-predict-rs` → `panicked at predict.rs:56:53: index out of bounds: the len is 0 but the index is 0` (exit 101).
- **Impact:** DoS on any service that loads attacker-supplied model files. `load_model` is the documented untrusted entry point and its doc claims downstream structural invariants — this is a contract gap.
- **Fix:** in `validate_model_header`, for classification require `label.len() == nr_class` **and** `n_sv.len() == nr_class` (not only when present). Reject with `ParseError` otherwise.

### F2 — Medium — CLI panics on a bare `-` (or any single-char arg) · [Verified]
- **Where:** `bins/svm-train-rs/src/main.rs:92`, `bins/svm-predict-rs/src/main.rs:59`, `bins/svm-scale-rs/src/main.rs:119` — `flag.as_bytes()[1]` indexes byte 1 of a 1-byte string.
- **Verified repro:** `svm-train-rs - …`, `svm-scale-rs - …`, `svm-predict-rs - …` each `panicked … index out of bounds: the len is 1 but the index is 1` (exit 101).
- **Impact:** Ungraceful crash instead of usage/help on malformed CLI input. Low severity (local CLI), but trivially fixable.
- **Fix:** guard `flag.len() < 2` (or match the whole flag string) before indexing; fall through to `exit_with_help`.

### F3 — Low — Model numeric fields accept NaN/Inf; `degree` unbounded · [Reviewer]
- **Where:** `crates/libsvm/src/io.rs` degree/gamma/coef0 (~`:689-696`), `rho`/`coef`/feature values.
- **What:** `gamma nan`, `rho inf`, `1:nan`, or `degree 2000000000` are accepted (`load_model` never calls `SvmParameter::validate()`). Propagates NaN into kernel math / silently wrong predictions; large `degree` makes `powi` cost attacker-controlled (O(log degree), bounded).
- **Impact:** Correctness/trust, not memory safety. Worth tightening since the module advertises caps as the trust boundary.
- **Fix:** reject non-finite values in model numeric fields and/or call `validate()` after load; bound `degree`.

### F4 — Low — Precomputed-kernel column index `as usize` silently mis-maps · [Reviewer]
- **Where:** `crates/libsvm/src/kernel.rs:104,171`. `node.value as usize` on a fractional/out-of-range float saturates and `.get()` returns 0.0 — **no panic**, but a silently wrong kernel value. Matches upstream C laxity; flagged as silent-failure only.

### F5 — Low — Latent dependency on `check_parameter` before QMatrix construction · [Reviewer]
- **Where:** `crates/libsvm/src/qmatrix.rs:46,99,156`. `(cache_size * 1048576.0) as usize` is safe on the validated CLI path (float→usize saturates, `Cache::new` clamps); only risky if a caller builds `SvcQ`/`OneClassQ`/`SvrQ` directly with an unvalidated param. Contract note, not a live bug.

### F6 — Low — `deny.toml` advisory policy not PR-blocking for yanked/unmaintained · [Reviewer]
- **Where:** `deny.toml` `[advisories]` has only `ignore = []` (no explicit `yanked = "deny"` / `unmaintained`), and `[bans] multiple-versions = "warn"`. Yanked/unmaintained crates warn rather than fail CI, and the behavior drifts with cargo-deny defaults.
- **Fix:** add explicit `yanked = "deny"` and `unmaintained = "workspace"` (or `"all"`).

## What's already solid (verified, do not re-audit)
- **Loader caps:** `read_line_capped` enforces byte + per-line caps with `saturating_add` *before* extending the buffer; rejects NUL + non-UTF8; SV section does **not** preallocate from `total_sv` (anti-DoS). `nr_class`/`total_sv` intersect per-call options with module hard caps. Feature indices reject negatives + enforce `MAX_FEATURE_INDEX` + ascending order.
- **No `unsafe`** anywhere in `crates/`+`bins/` (only a doc comment). Candidate for `#![forbid(unsafe_code)]`.
- **Arithmetic:** solver clamps `quad_coef <= 0` to `TAU` before division and guards `nr_free == 0` in `calculate_rho`; `max_iter` overflow-guarded; cache LRU math uses `saturating_sub` + empty-list guard. Full suite green under overflow checks.
- **Supply chain:** all GitHub Actions SHA-pinned (mutable-tag class closed); least-privilege `permissions`; `persist-credentials: false`; no script-injection sinks (untrusted `github.*` routed via `env:`). Runtime surface verified = `thiserror` (+ optional `rayon`/`crossbeam`); only native `build.rs` (`cc` via `libfuzzer-sys`) confined to the `publish = false` fuzz crate. `[sources]` denies unknown registry/git.
- **CI gates on PRs:** `cargo deny --locked --all-features`, `cargo audit`, tests (default + `--no-default-features`), `clippy -D warnings`, MSRV, lockfile-drift guard, gitleaks, fuzz smoke 300s/target (7200s weekly).

## Recommended fix order
1. **F1** (High) — `validate_model_header` require label/n_sv length on classification. Add regression test (load incomplete model → expect `ParseError`, not panic) + fuzz corpus case.
2. **F2** (Medium) — guard single-char flags in all three bins; add CLI tests for bare `-` and trailing-operand-missing.
3. **F3** (Low) — reject non-finite model numerics / bound `degree`.
4. **F6** (Low) — harden `deny.toml` advisories.
5. **F4/F5** — document as accepted/contract; optional.

## Verification gaps (honest)
- `cargo-deny` / `cargo-audit` / `cargo-fuzz` not installed locally → F6 and advisory state **[Unverified locally; run in CI]**.
- F3/F4/F5 are code-confirmed but not separately reproduced by the coordinator.
