# Goal: libsvm-rs-c3r — Release-gate verification sweep for 0.9.0

Mode: supervisor-driven tool run (verification-only; no coder dispatch).

## Acceptance criteria
- Every gate passes with logged output, or carries an explicit logged justification.
- Gate summary (command → result, differential pass/warn/fail/skip counts) posted via `br comment`.
- No gate weakened to pass.

## Gates
Scripted in `.sc/c3r.gate.sh` (logs to `.sc/c3r.gate.out`):
1. fmt --check
2. clippy --workspace --all-targets --all-features -D warnings
3. Feature matrix tests: no-default / default / serde / rayon / all-features
4. doc tests --all-features
5. cargo deny check
6. MSRV: cargo +1.75.0 check --workspace --all-features
7. Differential parity: reference lock check -> release build -> compare_references.sh
8. Coverage thresholds (skip with note if cargo-llvm-cov absent; rely on CI)
9. Rayon determinism: rayon suite run twice, identical outcomes
Plus: confirm CI on main is green (gh run list).

## Scope
- Always: run gates, log output, br comment, ledger entry
- Ask-first: any non-trivial fix a failing gate requires
- Never: weaken tests/lints/tolerances; regenerate reference baselines
