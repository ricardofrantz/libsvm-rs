# Goal: Re-baseline differential reference artifacts on macOS   (bead: libsvm-rs-dt3)

SUPERVISOR-DRIVEN CYCLE (user-approved): no coder dispatch. The supervisor on
nexus-dev drives the Mac over ssh — run the suite, review, commit there, pull
back. Approved decisions: per-platform baselines for BOTH differential counts
and benchmark notes.

## 1. Objective
libsvm-rs-4r5 switched public k-fold CV shuffling to the process-global libc
rand() replica, changing macOS fold splits. The committed Mac baseline
(236/4/0/10) and reference/differential_{report.md,results.json} predate it.
Re-run `DIFF_SCOPE=full scripts/run_differential_suite.py` on the Mac
(checkout at 283339e), regenerate and commit the artifacts, and record
per-platform baselines.

## 2. Acceptance Criteria
- [ ] Full suite completes on macOS with exit 0; counts captured (expect
      warns to drop from 4; fail must stay 0).
- [ ] reference/differential_results.json + differential_report.md are the
      fresh Mac run, committed.
- [ ] README Parity Status table gains a platform column: macOS <new counts>
      and Linux 240/0/0/10 (Linux counts from the eo9 full run).
- [ ] reference/tolerance_policy.md records the current per-platform
      differential baselines and a benchmark note: benchmark_report.md is
      macOS-generated; Linux full-bench train_probability median 1.074 /
      worst 2.058 (pop-os) noted alongside.
- [ ] Diff confined to reference/ + README.md (plus ledger at impl commit).

## 3. Verification
- `ssh mac 'cd ~/Projects/libsvm-rs && DIFF_SCOPE=full python3 scripts/run_differential_suite.py'` exits 0
- Counts in report.md == summary in results.json
- `git diff --stat` on the Mac shows only the files above

## 4. Scope
✅ ALWAYS:    reference/differential_results.json, reference/differential_report.md,
              reference/tolerance_policy.md, README.md (parity table + one note)
⚠️ ASK FIRST: any failing case (fail > 0), warn count INCREASING, edits to
              scripts/, regenerating benchmark_report.md
🚫 NEVER:     src/, tests/, vendor/, Cargo.toml, ci.yml

## 5. Stop Conditions
DONE when criteria pass. STOP if fail > 0 or warns increase — that contradicts
the RNG-change rationale and needs triage, not re-baselining.
