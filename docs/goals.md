# Goal: libsvm-rs-lx7 — remove plan.md and goal-loop artifacts (supervisor-driven)
git rm plan.md docs/goals.md docs/goals-ledger.md; only historical mentions in
docs/improve-ledger.md remain (kept by design). Verify: ls-files clean + tests.
