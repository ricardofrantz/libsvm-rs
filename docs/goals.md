# Goal: libsvm-rs-80n — CHANGELOG: write the 0.9.0 entry covering all post-0.8.1 work

Context pointers (read these first): `br show libsvm-rs-80n` · VISION.md ·
CHANGELOG.md (match its style exactly).

## Task
Write the complete entry under `## [Unreleased]` in CHANGELOG.md (a short
Security subsection already exists there — keep it, merge into your grouping).
Do NOT retitle to 0.9.0 — the release bead does that. Sweep
`git log v0.8.1..HEAD --oneline` and map every commit to a bullet or a
conscious exclusion (pure CI/goal-loop/dependabot commits may be folded or
omitted; user-visible changes may NOT).

Must cover (verify against the log, don't trust blindly): builder API; serde
feature; rayon feature (parallel CV + probability folds); migration guide
docs/MIGRATION.md; wasm CI check; glibc-rand probability-CV shuffle on Linux;
macOS differential re-baseline; master→main migration (call out link/clone
impact); security audit + negative-gamma load rejection (already drafted in
the file).

## Acceptance criteria
- Every v0.8.1..HEAD commit represented or consciously excluded; post the
  commit→bullet mapping in your CODER REPORT (not in the file).
- The Linux probability behavior change has its own explicit Changed bullet:
  one sentence on WHY (glibc rand replication restores C parity for the
  stratified shuffle) and WHO is affected (Linux users of probability
  estimation/CV) — outputs differ from 0.8.1 by design.
- Keep-a-Changelog grouping (Added/Changed/Fixed/Security/Documentation);
  user-facing language; NO commit hashes or bead IDs in the file.

## Verification
`grep -n "Unreleased" CHANGELOG.md` + read your own diff; the mapping list is
the completeness proof.

## Scope
✅ CHANGELOG.md only.
🚫 Everything else. NOTE: README.md and assets/ have uncommitted changes from
another session — do not touch or stage them.
