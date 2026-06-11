# Goal: libsvm-rs-9dt — bump MSRV to 1.80 (rayon feature requires it)

Decision (user-confirmed): set rust-version = "1.80" workspace-wide; fix the CI
MSRV job so it tests with the matching cargo and fails honestly; document the
change.

## Tasks
1. Cargo.toml (workspace root): rust-version = "1.80".
2. .github/workflows/ci.yml MSRV job: rename to "MSRV (1.80.0)", toolchain
   1.80.0 (update the dtolnay/rust-toolchain ref comment AND the `toolchain:`
   input; keep the action SHA-pinned — if the pinned SHA encodes the version
   tag, point the ref at the 1.80.0 tag's SHA, verifiable via
   `gh api repos/dtolnay/rust-toolchain/git/ref/tags/1.80.0`). Ensure the job's
   cargo is the 1.80 toolchain's own cargo (rustup default behavior is fine once
   the toolchain input is right).
3. README.md: update the MSRV paragraph — MSRV is 1.80 for all builds; drop the
   rayon caveat sentence.
4. CHANGELOG.md [Unreleased] Changed: one bullet — MSRV raised 1.75 → 1.80
   (required by rayon-core 1.13; previously the 1.75 claim was unsatisfiable
   with the rayon feature). No bead IDs in CHANGELOG.
5. Check for any other hardcoded 1.75 references: grep -rn '1\.75' --include='*.md' --include='*.toml' --include='*.yml' (update docs that state MSRV; leave unrelated matches).

## Acceptance criteria
- rust-version = "1.80" in workspace Cargo.toml.
- CI MSRV job pins toolchain 1.80.0 and would fail under a genuine 1.80 cargo
  if a dep needed newer.
- README + CHANGELOG updated, no bead IDs in shipped prose.
- Gate passes.

## Verification (gate)
- `rustup toolchain install 1.80.0 --profile minimal` (if absent), then
  `cargo +1.80.0 check --locked --all-features` → must succeed.
- `cargo +1.80.0 test --workspace --all-features` → green.
- `cargo fmt --all -- --check` and
  `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.
- `grep -rn 'rust-version' Cargo.toml` shows 1.80.

## Scope
- ✅ Always: Cargo.toml, .github/workflows/ci.yml (MSRV job only), README.md
  (MSRV paragraph), CHANGELOG.md ([Unreleased]), docs/*.md MSRV mentions
- ⚠️ Ask-first: touching other CI jobs, Cargo.lock changes, crate Cargo.tomls
- 🚫 Never: version bumps, reference/ artifacts, source code, tests
