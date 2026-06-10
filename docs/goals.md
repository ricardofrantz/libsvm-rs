# Goal: wasm32 build check in CI for the core library   (bead: libsvm-rs-5yo)

## 1. Objective
Make wasm32 a checked target for the CORE crate. CI already has a
`wasm-integration` job (ci.yml:168) that exercises the example module via
run.sh — that covers the example, slowly. Missing: a fast, direct compile
check `cargo build --locked -p libsvm-rs --target wasm32-unknown-unknown`
(default features) so a core change that breaks wasm compat fails loudly.
Full spec: `br show libsvm-rs-5yo`.

## 2. Acceptance Criteria
- [ ] New small CI job (or step) in .github/workflows/ci.yml:
      `rustup target add wasm32-unknown-unknown` (via dtolnay/rust-toolchain
      `targets:` input, matching existing pins) then
      `cargo build --locked -p libsvm-rs --target wasm32-unknown-unknown`.
      Library only, default features. NOT bins, NOT --features rayon (threads).
      Do NOT duplicate or modify the existing wasm-integration job.
- [ ] Conventions: SHA-pinned actions identical to existing pins in ci.yml,
      `persist-credentials: false`, `permissions: contents: read`,
      Swatinem/rust-cache with a distinct cache key.
- [ ] Local proof-of-teeth experiment (NOT committed): temporarily add an
      `std::fs` call to lib.rs, show the wasm build command fails, revert.
      Report the failing output snippet in the CODER REPORT.
- [ ] README: one line/sentence noting wasm32-unknown-unknown is a CI-checked
      build target for the core crate (extend an existing targets/CI line if
      one exists).
- [ ] Gates: bash .sc/5yo.gate.sh green (wasm build, fmt, clippy, default tests,
      actionlint if installed — script skips it gracefully if absent).

## 3. Verification
- Quick: `cargo build --locked -p libsvm-rs --target wasm32-unknown-unknown`
- Full: `bash .sc/5yo.gate.sh`

## 4. Scope
✅ ALWAYS:    .github/workflows/ci.yml (one new job/step), README.md (one line)
⚠️ ASK FIRST: any cfg change in crates/libsvm/src/ — ONLY if the core crate
              does not compile for wasm32 today; report findings first if the
              fix would touch solver/training logic (STOP-and-split rule)
🚫 NEVER:     examples/, bins/, vendor/, scripts/, reference/, data/,
              Cargo.toml deps, the existing wasm-integration job
## 5. Non-Goals / Constraints
- No wasm test execution (wasm-pack test / node runners) — compile check only.
- rustup target add locally is fine (already installed on this box, verify).

## 6. Context Pointers
- `br show libsvm-rs-5yo` — full spec incl. conventions reasoning.
- Existing ci.yml jobs for pin hashes and permissions patterns.
- VISION.md — WASM inference is a named deployment target.

## 7. Stop Conditions
- DONE when criteria pass. STOP if the core crate fails to compile for wasm32
  in a way that needs more than tiny cfg gating.
