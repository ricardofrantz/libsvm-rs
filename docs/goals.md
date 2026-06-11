# Goal: Fix WASM Core Build CI job — install wasm32 target for pinned toolchain   (bead: libsvm-rs-vbx)

SUPERVISOR-DRIVEN MICRO-CYCLE: one-line CI fix, within the supervisor touch-up
bound; no coder dispatch.

## 1. Objective
CI is red on master: the `wasm-core` job installs wasm32-unknown-unknown only
for the dtolnay stable toolchain, but rust-toolchain.toml pins 1.93.1, so the
build runs on a toolchain without the target (E0463).

## 2. Change
Add one step to the `wasm-core` job in .github/workflows/ci.yml, after the
cache step and before the build:

    - run: rustup target add wasm32-unknown-unknown

Pin-independent (applies to whichever toolchain the repo pins). The
`wasm-integration` job is NOT affected — run.sh:31 already does
`rustup target add` itself.

## 3. Acceptance Criteria
- [ ] wasm-core job passes on the default branch; full CI run green.
- [ ] Fix uses `rustup target add` (no hardcoded toolchain version).
- [ ] Diff confined to .github/workflows/ci.yml (one step).

## 4. Verification
Quick (local): `cargo build --locked -p libsvm-rs --target wasm32-unknown-unknown`
Full (CI): push, `gh run watch` — requires user OK to push.
