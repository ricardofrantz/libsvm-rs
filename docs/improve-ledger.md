# Improvement Ledger

Decisions about improvement ideas live here (accepted/rejected/deferred and
why). Work lives in the bead database. One entry per deep-improve run.

## Run 2026-06-11 — refresh: close-out for stable release
**Studied:** Delta since 2026-06-10 run. All 9 beads closed (6 planned + 3
emergent: 4r5 glibc-rand Linux parity, aud probability-fold parallelism, dt3
macOS re-baseline). Unreleased on master since v0.8.1: builder, migration
guide, serde feature, rayon feature, Linux parity fix, wasm CI job. Findings:
CI RED on master (wasm job: rust-toolchain.toml pins 1.93.1, wasm32 target
only installed for the action's stable toolchain); CHANGELOG [Unreleased]
empty; local tests pass. User direction (gate): release 0.9.0 stable, full
docs.rs pass in scope, archive working files, NO new implementations —
close-out run.

**Ranked list (rev. after mid-run user additions: master→main migration,
dependabot branch cleanup, fresh SECURITY_AUDIT.md, plan.md removal):**
1. Fix WASM CI job (toolchain/target mismatch) — CI green is a release blocker — P0/S
2. Full release-gate verification: fmt, clippy, deny, test matrix across feature combos (none/serde/rayon/all), MSRV, differential parity suite, coverage — P0/M
3. Branch migration master→main + delete dependabot branches (user-requested; workflows reference master in 4 files) — P1/M
4. Fresh security audit → rewrite SECURITY_AUDIT.md against current tree (new surfaces since 2026-06-02 audit: serde deserialize, rayon, builder, glibc-rand LCG) — P1/M
5. CHANGELOG: write the 0.9.0 entry covering all post-0.8.1 work — P1/S
6. README accuracy pass: document serde/rayon features, builder API, link MIGRATION.md; humanizer before ship — P1/S
7. docs.rs full pass: all public items documented, missing_docs lint, doctests, docs.rs feature metadata — P1/M
8. cargo-semver-checks vs 0.8.1 (additive-only confirmation before stabilizing) — P2/S
9. Housekeeping: remove plan.md (user-requested) and docs/goals*.md goal-loop artifacts — P2/S
10. Release 0.9.0: version reconciliation, tag, GH release, cargo publish (user approval at publish time) — P1/S, last in sequence

**Cut line: all 10** (user-confirmed; this is a close-out, nothing below the line).

**Accepted → beads:** WASM CI fix — libsvm-rs-vbx; release-gate sweep —
libsvm-rs-c3r; master→main + dependabot branch cleanup — libsvm-rs-4bd;
fresh security audit / SECURITY_AUDIT.md rewrite — libsvm-rs-cww; CHANGELOG
0.9.0 entry — libsvm-rs-80n; README accuracy pass — libsvm-rs-wa5; docs.rs
full pass — libsvm-rs-246; cargo-semver-checks vs 0.8.1 — libsvm-rs-4pz;
housekeeping (plan.md, goals files) — libsvm-rs-lx7; release 0.9.0 —
libsvm-rs-2wg (publish step requires explicit user OK).
Dependency shape: vbx → {c3r, 4bd}; cww → {c3r, wa5, 80n}; 246 → c3r;
4bd → 80n; {everything} → lx7 → 2wg. Critical path:
vbx → 4bd → 80n → lx7 → 2wg; max parallel width 4.
Phase 4 audit done: `br dep cycles` clean, bv DAG healthy (no cycles,
19 nodes / 22 edges incl. closed history); quick-tier verification commands
run on pop-os: fmt clean, clippy --all-features clean, full all-features
test suite green, wasm32 local build green; reference scripts confirmed on
disk; tooling gaps (cargo-deny, 1.75.0 toolchain not installed locally)
recorded as a comment on c3r.
**Rejected:** 1.0.0 now — user chose 0.9.0; any new feature work (ndarray,
Axum promotion, perf rounds) — explicit "no new implementations" directive.
**Deferred (carried, now closed-as-out-of-scope for this repo's close-out):**
ndarray interop; Axum recipe promotion; mutation testing of solver.

## Run 2026-06-10 — initial
**Studied:** Full first-run study. v0.8.1 published; all 5 SVM/kernel types,
93% line coverage, 250-config differential parity (236 pass / 4 warn / 0
fail / 10 skip). Open ends found: rayon feature declared but unwired; no
builder/serde/dense-data API; train ~8% slower than C (worst 1.39× on SVR
probability training); WASM + Axum server exist as untested examples.
VISION.md created this run (north star: definitive drop-in LIBSVM
replacement; user-confirmed direction gate, axes: ergonomics, performance,
integrations, docs).

**Ranked list (impact toward VISION × effort):**
1. Migration guide: C LIBSVM / libsvm-sys2 → libsvm-rs (flags, functions, formats mapping) — high/S
2. `SvmParameter` builder with construction-time validation (parity with `svm_check_parameter`) — high/M
3. Close SVR probability-training perf gap (1.39× worst case; profile, fix allocations/cache) — high/M
4. Wire `rayon` feature: parallel cross-validation folds + internal SVR-probability CV (deterministic results preserved) — high/M
5. Feature-gated `serde` support for `SvmModel`/`SvmParameter` (text format stays canonical) — med-high/S
6. WASM as a tested target: `wasm32-unknown-unknown` build check in CI — med/S
7. docs.rs polish: document all public items, `#![warn(missing_docs)]`, doctests — med/M
8. Dense-data helpers / feature-gated ndarray interop — med/M

**Recommended cut line: after #6** (items 1–6 → beads; 7–8 deferred).

**Accepted → beads:** migration guide — libsvm-rs-966; SvmParameter builder —
libsvm-rs-zrt; SVR probability perf gap — libsvm-rs-jeh; rayon parallel CV —
libsvm-rs-eo9 (depends on jeh); serde feature — libsvm-rs-8ox; WASM CI build
check — libsvm-rs-5yo. Cut line set by user after #6; `.beads/` gitignored by
user decision. Phase 4 audit done: no dep cycles (`br dep cycles` clean, bv
DAG healthy), all verification paths/scripts confirmed to exist, quick-tier
test filters match 20 real tests. Reproducibility pass: pinned toolchains,
deterministic PRNG, and reference provenance already in place — no new beads
needed.
**Rejected:** linfa estimator bridge — diverges from drop-in-replacement
vision, large maintenance surface, VISION non-goal; `no_std` support — edge
vision not chosen, heavy effort, weak alignment; svm-toy GUI — VISION
non-goal; online/incremental learning — upstream LIBSVM doesn't have it,
parity vision excludes; kernel SIMD micro-optimization — parity-drift risk
outweighs single-digit gains while bigger wins (3, 4) remain.
**Deferred (below cut line):** docs.rs polish (#7) — valuable, after API
additions land so docs are written once; ndarray interop (#8) — wait for
builder/serde to settle the additive-API pattern first; Axum
prediction-server recipe — keep as example until WASM target proves the
integration-promotion pattern; mutation testing of solver — coverage already
93%, revisit if a parity bug ever escapes the differential suite.
