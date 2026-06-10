# Improvement Ledger

Decisions about improvement ideas live here (accepted/rejected/deferred and
why). Work lives in the bead database. One entry per deep-improve run.

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
