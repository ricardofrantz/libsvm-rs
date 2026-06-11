## libsvm-rs-966 — Migration guide (2026-06-10)
- AC1 six sections: PASS (docs/MIGRATION.md, 157 lines)
- AC2 all CLI flags in table: PASS (grep sweep clean, -wi included)
- AC3 API rows verified vs svm.h + lib.rs/types.rs: PASS (supervisor re-grepped
  check_parameter, svm_get_*, accessors, LoadOptions paths — all exist;
  check_parameter reachable at root via `pub use types::*`)
- AC4 README link: PASS (README.md:213)
- AC5 parity wording: PASS (numerical equivalence framing, "not bitwise")
- Follow-ups: none
## libsvm-rs-zrt — SvmParameter builder (2026-06-10)
- AC1 builder module, one method/param, docs w/ defaults: PASS (builder.rs, 243 lines)
- AC2 build() delegates to existing validate(), no duplicated rules: PASS (builder.rs:141-144)
- AC3 no-method build == default (test): PASS
- AC4 lib.rs export + Quick-Start doctest: PASS (module doc example)
- AC5 unit tests, 4+ invalid cases: PASS (7 tests; gamma/eps/cache_size/degree)
- AC6 no semantic changes: PASS (diff touches builder.rs + 2-line lib.rs only)
- Gates re-run by supervisor: 10/10 test targets ok, clippy -D warnings clean, fmt clean
- Follow-ups: none (README builder example was optional, skipped — doctest covers it)
## libsvm-rs-jeh — SVR probability perf gap (2026-06-10) — BLOCKED
- Coder stopped correctly on two stop conditions; no code changes made.
- Profile (callgrind; perf blocked by perf_event_paranoid=4): 42.6% of
  instructions in Kernel::evaluate — hotspot is kernel.rs (Ask-first scope).
  Evidence: reference/perf_svr_probability_notes.md (uncommitted).
- Differential gate: brief omitted DIFF_SCOPE=full (default=quick, 45 cfgs) —
  supervisor brief error. Full-scope re-run on THIS box: 139/78/23/10 vs
  committed Mac baseline 236/4/0/10, unmodified tree.
- ROOT CAUSE (supervisor triage): probability.rs c_rand() replicates BSD
  rand() on macOS but uses a non-glibc LCG fallback elsewhere → on Linux the
  internal probability-CV fold shuffle differs from the C reference's glibc
  rand() → different Platt fits → 23 probability label-flip fails. Parity
  baseline is platform-scoped (Mac/clang) — README claim needs that caveat.
- Follow-up: parity-fix bead to be filed (glibc rand replication on Linux +
  re-baseline + document platform scope). jeh parked pending that gate.
## libsvm-rs-4r5 — glibc rand() parity on Linux (2026-06-10)
- AC1 glibc TYPE_3 RNG under cfg(linux): PASS (util.rs; seeding/discard/output
  verified against glibc random_r.c semantics by supervisor)
- AC2 hermetic unit test, 20 hardcoded glibc constants: PASS (seed-1 sequence)
- AC3 macOS path byte-identical, fallback LCG doc-scoped: PASS (code moved to
  util.rs verbatim; cfg gates correct — macOS build untested on this box)
- AC4 public CV shuffle now consumes c_rand (C uses global rand() stream):
  PASS — mandated by bead; changes mac public-CV splits toward C parity
- AC5 DIFF_SCOPE=full on pop-os: 240 pass / 0 warn / 0 fail / 10 skip
  (was 139/78/23/10) — exceeds Mac baseline 236/4/0/10
- AC6 README + tolerance_policy platform-scope sentences: PASS
- AC7 fmt/clippy -D warnings/cargo test --all-features (136+): PASS (re-run)
- reference/ artifacts correctly reverted by coder; re-baseline deferred
- Follow-up: re-run differential on macOS + re-baseline reference/ artifacts
  (CV-shuffle change affects mac splits) — new bead filed
## libsvm-rs-jeh — SVR probability perf gap, round 2 (2026-06-10) — NEGATIVE RESULT
- AC1 pre-edit baseline models captured: PASS (.sc/jeh-models-before/)
- AC2 hot-path work: kernel.rs borrow-caching cleanup only — safe but
  essentially perf-neutral (LLVM likely already did it); no justified
  serial win found without algorithmic/cache redesign (out of scope)
- AC3 bitwise guard: PASS — supervisor independently retrained housing+heart
  post-change, cmp byte-identical to pre-edit baselines
- AC4 benchmark target worst ≤1.15: NOT MET on this box (median 1.074,
  worst 2.058 s1_t3_iris, 1.730 s3_t0_housing). Caveat: committed
  benchmark_report.md is Mac-generated; no on-box before/after pair exists,
  and post-4r5 fold splits differ — Linux ratios not comparable to the Mac
  baseline. Coder's noise claim on tiny cases is [Unverified].
- AC5 differential 240/0/0/10: PASS (re-checked from log)
- AC6 fmt/clippy/tests: PASS (re-run by supervisor)
- Verdict: documented-negative-result path accepted; serial-waste hypothesis
  exhausted within scope. Path forward = rayon parallel CV (libsvm-rs-eo9).
- Follow-up: per-platform benchmark baseline noted on libsvm-rs-dt3.
## libsvm-rs-8ox — feature-gated serde support (2026-06-10)
- Coder reported BLOCKED (exit 4) on a SUPERVISOR BRIEF BUG: the dep gate
  `cargo tree --no-default-features | grep -c serde` counts dev-deps
  (serde_json, criterion→serde). Correct gate is `-e normal`: passes with 0.
  Work itself was complete; accepted after corrected-gate verification.
- AC1 optional serde dep + feature, serde_json dev-only: PASS (-e normal = 0)
- AC2 derives + enums as pinned LIBSVM integer codes + snapshot test: PASS
- AC3 SvmModel Deserialize via raw struct + shared validate_model() factored
  into io.rs, used by BOTH text-load and serde paths: PASS (and load_model
  now validates strictly more than before — 136+19 tests still green)
- AC4 round-trip tests (classification/probability/SVR, to_bits): PASS (4)
- AC5 malicious-input serde cases: PASS — note: non-finite cases hit
  serde_json's own null-for-NaN rejection, not validate_model (which still
  guards non-JSON formats like bincode); error-not-panic holds either way
- AC6 docs (Cargo.toml/lib.rs/README): PASS
- AC7 all five gates: PASS (re-run by supervisor after touch-up)
- TOUCH-UP (supervisor): reverted coder's criterion→optional-bench-feature
  move (workaround for the buggy gate; would have made `cargo bench` a
  silent no-op without --features bench); criterion restored as dev-dep.
- Follow-ups: none
## libsvm-rs-eo9 — rayon parallel CV folds (2026-06-10)
- AC fold assignment serial + disjoint slice writes: PASS (split_at_mut fold
  slices, perm computed before par_iter; reviewed by supervisor + Codex)
- AC bitwise parity test: PASS after supervisor TOUCH-UP — coder's pinned
  digests were Linux-glibc-specific and would break macOS/Windows CI
  (build-matrix runs --all-features tests); digest asserts now
  cfg(target_os="linux"), both code paths still exercised everywhere
- AC no rayon in default tree (-e normal == 0): PASS
- AC quiet/print: PASS — with_suppressed_info depth guard, documented in lib.rs
- AC memory note + docs + CI step: PASS
- DEVIATION accepted: probability-mode outer folds stay serial — probability
  training consumes the shared c_rand stream, parallel would break determinism
  (Codex concurs). probability.rs internal CV untouched → follow-up filed.
- Gates: .sc/eo9.gate.sh GATE_EXIT=0 (supervisor re-run); full differential
  240/0/0/10 from coder log [coder-reported]; speedup ~1.98× on 5-fold CV.
- TOUCH-UP 2: restored reference/ + data/generated/ artifacts overwritten by
  supervisor's quick-scope gate run (Codex P2 — self-inflicted, not coder).
## libsvm-rs-5yo — wasm32 build check in CI (2026-06-10)
- AC new wasm-core CI job: PASS — checkout/toolchain/cache SHAs identical to
  existing pins, persist-credentials false, distinct cache key wasm32-core,
  workflow-level permissions contents:read inherited; library-only, default
  features; existing wasm-integration job untouched
- AC proof-of-teeth: PASS — coder's temporary std::os::unix probe failed the
  wasm build (E0433/E0599 snippet in report), reverted; tree shows 2 files only
- AC README one sentence: PASS
- Gates: .sc/5yo.gate.sh GATE_EXIT=0 (supervisor re-run; actionlint absent,
  skipped gracefully). Core crate compiles for wasm32 with zero cfg changes.
- Brief correction vs bead: bead background was stale (wasm-integration job
  already exists); scope narrowed to the direct core compile check.
- Follow-ups: none
## libsvm-rs-aud — parallel binary-SVC probability internal CV (2026-06-10)
- AC shuffle serial, fold bodies rand-free, scatter after collect: PASS
  (supervisor read full diff; Codex: no actionable correctness issues)
- AC serial path pure extraction (evaluate_binary_svc_probability_fold),
  degenerate class-count branches preserved as fill vecs: PASS
- AC bitwise regression — pinned rayon_parity digests unchanged, test green
  with AND without --features rayon: PASS (supervisor re-run)
- AC with_suppressed_info under rayon only, lib.rs doc sentence: PASS
- Gates: .sc/aud.gate.sh GATE_EXIT=0 (fmt/clippy/3-way tests/dep-tree/quick
  differential 45/0/0/0; reference+data artifacts auto-restored by gate)
- Informational: rayon_parity wall-clock 1.55s serial → 0.69s rayon (~2.2x)
- Scope note: SVR side needed nothing — svm_svr_probability already routes
  through eo9's parallel branch (probability=false); brief narrowed the bead.
- Follow-ups: none
## libsvm-rs-dt3 — re-baseline differential artifacts on macOS (2026-06-10)
- Supervisor-driven cycle (user-approved): no coder; Mac had NO checkout —
  fresh clone to ~/Projects/libsvm-rs (symlink → Documents/projects.nosync);
  local master rebased onto origin (+1 Mac benchmark commit) and pushed first.
- AC full suite exit 0 on macOS: PASS — 237/3/0/10 (was 236/4/0/10); warn
  resolved: gen_regression_sparse_scale_s4_t3_tuned probA drift now under
  threshold post CV-shuffle alignment. fail=0. report == json summary.
- AC artifacts regenerated + committed: PASS — also provenance/build files:
  Apple clang 17→21, new C-binary SHAs (pinned v337 commit unchanged);
  parity improved under the newer compiler.
- AC README per-platform parity table (macOS 237/3/0/10, Linux 240/0/0/10): PASS
- AC tolerance_policy.md per-platform baselines + benchmark note (Mac-generated
  bench artifacts; Linux train_probability median 1.074 / worst 2.058): PASS
- Diff confined to reference/ + README.md: PASS
- Follow-ups: none

## Batch contract — 2026-06-11 (deep-improve close-out, user-confirmed)
AUTO batch: ALL 9 open beads in `br ready` order, starting libsvm-rs-vbx.
Beads: vbx (CI wasm fix, P0) → c3r (gate sweep, P0) · 4bd (master→main +
dependabot cleanup) · cww (security audit) · 80n (CHANGELOG) · wa5 (README) ·
246 (docs.rs pass) · 4pz (semver-checks) · lx7 (housekeeping) → 2wg (release).
Stop conditions: BEFORE the publish/tag/push step of libsvm-rs-2wg (explicit
user OK required); standard BLOCKED / Ask-first stops. /clear checkpoint
after 4 impl: commits. No per-bead confirmation needed.

## libsvm-rs-vbx — fix WASM Core Build CI job (2026-06-11)
- Supervisor micro-cycle (one-line fix, no coder dispatch).
- AC pin-independent fix: PASS — added `rustup target add wasm32-unknown-unknown`
  step before the build in wasm-core; no hardcoded toolchain version.
- AC diff confined to ci.yml: PASS (1 insertion).
- wasm-integration job checked: not affected — run.sh:31 already adds the target.
- Quick gate: local `cargo build --locked -p libsvm-rs --target wasm32-unknown-unknown` EXIT=0.
- Full gate PENDING: CI green requires a push to master — awaiting user OK.
- Full gate CONFIRMED post-push: CI run 27327338121 success (incl. WASM Core
  Build); Fuzz 27327338138 success. Bead closed.
