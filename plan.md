# libsvm-rs — Full Compliance & Security Audit Plan

Status: **draft for approval** · Scope: license compliance + security audit · No edits applied yet.

License changes are a STOP trigger — every item below is a recommendation. Nothing
is implemented until you approve the specific item.

---

## Part A — License Compliance

### A.0 Current state (verified)

| Item | State | Verified by |
|---|---|---|
| Root `LICENSE` | BSD-3-Clause, `Copyright (c) 2026, Ricardo Frantz` (sole holder) | Read LICENSE |
| `Cargo.toml` | `license = "BSD-3-Clause"` (consistent with LICENSE) | Read Cargo.toml:14 |
| Vendored upstream | LIBSVM **v337**, `cjlin1/libsvm` commit `6b90713`, redistributed verbatim under `vendor/libsvm/` (C/C++ + Python iface + tools + datasets) | `reference/libsvm_upstream_lock.json` |
| Upstream `COPYRIGHT` | **MISSING from repo** — never tracked in git | `git log --all -- vendor/libsvm/COPYRIGHT` empty |
| Upstream holder/text | `Copyright (c) 2000-2023 Chih-Chung Chang and Chih-Jen Lin`, BSD-3-Clause | Fetched from pinned commit `6b90713` |
| Root `NOTICE` | **absent** | `ls NOTICE` |
| Rust source headers | no SPDX / copyright headers | `grep -rln SPDX crates/ bins/` empty |
| Branch | local + remote default both `master` | `git branch -a` |

### A.1 Findings (severity-ordered)

1. **🔴 Critical — vendored source redistributed without its required license.**
   `vendor/libsvm/` ships `svm.cpp`, `svm.h`, `svm-train.c`, `svm-predict.c`,
   `svm-scale.c`, `python/`, `tools/` verbatim, but the upstream `COPYRIGHT`
   file is absent. BSD-3-Clause clause 1 requires source redistributions to
   retain the copyright notice + disclaimer. `vendor/libsvm/README` itself says
   *"Please read the COPYRIGHT file"* — a file not present. **This is the one
   actual license violation.**

2. **🟠 High — root `LICENSE` does not retain upstream copyright.** Project is an
   explicit port/reimplementation of LIBSVM *and* redistributes its source. The
   Chang & Lin copyright appears in no LICENSE/NOTICE. Your re-expressed Rust code
   may carry its own copyright, but the upstream notice must be retained for the
   derived + vendored material.

3. **🟡 Medium — no `NOTICE`/provenance file.** README names LIBSVM and cites the
   papers but states no derivative-work + retained-copyright relationship.

4. **🟡 Medium — vendored datasets provenance.** `data/heart_scale`,
   `data/iris.scale`, `data/housing_scale` and `vendor/libsvm/heart_scale` are
   LIBSVM-distributed datasets (ultimately UCI-derived). Worth a provenance line;
   low legal risk but currently unattributed.

5. **🔵 Info — license-family choice.** Global default is Apache-2.0, but **keeping
   BSD-3-Clause is the lower-risk choice** for a LIBSVM derivative (same license
   as upstream, no relicensing question). Recommendation: **do not** migrate to
   Apache-2.0 here. (Explicit deviation from the license skill default, justified
   by the derivative relationship.)

6. **🔵 Info — branch `master`.** Skill prefers `main` (`both-master` state).
   Cosmetic relative to the above.

### A.2 Remediation checklist

- [ ] **A2.1 (Critical)** Add `vendor/libsvm/COPYRIGHT` with the verbatim upstream
      text for v337 (`Copyright (c) 2000-2023 Chih-Chung Chang and Chih-Jen Lin`).
      - Acceptance: file exists; `vendor/libsvm/README` reference resolves.
      - Verify: `test -f vendor/libsvm/COPYRIGHT && grep -q "Chih-Jen Lin" vendor/libsvm/COPYRIGHT`
- [ ] **A2.2 (High)** Add root `NOTICE`: state libsvm-rs is a derivative of LIBSVM
      by Chih-Chung Chang and Chih-Jen Lin (BSD-3-Clause), original copyright
      retained, upstream license at `vendor/libsvm/COPYRIGHT`; add Ricardo
      provenance/contact.
      - Verify: `test -f NOTICE`
- [ ] **A2.3 (High)** Amend root `LICENSE` to a dual-copyright block — retain
      `Copyright (c) 2000-2023 Chih-Chung Chang and Chih-Jen Lin` alongside
      `Copyright (c) 2026 Ricardo A S Frantz` — or reference `NOTICE` for upstream
      holders. Keep BSD-3-Clause body so GitHub license detection still works.
      - Verify: `grep -q "Chih-Jen Lin" LICENSE`
- [ ] **A2.4 (Medium)** README License section: expand from one line to name the
      derivative relationship and point to `NOTICE` + `vendor/libsvm/COPYRIGHT`.
- [ ] **A2.5 (Medium)** Add a short provenance note for `data/` datasets (source =
      LIBSVM datasets page; original UCI attribution where applicable).
- [ ] **A2.6 (Optional)** Branch migration `master → main`:
      `git branch -m master main && git push -u origin main && gh repo edit --default-branch main`,
      then (only after confirming GitHub default moved) `git push origin --delete master`.
      Do not run on a dirty tree.
- [ ] **A2.7 (Verify)** `cargo deny check licenses` passes; `gh repo view --json licenseInfo`
      still detects BSD-3-Clause.

### A.3 Decisions needed from you
- D1: Keep BSD-3-Clause (recommended) vs migrate to Apache-2.0?
- D2: Dual-copyright in `LICENSE` (A2.3) vs upstream copyright only in `NOTICE`?
- D3: Do the `master → main` migration now or defer?

---

## Part B — Security Audit

Context: pure-Rust crate whose primary attack surface is **parsing untrusted
LIBSVM model/problem files**. Prior commit `247d6cf` ("harden trust boundaries")
already added loader caps, structural model validation, fuzz targets, `deny.toml`,
and pinned toolchains. This audit **verifies and extends** that work rather than
assuming it.

### B.0 Current state (verified)
- `unsafe` in `crates/`+`bins/`: effectively none (1 grep hit, to confirm it's not real unsafe).
- Fuzz targets: `parse_model.rs`, `parse_problem.rs` with seeded corpus incl. malicious cases.
- `deny.toml`: advisories + license allow-list + source pinning + bans
  (`openssl-sys`, `ring`) + `unknown-git = deny`.
- Parsers / attack surface: `io.rs` (model + problem loaders), `types.rs`,
  CLI arg parsing in all three `bins/`.

### B.1 Audit tasks (each = scan + finding + verification)

- [ ] **B1.1 Untrusted-input parsers (`io.rs`).** Review model + problem loaders for:
      unbounded allocation from attacker-controlled counts (`nr_class`, `total_sv`,
      feature indices), integer overflow on `i32` index math, line-length /
      node-count caps, and `unwrap()/expect()/panic!/[]`-index on parsed data.
      - Acceptance: every loop/alloc sized from file input has a documented cap;
        no panic path on malformed input.
      - Verify: `grep -rn "unwrap\|expect\|panic!\|unreachable!" crates/libsvm/src/io.rs`
        — each remaining one justified.
- [ ] **B1.2 Panic-freedom on parse path.** Confirm loaders return `Result`, never
      panic, across `predict`, `types`, `io`.
      - Verify: run both fuzz targets ≥5 min each, zero crashes:
        `cargo +nightly fuzz run parse_model -- -max_total_time=300` and same for `parse_problem`.
- [ ] **B1.3 Arithmetic safety.** Check kernel/solver/cache for overflow in index
      and size arithmetic (`cache.rs` line/column products, `qmatrix.rs` strides).
      - Verify: `cargo build` clean; consider `RUSTFLAGS=-Coverflow-checks=on cargo test`.
- [ ] **B1.4 `unsafe` audit.** Resolve the single grep hit; target
      `#![forbid(unsafe_code)]` on `crates/libsvm` if none remains.
      - Verify: `grep -rn "unsafe" crates/libsvm/src` and, if clean, add the lint.
- [ ] **B1.5 CLI robustness (`bins/`).** Manual arg parsers (`-wi`, `-b`, `-q`):
      check index-out-of-bounds on flags missing operands, numeric parse errors,
      and untrusted file paths.
      - Verify: each bin handles `--` with missing operand without panic.
- [ ] **B1.6 Dependency / advisory scan.**
      - Verify: `cargo deny check` (all) and `cargo audit` (RUSTSEC) — zero
        unignored advisories; `advisories.ignore = []` stays empty or each entry justified.
- [ ] **B1.7 Supply chain.** Confirm `unknown-git = deny`, registry pinning, and the
      `openssl-sys`/`ring` bans still hold; scan `Cargo.lock` for unexpected
      transitive native-build crates.
      - Verify: `cargo deny check bans sources`.
- [ ] **B1.8 Resource-exhaustion / DoS.** Confirm loader caps bound memory for a
      hostile small file (e.g. huge `total_sv` corpus case already present).
      - Verify: feed `fuzz/corpus/parse_model/huge_total_sv.model` and
        `huge_nr_class.model` to `defensive_load` example; bounded memory, clean error.
- [ ] **B1.9 Integer/float format round-trip.** `%.17g` formatter and parser:
      check for locale assumptions, NaN/Inf handling in model I/O (untrusted
      `rho`, `coef`, decision values).
      - Verify: load a model with `inf`/`nan` fields → rejected or handled, no panic.
- [ ] **B1.10 CI gates.** Confirm `ci.yml` runs `cargo deny`, fuzz smoke, and tests
      on PRs so regressions are caught.
      - Verify: read `.github/workflows/ci.yml` + `fuzz.yml`.

### B.2 Security deliverable
A `SECURITY_AUDIT.md` (or `reference/security_audit_2026-06.md`) recording: tasks
run, commands, findings with severity, fixes vs accepted-risk, and fuzz runtime /
coverage. Optionally a `SECURITY.md` disclosure policy if none exists.

### B.3 Decisions needed from you
- D4: Run the audit as **read-only findings** first (recommended), or audit-and-fix
      in one pass?
- D5: Fuzz duration for sign-off — quick smoke (5 min/target) vs extended
      (e.g. 1 h/target in CI nightly)?
- D6: Add `#![forbid(unsafe_code)]` if B1.4 confirms zero unsafe?

---

## Suggested execution order & commits
1. A2.1 `COPYRIGHT` restore (critical, isolated commit).
2. A2.2–A2.5 NOTICE + LICENSE + README + dataset provenance (license commit).
3. B1.* security audit → `SECURITY_AUDIT.md` (findings commit), then fixes per finding.
4. A2.6 branch migration (separate, last).

Each group is a small, single-purpose commit. Workers return diffs; coordinator stages.
