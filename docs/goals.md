# Goal: Write the C LIBSVM → libsvm-rs migration guide   (bead: libsvm-rs-966)

## 1. Objective
Create `docs/MIGRATION.md`: the document that takes a current C LIBSVM user
(CLI or C API / libsvm-sys2) to the exact libsvm-rs equivalent. This is the
core "drop-in replacement" story of VISION.md made concrete.

## 2. Acceptance Criteria
- [ ] `docs/MIGRATION.md` exists with all six sections from the bead
      (`br show libsvm-rs-966`): CLI flag table, C API → Rust API table,
      memory-model differences, format-compatibility statement,
      libsvm-sys2 section, untrusted-input differences.
- [ ] Every flag in the usage text of `bins/svm-train-rs/src/main.rs`,
      `bins/svm-predict-rs/src/main.rs`, `bins/svm-scale-rs/src/main.rs`
      appears in the CLI table.
- [ ] Every API-table row verified against `vendor/libsvm/svm.h` and
      `crates/libsvm/src/lib.rs` exports — no invented names.
- [ ] README.md links to the guide (one line, in the existing docs area).
- [ ] Parity wording reuses the README "Parity Claim" framing — numerical
      equivalence within tolerance, never bitwise.

## 3. Verification
- `for f in s t d g r c n p m e h b q v w; do grep -q -- "-$f" docs/MIGRATION.md || echo "MISSING -$f"; done`  (no output = pass; `-wi` counts for `w`)
- Every markdown link target in MIGRATION.md exists:
  `grep -oP '\]\(\K[^)#]+' docs/MIGRATION.md | while read -r p; do [ -e "docs/$p" ] || [ -e "$p" ] || echo "BROKEN $p"; done`
- `grep -q 'MIGRATION' README.md`

## 4. Scope
✅ ALWAYS:    docs/MIGRATION.md (new), README.md (one link line only)
⚠️ ASK FIRST: any other README change
🚫 NEVER:     source code, vendor/, reference/, .github/, Cargo.*

## 5. Non-Goals / Constraints
- Documentation only — zero code changes this cycle.
- Do not overclaim parity; do not document features that don't exist (verify
  each claim by reading the named source file first).

## 6. Context Pointers
- `br show libsvm-rs-966` — full background, reasoning, section-by-section spec.
- `VISION.md` — the north star this guide serves.
- Sources of truth: `vendor/libsvm/svm.h`, `bins/*/src/main.rs`,
  `crates/libsvm/src/lib.rs`, `reference/tolerance_policy.md`,
  README "Parity Claim" section.
- Skills: humanizer (run over the prose before finishing).

## 7. Task Breakdown
1. Read the three CLI main.rs usage texts + svm.h → draft both tables.
2. Write sections 3–6 (memory model, formats, libsvm-sys2, untrusted input).
3. Add README link; run the Verification checks; humanize the prose.

## 8. Stop Conditions
- DONE when all Acceptance Criteria pass and Verification is clean.
- STOP and report if any mapped item cannot be verified in source (do not
  guess a mapping), or if a needed file is missing.
