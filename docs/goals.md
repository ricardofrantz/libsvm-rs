# Goal: Feature-gated serde support for SvmModel/SvmParameter   (bead: libsvm-rs-8ox)

## 1. Objective
Optional `serde` feature so embedding applications can carry models in their own
serialized state (JSON/bincode) without round-tripping through the text format.
Text format stays canonical for C interchange (VISION point 2). The bulk of the
work is the deserialization trust boundary — serde must enforce the SAME
validation as io.rs load_model. Full spec: `br show libsvm-rs-8ox`.

## 2. Acceptance Criteria
- [ ] `serde = { version = "1", features = ["derive"], optional = true }` +
      a `serde` feature in crates/libsvm/Cargo.toml; serde_json as
      dev-dependency only. No new default deps:
      `cargo tree --no-default-features --prefix none | grep -c serde` == 0.
- [ ] cfg_attr derives for SvmNode, SvmParameter, SvmType, KernelType (+ only
      what compilation requires from types.rs). Enums serialize as LIBSVM
      integer codes; the choice is documented and pinned by a snapshot test.
- [ ] SvmModel Deserialize goes through a validating path: deserialize a
      private raw struct, then run a shared `validate_model()` factored out of
      io.rs — ONE invariant list used by both the text-load and serde paths
      (header sanity, label/coef length invariants, NaN/Inf rejection).
- [ ] Round-trip tests (new crates/libsvm/tests/serde_roundtrip.rs): train on
      heart_scale (classification + probability) and an SVR model; model →
      serde_json → model → save_model text == original text; f64 fields
      asserted via to_bits, not approximate equality.
- [ ] Malicious-input tests: serde-path cases in tests/malicious_input.rs
      mirroring the text-format cases (length mismatches, non-finite
      gamma/rho/sv_coef, negative degree) — each returns SvmError, never panics.
- [ ] Docs: Cargo.toml feature comment, lib.rs feature docs, README one
      paragraph (serde vs save_model guidance).
- [ ] Gates: `cargo test --locked --features serde -p libsvm-rs` green;
      `cargo test --locked --all-features` green;
      `cargo test --locked --no-default-features` green;
      `cargo clippy --locked --all-targets --all-features -- -D warnings`;
      `cargo fmt --all -- --check`.

## 3. Verification
- Quick: `cargo test --locked --features serde -p libsvm-rs serde`
- Full: the five gate commands in AC7 (none are long; no .sc/ logging needed).

## 4. Scope
✅ ALWAYS:    crates/libsvm/Cargo.toml, crates/libsvm/src/types.rs, io.rs
              (validate_model factoring only — no behavior change to text I/O),
              lib.rs (docs/feature), tests/malicious_input.rs,
              tests/serde_roundtrip.rs (new), README.md (one paragraph)
⚠️ ASK FIRST: SvmProblem serde (only if literally a bare derive with zero
              validation surface — otherwise defer and note it), any pub API
              signature change beyond additive derives + validate_model
🚫 NEVER:     vendor/, scripts/, .github/, bins/, io.rs text-format semantics,
              solver/kernel/probability code, workspace root Cargo.toml deps

## 5. Non-Goals / Constraints
- serde is NOT a model-interchange format with C — text format remains the
  only compat surface; say so in the docs paragraph.
- Lockfile will change (serde/serde_json pins) — that is expected and allowed.

## 6. Context Pointers
- `br show libsvm-rs-8ox` — full spec incl. trust-boundary reasoning.
- io.rs load_model validation (v0.8.1 security work) — the invariant source.
- `VISION.md` points 2 and 5 — text format canonical; additive features only.

## 7. Stop Conditions
- DONE when criteria pass. STOP if validation cannot be cleanly shared between
  the two paths, or an ⚠️ item is needed.
