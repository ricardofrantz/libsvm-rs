# Goal: Merge dependabot PRs + migrate master→main   (bead: libsvm-rs-4bd)

SUPERVISOR-DRIVEN CYCLE (user-directed 2026-06-11): all steps are gh/push
operations; no coder dispatch. User explicitly asked to MERGE (take) the 5
dependabot PRs and rename master→main.

## 1. Steps
1. Merge PRs #4, #5, #3, #1 (squash). After #1: add explicit
   `toolchain: 1.75.0` to the MSRV job (ci.yml:93) — PR #1 repoints the
   1.75.0 tag SHA to the stable tag SHA, which would silently change the
   MSRV job's toolchain (dtolnay tag commits encode the toolchain).
   Fuzz + dynamic-matrix jobs already pass explicit `toolchain:` inputs.
2. Merge PR #2 (criterion 0.5.1→0.8.2, major); pull; `cargo bench --no-run`
   — fix bench compile breaks if any (criterion 0.8 API).
3. Update workflow branch refs master→main (ci.yml:5,7; gitleaks.yml:5,7;
   fuzz.yml:5; scientific-demo-benchmark-scheduled.yml:13); grep residuals.
4. `git branch -m master main && git push -u origin main`; switch GitHub
   default via API; `git push origin --delete master`;
   `git remote set-head origin -a`.
5. Confirm CI green on main.

## 2. Acceptance Criteria
- [ ] All 5 dependabot PRs merged (not closed); branches gone.
- [ ] MSRV job still checks 1.75.0 explicitly.
- [ ] Benches compile under criterion 0.8.
- [ ] Default branch is main; no stale master refs in workflows/docs;
      old master branch deleted.
- [ ] CI green on main end-to-end.
