# CLAUDE.md — standing rules for this repo

Project: **DeltaAudit**. The repo, Python package and CLI are all named `deltaaudit`.
Write-up title (fixed, do not change): *What the metric hides: predicting unseen gene knockdowns*.
The full plan is in `BUILD_GUIDE.md`. Read it before starting work, and work through its phases (Part H) in order.

## Where we are
Keep this block updated at the end of every session:
- Current phase: 2 (Phase 1 done: download/extract/pin/verify, SOURCES.md, dataset cards for replogle_k562 + norman)
- Last green commit: phase-1 (see `git log`)
- Open decisions: none. Dataset = Replogle K562 essential (user confirmed 2026-11-09). Dev box is an M1 with 8 GB RAM; card scans peaked at 0.78 GB.
- Facts for later phases (from `data/cards/*.json`): Replogle has 1,092 single-gene perturbations, but only 412 targets are in the 5,000-gene panel (common support for B2), and 22 have fewer than 30 cells. GEARS lists `A+ctrl` and `ctrl+A` separately; `card.canonical_condition` merges them (Norman goes from 152 to 105 singles).

## Hard rules (never break these)
1. **No test peeking.** Nothing is scored on a test fold until `reports/preregistration.md` is committed and tagged `prereg`. Every hyperparameter or model choice uses inner-validation perturbations only.
2. **Split by whole perturbation.** Train and test must share zero target genes (both genes of a double count) and zero guides. Run `deltaaudit verify` (checksums + leakage asserts) before every fit and in CI.
3. **Train-only statistics.** μ, PCA embeddings, scalers, ridge weights and kNN neighbours are computed from `train` (+`val`) indices only. Pass index arrays explicitly; never fit on the full matrix "for convenience".
4. **No hand-typed numbers.** Every number in `reports/writeup.md`, `README.md` and the site post is rendered from `results/summary/*`. If a number is needed and not in a summary file, add it to the pipeline.
5. **The statistical unit is the held-out perturbation.** Use paired bootstrap over perturbations (B = 10,000, seeded) and Holm across the pre-registered family. Do not compute p-values over cells or genes.
6. **The target gene is excluded** from every evaluation metric.
7. **Data safety.** Downloads go into their own empty directory under `data/raw/`, are extracted with path-traversal protection, and are never executed or imported. Load no pickles unless the GAF route has failed and the pickle's checksum is pinned. Never modify `data/raw/`.
8. **Pinned integrity.** Every raw file has a SHA-256 in `data/checksums.sha256`. A mismatch is a hard failure; do not "update the checksum" to make it pass without telling the user.
9. **Mac memory.** Open h5ad files with `backed="r"` and pseudobulk in chunks. Never densify a full cell × gene matrix.
10. **Null results are results.** If the prior does not help, report that. Do not add unregistered comparisons to the primary family after seeing results; label anything new "exploratory".

## How to work
- One phase at a time. A phase is done when its **Accept** criteria in BUILD_GUIDE.md are met, `make test` is green, and you have committed (`phase-N: <summary>`).
- Write the test first for metrics, splits, leakage and statistics, using `tests/fixtures/tiny.h5ad` (synthetic, with known answers).
- Every pipeline step writes `MANIFEST.json` (git SHA, config hash, input hashes, device, versions, wall time). `make all` skips a step when its manifest is up to date.
- Device order: cuda > mps > cpu. Time a smoke run before launching any grid. If the projected time is over 3 h on the Mac, prepare the Colab bundle (`make colab-bundle`) and tell the user.
- Ask the user before deleting files, changing the pre-registered family, or downloading anything over 5 GB (VCC).
- Keep `docs/DECISIONS.md` as a running log: date, decision, reason, alternatives rejected.
- End each session by updating "Where we are" above.

## Commands
```
make env | data | verify | baselines | train | evaluate | stats | report | all | test | tiny
deltaaudit --help
```

## Style
Python 3.11; type hints; `ruff` + `black` defaults; small pure functions in `eval/` and `data/`; config in YAML, no magic numbers in code; figures produced only by `report/figures.py`.
