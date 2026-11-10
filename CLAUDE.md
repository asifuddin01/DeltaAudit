# CLAUDE.md — standing rules for this repo

Project: **DeltaAudit**. The repo, Python package and CLI are all named `deltaaudit`.
Write-up title (fixed, do not change): *What the metric hides: predicting unseen gene knockdowns*.
The full plan is in `BUILD_GUIDE.md`. Read it before starting work, and work through its phases (Part H) in order.

## Where we are
Keep this block updated at the end of every session:
- Current phase: 10 (Phase 9 done: `make sensitivity` = `deltaaudit hvg-panels` (2,000-gene panels re-selected inside GEARS' 5,000 from all cells vs control + fold-0 train cells, derived datasets `replogle_k562__hvg_{all,train}`, fold-0 refits, ~3 min) + `deltaaudit sensitivity` -> results/summary/{sensitivity.csv, sensitivity.md}. Phase 8 outputs: results/summary/{scores.csv, model_table.csv, diagnosis.json, comparisons.csv, decisions.json, ladder.csv}. Secondary/sensitivity configs in `configs/secondary.yaml`, ladder in `configs/ladder.yaml` (rung names still placeholders for the user's scoping-review ladder). VCC, M4 and GEARS-on-Colab not done (optional; VCC needs a > 5 GB download, so ask first). Phase 10 = figures, write-up rendered from results/summary, README, one-command reproduction, Colab bundle. `prereg` tag = 8f96e03; scoring commands call `require_prereg`.
- Last green commit: phase-9 (see `git log`)
- Open decisions: none. Dataset = Replogle K562 essential (user confirmed 2026-11-09). Dev box is an M1 with 8 GB RAM; preprocess peaked at 1.37 GB (Replogle, 113 s).
- Facts for later phases (from `data/cards/*.json`): Replogle has 1,092 single-gene perturbations, but only 412 targets are in the 5,000-gene panel (common support for B2), and 22 have fewer than 30 cells (dropped; 1,070 kept). GEARS lists `A+ctrl` and `ctrl+A` separately; `card.canonical_condition` merges them (Norman goes from 152 to 105 singles).
- GO coverage: Replogle 1,063/1,070 targets have a GO edge, Norman 98/105 (`results/summary/*_go_coverage.json`). B3/B2-GO reach pspec_r ≈ 0.42–0.46 on Replogle (label-permuted graph: 0.006). B3 beats the split-half ceiling for 49% of perturbations, so the ceiling is conservative (DECISIONS). This sets a high bar for M2 in C4/C5.
- Primary results (Replogle common support, n=406; read numbers from results/summary, never type them): C1 M2 > B1 on raw_r, so H1 not supported; H2 supported; C3 null (H3 stands); C4/C5 significant (H4a/H4b supported); C6 B2-GO > B2. Best learned = M2 by val raw_r on both datasets. On Norman, H1 is "supported" only because M2 is below B1 (one-sided rule, logged in DECISIONS). The cluster bootstrap (177 GO clusters) changes no primary conclusion (CI lower bounds stay > 0 for C1, C2, C4–C6; C3 still spans 0). Ladder top: M2 L6, B2-GO/B3 L5, B2/M1 L4.
- Sensitivity (results/summary/sensitivity.md): Replogle primary verdicts hold under μ_all, top-20 genes, cluster bootstrap and test-cell-free HVG selection; the full set makes C3 significant (B2 falls back to μ out of panel). The Norman replication is fragile: no comparison survives the cluster bootstrap, and C2/C4–C6 fail under μ_all (which includes the doubles). Logged in DECISIONS.
- Every pipeline .py is in code_hash, so code edits make the fit manifests stale. Phases 8 and 9 each re-ran the full chain once and got byte-identical summaries.
- B0 scores pspec_r > 0 (Replogle ~0.06–0.09) while B1 = 0 by construction; read pspec_r against B0 too (logged in DECISIONS as exploratory).

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
