# Pre-registration

**Project:** DeltaAudit, *What the metric hides: predicting unseen gene knockdowns*
**Registered:** 2026-11-09, by the commit tagged `prereg` in <https://github.com/asifuddin01/DeltaAudit>
**Status at registration:** data downloaded and checksum-pinned; pseudobulks, split-half noise ceilings and systematic-variation descriptives computed on all perturbations (no model); splits written. **No model has been fitted or scored on any fold.**

The machine-readable form of this document is `configs/eval.yaml` and `configs/models.yaml`. Analysis code reads those files, so the files at the `prereg` tag are what is registered. Where this text and those files disagree, the files win, and the disagreement is reported as a deviation.

## 1. Questions and hypotheses

| ID | Question | Hypothesis (direction fixed now) |
|---|---|---|
| Q1 | How much of the headline score is the average perturbation effect? | **H1:** On raw PearsonΔ, the train-mean baseline B1 scores within 0.05 of the best learned model, and B1's per-perturbation raw_r correlates with that perturbation's similarity to the mean effect (Spearman ρ > 0.5). |
| Q2 | Is there any perturbation-specific skill? | **H2:** Under the perturbation-specific metric (pspec_r), the linear baseline B2 is > 0 but far below the split-half noise ceiling. |
| Q3 | Does a learned non-linear model add anything over linear? | **H3:** M1 (MLP, no prior) is *not* significantly better than B2. |
| Q4 | Does a gene-graph prior help, and is it the biological content? | **H4a:** M2 (GNN + GO) beats M1. **H4b:** M2 beats M3 (same GNN, GO graph with gene labels shuffled). |

The headline findings behind H1–H3 are not ours: Ahlmann-Eltze, Huber & Anders (*Nat Methods* 2025) and Viñas Torné et al. (Systema, *Nat Biotechnol* 2025). This project re-computes them; it does not cite their numbers. A null on any hypothesis is a result and will be reported as such.

## 2. Data

- **Primary:** Replogle et al. 2022 K562 essential CRISPRi, GEARS-processed (Dataverse file 7458695), 5,000 expression genes. After QC: **1,070** single-gene perturbations, of which **406** have their target in the expression panel.
- **Replication (secondary):** Norman et al. 2019 K562 CRISPRa (GEARS, Dataverse file 6154020): 105 singles, 131 doubles. CRISPRa, so effects are activation.
- **GO prior:** GO human GAF, release 2026-08-05.
- Raw files are pinned by SHA-256 in `data/checksums.sha256`; descriptions are in `data/SOURCES.md` and `data/cards/`.
- **Unit of modelling:** pseudobulk mean of log-normalized expression per perturbation; Δ = pert − control.
- **Exclusions (fixed):** perturbations with fewer than 30 cells (22 in Replogle, 0 in Norman). Weak knockdowns are **not** excluded. GEARS labels that differ only in order (`A+ctrl` / `ctrl+A`) are merged. No other exclusions.

## 3. Splits and leakage guards

- **Replogle and Norman singles (`unseen_single`):** 5 folds over single-gene perturbations, seed 0, stratified by "target in expression panel". Each perturbation is in exactly one test fold. **Inner validation** = 10% of each fold's training perturbations (same stratification). Norman doubles are not in these splits (strict: no held-out gene appears in any training perturbation).
- **Norman doubles (`seen_2of2`):** test = the 131 doubles whose two singles are both measured; train = all 105 singles.
- **Enforced in code** (`deltaaudit verify`, CI, and before every fit via `fit_rows`): zero target-gene overlap between train+val and test (a double counts both genes); ctrl in no list; train/val/test disjoint; test folds partition the perturbation set; guide overlap would be checked if guide ids existed. Replogle uses one dual-guide construct per gene, so gene-disjoint means guide-disjoint.
- **Train-only statistics:** every fitted quantity (μ, PCA and spectral embeddings, ridge weights, kNN neighbour sets, network weights) receives only train (+val) rows.
- **Known, unfixed leakage:** GEARS chose the 5,000-gene panel using all cells, including test-perturbation cells. A sensitivity check re-selects genes on control + training cells (§8).

Split files and their hashes are in §10.

## 4. Models

Every model outputs Δ̂_q for a held-out perturbation q. Details and grids are in `configs/models.yaml`.

| ID | Model |
|---|---|
| B0 | No change: Δ̂ = 0 |
| B1 | Train mean: Δ̂ = μ (mean Δ over the fold's train+val perturbations) |
| B2 | Ridge-linear, matching `const-ae/linear_perturbation_prediction-Paper` at commit `95e831e`, `run_linear_pretrained_model.R`. Gene embedding G = top-K PCs of the training pseudobulk **expression** matrix (genes × training conditions incl. ctrl, column-centred). Perturbation embedding = the row of G for the target gene (ctrl = 0). Fit Y − c = G W Pᵀ with two-sided ridge λ, where c = per-gene mean over training conditions. Predict Δ̂ = G W p_q + c. Requires the target in the panel. *Deviation from BUILD_GUIDE E1, which describes PCA of Δ: we follow the reference code.* |
| B2-GO | As B2, but p_q is a K-dim spectral embedding of the GO graph (smallest non-trivial eigenvectors of the normalized Laplacian), so it exists for out-of-panel targets with GO annotations. |
| B3 | kNN-GO: Jaccard-weighted mean Δ of q's GO-graph neighbours among train+val perturbations; μ if none. |
| B4 | Additive (Norman doubles only): Δ̂_{A+B} = Δ_A + Δ_B. |
| M1 | MLP, no prior: Δ̂ = μ_train + MLP(X[q]); X = [E_Δ (32-d PCA of Δ_trainᵀ, train only), E_ctrl (32-d PCA of control-cell gene–gene correlation), missing indicator]. 2 layers, width 256, GELU, dropout 0.2. |
| M2 | GNN + GO: H = Â² X W₀ (SGC, 2 hops, symmetric normalization with self-loops), then the same MLP head on H[q]. |
| M3 | As M2 with the GO graph's node labels permuted (3 permutations; topology identical). |
| M3b | As M2 with a degree-preserving double-edge-swap rewired graph (secondary null). |

**GO graph:** per gene, its set of GO terms (all aspects, all evidence codes). Jaccard similarity; keep the top 20 neighbours with Jaccard ≥ 0.1; symmetrize. Coverage is reported.

**Selection uses inner validation only.** B2 and B2-GO choose K ∈ {5, 10, 20, 50} and λ ∈ {0.01, 0.1, 1, 10} per fold by mean validation pspec_r (fitted on train), then refit on train+val. Learned models train on train with AdamW (lr 1e-3, weight decay 1e-4), MSE loss, full batch, ≤ 300 epochs, early stopping on validation pspec_r (patience 30); no refit. Learned models run 5 seeds × 5 folds. A perturbation's score is the **mean over seeds** (for M3, over seeds and the 3 permutations).

## 5. Metrics

Per test perturbation p, over the expression genes **excluding p's target gene(s)**. μ = mean Δ over the fold's train+val perturbations.

- **raw_r** = Pearson(Δ̂_p, Δ_p), the headline "PearsonΔ"; raw_r20 on the top 20 genes by |Δ_p|; mse, mse20.
- **pspec_r** = Pearson(Δ̂_p − μ, Δ_p − μ). **The primary metric.** If std(Δ̂_p − μ) < 1e-8 (e.g. B1), pspec_r = 0 and the row is flagged degenerate. pspec_r20 on the top 20 by |Δ_p − μ|.
- dir20, pspec_dir20 (sign agreement on the top 20); centroid_acc and pds (rank among the test fold's perturbations, ties averaged).
- **Noise ceilings:** split-half Pearson between the two half-pseudobulk Δs (each half with its own control half), raw and perturbation-specific (with the fold's μ). Systematic variation: sys_cos(p) = cos(Δ_p, μ_all).

If a score is undefined (zero variance, apart from the degenerate rule above), that perturbation is dropped from comparisons involving it, and the count is reported.

## 6. Statistics

- **Statistical unit:** the held-out perturbation, pooled over the 5 folds. No p-values over cells or genes.
- **Primary analysis set:** Replogle perturbations whose target is in the expression panel (n = 406), so every model can predict every perturbation.
- For each comparison: d_p = score_A(p) − score_B(p); obs = mean(d). 95% CI = percentile bootstrap, 10,000 resamples of perturbations, seed 20261109. Two-sided p = (1 + #{|mean(d*) − obs| ≥ |obs|}) / (B + 1). Also reported: Wilcoxon signed-rank p, median d, fraction of perturbations with d > 0.
- **Primary family** (Holm, α = 0.05; metric pspec_r unless stated):

| # | Comparison | Hypothesis |
|---|---|---|
| C1 | best learned model vs B1, on **raw_r** | H1 |
| C2 | B2 vs B1 | H2 |
| C3 | M1 vs B2 | H3 |
| C4 | M2 vs M1 | H4a |
| C5 | M2 vs M3 (mean of 3 permutations) | H4b |
| C6 | B2-GO vs B2 | prior in a linear model (two-sided, no direction) |

"Best learned model" in C1 is whichever of M1 or M2 has the higher mean inner-validation raw_r, decided before any test scoring.

## 7. Decision rules (what counts as support and as a null)

- **H1 supported** if the upper bound of the 95% CI of mean(raw_r[best] − raw_r[B1]) is < 0.05 **and** Spearman ρ(B1 raw_r, sys_cos) > 0.5 (ρ reported with a bootstrap CI). Otherwise not supported. C1's Holm p tests d = 0 and is reported either way.
- **H2 supported** if C2 has d > 0 with p_holm < 0.05, **and** mean B2 pspec_r < 0.5 × mean noise ceiling (pspec) over the same perturbations. If p_holm ≥ 0.05: no detectable specific skill (null).
- **H3** predicts a null. It is **contradicted** if C3 has d > 0 with p_holm < 0.05; otherwise it stands.
- **H4a / H4b supported** if C4 / C5 has d > 0 with p_holm < 0.05. Otherwise the prior / its content shows no detectable benefit (null), reported with the CI.
- **C6** is reported with its CI and p_holm; no direction is claimed.

## 8. Secondary and sensitivity analyses (reported, not Holm-gated)

- Every model on every metric; B3, M3b; the full perturbation set (B2 falls back to μ for out-of-panel targets); the out-of-panel subset for C4 and C5, stratified by whether the target has a GO neighbour among training perturbations.
- **Norman replication:** the same models and comparisons C1–C6 on singles (Holm within Norman reported separately). Doubles: B4 vs B1 vs B0 on the 131 seen-2/2 doubles.
- **Sensitivity:** μ = all-perturbation centroid instead of μ_train; top-20 genes vs all genes; common support vs full set; cluster bootstrap resampling GO-similarity clusters (average linkage on 1 − GO Jaccard, cut at 0.8) instead of single perturbations; gene-panel re-selection on control + training cells (fold 0).
- Optional, if time allows: VCC 2025 H1 replication; CollecTRI prior (M4); GEARS reference run.
- Anything not listed here is labelled **exploratory**.

## 9. Deviations

Any departure from this document is logged in `docs/DECISIONS.md` and listed in the write-up under "Deviations from pre-registration", with its reason. The primary family is not changed after test scoring begins.

## 10. Frozen artefacts (SHA-256)

| File | SHA-256 |
|---|---|
| `data/splits/norman/doubles_seen_2of2.json` | `172bcf63b842ab90e8a25a84d7933388ff300e9da9d3bf7f887680527e5eb9bc` |
| `data/splits/norman/fold_0.json` | `79d6515b80fde2d2f79b4e461f591c8edca579d2db70025b44cd3479aa177ad2` |
| `data/splits/norman/fold_1.json` | `7505bb8d9086c8e11204aeefd580d62eeebf34e8cc84e509c91928ae040582ec` |
| `data/splits/norman/fold_2.json` | `0984cb10bba3d5169ebc71ef67bec5174354bbddae63a0eb9b9c234f92c5434f` |
| `data/splits/norman/fold_3.json` | `042a1c652ea0e68b7d0a85e1494419456b041e13e3b8e3c12467c658f2c6ff5c` |
| `data/splits/norman/fold_4.json` | `197615f139dabf63d6a6bc695239b84485281ccdbe90c68f7f9deeedbf84fe90` |
| `data/splits/norman/summary.json` | `f1d1951de095c7dcb8e2f623ed79bd728535c5716a0fe4fb630b997fc59ac995` |
| `data/splits/replogle_k562/fold_0.json` | `8eaa23cc1cae40589207bdb99fe0be494a296e96251a2f8209566f10ff788f53` |
| `data/splits/replogle_k562/fold_1.json` | `7ac8117ac44a8c150a3b9e06415b9f8f10e1bea433c0ed8dd97743069e745b1a` |
| `data/splits/replogle_k562/fold_2.json` | `01f6ca8c042c6c41dbbd1b3625d549b22750ec0ceba5d6889cb0f8e5a6f0bdaf` |
| `data/splits/replogle_k562/fold_3.json` | `3781d9b643ea48b3e47611052fdd1d6a00deca184713d3647f0c541b17bbfd1c` |
| `data/splits/replogle_k562/fold_4.json` | `08efdb2cb1a0b63d6f8b68f93e2dde62b907ec66e8b8192f2e8be48044060a85` |
| `data/splits/replogle_k562/summary.json` | `01d701d9d65c9ed7031cfd7bcf63a6c2c6e64c5ae936717765b64222d2d9f813` |
| `configs/eval.yaml` | `fe3f88abedc3ecca01ea2426a3ac7e4ecf35e1d1d4c2c22ffa9f7b9d9d6895c3` |
| `configs/models.yaml` | `9323c672aa35bd60b44e24d528fc5c45530ebcc90ed759cd9cf0298cdafa24ed` |
| `data/processed/norman/meta.parquet` | `7fc10042ef299f4163894e00a507b8c328c05b12c41e4f7c4812e2b11498d171` |
| `data/processed/replogle_k562/meta.parquet` | `c8fb4c2564b1659e43851f978d879ff0ae0668ca7271cee3b0c6466ac2beed46` |
| `uv.lock` | `5cb0ad102ca416a445fd598e41a901989073584514e76d91f6f9c65ec90cac39` |

The `data/processed/*/meta.parquet` files are not committed (size); their hashes tie the split files to the processed data that produced them.
