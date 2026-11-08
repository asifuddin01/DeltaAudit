# Decisions log

| Date | Decision | Reason | Alternatives rejected |
|---|---|---|---|
| 2026-11-09 | Primary dataset: Replogle 2022 K562 essential (GEARS-processed) | User confirmed in Phase 0; ~1,000+ single-gene knockdowns, well above any thesis-dataset alternative offered | Thesis dataset (declined) |
| 2026-11-09 | Environment: `uv` + `uv.lock`, Python 3.11, hatchling build | Fast, exact lockfile, works on Mac and CI | mamba / environment.yml |
| 2026-11-09 | CPU-only torch wheel index on Linux (CI/Colab-free path); PyPI torch on macOS | Avoids ~2 GB CUDA wheels in CI; macOS wheel includes MPS | Single PyPI index everywhere |
| 2026-11-09 | Dev machine: Apple M1, 8 GB RAM | Peak-RAM budget for Phase 2 must stay well under 8 GB; backed + chunked reads are mandatory | — |
| 2026-11-09 | GO GAF: release 2026-08-05, file `annotations/gaf/HUMAN-uniprot.gaf.gz` | Newer GO releases no longer ship `goa_human.gaf.gz` under `annotations/`; HUMAN-uniprot is the same UniProt-GOA human set | `current.geneontology.org` (undated); GEARS `gene2go_all.pkl` (pickle, fallback only) |
| 2026-11-09 | Dataverse IDs 7458695 / 6154020 unchanged; also check Dataverse-published MD5 on download | IDs still resolve (303 → S3); MD5 gives an independent check besides our SHA-256 pin | — |
| 2026-11-09 | Pin archives **and** every extracted file; `verify` also fails on unpinned files in `data/raw` (dotfiles ignored) | Rule 8 says every raw file is pinned; catches stray or partial files | Pin archives only |
| 2026-11-09 | Download records kept in committed `data/downloads.json`; `data/SOURCES.md` is generated from it + configs | No hand-typed hashes or sizes | Hand-written SOURCES.md |
| 2026-11-09 | Knockdown efficiency = log2((mean_pert + 0.01) / (mean_ctrl + 0.01)) on normalized expression (expm1 of stored log1p) | Interpretable as fold change; pseudocount in config | Difference of mean log1p values |
| 2026-11-09 | GEARS Norman `obs.cell_type` says "A549"; the screen is K562 (Norman 2019). Treated as a GEARS labelling quirk, noted in the card | Original paper is K562 CRISPRa | — |
| 2026-11-09 | Pseudobulk columns keyed by Ensembl id; `genes.parquet` maps symbols. A target symbol masks every column with that symbol | Replogle has a duplicated symbol (TBCE, two Ensembl ids) | Dedupe symbols by suffix |
| 2026-11-09 | Split-half ceiling uses half-specific controls: Δ^A = pb_A[p] − pb_A[ctrl], Δ^B = pb_B[p] − pb_B[ctrl] | A shared control mean would put the same control noise in both halves and inflate the ceiling | One control mean for both halves |
| 2026-11-09 | sys_cos and ceilings exclude the target gene(s), like every metric | Knockdown of the target dominates Δ and would inflate both | Include target in descriptives only |
| 2026-11-09 | Control embedding = top-K eigenvectors of control-cell gene–gene correlation × sqrt(eigenvalue), sign fixed by the largest-|loading| entry; zero-variance genes get a zero row | Deterministic, uses control cells only | sklearn PCA on cells (needs dense cells) |
| 2026-11-09 | `make all` skip rule: rerun a step unless MANIFEST.json matches config hash, package code hash and input hashes | Git SHA changes on every commit; a code hash is what actually invalidates outputs | Skip on git SHA |
| 2026-11-09 | `data/processed/` parquet (~80 MB for Replogle) not committed; will ship in the Colab bundle / release | Above the 50 MB guideline | Commit to git |
