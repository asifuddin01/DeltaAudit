# Decisions log

| Date | Decision | Reason | Alternatives rejected |
|---|---|---|---|
| 2026-11-09 | Primary dataset: Replogle 2022 K562 essential (GEARS-processed) | User confirmed in Phase 0; ~1,000+ single-gene knockdowns, well above any thesis-dataset alternative offered | Thesis dataset (declined) |
| 2026-11-09 | Environment: `uv` + `uv.lock`, Python 3.11, hatchling build | Fast, exact lockfile, works on Mac and CI | mamba / environment.yml |
| 2026-11-09 | CPU-only torch wheel index on Linux (CI/Colab-free path); PyPI torch on macOS | Avoids ~2 GB CUDA wheels in CI; macOS wheel includes MPS | Single PyPI index everywhere |
| 2026-11-09 | Dev machine: Apple M1, 8 GB RAM | Peak-RAM budget for Phase 2 must stay well under 8 GB; backed + chunked reads are mandatory | — |
