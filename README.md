# DeltaAudit

**What the metric hides: predicting unseen gene knockdowns**

A pre-registered, leakage-checked replication audit of perturbation-response prediction on held-out CRISPRi knockdowns (Replogle 2022 K562 essential), plus a controlled test of whether a gene-graph prior helps because of its biological content.

Status: Phase 0 (scaffold). See `BUILD_GUIDE.md` for the plan and `CLAUDE.md` for the rules.

## Quick start

```bash
make env     # uv sync from the lockfile (Python 3.11)
make test
uv run deltaaudit info
```

Results will be filled in here automatically from `results/summary/` (no hand-typed numbers).
