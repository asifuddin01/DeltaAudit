"""Generate tests/fixtures/tiny.h5ad: small synthetic GEARS-like data with known structure.

    uv run python tests/fixtures/make_tiny.py

Layout (mirrors the quirks of the real GEARS files):
- 200 genes, indexed by fake Ensembl ids; the last symbol duplicates "G198" (like TBCE in Replogle).
- 300 control cells; 40 in-panel singles (G000-G039), 4 out-of-panel singles (X00-X03),
  6 doubles, 2 singles with only 10 cells (G040, G041; below min_cells, dropped in QC).
- G005's cells are split across "G005+ctrl" and "ctrl+G005"; one double is written "G003+G002".
- Cell expression (log1p scale) = base + a_p * systematic + specific_p + noise, target genes
  knocked down to 10% of base, clipped at 0, ~30% dropout. Cells are shuffled.
- uns["truth_systematic"], uns["truth_base"] hold the generating vectors.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 0
N_GENES = 200
N_CTRL = 300
CELLS = 40
LOW_CELLS = 10
OUT = Path(__file__).with_name("tiny.h5ad")


def main() -> None:
    rng = np.random.default_rng(SEED)
    symbols = [f"G{i:03d}" for i in range(N_GENES)]
    symbols[-1] = "G198"  # duplicated symbol, distinct Ensembl id
    col_of = {s: i for i, s in enumerate(symbols[:-1])}
    base = rng.gamma(2.0, 0.5, N_GENES)
    systematic = rng.normal(0.0, 0.3, N_GENES)

    singles = [f"G{i:03d}" for i in range(40)] + [f"X{i:02d}" for i in range(4)]
    doubles = [("G000", "G001"), ("G002", "G003"), ("G004", "G005"),
               ("G010", "G020"), ("G011", "G021"), ("G012", "G022")]  # fmt: skip
    effect = {}
    for g in singles + ["G040", "G041"]:
        e = np.zeros(N_GENES)
        hit = rng.choice(N_GENES, 10, replace=False)
        e[hit] = rng.normal(0.0, 0.8, 10)
        effect[g] = e

    perts = [((g,), CELLS) for g in singles] + [(d, CELLS) for d in doubles]
    perts += [(("G040",), LOW_CELLS), (("G041",), LOW_CELLS)]

    blocks, labels = [], []
    ctrl = base + rng.normal(0.0, 0.5, (N_CTRL, N_GENES))
    blocks.append(ctrl)
    labels += ["ctrl"] * N_CTRL
    for targets, n in perts:
        mean = (
            base + rng.uniform(0.5, 1.5) * systematic + sum(effect[t] for t in targets)
        )
        x = mean + rng.normal(0.0, 0.5, (n, N_GENES))
        for t in targets:
            if t in col_of:
                x[:, col_of[t]] = 0.1 * base[col_of[t]] + rng.normal(0.0, 0.05, n)
        blocks.append(x)
        if targets == ("G005",):
            labels += ["G005+ctrl"] * (n // 2) + ["ctrl+G005"] * (n - n // 2)
        elif targets == ("G002", "G003"):
            labels += ["G003+G002"] * n
        elif len(targets) == 1:
            labels += [f"{targets[0]}+ctrl"] * n
        else:
            labels += ["+".join(targets)] * n

    x = np.clip(np.vstack(blocks), 0.0, None)
    x[rng.random(x.shape) < 0.3] = 0.0
    order = rng.permutation(len(labels))
    x, labels = x[order], np.array(labels)[order]

    adata = ad.AnnData(
        X=sp.csr_matrix(x.astype(np.float32)),
        obs=pd.DataFrame(
            {
                "condition": pd.Categorical(labels),
                "cell_type": "TINY",
                "control": (labels == "ctrl").astype(int),
            },
            index=[f"cell{i:05d}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(
            {"gene_name": symbols}, index=[f"ENSGFAKE{i:05d}" for i in range(N_GENES)]
        ),
        uns={"truth_systematic": systematic, "truth_base": base},
    )
    adata.write_h5ad(OUT, compression="gzip")
    print(f"wrote {OUT} {adata.shape}")


if __name__ == "__main__":
    main()
