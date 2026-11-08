"""deltaaudit command-line interface. Commands are added phase by phase (BUILD_GUIDE Part H)."""

import platform
from pathlib import Path

import typer
import yaml

from deltaaudit import __version__
from deltaaudit.manifest import pick_device

app = typer.Typer(
    no_args_is_help=True, help="DeltaAudit: audit of unseen-perturbation prediction."
)
ROOT = Path.cwd()  # Makefile and CI run from the repo root
DATASET_DIR = Path("configs/datasets")


def load_configs(names: list[str] | None, all_: bool) -> list[dict]:
    if all_:
        paths = sorted((ROOT / DATASET_DIR).glob("*.yaml"))
    elif names:
        paths = [ROOT / DATASET_DIR / f"{n}.yaml" for n in names]
    else:
        raise typer.BadParameter("give dataset names or --all")
    return [yaml.safe_load(p.read_text()) for p in paths]


@app.command()
def info() -> None:
    """Print version, Python and the compute device that would be used."""
    typer.echo(
        f"deltaaudit {__version__} | python {platform.python_version()} | device {pick_device()}"
    )


@app.command()
def download(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(False, "--all", help="Every config in configs/datasets."),
) -> None:
    """Download, extract and pin raw assets into data/raw/<dataset>/<asset>/."""
    from deltaaudit.io.download import fetch_asset, render_sources

    configs = load_configs(names, all_)
    for cfg in configs:
        for asset in cfg["assets"]:
            res = fetch_asset(ROOT, cfg["name"], asset)
            typer.echo(f"{cfg['name']}/{asset['id']}: {res['status']}")
    every = load_configs(None, True)
    typer.echo(f"wrote {render_sources(ROOT, every).relative_to(ROOT)}")


@app.command()
def verify(
    skip_raw: bool = typer.Option(
        False, help="Skip raw-file checksums (CI and Colab have no data/raw)."
    ),
) -> None:
    """Checksums of every pinned raw file + leakage assertions on every split file."""
    from deltaaudit.data.leakage import verify_split_files
    from deltaaudit.io.checksums import read_pins
    from deltaaudit.io.checksums import verify as verify_pins

    problems = [] if skip_raw else verify_pins(ROOT)
    problems += verify_split_files(ROOT)
    for p in problems:
        typer.echo(p, err=True)
    if problems:
        typer.echo(f"verify FAILED: {len(problems)} problem(s)", err=True)
        raise typer.Exit(1)
    n_splits = len(list((ROOT / "data" / "splits").rglob("fold_*.json")))
    raw = "skipped" if skip_raw else f"{len(read_pins(ROOT))} pinned file(s) ok"
    typer.echo(f"verify ok: raw {raw}; {n_splits} split file(s) leak-free")


@app.command()
def card(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(False, "--all", help="Every config that has an h5ad."),
) -> None:
    """Write data/cards/<dataset>.{md,json}: cells, perturbations, genes, controls, knockdown."""
    import json
    import time

    from deltaaudit.data.card import render_card, scan_h5ad, summarize
    from deltaaudit.manifest import build_manifest

    out_dir = ROOT / "data" / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in (c for c in load_configs(names, all_) if "h5ad" in c):
        t0 = time.time()
        per, facts = scan_h5ad(
            str(ROOT / cfg["h5ad"]),
            cfg["preprocess"]["chunk_cells"],
            cfg["qc"]["knockdown_pseudocount"],
        )
        summary = summarize(per, facts, cfg["qc"]["min_cells"])
        summary["manifest"] = build_manifest(ROOT, cfg, [cfg["h5ad"]], t0)
        (out_dir / f"{cfg['name']}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (out_dir / f"{cfg['name']}.md").write_text(render_card(cfg, summary))
        gb = summary["manifest"]["peak_rss_bytes"] / 1e9
        typer.echo(f"{cfg['name']}: card written; peak RSS {gb:.2f} GB")


PROCESSED = Path("data/processed")
SUMMARY = Path("results/summary")


@app.command()
def preprocess(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(False, "--all", help="Every config that has an h5ad."),
    force: bool = typer.Option(
        False, help="Rebuild even if MANIFEST.json is up to date."
    ),
) -> None:
    """Pseudobulk (+ half-splits, delta, meta, genes, ctrl_embed) into data/processed/<dataset>/."""
    from deltaaudit.data.pseudobulk import preprocess as run
    from deltaaudit.manifest import up_to_date

    for cfg in (c for c in load_configs(names, all_) if "h5ad" in c):
        out = ROOT / PROCESSED / cfg["name"]
        if not force and up_to_date(out / "MANIFEST.json", ROOT, cfg, [cfg["h5ad"]]):
            typer.echo(f"{cfg['name']}: up to date, skipped")
            continue
        res = run(ROOT, cfg, out)
        m, q = res["manifest"], res["qc"]
        typer.echo(
            f"{cfg['name']}: kept {q['n_kept']}/{q['n_conditions_total']} perturbations "
            f"({len(q['dropped'])} dropped, n_cells < {q['min_cells']}); "
            f"{m['wall_time_s']:.0f} s, peak RSS {m['peak_rss_bytes'] / 1e9:.2f} GB"
        )


@app.command()
def describe(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(False, "--all", help="Every config that has an h5ad."),
) -> None:
    """sys_cos and split-half ceilings on all perturbations (descriptive; no model, no split)."""
    import json

    import pandas as pd

    from deltaaudit.data.card import QUANTILES
    from deltaaudit.eval.ceilings import describe as run
    from deltaaudit.eval.ceilings import summarize

    (ROOT / SUMMARY).mkdir(parents=True, exist_ok=True)
    for cfg in (c for c in load_configs(names, all_) if "h5ad" in c):
        d = ROOT / PROCESSED / cfg["name"]
        names_ = ("delta", "pb_half_A", "pb_half_B", "meta", "genes")
        desc = run(*(pd.read_parquet(d / f"{n}.parquet") for n in names_))
        desc.to_parquet(d / "descriptive.parquet")
        summary = {"dataset": cfg["name"], **summarize(desc, QUANTILES)}
        out = ROOT / SUMMARY / f"{cfg['name']}_descriptive.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        typer.echo(
            f"{cfg['name']}: median sys_cos {summary['sys_cos']['q50']:.3f}, "
            f"median ceil_raw {summary['ceil_raw']['q50']:.3f}, "
            f"median ceil_pspec {summary['ceil_pspec']['q50']:.3f}"
        )


FIGURES = Path("reports/figures")


@app.command()
def figures(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(False, "--all", help="Every config that has an h5ad."),
) -> None:
    """Render every figure whose inputs exist into reports/figures/."""
    import pandas as pd

    from deltaaudit.report.figures import fig_sys_variation

    for cfg in (c for c in load_configs(names, all_) if "h5ad" in c):
        desc = pd.read_parquet(ROOT / PROCESSED / cfg["name"] / "descriptive.parquet")
        title = f"{cfg['name']} ({cfg.get('modality', '')})"
        out = fig_sys_variation(
            desc, title, ROOT / FIGURES / f"fig1_sys_variation_{cfg['name']}.png"
        )
        typer.echo(f"wrote {out.relative_to(ROOT)}")


@app.command()
def split(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(
        False, "--all", help="Every config with a splits section."
    ),
) -> None:
    """Write data/splits/<dataset>/fold_k.json (+ doubles split, summary.json) from meta."""
    import json

    import pandas as pd

    from deltaaudit.data.leakage import check_folds, check_split
    from deltaaudit.data.splits import (
        seen_2of2_split,
        split_summary,
        unseen_single_splits,
    )

    for cfg in (c for c in load_configs(names, all_) if "splits" in c):
        meta = pd.read_parquet(ROOT / PROCESSED / cfg["name"] / "meta.parquet")
        folds = unseen_single_splits(meta, cfg["splits"], cfg["name"])
        extra = (
            [seen_2of2_split(meta, cfg["name"])]
            if cfg["splits"].get("doubles_seen_2of2")
            else []
        )
        problems = check_folds(folds, meta) + [
            p for s in folds + extra for p in check_split(s, meta)
        ]
        if problems:
            raise typer.BadParameter("; ".join(problems))
        out = ROOT / "data" / "splits" / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        for s in folds:
            (out / f"fold_{s['fold']}.json").write_text(json.dumps(s, indent=1) + "\n")
        for s in extra:
            (out / "doubles_seen_2of2.json").write_text(json.dumps(s, indent=1) + "\n")
        summary = split_summary(folds + extra, meta)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        typer.echo(
            f"{cfg['name']}: {len(folds)} folds + {len(extra)} extra split(s) written"
        )
