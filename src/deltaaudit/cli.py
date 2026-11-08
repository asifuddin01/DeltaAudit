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
def verify() -> None:
    """Check every pinned raw file (SHA-256) and that nothing in data/raw is unpinned."""
    from deltaaudit.io.checksums import read_pins
    from deltaaudit.io.checksums import verify as verify_pins

    problems = verify_pins(ROOT)
    for p in problems:
        typer.echo(p, err=True)
    if problems:
        typer.echo(f"verify FAILED: {len(problems)} problem(s)", err=True)
        raise typer.Exit(1)
    typer.echo(f"verify ok: {len(read_pins(ROOT))} pinned file(s)")


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
            cfg["qc"]["chunk_cells"],
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
