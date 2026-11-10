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


@app.command("go-graph")
def go_graph_cmd(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(False, "--all", help="Every config that has an h5ad."),
    force: bool = typer.Option(
        False, help="Rebuild even if MANIFEST.json is up to date."
    ),
) -> None:
    """GO Jaccard kNN graphs (targets -> go/, panel+targets -> go_union/ with the M3b rewire) and coverage."""
    import json
    import time

    import pandas as pd
    import scipy.sparse as sp

    from deltaaudit.data import go_graph
    from deltaaudit.manifest import build_manifest, up_to_date

    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    go_cfg, m3b = models_cfg["go_graph"], models_cfg["learned"]["M3b"]
    src = yaml.safe_load((ROOT / DATASET_DIR / "go.yaml").read_text())
    if src["release"] != go_cfg["release"]:
        raise typer.BadParameter(
            f"models.yaml GO release {go_cfg['release']} != go.yaml {src['release']}"
        )
    asset = src["assets"][0]
    gaf = f"data/raw/{src['name']}/{asset['id']}/{asset['filename']}"
    (ROOT / SUMMARY).mkdir(parents=True, exist_ok=True)
    for cfg in (c for c in load_configs(names, all_) if "h5ad" in c):
        d = PROCESSED / cfg["name"]
        meta = pd.read_parquet(ROOT / d / "meta.parquet")
        genes = pd.read_parquet(ROOT / d / "genes.parquet")
        graphs = {
            "targets": (go_graph.GO_DIR, go_graph.target_nodes(meta), go_cfg),
            "union": (go_graph.GO_UNION_DIR, go_graph.union_nodes(meta, genes), {**go_cfg, "M3b": m3b}),
        }  # fmt: skip
        summary_path = ROOT / SUMMARY / f"{cfg['name']}_go_coverage.json"
        old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary = {"dataset": cfg["name"], **{k: old[k] for k in graphs if k in old}}
        for key, (sub, nodes, conf) in graphs.items():
            out = ROOT / d / sub
            inputs = [gaf, f"{d}/meta.parquet", f"{d}/genes.parquet"]
            if not force and up_to_date(out / "MANIFEST.json", ROOT, conf, inputs):
                typer.echo(f"{cfg['name']}/{key}: GO graph up to date, skipped")
                continue
            t0 = time.time()
            cov = go_graph.build(nodes, ROOT / gaf, go_cfg, out)
            if key == "union":
                nodes_, adj = go_graph.load(ROOT / d, sub)
                rw = go_graph.rewire(adj, m3b["nswap_per_edge"], m3b["seed"])
                sp.save_npz(out / "rewired.npz", rw)
            m = build_manifest(ROOT, conf, inputs, t0)
            (out / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
            summary[key] = cov
            typer.echo(
                f"{cfg['name']}/{key}: {cov['n_with_edge']}/{cov['n_nodes']} nodes have a GO edge "
                f"({cov['n_annotated']} annotated, {cov['n_via_synonym']} via synonym), "
                f"{cov['n_edges']} edges, {time.time() - t0:.0f} s"
            )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


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


@app.command()
def baselines(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(
        False, "--all", help="Every config with a splits section."
    ),
) -> None:
    """Fit/score the baselines on every split -> results/<dataset>/<model>/..., results/summary/baselines.csv."""
    import pandas as pd

    from deltaaudit.data.leakage import require_prereg, verify_split_files
    from deltaaudit.models.baselines import run_baselines
    from deltaaudit.pipeline import load_dataset, load_splits, summarize

    require_prereg(ROOT)
    if problems := verify_split_files(ROOT):
        raise typer.BadParameter("; ".join(problems))
    eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    if not names and not all_:
        all_ = True
    frames = []
    for cfg in (c for c in load_configs(names, all_) if "splits" in c):
        data, splits = load_dataset(ROOT, cfg["name"]), load_splits(ROOT, cfg["name"])
        frames.append(
            run_baselines(ROOT, data, splits, eval_cfg, models_cfg, typer.echo)
        )
    table = summarize(pd.concat(frames), eval_cfg["metrics"], eval_cfg["statistics"])
    (ROOT / SUMMARY).mkdir(parents=True, exist_ok=True)
    table.to_csv(ROOT / SUMMARY / "baselines.csv", index=False, float_format="%.6g")
    typer.echo(f"wrote {SUMMARY}/baselines.csv ({len(table)} rows)")


@app.command()
def train(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(
        False, "--all", help="Every config with a splits section."
    ),
    models: str = typer.Option("M1,M2,M3,M3b", help="Comma-separated learned models."),
    folds: str = typer.Option("", help="Comma-separated folds (default: all)."),
    seeds: str = typer.Option(
        "", help="Comma-separated seeds (default: eval.yaml learned_seeds)."
    ),
    device: str = typer.Option(
        "auto", help="auto (cuda > mps > cpu), cuda, mps or cpu."
    ),
    smoke: bool = typer.Option(
        False, help="Fold 0, seed 0, every model; time it and project the grid."
    ),
) -> None:
    """Train/score learned models (resumable) -> results/<ds>/<model>/<run>/fold_k/; timing + summary."""
    import json

    import pandas as pd

    from deltaaudit.data.leakage import require_prereg, verify_split_files
    from deltaaudit.manifest import pick_device
    from deltaaudit.models.train import run_learned, runs
    from deltaaudit.pipeline import (
        RESULTS,
        load_dataset,
        load_splits,
        split_tag,
        summarize,
    )

    require_prereg(ROOT)
    if problems := verify_split_files(ROOT):
        raise typer.BadParameter("; ".join(problems))
    eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["learned"]
    dev = pick_device() if device == "auto" else device
    model_list = models.split(",")
    seed_list = (
        [0]
        if smoke
        else [int(s) for s in seeds.split(",")] if seeds else eval_cfg["learned_seeds"]
    )
    fold_list = [0] if smoke else [int(f) for f in folds.split(",")] if folds else None
    if not names and not all_:
        all_ = True
    frames, timing = [], {"device": dev, "datasets": {}}
    for c in (c for c in load_configs(names, all_) if "splits" in c):
        data = load_dataset(ROOT, c["name"])
        splits = [
            s for s in load_splits(ROOT, c["name"])
            if s["split_type"] == "unseen_single" and (fold_list is None or s["fold"] in fold_list)
        ]  # fmt: skip
        frames.append(
            run_learned(
                ROOT,
                data,
                splits,
                eval_cfg,
                cfg,
                model_list,
                seed_list,
                dev,
                typer.echo,
            )
        )
        if smoke:
            n_folds, n_seeds = c["splits"]["n_folds"], len(eval_cfg["learned_seeds"])
            per_model = {}
            for m in model_list:
                secs = [
                    json.loads((ROOT / RESULTS / c["name"] / m / r / split_tag(splits[0]) / "MANIFEST.json").read_text())["wall_time_s"]
                    for r, _, _ in runs(m, cfg, eval_cfg, [0])
                ]  # fmt: skip
                per_model[m] = {
                    "s_per_run": sum(secs) / len(secs),
                    "n_runs_full_grid": len(secs) * n_folds * n_seeds,
                }
            total = sum(
                v["s_per_run"] * v["n_runs_full_grid"] for v in per_model.values()
            )
            timing["datasets"][c["name"]] = {
                "models": per_model,
                "projected_full_grid_h": total / 3600,
            }
    scores = pd.concat(frames, ignore_index=True)
    (ROOT / SUMMARY).mkdir(parents=True, exist_ok=True)
    if smoke:
        timing["projected_full_grid_h"] = sum(
            d["projected_full_grid_h"] for d in timing["datasets"].values()
        )
        (ROOT / SUMMARY / "train_timing.json").write_text(
            json.dumps(timing, indent=2) + "\n"
        )
        typer.echo(
            f"projected full grid on {dev}: {timing['projected_full_grid_h']:.2f} h"
        )
        return
    table = summarize(scores, eval_cfg["metrics"], eval_cfg["statistics"])
    table.to_csv(
        ROOT / SUMMARY / "learned_by_run.csv", index=False, float_format="%.6g"
    )
    typer.echo(f"wrote {SUMMARY}/learned_by_run.csv ({len(table)} rows)")


def _phase8_configs() -> tuple[dict, dict]:
    from deltaaudit.data.leakage import require_prereg, verify_split_files

    require_prereg(ROOT)
    if problems := verify_split_files(ROOT):
        raise typer.BadParameter("; ".join(problems))

    def load(n: str) -> dict:
        return yaml.safe_load((ROOT / "configs" / f"{n}.yaml").read_text())

    return load("eval"), load("secondary")


@app.command()
def evaluate(
    names: list[str] = typer.Argument(None, help="Dataset config names."),
    all_: bool = typer.Option(
        False, "--all", help="Every config with a splits section."
    ),
) -> None:
    """Seed-averaged per-perturbation scores + F2 diagnostics -> results/summary/{scores.csv, model_table.csv, diagnosis.json}."""
    from deltaaudit.eval.audit import run_evaluate

    eval_cfg, sec_cfg = _phase8_configs()
    if not names and not all_:
        all_ = True
    ds = [c["name"] for c in load_configs(names, all_) if "splits" in c]
    out = run_evaluate(ROOT, ds, eval_cfg, sec_cfg)
    for n in ds:
        best = out["diagnosis"][n]["best_learned"]
        typer.echo(
            f"{n}: best learned by inner-val raw_r = {best['selected']} {best['val_raw_r']}"
        )
    typer.echo(
        f"wrote {SUMMARY}/scores.csv ({len(out['scores'])} rows), model_table.csv, diagnosis.json"
    )


@app.command()
def stats() -> None:
    """Paired bootstrap + Holm (C1-C6 and secondary), cluster bootstrap, decisions, ladder ->
    results/summary/{comparisons.csv, decisions.json, ladder.csv}. Run `evaluate` first.
    """
    from deltaaudit.eval.audit import run_stats

    eval_cfg, sec_cfg = _phase8_configs()
    go_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["go_graph"]
    src = yaml.safe_load((ROOT / DATASET_DIR / "go.yaml").read_text())
    gaf = (
        ROOT
        / "data/raw"
        / src["name"]
        / src["assets"][0]["id"]
        / src["assets"][0]["filename"]
    )
    rungs = yaml.safe_load((ROOT / "configs" / "ladder.yaml").read_text())["rungs"]
    out = run_stats(ROOT, eval_cfg, sec_cfg, rungs, gaf, go_cfg)
    c = out["comparisons"]
    fam = c[(c.family == "primary") & (c.stratum == "")]
    for r in fam.itertuples():
        typer.echo(f"{r.id} {r.a} vs {r.b} ({r.metric}): d={r.obs:+.4f} "
                   f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] p={r.p_raw:.4g} p_holm={r.p_holm:.4g}")  # fmt: skip
    typer.echo(
        f"wrote {SUMMARY}/comparisons.csv ({len(c)} rows), decisions.json, ladder.csv"
    )


def _hvg_names(cfg: dict) -> dict[str, str]:
    return {v: f"{cfg['dataset']}__hvg_{v}" for v in ("all", "train")}


@app.command("hvg-panels")
def hvg_panels_cmd(
    device: str = typer.Option(
        "auto", help="auto (cuda > mps > cpu), cuda, mps or cpu."
    ),
) -> None:
    """Phase 9: re-select the gene panel from all cells vs control + train cells (one fold),
    build both derived datasets and fit every single-perturbation model on that fold."""
    from deltaaudit.eval.sensitivity import fit_fold, hvg_panels
    from deltaaudit.pipeline import load_splits

    eval_cfg, sec_cfg = _phase8_configs()
    cfg = sec_cfg["sensitivity"]["hvg_reselection"]
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    src = yaml.safe_load((ROOT / DATASET_DIR / "go.yaml").read_text())
    gaf = (
        ROOT
        / "data/raw"
        / src["name"]
        / src["assets"][0]["id"]
        / src["assets"][0]["filename"]
    )
    ds_cfg = load_configs([cfg["dataset"]], False)[0]
    names = hvg_panels(ROOT, ds_cfg, cfg, load_splits(ROOT, cfg["dataset"]),
                       models_cfg["go_graph"], models_cfg["learned"]["M3b"], gaf)  # fmt: skip
    assert names == _hvg_names(cfg)
    dev = pick_device() if device == "auto" else device
    for n in names.values():
        fit_fold(
            ROOT,
            n,
            eval_cfg,
            models_cfg,
            cfg["fold"],
            eval_cfg["learned_seeds"],
            dev,
            typer.echo,
        )
    typer.echo(f"built and fitted {', '.join(names.values())} (fold {cfg['fold']})")


@app.command()
def sensitivity() -> None:
    """Phase 9 sensitivity analyses -> results/summary/{sensitivity.csv, sensitivity.md}.
    Run `evaluate`, `stats` and `hvg-panels` first."""
    from deltaaudit.eval.sensitivity import run_sensitivity

    eval_cfg, sec_cfg = _phase8_configs()
    names = _hvg_names(sec_cfg["sensitivity"]["hvg_reselection"])
    if not all(
        (ROOT / "data/processed" / n / "delta.parquet").exists() for n in names.values()
    ):
        raise typer.BadParameter(
            "HVG panels missing: run `deltaaudit hvg-panels` first"
        )
    run_sensitivity(ROOT, eval_cfg, sec_cfg, names)
    typer.echo((ROOT / SUMMARY / "sensitivity.md").read_text())
