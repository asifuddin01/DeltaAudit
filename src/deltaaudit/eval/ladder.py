"""Validation ladder table (BUILD_GUIDE F4). Rungs, names and criteria come from configs/ladder.yaml."""

import numpy as np
import pandas as pd

from deltaaudit.eval.stats import bootstrap_ci, paired_test


def _beats(w: dict, model: str, vs: dict, boot: tuple) -> tuple[bool, float]:
    m = w[vs["metric"]]
    if model == vs["model"] or model not in m or vs["model"] not in m:
        return False, np.nan
    pair = m[[model, vs["model"]]].dropna()
    t = paired_test((pair[model] - pair[vs["model"]]).to_numpy(), *boot)
    cleared = t["ci_lo"] > 0 if vs["sign"] > 0 else t["ci_hi"] < 0
    return bool(cleared), t["obs"] * vs["sign"]


def _rungs(s: pd.DataFrame, model: str, rungs: list[dict], boot: tuple) -> dict:
    """Per rung: (cleared, signed effects). Signed effect > 0 means 'in the rung's direction'."""
    metrics = {v["metric"] for r in rungs for v in r.get("vs", [])} | {
        r["metric"] for r in rungs if "metric" in r
    }
    w = {m: s.pivot(index="condition", columns="model", values=m) for m in metrics}
    mine = s[s.model == model]
    out = {}
    for r in rungs:
        if r["test"] == "integrity":  # stats refuses to run unless `verify` passes
            out[r["id"]] = (True, [])
        elif r["test"] == "fit":
            out[r["id"]] = (bool(mine.raw_r.notna().any()), [])
        elif r["test"] == "positive":
            x = mine[r["metric"]].dropna().to_numpy()
            out[r["id"]] = (
                bool(len(x)) and bootstrap_ci(x, *boot)[0] > 0,
                [float(x.mean())],
            )
        elif r["test"] == "beats":
            if "applies_to" in r and model not in r["applies_to"]:
                out[r["id"]] = (False, [])
                continue
            res = [_beats(w, model, v, boot) for v in r["vs"]]
            out[r["id"]] = (all(c for c, _ in res), [e for _, e in res])
    return out


def ladder(
    primary: pd.DataFrame, replication: pd.DataFrame, rungs: list[dict], stats_cfg: dict
) -> pd.DataFrame:
    """One row per model: each rung cleared or not, and the highest rung cleared continuously.
    `replicates` holds when every rung cleared before it has the same-direction effect in the
    replication set (CI not required there)."""
    boot = (stats_cfg["bootstrap_B"], stats_cfg["bootstrap_seed"], stats_cfg["ci"])
    rows = []
    for model in sorted(primary.model.unique()):
        here = _rungs(
            primary, model, [r for r in rungs if r["test"] != "replicates"], boot
        )
        there = (
            _rungs(replication, model, [r for r in rungs if r["test"] != "replicates"], boot)
            if model in set(replication.model) else {}
        )  # fmt: skip
        row = {"model": model}
        for r in rungs:
            if r["test"] == "replicates":
                tested = [k for k, (c, e) in here.items() if c and e]
                row[r["id"]] = (
                    bool(there)
                    and bool(tested)
                    and all(
                        len(there[k][1]) and all(e > 0 for e in there[k][1])
                        for k in tested
                    )
                )
            else:
                row[r["id"]] = here[r["id"]][0]
        top = -1
        while top + 1 < len(rungs) and row[rungs[top + 1]["id"]]:
            top += 1
        row["highest_rung"] = rungs[top]["id"] if top >= 0 else ""
        row["highest_rung_name"] = rungs[top]["name"] if top >= 0 else ""
        rows.append(row)
    return pd.DataFrame(rows)
