"""Learned models M1, M2, M3 (label-shuffled GO), M3b (rewired GO): residual over mu_train,
full-batch AdamW on train perturbations, early stopping on inner-validation pspec_r, no refit.

Resume is per run: `run_model(resume=True)` skips any (model, run, fold) whose MANIFEST is up to date.
"""

import copy

import numpy as np
import pandas as pd
import torch

from deltaaudit.data.features import node_features, propagate
from deltaaudit.data.go_graph import permute_labels
from deltaaudit.data.leakage import fit_rows
from deltaaudit.eval.metrics import pearson_or_zero, target_mask
from deltaaudit.models.nets import make_net
from deltaaudit.pipeline import run_model

VARIANTS = {"M1": None, "M2": "go", "M3": "permuted", "M3b": "rewired"}


def fit_net(
    x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    keep_val: np.ndarray, cfg: dict, seed: int, device: str, tol: float,
) -> tuple[torch.nn.Module, dict]:  # fmt: skip
    torch.manual_seed(seed)
    net = make_net(x_tr.shape[1], y_tr.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    xt, yt, xv = t(x_tr), t(y_tr), t(x_val)

    def val_score() -> float:
        net.eval()
        with torch.no_grad():
            f = net(xv).cpu().numpy().astype(np.float64)
        return float(pearson_or_zero(f, y_val, keep_val, tol)[0].mean())

    patience = cfg["early_stopping"]["patience"]
    best, best_epoch, best_state, epoch = -np.inf, 0, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        net.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
        score = (
            val_score() if len(x_val) else float(epoch)
        )  # no val: train to max_epochs
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= patience:
            break
    net.load_state_dict(best_state)
    info = {
        "best_epoch": best_epoch, "epochs_run": epoch, "final_train_loss": float(loss.detach()),
        "best_val_pspec_r": best if len(x_val) else None,
        "val_pspec_r_restored": val_score() if len(x_val) else None,
        "n_train": len(x_tr), "n_val": len(x_val), "device": device, "seed": seed,
    }  # fmt: skip
    return net, info


def _graph(data: dict, kind: str | None, perm: int | None):
    adj = data["go_union"][1]
    return {
        None: lambda: None, "go": lambda: adj,
        "permuted": lambda: permute_labels(adj, perm),
        "rewired": lambda: data["go_rewired"],
    }[kind]()  # fmt: skip


def make_learned(
    cfg: dict, variant: str, seed: int, device: str, tol: float, perm: int | None = 0
):
    def predict(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
        x = node_features(data, split, cfg["features"])
        graph = _graph(data, VARIANTS[variant], perm)
        if graph is not None:
            x = propagate(graph, x, cfg["sgc"]["hops"])
        idx = {n: i for i, n in enumerate(data["go_union"][0])}
        targets = data["meta"]["targets"]
        rows = lambda conds: [idx[targets[c][0]] for c in conds]  # noqa: E731
        train = fit_rows(data["delta"], split, include_val=False)
        mu = train.mean(axis=0).to_numpy(np.float64)  # residual_over: mu_train
        val = split["val"]
        y_val = data["delta"].loc[val].to_numpy(np.float64) - mu
        keep_val = ~target_mask(targets.loc[val].tolist(), data["symbols"])
        net, info = fit_net(
            x[rows(train.index)], train.to_numpy(np.float64) - mu, x[rows(val)], y_val,
            keep_val, cfg, seed, device, tol,
        )  # fmt: skip
        net.eval()
        with torch.no_grad():
            xt = torch.as_tensor(
                x[rows(split["test"])], dtype=torch.float32, device=device
            )
            f = net(xt).cpu().numpy().astype(np.float64)
        out = pd.DataFrame(mu + f, index=split["test"], columns=data["delta"].columns)
        out.attrs["fit_info"] = {**info, "variant": variant, "perm": perm}
        return out

    return predict


def runs(
    model: str, cfg: dict, eval_cfg: dict, seeds: list[int]
) -> list[tuple[str, int, int | None]]:
    """(run tag, seed, permutation) for one model; M3 expands over its label permutations."""
    perms = (
        eval_cfg["shuffle_permutations"][: cfg["M3"]["permutations"]]
        if model == "M3"
        else [None]
    )
    return [
        (f"seed_{s}" if p is None else f"perm_{p}/seed_{s}", s, p)
        for p in perms
        for s in seeds
    ]


def run_learned(
    root, data: dict, splits: list[dict], eval_cfg: dict, cfg: dict,
    models: list[str], seeds: list[int], device: str, log=print,
) -> pd.DataFrame:  # fmt: skip
    singles = [s for s in splits if s["split_type"] == "unseen_single"]
    frames = []
    for model in models:
        for run, seed, perm in runs(model, cfg, eval_cfg, seeds):
            predict = make_learned(
                cfg, model, seed, device, eval_cfg["degenerate_std"], perm
            )
            conf = {**cfg, "variant": model, "seed": seed, "perm": perm}
            frames.append(
                run_model(
                    root,
                    data,
                    singles,
                    eval_cfg,
                    model,
                    predict,
                    conf,
                    run=run,
                    resume=True,
                    log=log,
                )
            )
    return pd.concat(frames, ignore_index=True)
