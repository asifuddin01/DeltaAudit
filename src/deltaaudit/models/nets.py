"""W0 (SGC linear map) + MLP head. M1 uses the same net on unpropagated features, so M1 and M2
differ only in the graph propagation of the inputs."""

import torch.nn as nn


def make_net(n_in: int, n_out: int, cfg: dict) -> nn.Sequential:
    mlp, width = cfg["mlp"], cfg["sgc"]["width"]
    assert mlp["activation"] == "gelu", mlp["activation"]
    mods: list[nn.Module] = [nn.Linear(n_in, width)]  # W0
    d = width
    for _ in range(mlp["layers"]):
        mods += [nn.Linear(d, mlp["hidden"]), nn.GELU(), nn.Dropout(mlp["dropout"])]
        d = mlp["hidden"]
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)
