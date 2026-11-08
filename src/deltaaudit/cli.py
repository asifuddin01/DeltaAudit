"""deltaaudit command-line interface. Commands are added phase by phase (BUILD_GUIDE Part H)."""

import platform

import typer

from deltaaudit import __version__

app = typer.Typer(
    no_args_is_help=True, help="DeltaAudit: audit of unseen-perturbation prediction."
)


@app.callback()
def main() -> None:
    """Keeps subcommand mode while only one command exists."""


def pick_device() -> str:
    """Device order from CLAUDE.md: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def info() -> None:
    """Print version, Python and the compute device that would be used."""
    typer.echo(
        f"deltaaudit {__version__} | python {platform.python_version()} | device {pick_device()}"
    )
