"""MANIFEST.json contents for every pipeline step: git SHA, config hash, input hashes,
device, library versions, wall time, peak RAM."""

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from deltaaudit.io.checksums import read_pins, sha256_file

PACKAGES = ["numpy", "scipy", "pandas", "anndata", "h5py", "scikit-learn", "torch"]


def pick_device() -> str:
    """Device order from CLAUDE.md: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_sha(root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        ).stdout.strip()

    sha = git("rev-parse", "HEAD") or "unknown"
    return sha + (
        "-dirty" if git("status", "--porcelain", "--untracked-files=no") else ""
    )


def code_hash() -> str:
    """SHA-256 over the package source, so a code change invalidates cached outputs."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def up_to_date(
    manifest_path: Path, root: Path, config: dict, inputs: list[str]
) -> bool:
    """True when an existing MANIFEST.json matches this config, code and input hashes."""
    if not manifest_path.exists():
        return False
    old = json.loads(manifest_path.read_text())
    pins = read_pins(root)
    return (
        old.get("config_hash") == config_hash(config)
        and old.get("code_hash") == code_hash()
        and old.get("inputs")
        == {p: pins.get(p) or sha256_file(root / p) for p in inputs}
    )


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports KiB


def build_manifest(root: Path, config: dict, inputs: list[str], t0: float) -> dict:
    """`inputs` are repo-relative paths; pinned hashes are reused (verify already checked them)."""
    pins = read_pins(root)
    return {
        "git_sha": git_sha(root),
        "config_hash": config_hash(config),
        "code_hash": code_hash(),
        "inputs": {p: pins.get(p) or sha256_file(root / p) for p in inputs},
        "device": pick_device(),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in PACKAGES},
        "wall_time_s": round(time.time() - t0, 2),
        "peak_rss_bytes": peak_rss_bytes(),
    }
