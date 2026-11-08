"""SHA-256 pinning and verification of raw data (CLAUDE.md rule 8).

Pins live in `data/checksums.sha256` (`sha256sum` format, paths relative to the repo root).
A mismatch is a hard failure and is never "fixed" by re-pinning.
"""

import hashlib
from pathlib import Path

PIN_FILE = Path("data/checksums.sha256")
RAW_DIR = Path("data/raw")
CHUNK_BYTES = 1 << 20


class ChecksumError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK_BYTES):
            h.update(block)
    return h.hexdigest()


def read_pins(root: Path) -> dict[str, str]:
    pin_file = root / PIN_FILE
    if not pin_file.exists():
        return {}
    pins = {}
    for line in pin_file.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            pins[rel] = digest
    return pins


def pin(root: Path, path: Path, digest: str) -> None:
    """Record a new pin. An existing pin for the same path must match exactly."""
    rel = path.relative_to(root).as_posix()
    pins = read_pins(root)
    if rel in pins:
        if pins[rel] != digest:
            raise ChecksumError(
                f"{rel}: got sha256 {digest}, pinned {pins[rel]}. Not re-pinning; "
                "investigate the source before touching data/checksums.sha256."
            )
        return
    pins[rel] = digest
    (root / PIN_FILE).parent.mkdir(parents=True, exist_ok=True)
    (root / PIN_FILE).write_text(
        "".join(f"{d}  {r}\n" for r, d in sorted(pins.items()))
    )


def raw_files(root: Path) -> list[Path]:
    """Every non-hidden file under data/raw (Finder/AppleDouble dotfiles are ignored)."""
    base = root / RAW_DIR
    if not base.exists():
        return []
    return sorted(
        p
        for p in base.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(base).parts)
    )


def verify(root: Path) -> list[str]:
    """Return a list of problems; empty means every pinned file matches and nothing is unpinned."""
    pins = read_pins(root)
    problems = []
    for rel, digest in pins.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"MISMATCH: {rel}")
    for p in raw_files(root):
        rel = p.relative_to(root).as_posix()
        if rel not in pins:
            problems.append(f"unpinned: {rel}")
    return problems
