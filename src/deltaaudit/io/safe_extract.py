"""Path-traversal-safe archive extraction into a new, empty directory (CLAUDE.md rule 7)."""

import stat
import tarfile
import zipfile
from pathlib import Path


class UnsafeArchive(Exception):
    pass


def _check_zip_member(dest: Path, info: zipfile.ZipInfo) -> None:
    if stat.S_ISLNK(info.external_attr >> 16):
        raise UnsafeArchive(f"symlink in archive: {info.filename}")
    if not (dest / info.filename).resolve().is_relative_to(dest.resolve()):
        raise UnsafeArchive(f"path escapes destination: {info.filename}")


def safe_extract(archive: Path, dest: Path) -> list[Path]:
    """Extract a zip or tar into `dest` (must be new or empty). Returns the extracted files."""
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"extraction target is not empty: {dest}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _check_zip_member(dest, info)
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            # filter="data" rejects absolute paths, '..' and links pointing outside dest
            tf.extractall(dest, filter="data")
    else:
        raise UnsafeArchive(f"not a zip or tar archive: {archive}")
    return sorted(p for p in dest.rglob("*") if p.is_file())
