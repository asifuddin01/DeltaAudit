import hashlib
import zipfile

import pytest

from deltaaudit.io.checksums import ChecksumError, pin, read_pins, sha256_file, verify
from deltaaudit.io.download import DownloadError, fetch_asset, stream_download


def _raw_file(root, rel="data/raw/ds/a/file.bin", data=b"hello world"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_sha256_matches_hashlib(tmp_path):
    p = _raw_file(tmp_path)
    assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_verify_ok_then_corrupted_byte_fails(tmp_path):
    p = _raw_file(tmp_path)
    pin(tmp_path, p, sha256_file(p))
    assert verify(tmp_path) == []

    data = bytearray(p.read_bytes())
    data[3] ^= 0x01
    p.write_bytes(bytes(data))
    problems = verify(tmp_path)
    assert len(problems) == 1 and problems[0].startswith("MISMATCH")


def test_missing_and_unpinned_files_fail(tmp_path):
    p = _raw_file(tmp_path)
    pin(tmp_path, p, sha256_file(p))
    _raw_file(tmp_path, "data/raw/ds/a/extra.bin")
    assert any(m.startswith("unpinned") for m in verify(tmp_path))
    p.unlink()
    assert any(m.startswith("missing") for m in verify(tmp_path))


def test_dotfiles_ignored(tmp_path):
    p = _raw_file(tmp_path)
    pin(tmp_path, p, sha256_file(p))
    _raw_file(tmp_path, "data/raw/ds/a/.DS_Store")
    assert verify(tmp_path) == []


def test_pin_conflict_is_refused(tmp_path):
    p = _raw_file(tmp_path)
    pin(tmp_path, p, sha256_file(p))
    with pytest.raises(ChecksumError):
        pin(tmp_path, p, "0" * 64)
    assert read_pins(tmp_path)["data/raw/ds/a/file.bin"] == sha256_file(p)


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_stream_download_md5_mismatch_leaves_nothing(tmp_path):
    src = _raw_file(tmp_path, "src/blob.bin")
    dest = tmp_path / "out" / "blob.bin"
    dest.parent.mkdir()
    with pytest.raises(DownloadError):
        stream_download(src.as_uri(), dest, expected_md5="0" * 32)
    assert list(dest.parent.iterdir()) == []


def test_fetch_asset_pins_extracts_and_is_idempotent(tmp_path):
    src = _zip(tmp_path / "pkg.zip", {"inner/a.txt": "A", "inner/b.txt": "B"})
    asset = {
        "id": "toy",
        "url": src.as_uri(),
        "filename": "pkg.zip",
        "md5": hashlib.md5(src.read_bytes()).hexdigest(),
        "archive": True,
    }
    rec = fetch_asset(tmp_path, "ds", asset)
    assert rec["status"] == "downloaded"
    pins = read_pins(tmp_path)
    assert "data/raw/ds/toy/pkg.zip" in pins
    assert "data/raw/ds/toy/extracted/inner/a.txt" in pins
    assert verify(tmp_path) == []
    assert fetch_asset(tmp_path, "ds", asset)["status"] == "present (verified)"


def test_fetch_asset_refuses_dirty_dir_and_pin_mismatch(tmp_path):
    src = _zip(tmp_path / "pkg.zip", {"a.txt": "A"})
    asset = {"id": "toy", "url": src.as_uri(), "filename": "pkg.zip", "archive": True}
    _raw_file(tmp_path, "data/raw/ds/toy/stray.bin")
    with pytest.raises(DownloadError):
        fetch_asset(tmp_path, "ds", asset)

    # A committed pin that disagrees with what the server sends is a hard failure.
    other = tmp_path / "other"
    (other / "data").mkdir(parents=True)
    (other / "data" / "checksums.sha256").write_text(
        f"{'0' * 64}  data/raw/ds/toy/pkg.zip\n"
    )
    with pytest.raises(ChecksumError):
        fetch_asset(other, "ds", asset)
