import io
import stat
import tarfile
import zipfile

import pytest

from deltaaudit.io.safe_extract import UnsafeArchive, safe_extract


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_zip_roundtrip(tmp_path):
    arc = _zip(tmp_path / "ok.zip", {"d/a.txt": "A", "b.txt": "B"})
    files = safe_extract(arc, tmp_path / "out")
    assert [f.relative_to(tmp_path / "out").as_posix() for f in files] == [
        "b.txt",
        "d/a.txt",
    ]


@pytest.mark.parametrize("name", ["../evil.txt", "d/../../evil.txt", "/tmp/evil.txt"])
def test_zip_traversal_rejected(tmp_path, name):
    arc = _zip(tmp_path / "bad.zip", {name: "x"})
    with pytest.raises(UnsafeArchive):
        safe_extract(arc, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_zip_symlink_rejected(tmp_path):
    arc = tmp_path / "link.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(UnsafeArchive):
        safe_extract(arc, tmp_path / "out")


def test_tar_traversal_rejected(tmp_path):
    arc = tmp_path / "bad.tar"
    with tarfile.open(arc, "w") as tf:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        safe_extract(arc, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_dest_must_be_empty(tmp_path):
    arc = _zip(tmp_path / "ok.zip", {"a.txt": "A"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing").write_text("x")
    with pytest.raises(FileExistsError):
        safe_extract(arc, out)
