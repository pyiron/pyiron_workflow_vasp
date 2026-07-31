import pathlib
import tarfile

import pyiron_workflow as pwf

from pyiron_workflow_vasp.generic import (
    compress_directory,
    delete_files_recursively,
    is_line_in_file,
    remove_directory,
)


def test_delete_files_recursively_removes_only_named_files(fake_vasp_dir):
    run = pwf.node(delete_files_recursively).run(
        workdir=str(fake_vasp_dir), files_to_be_deleted=["CHGCAR", "WAVECAR"]
    )
    assert run.status == "finished"
    assert run.outputs.workdir == str(fake_vasp_dir)
    assert not (fake_vasp_dir / "CHGCAR").exists()
    assert not (fake_vasp_dir / "WAVECAR").exists()
    assert (fake_vasp_dir / "OUTCAR").exists()


def test_delete_files_recursively_tolerates_missing_directory(tmp_path):
    missing = str(tmp_path / "nope")
    run = pwf.node(delete_files_recursively).run(
        workdir=missing, files_to_be_deleted=["CHGCAR"]
    )
    assert run.status == "finished"
    assert run.outputs.workdir == missing


def test_compress_directory_creates_tarball_and_returns_path(fake_vasp_dir):
    run = pwf.node(compress_directory).run(
        directory_path=str(fake_vasp_dir), actually_compress=True, inside_dir=True
    )
    assert run.outputs.directory_path == str(fake_vasp_dir)
    tarballs = list(pathlib.Path(fake_vasp_dir).glob("*.tar.gz"))
    assert len(tarballs) == 1
    with tarfile.open(tarballs[0]) as tf:
        assert any(n.endswith("OUTCAR") for n in tf.getnames())


def test_compress_directory_is_a_noop_when_disabled(fake_vasp_dir):
    run = pwf.node(compress_directory).run(
        directory_path=str(fake_vasp_dir), actually_compress=False
    )
    assert run.outputs.directory_path == str(fake_vasp_dir)
    assert list(pathlib.Path(fake_vasp_dir).glob("*.tar.gz")) == []


def test_remove_directory_forwards_payload_and_deletes(fake_vasp_dir):
    sentinel = {"energy": -8.21}
    run = pwf.node(remove_directory).run(
        directory_path=str(fake_vasp_dir), actually_remove=True, payload=sentinel
    )
    assert run.outputs.payload == sentinel
    assert not fake_vasp_dir.exists()


def test_remove_directory_forwards_payload_without_deleting(fake_vasp_dir):
    run = pwf.node(remove_directory).run(
        directory_path=str(fake_vasp_dir), actually_remove=False, payload="keep"
    )
    assert run.outputs.payload == "keep"
    assert fake_vasp_dir.exists()


def test_is_line_in_file_substring_match(fake_vasp_dir):
    run = pwf.node(is_line_in_file).run(
        filepath=str(fake_vasp_dir / "vasp.log"),
        line="reached required accuracy",
        exact_match=False,
    )
    assert run.outputs.line_found is True
