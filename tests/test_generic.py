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


import os

from pyiron_workflow_vasp.generic import run_shell


def test_run_shell_executes_in_workdir(abs_workdir):
    run = pwf.node(run_shell).run(command="pwd", workdir=abs_workdir)
    assert run.outputs.output.return_code == 0
    assert run.outputs.output.stdout.strip() == abs_workdir


def test_run_shell_does_not_mutate_process_cwd(abs_workdir):
    before = os.getcwd()
    pwf.node(run_shell).run(command="pwd", workdir=abs_workdir)
    assert os.getcwd() == before


def test_run_shell_captures_failure(abs_workdir):
    run = pwf.node(run_shell).run(command="exit 3", workdir=abs_workdir)
    assert run.outputs.output.return_code == 3


def test_run_shell_is_concurrency_safe(abs_workdir, tmp_path):
    """Two shell nodes in one DAG layer must not corrupt each other's cwd.

    This is the regression test for removing os.chdir: with the chdir in
    place, interleaved execution makes one of the two `pwd` results wrong.
    """
    import concurrent.futures

    other = tmp_path / "other"
    other.mkdir()
    other_abs = str(other.resolve())

    def run_in(d):
        return pwf.node(run_shell).run(command="pwd", workdir=d).outputs.output.stdout.strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(run_in, d) for d in (abs_workdir, other_abs) for _ in range(20)]
        results = {f.result() for f in futures}

    assert results == {abs_workdir, other_abs}
