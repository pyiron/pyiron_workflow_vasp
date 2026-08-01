import os
import pathlib
import tarfile

import pyiron_workflow as pwf

from pyiron_workflow_vasp.generic import (
    compress_directory,
    delete_files_recursively,
    is_line_in_file,
    remove_directory,
    run_shell,
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


def test_delete_files_recursively_tolerates_none_files_to_be_deleted(fake_vasp_dir):
    """``files_to_be_deleted=None`` must be treated as "delete nothing", not
    raise TypeError from ``file in None`` -- this is a defensive, low-level
    guard for direct/standalone callers of this node. ``vasp_job`` itself no
    longer passes ``None`` through unchanged: its ``resolve_cleanup_files``
    node intercepts ``None`` and substitutes the documented default cleanup
    list (``["CHG", "CHGCAR", "WAVECAR"]``) before this node ever sees it --
    see ``test_vasp_job.py::test_vasp_job_runs_without_files_to_be_deleted_argument``."""
    run = pwf.node(delete_files_recursively).run(
        workdir=str(fake_vasp_dir), files_to_be_deleted=None
    )
    assert run.status == "finished"
    assert (fake_vasp_dir / "CHGCAR").exists()
    assert (fake_vasp_dir / "WAVECAR").exists()


def test_compress_directory_creates_tarball_and_returns_path(fake_vasp_dir):
    run = pwf.node(compress_directory).run(
        directory_path=str(fake_vasp_dir), actually_compress=True, inside_dir=True
    )
    assert run.outputs.directory_path == str(fake_vasp_dir)
    tarballs = list(pathlib.Path(fake_vasp_dir).glob("*.tar.gz"))
    assert len(tarballs) == 1
    with tarfile.open(tarballs[0]) as tf:
        assert any(n.endswith("OUTCAR") for n in tf.getnames())


def test_compress_directory_members_are_prefixed_with_directory_basename(
    fake_vasp_dir,
):
    """Archive members must be directory-rooted (e.g. ``vaspcalc/OUTCAR``),
    matching every archive produced by this package historically. A flat
    layout (bare ``OUTCAR``) is an on-disk format split that downstream
    consumers (ASSYST) would silently inherit, since archives extract flat
    into the current directory instead of under a named subdirectory."""
    run = pwf.node(compress_directory).run(
        directory_path=str(fake_vasp_dir), actually_compress=True, inside_dir=True
    )
    tarballs = list(pathlib.Path(fake_vasp_dir).glob("*.tar.gz"))
    assert len(tarballs) == 1
    base = fake_vasp_dir.name
    with tarfile.open(tarballs[0]) as tf:
        names = tf.getnames()
        assert names, "tarball has no members"
        assert all(n.startswith(f"{base}/") for n in names), names


def test_compress_directory_excludes_only_the_real_tarball_not_namesakes(
    fake_vasp_dir,
):
    """Self-exclusion must compare the FULL path, not the bare filename. A
    legitimate user file living in a subdirectory but sharing the tarball's
    basename (e.g. ``calc/sub/calc.tar.gz``) must NOT be dropped from the
    archive -- doing so is silent data loss, especially since
    remove_calc_dir=True subsequently deletes the source directory."""
    base = fake_vasp_dir.name
    sub = fake_vasp_dir / "sub"
    sub.mkdir()
    namesake = sub / f"{base}.tar.gz"
    namesake.write_text("this is real user data, not the output archive")

    run = pwf.node(compress_directory).run(
        directory_path=str(fake_vasp_dir), actually_compress=True, inside_dir=True
    )
    tarballs = list(pathlib.Path(fake_vasp_dir).glob("*.tar.gz"))
    assert len(tarballs) == 1

    with tarfile.open(tarballs[0]) as tf:
        names = tf.getnames()
        # the real output tarball must not have archived itself
        assert f"{base}/{base}.tar.gz" not in names
        # but the subdirectory namesake IS a legitimate file and must survive
        assert f"{base}/sub/{base}.tar.gz" in names


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


def test_run_shell_preserves_arguments_via_full_command_string(abs_workdir):
    """Regression test for the shell=True + list-argv bug: passing a list to
    subprocess.run under shell=True silently drops every element past the
    first on POSIX (only argv[0] reaches /bin/sh -c). run_shell must compose
    a single command string (command + arguments) instead."""
    run = pwf.node(run_shell).run(
        command="echo", workdir=abs_workdir, arguments=["hello", "world"]
    )
    assert run.outputs.output.stdout.strip() == "hello world"


def test_run_shell_concurrent_calls_preserve_process_cwd(tmp_path):
    """Regression test for the removed ``os.chdir`` race in ``run_shell``.

    A prior implementation did::

        curr_dir = os.getcwd()
        os.chdir(workdir)
        ...
        os.chdir(curr_dir)

    Under concurrent execution this save/restore pair races: thread B can
    call ``os.getcwd()`` for its own "curr_dir" bookkeeping *after* thread A
    has already ``chdir``'d to thread A's workdir, so thread B captures
    thread A's workdir as "the directory to restore to" and later leaves the
    *process* working directory pointed at the wrong place once all calls
    finish.

    Asserting on subprocess ``pwd`` stdout does NOT catch this: the child
    process's directory is scoped directly by ``subprocess.run(cwd=...)``,
    independent of the parent's ``os.getcwd()`` state, so every child always
    reports the directory it was launched with regardless of any parent-side
    chdir race. Only checking ``os.getcwd()`` of the calling process itself,
    after many concurrent calls have interleaved, can observe the corruption.

    The CWD is restored in a ``finally`` block so a failure here can't poison
    the working directory for the rest of the test session.
    """
    import concurrent.futures

    workdirs = []
    for i in range(4):
        d = tmp_path / f"work_{i}"
        d.mkdir()
        workdirs.append(str(d.resolve()))

    original_cwd = os.getcwd()
    try:
        for trial in range(5):
            targets = [workdirs[i % len(workdirs)] for i in range(24)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futures = [
                    ex.submit(
                        lambda d: pwf.node(run_shell).run(command="pwd", workdir=d),
                        d,
                    )
                    for d in targets
                ]
                for f in futures:
                    f.result()
            assert os.getcwd() == original_cwd, (
                f"trial {trial}: process cwd corrupted to {os.getcwd()!r}, "
                f"expected {original_cwd!r}"
            )
    finally:
        os.chdir(original_cwd)


def test_submit_to_slurm_is_gone():
    """Submission moved to assystant; the old helper imported PickleStorage,
    which pyiron_workflow 0.19 removed outright."""
    import pyiron_workflow_vasp.generic as generic

    assert not hasattr(generic, "submit_to_slurm")


def test_no_pickle_storage_references_remain():
    import pathlib

    import pyiron_workflow_vasp

    root = pathlib.Path(pyiron_workflow_vasp.__file__).parent
    offenders = [
        p.name
        for p in root.rglob("*.py")
        if "PickleStorage" in p.read_text()
    ]
    assert offenders == []
