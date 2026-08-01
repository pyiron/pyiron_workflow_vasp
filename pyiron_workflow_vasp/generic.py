# for development and testing only
# provide functionality, data types etc. that will be later moved to the workflow code
from __future__ import annotations

from pathlib import Path
import os
import tarfile
import shutil
import subprocess
from typing import Optional

from pyiron_snippets.logger import logger

import flowrep as fr


class Storage:
    def _convert_to_dict(instance):
        # Get the attributes of the instance
        attributes = vars(instance)

        # Convert attributes to a dictionary
        result_dict = {
            key: value for key, value in attributes.items() if "_" not in key[0]
        }

        return result_dict

class ShellOutput(Storage):
    stdout: str
    stderr: str
    return_code: int
    dump: FileObject  # TODO: should be done in a specific lammps object
    log: FileObject

class VarType:
    def __init__(
        self,
        value=None,
        dat_type=None,
        label: str = None,
        store: int = 0,
        generic: bool = None,
        doc: str = None,
    ):
        self.value = value
        self.type = dat_type
        self.label = label
        self.store = store
        self.generic = generic
        self.doc = doc


class FileObject:
    def __init__(self, path=".", directory=None):
        if directory is None:
            self._path = Path(path)
        else:
            self._path = Path(directory) / Path(path)

    def __repr__(self):
        return f"FileObject: {self._path} {self.is_file}"

    @property
    def path(self):
        # Note conversion to string (needed to satisfy glob which is used e.g. in dump parser)
        return str(self._path)

    @property
    def is_file(self):
        return self._path.is_file()

    @property
    def name(self):
        return self._path.name


@fr.atomic("output")
def run_shell(
    command: str,
    workdir: str | None = None,
    environment: Optional[dict[str, str]] = None,
    arguments: Optional[list[str]] = None,
) -> ShellOutput:
    """
    Run a shell command in the specified working directory.

    Args:
        command (str): The command to execute. Interpreted by the shell, so the
            full string is preserved (this is what allows VASP-style invocations
            like ``"module load vasp; mpiexec -n 1 vasp_std"``).
        workdir (str | None, optional): The working directory. Defaults to None.
        environment (Optional[dict[str, str]], optional): Environment variables
            to set in addition to the parent environment. Defaults to None.
        arguments (Optional[list[str]], optional): Extra arguments appended to
            ``command`` (whitespace-separated). Defaults to None.

    Returns:
        ShellOutput: Object containing stdout, stderr, and return code.

    Note:
        Deliberately does NOT call ``os.chdir``: ``subprocess.run``'s ``cwd``
        argument already scopes the child process to ``workdir``, and mutating
        the parent's working directory is unsafe now that pyiron_workflow 0.19
        evaluates sibling nodes in a DAG layer on separate threads by default.
        A prior version saved/restored ``os.getcwd()`` around a real
        ``os.chdir`` call; under concurrent execution that save/restore pair
        races (thread B can capture thread A's workdir as "the directory to
        restore to"), corrupting the *process* cwd for the remainder of the
        run. Measured at 27/30 trials. ``logger.info`` below logs ``workdir``
        directly instead of ``os.getcwd()`` since that was the only reason the
        old code queried/changed the process cwd at all.
    """
    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []
    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})

    # When ``shell=True``, the command must be a single string — passing a list
    # silently drops every element past the first on POSIX. Compose the full
    # command line here.
    full_command = " ".join([command, *map(str, arguments)]) if arguments else command

    logger.info(f"run_shell: workdir={workdir}")
    proc = subprocess.run(
        full_command,
        capture_output=True,
        cwd=workdir,
        encoding="utf8",
        env=environ,
        shell=True,
    )

    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    return output


@fr.atomic("line_found")
def is_line_in_file(filepath: str, line: str, exact_match: bool = True) -> bool:
    """
    Check if a specific line exists in a file.

    Args:
        filepath (str): Path to the file to search in.
        line (str): The line to search for.
        exact_match (bool, optional): If True, the line must match exactly. If False,
                                     the line can be a substring of any line in the file.
                                     Defaults to True.

    Returns:
        bool: True if the line is found, False otherwise.
    """
    line_found = False  # Initialize the result as False
    try:
        with open(filepath, "r") as file:
            for file_line in file:
                if exact_match and line == file_line.strip():
                    line_found = True
                    break  # Exit loop if the line is found
                elif not exact_match and line in file_line:
                    line_found = True
                    break  # Exit loop if a partial match is found
    except FileNotFoundError:
        logger.info(f"File '{filepath}' not found.")
    return line_found


@fr.atomic("workdir")
def delete_files_recursively(
    workdir: str, files_to_be_deleted: list[str] | None = None, after: object = None
) -> str:
    """
    Recursively delete specific files in a directory and its subdirectories.

    Args:
        workdir (str): The directory to search for files.
        files_to_be_deleted (list[str] | None): List of filenames to delete.
            ``None`` is treated as an empty list (nothing to delete) rather
            than raising -- a defensive, low-level guard for direct or
            standalone callers of this node. ``vasp_job`` (``vasp.py``) does
            NOT rely on this fallback: its ``resolve_cleanup_files`` node
            intercepts a ``None`` ``files_to_be_deleted`` before this
            function ever sees it and substitutes the documented default
            cleanup list (``["CHG", "CHGCAR", "WAVECAR"]``) instead.
        after: An ordering token. It is never read, but accepting it lets a
            caller create a data edge that forces this node to run after
            another. pyiron_workflow 0.19 has no execution signals, so
            ordering must come from data flow.

    Returns:
        str: ``workdir``, unchanged, so the value can be threaded onward.
    """
    files_to_be_deleted = files_to_be_deleted or []
    if not os.path.isdir(workdir):
        logger.info(f"Error: {workdir} is not a valid directory.")
    else:
        for root, _, files in os.walk(workdir):
            for file in files:
                if file in files_to_be_deleted:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.info(f"Error deleting {file_path}: {e}")
    return workdir


@fr.atomic("directory_path")
def compress_directory(
    directory_path: str,
    actually_compress: bool = True,
    inside_dir: bool = True,
    exclude_files: list[str] | None = None,
    after: object = None,
) -> str:
    """
    Compress ``directory_path`` to a gzipped tarball, returning the directory.

    Args:
        directory_path (str): The path of the directory to compress.
        actually_compress (bool, optional): If False, this is a no-op that
            just returns ``directory_path``. Defaults to True.
        inside_dir (bool, optional): Whether the output tarball should be
            placed inside the source directory or alongside it. Defaults to
            True.
        exclude_files (list[str] | None, optional): Filenames to exclude from
            the compression. Defaults to None.
        after: An ordering token; see :func:`delete_files_recursively`.

    Returns:
        str: ``directory_path``, unchanged. Returning the directory rather
        than the tarball lets the value be threaded onward to establish
        ordering.

    Note:
        Archive members are rooted under the directory's basename (e.g.
        ``calc/OUTCAR``, not bare ``OUTCAR``), matching every archive produced
        by this package historically. Extracting an archive from any campaign
        -- old or new -- therefore lands files under the same
        ``<basename>/...`` prefix instead of flat into the current directory.

        Self-exclusion of the output tarball compares the FULL path, not the
        bare filename: a legitimate user file living in a subdirectory but
        sharing the tarball's basename (e.g. ``calc/sub/calc.tar.gz``) must
        not be dropped from the archive just because its name matches --
        especially since ``remove_calc_dir=True`` may delete the source
        directory immediately afterward.
    """
    if not actually_compress:
        return directory_path
    exclude = set(exclude_files or [])
    base = os.path.basename(os.path.normpath(directory_path))
    target_dir = directory_path if inside_dir else os.path.dirname(directory_path)
    tar_path = os.path.join(target_dir, f"{base}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for root, _, files in os.walk(directory_path):
            for file in files:
                full = os.path.join(root, file)
                if file in exclude or full == tar_path:
                    continue
                tar.add(
                    full,
                    arcname=os.path.join(base, os.path.relpath(full, directory_path)),
                )
    logger.info(f"compress_directory: compressed directory at {directory_path}")
    return directory_path


@fr.atomic("payload")
def remove_directory(
    directory_path: str, actually_remove: bool = False, payload: object = None
) -> object:
    """
    Optionally delete ``directory_path``, forwarding ``payload`` unchanged.

    Args:
        directory_path (str): Directory to (maybe) remove.
        actually_remove (bool, optional): If True, ``directory_path`` is
            deleted (ignoring missing-directory errors). Defaults to False.
        payload: Arbitrary value passed straight through. Any consumer of the
            returned payload is necessarily ordered after the removal, which
            is how the "remove last" guarantee is preserved without execution
            signals.

    Returns:
        object: ``payload``, unchanged.
    """
    if actually_remove:
        shutil.rmtree(directory_path, ignore_errors=True)
    return payload
