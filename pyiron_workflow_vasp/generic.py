# for development and testing only
# provide functionality, data types etc. that will be later moved to the workflow code
from __future__ import annotations

from pathlib import Path
import os
from typing import Optional
import tarfile
import fnmatch
import shutil
import subprocess
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
def shell(
    command: str,
    workdir: str | None = None,
    environment: Optional[dict[str, str]] = None,
    arguments: Optional[list[str]] = None,
) -> ShellOutput:
    """
    Run a shell command in the specified working directory.
    
    Args:
        command (str): The command to execute.
        workdir (str | None, optional): The working directory. Defaults to None.
        environment (Optional[dict[str, str]], optional): Environment variables to set. Defaults to None.
        arguments (Optional[list[str]], optional): Command line arguments. Defaults to None.
    
    Returns:
        ShellOutput: Object containing stdout, stderr, and return code.
    """
    curr_dir = os.getcwd()
    os.chdir(workdir)
    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []
    logger.info(f"shell is in {os.getcwd()}")
    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    proc = subprocess.run(
        [command, *map(str, arguments)],
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
    os.chdir(curr_dir)
    return output

@fr.atomic("line_found")
def is_line_in_file(filepath: str, line: str, exact_match: bool = True) -> bool:
    line_found = False
    try:
        with open(filepath) as f:
            for file_line in f:
                if (exact_match and file_line.strip() == line) or (
                    not exact_match and line in file_line
                ):
                    line_found = True
                    break
    except FileNotFoundError:
        logger.info(f"File '{filepath}' not found.")
    return line_found


@fr.atomic("workdir")
def delete_files_recursively(
    workdir: str, files_to_be_deleted: list[str], after: object = None
) -> str:
    """Recursively delete named files under ``workdir``.

    ``after`` is an ordering token: it is never read, but accepting it lets a
    caller create a data edge that forces this node to run after another.
    0.19 has no execution signals, so ordering must come from data flow.
    """
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
    """Compress ``directory_path`` to a gzipped tarball, returning the directory.

    Returns the directory rather than the tarball so the value can be threaded
    onward to establish ordering. ``after`` is an ordering token; see
    :func:`delete_files_recursively`.
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
                if file in exclude or file == f"{base}.tar.gz":
                    continue
                full = os.path.join(root, file)
                tar.add(full, arcname=os.path.relpath(full, directory_path))
    return directory_path


def submit_to_slurm(
    node,
    /,
    job_name=None,
    output_file=None,
    error_file=None,
    time_limit="00:05:00",
    partition="s.cmmg",
    nodes=1,
    ntasks=1,
    cpus_per_task=1,
    memory="1GB",
):
    """
    An example of a helper function for running nodes on slurm.

    - Saves the node
    - Writes a slurm batch script that 
        - Loads the node
        - Runs it
        - Saves it again
    - Runs the batch script
    """
    if node.graph_root is not node:
        raise ValueError(
            f"Can only submit parent-most nodes, but {node.full_label} "
            f"has root {node.graph_root.full_label}"
        )
        
    node.save(backend="pickle")
    p = node.as_path()
    
    if job_name is None:
        job_name = node.full_label 
        job_name = job_name.replace(node.lexical_delimiter, "_")
        job_name = "pwf" + job_name
        
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name} 
#SBATCH --output={p.joinpath("slurm.out").resolve() if output_file is None else output_file}
#SBATCH --error={p.joinpath("slurm.err").resolve() if error_file is None else error_file}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}

# Execute Python script inline
python - <<EOF
from pyiron_workflow import PickleStorage
node = PickleStorage().load(filename="{node.as_path().joinpath('picklestorage').resolve()}")  # Load
node.run()  # Run
node.save(backend="pickle")  # Save again
EOF
"""
    submission_script = p.joinpath("node_submission.sh")
    submission_script.write_text(script_content)
    import subprocess
    submission = subprocess.run(["sbatch", submission_script.resolve()])
    return submission
    
@fr.atomic("payload")
def remove_directory(
    directory_path: str, actually_remove: bool = False, payload: object = None
) -> object:
    """Optionally delete ``directory_path``, forwarding ``payload`` unchanged.

    The pass-through is deliberate: any consumer of the returned payload is
    necessarily ordered after the removal, which is how the old signal chain's
    "remove last" guarantee is preserved.
    """
    if actually_remove and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
    return payload