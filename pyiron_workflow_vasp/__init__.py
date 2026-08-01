"""pyiron_workflow_vasp - VASP integration for the pyiron workflow system.

Public API:
    VaspEngine - satisfies pyiron_workflow_atomistics.engine.Engine
                  for Static + Minimize modes; wraps the helpers in
                  vasp.py and vasp_parser/output.py.

Plus the flowrep-based nodes in vasp.py and generic.py (create/write/run/
parse/check/cleanup a VASP calculation), and the legacy bundled parser in
vasp_parser (superseded as the default parser by the external ``vaspparser``
package, but still shipped and importable).
"""

from ._version import get_versions
from .engine import VaspEngine
from .generic import (
    ShellOutput,
    compress_directory,
    delete_files_recursively,
    is_line_in_file,
    remove_directory,
    run_shell,
)
from .vasp import (
    VaspInput,
    check_convergence,
    construct_sequential_vasp_input,
    create_working_directory,
    generate_modified_incar,
    generate_vasp_input,
    parse_vasp_output,
    vasp_job,
    write_vasp_input_set,
)
from .vasp_parser import Outcar, parse_vasp_directory

__version__ = get_versions()["version"]

__all__ = [
    "VaspEngine",
    "Outcar",
    "ShellOutput",
    "VaspInput",
    "check_convergence",
    "compress_directory",
    "construct_sequential_vasp_input",
    "create_working_directory",
    "delete_files_recursively",
    "generate_modified_incar",
    "generate_vasp_input",
    "is_line_in_file",
    "parse_vasp_directory",
    "parse_vasp_output",
    "remove_directory",
    "run_shell",
    "vasp_job",
    "write_vasp_input_set",
]
