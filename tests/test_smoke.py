"""Smoke tests that do not require VASP, POTCARs, or a real config file.

The behaviours locked in here:

* the package can be imported without ``~/.pyiron_vasp_config`` (regression
  guard for the lazy-config refactor);
* ``read_potcar_config`` handles valid input and surfaces the right errors;
* a handful of small utilities behave as expected on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ----------------------------------------------------------------------------
# Importability
# ----------------------------------------------------------------------------

def test_package_imports_without_config():
    """The package must import even when no config file is on disk."""
    import pyiron_workflow_vasp  # noqa: F401
    from pyiron_workflow_vasp.vasp import VaspInput, vasp_job  # noqa: F401
    from pyiron_workflow_vasp.generic import run_shell, is_line_in_file  # noqa: F401


def test_lazy_config_raises_only_on_use():
    """Accessing the lazy accessor without a config should raise FileNotFoundError."""
    from pyiron_workflow_vasp.vasp import _get_potcar_config

    with pytest.raises(FileNotFoundError):
        _get_potcar_config()


# ----------------------------------------------------------------------------
# read_potcar_config
# ----------------------------------------------------------------------------

def test_read_potcar_config_happy_path(valid_config_file: Path):
    from pyiron_workflow_vasp.vasp import read_potcar_config

    cfg = read_potcar_config(valid_config_file)

    assert cfg["default_POTCAR_set"] == "potpaw64"
    assert cfg["default_functional"] == "GGA"
    # default_POTCAR_path is derived from default_POTCAR_set
    assert cfg["default_POTCAR_path"].endswith("potpaw_64")
    # CSV suffix falls back to functional when default_pseudopotential_set is missing
    assert cfg["pseudopotential_csv_suffix"] == "GGA"


def test_read_potcar_config_missing_file(tmp_path: Path):
    from pyiron_workflow_vasp.vasp import read_potcar_config

    with pytest.raises(FileNotFoundError):
        read_potcar_config(tmp_path / "does_not_exist")


def test_read_potcar_config_unknown_default(tmp_path: Path):
    from pyiron_workflow_vasp.vasp import read_potcar_config

    bad = tmp_path / "bad.cfg"
    bad.write_text(
        "default_POTCAR_set = potpaw99\n"
        "default_functional = GGA\n"
        "pyiron_vasp_resources = /tmp\n"
        "vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64\n"
    )

    with pytest.raises(ValueError, match="default_POTCAR_set"):
        read_potcar_config(bad)


def test_read_potcar_config_unknown_functional(tmp_path: Path):
    from pyiron_workflow_vasp.vasp import read_potcar_config

    bad = tmp_path / "bad.cfg"
    bad.write_text(
        "default_POTCAR_set = potpaw64\n"
        "default_functional = MEHGA\n"
        "pyiron_vasp_resources = /tmp\n"
        "vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64\n"
    )

    with pytest.raises(ValueError, match="default_functional"):
        read_potcar_config(bad)


# ----------------------------------------------------------------------------
# VaspInput
# ----------------------------------------------------------------------------

def test_vasp_input_with_explicit_potcars_skips_config_lookup():
    """If the user passes potcar_paths, no config file is needed."""
    from ase.build import bulk
    from pymatgen.io.vasp.inputs import Incar
    from pyiron_workflow_vasp.vasp import VaspInput

    structure = bulk("Fe", cubic=True, a=2.83)
    incar = Incar.from_dict({"ENCUT": 400})

    vi = VaspInput(structure=structure, incar=incar, potcar_paths=["/dev/null/POTCAR"])

    assert vi.potcar_paths == ["/dev/null/POTCAR"]
    # pseudopot_lib_path stays unset because we shortcut on explicit POTCARs.
    assert vi.pseudopot_lib_path is None


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def test_stack_element_string_groups_contiguous_runs():
    from ase import Atoms
    from pyiron_workflow_vasp.vasp import stack_element_string

    atoms = Atoms(symbols=["Fe", "Fe", "O", "O", "O", "Fe"])
    elements, counts = stack_element_string(atoms)
    assert elements == ["Fe", "O", "Fe"]
    assert counts == [2, 3, 1]


def test_is_line_in_file_finds_substring(tmp_path: Path):
    from pyiron_workflow_vasp.generic import is_line_in_file

    f = tmp_path / "log.txt"
    f.write_text("alpha\nbeta gamma delta\nepsilon\n")

    assert is_line_in_file(filepath=str(f), line="gamma", exact_match=False)
    # exact_match requires the whole line to match the stripped target
    assert not is_line_in_file(filepath=str(f), line="gamma", exact_match=True)
    assert is_line_in_file(filepath=str(f), line="beta gamma delta", exact_match=True)


def test_is_line_in_file_missing_file_returns_false(tmp_path: Path):
    from pyiron_workflow_vasp.generic import is_line_in_file

    assert not is_line_in_file(
        filepath=str(tmp_path / "no_such_file"), line="x", exact_match=False
    )


def test_run_shell_propagates_subprocess_exceptions(tmp_path: Path, monkeypatch):
    """run_shell must not swallow an exception raised by subprocess.run.

    (This package no longer chdirs around the subprocess call at all -- see
    test_generic.py::test_run_shell_concurrent_calls_preserve_process_cwd for
    the regression test covering the removed, thread-unsafe os.chdir --
    so there is no cwd-restore behaviour left to assert on here.)
    """
    import subprocess

    from pyiron_workflow_vasp.generic import run_shell

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated subprocess failure")

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(RuntimeError):
        run_shell(command="true", workdir=str(tmp_path))
