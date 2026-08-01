"""Test fixtures.

These tests deliberately do NOT depend on a working ``.pyiron_vasp_config``,
the presence of POTCAR files, or a VASP binary. Anything that needs DFT
results is mocked or skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _point_config_at_nonexistent(monkeypatch):
    """Force every test to import the package as if no config existed.

    Tests that want to exercise the config-loading path build their own
    config file under ``tmp_path`` and point ``PYIRON_VASP_CONFIG`` at it
    explicitly. The cache is cleared on entry/exit to keep tests isolated.
    """
    monkeypatch.setenv("PYIRON_VASP_CONFIG", "/this/path/does/not/exist")

    # Defer the import so this fixture is harmless even if pyiron_workflow_vasp
    # somehow can't be imported in a particular environment.
    try:
        from pyiron_workflow_vasp import vasp as _vasp
    except Exception:
        yield
        return

    _vasp._get_potcar_config.cache_clear()
    # Reset DEFAULT_CONFIG_PATH to honour the freshly-monkeypatched env var.
    _vasp.DEFAULT_CONFIG_PATH = Path(os.environ["PYIRON_VASP_CONFIG"])
    try:
        yield
    finally:
        _vasp._get_potcar_config.cache_clear()


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Write a fake but structurally valid config file and return its path."""
    cfg = tmp_path / ".pyiron_vasp_config"
    cfg.write_text(
        "default_POTCAR_set = potpaw64\n"
        "default_functional = GGA\n"
        f"pyiron_vasp_resources = {tmp_path}\n"
        "vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64\n"
        "vasp_POTCAR_path_potpaw54 = {pyiron_vasp_resources}/potpaw_54\n"
    )
    return cfg


MINIMAL_OUTCAR = """\
 vasp.6.4.3 complex
 POTCAR:    PAW_PBE Fe 06Sep2000
 energy  without entropy=      -8.21000000  energy(sigma->0) =       -8.21000000
 reached required accuracy - stopping structural energy minimisation
"""

MINIMAL_POSCAR = """\
Fe
1.0
2.83 0.00 0.00
0.00 2.83 0.00
0.00 0.00 2.83
Fe
1
Direct
0.0 0.0 0.0
"""


@pytest.fixture
def abs_workdir(tmp_path: Path) -> str:
    d = tmp_path / "calc"
    d.mkdir()
    return str(d.resolve())


@pytest.fixture
def fake_vasp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "vaspcalc"
    d.mkdir()
    (d / "OUTCAR").write_text(MINIMAL_OUTCAR)
    (d / "POSCAR").write_text(MINIMAL_POSCAR)
    (d / "vasp.log").write_text(
        "reached required accuracy - stopping structural energy minimisation\n"
    )
    (d / "CHGCAR").write_text("bulk charge density placeholder\n")
    (d / "WAVECAR").write_text("wavefunction placeholder\n")
    return d
