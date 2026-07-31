"""Regression tests for the OUTCAR parsing pipeline against a genuine OUTCAR.

Context: `pyiron_workflow_vasp/vasp_parser/outcar.py` used to call
``zopen(filename, "r")``. The monty version resolved by this package's own
dependency spec (``monty`` with no floor -- currently 2026.7.16) raises::

    RuntimeError: Implicit text/binary mode is not allowed, please pass t/b
    explicitly in mode.

on every call, i.e. `Outcar.from_file` raised on *every real OUTCAR*.
`pyiron_workflow_vasp/vasp_parser/output.py` wrapped all of its parsing in
bare ``except:`` clauses that swallowed this (and everything else, including
Ctrl-C) and substituted ``np.nan``, so `parse_vasp_directory` silently
returned NaN energies/forces/stresses for genuine VASP output with no error
and no warning.

This bug shipped through the entire pyiron_workflow 0.19 port because no
unit test ever fed the parser a real OUTCAR -- every existing test either
mocked the parser or used a hand-written OUTCAR fixture too synthetic to
exercise `Outcar.from_file`'s trigger-line search logic realistically.

The fixture ``tests/fixtures/vasp_outcar_fe_bcc/`` is a genuine OUTCAR from a
real VASP 5.4.4 run (2-atom bcc Fe, ISPIN=2 static calculation; POTCAR/
vasprun.xml/vasp.log are deliberately not included -- their absence is a
legitimately-tolerated case and is exercised here too). These tests assert
the parsed energy is a *finite number*, not merely that no exception was
raised -- a test that only checked "no exception" would still pass against
the broken code, since the exception was swallowed.
"""

import math
import pathlib
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Structure

from pyiron_workflow_vasp.vasp_parser.outcar import Outcar
from pyiron_workflow_vasp.vasp_parser.output import (
    parse_vasp_directory,
    process_outcar,
)

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "vasp_outcar_fe_bcc"


def test_outcar_from_file_parses_real_outcar_to_finite_energy():
    outcar = Outcar()
    outcar.from_file(str(FIXTURE_DIR / "OUTCAR"))

    energies = outcar.parse_dict["energies"]
    assert len(energies) > 0, "no ionic-step energies were parsed at all"
    for value in energies:
        assert math.isfinite(value), f"parsed energy is not finite: {value!r}"
    assert np.isfinite(energies).all()

    # This specific calculation's free energy is about -16.5 eV; a NaN
    # substituted by a swallowed exception would fail both the isfinite
    # checks above and this sanity bound.
    assert -20.0 < energies[-1] < -10.0


def test_parse_vasp_directory_real_outcar_reports_finite_energy(tmp_path):
    # parse_vasp_directory globs a directory rather than taking a single
    # filename, so the fixture is copied into a scratch directory.
    work_dir = tmp_path / "vasp_calc"
    shutil.copytree(FIXTURE_DIR, work_dir)

    with warnings.catch_warnings():
        # POTCAR/vasprun.xml/vasp.log/error.out are intentionally absent
        # from this trimmed fixture; the resulting warnings are expected
        # and orthogonal to what this test checks.
        warnings.simplefilter("ignore")
        results_df = parse_vasp_directory(str(work_dir), extract_error_dirs=False)

    assert len(results_df) == 1
    energy = results_df["energy"].iloc[0]
    assert not (
        isinstance(energy, float) and math.isnan(energy)
    ), "energy came back as a bare NaN -- parser silently failed"

    energy_arr = np.asarray(energy, dtype=float)
    assert energy_arr.size > 0
    assert np.isfinite(
        energy_arr
    ).all(), f"parsed energy contains non-finite values: {energy_arr}"
    assert -20.0 < float(energy_arr[-1]) < -10.0


def test_parse_vasp_directory_without_vasprun_or_potcar_emits_no_warnings(tmp_path):
    """A missing vasprun.xml (documented fallback to vasp.log/error.out) and
    a missing POTCAR (routinely stripped from archived directories for
    licensing reasons) are both DESIGNED, expected-absence paths, not
    failures -- parsing a directory that has everything else but those two
    files must not warn at all. Measured over a real campaign this bug was
    ~2 spurious warnings per archived directory (40 over 20 dirs), none
    deduplicated since the message embeds a per-directory path.

    The fixture directory (tests/fixtures/vasp_outcar_fe_bcc/) already lacks
    POTCAR and vasprun.xml (see module docstring), so it is reused as-is.
    """
    work_dir = tmp_path / "vasp_calc"
    shutil.copytree(FIXTURE_DIR, work_dir)
    assert not (work_dir / "vasprun.xml").exists()
    assert not (work_dir / "POTCAR").exists()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_vasp_directory(str(work_dir), extract_error_dirs=False)

    assert caught == [], [str(w.message) for w in caught]


def test_process_outcar_warns_instead_of_silently_swallowing_missing_field():
    """The bare `except:` clauses in process_outcar now warn on failure.

    Confirms the item-3 fix: a genuinely-missing field still degrades to
    np.nan (unchanged behaviour), but the failure is no longer invisible.
    """
    outcar = Outcar()
    outcar.from_file(str(FIXTURE_DIR / "OUTCAR"))
    structure = Structure.from_file(str(FIXTURE_DIR / "POSCAR"))

    del outcar.parse_dict["forces"]

    with pytest.warns(UserWarning, match="forces"):
        result_df = process_outcar(outcar, structure)

    assert pd.isna(result_df["forces"].iloc[0])
    # Unrelated fields are unaffected by the one missing key.
    assert np.isfinite(np.asarray(result_df["energy"].iloc[0], dtype=float)).all()
