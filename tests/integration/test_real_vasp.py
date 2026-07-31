"""End-to-end validation against a real VASP binary.

Skipped unless ASSYST_VASP_BINARY is set, so the unit suite stays hermetic.
"""

import os

import pyiron_workflow as pwf
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar

from pyiron_workflow_vasp import VaspInput, vasp_job

VASP_BINARY = os.environ.get("ASSYST_VASP_BINARY")

pytestmark = pytest.mark.skipif(
    not VASP_BINARY or not os.path.isfile(VASP_BINARY),
    reason="set ASSYST_VASP_BINARY to a real vasp_std to run this",
)


@pytest.mark.integration
def test_single_point_bcc_fe_against_real_vasp(tmp_path):
    """One BCC Fe atom, single point, coarse settings -- seconds of runtime.

    Asserts on structure and sign of the result rather than an exact energy,
    so the test pins the plumbing without becoming a regression trap for
    POTCAR or VASP version changes.

    Extraction note: ``parse_vasp_output`` (the default, real parser --
    ``vasp_parser.output.parse_vasp_directory``) returns a ``pandas.DataFrame``
    with one row per parsed OUTCAR and an ``"energy"`` column holding, per
    row, a ``numpy.ndarray`` of per-ionic-step energies (see
    ``Outcar.get_total_energies``). Two things a naive
    ``float(output["energy"])`` gets wrong, discovered while writing this
    test against the versions actually resolved by ``pip install -e .``:

    1. ``output["energy"]`` is a column lookup on a DataFrame, so it returns
       a ``pandas.Series`` (one entry per row), not a scalar -- it must be
       indexed with ``.iloc[0]`` to reach the per-row array.
    2. ``pandas.Series.__float__`` was removed in pandas 3.0 (deprecated
       since 2.1); this environment resolves ``pandas==3.0.5``, so
       ``float(a_length_one_series)`` raises ``TypeError`` unconditionally,
       never mind what value it holds. ``.iloc[0]`` sidesteps this too.

    Neither point changes what is asserted (still ``-20 < energy < 0`` on
    the last ionic step's value) -- both are about correctly reaching the
    scalar pandas actually stored, not about relaxing the check.
    """
    workdir = str((tmp_path / "fe_bcc").resolve())
    incar = Incar.from_dict(
        {
            "ENCUT": 300,
            "ISMEAR": 1,
            "SIGMA": 0.2,
            "NSW": 0,
            "IBRION": -1,
            "ISPIN": 2,
            "PREC": "Low",
            "NELM": 30,
            "LWAVE": False,
            "LCHARG": False,
            "KSPACING": 0.5,
        }
    )
    structure = Structure(Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]])

    command = f"srun -n 4 --hint=nomultithread {VASP_BINARY} > vasp.log 2>&1"

    run = pwf.node(vasp_job).run(
        workdir=workdir,
        vasp_input=VaspInput(structure=structure, incar=incar, potcar_paths=None),
        command=command,
        files_to_be_deleted=["CHG", "CHGCAR", "WAVECAR"],
        compress=False,
        compressed_file_in_dir=False,
        remove_calc_dir=False,
        vasp_parser_function=None,
        vasp_parser_args=None,
    )

    assert run.status == "finished", f"graph failed: {run.exception}"

    output = run.outputs.vasp_output
    assert output is not None, "parser returned nothing for a real VASP directory"

    print(f"parsed output type: {type(output)}")
    if hasattr(output, "columns"):
        print(f"parsed output columns: {list(output.columns)}")
        print(output.to_string())

    if "energy" in output:
        per_row = output["energy"].iloc[0]
    elif "energies" in output:
        per_row = output["energies"].iloc[0]
    else:
        per_row = None
    assert per_row is not None, f"no energy key in parsed output: {list(output)}"
    print(f"per-ionic-step energies: {per_row!r}")

    last = per_row[-1] if hasattr(per_row, "__len__") else per_row
    assert -20.0 < float(last) < 0.0, f"implausible energy for one Fe atom: {last}"

    assert not os.path.exists(os.path.join(workdir, "CHGCAR")), "cleanup did not run"
    assert os.path.exists(os.path.join(workdir, "OUTCAR")), "OUTCAR should survive"
