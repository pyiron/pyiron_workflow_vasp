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

    Extraction note: ``parse_vasp_output`` with ``vasp_parser_function=None``
    (the default) now goes through the EXTERNAL ``vaspparser`` package's
    ``parse_vasp_output`` -- this is the upstream default-parser switch this
    port preserves, and it is a real behavioural difference from the
    bundled/legacy ``pyiron_workflow_vasp.vasp_parser.output.parse_vasp_directory``
    (which returns a ``pandas.DataFrame`` with one row per OUTCAR). The
    external parser instead returns a single hierarchical ``dict`` (see
    ``vaspparser.vasp.output.Output.to_dict``) shaped roughly as::

        {"generic": {"energy_pot": [...], "energy_tot": [...], ...}, ...}

    i.e. one entry per ionic step, most recent last -- no restart-chain
    aggregation across multiple OUTCARs/error archives (that behaviour is
    specific to the bundled legacy parser and is exercised separately via
    ``construct_sequential_vasp_input``, which explicitly asks for the
    bundled parser instead of relying on this default).
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
    assert isinstance(output, dict), (
        f"default parser (external vaspparser package) is expected to return "
        f"a dict, got {type(output)}"
    )

    generic = output.get("generic", {})
    print(f"parsed output top-level keys: {list(output.keys())}")
    print(f"generic keys: {list(generic.keys())}")

    energies = generic.get("energy_pot") or generic.get("energy_tot")
    assert energies, f"no per-ionic-step energy in parsed output: {list(generic)}"
    last = energies[-1]
    print(f"per-ionic-step energies: {energies!r}")

    assert -20.0 < float(last) < 0.0, f"implausible energy for one Fe atom: {last}"
    # convergence_status comes from check_convergence (vasprun.xml/vasp.log
    # marker search), a separate node from the parser -- don't guess at the
    # exact key path for "converged" inside the external parser's dict shape.
    assert run.outputs.convergence_status is True

    assert not os.path.exists(os.path.join(workdir, "CHGCAR")), "cleanup did not run"
    assert os.path.exists(os.path.join(workdir, "OUTCAR")), "OUTCAR should survive"
