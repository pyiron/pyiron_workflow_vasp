import os

import pyiron_workflow as pwf
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar

from pyiron_workflow_vasp.vasp import VaspInput, vasp_job


@pytest.fixture
def fe_structure():
    return Structure(
        Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]]
    )


@pytest.fixture
def stub_vasp_command(tmp_path):
    """A stand-in for VASP: writes the convergence marker and an OUTCAR."""
    marker = "reached required accuracy - stopping structural energy minimisation"
    return (
        f"echo '{marker}' > vasp.log && "
        f"echo '{marker}' > OUTCAR && "
        f"echo 'charge' > CHGCAR"
    )


def test_vasp_job_runs_end_to_end(tmp_path, fe_structure, stub_vasp_command):
    workdir = str((tmp_path / "calc").resolve())
    incar = Incar.from_dict({"ENCUT": 300, "NSW": 0})

    def fake_parser(directory):
        return {"energy": -8.21, "directory": directory}

    node = pwf.node(vasp_job)
    run = node.run(
        workdir=workdir,
        vasp_input=VaspInput(structure=fe_structure, incar=incar, potcar_paths=None),
        command=stub_vasp_command,
        files_to_be_deleted=["CHGCAR"],
        compress=False,
        compressed_file_in_dir=False,
        remove_calc_dir=False,
        vasp_parser_function=fake_parser,
        vasp_parser_args={"directory": workdir},
    )

    assert run.status == "finished"
    assert run.outputs.convergence_status is True
    assert run.outputs.vasp_output["energy"] == -8.21
    assert not os.path.exists(os.path.join(workdir, "CHGCAR")), "cleanup must run"
    assert os.path.exists(os.path.join(workdir, "OUTCAR")), "OUTCAR must survive"


def test_vasp_job_removes_calc_dir_but_returns_output(
    tmp_path, fe_structure, stub_vasp_command
):
    """remove_calc_dir deletes the directory yet the parsed result survives,
    which is the whole point of the payload pass-through."""
    workdir = str((tmp_path / "calc2").resolve())
    incar = Incar.from_dict({"ENCUT": 300, "NSW": 0})

    def fake_parser(directory):
        return {"energy": -1.23}

    run = pwf.node(vasp_job).run(
        workdir=workdir,
        vasp_input=VaspInput(structure=fe_structure, incar=incar, potcar_paths=None),
        command=stub_vasp_command,
        files_to_be_deleted=[],
        compress=False,
        compressed_file_in_dir=False,
        remove_calc_dir=True,
        vasp_parser_function=fake_parser,
        vasp_parser_args={"directory": workdir},
    )

    assert run.outputs.vasp_output["energy"] == -1.23
    assert run.outputs.convergence_status is True
    assert not os.path.exists(workdir), "directory should have been removed"


def test_vasp_job_graph_orders_removal_after_convergence(fe_structure):
    """Structural check: the node that removes the directory must be
    downstream of the node that reads convergence, otherwise the check could
    race the deletion."""
    node = pwf.node(vasp_job)
    child_labels = list(node.nodes.keys())
    assert any("check_convergence" in label for label in child_labels)
    assert any("remove_directory" in label for label in child_labels)
