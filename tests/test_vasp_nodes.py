import os

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
import pytest
from ase import Atoms as AseAtoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar

import pyiron_workflow_vasp.vasp as vasp_mod
from pyiron_workflow_vasp.vasp import (
    check_convergence,
    construct_sequential_vasp_input,
    create_working_directory,
    generate_modified_incar,
    generate_vasp_input,
    parse_vasp_output,
)


def test_create_working_directory_makes_it(tmp_path):
    target = str((tmp_path / "fresh").resolve())
    run = pwf.node(create_working_directory).run(workdir=target)
    assert run.outputs.workdir == target
    assert os.path.isdir(target)


def test_create_working_directory_is_idempotent(abs_workdir):
    run = pwf.node(create_working_directory).run(workdir=abs_workdir, quiet=True)
    assert run.outputs.workdir == abs_workdir
    assert os.path.isdir(abs_workdir)


def test_check_convergence_reads_vasp_log(fake_vasp_dir):
    run = pwf.node(check_convergence).run(workdir=str(fake_vasp_dir))
    assert run.outputs.convergence is True


def test_check_convergence_false_when_no_marker(abs_workdir):
    run = pwf.node(check_convergence).run(workdir=abs_workdir)
    assert run.outputs.convergence is False


def test_parse_vasp_output_uses_supplied_function(abs_workdir):
    def fake_parser(directory):
        return {"directory": directory, "energy": -8.21}

    run = pwf.node(parse_vasp_output).run(
        workdir=abs_workdir,
        function=fake_parser,
        parser_args={"directory": abs_workdir},
    )
    assert run.outputs.output_dict["energy"] == -8.21


def test_parse_vasp_output_after_token_does_not_change_result(abs_workdir):
    def fake_parser(directory):
        return {"directory": directory}

    run = pwf.node(parse_vasp_output).run(
        workdir=abs_workdir,
        function=fake_parser,
        parser_args={"directory": abs_workdir},
        after="ignored-ordering-token",
    )
    assert run.outputs.output_dict["directory"] == abs_workdir


def test_generate_modified_incar_overrides_tag():
    base = Incar.from_dict({"ENCUT": 400, "ISIF": 3, "NSW": 100})
    run = pwf.node(generate_modified_incar).run(
        incar=base, modifications={"ISIF": 7}
    )
    assert run.outputs.incar["ISIF"] == 7
    assert run.outputs.incar["ENCUT"] == 400


def test_generate_modified_incar_does_not_mutate_input():
    base = Incar.from_dict({"ENCUT": 400, "ISIF": 3})
    pwf.node(generate_modified_incar).run(incar=base, modifications={"ISIF": 7})
    assert base["ISIF"] == 3, "the source INCAR must be left untouched"


def test_get_multiple_input_is_gone():
    """ForEach broadcasts non-iterated inputs, so this helper is obsolete."""
    assert not hasattr(vasp_mod, "get_multiple_input")


# --- VaspInput.structure normalization (moved to __post_init__) -----------
#
# write_POSCAR/get_default_POTCAR_paths only understand ase.Atoms, but there
# are three producers of VaspInput.structure (generate_vasp_input,
# construct_sequential_vasp_input, and direct construction) that may each be
# handed either an ase.Atoms or a pymatgen Structure. Normalization lives in
# VaspInput.__post_init__ so every producer benefits without mutating the
# caller's original object.


def test_generate_vasp_input_normalizes_pymatgen_structure_without_mutating_it():
    source = Structure(Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]])
    incar = Incar.from_dict({"ENCUT": 300})

    run = pwf.node(generate_vasp_input).run(
        structure=source, incar=incar, potcar_paths=None
    )

    assert isinstance(run.outputs.vasp_input.structure, AseAtoms)
    assert isinstance(source, Structure), "the caller's pymatgen Structure must survive unchanged"


def test_generate_vasp_input_leaves_ase_atoms_as_is():
    atoms = AseAtomsAdaptor.get_atoms(
        Structure(Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]])
    )
    incar = Incar.from_dict({"ENCUT": 300})

    run = pwf.node(generate_vasp_input).run(
        structure=atoms, incar=incar, potcar_paths=None
    )

    assert run.outputs.vasp_input.structure is atoms


def test_construct_sequential_vasp_input_round_trips_last_structure():
    """``vasp_output.structures`` (as produced by parse_vasp_directory, see
    vasp_parser/output.py) holds an array of ``Structure.to_json()`` strings
    per row, one per ionic step -- NOT structure objects. This must survive
    a real JSON round-trip and land as a genuine structure, not a string."""
    first_step = Structure(Lattice.cubic(4.0), ["Ni"], [[0.0, 0.0, 0.0]])
    last_step = Structure(Lattice.cubic(3.5), ["Ni"], [[0.0, 0.0, 0.0]])
    fake_vasp_output = pd.DataFrame(
        [{"structures": np.array([first_step.to_json(), last_step.to_json()])}]
    )
    incar = Incar.from_dict({"ENCUT": 300})

    run = pwf.node(construct_sequential_vasp_input).run(
        vasp_output=fake_vasp_output, incar=incar, potcar_paths=None
    )

    result_structure = run.outputs.vasp_input.structure
    assert not isinstance(result_structure, str)
    assert isinstance(result_structure, AseAtoms)
    assert len(result_structure) == 1
    assert result_structure.cell.cellpar()[0] == pytest.approx(3.5), (
        "must pick up the LAST ionic step (3.5 A), not the first (4.0 A)"
    )
