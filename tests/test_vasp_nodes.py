import os
import warnings

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

# Tests below that only care about structure normalization/ordering pass
# potcar_paths=[] (an empty list) rather than None. VaspInput.__post_init__
# only resolves the lazy POTCAR-config default when potcar_paths IS None
# (`is None`, not falsy), so [] sidesteps config lookup exactly like an
# explicit path list would -- without requiring a real ~/.pyiron_vasp_config
# under the suite's autouse config-nonexistence fixture (see conftest.py).


def test_create_working_directory_makes_it(tmp_path):
    target = str((tmp_path / "fresh").resolve())
    run = pwf.node(create_working_directory).run(workdir=target)
    assert run.outputs.workdir == target
    assert os.path.isdir(target)


def test_create_working_directory_is_idempotent(abs_workdir):
    run = pwf.node(create_working_directory).run(workdir=abs_workdir, quiet=True)
    assert run.outputs.workdir == abs_workdir
    assert os.path.isdir(abs_workdir)


def test_create_working_directory_quiet_suppresses_warning(abs_workdir):
    """``quiet`` was accepted but silently ignored by a prior implementation
    -- the "already exists" warning fired unconditionally. Assert both
    directions explicitly since a parameter that is accepted but never read
    is exactly the kind of bug a "no exception was raised" check can't
    catch."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pwf.node(create_working_directory).run(workdir=abs_workdir, quiet=True)
    assert caught == [], [str(w.message) for w in caught]


def test_create_working_directory_warns_when_not_quiet(abs_workdir):
    with pytest.warns(UserWarning, match="already exists"):
        pwf.node(create_working_directory).run(workdir=abs_workdir, quiet=False)


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


def test_parse_vasp_output_default_uses_external_vaspparser_package(
    abs_workdir, monkeypatch
):
    """When no ``function`` is supplied, the default parser must be the
    EXTERNAL ``vaspparser`` package's ``parse_vasp_output``, called with
    ``working_directory=workdir`` -- NOT the bundled legacy
    ``pyiron_workflow_vasp.vasp_parser`` parser. This is the upstream
    switch this port must preserve."""
    calls = []

    def fake_external_parser(**kwargs):
        calls.append(kwargs)
        return {"generic": {"energy_pot": [-1.0]}}

    import vaspparser.vasp.output as external_module

    monkeypatch.setattr(external_module, "parse_vasp_output", fake_external_parser)

    run = pwf.node(parse_vasp_output).run(workdir=abs_workdir, function=None)

    assert len(calls) == 1
    assert calls[0] == {"working_directory": abs_workdir}
    assert run.outputs.output_dict == {"generic": {"energy_pot": [-1.0]}}


def test_generate_modified_incar_overrides_tag():
    base = Incar.from_dict({"ENCUT": 400, "ISIF": 3, "NSW": 100})
    run = pwf.node(generate_modified_incar).run(incar=base, modifications={"ISIF": 7})
    assert run.outputs.incar["ISIF"] == 7
    assert run.outputs.incar["ENCUT"] == 400


def test_generate_modified_incar_does_not_mutate_input():
    base = Incar.from_dict({"ENCUT": 400, "ISIF": 3})
    pwf.node(generate_modified_incar).run(incar=base, modifications={"ISIF": 7})
    assert base["ISIF"] == 3, "the source INCAR must be left untouched"


def test_get_multiple_input_is_gone():
    """ForEach broadcasts non-iterated inputs, so this helper is obsolete."""
    assert not hasattr(vasp_mod, "get_multiple_input")


def test_submit_to_slurm_is_gone_from_vasp_module():
    assert not hasattr(vasp_mod, "submit_to_slurm")


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

    # Capture real, comparable values BEFORE the call -- these are what can
    # actually change if __post_init__ mutates the caller's object in place
    # (e.g. via scale_lattice), unlike a bare `isinstance(source, Structure)`
    # check, which stays true regardless of whether `source` was mutated
    # (only rebinding the local, which nothing in the call path can do,
    # would make it false).
    original_volume = source.volume
    original_abc = source.lattice.abc
    original_num_sites = len(source)

    run = pwf.node(generate_vasp_input).run(
        structure=source, incar=incar, potcar_paths=[]
    )

    assert isinstance(run.outputs.vasp_input.structure, AseAtoms)
    assert source.volume == pytest.approx(
        original_volume
    ), "the caller's pymatgen Structure must survive unchanged (volume changed)"
    assert source.lattice.abc == pytest.approx(
        original_abc
    ), "the caller's pymatgen Structure must survive unchanged (lattice changed)"
    assert (
        len(source) == original_num_sites
    ), "the caller's pymatgen Structure must survive unchanged (site count changed)"
    assert run.outputs.vasp_input.structure is not source, (
        "normalization must produce a new ase.Atoms object, not alias the "
        "caller's pymatgen Structure"
    )


def test_generate_vasp_input_leaves_ase_atoms_as_is():
    atoms = AseAtomsAdaptor.get_atoms(
        Structure(Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]])
    )
    incar = Incar.from_dict({"ENCUT": 300})

    run = pwf.node(generate_vasp_input).run(
        structure=atoms, incar=incar, potcar_paths=[]
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
        vasp_output=fake_vasp_output, incar=incar, potcar_paths=[]
    )

    result_structure = run.outputs.vasp_input.structure
    assert not isinstance(result_structure, str)
    assert isinstance(result_structure, AseAtoms)
    assert len(result_structure) == 1
    assert result_structure.cell.cellpar()[0] == pytest.approx(
        3.5
    ), "must pick up the LAST ionic step (3.5 A), not the first (4.0 A)"


def test_construct_sequential_vasp_input_uses_most_recent_row_not_oldest():
    """parse_vasp_directory returns one row per OUTCAR* plus one row per
    error archive, sorted ASCENDING by calc_start_time -- so a directory
    with a custodian restart or an error archive present yields MORE THAN
    ONE row. construct_sequential_vasp_input must continue from the most
    recent calculation (the last row after the ascending sort), not the
    oldest: picking the oldest would feed a crashed/earlier attempt's
    structure into the next stage of e.g. an ISIF7 -> ISIF5 -> ISIF2 chain
    run under custodian with max_errors>0."""
    older_first_step = Structure(Lattice.cubic(4.0), ["Ni"], [[0.0, 0.0, 0.0]])
    older_last_step = Structure(Lattice.cubic(3.9), ["Ni"], [[0.0, 0.0, 0.0]])
    newer_first_step = Structure(Lattice.cubic(3.6), ["Ni"], [[0.0, 0.0, 0.0]])
    newer_last_step = Structure(Lattice.cubic(3.5), ["Ni"], [[0.0, 0.0, 0.0]])

    fake_vasp_output = pd.DataFrame(
        [
            {
                "calc_start_time": pd.Timestamp("2026-01-01T00:00:00"),
                "structures": np.array(
                    [older_first_step.to_json(), older_last_step.to_json()]
                ),
            },
            {
                "calc_start_time": pd.Timestamp("2026-01-02T00:00:00"),
                "structures": np.array(
                    [newer_first_step.to_json(), newer_last_step.to_json()]
                ),
            },
        ]
    )
    incar = Incar.from_dict({"ENCUT": 300})

    run = pwf.node(construct_sequential_vasp_input).run(
        vasp_output=fake_vasp_output, incar=incar, potcar_paths=[]
    )

    result_structure = run.outputs.vasp_input.structure
    assert result_structure.cell.cellpar()[0] == pytest.approx(3.5), (
        "must pick up the last ionic step (3.5 A) of the NEWER row (later "
        "calc_start_time), not the older row's last step (3.9 A) or either "
        "row's first step (3.6/4.0 A)"
    )
