import os

import pyiron_workflow as pwf
import pytest

from pyiron_workflow_vasp.vasp import (
    check_convergence,
    create_working_directory,
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


from pymatgen.io.vasp.inputs import Incar

import pyiron_workflow_vasp.vasp as vasp_mod
from pyiron_workflow_vasp.vasp import generate_modified_incar


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
