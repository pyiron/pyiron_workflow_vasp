import os

import pyiron_workflow as pwf
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar

from pyiron_workflow_vasp.vasp import VaspInput, vasp_job

# potcar_paths=[] (an empty list, not None) below: VaspInput.__post_init__
# only resolves the lazy POTCAR-config default when potcar_paths IS None, so
# [] sidesteps config lookup exactly like an explicit path list would --
# these tests only care about vasp_job's ordering/cleanup behaviour, not
# real POTCAR content, and must stay hermetic under the suite's autouse
# config-nonexistence fixture (see conftest.py). write_POTCAR handles an
# empty list by writing a (valid but empty) POTCAR file.


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
        vasp_input=VaspInput(structure=fe_structure, incar=incar, potcar_paths=[]),
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
        vasp_input=VaspInput(structure=fe_structure, incar=incar, potcar_paths=[]),
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


def _ancestors(edges, target_label):
    """Transitive ancestors of ``target_label`` in a flowrep node's ``.edges``.

    Each edge's ``.source``/``.target`` exposes a ``.node`` attribute that is
    ``None`` for graph-level inputs/outputs and a child label otherwise; only
    node-to-node edges matter for dependency ordering.
    """
    parents: dict[str, set[str]] = {}
    for edge in edges:
        src_node = edge.source.node
        dst_node = edge.target.node
        if src_node is None or dst_node is None:
            continue
        parents.setdefault(dst_node, set()).add(src_node)

    seen: set[str] = set()
    stack = list(parents.get(target_label, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def test_vasp_job_graph_orders_removal_after_convergence(fe_structure):
    """Structural check: remove_directory_0 must be a transitive descendant
    of check_convergence_0 -- i.e. check_convergence_0 must be a real
    ancestor of remove_directory_0 in the built graph's edges, not just a
    node that happens to exist somewhere in the same graph. Without this,
    the calc directory could be compressed/removed before convergence is
    ever read.

    A version of this test that only checked that both labels appeared in
    ``node.nodes.keys()`` would be true regardless of edges and would stay
    true even if the ``after=convergence`` argument were deleted from
    ``compress_directory(...)`` in vasp_job -- i.e. it could not detect the
    exact race it is written to prevent. This version asserts on the actual
    ancestor relationship and fails when that edge is removed.
    """
    node = pwf.node(vasp_job)
    ancestors = _ancestors(node.edges, "remove_directory_0")
    assert "check_convergence_0" in ancestors


def test_vasp_job_graph_orders_parse_after_shell(fe_structure):
    """parse_vasp_output must run after run_shell -- VASP has to have
    actually executed before its output directory is parsed."""
    node = pwf.node(vasp_job)
    ancestors = _ancestors(node.edges, "parse_vasp_output_0")
    assert "run_shell_0" in ancestors


def test_vasp_job_graph_orders_cleanup_after_parse(fe_structure):
    """delete_files_recursively must run after parse_vasp_output -- deleting
    CHG/CHGCAR/WAVECAR before the parser has read the output directory would
    risk deleting files the parser still needs."""
    node = pwf.node(vasp_job)
    ancestors = _ancestors(node.edges, "delete_files_recursively_0")
    assert "parse_vasp_output_0" in ancestors


def test_vasp_job_runs_without_files_to_be_deleted_argument(
    tmp_path, fe_structure, stub_vasp_command
):
    """files_to_be_deleted defaults to None in vasp_job's signature, and
    delete_files_recursively must tolerate that (treat it as "delete
    nothing") rather than raising TypeError from `file in None`."""
    workdir = str((tmp_path / "calc3").resolve())
    incar = Incar.from_dict({"ENCUT": 300, "NSW": 0})

    def fake_parser(directory):
        return {"energy": -3.21}

    run = pwf.node(vasp_job).run(
        workdir=workdir,
        vasp_input=VaspInput(structure=fe_structure, incar=incar, potcar_paths=[]),
        command=stub_vasp_command,
        compress=False,
        compressed_file_in_dir=False,
        remove_calc_dir=False,
        vasp_parser_function=fake_parser,
        vasp_parser_args={"directory": workdir},
    )

    assert run.status == "finished"
    assert run.outputs.vasp_output["energy"] == -3.21
    assert os.path.exists(os.path.join(workdir, "CHGCAR")), (
        "with no files_to_be_deleted, nothing should have been cleaned up"
    )
