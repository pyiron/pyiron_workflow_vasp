"""Internal: assemble VASP inputs, run the binary, parse -> EngineOutput.

Not part of the public API - callers should go through
``VaspEngine.get_calculate_fn`` which returns this function as its
callable. The kwargs surface mirrors the keys VaspEngine bakes into
the return tuple.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ase import Atoms
from pyiron_workflow_atomistics.engine import (
    CalcInputMinimize,
    CalcInputStatic,
    EngineOutput,
)


def run_vasp(
    structure: Atoms,
    working_directory: str,
    engine_input: CalcInputStatic | CalcInputMinimize,
    potcar_config_file: Path | None,
    functional: str,
    encut: float,
    kpoints_density: float,
    command: str,
    mode: str,
) -> EngineOutput:
    """Run a VASP single-point or relaxation and return an EngineOutput.

    Workflow:
        1. mkdir working_directory
        2. write POSCAR (via ASE)
        3. write INCAR (built from engine_input + mode + encut)
        4. write KPOINTS (built from kpoints_density)
        5. write POTCAR (via potcar_config_file)
        6. shell out to `command`
        7. parse via vaspparser.vasp.output.parse_vasp_output
        8. map parsed dict -> EngineOutput
    """
    from vaspparser.vasp.output import parse_vasp_output

    from pyiron_workflow_vasp.generic import shell
    from pyiron_workflow_vasp.vasp import (
        VaspInput,
        get_default_POTCAR_paths,
        read_potcar_config,
        write_INCAR,
        write_KPOINTS,
        write_POSCAR,
        write_POTCAR,
    )

    os.makedirs(working_directory, exist_ok=True)

    # 1. POSCAR (via the existing helper - ASE-Atoms-aware)
    write_POSCAR(workdir=working_directory, structure=structure)

    # 2. INCAR - build dict from engine_input + mode, hand to pymatgen Incar
    incar = _build_incar(engine_input=engine_input, mode=mode, encut=encut)
    write_INCAR(workdir=working_directory, incar=incar)

    # 3. KPOINTS - automatic density via pymatgen Kpoints
    kpoints = _build_kpoints(structure=structure, kpoints_density=kpoints_density)
    write_KPOINTS(workdir=working_directory, kpoints=kpoints)

    # 4. POTCAR - resolve via potcar_config_file
    if potcar_config_file is not None:
        config = read_potcar_config(potcar_config_file)
        pseudopot_lib_path = config["default_POTCAR_library_path"]
    else:
        pseudopot_lib_path = None

    potcar_paths = get_default_POTCAR_paths(
        structure=structure,
        pseudopot_lib_path=pseudopot_lib_path,
        pseudopot_functional=functional,
    )
    vasp_input = VaspInput(
        structure=structure,
        potcar_paths=potcar_paths,
        pseudopot_lib_path=pseudopot_lib_path,
        pseudopot_functional=functional,
    )
    write_POTCAR(workdir=working_directory, vasp_input=vasp_input)

    # 5. Run the binary - shell is a pyiron_workflow Node; use the raw callable
    shell.node_function(command=command, workdir=working_directory)

    # 6. Parse the output
    parsed = parse_vasp_output(working_directory=working_directory)

    # 7. Map -> EngineOutput
    return _to_engine_output(parsed=parsed)


def _build_incar(
    engine_input: CalcInputStatic | CalcInputMinimize,
    mode: str,
    encut: float,
):
    """Build a pymatgen Incar from the engine_input + mode + encut."""
    from pymatgen.io.vasp.inputs import Incar

    params: dict[str, Any] = {
        "ENCUT": encut,
        "PREC": "Accurate",
        "EDIFF": 1e-5,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LREAL": "Auto",
    }

    if mode == "static":
        params["NSW"] = 0
        params["IBRION"] = -1
    elif mode == "minimize":
        assert isinstance(engine_input, CalcInputMinimize)
        params["NSW"] = engine_input.max_iterations
        params["IBRION"] = 2  # conjugate gradient
        params["EDIFFG"] = -abs(engine_input.force_convergence_tolerance)
        params["ISIF"] = 3 if engine_input.relax_cell else 2
        if engine_input.energy_convergence_tolerance is not None:
            params["EDIFF"] = engine_input.energy_convergence_tolerance
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    return Incar.from_dict(params)


def _build_kpoints(structure: Atoms, kpoints_density: float):
    """Build a pymatgen Kpoints using automatic density (A^-1)."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Kpoints

    pmg_structure = AseAtomsAdaptor.get_structure(structure)
    return Kpoints.automatic_density(
        structure=pmg_structure,
        kppa=int(1000 / max(kpoints_density, 0.01)),  # rough density mapping
    )


def _to_engine_output(parsed: dict) -> EngineOutput:
    """Map the vaspparser parse_vasp_output return dict to EngineOutput.

    The parse_vasp_output result is structured as
        {"generic": {"energy_tot": [...], "positions": [...], "cells": [...],
                     "forces": [...], "stresses": [...], ...},
         "converged": bool, ...}
    Specific keys depend on vaspparser version - adapt here.
    """
    from ase import Atoms as ASEAtoms

    generic = parsed.get("generic", parsed)

    # Trajectory
    energies = list(generic.get("energy_pot", generic.get("energy_tot", [])))
    forces_traj = list(generic.get("forces", []))
    stresses_traj = list(generic.get("stresses", generic.get("pressures", [])))

    # Reconstruct trajectory structures if cell/positions are present
    structures_traj: list = []
    cells = generic.get("cells")
    positions = generic.get("positions")
    indices = generic.get("indices")
    if cells is not None and positions is not None and indices is not None:
        species = parsed.get("species") or parsed.get("species_list")
        for c, p, idx in zip(cells, positions, indices):
            if species is None:
                # Best effort: fall back to atom symbols from initial structure
                continue
            symbols = [species[i] for i in idx]
            structures_traj.append(
                ASEAtoms(symbols=symbols, positions=p, cell=c, pbc=True)
            )

    final_structure = (
        structures_traj[-1] if structures_traj else parsed.get("final_structure")
    )
    final_energy = energies[-1] if energies else parsed.get("final_total_energy")
    converged = bool(parsed.get("converged", False))

    return EngineOutput(
        final_structure=final_structure,
        final_energy=final_energy,
        converged=converged,
        final_forces=forces_traj[-1] if forces_traj else None,
        final_stress=stresses_traj[-1] if stresses_traj else None,
        final_volume=(
            final_structure.get_volume() if final_structure is not None else None
        ),
        energies=energies or None,
        forces=forces_traj or None,
        stresses=stresses_traj or None,
        structures=structures_traj or None,
        n_ionic_steps=len(energies) if energies else None,
    )
