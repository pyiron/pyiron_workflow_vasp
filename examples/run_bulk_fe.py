"""Minimal end-to-end example: relax a bcc-Fe primitive cell with VASP.

This script avoids any cluster-specific paths. Override the VASP command and
output directory via CLI flags to match your environment.

Usage
-----
    # Single calculation (writes inputs, runs VASP, parses output)
    python examples/run_bulk_fe.py \
        --workdir ./fe_bulk \
        --command "mpirun -n 4 vasp_std"

    # Equation-of-state volume scan
    python examples/run_bulk_fe.py --eos \
        --workdir ./fe_eos \
        --command "mpirun -n 4 vasp_std"

Configuration
-------------
You still need a ~/.pyiron_vasp_config (see .pyiron_vasp_config.example) so
the default POTCARs can be located. To bypass POTCAR auto-generation, pass
``--potcar`` one or more times with explicit paths to per-element POTCARs.

For a dry run that only writes input files (no VASP execution), use
``--command /bin/true`` — useful for verifying your local setup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase.build import bulk
from pymatgen.io.vasp.inputs import Incar

from pyiron_workflow_vasp.vasp import VaspInput, vasp_job


DEFAULT_INCAR = {
    "ENCUT": 400,
    "EDIFF": 1e-5,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "ISPIN": 2,
    "MAGMOM": "2*3.0",
    "PREC": "Accurate",
    "LREAL": "Auto",
}


def build_incar(extra: dict | None = None) -> Incar:
    params = dict(DEFAULT_INCAR)
    if extra:
        params.update(extra)
    return Incar.from_dict(params)


def run_single(args: argparse.Namespace) -> None:
    structure = bulk("Fe", cubic=True, a=args.a0)
    incar = build_incar()

    vasp_input = VaspInput(
        structure=structure,
        incar=incar,
        potcar_paths=list(args.potcar) if args.potcar else None,
    )

    # vasp_job is a plain @fr.workflow-decorated function: calling it
    # directly (rather than via pyiron_workflow.node(vasp_job).run(...))
    # just executes the pipeline eagerly, as ordinary Python, and returns
    # its (vasp_output, convergence_status) tuple immediately.
    vasp_output, convergence_status = vasp_job(
        workdir=str(args.workdir),
        vasp_input=vasp_input,
        command=args.command,
    )

    print(f"converged: {convergence_status}")
    if vasp_output is not None:
        # "energy_pot" is only populated when vasprun.xml was parsed; an
        # OUTCAR-only run (vasprun.xml missing/disabled) only has
        # "energy_tot" -- fall back rather than assume either is present.
        generic = vasp_output["generic"]
        energy = generic.get("energy_pot", generic.get("energy_tot"))
        print(f"final energy: {energy[-1]:.6f} eV")


def run_eos(args: argparse.Namespace) -> None:
    """Tiny equation-of-state scan using a Python for-loop (no pyiron for_node).

    Kept deliberately simple so the example reads top-to-bottom; see the
    QuickStart notebook for the for_node-based variant that runs in parallel.
    """
    base = bulk("Fe", cubic=True, a=args.a0)
    strains = np.linspace(args.strain_min, args.strain_max, args.num_points)

    incar = build_incar({"ISIF": 2, "NSW": 0})

    volumes: list[float] = []
    energies: list[float] = []

    for strain in strains:
        structure = base.copy()
        structure.set_cell(structure.get_cell() * (1.0 + float(strain)), scale_atoms=True)

        run_dir = Path(args.workdir) / f"eos_{strain:+.3f}"
        vi = VaspInput(
            structure=structure,
            incar=incar,
            potcar_paths=list(args.potcar) if args.potcar else None,
        )
        vasp_output, _convergence_status = vasp_job(
            workdir=str(run_dir), vasp_input=vi, command=args.command
        )
        if vasp_output is None:
            print(f"[strain={strain:+.3f}] no output parsed; skipping")
            continue
        generic = vasp_output["generic"]
        volumes.append(generic["volume"][-1])
        energies.append(generic.get("energy_pot", generic.get("energy_tot"))[-1])

    if not volumes:
        print("No converged points collected; nothing to fit.")
        return

    try:
        from ase.eos import EquationOfState

        eos = EquationOfState(volumes, energies, eos="sj")
        v0, e0, B = eos.fit()
        print(f"v0 = {v0:.4f} A^3   e0 = {e0:.6f} eV   B = {B:.4f} eV/A^3")
    except Exception as exc:  # pragma: no cover - ase plotting is optional
        print(f"EOS fit failed: {exc!r}")
        print("raw points (volume, energy):")
        for v, e in zip(volumes, energies):
            print(f"  {v:.4f}  {e:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, default=Path.cwd() / "pwf_vasp_example")
    p.add_argument(
        "--command",
        default=os.environ.get("VASP_CMD", "mpirun -n 1 vasp_std"),
        help="Shell command used to run VASP (also reads $VASP_CMD).",
    )
    p.add_argument("--a0", type=float, default=2.83, help="Lattice constant (A) for bcc Fe.")
    p.add_argument(
        "--potcar",
        action="append",
        default=[],
        help="Explicit POTCAR path (repeatable). When omitted, POTCARs are "
             "auto-resolved from ~/.pyiron_vasp_config.",
    )
    p.add_argument("--eos", action="store_true", help="Run a volume scan instead of a single calc.")
    p.add_argument("--num-points", type=int, default=7)
    p.add_argument("--strain-min", type=float, default=-0.08)
    p.add_argument("--strain-max", type=float, default=0.08)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    (run_eos if args.eos else run_single)(args)
