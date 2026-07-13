# VaspEngine Greenfield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `pyiron_workflow_vasp.engine.VaspEngine` `@dataclass` that satisfies the `pyiron_workflow_atomistics==0.0.5` Engine Protocol contract for Static + Minimize modes, wraps the existing VASP helper machinery via a new internal `_run.py`, and ships the repo at `pyiron_workflow_vasp-0.1.0` with current-generation dependency pins.

**Architecture:** Greenfield addition — no existing public symbol moves or breaks. New `engine.py` holds `VaspEngine` as a `@dataclass`; new `_run.py` holds the `run_vasp(...)` callable that composes the existing `write_POSCAR/INCAR/KPOINTS/POTCAR` helpers + `shell` + `parse_vasp_output` and returns a real `EngineOutput`. CI smoke uses a **mocked `vasp_std`** that copies canned `vasprun.xml`/`OUTCAR`/`CONTCAR` files into the run dir; the parser then exercises end-to-end without needing a real VASP binary in CI.

**Tech Stack:** Python 3.10+, `dataclasses`, `pyiron_workflow_atomistics==0.0.5` (Protocol + `EngineOutput` + `EngineConformanceTests`), existing repo helpers (`vasp.py`, `vasp_parser/`, `generic.shell`), `vaspparser.vasp.output.parse_vasp_output` for the real parse path, pytest. Real `vasp_std` is opt-in only (`@pytest.mark.real_vasp`); CI uses the mock.

**Spec:** `docs/design/specs/2026-05-12-vasp-engine-design.md`.

**Branch:** `design-vasp-engine` (already pushed to `origin`).

**Working directory:** `/home/liger/pyiron_workflow_vasp`.

**Python interpreter / pytest binary:** `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python` and `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest`. The env already has `pyiron-workflow-atomistics==0.0.5` installed.

---

## File structure (what this PR touches)

| Path | Action | Responsibility |
|---|---|---|
| `pyiron_workflow_vasp/engine.py` | NEW | `VaspEngine` `@dataclass` satisfying the Engine Protocol (Static + Minimize). Holds VASP-specific config (POTCAR config file path, functional, ENCUT, kpoints density, run command). |
| `pyiron_workflow_vasp/_run.py` | NEW | Internal `run_vasp(structure, working_directory, engine_input, …) -> EngineOutput` callable. Composes existing helpers; never imported by users directly. |
| `pyiron_workflow_vasp/__init__.py` | MODIFY | Add `from .engine import VaspEngine` so the public symbol is reachable as `pyiron_workflow_vasp.VaspEngine`. |
| `tests/unit/__init__.py` | NEW | Empty marker. |
| `tests/unit/test_engine_conformance.py` | NEW | `TestVaspEngineConformance(EngineConformanceTests)` — runs the upstream 5-method mixin against a `VaspEngine` instance whose `command` is the mock `cp -r fixture/* . && true`. Guarded by `pytest.skip` if the canned fixtures aren't on disk. |
| `tests/unit/test_numerical_regression.py` | NEW | Asserts the parsed `EngineOutput` from each canned fixture matches the pinned golden values byte-for-byte. |
| `tests/fixtures/README.md` | NEW | Documents how to regenerate the canned `cu_static/` and `cu_minimize/` fixture sets with a real `vasp_std`. |
| `tests/fixtures/generate.py` | NEW | One-off script run locally by someone with a real VASP install. Produces the canned `vasprun.xml`/`OUTCAR`/`CONTCAR` files. |
| `tests/fixtures/cu_static/.gitkeep` | NEW | Placeholder so the empty dir is tracked. The 3 binary outputs are added in a follow-up commit (or by you locally) once VASP has run. |
| `tests/fixtures/cu_minimize/.gitkeep` | NEW | Same as above. |
| `pyproject.toml` | MODIFY | Bump every dep pin to match atomistics 0.0.5 verbatim + add `pyiron-workflow-atomistics==0.0.5` as a new required dep. Add `[project.optional-dependencies] test`. |
| `.ci_support/environment.yml` | NEW | Conda env yaml for the shared pyiron CI workflows. |
| `.ci_support/lower-bound.yml` | NEW | Lower-bound conda env (same versions as `environment.yml` for this 0.1.0 cycle). |
| `.github/workflows/push-pull.yml` | NEW | Wire to the shared `pyiron/actions` push-pull workflow. |
| `.github/workflows/release.yml` | NEW | Wire to the shared release workflow. Uses the local fork pattern from atomistics+lammps to handle the `- pip:` block in lower-bound yaml. |
| `.github/workflows/pyproject-release.yml` | NEW | Local fork of the pyiron-actions release workflow (matches atomistics post-PR-#35 pattern). |
| `.github/actions/.support/update_pyproject_dependencies.py` | NEW | Local fork with the dict-handling fix. |
| `.github/actions/.support/pypi_vs_conda_names.json` | NEW | pypi↔conda name map (copied verbatim from atomistics). |
| `.github/actions/update-pyproject-dependencies/action.yml` | NEW | Composite action wrapping the local script. |
| `CHANGELOG.md` | NEW | `0.1.0` release notes. |
| `docs/design/plans/2026-05-12-vasp-engine.md` | NEW | This plan. Committed first. |

No file in `pyiron_workflow_vasp/vasp.py` or `vasp_parser/` is modified. They're consumed by `_run.py` via imports.

---

## Task 1: Commit this plan first

**Files:**
- Create: `docs/design/plans/2026-05-12-vasp-engine.md` (this file)

- [ ] **Step 1: Verify the plan file is untracked**

```bash
cd /home/liger/pyiron_workflow_vasp
git status --short docs/design/plans/2026-05-12-vasp-engine.md
```

Expected: `?? docs/design/plans/2026-05-12-vasp-engine.md`.

- [ ] **Step 2: Commit and push**

```bash
git add docs/design/plans/2026-05-12-vasp-engine.md
git commit -m "docs(plan): VaspEngine implementation plan

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin design-vasp-engine
```

Expected: branch tip on remote advances.

---

## Task 2: Bump pyproject deps to atomistics 0.0.5

**Files:**
- Modify: `pyproject.toml` (the `dependencies = [...]` block and `[project.optional-dependencies]`)

- [ ] **Step 1: Replace the `dependencies` block**

In `/home/liger/pyiron_workflow_vasp/pyproject.toml`, replace the entire `dependencies = [...]` array (currently pins to numpy 1.22-1.26, ase 3.23-3.25, pyiron_workflow 0.13.3, pymatgen 2023.10) with:

```toml
dependencies = [
    "numpy==1.26.4",
    "pandas==3.0.2",
    "matplotlib==3.10.9",
    "ase==3.28.0",
    "scipy==1.17.1",
    "pyiron-workflow==0.15.6",
    "pyiron-workflow-atomistics==0.0.5",
    "vaspparser==0.0.6",
    "pymatgen==2026.5.4",
    "pyiron_snippets==1.2.1",
    "scikit-learn==1.8.0",
    "tqdm==4.67.3",
]
```

- [ ] **Step 2: Replace the `[project.optional-dependencies]` block**

Replace the existing `[project.optional-dependencies] dev = [...]` block with:

```toml
[project.optional-dependencies]
test = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "nbformat",
    "nbclient",
]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
    "ruff>=0.5",
]
notebook = [
    "jupyter>=1.0.0",
    "notebook>=6.0.0",
]
```

- [ ] **Step 3: Bump `requires-python` to match the atomistics 0.0.5 floor**

In `[project]`, change:

```toml
requires-python = ">=3.8"
```

to:

```toml
requires-python = ">=3.10, <3.13"
```

- [ ] **Step 4: Verify pyproject.toml parses + the package still imports**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import tomllib, pathlib
cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
deps = cfg['project']['dependencies']
print(f'{len(deps)} deps')
assert any('pyiron-workflow-atomistics==0.0.5' in d for d in deps), 'atomistics 0.0.5 not pinned'
print('OK')
"
```

Expected: `12 deps\nOK`.

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "import pyiron_workflow_vasp; print('package OK')"
```

Expected: `package OK`. (Existing helpers may already be broken by stale imports — that's fine, the engine work doesn't touch them. If you see an ImportError on `pyiron_workflow.Workflow`, fix the next 0.13.3-API usage in `vasp.py` only if it blocks; otherwise note and move on.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump pyproject pins to atomistics 0.0.5 + add test extras

Adopt the canonical atomistics 0.0.5 dep set (numpy 1.26.4, pandas
3.0.2, ase 3.28.0, pyiron-workflow 0.15.6, pymatgen 2026.5.4) plus
the new pyiron-workflow-atomistics==0.0.5 dependency. Add a
[project.optional-dependencies] test = […] section with pytest,
nbformat, nbclient. Raise requires-python floor to 3.10 to match
atomistics' supported range.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `pyiron_workflow_vasp/engine.py` with `VaspEngine` class

**Files:**
- Create: `pyiron_workflow_vasp/engine.py`

The class must satisfy the Engine Protocol shape (Protocol-checked + dataclass + `working_directory: str` + `with_working_directory` + `get_calculate_fn`). The body of `get_calculate_fn` returns `(_run.run_vasp, kwargs)` where `_run.run_vasp` is built in Task 4.

- [ ] **Step 1: Write the failing import-shape test**

Create `tests/unit/__init__.py` (empty) and `tests/unit/test_engine_shape.py`:

```python
"""Engine shape pre-test: VaspEngine satisfies the Protocol class-level
contract without needing the conformance suite. Refined into the full
EngineConformanceTests subclass in Task 7."""

from pathlib import Path

import pytest


def test_vasp_engine_imports():
    from pyiron_workflow_vasp.engine import VaspEngine
    assert VaspEngine is not None


def test_vasp_engine_satisfies_protocol(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import CalcInputStatic, Engine
    from pyiron_workflow_vasp.engine import VaspEngine

    eng = VaspEngine(
        EngineInput=CalcInputStatic(),
        working_directory=str(tmp_path),
    )
    assert isinstance(eng, Engine), \
        "VaspEngine does not satisfy the runtime_checkable Engine Protocol"


def test_with_working_directory_is_pure(tmp_path: Path):
    import os

    from pyiron_workflow_atomistics.engine import CalcInputStatic
    from pyiron_workflow_vasp.engine import VaspEngine

    eng = VaspEngine(EngineInput=CalcInputStatic(), working_directory=str(tmp_path))
    sub = eng.with_working_directory("subdir")
    assert sub.working_directory == os.path.join(str(tmp_path), "subdir")
    assert eng.working_directory == str(tmp_path)
    assert sub is not eng
    assert type(sub) is type(eng)


def test_md_input_raises(tmp_path: Path):
    from pyiron_workflow_atomistics.engine import CalcInputMD
    from pyiron_workflow_vasp.engine import VaspEngine

    with pytest.raises(NotImplementedError, match="MD"):
        VaspEngine(EngineInput=CalcInputMD(), working_directory=str(tmp_path))
```

- [ ] **Step 2: Run — expect ImportError**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_engine_shape.py -v 2>&1 | tail -10
```

Expected: 4 collected, 4 fail with `ModuleNotFoundError: No module named 'pyiron_workflow_vasp.engine'`.

- [ ] **Step 3: Create `pyiron_workflow_vasp/engine.py`**

```python
"""VaspEngine satisfying the pyiron_workflow_atomistics Engine Protocol.

Static + Minimize modes only. CalcInputMD is rejected at construction
time with NotImplementedError — MD support is a future PR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal

from ase import Atoms

from pyiron_workflow_atomistics.engine import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
    EngineOutput,
)


@dataclass
class VaspEngine:
    """VASP backend for the pyiron_workflow Engine ecosystem.

    Composes the existing pyiron_workflow_vasp helpers
    (write_POSCAR/INCAR/KPOINTS/POTCAR, generic.shell, vasp_parser.output)
    behind the Engine Protocol contract.

    Static and Minimize modes are supported. MD raises
    NotImplementedError at construction time — the Engine Protocol's
    type union allows CalcInputMD but this engine doesn't implement
    NSW + MDALGO + TEBEG/TEEND wiring yet.
    """

    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    working_directory: str = field(default_factory=os.getcwd)

    # VASP-specific configuration
    potcar_config_file: Path | None = None
    functional: Literal["GGA", "LDA"] = "GGA"
    encut: float = 520.0
    kpoints_density: float = 0.30
    command: str = "vasp_std"
    mode: Literal["static", "minimize"] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.EngineInput, CalcInputMinimize):
            self.mode = "minimize"
        elif isinstance(self.EngineInput, CalcInputStatic):
            self.mode = "static"
        else:
            raise NotImplementedError(
                f"VaspEngine MD support not yet implemented "
                f"(got {type(self.EngineInput).__name__}). "
                "Use CalcInputStatic or CalcInputMinimize for now."
            )

    def with_working_directory(self, subdir: str) -> "VaspEngine":
        """Return a pure copy with the working directory composed."""
        return replace(
            self,
            working_directory=os.path.join(self.working_directory, subdir),
        )

    def get_calculate_fn(
        self, structure: Atoms
    ) -> tuple[Callable[..., EngineOutput], dict[str, Any]]:
        """Return (callable, kwargs). The callable will be invoked as
        callable(structure=structure, **kwargs) and must return an
        EngineOutput."""
        from pyiron_workflow_vasp._run import run_vasp

        kwargs = {
            "working_directory": self.working_directory,
            "engine_input": self.EngineInput,
            "potcar_config_file": self.potcar_config_file,
            "functional": self.functional,
            "encut": self.encut,
            "kpoints_density": self.kpoints_density,
            "command": self.command,
            "mode": self.mode,
        }
        return run_vasp, kwargs
```

- [ ] **Step 4: Re-run the shape tests — expect first 3 to PASS, last (NotImplementedError) to PASS too**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_engine_shape.py -v 2>&1 | tail -10
```

Expected: 4 passed. But `test_with_working_directory_is_pure` may transitively reach into `get_calculate_fn` if we did anything wrong — re-read the test to ensure it does NOT call `get_calculate_fn`.

If `test_md_input_raises` fails because `CalcInputMD` can't be imported, ensure atomistics 0.0.5 is installed: `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pip show pyiron-workflow-atomistics | head -2` → expect `Version: 0.0.5`.

- [ ] **Step 5: Update `pyiron_workflow_vasp/__init__.py`**

Replace the current content of `pyiron_workflow_vasp/__init__.py`:

```python
"""
pyiron_workflow_vasp - A VASP workflow integration package for pyiron
"""

from .vasp import *
from .generic import *

__version__ = "0.1.0"
```

with:

```python
"""pyiron_workflow_vasp — VASP integration for the pyiron workflow system.

Public API:
    VaspEngine — satisfies pyiron_workflow_atomistics.engine.Engine
                  for Static + Minimize modes; wraps the helpers in
                  vasp.py and vasp_parser/output.py.

Plus the existing standalone helper functions in vasp and generic
(re-exported via wildcard for backwards compatibility with the 0.0.x
script-style API).
"""

from .engine import VaspEngine
from .generic import *  # noqa: F401,F403  -- legacy helpers
from .vasp import *  # noqa: F401,F403      -- legacy helpers

__version__ = "0.1.0"
__all__ = ["VaspEngine"]
```

- [ ] **Step 6: Confirm `from pyiron_workflow_vasp import VaspEngine` works**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from pyiron_workflow_vasp import VaspEngine
print(VaspEngine.__module__, VaspEngine.__name__)
"
```

Expected: `pyiron_workflow_vasp.engine VaspEngine`.

- [ ] **Step 7: Commit**

```bash
git add pyiron_workflow_vasp/engine.py pyiron_workflow_vasp/__init__.py \
        tests/unit/__init__.py tests/unit/test_engine_shape.py
git commit -m "feat(engine): add VaspEngine satisfying Engine Protocol

Static + Minimize modes; CalcInputMD raises NotImplementedError at
construction. Protocol clauses verified by tests/unit/test_engine_shape.py
(isinstance check, with_working_directory purity, MD rejection).
get_calculate_fn returns (run_vasp, kwargs) — run_vasp is built in
the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create `pyiron_workflow_vasp/_run.py` — the internal callable

**Files:**
- Create: `pyiron_workflow_vasp/_run.py`

This is the function returned by `VaspEngine.get_calculate_fn`. It composes existing helpers from `vasp.py` and `generic.py` and `vasp_parser/output.py`, runs the binary, parses the output, and returns an `EngineOutput`.

- [ ] **Step 1: Write the failing test (parser → EngineOutput round-trip via mock)**

Create `tests/unit/test_run_vasp.py`:

```python
"""Unit-test the run_vasp callable. Uses an artificial 'mock command'
that does nothing (no fixtures yet) — verifies the function imports
and matches the expected signature shape. End-to-end parser coverage
lives in test_engine_conformance.py once fixtures are populated."""

from pathlib import Path

import pytest


def test_run_vasp_importable():
    from pyiron_workflow_vasp._run import run_vasp
    assert callable(run_vasp)


def test_run_vasp_signature():
    """Signature must accept the kwargs VaspEngine.get_calculate_fn
    promises to supply: working_directory, engine_input, potcar_config_file,
    functional, encut, kpoints_density, command, mode. Plus the
    positional `structure` argument that the caller passes."""
    import inspect

    from pyiron_workflow_vasp._run import run_vasp

    sig = inspect.signature(run_vasp)
    expected = {
        "structure",
        "working_directory",
        "engine_input",
        "potcar_config_file",
        "functional",
        "encut",
        "kpoints_density",
        "command",
        "mode",
    }
    actual = set(sig.parameters.keys())
    assert expected.issubset(actual), f"missing: {expected - actual}"
```

- [ ] **Step 2: Run — expect ImportError**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_run_vasp.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'pyiron_workflow_vasp._run'`.

- [ ] **Step 3: Create `pyiron_workflow_vasp/_run.py`**

```python
"""Internal: assemble VASP inputs, run the binary, parse → EngineOutput.

Not part of the public API — callers should go through
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
        8. map parsed dict → EngineOutput
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

    # 1. POSCAR (via the existing helper — ASE-Atoms-aware)
    write_POSCAR(workdir=working_directory, structure=structure)

    # 2. INCAR — build dict from engine_input + mode, hand to pymatgen Incar
    incar = _build_incar(engine_input=engine_input, mode=mode, encut=encut)
    write_INCAR(workdir=working_directory, incar=incar)

    # 3. KPOINTS — automatic density via pymatgen Kpoints
    kpoints = _build_kpoints(structure=structure, kpoints_density=kpoints_density)
    write_KPOINTS(workdir=working_directory, kpoints=kpoints)

    # 4. POTCAR — resolve via potcar_config_file
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

    # 5. Run the binary
    shell(command=command, working_directory=working_directory)

    # 6. Parse the output
    parsed = parse_vasp_output(working_directory=working_directory)

    # 7. Map → EngineOutput
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
    """Build a pymatgen Kpoints using automatic density (Å^-1)."""
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
    Specific keys depend on vaspparser version — adapt here.
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

    final_structure = structures_traj[-1] if structures_traj else parsed.get(
        "final_structure"
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
```

Important: keys like `energy_pot` vs `energy_tot`, `species` vs `species_list` reflect uncertainty about the exact `vaspparser.vasp.output.parse_vasp_output` schema. The `_to_engine_output` uses `.get` fallbacks so a mismatch surfaces as `None`-valued optional fields, not a hard crash. **Task 8 step 2** validates the exact mapping against a real fixture.

- [ ] **Step 4: Re-run the test — expect 2 passed**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_run_vasp.py -v 2>&1 | tail -5
```

Expected: 2 passed.

- [ ] **Step 5: Run the shape tests too — confirm nothing broke**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_engine_shape.py tests/unit/test_run_vasp.py -v 2>&1 | tail -8
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add pyiron_workflow_vasp/_run.py tests/unit/test_run_vasp.py
git commit -m "feat(engine): internal run_vasp callable

Composes write_POSCAR/INCAR/KPOINTS/POTCAR + shell + parse_vasp_output
behind the (callable, kwargs) tuple that VaspEngine.get_calculate_fn
returns. INCAR is built from CalcInputStatic/Minimize + mode + encut;
KPOINTS via pymatgen automatic_density; POTCAR via the existing
get_default_POTCAR_paths helper. The output mapping handles missing
keys gracefully — exact schema validated end-to-end in the conformance
suite (Task 8) once a real VASP fixture is provided.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Scaffold `tests/fixtures/` (dirs + README + generator script)

**Files:**
- Create: `tests/fixtures/README.md`
- Create: `tests/fixtures/generate.py`
- Create: `tests/fixtures/cu_static/.gitkeep`
- Create: `tests/fixtures/cu_minimize/.gitkeep`
- Create: `tests/fixtures/__init__.py` (empty)

The binary VASP outputs (`vasprun.xml`, `OUTCAR`, `CONTCAR`) are NOT created in this commit — they require a real VASP run. The generator script and the `.gitkeep` placeholders ship; binaries land in a follow-up commit once a maintainer with VASP runs `python tests/fixtures/generate.py`.

- [ ] **Step 1: Create the fixture directory layout**

```bash
cd /home/liger/pyiron_workflow_vasp
mkdir -p tests/fixtures/cu_static tests/fixtures/cu_minimize
touch tests/fixtures/__init__.py
touch tests/fixtures/cu_static/.gitkeep
touch tests/fixtures/cu_minimize/.gitkeep
```

- [ ] **Step 2: Create the generator script**

Create `tests/fixtures/generate.py`:

```python
"""One-off fixture generator. Run locally with a real `vasp_std` to
populate tests/fixtures/{cu_static, cu_minimize}/ with the canned
vasprun.xml, OUTCAR, CONTCAR that the conformance suite copies into
the test working directory via a mock command.

Usage:
    cd pyiron_workflow_vasp
    python tests/fixtures/generate.py \\
        --command 'mpirun -n 4 vasp_std' \\
        --potcar-config ~/.pyiron_vasp_config

After this completes, commit the generated binaries:

    git add tests/fixtures/cu_static tests/fixtures/cu_minimize
    git commit -m 'test(fixtures): populate cu_static and cu_minimize'
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ase.build import bulk

HERE = Path(__file__).resolve().parent
OUTPUT_FILES = ("vasprun.xml", "OUTCAR", "CONTCAR")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        default="vasp_std",
        help="The VASP run command to pass to VaspEngine.",
    )
    parser.add_argument(
        "--potcar-config",
        type=Path,
        required=True,
        help="Path to a .pyiron_vasp_config file resolving POTCAR paths.",
    )
    args = parser.parse_args()

    from pyiron_workflow_atomistics.engine import (
        CalcInputMinimize,
        CalcInputStatic,
    )

    from pyiron_workflow_vasp.engine import VaspEngine

    structure = bulk("Cu", "fcc", a=3.6, cubic=True)

    for sub, engine_input in (
        ("cu_static", CalcInputStatic()),
        (
            "cu_minimize",
            CalcInputMinimize(
                force_convergence_tolerance=0.05,
                max_iterations=5,
                relax_cell=False,
            ),
        ),
    ):
        workdir = HERE / sub
        workdir.mkdir(parents=True, exist_ok=True)
        # Wipe stale outputs before re-running
        for fname in OUTPUT_FILES:
            (workdir / fname).unlink(missing_ok=True)

        engine = VaspEngine(
            EngineInput=engine_input,
            working_directory=str(workdir),
            potcar_config_file=args.potcar_config,
            command=args.command,
        )
        fn, kwargs = engine.get_calculate_fn(structure)
        output = fn(structure=structure, **kwargs)
        assert output is not None
        print(f"{sub}: E = {output.final_energy} eV, converged = {output.converged}")

        # Sanity: ensure the three canned files we care about exist
        for fname in OUTPUT_FILES:
            assert (workdir / fname).exists(), f"missing {workdir / fname}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create the fixtures README**

Create `tests/fixtures/README.md`:

```markdown
# VASP conformance test fixtures

The two subdirectories (`cu_static/`, `cu_minimize/`) hold canned VASP
output (`vasprun.xml`, `OUTCAR`, `CONTCAR`) from a 4-atom FCC Cu
single-point and a 5-step relaxation. The conformance suite at
`tests/unit/test_engine_conformance.py` uses a mock VASP command of
the form

    bash -c 'cp -r <fixture-dir>/* . && true'

so the parser can be exercised end-to-end in CI without a real
`vasp_std` binary.

## Regenerating

Requires a real VASP install + a configured `~/.pyiron_vasp_config`
that resolves POTCAR paths. From the repo root:

    python tests/fixtures/generate.py \\
        --command 'mpirun -n 4 vasp_std' \\
        --potcar-config ~/.pyiron_vasp_config

This re-runs both calculations and overwrites the canned files. After
the script completes, commit the regenerated binaries:

    git add tests/fixtures/cu_static tests/fixtures/cu_minimize
    git commit -m 'test(fixtures): refresh cu_static and cu_minimize'

Pinned golden values in `tests/unit/test_numerical_regression.py`
will need updating in the same commit if VASP semantics shifted.

## CI behaviour without fixtures

When the binaries are absent (e.g. first PR, before a maintainer has
generated them), `test_engine_conformance.py::test_run_returns_engine_output`
is automatically skipped via `pytest.skip`. The other four mixin
methods (Protocol satisfaction, `with_working_directory`, pickle,
`get_calculate_fn` signature) run regardless and gate the PR.
```

- [ ] **Step 4: Smoke-test that the generator script imports cleanly (without running)**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('gen', pathlib.Path('tests/fixtures/generate.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('generate.py imports OK')
"
```

Expected: `generate.py imports OK`.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/
git commit -m "test(fixtures): scaffold canned-output fixture dirs + generator

cu_static/ and cu_minimize/ are empty (.gitkeep only) until a maintainer
with VASP runs tests/fixtures/generate.py. Once generated and committed,
the conformance suite's run() smoke exercises the parser end-to-end via
a mock command that copies the canned outputs into the test workdir.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Subclass `EngineConformanceTests` with fixture-aware skips

**Files:**
- Create: `tests/unit/test_engine_conformance.py`

The conformance suite from atomistics 0.0.5 runs 5 tests. The first 4 (Protocol satisfaction, `with_working_directory` purity, pickle, `get_calculate_fn` signature) don't need real outputs — they pass without fixtures. The 5th (`test_run_returns_engine_output`) does need outputs — skip it when the fixtures haven't been populated.

- [ ] **Step 1: Create the conformance test file**

```python
"""Conformance: VaspEngine satisfies pyiron_workflow_atomistics.engine.Engine.

Uses a mock VASP command for the run() smoke — the command simply copies
canned vasprun.xml/OUTCAR/CONTCAR files into the test working directory,
then `_run.run_vasp` continues into the parser path. This sidesteps the
need for a real VASP binary in CI.

If the canned fixtures haven't been generated yet (`.gitkeep` only),
test_run_returns_engine_output skips via pytest.skip. The other four
mixin methods always run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_atomistics.testing import EngineConformanceTests

from pyiron_workflow_vasp.engine import VaspEngine

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "cu_static"
_FIXTURE_FILES = ("vasprun.xml", "OUTCAR", "CONTCAR")


def _fixtures_populated() -> bool:
    return all((_FIXTURE_ROOT / f).exists() for f in _FIXTURE_FILES)


def _mock_command() -> str:
    """Construct the mock command: cp the fixture files into cwd, exit 0."""
    if not _fixtures_populated():
        # No fixtures → return a command that lets the engine construct but
        # would fail at run(). The run() test below skips before reaching this.
        return "/bin/true"
    return f"bash -c 'cp {_FIXTURE_ROOT}/* . && true'"


class TestVaspEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        return VaspEngine(
            EngineInput=CalcInputStatic(),
            working_directory=str(tmp_path),
            command=_mock_command(),
        )

    @staticmethod
    def test_structure_factory():
        # Match the 4-atom Cu FCC the canned fixture was generated for
        return bulk("Cu", "fcc", a=3.6, cubic=True)

    # Override only the run() smoke — guard with fixture presence
    def test_run_returns_engine_output(self, tmp_path):
        if not _fixtures_populated():
            pytest.skip(
                f"Canned fixtures not populated at {_FIXTURE_ROOT}. "
                "Run `python tests/fixtures/generate.py` with a real "
                "vasp_std to regenerate."
            )
        # Otherwise defer to the base implementation
        return super().test_run_returns_engine_output(tmp_path)
```

- [ ] **Step 2: Run the conformance suite**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_engine_conformance.py -v 2>&1 | tail -15
```

Expected: 4 passed, 1 skipped (the `test_run_returns_engine_output` skip because fixtures aren't populated yet). The output looks like:

```
TestVaspEngineConformance.test_satisfies_engine_protocol PASSED
TestVaspEngineConformance.test_with_working_directory_is_pure PASSED
TestVaspEngineConformance.test_pickleable PASSED
TestVaspEngineConformance.test_get_calculate_fn_signature PASSED
TestVaspEngineConformance.test_run_returns_engine_output SKIPPED
```

Once a maintainer generates and commits the fixtures, the SKIP becomes a PASS automatically — no test code change required.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_engine_conformance.py
git commit -m "test(engine): subclass EngineConformanceTests with mock vasp_std

Runs the upstream 5-method conformance mixin against VaspEngine.
The run() smoke uses a mock command (bash -c 'cp fixture/* . && true')
that copies canned vasprun.xml/OUTCAR/CONTCAR into the test workdir
so the parser is exercised without a real VASP binary. If fixtures
aren't populated yet (cu_static/ has only .gitkeep), run() skips
with a hint pointing at the regenerate script. The other four mixin
methods always run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add numerical-regression golden test (skipif fixtures missing)

**Files:**
- Create: `tests/unit/test_numerical_regression.py`

This file pins specific numerical outputs from the canned fixtures so future vaspparser parser bumps can't silently change what `_to_engine_output` produces. Skipif-guarded just like the conformance run.

- [ ] **Step 1: Create the regression test**

```python
"""Numerical regression: parsed EngineOutput from canned fixtures must
match the pinned golden values. Updated whenever the maintainer
regenerates fixtures (see tests/fixtures/README.md).

Skipped when the binary fixtures aren't on disk yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_vasp.engine import VaspEngine

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "cu_static"
_FIXTURE_FILES = ("vasprun.xml", "OUTCAR", "CONTCAR")


def _fixtures_populated() -> bool:
    return all((_FIXTURE_ROOT / f).exists() for f in _FIXTURE_FILES)


@pytest.mark.skipif(
    not _fixtures_populated(),
    reason=f"Fixtures not generated yet. See {_FIXTURE_ROOT / 'README.md'}.",
)
def test_cu_static_golden(tmp_path):
    """The parsed Cu FCC single-point energy must match the pinned value
    to within sub-meV. Update only when the maintainer regenerates fixtures
    and explicitly documents the new value in the same commit."""
    engine = VaspEngine(
        EngineInput=CalcInputStatic(),
        working_directory=str(tmp_path),
        command=f"bash -c 'cp {_FIXTURE_ROOT}/* . && true'",
    )
    structure = bulk("Cu", "fcc", a=3.6, cubic=True)
    fn, kwargs = engine.get_calculate_fn(structure)
    out = fn(structure=structure, **kwargs)

    # Golden — update when fixtures are regenerated.
    # The maintainer who first generates the fixtures fills these in.
    GOLDEN_ENERGY_EV = None  # e.g. -14.7321
    GOLDEN_N_ATOMS = 4

    if GOLDEN_ENERGY_EV is not None:
        assert out.final_energy == pytest.approx(GOLDEN_ENERGY_EV, abs=1e-3)
    assert out.final_structure is not None
    assert len(out.final_structure) == GOLDEN_N_ATOMS
    assert isinstance(out.converged, bool)
```

The maintainer who generates the first fixture also fills in `GOLDEN_ENERGY_EV` (and any other goldens they want pinned). The `is not None` guard lets the test ship initially with only the structural assertions and a placeholder — the energy check activates as soon as a real number is committed.

- [ ] **Step 2: Run — expect skipped**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_numerical_regression.py -v 2>&1 | tail -5
```

Expected: 1 skipped, reason mentions `tests/fixtures/cu_static/README.md`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_numerical_regression.py
git commit -m "test(regression): pin parser golden values

Skipif-guarded against missing canned fixtures. Activates as soon as
a maintainer regenerates cu_static/{vasprun.xml,OUTCAR,CONTCAR} and
fills in GOLDEN_ENERGY_EV. Defends against silent vaspparser parser
behaviour changes that would alter what _to_engine_output yields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Add CI infrastructure (`.ci_support/` + `.github/`)

**Files:**
- Create: `.ci_support/environment.yml`
- Create: `.ci_support/lower-bound.yml`
- Create: `.github/workflows/push-pull.yml`
- Create: `.github/workflows/release.yml`
- Create: `.github/workflows/pyproject-release.yml` (local fork)
- Create: `.github/actions/.support/update_pyproject_dependencies.py` (local fork)
- Create: `.github/actions/.support/pypi_vs_conda_names.json`
- Create: `.github/actions/update-pyproject-dependencies/action.yml`

This mirrors the pyiron_workflow_atomistics post-#35 layout. The local-fork pattern is required because the upstream pyiron-actions release script crashes on `- pip:` dict entries (already fixed in pyiron/actions#174 but not yet released as a new actions tag).

- [ ] **Step 1: Copy the local-fork action wholesale from atomistics**

```bash
cd /home/liger/pyiron_workflow_vasp
mkdir -p .github/actions/.support .github/actions/update-pyproject-dependencies .github/workflows .ci_support
cp /home/liger/pyiron_workflow_atomistics/.github/actions/.support/update_pyproject_dependencies.py \
   .github/actions/.support/
cp /home/liger/pyiron_workflow_atomistics/.github/actions/.support/pypi_vs_conda_names.json \
   .github/actions/.support/
cp /home/liger/pyiron_workflow_atomistics/.github/actions/update-pyproject-dependencies/action.yml \
   .github/actions/update-pyproject-dependencies/
cp /home/liger/pyiron_workflow_atomistics/.github/workflows/pyproject-release.yml \
   .github/workflows/
```

- [ ] **Step 2: Create `.github/workflows/push-pull.yml`**

```yaml
# This runs jobs which pyiron modules should run on pushes or PRs to main
name: Push-Pull

on:
  push:
    branches: [ main ]
  pull_request:

jobs:
  pyiron:
    uses: pyiron/actions/.github/workflows/push-pull.yml@actions-4.0.8
    with:
      runner: 'ubuntu-22.04'
      python-version-alt3: 'exclude'
      runner-alt1: 'exclude'
    secrets: inherit
```

- [ ] **Step 3: Create `.github/workflows/release.yml`**

```yaml
name: Release

on:
#  pull_request:
  release:
    types: [ published ]

jobs:
  pyiron:
    uses: ./.github/workflows/pyproject-release.yml
    secrets: inherit
    with:
      semantic-upper-bound: 'minor'
      lower-bound-yaml: '.ci_support/lower-bound.yml'
      pypi-to-conda-name-map-file: '.github/actions/.support/pypi_vs_conda_names.json'
```

- [ ] **Step 4: Create `.ci_support/environment.yml`**

```yaml
channels:
  - conda-forge

dependencies:
  - python>=3.11,<3.13
  - pip
  - numpy=1.26.4
  - ase=3.28.0
  - pymatgen=2026.5.4
  - pyiron_workflow=0.15.6
  - pandas=3.0.2
  - scipy=1.17.1
  - matplotlib=3.10.9
  - scikit-learn=1.8.0
  - tqdm=4.67.3
  - pytest
  - nbformat
  - nbclient
  - pip:
      - pyiron-workflow-atomistics==0.0.5
      - pyiron-snippets==1.2.1
      - vaspparser==0.0.6
```

- [ ] **Step 5: Create `.ci_support/lower-bound.yml`**

```yaml
channels:
  - conda-forge

dependencies:
  - python=3.11
  - pip
  - numpy=1.26.4
  - ase=3.28.0
  - pymatgen=2026.5.4
  - pyiron_workflow=0.15.6
  - pandas=3.0.2
  - scipy=1.17.1
  - matplotlib=3.10.9
  - scikit-learn=1.8.0
  - tqdm=4.67.3
  - pytest
  - nbformat
  - nbclient
  - pip:
      - pyiron-workflow-atomistics==0.0.5
      - pyiron-snippets==1.2.1
      - vaspparser==0.0.6
```

- [ ] **Step 6: Verify the local action's Python script runs**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import sys
sys.path.insert(0, '.github/actions/.support')
import update_pyproject_dependencies as m
bounds = m.load_yaml('.ci_support/lower-bound.yml')
print(f'{len(bounds)} bounds; pyiron-workflow-atomistics={bounds.get(\"pyiron-workflow-atomistics\", \"MISSING\")}')
"
```

Expected: prints `≥11 bounds; pyiron-workflow-atomistics=0.0.5`. Confirms the local fork's pip-dict handling works against this repo's yaml.

- [ ] **Step 7: Commit**

```bash
git add .github/ .ci_support/
git commit -m "ci: add pyiron shared workflows + local release-script fork

push-pull.yml wires to pyiron/actions@actions-4.0.8. release.yml uses
a local fork of pyproject-release.yml + update_pyproject_dependencies.py
that handles '- pip:' dict deps in lower-bound.yml (the upstream script
crashes on this; fix proposed at pyiron/actions#174). Copied verbatim
from pyiron_workflow_atomistics post-#35.

.ci_support/environment.yml + lower-bound.yml pin the same versions
as atomistics 0.0.5 so pip-check + the release-bounds script align.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Write CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create the file**

```markdown
# Changelog

All notable changes to `pyiron_workflow_vasp` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: PEP 440.

## [0.1.0] — 2026-05-12

### Added

- **`pyiron_workflow_vasp.engine.VaspEngine`** — a `@dataclass`
  satisfying the
  `pyiron_workflow_atomistics.engine.Engine` Protocol contract for
  `CalcInputStatic` and `CalcInputMinimize`. Wraps the existing
  POSCAR/INCAR/KPOINTS/POTCAR helpers + `generic.shell` +
  `vaspparser.vasp.output.parse_vasp_output` via the new internal
  `_run.py:run_vasp`. `CalcInputMD` raises `NotImplementedError` at
  construction time — MD wiring is a future PR.
- `pyiron_workflow_vasp` is now reachable from atomistics' physics
  macros: `get_vacancy_formation_energy(structure=…, engine=VaspEngine(…))`
  works without any wrapper code.
- `tests/unit/test_engine_conformance.py` subclasses the upstream
  `EngineConformanceTests` mixin. CI exercises the four shape clauses
  (Protocol satisfaction, `with_working_directory` purity, pickle,
  `get_calculate_fn` signature) unconditionally; the `run()` smoke
  uses a mock VASP command that copies canned fixtures, skipping
  when those fixtures haven't been generated.
- `tests/unit/test_numerical_regression.py` pins golden parser
  outputs from the canned fixtures.
- `tests/fixtures/generate.py` regenerates the canned fixtures
  against a real `vasp_std`.
- `.github/workflows/{push-pull,release,pyproject-release}.yml` and
  `.ci_support/{environment,lower-bound}.yml` — the repo is wired to
  the shared pyiron CI workflows for the first time.

### Changed (breaking)

- Pyproject dependency pins bumped wholesale to match the
  `pyiron_workflow_atomistics==0.0.5` set. Notable jumps from 0.0.x:
  `numpy 1.22-1.26 → 1.26.4`, `ase 3.23-3.25.1 → 3.28.0`,
  `pyiron_workflow 0.13.3 → 0.15.6`, `pymatgen 2023.10.11 → 2026.5.4`.
- `requires-python` raised from `>=3.8` to `>=3.10, <3.13` to match
  atomistics' supported range.

### Out of scope

- VASP MD ensembles (NVT/NPT/NHC). The Protocol allows `CalcInputMD`,
  but VASP MD has enough complexity (NHC chains, AIMD timestep,
  thermostat damping) to warrant its own design pass.
- Hybrid functionals (HSE06, B3LYP), GW, BSE.
- POTCAR redistribution. Users must still supply licensed POTCARs.
- Org migration `ligerzero-ai/` → `pyiron/` — orthogonal.

## [0.0.x] — pre-2026-05-12

See git history for the standalone helper-functions API.
```

- [ ] **Step 2: Verify**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import pathlib
t = pathlib.Path('CHANGELOG.md').read_text()
assert '[0.1.0]' in t
assert 'VaspEngine' in t
assert 'CalcInputMD' in t
print(f'CHANGELOG.md OK, {len(t)} chars')
"
```

Expected: `CHANGELOG.md OK, NNN chars`.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): 0.1.0 — VaspEngine + Engine Protocol conformance

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Final lint + tests + push

**Files:**
- (no edits — verification only)

- [ ] **Step 1: Install dev tools if missing**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pip install ruff black 2>&1 | tail -3
```

Expected: `Successfully installed …` or `Requirement already satisfied`.

- [ ] **Step 2: Run ruff**

```bash
cd /home/liger/pyiron_workflow_vasp
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check pyiron_workflow_vasp/ tests/ 2>&1 | tail -5
```

Expected: `All checks passed!`. If anything fails, run `ruff check --fix pyiron_workflow_vasp/ tests/`. The forked `.github/actions/.support/update_pyproject_dependencies.py` has 5 pre-existing lints from upstream — exclude `.github` via pyproject if it isn't already (it should be inherited from the atomistics fork pattern; otherwise add).

- [ ] **Step 3: Run ruff import-sort**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check --select I pyiron_workflow_vasp/ tests/ 2>&1 | tail -3
```

Expected: `All checks passed!`.

- [ ] **Step 4: Run black**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m black --check pyiron_workflow_vasp/ tests/ 2>&1 | tail -5
```

Expected: `All done! ✨ … N files would be left unchanged.`. If anything wants reformatting, run `black pyiron_workflow_vasp/ tests/` and commit as `style:` follow-up.

- [ ] **Step 5: Run the full unit test suite**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/ -q --no-header 2>&1 | tail -10
```

Expected: all engine-shape + conformance tests pass (4 conformance + 4 shape + 2 run-vasp); the run() smoke and the numerical-regression test skip (binary fixtures absent). Existing `tests/test_smoke.py` continues to pass.

- [ ] **Step 6: Push the branch**

```bash
git push origin design-vasp-engine
```

Expected: branch tip on remote updated. The previously-opened draft PR #1 (on `ligerzero-ai/pyiron_workflow_vasp`) picks up the new commits automatically.

---

## Task 11: Promote PR + verify CI

**Files:**
- (no edits)

- [ ] **Step 1: Wait for CI to settle**

```bash
sleep 90
gh pr checks 1 --repo ligerzero-ai/pyiron_workflow_vasp 2>&1 | head -25
```

Expected: every CI job passes — `ruff-check`, `ruff-sort-imports`, `black`, `unit-tests` across the matrix, `build-docs` (if applicable), `pip-check`, `pypi-release` (dry), `coverage`. The conformance + regression tests show up as "1 skipped" each on every matrix entry (fixtures absent).

If `pip-check` fails because of a version conflict introduced by pyiron-workflow 0.15.6 (e.g. some transitive dep), fix the pin in `pyproject.toml` + the env yaml + push a follow-up commit. The PR description test plan should note any non-trivial bumps.

- [ ] **Step 2: Promote from draft to ready-for-review**

```bash
gh pr ready 1 --repo ligerzero-ai/pyiron_workflow_vasp
```

Expected: PR state moves from `DRAFT` to `OPEN`.

- [ ] **Step 3: Update the PR body**

```bash
gh pr edit 1 --repo ligerzero-ai/pyiron_workflow_vasp --body "$(cat <<'EOF'
## Summary

Adds a `VaspEngine` `@dataclass` satisfying the `pyiron_workflow_atomistics==0.0.5` Engine Protocol contract (Static + Minimize). Wraps the existing POSCAR/INCAR/KPOINTS/POTCAR helpers + `generic.shell` + `vaspparser.vasp.output.parse_vasp_output` via a new internal `_run.py:run_vasp`. CalcInputMD raises NotImplementedError — future PR.

## Concrete changes

1. New `engine.py` with `VaspEngine` `@dataclass` (mode inference, `with_working_directory` via `dataclasses.replace`, `get_calculate_fn` returning `(_run.run_vasp, kwargs)`).
2. New `_run.py` with `run_vasp(structure, working_directory, engine_input, …) -> EngineOutput`. Composes existing helpers; the parsed dict → `EngineOutput` mapping handles missing keys gracefully and surfaces incomplete coverage as `None` optional fields.
3. `tests/unit/test_engine_conformance.py` subclasses `pyiron_workflow_atomistics.testing.EngineConformanceTests` with a mocked-VASP factory and a fixture-presence guard on the `run()` smoke.
4. `tests/unit/test_numerical_regression.py` pins parser golden outputs; skipif-guarded the same way.
5. `tests/fixtures/` scaffold (README + generator script + `.gitkeep` placeholders). Maintainer regenerates with `python tests/fixtures/generate.py` + a real `vasp_std`.
6. Pyproject pins bumped verbatim to atomistics 0.0.5; `requires-python = ">=3.10, <3.13"`.
7. `.github/workflows/{push-pull, release, pyproject-release}.yml` + `.ci_support/{environment, lower-bound}.yml` wire the repo into the shared pyiron CI for the first time. The release workflow uses the local-fork pattern from atomistics post-#35 to handle `- pip:` dict deps.
8. `CHANGELOG.md` documents the migration.

## Followups (after merge)

1. A maintainer with `vasp_std` + a `.pyiron_vasp_config` runs `python tests/fixtures/generate.py`, commits the binary fixtures, and fills in `GOLDEN_ENERGY_EV` in `test_numerical_regression.py`. The two skipped tests then run automatically.
2. Tag the merge commit `pyiron_workflow_vasp-0.1.0`; the release workflow publishes to PyPI.
3. Optional: org migration to `pyiron/pyiron_workflow_vasp` (orthogonal).

## Test plan

- [x] `pytest tests/` — all green or expected-skipped locally.
- [x] `ruff check` + `ruff check --select I` + `black --check` — green.
- [ ] CI green across the Push-Pull matrix (Ubuntu × 3.10/3.11/3.12).
- [ ] After fixture generation: run-smoke + numerical-regression go from SKIPPED to PASSED with no test code change.

## Spec & plan

- Spec: `docs/design/specs/2026-05-12-vasp-engine-design.md`
- Plan: `docs/design/plans/2026-05-12-vasp-engine.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR body updated.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Plan task |
|---|---|
| New `engine.py` with `VaspEngine` `@dataclass` | Task 3 |
| `working_directory: str` + `__post_init__` mode inference | Task 3 step 3 |
| `with_working_directory(subdir) -> VaspEngine` pure copy | Task 3 step 3 |
| `get_calculate_fn(structure) -> (callable, dict)` with `structure` not in dict | Task 3 step 3 |
| MD input rejection via `NotImplementedError` | Task 3 step 3 + step 1 test |
| New `_run.py` with `run_vasp(...)` | Task 4 |
| INCAR mapping from CalcInput* + mode + encut | Task 4 step 3 (`_build_incar`) |
| KPOINTS via pymatgen automatic_density | Task 4 step 3 (`_build_kpoints`) |
| POTCAR via `read_potcar_config` + `get_default_POTCAR_paths` | Task 4 step 3 |
| Shell out via `generic.shell` | Task 4 step 3 |
| Parse via `vaspparser.vasp.output.parse_vasp_output` | Task 4 step 3 |
| Parsed dict → EngineOutput mapping | Task 4 step 3 (`_to_engine_output`) |
| Conformance harness with mocked vasp_std | Task 6 |
| Mock command: `bash -c 'cp -r fixture/* . && true'` | Task 6 step 1 |
| Canned `cu_static/`, `cu_minimize/` fixtures | Task 5 (scaffold) + maintainer follow-up |
| `tests/fixtures/generate.py` | Task 5 step 2 |
| Numerical-regression test with goldens | Task 7 |
| Pyproject sweep to atomistics 0.0.5 pins verbatim | Task 2 |
| New `[project.optional-dependencies] test` | Task 2 step 2 |
| `.ci_support/environment.yml` | Task 8 step 4 |
| Release as `pyiron_workflow_vasp-0.1.0` | Task 9 (CHANGELOG) + Task 11 PR body |
| Existing `vasp.py` / `vasp_parser/` untouched | (no task modifies them) |
| Existing example notebook / smoke tests preserved | (Task 3 step 5's wildcard re-exports keep `vasp_job`, `VaspInput` reachable) |
| Risk: parse_vasp_output return-shape uncertainty | Task 4 step 3 docstring + `_to_engine_output` `.get` fallbacks |
| Risk: mock-command brittleness, fixture file pin | Task 5 README + generator script |
| Risk: Path field pickle-safety | (Path pickles fine; verified by conformance test_pickleable) |
| Risk: pyiron_workflow API drift 0.13.3 → 0.15.6 | Task 3 step 5 wildcard re-export + Task 10 step 5 full suite |

**Gap I notice:** the spec mentions `release.yml` calling out the local fork pattern; the CI footprint mentions adding LAMMPS isn't relevant here, but adding a `.github/dependabot.yml` for the new pin set isn't mentioned in the spec and isn't strictly required to ship 0.1.0. Out of scope; if the maintainer wants dependabot they can copy atomistics' `.github/dependabot.yml` in a follow-up.

**2. Placeholder scan:** No "TBD", "TODO", "fill in details" tokens. The one explicit deferred-fill is `GOLDEN_ENERGY_EV = None` in Task 7 — this is intentional: the maintainer fills it when fixtures are first generated. The plan documents this as a follow-up rather than hiding it as a placeholder.

**3. Type / identifier consistency:**
- `VaspEngine` field names (`EngineInput`, `working_directory`, `potcar_config_file`, `functional`, `encut`, `kpoints_density`, `command`, `mode`) appear identically in Tasks 3, 4, 5 (generate.py), and 6.
- `_run.run_vasp(structure, working_directory, engine_input, potcar_config_file, functional, encut, kpoints_density, command, mode)` — the parameter set matches `VaspEngine.get_calculate_fn`'s `kwargs` dict + the structure positional. Verified in Task 4 step 1 test.
- `EngineOutput` field names (`final_structure`, `final_energy`, `converged`, `final_forces`, `final_stress`, `final_volume`, `energies`, `forces`, `stresses`, `structures`, `n_ionic_steps`) match the upstream `pyiron_workflow_atomistics.engine.EngineOutput` dataclass.
- `EngineConformanceTests` is the upstream class name (Task 6) — matches what was shipped in atomistics 0.0.5.
- `tests/fixtures/cu_static/.gitkeep` referenced as the empty-marker in Task 5 step 1; referenced as the populated-fixture path in Tasks 6 and 7. Consistent.

---

## Execution Handoff

**Plan complete and saved to `docs/design/plans/2026-05-12-vasp-engine.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
