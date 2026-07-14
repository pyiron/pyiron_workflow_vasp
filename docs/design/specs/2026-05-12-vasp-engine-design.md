# VaspEngine — bring `pyiron_workflow_vasp` onto the pyiron_workflow_atomistics Engine Protocol

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-12 |
| Repo | `ligerzero-ai/pyiron_workflow_vasp` |
| Upstream contract | [pyiron_workflow_atomistics Engine Protocol](https://github.com/pyiron/pyiron_workflow_atomistics/blob/main/docs/design/specs/2026-05-12-pyiron-workflow-atomistics-cleanup-design.md) |
| Upstream conformance suite | [Engine conformance suite spec](https://github.com/pyiron/pyiron_workflow_atomistics/blob/main/docs/design/specs/2026-05-12-engine-conformance-suite-design.md) |
| Companion spec | `pyiron_workflow_lammps/docs/design/specs/2026-05-12-engine-protocol-migration-design.md` |

## Problem

`pyiron_workflow_vasp` (HEAD = `ff54816`) does not depend on `pyiron_workflow_atomistics` at all. It exposes a collection of standalone helper functions (`pyiron_workflow_vasp.vasp.read_potcar_config`, INCAR/KPOINTS construction, POTCAR resolution, `parse_vasp_output`) wrapping `vaspparser` and pymatgen, plus a handful of shell utilities in `generic.py`. There is no `Engine` class.

Consequences:

1. Atomistics' physics macros (`get_vacancy_formation_energy`, `calculate_surface_energy`, etc.) cannot be run against VASP — they require an object satisfying the `Engine` Protocol. Today a user wanting VASP results writes ad-hoc pyiron_workflow nodes by hand.
2. The Engine ecosystem has only one backend in production (`ASEEngine`) and one downstream that conforms after migration (`LammpsEngine`). VASP — the most-requested DFT backend in the pyiron user base — is conspicuously absent.
3. The pyproject pins are years out of date: `numpy>=1.22,<=1.26`, `ase>=3.23.0,<3.25.1`, `pyiron_workflow==0.13.3`, `pymatgen==2023.10.11`. Co-installation with atomistics 0.0.5 (`numpy==1.26.4`, `ase==3.28.0`, `pyiron-workflow==0.15.6`, `pymatgen==2026.5.4`) is impossible without a pin sweep.

The job here is to add a `VaspEngine` that satisfies the Engine Protocol (Static + Minimize modes only this cycle), wrap the existing parse/run helpers under it, and bring the whole repo current with the atomistics 0.0.5 dependency profile. Numbers from the existing helper paths must not drift.

## Approach

Greenfield-style: add a new `engine.py` module that hosts `VaspEngine` as a `@dataclass` satisfying the Protocol. The existing standalone helpers in `vasp.py` and `vasp_parser/` stay untouched — `VaspEngine` composes them via its `get_calculate_fn` and `_run.py` glue. The migration is fully additive on the code side; the pyproject sweep is the only breaking change for existing consumers.

Match the LAMMPS migration's break-freely posture: bump pyproject pins to match atomistics 0.0.5 verbatim, ship as a fresh `0.1.0` (current is `0.0.x`, no public ABI to preserve), document changes in CHANGELOG.md.

## Components

```
pyiron_workflow_vasp/
├── __init__.py                 # MODIFIED — re-export VaspEngine
├── engine.py                   # NEW — VaspEngine dataclass
├── _run.py                     # NEW — internal run_vasp callable wrapping existing helpers
├── vasp.py                     # UNCHANGED — keeps standalone helpers
├── generic.py                  # UNCHANGED — shell utilities
├── vasp_parser/                # UNCHANGED — parse_vasp_output
├── vasp_resources/             # UNCHANGED — POTCAR config templates
└── _version.py                 # NEW — versioneer-managed
tests/
├── unit/
│   ├── test_engine_conformance.py    # NEW — subclasses EngineConformanceTests, uses mock vasp_std
│   └── test_numerical_regression.py  # NEW — golden parse outputs pinned
└── fixtures/
    ├── cu_static/                    # NEW — canned vasprun.xml + OUTCAR + CONTCAR for 4-atom Cu
    └── cu_minimize/                  # NEW — same for a 6-step relaxation
docs/
└── design/
    ├── specs/2026-05-12-vasp-engine-design.md   # this file
    └── plans/2026-05-12-vasp-engine.md          # implementation plan
CHANGELOG.md                          # NEW
```

### `VaspEngine` shape

```python
"""VaspEngine satisfying the pyiron_workflow_atomistics Engine Protocol."""
from __future__ import annotations
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal

from ase import Atoms
from pyiron_workflow_atomistics.engine import (
    CalcInputMinimize, CalcInputStatic, EngineOutput,
)


@dataclass
class VaspEngine:
    """Static + Minimize VASP backend for the pyiron_workflow Engine ecosystem.

    The Engine Protocol's type union includes CalcInputMD, but this engine
    does NOT yet implement MD ensembles. Passing CalcInputMD raises
    NotImplementedError at construction time. Deferred to a follow-up.
    """

    EngineInput: CalcInputStatic | CalcInputMinimize
    working_directory: str = field(default_factory=os.getcwd)

    # VASP-specific configuration mirroring the existing helper-function call surface
    potcar_config_file: Path | None = None          # default: ~/.pyiron_vasp_config
    functional: Literal["GGA", "LDA"] = "GGA"
    encut: float = 520.0                             # eV
    kpoints_density: float = 0.30                    # Å^-1 — pymatgen Kpoints.automatic_density
    command: str = "vasp_std"
    mode: Literal["static", "minimize"] = field(init=False)

    def __post_init__(self):
        if isinstance(self.EngineInput, CalcInputMinimize):
            self.mode = "minimize"
        elif isinstance(self.EngineInput, CalcInputStatic):
            self.mode = "static"
        else:
            raise NotImplementedError(
                f"VaspEngine: MD support not yet implemented "
                f"(got {type(self.EngineInput).__name__}). See "
                f"https://github.com/.../issues/<TBD>."
            )

    def with_working_directory(self, subdir: str) -> "VaspEngine":
        """Pure copy with the working directory composed."""
        return replace(
            self, working_directory=os.path.join(self.working_directory, subdir)
        )

    def get_calculate_fn(
        self, structure: Atoms
    ) -> tuple[Callable[..., EngineOutput], dict[str, Any]]:
        from pyiron_workflow_vasp._run import run_vasp
        kwargs = {
            "working_directory":  self.working_directory,
            "engine_input":       self.EngineInput,
            "potcar_config_file": self.potcar_config_file,
            "functional":         self.functional,
            "encut":              self.encut,
            "kpoints_density":    self.kpoints_density,
            "command":            self.command,
            "mode":               self.mode,
        }
        return run_vasp, kwargs
```

### `_run.run_vasp` — internal callable

```python
"""Internal: assemble VASP inputs, run, parse → EngineOutput."""
from ase import Atoms
from pyiron_workflow_atomistics.engine import EngineOutput

def run_vasp(
    structure: Atoms,
    working_directory: str,
    engine_input,
    potcar_config_file,
    functional: str,
    encut: float,
    kpoints_density: float,
    command: str,
    mode: str,
) -> EngineOutput:
    """End-to-end: write inputs → run binary → parse → EngineOutput."""
    from pyiron_workflow_vasp.vasp import (
        read_potcar_config, write_incar, write_kpoints, write_poscar_from_ase,
        write_potcar,
    )
    from pyiron_workflow_vasp.generic import shell
    from vaspparser.vasp.output import parse_vasp_output

    os.makedirs(working_directory, exist_ok=True)
    write_poscar_from_ase(structure, working_directory)
    write_incar(working_directory, engine_input=engine_input, mode=mode, encut=encut)
    write_kpoints(structure, working_directory, density=kpoints_density)
    write_potcar(structure, working_directory,
                 config=read_potcar_config(potcar_config_file), functional=functional)
    shell(command, working_directory=working_directory)
    parsed = parse_vasp_output(working_directory)
    return _to_engine_output(parsed)


def _to_engine_output(parsed: dict) -> EngineOutput:
    return EngineOutput(
        final_structure    = parsed["final_structure"],          # ase.Atoms
        final_energy       = parsed["final_total_energy"],
        converged          = parsed["converged"],
        final_forces       = parsed.get("final_forces"),
        final_stress       = parsed.get("final_stress"),          # (3, 3) tensor
        final_stress_voigt = parsed.get("final_stress_voigt"),    # (6,) Voigt
        final_volume       = parsed.get("final_volume"),
        final_magmoms      = parsed.get("final_magmoms"),
        energies           = parsed.get("trajectory_energies"),
        forces             = parsed.get("trajectory_forces"),
        stresses           = parsed.get("trajectory_stresses"),
        structures         = parsed.get("trajectory_structures"),
        n_ionic_steps      = parsed.get("n_ionic_steps"),
    )
```

Caveat: `parse_vasp_output` from `vaspparser` returns a dict-like object; the exact keys above are placeholders that we'll match to whatever it actually yields during implementation. The contract is "map upstream keys to `EngineOutput` field names" — the mapping table goes in the implementation plan, not the spec.

### Mapping `CalcInput*` → INCAR

| `CalcInputStatic` | INCAR |
|---|---|
| (no fields) | `NSW = 0` (single point) |

| `CalcInputMinimize` | INCAR |
|---|---|
| `force_convergence_tolerance` | `EDIFFG = -<value>` (negative ⇒ force criterion in eV/Å) |
| `energy_convergence_tolerance` | `EDIFF = <value>` |
| `max_iterations` | `NSW = <value>` |
| `relax_cell = True` | `ISIF = 3`; otherwise `ISIF = 2` |

`encut` and `kpoints_density` come from the engine, not the CalcInput — those are convergence parameters that aren't engine-agnostic.

## Verification — conformance suite + numerical regression

### Conformance harness with a mocked VASP binary

A real `vasp_std` is unavailable in CI. Conformance still has to verify shape, not just compile. `tests/unit/test_engine_conformance.py`:

```python
from pyiron_workflow_atomistics.testing import EngineConformanceTests
from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_vasp.engine import VaspEngine

class TestVaspEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        # Use a fake `vasp_std` that copies canned outputs into cwd then exits 0.
        fixture = Path(__file__).parent.parent / "fixtures" / "cu_static"
        fake_cmd = f"bash -c 'cp -r {fixture}/* . && echo VASP OK'"
        return VaspEngine(
            EngineInput=CalcInputStatic(),
            working_directory=str(tmp_path),
            command=fake_cmd,
            potcar_config_file=fixture / "potcar.config",
        )
```

The mock copies a real (manually-generated, one-off) `vasprun.xml` + `OUTCAR` + `CONTCAR` from a tiny 4-atom Cu single-point calculation into the working directory, then the parser runs against those. Shape conformance is verified end-to-end without needing a VASP binary in CI.

A `@pytest.mark.real_vasp` companion marker exercises the same engine against an actual `vasp_std` — opt-in, run locally before tagging a release, never gates CI.

### Numerical-regression gate

Capture the parsed outputs of the canned fixtures into `tests/unit/test_numerical_regression.py` golden values. The fixtures are bytewise stable, so the golden values are immutable as long as `parse_vasp_output` itself doesn't change. If vaspparser ever bumps and the parser output drifts, the test trips immediately.

Beyond fixture-parsing: there is one example script in `example_notebooks/` and a smoke suite under `tests/`. Run both against `main` before this PR opens, record printed values, pin in the regression test.

## Commits in this PR

| # | Commit | Touches |
|--:|--------|---------|
| 1 | `chore: bump pyproject pins to atomistics 0.0.5` | `pyproject.toml`. |
| 2 | `feat(engine): add VaspEngine satisfying Engine Protocol` | `engine.py`, `_run.py`, `__init__.py`. |
| 3 | `test: add canned VASP-output fixtures (cu_static, cu_minimize)` | `tests/fixtures/`. |
| 4 | `test(engine): subclass EngineConformanceTests with mock vasp_std` | `tests/unit/test_engine_conformance.py`. |
| 5 | `test(numerical): pin parser golden values + existing example outputs` | `tests/unit/test_numerical_regression.py`. |
| 6 | `chore: docs, CHANGELOG, ship v0.1.0` | `docs/`, `CHANGELOG.md`, version tag. |

Release: tag `pyiron_workflow_vasp-0.1.0`; if there's a `pyproject-release.yml` workflow this publishes to PyPI; if not, manual `uv build` + `uv publish`.

## Pyproject sweep

Replace the entire `dependencies = [...]` block with atomistics 0.0.5's verbatim:

```toml
dependencies = [
    "numpy==1.26.4",
    "pandas==3.0.2",
    "matplotlib==3.10.9",
    "ase==3.28.0",
    "scipy==1.17.1",
    "pyiron-workflow==0.15.6",
    "pyiron-workflow-atomistics==0.0.5",
    "pymatgen==2026.5.4",
    "pyiron_snippets==1.2.1",
    "scikit-learn==1.8.0",
    "tqdm==4.67.3",
    "vaspparser==0.0.6",
]
```

`vaspparser==0.0.6` stays pinned — it's the VASP parse-output dependency, orthogonal to atomistics.

Add a `[project.optional-dependencies] test = ["pytest", "nbformat", "nbclient"]` matching atomistics.

## CI footprint

The repo doesn't currently have a `.ci_support/environment.yml` — it relies on pip + pyproject. Add one matching atomistics' shape so the same shared workflows (push-pull, pyproject-release) can run. Conda dependencies as listed in the sweep above; `pip:` for `pyiron-workflow-atomistics`, `vaspparser`, `pyiron_workflow` (whatever isn't on conda-forge). The conformance suite uses fixtures so no real VASP install is needed.

## Out of scope

- MD ensembles (NVT/NPT). Deferred — Protocol allows it but VASP MD has enough complexity (NHC chains, AIMD timestep selection, thermostat damping) to warrant its own design pass.
- Hybrid functionals (HSE06, B3LYP), GW, BSE. Base PBE/GGA + LDA only.
- POTCAR redistribution. The conformance fixture uses a tiny dummy POTCAR; real-world use requires user-provided licensed POTCARs as today.
- Reorganising `vasp.py` into smaller modules. Existing helpers stay where they are; `VaspEngine` composes them via thin adapters in `_run.py`.
- Org migration (`ligerzero-ai/` → `pyiron/`). Orthogonal — user decides separately whether to transfer the repo.

## Risk register

1. **`parse_vasp_output` return-shape uncertainty**: the implementation plan must inventory exactly which keys it yields before commit 2 can be merged. If keys are missing (e.g. no `final_stress_voigt`), the `EngineOutput` adapter computes them from raw stress tensor in `_to_engine_output`.
2. **Mock-command brittleness**: the canned fixtures must contain exactly the files `parse_vasp_output` expects to read. If a future vaspparser version requires new files (e.g. `vaspout.h5`), the conformance suite breaks. Pin `vaspparser==0.0.6` and explicitly bump only with a regenerated fixture set.
3. **Pickle safety of `Path` field**: `pathlib.Path` pickles fine; no concern. Functions / callables are not stored on the engine (only command strings), so the pickle test passes trivially.
4. **`pyiron_workflow` 0.13.3 → 0.15.6 API drift**: the existing helper functions use `pyiron_workflow.Workflow` directly. Audit during implementation; rename / adjust call sites as needed in commit 2.

## Companion repos

- [`pyiron_workflow_atomistics`](https://github.com/pyiron/pyiron_workflow_atomistics) — owns the Protocol contract + conformance suite. Must release 0.0.5 before this PR opens.
- [`pyiron_workflow_lammps`](https://github.com/pyiron/pyiron_workflow_lammps) — parallel migration on the same contract. Independent PR, lands first because it's the easier proof of pattern.
