# Changelog

All notable changes to `pyiron_workflow_vasp` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: PEP 440.

## [0.1.0] - 2026-05-12

### Added

- **`pyiron_workflow_vasp.engine.VaspEngine`** - a `@dataclass`
  satisfying the
  `pyiron_workflow_atomistics.engine.Engine` Protocol contract for
  `CalcInputStatic` and `CalcInputMinimize`. Wraps the existing
  POSCAR/INCAR/KPOINTS/POTCAR helpers + `generic.shell` +
  `vaspparser.vasp.output.parse_vasp_output` via the new internal
  `_run.py:run_vasp`. `CalcInputMD` raises `NotImplementedError` at
  construction time - MD wiring is a future PR.
- `pyiron_workflow_vasp` is now reachable from atomistics' physics
  macros: `get_vacancy_formation_energy(structure=..., engine=VaspEngine(...))`
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
  `.ci_support/{environment,lower-bound}.yml` - the repo is wired to
  the shared pyiron CI workflows for the first time.

### Changed (breaking)

- Pyproject dependency pins bumped wholesale to match the
  `pyiron_workflow_atomistics==0.0.5` set. Notable jumps from 0.0.x:
  `numpy 1.22-1.26 -> 1.26.4`, `ase 3.23-3.25.1 -> 3.28.0`,
  `pyiron_workflow 0.13.3 -> 0.15.6`, `pymatgen 2023.10.11 -> 2026.5.4`.
- `requires-python` raised from `>=3.8` to `>=3.10, <3.13` to match
  atomistics' supported range.

### Out of scope

- VASP MD ensembles (NVT/NPT/NHC). The Protocol allows `CalcInputMD`,
  but VASP MD has enough complexity (NHC chains, AIMD timestep,
  thermostat damping) to warrant its own design pass.
- Hybrid functionals (HSE06, B3LYP), GW, BSE.
- POTCAR redistribution. Users must still supply licensed POTCARs.
- Org migration `ligerzero-ai/` -> `pyiron/` - orthogonal.

## [0.0.x] - pre-2026-05-12

See git history for the standalone helper-functions API.
