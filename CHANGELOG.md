# Changelog

All notable changes to `pyiron_workflow_vasp` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: PEP 440.

## [Unreleased]

Ports `vasp.py`/`generic.py` from `pyiron_workflow`'s pre-0.19
`@Workflow.wrap.as_function_node`/`as_macro_node` API to `flowrep`'s
`@fr.atomic`/`@fr.workflow`, required because 0.19 removed that decorator
API and execution signals (`>>`, `starting_nodes`) outright.

### Changed (breaking)

- **Dependency bumps** (required together -- `pyiron-workflow-atomistics`
  0.2.1 itself pins `pyiron-workflow==0.19.0`):
  - `pyiron-workflow`: `0.15.6` -> `0.19.0`
  - `pyiron-workflow-atomistics`: `0.0.6` -> `0.2.1`
  - `vaspparser`: `0.0.6` -> `0.0.9`
  - `flowrep`: new explicit dependency, `0.6.2`
  - Transitive floors forced by the above (`pip install -e .` is
    unsatisfiable under the previous strict `==`-pins otherwise):
    `pyiron_snippets` `1.2.1` -> `1.4.0`, `matplotlib` `3.10.9` -> `3.11.1`,
    `tqdm` `4.67.3` -> `4.70.0`.
- **Renamed with no aliases** (a repo-wide sweep found zero downstream
  users of the old names, so this ships without a deprecation shim):
  - `generic.shell` -> `generic.run_shell`
  - `generic.isLineInFile` -> `generic.is_line_in_file`
  - `generic.remove_dir` -> `generic.remove_directory`
  - `vasp.create_WorkingDirectory` -> `vasp.create_working_directory`
  - `vasp.write_VaspInputSet` -> `vasp.write_vasp_input_set`
  - `vasp.parse_VaspOutput` -> `vasp.parse_vasp_output`
  - `vasp.generate_VaspInput` -> `vasp.generate_vasp_input`
  - `vasp.construct_sequential_VaspInput_from_vaspoutput_structure` ->
    `vasp.construct_sequential_vasp_input`
- **Removed, no replacement in this package**:
  - `generic.submit_to_slurm` -- imported `pyiron_workflow.PickleStorage`,
    which does not exist in `pyiron_workflow` 0.19. Job submission is no
    longer this package's concern; see `examples/run_bulk_fe.py` for the
    direct-call pattern any submission script should wrap. **One known
    downstream script still imports this and will `ImportError`:**
    `FePotentials/2025_12_30_MaterialsProject_Compatibility/
    submit_vasp_calculations.py` (outside this repo) -- it needs porting
    by its owner; there is no drop-in replacement.
  - `vasp.get_multiple_input` -- `ForEach` broadcasts non-iterated inputs
    automatically in the current atomistics ecosystem, so this had no
    purpose left.
- **`__init__.py`** switched from `from .generic import *` / `from .vasp
  import *` wildcard re-exports to an explicit, verified `__all__`
  allow-list (every name checked to actually resolve at import time).
- **`vasp_job`'s `files_to_be_deleted=None` semantics, clarified and
  restored**: `vasp_job(files_to_be_deleted=None)` (the default -- and the
  value `pyiron_workflow_assyst` passes explicitly on every call) resolves
  to `["CHG", "CHGCAR", "WAVECAR"]`, matching the pre-0.19 behaviour. This
  is intentionally DIFFERENT from the lower-level
  `generic.delete_files_recursively(files_to_be_deleted=None)`, which
  stays "delete nothing" (a defensive guard for direct/standalone callers
  of that node). Pass `files_to_be_deleted=[]` to `vasp_job` explicitly if
  you want to keep every file instead.
- `generic.run_shell` no longer calls `os.chdir` at all (a previous
  try/finally save-restore was not thread-safe under 0.19's default
  concurrent evaluation of sibling DAG-layer nodes); `subprocess.run`'s
  `cwd=` argument already scopes the child process.

### Fixed

- `vasp_parser/__init__.py` advertised `__all__ = ["VaspParser", ...]` --
  `VaspParser` was never defined anywhere in the package (only `Outcar`
  and `parse_vasp_directory` exist). Fixed to match.
- `vasp_parser/outcar.py`'s `Outcar.from_file` called
  `zopen(filename, "r")`, which the resolved `monty` version rejects
  outright (`RuntimeError: Implicit text/binary mode is not allowed`).
  Every parsing `try`/`except` in `vasp_parser/output.py` used a bare
  `except:`, silently swallowing that error (and substituting `np.nan`) on
  every real OUTCAR, with no error and no warning. Fixed the `zopen` call
  and replaced every bare `except:` with an explicit, warning-emitting
  handler (silent only for the two genuinely-expected-absence cases:
  missing `vasprun.xml`, missing `POTCAR`).
- `_run.py` (`VaspEngine`'s execution path) had three bugs that made every
  real invocation fail, none previously covered by a non-skipped test:
  `read_potcar_config`'s returned dict key is `"default_POTCAR_path"`, not
  `"default_POTCAR_library_path"` (`KeyError` whenever `potcar_config_file`
  was supplied); the no-`potcar_config_file` default path passed
  `pseudopot_lib_path=None` straight into `get_default_POTCAR_paths`,
  which crashes (so the common case -- no explicit config file -- never
  worked either); and `VaspInput(...)` was constructed without its
  required `incar` argument at all. Also fixed `EngineOutput.final_structure`
  always evaluating to `None` (looked for a `"final_structure"` key the
  parser's return value has never had; the real key is `"structure"`).
- `README.md`'s only two runnable code paths both raised on the first
  call under the current API (`pwa_engine.calculate.node_function(...)`
  and `vasp_job(...).run()`/`.outputs.*.value`, neither of which exist
  anymore). Rewritten and executed end-to-end against a real (committed)
  OUTCAR fixture to confirm they work.
- `construct_sequential_vasp_input` (the ISIF7 -> ISIF5 -> ISIF2 handoff)
  only accepted the bundled legacy parser's `pandas.DataFrame` shape, but
  `vasp_job`'s DEFAULT parser is the external
  `vaspparser.vasp.output.parse_vasp_output`, which returns a `dict` --
  every real ASSYST relaxation chain died at the ISIF7 -> ISIF5 handoff
  with `AttributeError: 'dict' object has no attribute 'structures'`,
  confirmed against a real 2-atom Fe VASP run. Now accepts both shapes
  (`Atoms(**vasp_output["structure"])` for the dict case; unchanged
  `.iloc[-1]`/`str(...)` DataFrame handling otherwise) and raises a clear
  `TypeError` naming both supported shapes for anything else.

## [0.1.0] - 2026-05-12

### Added

- **`pyiron_workflow_vasp.engine.VaspEngine`** - a `@dataclass`
  satisfying the
  `pyiron_workflow_atomistics.engine.Engine` Protocol contract for
  `CalcInputStatic` and `CalcInputMinimize`. Wraps the existing
  POSCAR/INCAR/KPOINTS/POTCAR helpers + `generic.run_shell` (named
  `generic.shell` at the time of this release; renamed in `[Unreleased]`
  above) + `vaspparser.vasp.output.parse_vasp_output` via the new internal
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
