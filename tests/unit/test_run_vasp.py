"""Unit tests for the run_vasp callable. Uses an artificial 'mock command'
that does nothing (no fixtures yet) - verifies the function imports and
matches the expected signature shape. Full parser coverage against canned
VaspEngine fixtures (cu_static/cu_minimize) lives in
test_engine_conformance.py once those are populated.

TestRunVaspEndToEnd below is a *cheaper* end-to-end check that doesn't need
the cu_static/cu_minimize fixtures: it points run_vasp's mock command at the
already-committed tests/fixtures/vasp_outcar_fe_bcc/OUTCAR fixture and a
throwaway fake POTCAR directory, so it always runs (no skip guard). It
exists because run_vasp had never actually been exercised end-to-end by
this suite -- test_engine_conformance.py/test_numerical_regression.py are
both skipped without real fixtures, and TestRunVaspSignature below only
checks the function's signature shape, never calls it. That gap is how
three real bugs shipped silently: read_potcar_config's returned dict key is
"default_POTCAR_path", not "default_POTCAR_library_path" (KeyError on any
call that supplied potcar_config_file); the no-config-file branch passed
pseudopot_lib_path=None straight into get_default_POTCAR_paths, which
crashes on os.path.join(None, ...) (so the "auto-resolve from
~/.pyiron_vasp_config" default path -- VaspEngine's most common usage --
never worked either); and VaspInput(...) was constructed without its
required `incar` argument at all (TypeError on every call). All three are
fixed in this same change; TestRunVaspEndToEnd guards against a repeat.

Written as ``unittest.TestCase`` subclasses so the pyiron shared CI
(which runs ``unittest discover``) picks them up.
"""

from __future__ import annotations

import inspect
import math
import tempfile
import unittest
from pathlib import Path


class TestRunVaspSignature(unittest.TestCase):
    def test_run_vasp_importable(self) -> None:
        from pyiron_workflow_vasp._run import run_vasp

        self.assertTrue(callable(run_vasp))

    def test_run_vasp_signature(self) -> None:
        """Signature must accept the kwargs VaspEngine.get_calculate_fn
        promises to supply: working_directory, engine_input,
        potcar_config_file, functional, encut, kpoints_density, command,
        mode. Plus the positional `structure` argument that the caller
        passes."""
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
        missing = expected - actual
        self.assertFalse(missing, msg=f"missing parameters: {missing}")


class TestRunVaspEndToEnd(unittest.TestCase):
    """run_vasp against a real (committed) OUTCAR fixture, no real VASP
    binary or populated cu_static/cu_minimize fixtures required."""

    _FIXTURE_OUTCAR = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "vasp_outcar_fe_bcc"
        / "OUTCAR"
    )

    def setUp(self) -> None:
        from ase.build import bulk

        from pyiron_workflow_vasp import vasp as _vasp

        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

        # Fake POTCAR library: <lib>/GGA/Fe/POTCAR (content is irrelevant --
        # write_POTCAR just concatenates file bytes, it never parses them).
        potcar_root = self.tmp_path / "potpaw_64"
        (potcar_root / "GGA" / "Fe").mkdir(parents=True)
        (potcar_root / "GGA" / "Fe" / "POTCAR").write_text("fake potcar\n")

        self.config_file = self.tmp_path / ".pyiron_vasp_config"
        self.config_file.write_text(
            "default_POTCAR_set = potpaw64\n"
            "default_functional = GGA\n"
            f"pyiron_vasp_resources = {self.tmp_path}\n"
            "vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64\n"
        )
        self.structure = bulk("Fe", cubic=True, a=2.83)

        # get_default_POTCAR_paths's own CSV-suffix lookup always falls back
        # to the *global* lazy config (DEFAULT_CONFIG_PATH), independent of
        # any potcar_config_file a caller passes explicitly -- a separate,
        # pre-existing quirk from the one fixed in this change. The suite's
        # autouse tests/conftest.py fixture points DEFAULT_CONFIG_PATH at a
        # nonexistent file for hermeticity everywhere else; override that
        # here so this scenario is internally consistent (same fake config
        # for both the explicit and the global path), and restore/clear the
        # cache afterward so this test can't leak into others.
        self._orig_default_config_path = _vasp.DEFAULT_CONFIG_PATH
        _vasp.DEFAULT_CONFIG_PATH = self.config_file
        _vasp._get_potcar_config.cache_clear()
        self._vasp_module = _vasp

    def tearDown(self) -> None:
        self._vasp_module.DEFAULT_CONFIG_PATH = self._orig_default_config_path
        self._vasp_module._get_potcar_config.cache_clear()
        self._tmpdir.cleanup()

    def _run(self, *, potcar_config_file):
        from pyiron_workflow_atomistics.engine import CalcInputStatic

        from pyiron_workflow_vasp._run import run_vasp

        return run_vasp(
            structure=self.structure,
            working_directory=str(self.tmp_path / "run"),
            engine_input=CalcInputStatic(),
            potcar_config_file=potcar_config_file,
            functional="GGA",
            encut=400.0,
            kpoints_density=0.30,
            command=f"cp {self._FIXTURE_OUTCAR} .",
            mode="static",
        )

    def test_run_vasp_with_explicit_potcar_config_file(self) -> None:
        """Regression guard: potcar_config_file used to raise KeyError
        (wrong dict key) before reaching parsing at all."""
        out = self._run(potcar_config_file=self.config_file)
        self.assertTrue(math.isfinite(out.final_energy))
        self.assertIsNotNone(out.final_structure)

    def test_run_vasp_incar_is_actually_written(self) -> None:
        """Regression guard: VaspInput(...) used to be constructed without
        `incar`, raising TypeError before an INCAR was ever written."""
        self._run(potcar_config_file=self.config_file)
        incar_path = self.tmp_path / "run" / "INCAR"
        self.assertTrue(incar_path.is_file())
        self.assertIn("ENCUT", incar_path.read_text())


if __name__ == "__main__":
    unittest.main()
