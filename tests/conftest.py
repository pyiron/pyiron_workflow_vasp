import pathlib

import pytest

MINIMAL_OUTCAR = """\
 vasp.6.4.3 complex
 POTCAR:    PAW_PBE Fe 06Sep2000
 energy  without entropy=      -8.21000000  energy(sigma->0) =       -8.21000000
 reached required accuracy - stopping structural energy minimisation
"""

MINIMAL_POSCAR = """\
Fe
1.0
2.83 0.00 0.00
0.00 2.83 0.00
0.00 0.00 2.83
Fe
1
Direct
0.0 0.0 0.0
"""


@pytest.fixture
def abs_workdir(tmp_path: pathlib.Path) -> str:
    d = tmp_path / "calc"
    d.mkdir()
    return str(d.resolve())


@pytest.fixture
def fake_vasp_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "vaspcalc"
    d.mkdir()
    (d / "OUTCAR").write_text(MINIMAL_OUTCAR)
    (d / "POSCAR").write_text(MINIMAL_POSCAR)
    (d / "vasp.log").write_text(
        "reached required accuracy - stopping structural energy minimisation\n"
    )
    (d / "CHGCAR").write_text("bulk charge density placeholder\n")
    (d / "WAVECAR").write_text("wavefunction placeholder\n")
    return d
