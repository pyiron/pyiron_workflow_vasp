# pyiron_workflow_vasp

[![PyPI](https://img.shields.io/pypi/v/pyiron-workflow-vasp.svg)](https://pypi.org/project/pyiron-workflow-vasp/)
[![Python](https://img.shields.io/pypi/pyversions/pyiron-workflow-vasp.svg)](https://pypi.org/project/pyiron-workflow-vasp/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

VASP backend for the [pyiron_workflow](https://github.com/pyiron/pyiron_workflow)
ecosystem. Exposes a `VaspEngine` that satisfies the
[`pyiron_workflow_atomistics`](https://github.com/pyiron/pyiron_workflow_atomistics)
Engine Protocol, so physics workflows written against the Protocol
(vacancy formation, surface energies, EOS, etc.) can swap VASP in for
LAMMPS or ASE without code changes.

## Installation

```bash
pip install pyiron_workflow_vasp

# With dev tools
pip install "pyiron_workflow_vasp[dev]"

# With Jupyter
pip install "pyiron_workflow_vasp[notebook]"
```

You also need a working `vasp_std` (or equivalent) binary on `$PATH` and a
`~/.pyiron_vasp_config` pointing at your POTCAR library — see the
[VASP Configuration](#vasp-configuration) section below.

## Dependencies

- Python `>=3.10, <3.13`
- `pyiron-workflow-atomistics >= 0.0.6` (Engine Protocol)
- `pyiron-workflow >= 0.15.6`
- `vaspparser >= 0.0.6` (POSCAR/INCAR/KPOINTS/POTCAR + vasprun parsing)
- `ase`, `pymatgen`, `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`

Exact pins are tracked in [`.ci_support/environment.yml`](.ci_support/environment.yml).

## Quick start (VaspEngine — recommended)

`VaspEngine` is the Protocol-compliant entry point. Import the
`pyiron_workflow_atomistics.engine` module once, dot-access the
shared primitives (`CalcInputStatic`, `CalcInputMinimize`, `calculate`),
and hand the engine to any physics node that takes an `Engine`:

```python
from ase.build import bulk
from pyiron_workflow_atomistics import engine as pwa_engine
from pyiron_workflow_vasp.engine import VaspEngine

structure = bulk("Fe", cubic=True, a=2.83)

engine = VaspEngine(
    EngineInput=pwa_engine.CalcInputMinimize(),   # or pwa_engine.CalcInputStatic()
    working_directory="./bulk_fe_run",
    functional="GGA",
    encut=400,
    kpoints_density=0.30,
    command="mpirun -n 4 vasp_std",
)

# Direct execution — returns an EngineOutput dataclass
output = pwa_engine.calculate.node_function(structure=structure, engine=engine)

print("converged:    ", output.converged)
print("final energy: ", output.final_energy, "eV")
print("final cell:   ", output.final_structure.get_cell())

# ...or compose it into a pyiron_workflow graph
import pyiron_workflow as pwf
wf = pwf.Workflow("fe_relax")
wf.relax = pwa_engine.calculate(structure=structure, engine=engine)
wf.run()
print(wf.relax.outputs.engine_output.value.final_energy)
```

### Swapping engines

That last block is engine-agnostic — `pwa_engine.calculate` is the
single physics-side entry point for the whole ecosystem. Switching
between backends is a one-line change to the engine constructor; the
workflow doesn't care:

```python
# VASP (this package)
from pyiron_workflow_vasp.engine import VaspEngine
engine = VaspEngine(EngineInput=pwa_engine.CalcInputMinimize(), command="mpirun -n 4 vasp_std")

# LAMMPS (pyiron_workflow_lammps)
from pyiron_workflow_lammps.engine import LammpsEngine
engine = LammpsEngine(EngineInput=pwa_engine.CalcInputMinimize(), potential=...)

# ASE / EMT, MACE, GRACE, ... (built into pyiron_workflow_atomistics)
from ase.calculators.emt import EMT
engine = pwa_engine.ASEEngine(EngineInput=pwa_engine.CalcInputMinimize(), calculator=EMT())

# Same call site for all three:
output = pwa_engine.calculate.node_function(structure=structure, engine=engine)
```

This is the whole point of the Engine Protocol — physics workflows
(vacancy formation, EOS, surface energies, ...) written against
`pwa_engine.calculate` work with VASP, LAMMPS, or any ASE calculator
without modification.

Static and Minimize modes are supported in 0.1.0. MD raises
`NotImplementedError` at construction — see the [release notes](https://github.com/pyiron/pyiron_workflow_vasp/releases/tag/pyiron_workflow_vasp-0.1.0)
for the follow-up plan.

## Legacy script API

The original `VaspInput` / `vasp_job` helpers still ship for users with
existing scripts:

```python
from ase.build import bulk
from pymatgen.io.vasp.inputs import Incar
from pyiron_workflow_vasp.vasp import VaspInput, vasp_job

structure = bulk("Fe", cubic=True, a=2.83)
incar = Incar.from_dict({
    "ENCUT": 400, "ISMEAR": 1, "SIGMA": 0.1, "ISPIN": 2, "MAGMOM": "2*3.0",
})

vasp_input = VaspInput(structure=structure, incar=incar)
job = vasp_job(workdir="./bulk_fe_run", vasp_input=vasp_input)
job.run()

print("converged:", job.outputs.convergence_status.value)
```

New code should prefer the `VaspEngine` API — it composes with the
Protocol-aware physics nodes in `pyiron_workflow_atomistics`.

## More examples

- [`examples/run_bulk_fe.py`](examples/run_bulk_fe.py) — runnable bcc-Fe single-point + EOS scan, no hardcoded cluster paths.
- [`example_notebooks/QuickStart.ipynb`](example_notebooks/QuickStart.ipynb) — original walkthrough (uses MPIE paths).
- [`.pyiron_vasp_config.example`](.pyiron_vasp_config.example) — annotated config template.

## VASP Configuration

The `.pyiron_vasp_config` file is essential for configuring the paths to the VASP pseudopotential files (POTCAR files) used in pyiron_workflow's VASP nodes. The file specifies the locations of different VASP potential sets and the default potential set to be used.

For detailed configuration instructions, see [generate_vasp_pyiron_config.md](generate_vasp_pyiron_config.md), which includes:
- Basic configuration setup
- Configuring custom pseudopotential sets
- Adding new pseudopotential variants to CSV files

## 1. Determine the Locations of Your VASP POTCAR Files

Before creating the `.pyiron_vasp_config` file, ensure that you know the locations of the different POTCAR sets on your system. The common VASP potential directories are `potpaw_64`, `potpaw_54`, and `potpaw_52`, but your setup may vary. The directory structure should look something like this:

```
/home/pyiron_resources_cmmc/vasp/potpaw_64/LDA
/home/pyiron_resources_cmmc/vasp/potpaw_64/GGA
/home/pyiron_resources_cmmc/vasp/potpaw_54/LDA
/home/pyiron_resources_cmmc/vasp/potpaw_54/GGA
/home/pyiron_resources_cmmc/vasp/potpaw_52/LDA
/home/pyiron_resources_cmmc/vasp/potpaw_52/GGA
```

- **LDA** and **GGA** refer to the functional types for the VASP potentials.

## 2. Create the .pyiron_vasp_config File

1. Open a terminal and navigate to your home directory:

    ```bash
    cd ~
    ```

2. Use a text editor (such as `nano` or `vim`) to create and open the `.pyiron_vasp_config` file. For example:

    ```bash
    nano .pyiron_vasp_config
    ```

3. Add the following lines to the file. Make sure to modify the paths according to your setup.

    ```ini
    # Set the default POTCAR set
    # Make sure that the default_POTCAR_set matches one of the suffixes in the vasp_POTCAR_path_*
    default_POTCAR_set = potpaw64
    
    # Path to the root directory containing the VASP pseudopotential files
    pyiron_vasp_resources = /home/pyiron_resources_cmmc/vasp
    
    # Path for different POTPAW versions (adjust these paths according to your setup)
    vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64
    vasp_POTCAR_path_potpaw54 = {pyiron_vasp_resources}/potpaw_54
    vasp_POTCAR_path_potpaw52 = {pyiron_vasp_resources}/potpaw_52
    # Note that pyiron vasp nodes can detect variants of vasp_POTCAR_path_{randomsuffix}
    # So if you want to do something with custom pseudopotentials, you can... 
    # Each of these dirs must have a "GGA" and "LDA" subdirectory structure
    # i.e. 
    # The structure should look like
    # .../vasp/potpaw_64/LDA
    # .../vasp/potpaw_64/GGA
    # .../vasp/potpaw_54/LDA etc.
    ```

4. After you have added the configuration details, save the file:
   - If you're using `nano`, press `Ctrl + O`, then `Enter` to save. Press `Ctrl + X` to exit.
   - If you're using `vim`, press `Esc`, type `:wq`, and press `Enter` to save and exit.

## 3. Verify File Permissions

Ensure that your `.pyiron_vasp_config` file and the POTCAR directories are readable by your user. Run the following command to check the file permissions of the `.pyiron_vasp_config`:

```bash
ls -l ~/.pyiron_vasp_config
```

If necessary, you can modify the permissions to ensure read access:

```bash
chmod 644 ~/.pyiron_vasp_config
```

Also, verify that you have read/copy access to the files inside the VASP resource directories (`potpaw_64`, `potpaw_54`, etc.):

```bash
ls -l /home/pyiron_resources_cmmc/vasp/potpaw_64
```

## 4. Testing Your Configuration

To verify that pyiron is correctly reading the `.pyiron_vasp_config` file, you can either check within your pyiron scripts or write a simple Python script to test the configuration:

```python
import os
from pathlib import Path

# Read the config file
config_file = Path.home().joinpath(".pyiron_vasp_config")
with open(config_file, "r") as f:
    print(f.read())
```

This should print out the contents of your `.pyiron_vasp_config`, and you can check if the paths are correctly generated.

### Additional Notes:
- **Ensure Correct Paths**: Double-check that the paths to your POTCAR directories are correct. Incorrect paths will lead to pyiron not being able to find the necessary POTCAR files for your VASP calculations.

By following these instructions, you'll have a correctly configured `.pyiron_vasp_config` file that points to the appropriate VASP pseudopotential directories.
