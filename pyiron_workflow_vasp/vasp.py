from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms

import flowrep as fr

from pyiron_workflow_vasp.generic import (
    compress_directory,
    delete_files_recursively,
    remove_directory,
    run_shell,
)

from pyiron_snippets.logger import logger

# Default location of the user config — overridable via env var to ease testing / CI.
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("PYIRON_VASP_CONFIG", Path.home() / ".pyiron_vasp_config")
)

def read_potcar_config(config_file: Path) -> dict:
    """
    Reads the POTCAR configuration from a file and resolves the paths dynamically based on config content.

    Args:
        config_file (Path): Path to the configuration file.

    Returns:
        dict: A dictionary containing the resolved paths and the default POTCAR path.

    Raises:
        ValueError:
            - If no valid `default_POTCAR_set` is found in the config file.
            - If no valid `default_functional` (e.g., GGA or LDA) is found in the config.
        FileNotFoundError: If the configuration file does not exist.
        Exception: For any other unexpected issues encountered while reading the file.
    """

    config_data = {}

    # Read the configuration file
    try:
        with open(config_file, "r") as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split the line into key and value
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Store the configuration
                    config_data[key] = value

        # Resolve the pyiron_vasp_resources path
        pyiron_vasp_resources = config_data.get("pyiron_vasp_resources", "")
        default_POTCAR_set = config_data.get("default_POTCAR_set")
        default_functional = config_data.get("default_functional")
        default_pseudopotential_set = config_data.get("default_pseudopotential_set")

        # Dynamically identify all POTCAR sets based on keys in the config file
        potcar_sets = []
        for key in config_data:
            if key.startswith("vasp_POTCAR_path_"):
                potcar_sets.append(key.split("vasp_POTCAR_path_")[1])

        # Check if a valid default_POTCAR_set is provided
        if not default_POTCAR_set or default_POTCAR_set not in potcar_sets:
            raise ValueError(
                f"Unknown or missing default_POTCAR_set: {default_POTCAR_set}. Valid options: {potcar_sets}"
            )

        # Check if a valid default_functional is provided
        if not default_functional or default_functional not in ["GGA", "LDA"]:
            raise ValueError(
                f"Unknown or missing default_functional: {default_functional}. Valid options: GGA, LDA"
            )

        # Dynamically generate the paths for all potpaw sets based on the config content
        for potcar_set in potcar_sets:
            key = f"vasp_POTCAR_path_{potcar_set}"
            # Preserve the underscores in the path
            config_data[key] = os.path.join(
                pyiron_vasp_resources, potcar_set.replace("potpaw", "potpaw_")
            )

        # Set the default POTCAR path
        config_data["default_POTCAR_path"] = config_data[
            f"vasp_POTCAR_path_{default_POTCAR_set}"
        ]

        # Set the pseudopotential CSV file suffix (defaults to functional if not specified)
        if default_pseudopotential_set:
            config_data["pseudopotential_csv_suffix"] = default_pseudopotential_set
        else:
            config_data["pseudopotential_csv_suffix"] = default_functional

        return config_data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_file}") from e
    except ValueError as e:
        raise e  # Let specific ValueErrors propagate
    except Exception as e:
        raise Exception(f"Error reading configuration file: {e}") from e


@lru_cache(maxsize=1)
def _get_potcar_config() -> dict:
    """Lazily read and cache the user's POTCAR config from ``DEFAULT_CONFIG_PATH``.

    Reading happens on first call rather than at import, so importing the
    package never fails on a machine that hasn't yet created the config file
    (useful for tests, docs builds, and inspecting function signatures).
    """
    return read_potcar_config(DEFAULT_CONFIG_PATH)


def _default_POTCAR_library_path() -> str:
    return _get_potcar_config()["default_POTCAR_path"]


def _default_functional() -> str:
    return _get_potcar_config()["default_functional"]


def _default_POTCAR_generation_path() -> str:
    cfg = _get_potcar_config()
    return os.path.join(cfg["default_POTCAR_path"], cfg["default_functional"])


def _default_POTCAR_specification_csv() -> str:
    cfg = _get_potcar_config()
    suffix = cfg.get("pseudopotential_csv_suffix", cfg["default_functional"])
    return str(
        Path(__file__).parent.joinpath(
            "vasp_resources", f"vasp_pseudopotential_{suffix}_data.csv"
        )
    )


@dataclass
class VaspInput:
    """
    Class to represent the input settings for a VASP calculation, including structure, INCAR parameters, and pseudopotential paths.

    Attributes:
        structure (ase Atoms): The atomic structure of the system to be used in the VASP calculation.
        incar (pymatgen Incar): The INCAR object containing the VASP input parameters.
        pseudopot_lib_path (str): The path to the library of VASP pseudopotentials (POTCAR files).
        Defaults to `default_POTCAR_generation_path` generated by the .pyiron_vasp_config.
        potcar_paths (Optional[list[str]]): A list of paths to user-specified POTCAR files associated with the calculation.
            When specified, this _OVERRIDES_ default POTCAR generation.
            This list must contain all the necessary POTCAR files for the elements in the structure.
            For example, if the structure contains H O H atoms, the list should contain paths like:
            - ".../vasp/potpaw_64/H/POTCAR"
            - ".../vasp/custom_charged_potential/O/POTCAR"
            - ".../vasp/potpaw_64/H/POTCAR"
            No checks are performed to ensure the POTCAR files match the species in the structure, so it is the user's responsibility
            to ensure consistency between the elements in the structure and the provided POTCAR files when using this functionality.

        kpoints (Optional[Kpoints]): The KPOINTS object defining the k-point mesh for the calculation. If not provided, default settings will be used.

    Notes:
        - The `potcar_paths` attribute is exposed to allow users to make one-off changes to the pseudopotentials. This is
          useful when custom/non-default potentials are required for specific elements.
        - When potcar_paths is used, the class does not validate that the provided POTCAR files match the elements in the structure,
          so incorrect configurations may lead to invalid calculations.
        - ``structure`` may be given as either an ``ase.Atoms`` or a ``pymatgen.core.Structure``; ``__post_init__``
          normalizes a pymatgen ``Structure`` to ``ase.Atoms`` once, at construction, since the file-writing helpers
          (``write_POSCAR``, ``get_default_POTCAR_paths``) only understand ``ase.Atoms``. The normalization never
          mutates the object the caller passed in.
        - **Per-site magnetic moments are NOT propagated to the INCAR.** If ``structure`` carries
          ``site_properties={"magmom": ...}`` (pymatgen) or per-atom initial magnetic moments (ase), those are
          silently dropped -- neither the ase nor the pymatgen write path ever wrote a MAGMOM tag from the
          structure. Set MAGMOM explicitly as an INCAR string (e.g. ``"3*2.5 1*-2.5"``) if the calculation needs it;
          this is the contract downstream ASSYST campaigns already rely on.
    """

    structure: Atoms
    incar: Incar
    pseudopot_lib_path: Optional[str] = field(default=None)
    pseudopot_functional: str = "GGA"
    potcar_paths: Optional[list[str]] = None
    kpoints: Optional[Kpoints] = None

    def __post_init__(self) -> None:
        # Normalize once, at construction, so every producer of a VaspInput
        # (generate_vasp_input, construct_sequential_vasp_input, or direct
        # construction by a caller/test) ends up with an ase.Atoms structure
        # regardless of what representation it was handed. This never
        # mutates the caller's original object -- AseAtomsAdaptor.get_atoms
        # returns a new object; self.structure is simply rebound. Doing this
        # here (rather than at each call site) also matters because
        # pyiron_workflow 0.19 evaluates sibling nodes concurrently, so a
        # call-site mutation of a shared structure object would race.
        if isinstance(self.structure, Structure):
            self.structure = AseAtomsAdaptor.get_atoms(self.structure)

        # Resolve the POTCAR library path lazily so importing the package
        # doesn't require a config file. Only resolve when the user hasn't
        # supplied explicit ``potcar_paths``.
        if self.pseudopot_lib_path is None and self.potcar_paths is None:
            self.pseudopot_lib_path = _default_POTCAR_library_path()
    # generic_input: GenericDFTInput
#    spin_constraints: []

# def GenericDFTInput:
#     spin_constraints=[]
#     kpoints=[]
#     plane_wave_cutoff=[]

def write_POSCAR(workdir: str, structure: Atoms, filename: str = "POSCAR") -> str:
    poscar_path = os.path.join(workdir, filename)
    # structure.to(fmt="poscar", filename=poscar_path)
    structure.write(poscar_path, format="vasp")
    return poscar_path


def write_INCAR(workdir: str, incar: Incar, filename: str = "INCAR") -> str:
    incar_path = os.path.join(workdir, filename)
    incar.write_file(incar_path)
    return incar_path


def write_POTCAR(workdir: str, vasp_input: VaspInput, filename: str = "POTCAR") -> str:
    class PotcarNotGeneratedError(Exception):
        pass

    if vasp_input.potcar_paths is None:
        potcar_paths = get_default_POTCAR_paths(
            vasp_input.structure, pseudopot_lib_path=vasp_input.pseudopot_lib_path, pseudopot_functional=vasp_input.pseudopot_functional
        )
    else:
        potcar_paths = vasp_input.potcar_paths

    potcar_path = os.path.join(workdir, filename)

    with open(potcar_path, "wb") as wfd:
        for f in potcar_paths:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)

    return potcar_path


def write_KPOINTS(
    workdir: str, kpoints: Optional[Kpoints] = None, filename: str = "KPOINTS"
) -> str:
    kpoint_path = os.path.join(workdir, filename)
    if kpoints is not None:
        kpoints.write_file(kpoint_path)
    else:
        with open(kpoint_path, "w") as f:
            f.write(
                "Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n"
            )  # Example KPOINTS content
    return kpoint_path


@fr.atomic("workdir")
def create_working_directory(workdir: str, quiet: bool = False) -> str:
    """
    Create a working directory if it doesn't exist.

    Args:
        workdir (str): Path to the directory to create.
        quiet (bool, optional): If True, suppress the "already exists"
            warning when ``workdir`` is pre-existing. Defaults to False.

    Returns:
        str: Path to the created or existing directory.
    """
    # Check if workdir exists
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        logger.info(f"made directory '{workdir}'")
    elif not quiet:
        warnings.warn(
            f"Directory '{workdir}' already exists. Existing files may be overwritten."
        )
    return workdir


@fr.atomic("workdir")
def write_vasp_input_set(workdir: str, vasp_input: VaspInput) -> str:
    """
    Write VASP input files to the working directory.

    Args:
        workdir (str): Path to the working directory.
        vasp_input (VaspInput): VASP input object containing structure, INCAR, and POTCAR information.

    Returns:
        str: Path to the working directory.
    """
    # vasp_input.structure is already an ase.Atoms here: VaspInput.__post_init__
    # normalizes a pymatgen Structure once, at construction, for every producer.
    _ = write_POSCAR(workdir=workdir, structure=vasp_input.structure)
    _ = write_INCAR(workdir=workdir, incar=vasp_input.incar)
    _ = write_POTCAR(workdir=workdir, vasp_input=vasp_input)
    if vasp_input.kpoints is not None:
        _ = write_KPOINTS(workdir=workdir, kpoints=vasp_input.kpoints)

    return workdir


@fr.atomic("output_dict")
def parse_vasp_output(
    workdir: str,
    function=None,
    parser_args: Optional[dict] = None,
    after: object = None,
):
    """
    Parse VASP output files in the working directory.

    Args:
        workdir (str): Path to the working directory containing VASP output files.
        function (callable, optional): Custom parser to use. If ``None``, the
            default ``vaspparser.vasp.output.parse_vasp_output`` is used and
            ``parser_args`` is overridden with ``{"working_directory": workdir}``.
        parser_args (dict, optional): Keyword arguments to pass to ``function``.
            Ignored when ``function`` is ``None``.
        after: An ordering token; see :func:`pyiron_workflow_vasp.generic.delete_files_recursively`.
            Never read -- accepting it lets a caller (``vasp_job``) create a
            data edge that forces parsing to run after the shell command has
            finished, since pyiron_workflow 0.19 has no execution signals.

    Returns:
        Whatever ``function`` returns. The default parser
        (``vaspparser.vasp.output.parse_vasp_output``) returns a hierarchical
        ``dict``, not the ``"output_dict"`` output-port name's literal type --
        that name predates this port and is kept for compatibility with
        existing callers/tests.
    """
    if function is None:
        from vaspparser.vasp.output import parse_vasp_output as parse_vasp_directory
        parser_args = {"working_directory": workdir}
    else:
        parse_vasp_directory = function
        if parser_args is None:
            parser_args = {}
    return parse_vasp_directory(**parser_args)


@fr.atomic("convergence")
def check_convergence(
    workdir: str,
    filename_vasprun: str = "vasprun.xml",
    filename_vasplog: str = "vasp.log",
    backup_vasplog: str = "error.out",
    after: object = None,
) -> bool:
    """
    Check if a VASP calculation has converged.

    Args:
        workdir (str): Path to the working directory.
        filename_vasprun (str, optional): Name of the vasprun.xml file. Defaults to "vasprun.xml".
        filename_vasplog (str, optional): Name of the VASP log file. Defaults to "vasp.log".
        backup_vasplog (str, optional): Name of the backup log file. Defaults to "error.out".
        after: An ordering token; see :func:`pyiron_workflow_vasp.generic.delete_files_recursively`.

    Returns:
        bool: True if the calculation has converged, False otherwise.
    """
    line_converged = (
        "reached required accuracy - stopping structural energy minimisation"
    )

    # Try vasprun.xml first (authoritative), fall back to scanning the run log
    # and finally the backup log captured by the queueing system.
    try:
        vr = Vasprun(filename=os.path.join(workdir, filename_vasprun))
        return bool(vr.converged)
    except Exception:
        pass

    for logname in (filename_vasplog, backup_vasplog):
        path = os.path.join(workdir, logname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as handle:
                if any(line_converged in file_line for file_line in handle):
                    return True
        except Exception:
            continue

    return False


@fr.workflow("vasp_output", "convergence_status")
def vasp_job(
    workdir: str,
    vasp_input: VaspInput,
    command: str = "module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 1 vasp_std",
    files_to_be_deleted: Optional[list[str]] = None,
    compress: bool = False,
    compressed_file_in_dir: bool = False,
    remove_calc_dir: bool = False,
    vasp_parser_function=None,
    vasp_parser_args: Optional[dict] = None,
):
    """
    Run a VASP calculation with the specified input parameters.

    pyiron_workflow 0.19 removed execution signals (``>>``/``starting_nodes``),
    so the ordering that a pre-0.19 macro would express with ``>>`` is rebuilt
    here as data dependencies:

        create_working_directory -> write_vasp_input_set -> run_shell
        parse_vasp_output(after=shell_result), check_convergence(after=shell_result)
        delete_files_recursively(after=raw_output) -> compress_directory(after=convergence)
        -> remove_directory(payload=raw_output)

    The ``after=`` arguments are ordering tokens that are never read by the
    receiving node; they exist only to create the data edge. This guarantees:
    inputs are written before VASP runs; output is parsed before anything is
    deleted; convergence is determined before the directory is compressed or
    removed.

    Args:
        workdir (str): Path to the working directory.
        vasp_input (VaspInput): VASP input object containing structure, INCAR, and POTCAR information.
        command (str, optional): Command to run VASP. Defaults to a specific module loading command.
        files_to_be_deleted (list[str], optional): List of files to delete after the calculation.
            Defaults to ``None`` (nothing is deleted).
        compress (bool, optional): Whether to compress the output directory. Defaults to False.
        compressed_file_in_dir (bool, optional): Whether to place the compressed file in the output directory.
                                                Defaults to False.
        remove_calc_dir (bool, optional): Whether to remove the calculation directory after compression.
                                         Defaults to False.
        vasp_parser_function (callable, optional): Function to parse VASP output. This should be a regular Python function
                                                 that takes a workdir parameter and returns parsed output. If None, the
                                                 default parse_vasp_output function will be used.
        vasp_parser_args (dict, optional): Additional arguments to pass to the vasp_parser_function. Defaults to {}.

    Returns:
        tuple: (vasp_output, convergence_status)
    """
    wd_created = create_working_directory(workdir)
    wd_written = write_vasp_input_set(wd_created, vasp_input)
    shell_result = run_shell(command, wd_written)
    raw_output = parse_vasp_output(
        wd_written, vasp_parser_function, vasp_parser_args, after=shell_result
    )
    convergence = check_convergence(wd_written, after=shell_result)
    # NOTE: there is no explicit data edge from `convergence` into
    # delete_files_recursively below (it is ordered after `raw_output` /
    # parse_vasp_output instead, not after check_convergence). That is safe
    # ONLY because 0.19's DAG-layer barrier places delete_files_recursively
    # in a layer after check_convergence has already run, so convergence is
    # always read before any deletion happens -- this is a dependency on
    # executor/layering semantics, not on a data edge. If a future edit
    # changes the execution model (e.g. separate executors without a shared
    # layer barrier) this ordering guarantee disappears silently; add an
    # explicit `after=convergence` edge if that assumption is ever relaxed.
    wd_cleaned = delete_files_recursively(
        wd_written, files_to_be_deleted, after=raw_output
    )
    wd_archived = compress_directory(
        wd_cleaned, compress, compressed_file_in_dir, after=convergence
    )
    vasp_output = remove_directory(wd_archived, remove_calc_dir, raw_output)
    return vasp_output, convergence


def stack_element_string(structure) -> tuple[list[str], list[int]]:
    #site_element_list = [site.species_string for site in structure]
    site_element_list = [atom.symbol for atom in structure]
    past_element = site_element_list[0]
    element_list = [past_element]
    element_count = []
    count = 0

    for element in site_element_list:
        if element == past_element:
            count += 1
        else:
            element_count.append(count)
            element_list.append(element)
            count = 1
            past_element = element

    element_count.append(count)
    return element_list, element_count


def get_default_POTCAR_paths(
    structure: Atoms,
    pseudopot_lib_path: str,
    pseudopot_functional: str = "GGA",
    potcar_df: Optional[pd.DataFrame] = None,
) -> list[str]:
    if potcar_df is None:
        potcar_df = pd.read_csv(_default_POTCAR_specification_csv())
    ele_list, _ = stack_element_string(structure)
    potcar_paths = []
    for element in ele_list:
        ele_default_potcar_path = potcar_df[
            (potcar_df["symbol"] == element) & (potcar_df["default"] == True)
        ].potential_name.values[0]
        potcar_paths.append(
            os.path.join(pseudopot_lib_path, pseudopot_functional, ele_default_potcar_path, "POTCAR")
        )

    return potcar_paths


#%% These are utilities
@fr.atomic("vasp_input")
def generate_vasp_input(structure, incar: Incar, potcar_paths=None) -> VaspInput:
    return VaspInput(structure=structure, incar=incar, potcar_paths=potcar_paths)


@fr.atomic("incar")
def generate_modified_incar(incar: Incar, modifications: dict) -> Incar:
    """
    Generates a modified INCAR dictionary by updating specific keys.

    Parameters:
    - incar (dict): Original INCAR dictionary to modify.
    - modifications (dict): Dictionary of keys and their corresponding new values to update.

    Returns:
    - dict: A modified INCAR dictionary with the specified changes.

    Example:
    --------
    Original INCAR:
        incar = {
            "ENCUT": 520,
            "EDIFF": 1e-5,
            "ISMEAR": 0,
            "SIGMA": 0.1
        }

    Modifications:
        modifications = {
            "ISIF": 2,
            "LREAL": "Auto",
            "NSW": 100
        }

    Call:
        modified_incar = generate_single_modified_incar(incar, modifications)

    Result:
        modified_incar = {
            "ENCUT": 520,
            "EDIFF": 1e-05,
            "ISMEAR": 0,
            "SIGMA": 0.1,
            "ISIF": 2,
            "LREAL": "Auto",
            "NSW": 100
        }
    """
    if not isinstance(modifications, dict):
        raise ValueError("Modifications must be provided as a dictionary.")

    # Create a copy of the original INCAR and apply modifications -- copying
    # (rather than mutating in place) matters because pyiron_workflow 0.19
    # evaluates sibling nodes concurrently, and the same Incar object may be
    # fanned out to several of them.
    modified_incar = incar.copy()
    for key, value in modifications.items():
        modified_incar[key] = value

    return modified_incar


@fr.atomic("vasp_input")
def construct_sequential_vasp_input(
    vasp_output: pd.DataFrame, incar: Incar, potcar_paths=None
) -> VaspInput:
    """Takes the most recent calculation in the directory and its final
    ionic step, and builds the next VaspInput from it.

    ``vasp_output`` is the DataFrame produced by the bundled legacy parser
    (see ``vasp_parser/output.py``, ``parse_vasp_directory``): one row per
    OUTCAR* found plus one row per error archive, sorted ASCENDING by
    ``calc_start_time``. Its ``structures`` column holds, per row, an array
    of ``Structure.to_json()`` strings, one per ionic step -- NOT structure
    objects.

    ``.iloc[-1]`` selects the LAST row, i.e. the most recent calculation,
    not ``.iloc[0]`` (the oldest/earliest). This matters whenever a
    directory holds more than one row -- e.g. a custodian restart or an
    error archive -- such as ASSYST's ISIF7 -> ISIF5 -> ISIF2 chains run
    with max_errors>0: picking the oldest row would feed the
    crashed/earliest attempt's structure into the next stage instead of
    continuing from where the calculation actually left off.
    ``vasp_output.structures.iloc[-1][-1]`` is therefore the JSON string for
    the last ionic step of the most recent row, and must be round-tripped
    back through ``Structure.from_str(..., fmt="json")`` before it is
    usable. ``VaspInput.__post_init__`` then normalizes that pymatgen
    Structure to ase.Atoms, same as every other VaspInput producer.
    """
    # str(...): parse_vasp_directory stores the per-step JSON strings in a
    # numpy array, so indexing it back out yields numpy.str_ rather than a
    # plain str. pymatgen's JSON reader (orjson) rejects numpy.str_ even
    # though it is a str subclass, so it must be coerced explicitly.
    structure = Structure.from_str(
        str(vasp_output.structures.iloc[-1][-1]), fmt="json"
    )
    return VaspInput(structure=structure, incar=incar, potcar_paths=potcar_paths)
