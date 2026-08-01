import pyiron_workflow as pwf


def test_pyiron_workflow_is_019():
    assert pwf.__version__.startswith("0.19")


def test_package_imports_without_pyiron_workflow_legacy_api():
    import pyiron_workflow_vasp

    assert hasattr(pyiron_workflow_vasp, "vasp_job")


def test_parser_subpackage_star_import_works():
    import pyiron_workflow_vasp.vasp_parser as vp

    for name in vp.__all__:
        assert hasattr(vp, name), f"{name} advertised in __all__ but missing"


EXPECTED_EXPORTS = [
    "VaspEngine",
    "VaspInput",
    "vasp_job",
    "create_working_directory",
    "write_vasp_input_set",
    "parse_vasp_output",
    "check_convergence",
    "generate_modified_incar",
    "generate_vasp_input",
    "construct_sequential_vasp_input",
    "run_shell",
    "delete_files_recursively",
    "compress_directory",
    "remove_directory",
    "is_line_in_file",
    "parse_vasp_directory",
    "Outcar",
]


def test_public_exports_present():
    import pyiron_workflow_vasp as pwv

    missing = [name for name in EXPECTED_EXPORTS if not hasattr(pwv, name)]
    assert missing == []


def test_all_advertised_exports_actually_resolve():
    """Every name in __all__ must be a real, importable attribute -- a
    previous version advertised a nonexistent name (``VaspParser`` in
    ``vasp_parser/__init__.py``, which no module in the package ever
    defined) and star-importing or attribute-accessing it would have
    raised AttributeError."""
    import pyiron_workflow_vasp as pwv

    for name in pwv.__all__:
        assert hasattr(pwv, name), f"{name} advertised in __all__ but missing"


def test_no_legacy_names_remain():
    import pyiron_workflow_vasp as pwv

    for legacy in (
        "get_multiple_input",
        "submit_to_slurm",
        "isLineInFile",
        "remove_dir",
        "shell",
    ):
        assert not hasattr(pwv, legacy), f"{legacy} should have been removed or renamed"
