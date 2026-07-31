import pyiron_workflow as pwf
import flowrep as fr


def test_pyiron_workflow_is_019():
    assert pwf.__version__.startswith("0.19")


def test_package_imports_without_pyiron_workflow_legacy_api():
    import pyiron_workflow_vasp

    assert hasattr(pyiron_workflow_vasp, "vasp_job")


def test_parser_subpackage_star_import_works():
    import pyiron_workflow_vasp.vasp_parser as vp

    for name in vp.__all__:
        assert hasattr(vp, name), f"{name} advertised in __all__ but missing"
