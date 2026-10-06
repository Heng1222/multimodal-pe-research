import importlib
import pkgutil
from pathlib import Path

from pe_research import __version__


def test_package_is_importable() -> None:
    assert __version__


def test_research_domains_are_importable() -> None:
    import pe_research.data
    import pe_research.experiment
    import pe_research.model

    assert pe_research.data.__doc__
    assert pe_research.model.__doc__
    assert pe_research.experiment.__doc__


def test_dataset_packages_and_modules_are_importable() -> None:
    from pe_research.data import api_traces_malware_detection, au_pemal_2025

    for package in (au_pemal_2025, api_traces_malware_detection):
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)


def test_dataset_workspaces_match_package_names() -> None:
    from pe_research.data.api_traces_malware_detection import load_api_trace_config
    from pe_research.data.api_traces_malware_detection.config import resolve_api_workspace
    from pe_research.data.au_pemal_2025 import load_config
    from pe_research.data.au_pemal_2025.config import resolve_workspace

    root = Path.cwd()
    au_config = load_config(root / "configs/data/au_pemal_pilot_v1.yaml")
    api_config = load_api_trace_config(root / "configs/data/api_traces_pilot_v1.yaml")

    assert resolve_workspace(au_config, root) == root / "data/au_pemal_2025/pilot_v1"
    assert resolve_api_workspace(api_config, root) == (
        root / "data/api_traces_malware_detection/pilot_v1"
    )
