"""Shared pytest config for the LayerNorm test suite.

- Skips the whole directory unless an A100 is detected (env override available).
- Registers the `bench` marker so perf/bandwidth/launch-count/alignment files
  are deselected from the default `pytest` run and only fire under `-m bench`.
- Seeds torch globally before each test for reproducibility.
- Inserts the kernel-POCs root onto sys.path so `from kernels.layernorm import ...`
  works regardless of the cwd pytest is invoked from.
"""
import os
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../kernel-POCs
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            item.add_marker(skip)
        return

    if os.environ.get("FORGE_ALLOW_NON_A100") == "1":
        return

    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        skip = pytest.mark.skip(
            reason=f"Suite is A100-only (got {name!r}); set FORGE_ALLOW_NON_A100=1 to override"
        )
        for item in items:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "bench: performance / memory / bandwidth tests (deselected by default; run with -m bench)",
    )


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)
    yield
