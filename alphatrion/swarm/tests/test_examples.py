import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[3] / "examples"


@pytest.mark.parametrize(
    "example", sorted(p.parent.name for p in EXAMPLES.glob("*/main.py"))
)
def test_example_runs(example):
    # Run each example as a user would, so they keep working as the format
    # changes.
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / example / "main.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "status: completed" in result.stdout
    assert "error:" not in result.stdout
