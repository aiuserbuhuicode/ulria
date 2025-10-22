"""Regression test for uia/gen_probing_samples.py config loading.

This test verifies that the script correctly reads the YAML
configuration passed via the ``--config`` option and prints it.  It
creates a temporary YAML file, invokes the script, and checks that
the loaded configuration appears in the standard output.
"""

import os
import tempfile
import subprocess


def test_gen_probing_samples_config_loading() -> None:
    # Create a temporary YAML config file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
        tmp.write("test_key: 42\n")
        tmp_path = tmp.name
    try:
        # Run the script and capture output
        result = subprocess.run(
            ["python", "uia/gen_probing_samples.py", "--config", tmp_path],
            capture_output=True,
            text=True,
            check=True,
        )
        # Check that the configuration appears in output
        assert "test_key" in result.stdout
        assert "42" in result.stdout
    finally:
        os.remove(tmp_path)
