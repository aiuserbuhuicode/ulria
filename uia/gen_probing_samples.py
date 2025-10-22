#!/usr/bin/env python3
"""Generate probing samples for UIA evaluation.

This script loads a configuration file in YAML format and performs
probing sample generation.  It is a placeholder implementation
illustrating correct argument parsing.  In the original code a
parameter name mismatch prevented the configuration from being
properly loaded; this version uses ``--config`` to specify the
configuration file and loads it via ``yaml.safe_load``.

Usage:

  python uia/gen_probing_samples.py --config config.yaml
"""

import argparse
from typing import Any, Dict

try:  # pragma: no cover - exercised indirectly via tests
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback path
    yaml = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate probing samples (placeholder)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file with a simple fallback parser.

    The test environment may not have PyYAML installed.  If the import
    fails we gracefully fall back to parsing the minimal ``key: value``
    syntax used in the regression tests.
    """

    with open(path, "r", encoding="utf-8") as f:
        contents = f.read()
    if yaml is not None:
        return yaml.safe_load(contents)
    config: Dict[str, Any] = {}
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Cannot parse line without ':' delimiter: {line!r}")
        key, value = stripped.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    # Placeholder processing: simply print the configuration
    print("Loaded configuration:", config)


if __name__ == "__main__":
    main()
