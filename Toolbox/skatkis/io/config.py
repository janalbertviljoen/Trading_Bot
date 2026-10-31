"""
skatkis/io/config.py
--------------------
Loads TOML configuration files (``config/*.toml``).

Owns
----
Reading a config file into a dictionary.

Must not
--------
Hold defaults or thresholds (they live in the TOML files); read the environment.

Public API
----------
load_config(path)
    Return the parsed TOML file.

Dependency rule
---------------
Standard library only.
"""

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Return the TOML file at ``path`` as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    with path.open("rb") as handle:
        return tomllib.load(handle)
