"""
skatkis/io/tables.py
--------------------
Parquet tables with a manifest, for every stage from cleaning onwards.

Each table ``<name>.parquet`` has a sidecar ``<name>.manifest.json`` with its row count,
columns, dtypes and a content hash. The hash is computed from the data, not the file
bytes, because Parquet metadata can differ between library versions while the data is
the same.

Owns
----
Writing and reading tables and their manifests; the content hash.

Must not
--------
Clean, validate or reshape data (the tool's stages).

Public API
----------
write_table(df, path)
    Write ``df`` to Parquet with its manifest; return the content hash.
read_table(path)
    Read a table written by ``write_table``.
content_hash(df)
    SHA-256 over column names, dtypes and values.

Dependency rule
---------------
pandas and the standard library; Parquet through ``pyarrow``.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


def content_hash(df: pd.DataFrame) -> str:
    """Return a SHA-256 over the column names, dtypes and values of ``df``.

    The index is ignored; row order matters.
    """
    digest = hashlib.sha256()
    for name, dtype in df.dtypes.items():
        digest.update(f"{name}:{dtype};".encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def write_table(df: pd.DataFrame, path: Path) -> str:
    """Write ``df`` to ``path`` (Parquet, without index) with a manifest beside it.

    Parameters
    ----------
    df : pd.DataFrame
        The table; not modified.
    path : Path
        Target ending in ``.parquet``; parent folders are created.

    Returns
    -------
    str
        The content hash, also recorded in the manifest.
    """
    if path.suffix != ".parquet":
        raise ValueError(f"table path must end in .parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    digest = content_hash(df)
    manifest = {
        "rows": len(df),
        "columns": {str(name): str(dtype) for name, dtype in df.dtypes.items()},
        "content_hash": digest,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return digest


def read_table(path: Path) -> pd.DataFrame:
    """Read a Parquet table written by ``write_table``."""
    return pd.read_parquet(path)
