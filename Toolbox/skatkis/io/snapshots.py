"""
skatkis/io/snapshots.py
-----------------------
Immutable raw snapshots of provider payloads, with a manifest.

A payload is fetched once per source, query and as-of date and stored gzipped in the
vendor's own format under ``<root>/<source>/<as_of>/``. Every later read uses the
stored copy, so cleaning can be rebuilt without calling the API again. Each folder has
a ``manifest.json`` recording, per query, the file, its size and the SHA-256 of the
uncompressed content, which is checked on every read.

Owns
----
The snapshot layout, the manifest and its integrity check.

Must not
--------
Call an API directly (the provider does); parse payloads (the tool's ``sources/``);
overwrite a stored snapshot.

Public API
----------
SnapshotStore(root)
    Where snapshots live, normally ``data/raw``.
fetch_snapshot(provider, query, as_of, store)
    Return the stored payload, fetching and storing it first if absent.
read_snapshot(store, source, query, as_of)
    Return a stored payload; never fetches.

Dependency rule
---------------
Standard library and ``skatkis.io.provider``.
"""

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

from Toolbox.skatkis.io.provider import DataProvider

MANIFEST_NAME = "manifest.json"


class SnapshotError(RuntimeError):
    """A snapshot is missing from its manifest or fails its integrity check."""


@dataclass(frozen=True)
class SnapshotStore:
    """Root folder of the raw snapshots (``data/raw``), passed in explicitly."""

    root: Path

    def folder(self, source: str, as_of: date) -> Path:
        """Return ``<root>/<source>/<as_of>``."""
        return self.root / source / as_of.isoformat()


def fetch_snapshot(
    provider: DataProvider, query: str, as_of: date, store: SnapshotStore
) -> bytes:
    """Return the payload for ``query``, fetching and storing it only if not stored yet.

    Parameters
    ----------
    provider : DataProvider
        The source; called only when the snapshot is absent.
    query : str
        The provider's query.
    as_of : date
        The snapshot date.
    store : SnapshotStore
        Where snapshots live.

    Returns
    -------
    bytes
        The uncompressed payload, as received from the provider.

    Notes
    -----
    A stored snapshot is never overwritten; take a new snapshot with a later ``as_of``.
    """
    manifest = _read_manifest(store.folder(provider.name, as_of))
    if query in manifest["entries"]:
        return read_snapshot(store, provider.name, query, as_of)
    payload = provider.fetch(query, as_of)
    folder = store.folder(provider.name, as_of)
    folder.mkdir(parents=True, exist_ok=True)
    file_name = f"{_slug(query)}.{payload.extension}.gz"
    _write_atomic(folder / file_name, gzip.compress(payload.content, mtime=0))
    manifest["entries"][query] = {
        "file": file_name,
        "extension": payload.extension,
        "bytes": len(payload.content),
        "sha256": hashlib.sha256(payload.content).hexdigest(),
    }
    manifest["source"] = provider.name
    manifest["as_of"] = as_of.isoformat()
    _write_atomic(folder / MANIFEST_NAME, _dump(manifest))
    return payload.content


def read_snapshot(store: SnapshotStore, source: str, query: str, as_of: date) -> bytes:
    """Return the stored payload for ``query``; never fetches.

    Raises
    ------
    SnapshotError
        If the query is not in the manifest, or the content's SHA-256 differs from the
        manifest's.
    """
    folder = store.folder(source, as_of)
    entry = _read_manifest(folder)["entries"].get(query)
    if entry is None:
        raise SnapshotError(f"no snapshot of {source} for query {query!r} on {as_of}")
    content = gzip.decompress((folder / str(entry["file"])).read_bytes())
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise SnapshotError(
            f"snapshot of {source} for query {query!r} on {as_of} is corrupt"
        )
    return content


def _read_manifest(folder: Path) -> dict[str, Any]:
    """Return the folder's manifest, or an empty one if it has none."""
    path = folder / MANIFEST_NAME
    if not path.exists():
        return {"entries": {}}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _dump(manifest: dict[str, Any]) -> bytes:
    """Serialise a manifest deterministically (sorted keys)."""
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _slug(query: str) -> str:
    """Return ``query`` as a safe file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", query)


def _write_atomic(path: Path, content: bytes) -> None:
    """Write through a temporary file so a crash never leaves a partial file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
