"""
skatkis/io/figures.py
---------------------
Saves matplotlib figures as image files.

Owns
----
Writing figures to disk.

Must not
--------
Build plots (``skatkis.analysis``); choose where results go (the caller).

Public API
----------
save_figure(figure, path, *, dpi)
    Write a figure as PNG, creating parent folders.

Dependency rule
---------------
matplotlib and the standard library.
"""

# matplotlib annotates savefig with untyped **kwargs, which strict pyright reports;
# the arguments passed here are typed.
# pyright: reportUnknownMemberType=false

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(figure: Figure, path: Path, *, dpi: int = 110) -> Path:
    """Write ``figure`` to ``path`` (PNG) and return the path; parent folders are created."""
    if path.suffix != ".png":
        raise ValueError(f"figure path must end in .png: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
