r"""
skatkis/report.py
-----------------
Writes a report as a LaTeX document and compiles it to PDF with a LaTeX engine.

A report is a title, an abstract and numbered sections (``\section``, ``\subsection``)
whose text is followed by their figures, each a floating figure with a numbered caption.
The document uses the ``article`` class, so it reads like a short academic paper. The
report knows nothing about markets: the text and figures come from the caller.

Owns
----
The report structure (``Section``, ``ReportFigure``), the LaTeX document and the call to
the LaTeX engine.

Must not
--------
Produce figures or write their text (callers such as ``market_data.report``); look for
an engine itself (the command line passes the one it found).

Public API
----------
ReportFigure(path, caption)
    A saved image and its caption.
FigureGroup(panels, caption, columns)
    Several images in one figure, lettered (a), (b), ..., in columns.
Table(header, rows, columns)
    A table inside a section; long tables break across pages.
Section(title, text, figures, subsections)
    A numbered section.
Document(title, subtitle, abstract, sections, appendix)
    A whole report.
write_latex(path, document)
    Write the ``.tex`` file with a table of contents, links between sections and an
    appendix; figures are referenced relative to it.
compile_pdf(tex, *, engine, cache)
    Run the engine (``tectonic``, ``latexmk`` or ``pdflatex``) and return the PDF path.

Dependency rule
---------------
The standard library.
"""

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from os.path import relpath
from pathlib import Path

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{longtable}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\usepackage[hidelinks,bookmarksnumbered]{hyperref}
"""

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class ReportFigure:
    """A saved figure (PNG or PDF) and its caption."""

    path: Path
    caption: str


@dataclass(frozen=True)
class FigureGroup:
    """Several images in one figure, in ``columns`` columns, each with a sub-caption.

    The panels are lettered (a), (b), ... and the figure has one ``caption``.
    """

    panels: Sequence[ReportFigure]
    caption: str
    columns: int = 2


@dataclass(frozen=True)
class Table:
    """A simple table: column headings, rows of cells and LaTeX column types."""

    header: Sequence[str]
    rows: Sequence[Sequence[str]]
    columns: str


@dataclass(frozen=True)
class Section:
    """A numbered section: heading, text, an optional table, figures and subsections.

    ``label`` lets other text refer to the section with ``[[label]]``.
    """

    title: str
    text: str
    figures: Sequence[ReportFigure | FigureGroup] = ()
    subsections: Sequence["Section"] = field(default_factory=tuple["Section", ...])
    label: str | None = None
    table: Table | None = None


def escape(text: str) -> str:
    """Return ``text`` with LaTeX's special characters escaped.

    ``$...$`` stays maths, and ``[[label]]`` becomes a reference to the labelled section.
    """
    parts = re.split(r"(\$[^$]+\$|\[\[[A-Za-z0-9:_-]+\]\])", text)
    out: list[str] = []
    for part in parts:
        if len(part) > 1 and part.startswith("$") and part.endswith("$"):
            out.append(part)
        elif part.startswith("[[") and part.endswith("]]"):
            out.append(f"\\ref{{{part[2:-2]}}}")
        else:
            out.append("".join(_SPECIAL.get(c, c) for c in part))
    return "".join(out)


@dataclass(frozen=True)
class Document:
    """A whole report: title, subtitle, abstract, main sections and appendix."""

    title: str
    subtitle: str
    abstract: str
    sections: Sequence[Section]
    appendix: Sequence[Section] = ()


def write_latex(path: Path, document: Document) -> Path:
    """Write ``document`` to ``path`` (``.tex``) and return the path.

    Parameters
    ----------
    path : Path
        Target ending in ``.tex``; parent folders are created.
    document : Document
        The title, the line under it (typically the data date), a short abstract, the
        main sections (numbered 1, 1.1, ...) and the appendix (lettered A, A.1, ...).

    Notes
    -----
    The document opens with a table of contents; sections, figures and references are
    links in the PDF. Text is escaped for LaTeX except inline maths between dollar signs
    and ``[[label]]`` references. Figures are referenced by paths relative to the
    ``.tex`` file, so the folder moves as a whole. The document has no date of its own,
    so the same input gives the same file.
    """
    if path.suffix != ".tex":
        raise ValueError(f"report path must end in .tex: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        PREAMBLE,
        f"\\title{{{escape(document.title)}}}",
        f"\\author{{{escape(document.subtitle)}}}",
        "\\date{}",
        "\\begin{document}",
        "\\maketitle",
        "\\begin{abstract}",
        escape(document.abstract),
        "\\end{abstract}",
        "\\tableofcontents",
        "\\newpage",
    ]
    for section in document.sections:
        lines += _section(section, path.parent, level="section")
    if document.appendix:
        lines += ["\\newpage", "\\appendix"]
        for section in document.appendix:
            lines += _section(section, path.parent, level="section")
    lines.append("\\end{document}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def compile_pdf(tex: Path, *, engine: str, cache: Path | None = None) -> Path:
    """Compile ``tex`` with ``engine`` in its own folder and return the PDF's path.

    Parameters
    ----------
    tex : Path
        The ``.tex`` file written by ``write_latex``.
    engine : str
        Path or name of ``tectonic``, ``latexmk`` or ``pdflatex``.
    cache : Path, optional
        Folder for tectonic's downloaded packages (``TECTONIC_CACHE_DIR``), so they are
        kept inside the repository; the engine's default location otherwise.

    Raises
    ------
    subprocess.CalledProcessError
        If the engine fails; its log is in the folder.
    """
    name = Path(engine).name
    if name == "tectonic":
        command = [engine, "--keep-logs", tex.name]
    elif name == "latexmk":
        command = [engine, "-pdf", "-interaction=nonstopmode", tex.name]
    else:
        command = [engine, "-interaction=nonstopmode", tex.name]
    runs = 2 if name == "pdflatex" else 1  # pdflatex needs a second run for numbering
    env = dict(os.environ)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        env["TECTONIC_CACHE_DIR"] = str(cache.resolve())
    for _ in range(runs):
        subprocess.run(
            command, cwd=tex.parent, check=True, capture_output=True, env=env
        )
    return tex.with_suffix(".pdf")


def _section(section: Section, folder: Path, *, level: str) -> list[str]:
    """Return the LaTeX of one section, its figures and its subsections."""
    lines = [f"\\{level}{{{escape(section.title)}}}"]
    if section.label:
        lines.append(f"\\label{{{section.label}}}")
    if section.table is not None:
        lines += _table(section.table)
    if section.text:
        lines += [escape(paragraph) + "\n" for paragraph in section.text.split("\n\n")]
    for item in section.figures:
        lines += _figure(item, folder)
    deeper = "subsection" if level == "section" else "subsubsection"
    for sub in section.subsections:
        lines += _section(sub, folder, level=deeper)
    return lines


def _figure(item: ReportFigure | FigureGroup, folder: Path) -> list[str]:
    """Return the LaTeX of one image, or of a group of lettered panels in columns."""
    if isinstance(item, ReportFigure):
        return [
            "\\begin{figure}[H]",
            "\\centering",
            f"\\includegraphics[width=\\textwidth]{{{relpath(item.path, folder)}}}",
            f"\\caption{{{escape(item.caption)}}}",
            "\\end{figure}",
        ]
    width = 0.98 / item.columns - 0.01
    lines = ["\\begin{figure}[H]", "\\centering"]
    for index, panel in enumerate(item.panels):
        if index and index % item.columns == 0:
            lines.append("\\par\\medskip")
        lines += [
            f"\\begin{{subfigure}}[t]{{{width:.3f}\\textwidth}}",
            "\\centering",
            f"\\includegraphics[width=\\linewidth]{{{relpath(panel.path, folder)}}}",
            f"\\caption{{{escape(panel.caption)}}}",
            "\\end{subfigure}\\hfill",
        ]
    lines += [f"\\caption{{{escape(item.caption)}}}", "\\end{figure}"]
    return lines


def _table(table: Table) -> list[str]:
    """Return the LaTeX of a table with horizontal rules; long tables break across pages."""
    header = " & ".join(escape(h) for h in table.header) + " \\\\"
    lines = [
        "{\\small",
        f"\\begin{{longtable}}{{{table.columns}}}",
        "\\hline",
        header,
        "\\hline",
        "\\endhead",
    ]
    lines += [" & ".join(escape(c) for c in row) + " \\\\" for row in table.rows]
    lines += ["\\hline", "\\end{longtable}}"]
    return lines
