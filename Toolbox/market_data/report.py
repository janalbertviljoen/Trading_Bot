"""
market_data/report.py
---------------------
Assembles the market analysis report: the plots of a check run with the report's text.

The text (title, abstract, sector descriptions, and per section its description and
captions) is in ``config/report_market_analysis.toml``, which the secretary writes; this
module only places it. The report goes from coarse to fine: an introduction with a table
of the GICS sectors, the overview of sectors against the S&P 500, then one section per
analysis with its sector summary, its best and worst sector and their best and worst
companies, referring to the appendix, where every sector's companies are shown.
``skatkis.report`` writes the LaTeX document.

Owns
----
The mapping from a check run's plots, facts and sector table to sections, subsections,
captions, references and the appendix.

Must not
--------
Write the report's wording (the text file); produce plots (``market_data.checks``);
render LaTeX (``skatkis.report``).

Public API
----------
Plot
    A plot of a check run: section, name, caption kind, file, caption fields, appendix.
ReportContext(plots, facts, sectors)
    What a check run hands to the report.
build_report(context, text, *, folder, subtitle)
    Write ``report.tex`` into ``folder`` and return its path.

Dependency rule
---------------
The standard library and ``skatkis.report``.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from Toolbox.skatkis.report import Document, ReportFigure, Section, Table, write_latex


class Plot(NamedTuple):
    """A plot written by a check run.

    ``topic`` is its section, ``kind`` selects its caption in the text file, ``fields``
    fill the caption's placeholders, and ``appendix`` places it in the appendix.
    """

    topic: str
    name: str
    kind: str
    path: Path
    fields: Mapping[str, str] = {}
    appendix: bool = False


class ReportContext(NamedTuple):
    """What a check run hands to the report: plots, facts per analysis, sector table."""

    plots: Sequence[Plot]
    facts: Mapping[str, Mapping[str, str]]
    sectors: Sequence[Mapping[str, str]]


def build_report(
    context: ReportContext, text: Mapping[str, Any], *, folder: Path, subtitle: str
) -> Path:
    """Write the report's LaTeX into ``folder`` and return the ``report.tex`` path.

    Parameters
    ----------
    context : ReportContext
        The plots of the check run; per analysis the ``best`` and ``worst`` sector and
        their scores; per GICS sector its ``sector``, ``companies`` and ``fund``.
    text : Mapping[str, Any]
        The parsed report text file.
    folder : Path
        Where ``report.tex`` is written, next to the plot folders.
    subtitle : str
        The line under the title, e.g. the data date and universe size.

    Notes
    -----
    Captions and section texts may use the placeholders of the plot's fields or the
    analysis's facts (``{best}``, ``{worst}``, ``{best_score}``, ``{sector}``, ...), and
    ``[[app:<analysis>]]`` refers to the analysis's appendix section.
    """
    plots, facts = context.plots, context.facts
    sections = [_introduction(text, context.sectors)]
    for topic in text["order"]:
        own = [p for p in plots if p.topic == topic and not p.appendix]
        if own:
            sections.append(
                _analysis(topic, own, facts.get(topic, {}), text["sections"][topic])
            )
    appendix = [
        _appendix(
            topic,
            [p for p in plots if p.appendix and p.topic == f"appendix/{topic}"],
            text,
        )
        for topic in text["order"]
        if any(p.appendix and p.topic == f"appendix/{topic}" for p in plots)
    ]
    document = Document(
        title=str(text["title"]),
        subtitle=subtitle,
        abstract=str(text["abstract"]),
        sections=sections,
        appendix=appendix,
    )
    return write_latex(folder / "report.tex", document)


def _introduction(
    text: Mapping[str, Any], sectors: Sequence[Mapping[str, str]]
) -> Section:
    """Return the introduction with the table of GICS sectors."""
    intro = text["introduction"]
    descriptions: Mapping[str, str] = intro["sectors"]
    rows = [
        [s["sector"], descriptions.get(s["sector"], ""), s["companies"], s["fund"]]
        for s in sectors
    ]
    return Section(
        title=str(intro["title"]),
        text=str(intro["text"]),
        label="sec:introduction",
        table=Table(
            header=["GICS sector", "What it covers", "Companies", "Fund"],
            rows=rows,
            columns=r"p{3.4cm}p{8.6cm}rl",
        ),
    )


def _caption(spec: Mapping[str, Any], plot: Plot, facts: Mapping[str, str]) -> str:
    """Return the caption of ``plot``: its kind's template with its fields and facts."""
    captions: Mapping[str, str] = spec.get("captions", {})
    template = captions.get(plot.kind, plot.name.replace("_", " "))
    return template.format(**{**facts, **plot.fields})


def _analysis(
    topic: str, own: Sequence[Plot], facts: Mapping[str, str], spec: Mapping[str, Any]
) -> Section:
    """Return one analysis: summary figures, then its subsections in the file's order."""
    used = {
        p.kind
        for sub in spec.get("subsections", [])
        for p in own
        if p.kind in sub["kinds"]
    }
    summary = [
        ReportFigure(p.path, _caption(spec, p, facts))
        for p in own
        if p.kind not in used
    ]
    subsections = [
        Section(
            title=str(sub["title"]).format(**facts),
            text=str(sub.get("text", "")).format(**facts),
            figures=[
                ReportFigure(p.path, _caption(spec, p, facts))
                for kind in sub["kinds"]
                for p in own
                if p.kind == kind
            ],
        )
        for sub in spec.get("subsections", [])
        if any(p.kind in sub["kinds"] for p in own)
    ]
    return Section(
        title=str(spec["title"]),
        text=str(spec["text"]).format(**facts) if facts else str(spec["text"]),
        figures=summary,
        subsections=subsections,
        label=f"sec:{topic}",
    )


def _appendix(topic: str, own: Sequence[Plot], text: Mapping[str, Any]) -> Section:
    """Return the appendix section of an analysis: one subsection per GICS sector."""
    spec = text["sections"][topic]
    appendix = text["appendix"]
    return Section(
        title=f"{spec['title']}: {appendix['title']}",
        text=str(appendix["text"]),
        label=f"app:{topic}",
        subsections=[
            Section(
                title=p.fields.get("sector", p.name),
                text="",
                figures=[
                    ReportFigure(p.path, str(appendix["caption"]).format(**p.fields))
                ],
            )
            for p in sorted(own, key=lambda p: p.fields.get("sector", p.name))
        ],
    )
