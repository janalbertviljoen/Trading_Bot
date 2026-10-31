"""
market_data/report.py
---------------------
Assembles the market analysis report: the plots of a check run with the report's text.

The text (title, abstract, sector descriptions, section texts, figure lists and captions)
is in ``config/report_market_analysis.toml``, which the secretary writes; this module only
places it. Each section lists its figures by name; the plots of one figure become one
image, or a group of lettered panels in the given number of columns (a 3 x 3 grid of
sectors, two sectors side by side, three time views in rows). The appendix holds the
ticker list, every analysis's full history per sector and the data checks.
``skatkis.report`` writes the LaTeX document.

Owns
----
The mapping from a check run's plots, facts, sector table and ticker list to sections,
figures, panels, captions, references and the appendix.

Must not
--------
Write the report's wording (the text file); produce plots (``market_data.checks``);
render LaTeX (``skatkis.report``).

Public API
----------
Plot
    A plot of a check run: section, name, figure, panel, file, caption fields, appendix.
ReportContext(plots, facts, sectors, tickers)
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

from Toolbox.skatkis.report import (
    Document,
    FigureGroup,
    ReportFigure,
    Section,
    Table,
    write_latex,
)


class Plot(NamedTuple):
    """A plot written by a check run.

    ``topic`` is its section, ``kind`` selects its caption in the text file, ``fields``
    fill the caption's placeholders, ``appendix`` places it in the appendix. Plots with
    the same ``figure`` are joined into one figure, each with ``panel`` as sub-caption.
    """

    topic: str
    name: str
    kind: str
    path: Path
    fields: Mapping[str, str] = {}
    appendix: bool = False
    figure: str = ""
    panel: str = ""


class ReportContext(NamedTuple):
    """What a check run hands to the report.

    ``facts`` per analysis (best and worst sectors and companies), ``sectors`` for the
    introduction table and ``tickers`` (sector, symbol, name) for the ticker appendix.
    """

    plots: Sequence[Plot]
    facts: Mapping[str, Mapping[str, str]]
    sectors: Sequence[Mapping[str, str]]
    tickers: Sequence[Mapping[str, str]]


def build_report(
    context: ReportContext, text: Mapping[str, Any], *, folder: Path, subtitle: str
) -> Path:
    """Write the report's LaTeX into ``folder`` and return the ``report.tex`` path.

    Parameters
    ----------
    context : ReportContext
        The plots, facts, sector table and ticker list of the check run.
    text : Mapping[str, Any]
        The parsed report text file.
    folder : Path
        Where ``report.tex`` is written, next to the plot folders.
    subtitle : str
        The line under the title, e.g. the data date.

    Notes
    -----
    Texts and captions may use the placeholders of the analysis's facts (``{best1}``,
    ``{worst1}``, ``{n}``, ``{years}``, ...) and of the plot's fields; ``[[app:<name>]]``
    refers to an appendix section.
    """
    sections = [_introduction(text, context.sectors)]
    for topic in text["order"]:
        spec = text["sections"][topic]
        facts = context.facts.get(topic, {})
        main = [p for p in context.plots if p.topic == topic and not p.appendix]
        if main:
            sections.append(_section(spec, main, facts, label=f"sec:{topic}"))
    appendix = [_tickers(text["appendix"]["tickers"], context.tickers)]
    for topic in text["appendix_order"]:
        own = [p for p in context.plots if p.topic == f"appendix/{topic}"]
        if own:
            appendix.append(_appendix(topic, own, text["appendix"][topic]))
    document = Document(
        title=str(text["title"]),
        subtitle=subtitle,
        abstract=str(text["abstract"]),
        sections=sections,
        appendix=appendix,
    )
    return write_latex(folder / "report.tex", document)


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Return ``template`` with its placeholders filled from ``values``."""
    return template.format(**values) if values else template


def _figures(
    specs: Sequence[Mapping[str, Any]], plots: Sequence[Plot], facts: Mapping[str, str]
) -> list[ReportFigure | FigureGroup]:
    """Build the figures listed in ``specs`` from the plots with those figure names."""
    figures: list[ReportFigure | FigureGroup] = []
    for spec in specs:
        own = [p for p in plots if p.figure == spec["figure"]]
        if not own:
            continue
        caption = _fill(str(spec["caption"]), {**facts, **own[0].fields})
        if len(own) == 1 and not own[0].panel:
            figures.append(ReportFigure(own[0].path, caption))
        else:
            panels = [ReportFigure(p.path, p.panel) for p in own]
            figures.append(FigureGroup(panels, caption, int(spec.get("columns", 2))))
    return figures


def _section(
    spec: Mapping[str, Any],
    plots: Sequence[Plot],
    facts: Mapping[str, str],
    *,
    label: str,
) -> Section:
    """Return one main section with its figures and subsections."""
    return Section(
        title=_fill(str(spec["title"]), facts),
        text=_fill(str(spec["text"]), facts),
        figures=_figures(spec.get("figures", []), plots, facts),
        subsections=[
            Section(
                title=_fill(str(sub["title"]), facts),
                text=_fill(str(sub.get("text", "")), facts),
                figures=_figures(sub.get("figures", []), plots, facts),
            )
            for sub in spec.get("subsections", [])
        ],
        label=label,
    )


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


def _tickers(spec: Mapping[str, Any], tickers: Sequence[Mapping[str, str]]) -> Section:
    """Return the ticker appendix: per GICS sector, tickers and names in alphabetical order."""
    sectors = sorted({t["sector"] for t in tickers})
    return Section(
        title=str(spec["title"]),
        text=str(spec["text"]),
        label="app:tickers",
        subsections=[
            Section(
                title=sector,
                text="",
                table=Table(
                    header=["Ticker", "Company"],
                    rows=sorted(
                        (
                            [t["symbol"], t["name"]]
                            for t in tickers
                            if t["sector"] == sector
                        ),
                        key=lambda row: row[0],
                    ),
                    columns=r"lp{11cm}",
                ),
            )
            for sector in sectors
        ],
    )


def _appendix(topic: str, plots: Sequence[Plot], spec: Mapping[str, Any]) -> Section:
    """Return an appendix section: per GICS sector its full-history plot, plus extras."""
    per_sector = [p for p in plots if p.kind == "appendix"]
    others = [p for p in plots if p.kind != "appendix"]
    captions: Mapping[str, str] = spec.get("captions", {})
    return Section(
        title=str(spec["title"]),
        text=str(spec["text"]),
        label=f"app:{topic}",
        figures=[
            ReportFigure(p.path, _fill(captions.get(p.kind, p.name), p.fields))
            for p in others
        ],
        subsections=[
            Section(
                title=p.fields.get("sector", p.name),
                text="",
                figures=[ReportFigure(p.path, _fill(str(spec["caption"]), p.fields))],
            )
            for p in sorted(per_sector, key=lambda p: p.fields.get("sector", p.name))
        ],
    )
