# Midas, writing

The rules for writing any file, shared by every hat that writes: code, documentation,
records and instructions alike. They add to [AGENTS.md](AGENTS.md); where they differ,
`AGENTS.md` wins.

## Writing for the reader

Records and documentation are written for a person reading them, not only for agents:
the conclusion first, then the reasoning; plain words; production identifiers exactly
as in the code. The persona stays in conversation; nothing written for others carries
it.

## Formulas

In Markdown, every formula is LaTeX: `$…$` inline, `$$…$$` on its own lines for a
display equation.

- One definition per line, spaces around operators; an inline formula stays on one
  source line.
- A display formula is part of its sentence and carries the sentence's punctuation.
- Each symbol has one meaning in a document and is defined at first use, right after
  the term it names or in a "where" clause.
- Floors, caps and clipping the code applies are stated in the sentence, not written
  into the formula with $\min$ or $\max$.
- A calculation is described once, in the earliest section where everything it uses
  has been described; later sections refer back to it.
- Words inside a formula in `\text{…}`; `\frac` and `\sum`, never ASCII stand-ins.

Code blocks hold code, never formulas. Python docstrings keep plain Unicode notation,
because they are read as plain text.

## Human-readable layout

When proposing folders, file names or document structure, design for a person browsing
the repository: a folder's top level shows what a reader needs first, related files sit
together, names say what a file is, and loose or generated files are not left lying
around.

## Renames and the trail

When a name, module or path is renamed or moved, update every reference to the current
name. Created and renamed in the same session: keep no trail. Long-standing (in other
branches, commits or notebooks): ask the user whether to keep a trail before writing
one.
