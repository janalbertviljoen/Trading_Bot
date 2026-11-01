# VI_202611_002: Report: general worst list, sells tracked in the wallet's cash flows

| Field | Value |
|---|---|
| Status | in progress |
| Priority | medium |
| Kind | new capability (the report's content; no change to the scores) |
| Depends on | VI_202610_006 |
| Branch | `feature/skatkis/s-202611-003-calibration-infrastructure` (shared with VI_202611_001) |
| Raised | 2026-11-01 |

## Description

The sell check of the value report lists what the reference investor holds and the share
the rule would sell (`holding.py`, `report.py`). That depends on a portfolio, so it is of
no use to a new user (user, 2026-11-01). Sells should appear only in the cash flows of the
wallet with sells, and the report should instead give a general list of the worst-ranked
stocks of the universe, independent of any holdings.

## Agreed approach

- The wallet with sells keeps its trades (`buy_and_sell`); its sells are shown as cash
  flows of that wallet only.
- The worst stocks are a general list of the 10 lowest-scoring companies on the latest date
  (the same count as the picks, `[wallet] names`), by `score_binned_weighted` and its 0-to-1
  share (`score_weighted_share`), for everyone.
- The list is coloured by the share (user, 2026-11-01: Green, Amber, Red, instead of the
  words "avoid" and "start thinking about selling"), with the edges taken from the cut-offs
  in use (today `sell_threshold = 0.4`; the fitted cut-offs of VI_202611_001 later). Proposed
  mapping, to confirm with the user: **Green** above the sell threshold (0.4); **Amber** from
  half of it (0.2) up to the threshold; **Red** below half of it, down to 0, where the stock
  is not worth anything any more. The legend is explained in the report's notes, and the list
  is "not a recommendation". The numbers are not fixed; they follow the cut-offs.
- Later, if the optimistic reading holds out-of-sample, a short position on this basis.
- The per-holding sell check is removed from the reader-facing text.
- A section on winning picks we missed, as opportunities (user, 2026-11-01): the companies
  with the highest realised excess return that the score did not pick, their score percentile
  and what they scored low on, by index (VI_202611_001, check 24). Hindsight, shown as a
  diagnostic of the rules and never as a recommendation.

## Further sections (user, 2026-11-01)

- **Approach and accreditation.** The report states how the model was tested and what each
  test found: the protocol steps in plain words and the results of the accreditation
  register (`midas/accreditation.md`): heavy hitters removed, company sampling bands, unseen
  test companies, independent indices, two periods, missed winners, with passes and fails
  stated.
- **Midas and the Investor: a Q&A.** The back and forth of the protocol (`quant.md`, step 15)
  appears as a dialogue between Midas (the quant: what the data showed) and the Investor (what
  the letters say, with the year), one exchange per fishy result, ending with what was tested
  and what was decided or parked. The section advises the reader how to improve their results.
- **Reading by risk profile.** Strategy literature the reader can download or bring, so that
  Midas can work from it (`assay/notes.md`, "Strategy literature by risk profile"). The
  letters and Graham's book are recommended first.

## Open decisions (user)

- Whether the tiers also carry sector and size information, as the picks do.

## Decisions

- 2026-11-01 (user): sells are tracked only in the wallet's cash flows; the worst stocks
  are a general list.
- 2026-11-01 (user): the report carries the approach and the accreditation results, and a
  Q&A between Midas and the Investor (above). This is the one place the two voices appear in
  something written for others (`writing.md`).
- 2026-11-01 (user): the list has as many companies as the picks (5 then, 10 from the rerun on the ten default rules); tiers as above; a
  short position is a later step. Superseded the same day: 10 companies, the picks' count
  (the report already has 10), and colours in place of the tier wording (below).
- 2026-11-01 (user, after the business-model review): the tiers become Green, Amber and
  Red, with the legend explained in the notes; 10 picks for now.

- 2026-11-01 (user, confirmed in the session that built the report): the tiers are Green, Amber
  and Red because it reads better, is intuitive and is the colour code of trading candles; the
  earlier wording tiers (0, 0.2, 0.4) are superseded. The change was made in the code by the other
  session after its business-model review (`_tier`, `_worst_table`, `fills` in `skatkis.report`) and
  is part of the first-light state. Reports share the weights and techniques but not the exact bin
  widths or the number of samplings (`scribe.md`, "Reports"): the text of the testing and the Q&A
  sections is to be cleaned of them.

## Notes

- 2026-11-01: the worst list is built (unstaged code: `report.py`, `__main__.py`,
  `config/report_value_investing.toml`, the scribe's wording to polish): the lowest scores of the
  whole index on the latest score date, 10 companies (the picks' count), sector, score, the
  0-to-1 score and the tier; `worst_stocks.csv` is written beside `next_picks.csv`; the
  holdings-based sell check is gone from the reader-facing text. Still to do: the sections on the
  approach and accreditation, the Q&A, the reading list and the missed-winners section; whether
  to show size beside sector; the report's other indices. For the scribe (inspector, 2026-11-01):
  `config/report_value_investing.toml` still says "five" (lines 50, 79, 204, 364; use the `{picks}`
  fact or reword) and "Sixteen rules" (line 41, ten are active), and the rule names for G3 ("10 years
  of profit") and G4 ("20 years of dividends") disagree with the 3-year windows.

- 2026-11-01: the sections on the approach and accreditation ("How the model was tested"), the
  Q&A between Midas and the Investor ("Midas and the Investor"), the missed winners ("Winning
  picks we missed") and the reading list ("Where to read more") are built in the report (unstaged
  code: `report.py`, `skatkis/report.py` for `**bold**`; first-draft text in
  `config/report_value_investing.toml`, for the Scribe to finish). Their numbers are static text
  from runs of 2026-11-01 on the ten default rules (accreditation tests, missed winners); the two
  tests not yet rerun on ten rules are marked "to rerun" in the table. Until the study
  machinery of S_202611_003 computes them, they are text and not computed by the command, and
  must be refreshed by hand when the model changes. The reading list's titles are unchecked.

- 2026-11-01 (user, after reading the report): "Selling below the threshold" has two
  subsections, "Sold in the wallet" (the wallet's sales, its cash flows) and "To sell or short"
  (the lowest scores of the whole index, as many as the picks, 10, with the tiers), and the worst
  list left "The next picks". The text says the wallet never sells short and a short position is a
  later step, untested. Figure 6 (the price history of the companies traded) shows 12 panels,
  `pick_ends = 6`: the 6 best on the first row and the 6 worst on the second. The S&P 400 and 600
  reports were still the old ones of 2026-10-31 10:00; regenerated with the ten default rules
  (results and sections as the S&P 500's; the testing and missed-winners text is the same in all
  three reports and describes all three indices).

- 2026-11-01 (business-model review): the Scribe has written the report text
  (`config/report_value_investing.toml`, unstaged): the section "The lowest scores" defines
  Green (above the threshold), Amber (the tier edge up to the threshold) and Red (below the
  edge), says "not a recommendation to buy, sell or sell short", carries the no-endorsement
  paragraph, qualifies "out-of-sample", and gives the ten-rule figures (2020 start 27.7% a year
  against 15.9%; MU 24% and LRCX 16% of the 2016 investor's gain). The investor profile
  (`docs/investor/starting_investor_profile.md`, unstaged) is aligned: a worked example, the
  chat sticks to Graham. The business proposal v2 doc is updated (figures, concentration,
  free-chat scope); its PDF export is not yet regenerated.
- 2026-11-01 (user): the colours are accepted, with the reading "Green is good, so buy,
  Amber is uncertain and Red is sell". This is wider than the report text, which says the
  list is not a recommendation, and it is the advice question of `assay/notes.md`, "Advice
  or tool": the report text is unchanged until the user decides which wording stands.
- 2026-11-01: built. `skatkis.report.Table` has `fills` (cell text to a background colour,
  `\cellcolor`, with `xcolor` `table` in the preamble); `value_investing.report._tier`
  returns Green (above the threshold), Amber (from the edge up to and including the
  threshold) or Red (strictly below the edge, down to 0); the column is headed "Colour". The
  `tier_ratio` docstring, its comment in `config/rules.toml` and the placeholder list in the
  report TOML are aligned. Black, ruff and pyright pass on the two report modules; 76 tests
  pass. The three value reports (S&P 500, 400, 600, as of 2026-10-31) were regenerated and
  compiled (tectonic outside the sandbox: inside it a certificate error stops the font
  download); in each, the worst ten are all Red. Unit tests for `fills` and `_tier` are for
  the Tester.

## Kickoff prompt

Pick up VI_202611_002. Read how `report.py` builds the sell check, then propose the layout
of the worst-stocks list before changing code.

## Done when

- The report shows the wallet's sells as cash flows and a general worst-stocks list; no
  section depends on what a user holds.
