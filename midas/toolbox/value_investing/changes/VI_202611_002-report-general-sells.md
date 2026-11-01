# VI_202611_002: Report: general worst list, sells tracked in the wallet's cash flows

| Field | Value |
|---|---|
| Status | in progress |
| Priority | medium |
| Kind | new capability (the report's content; no change to the scores) |
| Depends on | VI_202610_006 |
| Branch | not yet |
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
- The worst stocks are a general list of the 5 lowest-scoring companies on the latest date
  (the same count as the picks, `[wallet] names`), by `score_binned_weighted` and its 0-to-1
  share (`score_weighted_share`), for everyone.
- The list is tiered by the share, with the tier edges taken from the cut-offs in use
  (today `sell_threshold = 0.4`; the fitted cut-offs of VI_202611_001 later):
  1. at the sell threshold (0.4): **avoid**; hold it if you already have it;
  2. at half of it (0.2): **start thinking about selling**;
  3. at 0: it is not worth anything any more.
  The numbers are not fixed; they follow the cut-offs.
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
- 2026-11-01 (user): the list has 5 companies, the same as the picks; tiers as above; a
  short position is a later step.

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

## Kickoff prompt

Pick up VI_202611_002. Read how `report.py` builds the sell check, then propose the layout
of the worst-stocks list before changing code.

## Done when

- The report shows the wallet's sells as cash flows and a general worst-stocks list; no
  section depends on what a user holds.
