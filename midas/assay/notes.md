# Assay notes

Project Assay: a consumer app that turns this repository's value-investing research
into picks people choose for themselves. Midas stays the name of the chat assistant
inside it. These notes are drafts; nothing here is built or decided beyond what each
entry says.

## Planning (preliminary)

### Product first (2026-11-01, open)

The app is a long way from marketable, and nothing is decided or launched until the
tool is solid. The order is: a strategy that holds its own against the S&P 500
total-return index out-of-sample in this repository, then a working product, then
users, then paid tiers and infrastructure scaled with them. The research goals in
[../midas.md](../midas.md) §1 come first; Assay builds on their results.

### The promise: at least the market, without the guesswork (2026-11-01, open)

The sell is not beating the market. It is getting at least the market's return while
knowing what you own, not doing worse, and not having to decide what to pick, how much
to buy or when to sell. Assay makes those decisions transparently; the member follows
them or not.

What this asks of the research framework, before any app:

- Judge a strategy on how often and by how much it trails the S&P 500 total return
  over rolling periods, and on its drawdown against the index, not only on its average
  excess return.
- Position sizing and the buy and sell decisions are part of the product, so the wallet
  and weighting rules are as central as the picks.
- Every pick carries its reasons (which rules it passed), already an output of the
  rule catalogue.

### Origin: the user's own sleeve (2026-11-01, open)

The starting point is personal: a small fraction of the user's portfolio placed on
underdogs that can boom, expecting not to clear more than the market overall. Assay is
the same tool offered to others.

### The big picture sets the architecture (2026-11-01, open)

The tool is built for the user first, but the app decides how rigid the architecture
must be. Points to keep open while building, without building the app:

- Strategies and the wallet as pure functions of data and parameters (already a rule
  of the repository), so one member's settings are just another parameter set.
- Data sources behind ports, so a licensed price vendor can replace `yfinance`
  without touching the tools.
- Results per run kept apart, so many members' portfolios can be tracked from the same
  quarterly screen.

### Concept (2026-11-01, open)

- **Information to the masses.** Screening on Graham's rules is cheap to run, so the
  core is close to free. The aim is not to sell scarce information or to copy a hedge
  fund: the analysis is far lighter than a fund's, and so are the costs.
- **Do it yourself, at your own risk.** The user sets their own preferences and takes
  the risk, as with the crypto fad, but the app shows why the picks have given
  reasonable returns. The framing is counting cards, not a lottery: a measurable edge,
  shown openly.
- **The chat steers the user's settings.** Midas, the chatbot, helps the user express a
  risk appetite and turns it into the screen's parameters; the user makes the picks.
- **No money handled.** Assay manages subscriptions and member portfolios only; it
  never holds funds or executes trades.
- **Underdogs.** Find companies with the potential to become giants. Graham's net-nets
  (stocks below net current asset value) are where "underdog" fits the method honestly;
  the large, stable, cheap companies of his defensive rules do not.
- **Later niches:** picks within a GICS sector.

### Tiers (2026-11-01, open)

| Tier | Price | Offers |
|---|---|---|
| Standard | free or almost free | about 3 picks; a few adjustable parameters |
| Paid | small | more room to change risk parameters; an investment plan with more picks; the full report |
| Premium | small | tracks the picks the member says they bought and prints the full portfolio report |

Paid tiers start once there are users.

### Brand (2026-11-01, open)

- **Name: Assay** (user's choice, 2026-11-01). To assay is to test whether gold is
  genuine. The word comes from Old French *assai*, a variant of *essai* ("trial"), from
  Late Latin *exagium* ("a weighing"); it has been in English since the 14th century
  and shares its root with *essay*, which fits an app that writes a report. Trademark
  and domain availability not checked yet.
- **Theme:** old school meets fintech. Graham's almost forgotten principles with an AI
  chat. The look of modern hypercars that recall 1960s–70s aviation: steampunk
  refinement, brass, gold and exposed engineering; style and wealth, at small prices.
  Brand names (Pagani) are inspiration only and stay out of the marketing.
- **Positioning:** a quant in your pocket; measurable value against crypto's hype,
  keeping the spirit of adventure.
- **Image prompt for Midas** (drafted 2026-11-01): King Midas as a 1960s test pilot and
  coachbuilder, a crown of watch gears and an altimeter bezel, a leather flight jacket
  with gold-leaf stitching, in the cockpit of a sculpted hypercar with aluminium,
  carbon weave, exposed titanium and brass gear linkage and analogue aviation gauges;
  the wheel turns to gold where he touches it; a riveted hangar at golden hour.
  Negative prompt: cartoon, fantasy armour, neon, cyberpunk, crypto coins, cash,
  logos, text. App-icon variant: a gold crown whose points are gauge needles, inside an
  aviation instrument bezel, brushed brass on charcoal, no text.

## Market

### Competitors and overlap (2026-11-01, open)

From a web search on 2026-11-01; prices are from promotional pages and need checking.

| Product | Offers | Overlap |
|---|---|---|
| InvestingPro, ProPicks AI (investing.com) | Pro and Pro+ tiers; pre-built AI strategies updated monthly, the same for every subscriber | picks and tiers; not personalised, no chat |
| Validea | model portfolios on gurus' rules, including a Graham model | the closest on method; no personalisation or chat |
| GuruFocus, Old School Value | Graham Number and guru screens, valuation tools | Graham metrics as a tool for enthusiasts |
| Magnifi | conversational AI assistant with a brokerage | chat-steered investing; no stated method |
| Danelfin | AI score from 1 to 10 per stock | ranked picks; a black box |
| Sterling Stock Picker, StockPick AI | risk questionnaire, then picks | almost the standard tier; low-end |
| Simply Wall St, Seeking Alpha | portfolio tracking, visual reports, quant ratings | the premium tier |
| Composer | build, backtest and automate rule strategies | customisation for technical users |

Where Assay can differ: rules shown openly (why each test passed), a written report per
member, honest out-of-sample results against the S&P 500 total return, and a brand
unlike the terminal or crypto look of the rest.

## Open questions

### Advice or tool (2026-11-01, open)

Picks tailored to one person's risk profile count as investment advice in most
jurisdictions, whatever the disclaimer says; gamified framing draws regulators'
attention. The "do it yourself" design, where the user sets the parameters and the app
shows what passes the screen, sits closest to a tool. To settle with a securities
lawyer before launch.

- The user is based in South Africa and would likely register in Mauritius or Abu
  Dhabi. To confirm: whether South Africa's FAIS Act still applies to a foreign
  provider serving South African users (the FSCA generally expects authorisation), and
  which licence fits in Mauritius (FSC) or Abu Dhabi (ADGM FSRA).

### Data licensing (2026-11-01, open)

SEC EDGAR may be used commercially. Yahoo price data (`yfinance`) may not be
redistributed in a product, free or paid; a public app needs a licensed price vendor.
Using the name "S&P 500" commercially may need a licence from S&P Dow Jones Indices.
