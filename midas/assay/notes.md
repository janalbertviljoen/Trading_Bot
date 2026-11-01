# Assay notes

Project Assay: a consumer app that turns this repository's value-investing research
into picks people choose for themselves. Midas stays the name of the chat assistant
inside it. These notes are drafts; nothing here is built or decided beyond what each
entry says.

The business proposal written from these notes (2026-11-01) is a Claude doc:
[Assay: Business Proposal](https://claude.ai/artifact/4kpNDqC84SmaSHJkaN7xuW). It
names the tiers Standard, Plus and Premium.

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

The target (user, 2026-11-01): beat the index by about 2% a year, with not doing worse
as the minimum, reported also as inflation plus a margin (CPI + x%).

- **Safer against bubbles.** The rules exclude overvalued stocks by design, so a
  correction of an AI bubble or other expensive stocks should hurt the sleeve less than
  the index; this shifts the case towards betting on undervalued underdogs. A
  hypothesis, tested in `toolbox/value_investing/notes.md`, "Objective"; the cost is
  trailing the index while a bubble inflates.

### Interactive analysis (2026-11-01, open)

The member adjusts the inputs (how often the screen recalibrates, the number of picks,
risk settings) until the picks look attractive, then buys them cheaply on a platform of
their choice. This is the "quant in your pocket".

- **Frequency follows the data** (user, 2026-11-01): the screen never recalibrates more
  often than new data arrives. With paid data extractions it can run more often than
  quarterly. Filings arrive company by company through each quarter, so a screen that
  rescores each company when it files is possible even on EDGAR. A change to the
  rebalance schedule is a methodology change.

- Tuning settings until the history looks good fits the past, not the future. The app
  should show each setting's out-of-sample result (a hold-out the member cannot tune
  on), or the counting-cards promise does not hold. The repository applies the same
  rule to its own strategies (`instructions/AGENTS.md`, "Prohibited").

### Creator platform (2026-11-01, open)

A platform populated by its users, like YouTube or Instagram. Inquisitive members build,
test and optimise their own strategies and publish them; others follow, like and
subscribe; the creator earns a commission on the use of the model and posts their own
views on the market and the strategy.

- **Existing overlap** (from general knowledge, not yet searched): eToro's Popular
  Investor programme (creators paid on followers' copied assets), Seeking Alpha's
  Marketplace (paid author subscriptions), TradingView (published scripts and ideas),
  Collective2 and Darwinex (strategies followed for a fee), Composer's shared
  strategies, Stocktwits (social feed). Assay never executes trades, which keeps it
  nearer to TradingView or Seeking Alpha than to copy trading.
- **The assay is the platform's edge.** Every published strategy runs through the same
  point-in-time engine and is scored on the same out-of-sample hold-out against the
  S&P 500 total return. Among thousands of strategies the best in-sample are mostly
  lucky, so ranking and creator rewards should rest on results after publication and on
  the hold-out, never on the creator's own backtest.
- **Architecture:** strategies as declarative settings (rules, thresholds, weights,
  frequency) checked against a schema, as the `config/*.toml` files are today, not as
  code members upload. That keeps the engine deterministic and safe to run for anyone.
- **Regulation:** a creator whose published strategy others follow, for pay, may be
  giving advice in their own right; the platform's terms and licence need to cover
  creators. Part of "Advice or tool" below.

### Target market (2026-11-01, open)

People between the index investor and the day trader: willing to take more risk than
the S&P 500, but wanting to look at their portfolio only about once a year. Many
part-time traders have written strategy code but cannot tell whether its results were
luck, and lack the quant background to run a sound analysis.

### Two offers: house strategies and a testing platform (2026-11-01, open)

- **House strategies stay Graham.** The picks Assay publishes come from the Graham
  rules only.
- **Mavericks get a testing platform.** A member brings their own strategy, riskier or
  in another market (the BTC VWAP harness in `Toolbox/SandBox/btc_vwap/` is the
  example: someone with a strategy and no backtesting ability), and Midas backtests it
  and builds scenarios to show what it would have earned and whether it is viable.
  The scenario machinery is planned in
  [S_202610_008](../toolbox/skatkis/changes/S_202610_008-midas-wrath-scenario-testing.md)
  (`midas-wrath`).
- **Full Midas in the member's own environment** for the most capable members: their
  own workspace with the backtest engine, the scenarios and the reports. Running members'
  own code needs isolation (a sandbox per member, no network or data access beyond what
  the engine hands it); the declarative strategies of "Creator platform" are the safe
  default.

### The difference that holds: quant reports (2026-11-01, open)

Most of the plan overlaps with existing products; the transparent, quant-style report
does not. It shows every trade made to reach its conclusion, the good picks and the bad,
so the member's choice is informed. For a member's own strategy, the report answers
"was it luck?":

- out-of-sample results on a hold-out the strategy was not tuned on;
- the result against many random strategies of the same kind (a permutation or
  bootstrap test);
- the Sharpe ratio corrected for the number of variants tried (the deflated Sharpe
  ratio, Bailey and López de Prado, 2014);
- the S&P 500 total return as the benchmark throughout.

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

### Ambition: a hobby that may grow (2026-11-01, open)

A side project beside the user's day job, left to grow organically. Success is about
$50,000 a year in profit. That favours a self-service platform and creator content
over anything that needs staff: a few house models from this repository, the rest
content published by members.

- **What $50,000 takes, roughly:** at $4 a month ($48 a year), less the app stores'
  small-business fee of 15% and an assumed $8 a year per member for the chat and data,
  each member leaves about $33 a year, so about 1,500 to 2,000 paying members once fixed
  costs (price-data licence, hosting) are covered.
- **Platform defence, to check:** the user's model is the social-media one, where the
  platform is not liable for what members post. In the US that is Section 230, which
  covers liability for third-party content but not securities law for what the
  platform itself offers; paying creators, ranking their strategies and publishing
  house models may make the platform more than a host. South Africa's nearest
  protection is the conditional safe harbour for service providers in the Electronic
  Communications and Transactions Act (Chapter XI). Part of "Advice or tool" below.
- **Wording:** call the house models "house" or "Midas" models, not "accredited",
  which suggests a regulator's approval.

### First market and private beta (2026-11-01, open)

- **EasyEquities first** (user, 2026-11-01): its USD wallet holds the US shares the
  screen picks, in fractions, so small amounts work. South African retail investors
  rarely look beyond index funds, held back by not knowing which shares to choose or
  how much to put in each; that is the gap.
- **Private beta by invitation**, run like access to a private repository: the user
  first, then a few friends as testers. Free, with no fees and no public offer, which
  keeps it informal; once anyone pays or it opens to the public, "Advice or tool"
  below applies.
- **The first testers:** one friend who wants better trading advice; one who wanted to
  test a third-party strategy (Milk and Honey) that the user doubted but could not
  prove useless. That doubt is why the backtesting was built, and it is the
  testing-platform case in one person.

### EasyEquities partnership and Australia (2026-11-01, open)

The aim (user, 2026-11-01) is a partnership with EasyEquities: Assay moves its clients
beyond index funds into individual shares, which grows EasyEquities' portfolio base.
EasyEquities is a licensed provider, so a partnership may also answer "Advice or tool".
It serves clients in Australia, the first expansion beyond South Africa; advice there
needs an Australian Financial Services Licence from ASIC.

### Two model lines: Newtonian and Einsteinian (2026-11-01, open)

- **Newtonian** (user, 2026-11-01): Graham's rules, the `value_investing` tool;
  simple pass and fail buckets, every pick explained in a sentence.
- **Einsteinian** (user, 2026-11-01): models built with machine learning, the
  `hand_o_midas` tool; where Graham's buckets break into the AI generation.
- **One backbone:** both run through the same backtesting engine, hold-out and
  benchmark, so members compare them like for like. Machine-learning models overfit
  more easily, so the hold-out gate matters more for them; each pick shows which inputs
  drove it.
- Open: whether the Einsteinian line sits in a higher price tier.

### Members bring their own Claude (2026-11-01, open)

The second friend spends hours in Claude refining a strategy, without structure or
rules. The idea: testers connect their own Claude subscription to a restricted Midas
MCP server and develop the way the user does here, inside a controlled structure.

- The server exposes the rules, documents and backtests (the `index` and `lab`
  profiles of [../midas.md](../midas.md) §3), never the repository itself, and runs only
  the deterministic engine on declarative strategy settings.
- The member's Claude does the reasoning on the member's subscription, so the chat
  costs Assay nothing; this removes the per-member model cost from the tier pricing.
- In time members build their own strategies, a valuation model for example, in their
  own workspace (see "Two offers" and "Creator platform").
- What it needs, later: the server reachable remotely (HTTP with sign-in per tester)
  instead of the local process Claude Code starts today, one workspace per tester, and
  limits on how much each may run.

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

Paid tiers start once there are users. Ballpark prices (user, 2026-11-01), to revisit
later: $1 to $15 a month across the tiers, with about 90% of members expected on a $4
option. Entry costs almost nothing; costs and prices grow with revenue.

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
- **Main reference: fine watchmaking** (user, 2026-11-01). In a world of smartwatches,
  a mechanical watch from a maison such as Patek Philippe stands for old engineering
  precision and certainty; Assay adds modern flair, like a fine mechanical watch that can
  also pay contactlessly. Cues: the dial, the hand-finished movement behind a glass
  caseback, the chronometer certificate (as hold-out tests certify a strategy), and the
  hallmark that assay offices stamp on precious metal. Patek Philippe's "look after it
  for the next generation" is the spirit of value investing; idea only, not its words or
  marks. The Spyker reference below is kept as a second inspiration.
- **Second reference car** (user, 2026-11-01): the Spyker C8 Preliator coupé (2016). Spyker
  merged with a Dutch aircraft factory during the First World War; its logo is a
  propeller over a wire wheel and its motto *Nulla tenaci invia est via* ("for the
  tenacious, no road is impassable"). Its interior cues (turned-aluminium dashboard,
  exposed gear linkage, aviation-style switches) suit Midas's image prompt. Spirit
  only: no Spyker marks in Assay's branding.
- **Motto, proposed:** Seneca's *Ignis aurum probat*, "fire tests gold" (*De
  Providentia* 5.10): testing every strategy under stress before it is trusted.
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

Listed or not (2026-11-01, from general knowledge, not verified): of the direct
competitors only eToro (Nasdaq, `ETOR`, listed 2025) is public; the rest are private.
Listed neighbours: Robinhood (`HOOD`, whose Gold subscription is the nearest price
analogue to the $4 tier), Interactive Brokers (`IBKR`), Charles Schwab (`SCHW`),
Morningstar (`MORN`, S&P 400), Value Line (`VALU`, the old-school value research
publisher), and the data firms S&P Global (`SPGI`), MSCI (`MSCI`), Moody's (`MCO`) and
FactSet (`FDS`). All are GICS Financials, so the repository's universe lists the
S&P 500 and S&P 400 ones but holds no prices or fundamentals for them: financial
companies are excluded from the Graham screen (decided 2026-10-31).

### Revenue and costs of listed neighbours (2026-11-01, open)

Latest fiscal year, from the companies' results releases as summarised in a web search
on 2026-11-01; to check against the filings before relying on them.

| Company | Year | Revenue | Net income | Net margin | Per user |
|---|---|---|---|---|---|
| Robinhood | 2025 | $4.5 bn (net revenues) | $1.9 bn | about 42% | ARPU $191 a year; 4.2 m Gold subscribers |
| eToro | 2025 | $868 m (net contribution) | $216 m | about 25% | 3.81 m funded accounts, about $228 each |
| FactSet | FY Aug 2025 | $2.32 bn | $597 m | about 26% | institutional clients |
| Morningstar | 2025 | $2.4 bn | $374 m | about 16% (operating 21.5%) | mixed retail and institutional |
| Value Line | FY Apr 2026 | $33.4 m (publishing) | $21.6 m | operating income $4.0 m, about 12% | subscribers not disclosed |
| TradingView | private | estimates $25–300 m | unknown | unknown | valuation about $3 bn |

- **Robinhood Gold is the price benchmark.** Gold revenue was $50 m in the fourth
  quarter of 2025 over 4.2 m subscribers, about $4 a subscriber a month, the same as
  Assay's expected main tier. Gold is under 5% of Robinhood's revenue; the rest is
  interest and trading income, which Assay will not have.
- **Value Line is the warning.** The nearest pure value-research publisher: publishing
  revenue fell 4.7%, operating income fell a third, and its net income comes mostly
  from its stake in an asset manager and investment gains, not from subscriptions.
- **Costs** are revenue minus net income; these filings give no useful gross-profit
  line, because brokers and data firms report net revenues and operating expenses.

### Subscriber benchmarks (2026-11-01, open)

From a web search on 2026-11-01; most figures are vendor or third-party claims.

- **Conversion:** RevenueCat's 2025 report gives a median download-to-paid conversion
  of about 2.2% for freemium apps and 12.1% for hard paywalls. At 2%, 1,500 to 2,000
  paying members need about 75,000 to 100,000 sign-ups.
- **Paying subscribers:** Robinhood Gold 4.2 m (2025, reported); Stock Rover about
  170,000 (early 2024, third-party). Users, not payers: Simply Wall St claims 8 m
  investors, Seeking Alpha 20 m monthly users.
- **Prices:** Simply Wall St $131 to $258 a year; Stock Rover $348 to $948 a year.
  Assay's $48 sits well below both.

### Route to market (2026-11-01, open)

Proposed, not decided:

1. **Content before the app.** Publish the quarterly report free (the PDF pipeline
   exists), with the bad picks shown beside the good, in value-investing communities
   (Reddit, X, a newsletter). The audience comes before the product.
2. **A public, timestamped track record** from the first quarter, which cannot be
   backdated and grows more credible every quarter.
3. **Web first, app stores later**, to keep the 15–30% store fee and the review delays
   out until the product works.
4. **A narrow first niche.** Either South African investors (ZAR, tax-free savings
   accounts, US shares bought through local platforms), whom the large players ignore,
   or strategy coders who want to know whether their backtest was luck.
5. **No paid advertising early**, because clicks on finance keywords are expensive.

### The niche against large firms (2026-11-01, open)

From the user's audit work at large firms: their analysis may be deeper, but their
infrastructure runs on Excel and SAS, their reports are slow to produce and hard to
scale, and security and approval processes slow their adoption of AI. A small, nimble
provider with automated reports can reach the market first. Graham's rules are simple
enough that depth of analysis is not the deciding factor.

### Differences (2026-11-01, open)

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
