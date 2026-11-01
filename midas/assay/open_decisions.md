# Assay: assumptions and open decisions

The material assumptions and unresolved decisions behind the business proposal v2
(2026-11-01). "Blocks" says whether an item must be settled before the free private beta
or only before a later stage. Changes from v1 are in
[proposal_v2_changes.md](proposal_v2_changes.md).

## Product

| Item | Current position | Unknown | Why it matters | Needed | Blocks |
|---|---|---|---|---|---|
| Plus price | About $4 a month (founder, indicative) | Final price | Drives contribution per payer | Willingness to pay, measured in the beta | paid launch |
| Studio price | Up to about $15 a month; lower with own AI | Final prices, and whether creators pay at all | Creator supply against creator income | Test with first creators | marketplace |
| Free Newtonian access | "Limited" | What the limit is (picks, models, delay, history) | Free must be useful yet leave a reason to upgrade | Founder decision | beta |
| Einsteinian access | Plus only | All variants or a subset | Defines Plus | Founder decision, once models exist | paid launch |
| Accredited creator models | May be in Plus | All included, or some at a fee | Creator income and Plus value | Founder decision | marketplace |
| Separate model subscriptions | Not decided | Whether models are priced apart from Plus | Pricing complexity, attribution | Founder decision | marketplace |
| "Customisation" | Plus feature | What it means in the product | Scope of the build | Product design | paid launch |
| Holdings tracking | Decided (user, 2026-11-01): in Plus. The member enters the exact amount; Assay remembers each pick bought and says what to sell, when and how much. Free members manage their own | How the wallet memory is built and stored; the sell advice depends on a sell analysis that VI_202611_001 marks "later", and the report's wallet never sells short, so "what to sell, when and how much" is untested until then | The main convenience Plus sells | Product design; the sell analysis | paid launch |
| Bring your own AI | Decided (user, 2026-11-01): in Studio, at a reduced fee because the member carries the AI cost | The reduced fee | Studio pricing | Founder decision | marketplace |
| Midas chat for free users | Decided (user, 2026-11-01): free, to explain the picks, and to answer general questions from Graham's reference only ("stick to Graham"); no personal allocation. The founder's profile and glide path stay an example for starting investors and are not served as the answer to "what should I hold"; a cost to Assay | Cost per free user (model uses 2, 5 and 10 cents a year, illustrative; Haiku is estimated under 1 cent, before the Graham and Buffett documents are retrieved for each question) | Free users behind each payer add cost; at 1.5% conversion and 10 cents it is a fifth of a payer's contribution. Personal allocation advice is the advice zone of `assay/notes.md`, "Advice or tool" | Measure in the beta with retrieval included; choose a cheap model | beta |
| The worst list | Decided (user, 2026-11-01): the report's list of the lowest-scoring companies is shown in colour, Green, Amber and Red, with the legend explained in the report's notes; no "avoid" or "start thinking about selling" wording. 10 companies for now. Colours accepted (user, 2026-11-01): Green above the sell threshold, Amber from half of it up to it, Red below half of it; built and in the reports. The user's reading is Green buy, Amber uncertain, Red sell | Which wording stands: the report says "not a recommendation", the user's reading is a buy and sell signal; whether "not a recommendation" is enough with a securities lawyer | The list is open to everyone, so its wording is where advice risk sits | Developer and Scribe; securities lawyer | public launch |
| Picks per report | Decided (user, 2026-11-01): 10 picks for now, in every tier | Whether tiers differ later | Earlier documents say 3 (`notes.md`) and 5 (VI_202611_001, the report text) | Records and report text brought into line | beta |
| Specification versioning | A calibrated specification is frozen as the client's version and a recalibration makes a new one (VI_202611_001); more samples and reruns may be a paid option (S_202611_003) | Storage per member; the compute of 10 samples plus 12 reruns per member, which the cost model (a shared quarterly screen) does not include | Running costs of Appendix F | Not important yet (user, 2026-11-01) | later stage |
| "Unlock" of near-tied candidates | Open (`notes.md`) | Whether it belongs in any tier: gamified framing draws regulators' attention ("Advice or tool") and sits badly with a character that hides nothing | Brand and regulation | Founder decision before it enters a tier | paid launch |
| Creator layer's name | Decided (user, 2026-11-01): Studio | — | — | — | — |
| Midas line | *Turning everyday labour into wealth* offered | Approve or not | Must not imply promised returns | Founder decision | public launch |
| Slogan | Decided (user, 2026-11-01): *Pick like a Pro. Old-world value, with a quant in your pocket.* The Buffett line is dropped | Trademark availability of "Pick like a Pro" | Public use | Trademark search | public launch |
| Studio datasets | Cleaned, standardised datasets for Studio members; developing economies later | Which datasets may be offered, and in what form (raw, derived, member's own key) | Reshaping data does not by itself allow resale; the vendor contract decides | Lawyer reading each vendor and exchange contract; redistribution quotes | marketplace |
| Short picks and real estate | Later paid sources, if members ask, after the platform is stable | Evidence for short books; platform support for shorting (real estate means listed REITs; no conflict with the founder's private business) | Paid add-ons beyond the core | Research and member demand | later stage |
| Developing-economy data | Aim | Which markets first; local exchange or regional vendor terms and costs | Testing strategies outside the US | Market and vendor survey | later stage |

## Creators

| Item | Current position | Unknown | Why it matters | Needed | Blocks |
|---|---|---|---|---|---|
| Content creator income | Publishing is free | Whether and how content creators earn | Supply of free content | Options appraisal | public launch |
| Commission rate | "Generous"; none set | The rate | Each 10% of Plus net revenue costs $4.08 per subscriber a year | Founder decision with the model | marketplace |
| Commission basis and attribution | Open | Gross or net; per model or per member; several models; duration; fees first; refunds and chargebacks | Fairness and accounting | Founder decision | marketplace |
| Ownership and maintenance | Open | Who owns the model; removal and transfer; upkeep duties | Creator terms | Legal advice | marketplace |
| Accreditation criteria | Candidate list in v2 | Final criteria and who reviews | Quality and liability | Founder decision; parts depend on luck tests not yet built | marketplace |

## Commercial

| Item | Current position | Unknown | Why it matters | Needed | Blocks |
|---|---|---|---|---|---|
| The $50,000 hurdle | Decided (user, 2026-11-01): operating profit before tax | Before or after reinvestment and one-off development costs | Changes the members needed | Founder decision | no |
| Download-to-paid conversion | 2.2% base (RevenueCat 2025 freemium median); 1.5% and 3% illustrative | Assay's own rate | Downloads needed | Beta data | paid launch |
| Acquisition | Content-led, no paid ads early | Cost and volume per channel | Whether the downloads are reachable | Beta data | public launch |
| Churn | 6% / 4% / 3% a month, illustrative | Assay's own churn | At 4%, about half the base is replaced each year | Beta data | paid launch |
| Fixed costs | $12,000 / $6,000 / $3,000, illustrative | Real hosting and data costs | Directly reduces profit | Quotes | paid launch |
| Data licensing cost | Unknown | Vendor and price for each use | May be the largest fixed cost | Vendor quotes | paid launch |
| Billing | Model: app store 15%, card monthly about 10.4%, card annual about 3.5% | Monthly or annual, web or app store | Annual billing saves about $5,000 a year at 2,000 payers | Founder decision | paid launch |
| Chat model | Claude Haiku 5.5 assumed, about $0.0005 a question; one shared explanation per pick | Real usage per free user | The free chat's cost | Beta data | beta |
| Beta workspace | GitHub Codespaces on the private repository; testers bring their own Claude and fetch their own data. Decided (user, 2026-11-01): the testers are friends, so this is a trust exercise for now; they can read `midas/` and change seeds, config and the test split | Codespaces allowance and price to confirm; keys and vendor data kept out (the founder is happy to share his portfolio approach). Before any outside member, access moves to the restricted `member` profile (`midas.md` §3) and the engine enforces the honesty rules | The beta's cost and privacy | Check GitHub pricing; review committed files | beta |

## Regulation and data

| Item | Current position | Unknown | Why it matters | Needed | Blocks |
|---|---|---|---|---|---|
| South Africa (FSCA, FAIS) | Open | Whether research, picks, marketplace or commissions need a licence | Home market | Securities lawyer | paid or public launch |
| Mauritius (FSC) | Considered for registration | Fit, and whether it covers South African members | Licensing route | Securities lawyer | paid or public launch |
| Abu Dhabi (ADGM FSRA) | Considered for registration | Fit and cost | Licensing route | Securities lawyer | paid or public launch |
| Australia (ASIC) | Later market | General against personal advice | Expansion | Securities lawyer | Australia |
| Data redistribution | Yahoo data not redistributable; EDGAR usable. Index membership comes from Wikipedia lists (CC BY-SA, `indices.toml`, MD_202611_001); S&P Dow Jones Indices may hold rights in constituent data and the index names (S&P 500, 400, 600; the IJH and IJR funds named as benchmarks) | Rights for display, backtests, ML, creator models; the attribution and share-alike terms of the membership lists; use of the index and fund names | Legal exposure and cost | Vendor terms; lawyer | paid or public launch |
| Founder disclosure | The property business is disclosed in the profile and the proposal, and the free chat serves the profile as an example | Whether the wording is safe; the business may itself be regulated | A disclosure read as advice or a conflict | Securities lawyer to read the profile and the proposal | public launch |
| Quotation of sources | The 48 Berkshire letters and the downloaded papers are stored locally in git-ignored folders | Nothing | Copyright | Quote briefly with the year and link; never serve the files to members | beta |

## Go to market

| Item | Current position | Unknown | Why it matters | Needed | Blocks |
|---|---|---|---|---|---|
| Model gate | A house model must at least match the S&P 500 and pass the tests of `midas/accreditation.md`. Status 2026-11-01: not cleared. Launch scope decided (user, 2026-11-01): the S&P 500 only, the big one; the S&P 400 and 600 are not cleared (AT-1 and AT-2 fail there) and stay out | On ten rules the S&P 500 wallet leads in 12 of 12 company draws (2020 start: 27.7% a year against 15.9%), but the edge is concentrated (MU 24% and LRCX 16% of the 2016 investor's gain) and the bands cover company sampling only, not multiple testing or time. The unseen test companies (AT-3) and the two periods (AT-5) are not yet rerun on ten rules; AT-4, AT-6, AT-7 and AT-8 are open | Nothing to publish as proven without it; "out-of-sample" is not claimed without that qualifier | Research: the open tests; the protocol is a 10% company hold-out on the same market history, and the time dimension is covered only by the two-period test | beta |
| Beta milestones | Founder, then invited friends | Size, length, what is measured | Turns guesses into data | Founder decision | beta |
| EasyEquities partnership | An aim; none exists | Interest, terms | Distribution and possibly licensing | Contact once the beta has a record | public launch |
