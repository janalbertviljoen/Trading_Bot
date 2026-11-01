# value_investing

The value-investing tool: S&P 500 fundamentals extracted point-in-time, Graham /
Buffett / Munger rules, a quarterly top 10 and bottom 10, and strategies backtested
against the S&P 500.

Until the code is final, the methodology is specified in
[docs/value_investing_bot_spec.md](../../docs/value_investing_bot_spec.md); this README
replaces it once the tool is built.

The tool reads its data (sources, cleaned tables and shared metrics such as the P/E) from
the shared package [market_data](../market_data/README.md); it owns the rule thresholds,
gates, ranking and strategies.

## Default rules

Ten rules score by default: G6, G2a, G7, G2b, E2, NN, E1, G5, B1 and E4 (eight concepts,
because G2a and E1 and G2b and E2 rank on the same values). G3, G4, E3, E5, E6 and NI1
are off. To switch one on, remove the leading `#` of its table in
[config/rules.toml](../../config/rules.toml) (for example `[rules.G3]` and its `years`);
to switch one off, put the `#` back. The windows and thresholds are explained in
[docs/value_investing_bot_spec.md](../../docs/value_investing_bot_spec.md), section 2.
