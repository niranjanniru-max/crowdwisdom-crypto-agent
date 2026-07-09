cat > BACKTEST_FINDINGS.md << 'EOF'

# Backtesting Findings — CrowdWisdomTrading Crypto Prediction Agent

## Summary

This document reports the full backtesting process and results for the
6-agent crypto prediction pipeline, including two evaluation bugs found
and fixed along the way, and an honest correction to an earlier reported
accuracy number.

**Bottom line: no configuration tested (Kronos, technical indicator
confluence, or funding-rate mean-reversion) shows a statistically
significant directional edge on BTC/ETH across 1m–1h timeframes.**

---

## 1. Correction to earlier reported accuracy

The initial submission reported 61.8% BTC directional accuracy. That
number came from a small live run (~65 predictions) with no walk-forward
validation and no statistical controls. Once a proper backtesting harness
was built and validated, that result did not replicate. This document
supersedes that number.

---

## 2. Methodology

- **Walk-forward simulation**: predictions generated using only data
  available up to each point in time (no look-ahead bias).
- **Validation against a random-strategy control**: a `--strategy random`
  mode was added that ignores all real signals and picks direction with
  a coin flip. This confirmed the harness itself was unbiased once fixed
  (see bugs below) — random-mode accuracy landed at ~48–50%, as expected.
- **Confusion matrix cross-check**: every run's headline accuracy was
  verified by hand against its own confusion matrix.
- **Sample size discipline**: no result was trusted below n=40, and
  results were split into 1st-half / 2nd-half checks to catch
  small-sample luck.

## 3. Bugs found and fixed during evaluation

1. **Filter-mode leakage into random baseline**: `--strategy random`
   was not fully bypassing the signal pipeline, so changing
   `--filter-mode` changed "random" results — which is impossible for
   genuine randomness. Fixed by isolating the random-choice path from
   all real signal logic.
2. **Accuracy/confusion-matrix mismatch**: the headline "Accuracy on
   Trades Taken" stat did not match the accuracy implied by the
   confusion matrix on the same run (e.g. reported 38.5% vs. matrix-
   implied 48.7%). Root cause: the accuracy counter and the confusion
   matrix builder used different logic to determine "correct." Fixed
   by unifying both on the same comparison, and added an assertion to
   catch any future mismatch automatically.

Both bugs, uncorrected, made every real strategy look worse (or
occasionally artificially better) than it actually was. Fixing them
was a prerequisite for trusting any result below.

## 4. Results — Kronos + technical indicator confluence

| Timeframe | Filter Mode | Accuracy                      | Return |
| --------- | ----------- | ----------------------------- | ------ |
| 1h        | as-is       | 49.4%                         | -1.28% |
| 1h        | inverted    | 52.0% (n=50, not significant) | -0.44% |
| 1h        | removed     | 49.7%                         | -1.30% |
| 15m       | as-is       | 27–31%\*                      | -1.4%  |
| 1m        | as-is       | 38–39%\*                      | -1.7%  |

\*Lower timeframes show signals firing too often for their confidence,
consistent with 1m/15m direction being close to a random walk at this
resolution.

All results consistent with **no exploitable directional edge** once
corrected for the harness bugs above.

## 5. Results — Funding rate mean-reversion (independent hypothesis test)

Rationale: unlike price-derived indicators (RSI, MACD, EMA, etc.),
perpetual futures funding rate reflects trader positioning, not price
history — a genuinely independent information source. Tested as a
contrarian signal on funding-rate extremes (30-day rolling z-score).

| Test                              | n   | Accuracy | Verdict                                                 |
| --------------------------------- | --- | -------- | ------------------------------------------------------- |
| BTC, both directions, 15,000 bars | 85  | 50.6%    | No edge                                                 |
| BTC, long-only                    | 44  | 52.3%    | Not statistically significant, below fee-drag threshold |
| BTC, short-only                   | 41  | 48.8%    | Coin-flip                                               |

**Conclusion: funding-rate extremes do not currently show a reliable
tradeable edge on BTC at this sample size.** Recommend revisiting as a
monitoring alert (flagging crowded positioning) rather than a standalone
trading signal, and re-testing if funding regimes become more extreme
(e.g. during a leverage-driven market phase).

## 6. Overall conclusion and next steps

None of the tested configurations — Kronos alone, technical indicator
confluence, or funding-rate mean-reversion — clear the bar required for
live capital deployment. Given more time, next steps would be:

- Test longer timeframes (4h, 1d) where directional structure is more
  likely to exist than at 1m–1h resolution.
- Add order-book and on-chain features as additional independent
  information sources.
- Increase minimum sample-size requirements (500+ trades) before
  trusting any future signal, to avoid small-sample false positives —
  as happened initially with the funding-rate long signal (85% at n=20,
  regressed to 52% at n=44).

## 7. How to reproduce

\`\`\`bash
python backtest.py --asset BTC --bars 3000 --interval 1h
python backtest.py --asset BTC --bars 3000 --interval 1h --strategy random
python backtest.py --asset BTC --bars 15000 --interval 1h --strategy funding_only
python backtest.py --asset BTC --bars 15000 --interval 1h --strategy funding_only --direction long_only
\`\`\`

Full trade-level output: \`data/backtest_results.json\`, \`data/backtest_trades.csv\`
EOF
