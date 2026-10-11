# EGX / Thndr offline smoke lab (research only)

This is an isolated experiment inside the user's fork of `qusong0627/QuantMind`.
It **does not run QuantMind's service stack**, load broker credentials, issue orders,
or modify Thndr's source database or production Qlib/Quant projects.

## Baseline and experiment

- Source: immutable Thndr historical candle snapshot (last 2026-10-07)
  and frozen research predictions from MLP and LightGBM.
- Universe: 22 liquid stocks from a 2026-08-26 fundamentals/technical screen.
- Test signals: 26 observed sessions, 2026-09-01 through 2026-10-06.
- Fill assumption: next observed session open to same-day close,
  nominal 20 basis points **per side**, no slippage or taxes.
- Actual upstream QuantMind strategy-template configurations are parsed
  as literals from `strategy_templates/as18_trend_gate.py` and
  `strategy_templates/as22_pullback_in_uptrend.py` (no source execution).
- Adaptation: Chinese turnover threshold replaced with EGP 2,000,000
  trailing five-day value. RSI is a Wilder-style EWM approximation.
- This diagnostic only tests the template gates on frozen model scores.
  It **does not reproduce** QuantMind RedisRecordingStrategy's rebalancing,
  TopK Dropout, Chinese share restrictions, or broker accounting.

## Re-run locally

Use the **already provisioned**, standalone EGX Qlib research Python
environment and the research snapshot without copying it to Git:

```bash
cd $HOME/work/quantmind-egx-lab
THNDR_RESEARCH_ROOT=$HOME/.local/share/qlib-11-models-20261011-20261007 \
  $HOME/WebstormProjects/egx-qlib/.venv/bin/python egx_lab/run_thndr_smoke.py
```

Output: `egx_lab/results/offline_smoke_20261011.json`.

**Important restrictions:** Historical fundamentals have estimated
publication timestamps and are **not fully PIT-certified**. Current
test signals have already been examined during strategy development:
results are exploratory, not independent confirmation. Sharia
compliance, fair value grade, individual EGX T+0 eligibility,
broker execution rules, market impact and all actual transaction
costs remain unverified. **No trading authority is granted.**

QuantMind upstream's license is AGPL-3.0. Forking under its existing
license is separate from adopting its implementation into other
proprietary or independent projects.
