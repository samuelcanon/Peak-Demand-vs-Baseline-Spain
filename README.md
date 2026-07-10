# When Simple Beats Sophisticated: Forecasting Spain's Peak Electricity Demand
### A linear regression was pre-registered against a trivial "tomorrow = today" guess. The guess won and diagnosing exactly why, down to the one pattern that beat every method tested.

**Stack:** Python · pandas · SQL (SQLite) · scikit-learn · matplotlib/seaborn · ENTSO-E · Open-Meteo (ERA5)

---

## TL;DR

A three-source **SQL pipeline** (ENTSO-E demand, Open-Meteo weather, derived calendar) fed a linear regression **pre-registered** against one question: can an interpretable, feature-based model beat the trivial guess that tomorrow's peak looks like today's? **It couldn't by 32.8%.**

That's not where this stops. The loss was diagnosed in three steps: ruled out "the extra features hurt" (a lag-only model was worse still), ruled out "the model's shape is wrong" (a non-linear model still lost), and landed on the real cause, the model dilutes yesterday's signal to roughly a third of its true weight. That same diagnostic process surfaced the one pattern that survived every method tested: **weekends**. The model already tries to account for them, it's one of only seven features, and still can't close the gap.

| Approach | Test MAE (MW) | Result |
|---|---|---|
| **Naive persistence** (tomorrow = today) | **1,565** | the benchmark to beat |
| Feature-based model (temperature + calendar) | 2,078 | **32.8% worse, doesn't earn its complexity** |
| Non-linear diagnostic (gradient boosting, untuned) | 1,872 | still **19.6% worse**, rules out "wrong model shape" |

*Held-out test: all of 2024 (364 days), trained on 1,821 prior days.*

---

## The Diagnosis

**1. The model loses by design of the test.** This was a **pre-registered** question (locked before the data was seen) so a loss is a clean result. Daily peak demand is highly autocorrelated, which makes "assume tomorrow looks like today" a genuinely hard baseline to beat.

**2. Two explanations were tested and ruled out, not assumed.** A lag-only model (trained on yesterday's peak alone) scored *worse* than the full model, so the extra features aren't the problem, they help slightly. A gradient-boosting model that can capture non-linear demand patterns still lost to the baseline, the model's shape isn't the problem either.

**3. The real mechanism: the model dampens the one signal that matters most.** Linear regression assigns yesterday's peak a coefficient of roughly **0.3**; the naive baseline effectively uses it at **1.0**. The model dilutes exactly the signal it should be leaning on hardest.

**4. That same diagnostic process found the one pattern every method shares: Weekends.** Slicing the errors by day type: the naive baseline's error jumps **70%** on weekends (1,305 → 2,214 MW). The feature based
model's jumps only **27%** (1,927 → 2,456 MW), a smaller relative degradation, though still a higher absolute error than the baseline. The model already includes an explicit weekend indicator as one of its seven
features, and still can't close the gap. That rules out "a missing weekend feature" and points at something about weekend volatility the current feature set doesn't capture.

**5. A second pre-registered hypothesis was tested and disproven.** Error was expected to peak at temperature extremes. It didn't, the mildest days were hardest to predict. No mechanism is claimed; the pattern is real, the cause isn't established, and that's said plainly rather than papered over.

---

## The business takeaway

**Don't build a feature-based model for this task** a simple persistence rule is already the ceiling. That's a real resourcing recommendation, tested honestly rather than assumed, and it tells a team where *not* to spend modelling budget.

**If forecasting effort goes anywhere, it should go at weekend volatility specifically** not at more features or a fancier model in general, since neither closed that gap here. That's a sharper, evidence-backed
target than temperature or season ever were.

---

## Charts

| | |
|---|---|
| ![U-shape scatter](charts/chart1_ushape_scatter.png) | ![Three-way MAE](charts/chart4_three_way_mae.png) |
| **Demand vs. temperature:** the two-sided pattern that motivated testing a model at all. | **The result:** nothing beats the naive baseline, non-linear model included. |
| ![Linear model behaviour](charts/chart2_linear_behaviour.png) | ![Weekend gap](charts/chart3_weekend_gap.png) |
| **Model behaviour** over the test year, tracks well, misses on hard days. | **The standout pattern:** every method misses more on weekends. |

---

## The Data Engineering (the SQL work)
Three sources, joined relationally, the way it would be done in production:

- **Demand:** ENTSO-E hourly actual load for Spain (reduced to a daily peak).
- **Weather:** Open-Meteo ERA5 hourly temperature (daily max/min/mean).
- **Calendar:** Day-of-week, month, season, holiday, weekend (derived), zero leakage risk.

All three loaded into **SQLite** and combined with a real three-source `INNER JOIN` on a daily date key (2,189 joined rows, zero missing values). The data is small enough that pandas could technically do this join but SQL is used here because it's the relational, production-realistic approach and because it's a near universal tool used in the industry, not because of scale. The actual engineering challenge was upstream: getting three independently-sourced date columns onto one consistent, timezone-clean key so the join means something.

Built around one `COUNTRY_CODE` parameter, the same pipeline runs on any ENTSO-E country by changing a single value.

---

## Methodology
- **Split:** Chronological (train on earlier years), test on the held-out most recent year (2024). Never shuffled, since a forecast has to be evaluated the way it's actually used: predicting forward in time.
- **Leakage control:** Every input is knowable on the day it's used, same-day peak is excluded, and checked explicitly via correlation (max 0.4 no red flags).
- **Metrics:** MAE (average miss in MW, easy to explain to non-technical stakeholder) as the headline. RMSE as a secondary check for large misses.

---

## Limitations
- Trains on **actual** temperature; a real deployment would use **forecast** temperature, adding its own error. Forecasted data is available but not for thetime span's entirety.
- Single-city temperature (Madrid) proxies national weather.
- 2020–2021 excluded (COVID demand shock decoupled demand from its normal drivers).
- One production model, tested honestly against a simple baseline (not tuned to force a win).

## Future Work
- The weekend-volatility signal the diagnosis surfaced 
- Different model class or a volatility-specific feature, varing features.
- Multi-city temperature, population weighted.
- Genuinely forecasted temperature at a fixed lead time.

## Reproduce
bash
pip install -r requirements.txt
# add your ENTSO-E API token as an environment variable, then run the notebook top to bottom.

Outputs: `data_aligned.csv` (model-ready data) and `results_summary.csv` (headline metrics).

## Attribution & Disclaimer
Weather data:
[Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. 
Demand data:
**ENTSO-E Transparency Platform**. 
Educational / portfolio project — not an operational forecasting tool.
