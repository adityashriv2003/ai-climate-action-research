# Experimental results: forecasting GB grid intensity for flexible-load scheduling

## Result in one paragraph

On the chronologically later July--December 2024 retrospective test set,
validation-selected histogram gradient boosting (HGB) achieved aggregate MAE
30.66 gCO2/kWh across 0.5--24 h target-period completion leads, versus 36.09
for a stronger horizon-specific non-ML benchmark selected on validation: a
paired mean MAE reduction of 5.43 gCO2/kWh, or 15.05% (seven-day
circular-block-bootstrap 95% CI 1.37--9.14). This forecast gain did not imply
the best scheduling decision. In the primary simulation (one-kWh, three-hour
contiguous job within 18:00--06:00 UTC), the validation-selected training-only
climatology policy used 100.94 gCO2/job versus 102.17 for HGB and 151.97 for
immediate start. The signed climatology-minus-HGB difference was -1.22 g/job
(-1.21%), with a primary 95% CI of [-2.71, 0.16]; there is no reliable evidence
of incremental ML benefit in this configuration. These are average-intensity
accounting differences, not causal or marginal avoided emissions.

## Data and eligible samples

The input is the official NESO National Carbon Intensity Forecast CSV (resource
`0e5fde43-2de7-4fb4-833d-c7bca3b658b0`), retrieved 2026-09-05 21:42 UTC, SHA-256
`428476b2bab40d690f38309dd6eebe6e276631d570071785e5311f7ef45b6283`.
NESO defines `datetime` in UTC and `actual` as estimated national carbon
intensity from metered generation in gCO2/kWh. The downloaded file has 153,595
rows on a timestamp grid beginning in 2018 and extending beyond retrieval,
including forward-dated rows without finalized actual values; only targets
through 2024-12-31 enter the analysis.

The CKAN package metadata was modified at
`2026-09-05T21:10:38.004341` UTC and identifies licence ID `ESO`, title “NESO
Open Data Licence,” at
<https://www.neso.energy/data-portal/ngeso-open-licence>.

| Target period | Expected slots | Observed actual targets | Missing |
|---|---:|---:|---:|
| Train, 2022-01-01--2023-12-31 | 35,040 | 33,452 | 1,588 |
| Validation, 2024-01-01--2024-06-30 | 8,736 | 8,705 | 31 |
| Test, 2024-07-01--2024-12-31 | 8,832 | 8,832 | 0 |

Training missingness is concentrated rather than uniform: 1,441 of the 1,588
missing slots form one contiguous gap from 2023-03-06 14:30 through 2023-04-05
14:30 UTC.

Each horizon model has 33,452 training rows. Evaluation retains only model
origins whose full 24-hour horizon is temporally contained in its target
window: origins begin at or after the split start and end before the split
boundary minus 24 hours. After source missingness, each fitted horizon produces
8,657 validation model rows and 8,784 test rows. Common all-method pairing
leaves 8,578--8,595 validation rows per horizon, 412,033 validation instances
in total, and 421,632 =
8,784 x 48 test instances across 183 complete issue-date blocks (1 July--30
December).

## Leakage controls and model selection

The issue-time-ambiguous historical `forecast` field is not used because the
CSV provides no issue timestamp from which to identify an as-issued horizon.
Each of 48 models predicts one fixed horizon directly from the most recently
completed actual interval, older lags/rolling summaries, and target calendar
features. No predicted value becomes another model's input. All analogue
features point backward from the model-origin timestamp. The stored
`issue_time` is the start of the last completed settlement period, and the
operational decision is assumed at `issue_time + 30 minutes`; reported
0.5--24 h leads end at target-period completion. Evaluation uses full
forecast-origin cohorts, and HGB versus random-forest selection uses validation
aggregate MAE only. HGB was selected (34.10 versus 34.86 gCO2/kWh validation
MAE). A separate non-ML forecast benchmark selects the lowest-MAE baseline at
each horizon using validation only (persistence at 31 horizons and the
previous-day rule at 17). The scheduling comparator is likewise chosen on
validation only; training-only time-of-week climatology was best among the
four non-ML policies (108.62 g/job versus 109.16 for HGB and 111.79 for the
previous-day rule). The final selection calculations use validation outcomes
only, but the complete comparator audit followed initial test inspection; the
analysis is retrospective and not preregistered. Exact features and
hyperparameters are in `README.md`, `run_experiment.py`, and
`results/manifest.json`.

## Forecasting results

Aggregate later-period results on 421,632 paired forecast instances:

| Method | MAE | RMSE | Bias | MAE skill vs daily seasonal |
|---|---:|---:|---:|---:|
| HGB | 30.66 | 40.11 | +11.07 | 23.90% |
| Random forest | 32.34 | 41.96 | +12.04 | 19.73% |
| Validation-selected baseline ensemble | 36.09 | 48.16 | +0.11 | 10.42% |
| Persistence | 37.72 | 50.20 | +0.09 | 6.37% |
| Previous-day same slot | 40.29 | 52.59 | +0.18 | reference |
| Previous-week same slot | 53.50 | 68.88 | -0.64 | -32.77% |
| Train half-hour-of-week mean | 60.33 | 70.19 | +42.25 | -49.72% |

HGB MAE by representative horizon was 3.57 at 0.5 h, 14.78 at 3 h, 24.52 at
6 h, 34.24 at 12 h, and 40.64 at 24 h. At 24 h it was slightly worse than the
40.25 daily-seasonal baseline at that horizon; the aggregate gain is
concentrated at shorter and middle horizons. Its +11.07 overall bias also shows
systematic overprediction during a lower-intensity test period, consistent
with temporal distribution shift. Against the validation-selected baseline
ensemble, HGB's paired MAE improvement is 5.43 gCO2/kWh; 95% intervals are
3.09--7.80 for one-day IID blocks, 1.37--9.14 for the primary seven-day
circular blocks, and 0.41--9.75 for 14-day circular blocks (5,000 replicates,
seed 20240906). Longer 21- and 28-day block intervals cross zero
(-0.24--10.01 and -0.42--10.20), so the uncertainty conclusion is sensitive to
long-range dependence assumptions. The daily-minus-HGB MAE difference of 9.63
gCO2/kWh is retained as a secondary, less conservative comparison.

## Primary scheduling results

There are 183 paired decisions: each is made at 18:00 UTC using data through
the completed 17:30 interval, and the three-hour job may start in any half-hour
that keeps it within the following 12 hours. All strategies consume exactly
one kWh with the same uniform, contiguous six-slot trace.

| Strategy | Mean gCO2/job | Reduction vs immediate | Climatology minus strategy | Oracle potential captured | Delay mean / median / p95 |
|---|---:|---:|---:|---:|---:|
| Immediate | 151.97 | 0 | -51.03 | 0% | 0 / 0 / 0 h |
| Persistence | 151.97 | 0 | -51.03 | 0% | 0 / 0 / 0 h |
| Previous-day scheduler | 104.74 | 47.23 (31.08%) | -3.80 | 85.17% | 5.90 / 6.0 / 9.0 h |
| Previous-week scheduler | 104.60 | 47.37 (31.17%) | -3.66 | 85.43% | 5.90 / 6.0 / 9.0 h |
| Training climatology (validation-selected) | 100.94 | 51.03 (33.58%) | reference | 92.02% | 6.42 / 6.5 / 9.0 h |
| Random forest | 103.12 | 48.86 (32.15%) | -2.17 | 88.10% | 5.12 / 4.5 / 9.0 h |
| HGB | 102.17 | 49.81 (32.77%) | -1.22 | 89.81% | 5.27 / 5.0 / 9.0 h |
| Oracle (unattainable) | 96.52 | 55.45 (36.49%) | +4.42 | 100% | 5.89 / 6.0 / 9.0 h |

For the primary climatology-minus-HGB comparison, the 95% interval is
[-2.44, -0.05] g/job with one-day IID blocks, **[-2.71, 0.16]** with the primary
seven-day circular blocks, and [-2.90, 0.20] with 14-day circular blocks.
Positive values would favor HGB; the primary and 14-day intervals cross zero.
The 21- and 28-day intervals also cross zero ([-2.97, 0.18] and [-3.00, 0.26]).
For HGB versus immediate start, the corresponding seven-day interval is
44.16--55.31 g/job. The exploratory HGB-versus-previous-day contrast is +2.57
g/job (0.65--4.45), demonstrating how a weaker comparator can reverse the
conclusion.

Sensitivity results reinforce that flexibility assumptions matter. Signed
climatology-minus-HGB differences were -0.43 g/job for a three-hour job in a
six-hour window (7-day CI [-1.14, 0.04]), -1.39 for a one-hour job in a 12-hour
window ([-3.62, 0.65]), -1.22 for the primary three-hour job ([-2.71, 0.16]),
and -0.93 for a six-hour job ([-1.89, -0.03]). With a 24-hour window, HGB
instead improved on climatology by 7.00 g/job ([1.91, 12.29]). These five
contrasts are post-hoc, unadjusted, and dependent on block length: the
six-hour-job interval crosses zero with 14-day blocks ([-2.18, 0.16]), while
the 24-hour-window interval crosses zero with 21-day ([-0.42, 14.61]) and
28-day blocks ([-1.17, 15.36]).

## Scope and limitations

This is a predictive and accounting experiment, not a causal intervention.
Average national operational intensity does not measure the marginal generator
responding to load, lifecycle emissions, rebound, network constraints, user
noncompliance, or feedback from coordinated scheduling. NESO `actual` is an
estimate. The model omits weather, demand, generation and official as-issued
forecasts; its strength is a clean, auditable temporal baseline rather than
state-of-the-art accuracy. The fixed 18:00 UTC release corresponds to
different local clock times across the daylight-saving transition. The
pseudo-real-time replay treats the archived final `actual` for a completed
half-hour as immediately available; it does not reconstruct publication lag or
later revisions. Model selection uses aggregate 0.5--24 h MAE rather than a
loss aligned exactly with the 12-hour scheduling objective. Longer 21- and
28-day dependence blocks make the forecast interval cross zero. The complete
baseline comparison was refined after initial test inspection, so inference is
post hoc. The single later half-year limits temporal and geographic
generalization.

All result tables, paired predictions, inference JSON, plots, and executable
checks are in `results/`. `invariant_tests.json` records that all origin-boundary,
time-semantics, feature-availability, work-conservation, pairing, and
validation-only model-selection checks pass.
