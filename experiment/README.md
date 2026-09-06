# GB carbon-intensity forecast and load-shifting experiment

This folder contains a reproducible empirical study using National Energy
System Operator (NESO) national carbon-intensity data at 30-minute UTC
resolution. The measured target is NESO's estimated `actual` national average
operational carbon intensity in gCO2/kWh.

## Provenance

- Official dataset page: <https://www.neso.energy/data-portal/national-carbon-intensity-forecast/national_carbon_intensity_forecast>
- Direct official CSV: <https://api.neso.energy/dataset/f406810a-1a36-48d2-b542-1dfb1348096e/resource/0e5fde43-2de7-4fb4-833d-c7bca3b658b0/download/gb_carbon_intensity.csv>
- CKAN resource ID: `0e5fde43-2de7-4fb4-833d-c7bca3b658b0`
- CKAN package metadata last modified: `2026-09-05T21:10:38.004341` UTC
- Licence: `ESO`, **NESO Open Data Licence**,
  <https://www.neso.energy/data-portal/ngeso-open-licence>
- Retrieved: `2026-09-05T21:42Z` (UTC)
- Retrieved file SHA-256: `428476b2bab40d690f38309dd6eebe6e276631d570071785e5311f7ef45b6283`
- NESO documents `datetime` as UTC, and `forecast`/`actual` in gCO2/kWh;
  `actual` is estimated from metered generation.

The downloaded file is `data/gb_carbon_intensity.csv`. The script checks and
enforces the pinned checksum before attaching the recorded dataset metadata.
The current Data Portal file changes as new half-hours
arrive, so an exact rerun should retain the pinned input file rather than
redownload it.

## Critical anti-leakage decision

The source's historical `forecast` column is excluded. It has no issue-time
field, so its as-issued horizon cannot be identified and it cannot support a
fixed-horizon forecast evaluation. This experiment instead forecasts finalized
`actual` values using only historical `actual` values and calendar fields
available from settlement intervals completed by the operational decision
boundary.

Every split is chronological, based on timestamps rather than row order:

- train: 2022-01-01 through 2023-12-31;
- validation/model selection: 2024-01-01 through 2024-06-30;
- later retrospective test window: 2024-07-01 through 2024-12-31.

Validation and test evaluation use complete 24-hour forecast-origin cohorts:
an issue time must be at or after the split start and earlier than the split
end minus 24 hours. This purge prevents a validation forecast from being
issued before the training cutoff or a test forecast before validation has
finished, and it makes the same 48 horizons temporally eligible at every
retained origin. Source missingness can still remove individual validation
targets.

Training missingness is not approximately random: 1,441 of the 1,588 missing
training half-hours form one contiguous gap from 2023-03-06 14:30 through
2023-04-05 14:30 UTC. The data-quality artifact records this run explicitly.

The 48 direct models predict one to 48 settlement periods ahead. The stored
`issue_time` is the start of the most recently completed 30-minute period; lag
0 is assumed available at `decision_time = issue_time + 30 minutes`. Reported
0.5--24 hour horizons are leads from that decision boundary to completion of
the target period. Leads to target-period start are 0--23.5 hours. This
pseudo-real-time convention does not model data publication latency or later
corrections. Forecast features comprise
past lags (0, 1, 2, 3, 47, 48, 49, 95, 96, 97, 335 and 336 half-hours),
trailing mean and standard deviation over 6, 12, 48, 96 and 336 half-hours,
trailing minima/maxima over 12 and 48 half-hours, changes from 1, 2 and 48
half-hours earlier, horizon-specific prior-day/prior-two-day/prior-week target
analogues, and sine/cosine target-time encodings for time of day, time of week
and time of year plus a weekend flag. Lag 0 is the most recently completed
half-hour. Every analogue offset is nonnegative for horizons 1--48. The setup
is **direct**, with a separately fitted estimator for each horizon; predictions
are never fed recursively into later horizons. Median imputation is fitted on
training rows only.

The fixed HGB configuration uses squared-error loss, learning rate 0.06, 160
iterations, 31 maximum leaf nodes, minimum leaf size 30, L2 penalty 1.0, no
early stopping and seed 20240906. The fixed random forest uses 96 trees,
maximum depth 18, minimum leaf size 3, 75% of features per split, all available
cores and the same seed. Only validation aggregate MAE selects between these
two models in the final executable analysis. Baselines are: current
last value for all horizons; the same target slot one day earlier; the same
target slot one week earlier; and a half-hour-of-week mean calculated from
training targets only. The main forecast comparator selects the lowest
validation MAE among these four rules independently at each horizon.

The complete baseline comparison and dependence checks were added during
reproducibility audit after initial H2-2024 output had been inspected. The
final selection calculations use validation outcomes only, but this workflow
is not equivalent to preregistration. Test-period inference is retrospective
and requires independent replication.

## Scheduling estimand

At 18:00 UTC each test day, a controller schedules a one-kWh, uniform,
contiguous flexible job. The primary case is a three-hour job within the next
12 hours. It selects the lowest predicted contiguous block. Immediate start is
the no-shift baseline; perfect-future-information selection is labelled
`Oracle` and is only an unattainable upper bound. Sensitivities vary duration
and window length.

Reported grams are an accounting simulation: energy multiplied by **average**
grid intensity. They are not avoided marginal emissions and do not establish a
causal grid response.

All four declared non-ML forecasts are replayed through the identical
scheduler. The main corrected comparator is the non-ML policy with the lowest mean
assigned emissions in the primary configuration on validation data only;
training-only time-of-week climatology is selected. The previous-day and
immediate-start comparisons are secondary. The primary 95% interval is a
seven-day circular moving-block bootstrap over ordered UTC days (5,000
replicates); one-day IID and 14-, 21-, and 28-day circular blocks are reported
as dependence sensitivities. Delay medians and 95th percentiles are also
retained.

## Reproduction

From the repository root, install the pinned requirements in a Python 3.12
environment and run:

```sh
python experiment/run_experiment.py
```

After a completed model run, inference, invariant tests, and figures can be
regenerated without refitting:

```sh
python experiment/postprocess_results.py
```

Postprocessing verifies the fit-time experiment-code hash and the prediction
and sample-count artifact hashes before consuming them; it refuses to attach
current-code provenance to stale or modified fits.

For a clean environment, install `requirements.txt` and invoke the script with
Python 3.12. Results are written under `results/`; `manifest.json` records model
parameters, split boundaries, library versions, the validation-selected model,
the horizon-specific validation-selected forecast benchmark, and the
validation-selected scheduling comparator.

## Interpretation limits

- NESO `actual` is an estimate, not a direct emissions measurement.
- The pseudo-real-time replay treats the archived final `actual` for the
  just-completed half-hour as available at the next boundary; publication lag
  and any later corrections are not reconstructed.
- National average operational intensity does not represent location-specific
  or marginal emissions, upstream life-cycle emissions, or demand feedback.
- Weather, demand and generation forecasts are absent; this is intentionally a
  lightweight autoregressive/calendar benchmark, not a state-of-the-art NESO
  replacement.
- The fixed 18:00 UTC release corresponds to different local clock times across
  the daylight-saving transition.
- Missing values are reported; forecast metrics use a common paired sample and
  schedules require complete actual/predicted windows.
- Repeated overlapping forecasts are dependent. The reported uncertainty uses
  a seven-day circular moving-block bootstrap (one-, 14-, 21-, and 28-day
  sensitivities). Longer-block forecast intervals cross zero, and one later
  half-year cannot establish broad temporal or geographic generality.

`invariant_tests.json` records split-boundary, feature-availability,
constant-trace, zero-slack, paired-count, work-conservation and
validation-selection checks. All must pass for postprocessing to complete.
