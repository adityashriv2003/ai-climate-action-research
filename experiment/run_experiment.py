#!/usr/bin/env python3
"""Leakage-resistant GB grid carbon-intensity forecasting and scheduling study.

The input is NESO's National Carbon Intensity Forecast CSV. Importantly, the
historical ``forecast`` column is intentionally NOT used: the CSV does not
retain a forecast issue timestamp, so an as-issued horizon cannot be identified.
Models below forecast the finalized ``actual`` series using only settlement
intervals completed by the operational decision boundary.

Retrospective backtest design:
  * target: national average operational carbon intensity, gCO2/kWh, 30-min UTC
  * train targets: 2022-01-01 through 2023-12-31
  * validation origins: 2024-01-01 through 2024-06-29, with all targets in H1
  * later test origins: 2024-07-01 through 2024-12-30, with all targets in H2
  * direct forecasts: 1--48 half-hours ahead (0.5--24 h)
  * ML: fixed-configuration histogram gradient boosting and random forest
  * baselines: last value, previous-day same slot, previous-week same slot,
    and train-only half-hour-of-week climatology
  * scheduling: a fixed-energy contiguous job shifted within a fixed window

Run from the repository root:
  python experiment/run_experiment.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
SEED = 20240906
FREQ = pd.Timedelta("30min")
HORIZONS = tuple(range(1, 49))
TRAIN_START = pd.Timestamp("2022-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2024-07-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-01-01T00:00:00Z")

SOURCE_URL = (
    "https://api.neso.energy/dataset/"
    "f406810a-1a36-48d2-b542-1dfb1348096e/resource/"
    "0e5fde43-2de7-4fb4-833d-c7bca3b658b0/download/"
    "gb_carbon_intensity.csv"
)
DATASET_PAGE = (
    "https://www.neso.energy/data-portal/national-carbon-intensity-forecast/"
    "national_carbon_intensity_forecast"
)
CKAN_RESOURCE_ID = "0e5fde43-2de7-4fb4-833d-c7bca3b658b0"
DATASET_METADATA_MODIFIED_UTC = "2026-09-05T21:10:38.004341"
RETRIEVED_AT_UTC = "2026-09-05T21:42Z"
LICENSE_ID = "ESO"
LICENSE_TITLE = "NESO Open Data Licence"
LICENSE_URL = "https://www.neso.energy/data-portal/ngeso-open-licence"
EXPECTED_INPUT_SHA256 = (
    "428476b2bab40d690f38309dd6eebe6e276631d570071785e5311f7ef45b6283"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: str
    parameters: dict


MODEL_SPECS = (
    ModelSpec(
        "HistGradientBoosting",
        "sklearn.ensemble.HistGradientBoostingRegressor",
        {
            "loss": "squared_error",
            "learning_rate": 0.06,
            "max_iter": 160,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": SEED,
        },
    ),
    ModelSpec(
        "RandomForest",
        "sklearn.ensemble.RandomForestRegressor",
        {
            "n_estimators": 96,
            "max_depth": 18,
            "min_samples_leaf": 3,
            "max_features": 0.75,
            "n_jobs": -1,
            "random_state": SEED,
        },
    ),
)

BASELINES = ("Persistence", "DailySeasonal", "WeeklySeasonal", "Climatology")
ALL_METHODS = BASELINES + tuple(spec.name for spec in MODEL_SPECS)
SELECTED_BASELINE_COLUMN = "ValidationSelectedBaseline"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: Path) -> tuple[pd.Series, dict]:
    input_sha256 = sha256_file(path)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Input does not match the pinned NESO CSV checksum: "
            f"expected {EXPECTED_INPUT_SHA256}, received {input_sha256}"
        )
    raw = pd.read_csv(path)
    required = {"datetime", "forecast", "actual", "index"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    if raw["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps in NESO source file")
    raw = raw.sort_values("datetime").set_index("datetime")

    # Reindex explicitly, so an absent record is a missing value rather than an
    # accidental shift in positional lags.
    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq=FREQ)
    raw = raw.reindex(full_index)
    actual = pd.to_numeric(raw["actual"], errors="coerce").astype(float)
    actual.name = "actual"

    yearly = []
    for year in (2022, 2023, 2024):
        lo = pd.Timestamp(f"{year}-01-01T00:00:00Z")
        hi = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
        values = actual.loc[(actual.index >= lo) & (actual.index < hi)]
        yearly.append(
            {
                "year": year,
                "expected_half_hours": int((hi - lo) / FREQ),
                "records_after_reindex": int(len(values)),
                "actual_missing": int(values.isna().sum()),
                "actual_available": int(values.notna().sum()),
                "actual_mean_gCO2_per_kWh": float(values.mean()),
            }
        )
    training_values = actual.loc[
        (actual.index >= TRAIN_START) & (actual.index < TRAIN_END)
    ]
    missing_times = training_values.index[training_values.isna()]
    missing_runs = []
    if len(missing_times):
        run_id = pd.Series(missing_times, index=missing_times).diff().ne(FREQ).cumsum()
        for _, run in pd.Series(missing_times, index=missing_times).groupby(run_id):
            missing_runs.append(
                {
                    "start_utc": run.iloc[0].isoformat(),
                    "end_utc_inclusive": run.iloc[-1].isoformat(),
                    "half_hours": int(len(run)),
                }
            )
        missing_runs.sort(key=lambda item: item["half_hours"], reverse=True)
    try:
        portable_input_file = str(path.resolve().relative_to(ROOT))
    except ValueError:
        portable_input_file = str(path)
    quality = {
        "source_url": SOURCE_URL,
        "dataset_page": DATASET_PAGE,
        "ckan_resource_id": CKAN_RESOURCE_ID,
        "dataset_metadata_modified_utc": DATASET_METADATA_MODIFIED_UTC,
        "retrieved_at_utc": RETRIEVED_AT_UTC,
        "license_id": LICENSE_ID,
        "license_title": LICENSE_TITLE,
        "license_url": LICENSE_URL,
        "input_file": portable_input_file,
        "input_sha256": input_sha256,
        "source_rows": int(len(pd.read_csv(path, usecols=["datetime"]))),
        "source_first_timestamp_utc": raw.index.min().isoformat(),
        "source_last_timestamp_utc": raw.index.max().isoformat(),
        "yearly": yearly,
        "training_missingness": {
            "missing_half_hours": int(training_values.isna().sum()),
            "contiguous_run_count": int(len(missing_runs)),
            "largest_contiguous_run": missing_runs[0] if missing_runs else None,
        },
        "forecast_column_used": False,
        "forecast_exclusion_reason": (
            "The CSV has no forecast issue timestamp; treating its historical "
            "value as a fixed-horizon, as-issued forecast would therefore "
            "introduce temporal ambiguity."
        ),
    }
    return actual, quality


def calendar_features(target_time: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    slot = target_time.hour.to_numpy() * 2 + target_time.minute.to_numpy() / 30.0
    dow = target_time.dayofweek.to_numpy()
    # 365.2425 keeps the encoding smooth across leap years.
    doy = target_time.dayofyear.to_numpy() - 1
    return {
        "target_day_sin": np.sin(2 * np.pi * slot / 48.0),
        "target_day_cos": np.cos(2 * np.pi * slot / 48.0),
        "target_week_sin": np.sin(2 * np.pi * (dow * 48.0 + slot) / 336.0),
        "target_week_cos": np.cos(2 * np.pi * (dow * 48.0 + slot) / 336.0),
        "target_year_sin": np.sin(2 * np.pi * doy / 365.2425),
        "target_year_cos": np.cos(2 * np.pi * doy / 365.2425),
        "target_weekend": (dow >= 5).astype(float),
    }


def make_base_features(actual: pd.Series) -> pd.DataFrame:
    """Features shared by all horizons and available at decision time."""
    features: dict[str, pd.Series | np.ndarray] = {}
    for lag in (0, 1, 2, 3, 47, 48, 49, 95, 96, 97, 335, 336):
        features[f"actual_lag_{lag}"] = actual.shift(lag)
    for window in (6, 12, 48, 96, 336):
        past = actual.rolling(window=window, min_periods=window)
        features[f"rolling_mean_{window}"] = past.mean()
        features[f"rolling_std_{window}"] = past.std(ddof=0)
        if window in (12, 48):
            features[f"rolling_min_{window}"] = past.min()
            features[f"rolling_max_{window}"] = past.max()
    features["change_1"] = actual - actual.shift(1)
    features["change_2"] = actual - actual.shift(2)
    features["change_48"] = actual - actual.shift(48)
    return pd.DataFrame(features, index=actual.index, dtype=float)


def make_horizon_frame(
    actual: pd.Series, base: pd.DataFrame, horizon: int, climatology: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, dict[str, pd.Series]]:
    issue = actual.index
    target_time = issue + horizon * FREQ
    x = base.copy()
    for name, values in calendar_features(target_time).items():
        x[name] = values
    # Same target clock time on prior days. For h <= 48/336 these indices are
    # never later than the start timestamp of the most recently completed
    # settlement interval, which is asserted below.
    x["target_analog_1d"] = actual.shift(48 - horizon)
    x["target_analog_2d"] = actual.shift(96 - horizon)
    x["target_analog_7d"] = actual.shift(336 - horizon)
    assert horizon <= 48
    assert (48 - horizon) >= 0 and (336 - horizon) >= 0

    target = actual.shift(-horizon)
    how = target_time.dayofweek.to_numpy() * 48 + (
        target_time.hour.to_numpy() * 2 + target_time.minute.to_numpy() // 30
    )
    baselines = {
        "Persistence": actual,
        "DailySeasonal": actual.shift(48 - horizon),
        "WeeklySeasonal": actual.shift(336 - horizon),
        "Climatology": pd.Series(how, index=issue).map(climatology).astype(float),
    }
    return x, target, target_time, baselines


def build_model(spec: ModelSpec):
    if spec.name == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(**spec.parameters)
    elif spec.name == "RandomForest":
        estimator = RandomForestRegressor(**spec.parameters)
    else:
        raise KeyError(spec.name)
    # Imputer is fitted on training rows only. It also gives both algorithms an
    # identical, explicit missing-feature policy.
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)


def split_masks(target_time: pd.DatetimeIndex, target: pd.Series) -> dict[str, np.ndarray]:
    observed = target.notna().to_numpy()
    issue_time = target.index
    max_horizon = max(HORIZONS) * FREQ
    # Evaluation uses complete 24-hour origin cohorts wholly inside each split.
    # This purge prevents a validation forecast from being issued before the
    # training cutoff, or a test forecast before validation has finished.  It
    # also makes all 48 horizons temporally eligible at each retained origin;
    # source missingness can still remove individual targets in validation.
    return {
        "train": observed & (target_time >= TRAIN_START) & (target_time < TRAIN_END),
        "validation": (
            observed
            & (issue_time >= TRAIN_END)
            & (issue_time < VAL_END - max_horizon)
            & (target_time >= TRAIN_END)
            & (target_time < VAL_END)
        ),
        "test": (
            observed
            & (issue_time >= VAL_END)
            & (issue_time < TEST_END - max_horizon)
            & (target_time >= VAL_END)
            & (target_time < TEST_END)
        ),
    }


def select_non_ml_forecast_by_horizon(predictions: pd.DataFrame) -> dict[int, str]:
    """Choose one declared baseline per horizon using validation MAE only."""
    validation = predictions.loc[predictions["split"] == "validation"]
    selected: dict[int, str] = {}
    for horizon, frame in validation.groupby("horizon"):
        common = np.isfinite(frame[["actual", *BASELINES]].to_numpy()).all(axis=1)
        paired = frame.loc[common]
        scores = {
            method: float((paired[method] - paired["actual"]).abs().mean())
            for method in BASELINES
        }
        selected[int(horizon)] = min(scores, key=lambda method: (scores[method], method))
    if set(selected) != set(HORIZONS):
        raise ValueError("Could not select a non-ML forecast for every horizon")
    return selected


def add_validation_selected_baseline(
    predictions: pd.DataFrame, selection: dict[int, str]
) -> pd.DataFrame:
    """Materialize the validation-selected, horizon-specific baseline ensemble."""
    out = predictions.copy()
    values = np.full(len(out), np.nan)
    for horizon, method in selection.items():
        mask = out["horizon"] == horizon
        values[mask.to_numpy()] = out.loc[mask, method].to_numpy()
    out[SELECTED_BASELINE_COLUMN] = values
    return out


def evaluate_predictions(
    predictions: pd.DataFrame,
    split: str,
    methods: tuple[str, ...] | list[str] = ALL_METHODS,
) -> pd.DataFrame:
    rows = []
    frame = predictions.loc[predictions["split"] == split]
    for horizon, hdf in frame.groupby("horizon"):
        # A common paired sample prevents models benefiting from different rows.
        common = np.isfinite(hdf[["actual", *methods]].to_numpy()).all(axis=1)
        paired = hdf.loc[common]
        for method in methods:
            error = paired[method].to_numpy() - paired["actual"].to_numpy()
            rows.append(
                {
                    "split": split,
                    "scope": "horizon",
                    "horizon_half_hours": int(horizon),
                    "horizon_hours": horizon / 2.0,
                    "method": method,
                    "n": int(len(error)),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "bias": float(np.mean(error)),
                }
            )
    common = np.isfinite(frame[["actual", *methods]].to_numpy()).all(axis=1)
    paired = frame.loc[common]
    for method in methods:
        error = paired[method].to_numpy() - paired["actual"].to_numpy()
        rows.append(
            {
                "split": split,
                "scope": "aggregate",
                "horizon_half_hours": 0,
                "horizon_hours": 0.0,
                "method": method,
                "n": int(len(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
            }
        )
    out = pd.DataFrame(rows)
    # Compare each row with the daily-seasonal MAE at the identical scope and
    # horizon.  Using the aggregate denominator for horizon rows would distort
    # horizon-specific skill, particularly in validation when missingness
    # changes the paired cohort slightly.
    denominators = out.loc[
        out["method"] == "DailySeasonal",
        ["scope", "horizon_half_hours", "mae"],
    ].rename(columns={"mae": "daily_seasonal_mae"})
    out = out.merge(
        denominators,
        on=["scope", "horizon_half_hours"],
        how="left",
        validate="many_to_one",
    )
    out["mae_skill_vs_daily_seasonal"] = 1.0 - out["mae"] / out["daily_seasonal_mae"]
    out = out.drop(columns=["daily_seasonal_mae"])
    return out


def bootstrap_mean(
    values: np.ndarray, reps: int = 5000, block_days: int = 7
) -> tuple[float, float]:
    """Circular moving-block bootstrap CI for an ordered daily series.

    Seven days is the primary block length, chosen to retain day-to-day and
    weekly dependence. One-, 14-, 21- and 28-day results are reported as
    sensitivity checks. The longer blocks deliberately test whether a
    conclusion weakens under more persistent dependence. The circular
    construction gives every observation equal chance of appearing despite
    the short, finite later test interval.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    if block_days < 1 or block_days > len(values):
        raise ValueError("block_days must be between 1 and len(values)")
    rng = np.random.default_rng(SEED)
    n_blocks = math.ceil(len(values) / block_days)
    offsets = np.arange(block_days)
    estimates = np.empty(reps)
    for start in range(0, reps, 500):
        stop = min(reps, start + 500)
        block_starts = rng.integers(
            0, len(values), size=(stop - start, n_blocks, 1)
        )
        idx = (block_starts + offsets) % len(values)
        idx = idx.reshape(stop - start, -1)[:, : len(values)]
        estimates[start:stop] = values[idx].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def bootstrap_cis(values: np.ndarray, reps: int = 5000) -> dict[str, float]:
    out = {}
    for block_days in (1, 7, 14, 21, 28):
        low, high = bootstrap_mean(values, reps=reps, block_days=block_days)
        out[f"ci{block_days}d_low"] = float(low)
        out[f"ci{block_days}d_high"] = float(high)
    return out


def forecast_block_ci(
    predictions: pd.DataFrame, selected: str, comparator: str = "DailySeasonal"
) -> dict:
    test = predictions.loc[predictions["split"] == "test"].copy()
    common = np.isfinite(test[["actual", selected, comparator]].to_numpy()).all(axis=1)
    test = test.loc[common]
    test["selected_ae"] = (test[selected] - test["actual"]).abs()
    test["comparator_ae"] = (test[comparator] - test["actual"]).abs()
    # Issue-day blocks acknowledge serial dependence and the repeated targets
    # across horizons better than an IID half-hour bootstrap would.
    by_day = test.groupby(test["issue_time"].dt.floor("D"))[
        ["selected_ae", "comparator_ae"]
    ].mean()
    delta = by_day["comparator_ae"] - by_day["selected_ae"]
    cis = bootstrap_cis(delta.to_numpy())
    monthly = delta.groupby(delta.index.strftime("%Y-%m")).mean()
    return {
        "selected_model": selected,
        "comparator": comparator,
        "estimand": "paired reduction in MAE, comparator minus selected",
        "resampling_unit": "UTC issue day",
        "primary_bootstrap": "7-day circular moving blocks",
        "bootstrap_assumption": (
            "approximately stationary ordered daily contrasts within this "
            "fitted model and retrospective period; circular blocks wrap the "
            "December boundary to July"
        ),
        "sensitivity_bootstraps": [
            "1-day IID blocks",
            "14-day circular moving blocks",
            "21-day circular moving blocks",
            "28-day circular moving blocks",
        ],
        "n_days": int(len(delta)),
        "mean_reduction_gCO2_per_kWh": float(delta.mean()),
        "monthly_mean_reduction_gCO2_per_kWh": {
            str(month): float(value) for month, value in monthly.items()
        },
        **cis,
        "bootstrap_replicates": 5000,
    }


def contiguous_argmin(values: np.ndarray, duration_slots: int) -> int:
    totals = np.convolve(values, np.ones(duration_slots), mode="valid")
    return int(np.argmin(totals))


def simulate_scheduling(
    predictions: pd.DataFrame, split: str = "test"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every forecast policy on identical fixed-work scheduling tasks."""
    frame = predictions.loc[predictions["split"] == split].copy()
    # Exactly one decision per UTC day: at 18:00, immediately after the
    # 17:30--18:00 interval has completed. `issue_time` labels the last observed
    # interval start and is therefore 17:30.
    frame = frame.loc[
        (frame["issue_time"].dt.hour == 17)
        & (frame["issue_time"].dt.minute == 30)
    ]
    # Evaluating every declared forecast baseline is essential: forecast MAE
    # need not preserve the within-window ordering that drives the decision.
    strategies = list(ALL_METHODS)
    configs = [(12, 1), (12, 3), (12, 6), (6, 3), (24, 3)]
    daily_rows = []

    for issue_time, idf in frame.groupby("issue_time"):
        idf = idf.sort_values("horizon")
        for window_hours, duration_hours in configs:
            window_slots = window_hours * 2
            duration_slots = duration_hours * 2
            window = idf.loc[idf["horizon"].between(1, window_slots)]
            needed = ["actual", *strategies]
            if len(window) != window_slots or not np.isfinite(window[needed].to_numpy()).all():
                continue
            actual = window["actual"].to_numpy()
            starts = {"Immediate": 0, "Oracle": contiguous_argmin(actual, duration_slots)}
            for strategy in strategies:
                starts[strategy] = contiguous_argmin(
                    window[strategy].to_numpy(), duration_slots
                )
            for strategy, start in starts.items():
                # One kWh total, delivered uniformly: grams = mean g/kWh.
                grams = float(actual[start : start + duration_slots].mean())
                daily_rows.append(
                    {
                        "split": split,
                        "issue_time": issue_time,
                        "window_hours": window_hours,
                        "duration_hours": duration_hours,
                        "strategy": strategy,
                        "start_offset_hours": start / 2.0,
                        "emissions_g_for_1kWh_job": grams,
                    }
                )

    daily = pd.DataFrame(daily_rows)
    summary_rows = []
    for (window_hours, duration_hours), cdf in daily.groupby(
        ["window_hours", "duration_hours"]
    ):
        pivot = cdf.pivot(index="issue_time", columns="strategy", values="emissions_g_for_1kWh_job")
        delay = cdf.pivot(index="issue_time", columns="strategy", values="start_offset_hours")
        immediate = pivot["Immediate"]
        oracle = pivot["Oracle"]
        oracle_gain = float((immediate - oracle).mean())
        for strategy in pivot.columns:
            reduction = immediate - pivot[strategy]
            cis_immediate = bootstrap_cis(reduction.to_numpy())
            reduction_vs_daily = pivot["DailySeasonal"] - pivot[strategy]
            cis_daily = bootstrap_cis(reduction_vs_daily.to_numpy())
            captured = (
                float(reduction.mean() / oracle_gain) if oracle_gain > 0 else math.nan
            )
            summary_rows.append(
                {
                    "split": split,
                    "window_hours": int(window_hours),
                    "duration_hours": int(duration_hours),
                    "strategy": strategy,
                    "n_days": int(len(pivot)),
                    "mean_emissions_g_for_1kWh_job": float(pivot[strategy].mean()),
                    "mean_reduction_vs_immediate_g": float(reduction.mean()),
                    "reduction_vs_immediate_pct": float(
                        100 * reduction.mean() / immediate.mean()
                    ),
                    "reduction_vs_immediate_ci7d_low_g": cis_immediate["ci7d_low"],
                    "reduction_vs_immediate_ci7d_high_g": cis_immediate["ci7d_high"],
                    "reduction_vs_immediate_ci1d_low_g": cis_immediate["ci1d_low"],
                    "reduction_vs_immediate_ci1d_high_g": cis_immediate["ci1d_high"],
                    "reduction_vs_immediate_ci14d_low_g": cis_immediate["ci14d_low"],
                    "reduction_vs_immediate_ci14d_high_g": cis_immediate["ci14d_high"],
                    "mean_reduction_vs_daily_seasonal_g": float(reduction_vs_daily.mean()),
                    "reduction_vs_daily_seasonal_pct": float(
                        100 * reduction_vs_daily.mean() / pivot["DailySeasonal"].mean()
                    ),
                    "reduction_vs_daily_ci7d_low_g": cis_daily["ci7d_low"],
                    "reduction_vs_daily_ci7d_high_g": cis_daily["ci7d_high"],
                    "reduction_vs_daily_ci1d_low_g": cis_daily["ci1d_low"],
                    "reduction_vs_daily_ci1d_high_g": cis_daily["ci1d_high"],
                    "reduction_vs_daily_ci14d_low_g": cis_daily["ci14d_low"],
                    "reduction_vs_daily_ci14d_high_g": cis_daily["ci14d_high"],
                    "oracle_potential_captured": captured,
                    "mean_start_delay_hours": float(delay[strategy].mean()),
                    "median_start_delay_hours": float(delay[strategy].median()),
                    "p95_start_delay_hours": float(delay[strategy].quantile(0.95)),
                }
            )
    return daily, pd.DataFrame(summary_rows)


def select_non_ml_scheduler(validation_summary: pd.DataFrame) -> str:
    """Select the lowest-emission non-ML policy using validation data only."""
    candidates = validation_summary.loc[
        (validation_summary["window_hours"] == 12)
        & (validation_summary["duration_hours"] == 3)
        & validation_summary["strategy"].isin(BASELINES)
    ]
    if set(candidates["strategy"]) != set(BASELINES):
        raise ValueError("Validation scheduling summary is missing a baseline")
    return str(
        candidates.sort_values(
            ["mean_emissions_g_for_1kWh_job", "strategy"]
        ).iloc[0]["strategy"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "gb_carbon_intensity.csv",
    )
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results"
    )
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    started = time.time()

    actual, quality = load_data(args.data)
    base = make_base_features(actual)
    train_values = actual.loc[(actual.index >= TRAIN_START) & (actual.index < TRAIN_END)]
    train_how = train_values.index.dayofweek * 48 + (
        train_values.index.hour * 2 + train_values.index.minute // 30
    )
    climatology = train_values.groupby(train_how).mean()
    if len(climatology) != 336:
        raise ValueError("Training data do not cover all 336 half-hours of week")

    prediction_frames = []
    fit_counts = []
    for horizon in HORIZONS:
        x, target, target_time, baselines = make_horizon_frame(
            actual, base, horizon, climatology
        )
        masks = split_masks(target_time, target)
        train_mask = masks["train"]
        fitted_predictions: dict[str, dict[str, np.ndarray]] = {
            "validation": {},
            "test": {},
        }
        for spec in MODEL_SPECS:
            model = build_model(spec)
            model.fit(x.loc[train_mask], target.loc[train_mask])
            for split in ("validation", "test"):
                fitted_predictions[split][spec.name] = model.predict(x.loc[masks[split]])
            fit_counts.append(
                {
                    "horizon_half_hours": horizon,
                    "method": spec.name,
                    "train_rows": int(train_mask.sum()),
                    "validation_rows": int(masks["validation"].sum()),
                    "test_rows": int(masks["test"].sum()),
                }
            )
        for split in ("validation", "test"):
            mask = masks[split]
            frame = pd.DataFrame(
                {
                    "split": split,
                    # `issue_time` is retained for backward compatibility: it
                    # is the start of the most recently completed settlement
                    # period.  The operational decision is one interval later.
                    "issue_time": actual.index[mask],
                    "decision_time": actual.index[mask] + FREQ,
                    "target_time": target_time[mask],
                    "horizon": horizon,
                    "actual": target.loc[mask].to_numpy(),
                    **{
                        name: values.loc[mask].to_numpy()
                        for name, values in baselines.items()
                    },
                    **{
                        name: values
                        for name, values in fitted_predictions[split].items()
                    },
                }
            )
            prediction_frames.append(frame)
        print(f"completed horizon {horizon:02d}/48", flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    baseline_selection = select_non_ml_forecast_by_horizon(predictions)
    predictions = add_validation_selected_baseline(predictions, baseline_selection)
    evaluation_methods = (*ALL_METHODS, SELECTED_BASELINE_COLUMN)
    validation_metrics = evaluate_predictions(
        predictions, "validation", evaluation_methods
    )
    test_metrics = evaluate_predictions(predictions, "test", evaluation_methods)
    metrics = pd.concat([validation_metrics, test_metrics], ignore_index=True)

    val_aggregate = validation_metrics.loc[
        (validation_metrics["scope"] == "aggregate")
        & validation_metrics["method"].isin([s.name for s in MODEL_SPECS])
    ]
    selected = str(val_aggregate.sort_values(["mae", "method"]).iloc[0]["method"])
    ci = forecast_block_ci(
        predictions, selected, comparator=SELECTED_BASELINE_COLUMN
    )
    validation_schedule_daily, validation_schedule_summary = simulate_scheduling(
        predictions, split="validation"
    )
    selected_non_ml_scheduler = select_non_ml_scheduler(
        validation_schedule_summary
    )
    schedule_daily, schedule_summary = simulate_scheduling(predictions, split="test")

    metrics.to_csv(args.results / "forecast_metrics.csv", index=False)
    pd.DataFrame(fit_counts).to_csv(args.results / "sample_counts.csv", index=False)
    predictions.to_csv(
        args.results / "validation_test_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    schedule_daily.to_csv(args.results / "scheduling_daily.csv", index=False)
    schedule_summary.to_csv(args.results / "scheduling_summary.csv", index=False)
    validation_schedule_summary.to_csv(
        args.results / "scheduling_validation_summary.csv", index=False
    )
    with (args.results / "data_quality.json").open("w") as handle:
        json.dump(quality, handle, indent=2)

    manifest = {
        "seed": SEED,
        "time_basis": "UTC",
        "frequency_minutes": 30,
        "time_semantics": {
            "issue_time": (
                "Start timestamp of the most recently completed 30-minute "
                "settlement period; retained as a model-origin label."
            ),
            "decision_time": "issue_time + 30 minutes, when lag 0 is assumed available",
            "target_time": "Start timestamp of the settlement period being predicted",
            "horizon_hours": (
                "Lead from decision boundary to completion of the target period; "
                "lead to target-period start is 0.5 hours shorter."
            ),
        },
        "target": "NESO estimated actual national carbon intensity, gCO2/kWh",
        "data_provenance": {
            "source_url": SOURCE_URL,
            "dataset_page": DATASET_PAGE,
            "ckan_resource_id": CKAN_RESOURCE_ID,
            "input_sha256": quality["input_sha256"],
            "dataset_metadata_modified_utc": DATASET_METADATA_MODIFIED_UTC,
            "retrieved_at_utc": RETRIEVED_AT_UTC,
            "license_id": LICENSE_ID,
            "license_title": LICENSE_TITLE,
            "license_url": LICENSE_URL,
        },
        "target_windows": {
            "train": f"[{TRAIN_START.isoformat()}, {TRAIN_END.isoformat()})",
            "validation": f"[{TRAIN_END.isoformat()}, {VAL_END.isoformat()})",
            "test": f"[{VAL_END.isoformat()}, {TEST_END.isoformat()})",
        },
        "evaluation_origin_rule": (
            "Validation/test issue times must be at or after the split start "
            "and before split end minus 24 hours, making all 48 horizons "
            "temporally eligible and contained within each split."
        ),
        "horizons_half_hours": list(HORIZONS),
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "selected_on_validation_mae": selected,
        "validation_selected_baseline_by_horizon": {
            str(horizon): method
            for horizon, method in sorted(baseline_selection.items())
        },
        "selected_non_ml_scheduler_on_validation": selected_non_ml_scheduler,
        "forecast_block_bootstrap": ci,
        "scheduling_primary_configuration": {
            "decision_time": "18:00 UTC daily",
            "last_observation_interval_start": "17:30 UTC",
            "window_hours": 12,
            "job_duration_hours": 3,
            "job_energy_kWh": 1,
            "power_profile": "uniform and contiguous",
            "no_shift_reference": "start immediately at window opening",
            "primary_inferential_comparator": selected_non_ml_scheduler,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "code_provenance": {
            "fit_run_experiment_sha256": sha256_file(Path(__file__).resolve()),
            "postprocess_code_sha256_at_fit": sha256_file(
                Path(__file__).resolve().with_name("postprocess_results.py")
            ),
            "fit_predictions_sha256": sha256_file(
                (args.results / "validation_test_predictions.csv.gz").resolve()
            ),
            "sample_counts_sha256": sha256_file(
                (args.results / "sample_counts.csv").resolve()
            ),
        },
        "runtime_seconds": time.time() - started,
    }
    with (args.results / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps({"selected_model": selected, "forecast_ci": ci}, indent=2))
    print("\nTEST AGGREGATE")
    print(
        test_metrics.loc[test_metrics["scope"] == "aggregate", [
            "method", "n", "mae", "rmse", "bias", "mae_skill_vs_daily_seasonal"
        ]].sort_values("mae").to_string(index=False)
    )
    print("\nPRIMARY SCHEDULING (12 h window, 3 h job)")
    print(
        schedule_summary.loc[
            (schedule_summary["window_hours"] == 12)
            & (schedule_summary["duration_hours"] == 3)
        ].sort_values("mean_emissions_g_for_1kWh_job").to_string(index=False)
    )


if __name__ == "__main__":
    main()
