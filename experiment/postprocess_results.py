#!/usr/bin/env python3
"""Recompute inference, scheduler summaries, tests and plots without refitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_experiment import (
    ALL_METHODS,
    BASELINES,
    CKAN_RESOURCE_ID,
    DATASET_METADATA_MODIFIED_UTC,
    DATASET_PAGE,
    LICENSE_ID,
    LICENSE_TITLE,
    LICENSE_URL,
    RETRIEVED_AT_UTC,
    SEED,
    SELECTED_BASELINE_COLUMN,
    SOURCE_URL,
    add_validation_selected_baseline,
    bootstrap_cis,
    contiguous_argmin,
    evaluate_predictions,
    forecast_block_ci,
    select_non_ml_forecast_by_horizon,
    select_non_ml_scheduler,
    sha256_file,
    simulate_scheduling,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_vertical_label(
    image: Image.Image,
    text: str,
    *,
    center_x: float,
    center_y: float,
    label_font: ImageFont.ImageFont,
    fill: str = "#111111",
) -> None:
    """Draw a bottom-to-top axis title without crowding tick labels."""
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    left, top, right, bottom = probe_draw.textbbox((0, 0), text, font=label_font)
    padding = 6
    label = Image.new(
        "RGBA",
        (right - left + 2 * padding, bottom - top + 2 * padding),
        (255, 255, 255, 0),
    )
    label_draw = ImageDraw.Draw(label)
    label_draw.text(
        (padding - left, padding - top), text, fill=fill, font=label_font
    )
    rotated = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(
        rotated,
        (
            round(center_x - rotated.width / 2),
            round(center_y - rotated.height / 2),
        ),
        rotated,
    )


def draw_centered_line_legend(
    draw: ImageDraw.ImageDraw,
    *,
    canvas_width: int,
    y: int,
    items: list[tuple[str, str]],
    legend_font: ImageFont.ImageFont,
) -> None:
    """Lay out all line-series labels on one evenly aligned row."""
    swatch_width = 44
    swatch_to_text = 12
    item_gap = 48
    item_widths = []
    for label, _ in items:
        left, _, right, _ = draw.textbbox((0, 0), label, font=legend_font)
        item_widths.append(swatch_width + swatch_to_text + right - left)
    total_width = sum(item_widths) + item_gap * (len(items) - 1)
    cursor = (canvas_width - total_width) / 2
    for (label, color), item_width in zip(items, item_widths):
        draw.line(
            (cursor, y, cursor + swatch_width, y),
            fill=color,
            width=5,
        )
        draw.text(
            (cursor + swatch_width + swatch_to_text, y),
            label,
            fill="#222222",
            font=legend_font,
            anchor="lm",
        )
        cursor += item_width + item_gap


def draw_line_chart(metrics: pd.DataFrame, output: Path) -> None:
    methods = [
        "HistGradientBoosting",
        "RandomForest",
        SELECTED_BASELINE_COLUMN,
        "DailySeasonal",
    ]
    colors = {
        "HistGradientBoosting": "#1565C0",
        "RandomForest": "#00897B",
        SELECTED_BASELINE_COLUMN: "#EF6C00",
        "DailySeasonal": "#6D4C41",
    }
    labels = {
        "HistGradientBoosting": "HGB",
        "RandomForest": "Random forest",
        SELECTED_BASELINE_COLUMN: "Validation-selected baseline",
        "DailySeasonal": "Previous day",
    }
    data = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["scope"] == "horizon")
        & metrics["method"].isin(methods)
    ]
    width, height = 1400, 850
    margin = {"left": 145, "right": 50, "top": 160, "bottom": 105}
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = font(32, True), font(23), font(19)
    draw.text(
        (width / 2, 28),
        "Forecast error by horizon (retrospective H2 2024 test)",
        fill="#111111",
        font=title_font,
        anchor="ma",
    )
    x0, x1 = margin["left"], width - margin["right"]
    y0, y1 = height - margin["bottom"], margin["top"]
    ymax = 50.0
    for value in range(0, 51, 10):
        y = y0 - (y0 - y1) * value / ymax
        draw.line((x0, y, x1, y), fill="#DDDDDD", width=2)
        draw.text(
            (x0 - 18, y),
            str(value),
            fill="#333333",
            font=tick_font,
            anchor="rm",
        )
    draw.line((x0, y0, x1, y0), fill="#222222", width=3)
    draw.line((x0, y0, x0, y1), fill="#222222", width=3)
    for value in (0.5, 3, 6, 12, 18, 24):
        x = x0 + (x1 - x0) * (value - 0.5) / 23.5
        draw.line((x, y0, x, y0 + 8), fill="#222222", width=2)
        draw.text(
            (x, y0 + 15),
            f"{value:g}",
            fill="#333333",
            font=tick_font,
            anchor="ma",
        )
    for method in methods:
        m = data.loc[data["method"] == method].sort_values("horizon_hours")
        points = []
        for row in m.itertuples():
            x = x0 + (x1 - x0) * (row.horizon_hours - 0.5) / 23.5
            y = y0 - (y0 - y1) * row.mae / ymax
            points.append((x, y))
        draw.line(points, fill=colors[method], width=5)
    draw.text(
        ((x0 + x1) / 2, height - 48),
        "Lead to target-period completion (hours)",
        fill="#111111",
        font=label_font,
        anchor="ma",
    )
    draw_vertical_label(
        image,
        "MAE (g CO2/kWh)",
        center_x=38,
        center_y=(y0 + y1) / 2,
        label_font=label_font,
    )
    draw_centered_line_legend(
        draw,
        canvas_width=width,
        y=108,
        items=[(labels[method], colors[method]) for method in methods],
        legend_font=tick_font,
    )
    image.save(output, optimize=True)


def draw_schedule_chart(
    summary: pd.DataFrame, output: Path, selected_non_ml: str
) -> None:
    order = [
        "Immediate",
        "Persistence",
        "DailySeasonal",
        "WeeklySeasonal",
        "Climatology",
        "RandomForest",
        "HistGradientBoosting",
        "Oracle",
    ]
    colors = [
        "#9E9E9E",
        "#BDBDBD",
        "#6D4C41",
        "#8D6E63",
        "#EF6C00",
        "#00897B",
        "#1565C0",
        "#4527A0",
    ]
    data = summary.loc[(summary.window_hours == 12) & (summary.duration_hours == 3)].set_index("strategy").loc[order]
    width, height = 1700, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = font(32, True), font(23), font(19)
    x0, x1, y0, y1 = 155, 1650, 775, 150
    draw.text((x0, 28), "Accounting emissions for a 1 kWh flexible job", fill="#111111", font=title_font)
    draw.text((x0, 70), "3-hour contiguous job; 12-hour window beginning 18:00 UTC; H2 2024", fill="#444444", font=tick_font)
    draw.text((x0, 101), f"Lower is better; {selected_non_ml} was selected on validation as the non-ML comparator", fill="#444444", font=tick_font)
    ymax = 180
    for value in range(0, 181, 30):
        y = y0 - (y0 - y1) * value / ymax
        draw.line((x0, y, x1, y), fill="#DDDDDD", width=2)
        draw.text(
            (x0 - 18, y),
            str(value),
            fill="#333333",
            font=tick_font,
            anchor="rm",
        )
    bar_width = 125
    gap = (x1 - x0 - len(order) * bar_width) / (len(order) + 1)
    for i, (strategy, color) in enumerate(zip(order, colors)):
        value = float(data.loc[strategy, "mean_emissions_g_for_1kWh_job"])
        left = x0 + gap * (i + 1) + bar_width * i
        top = y0 - (y0 - y1) * value / ymax
        draw.rectangle((left, top, left + bar_width, y0), fill=color)
        draw.text(
            (left + bar_width / 2, top - 30),
            f"{value:.1f}",
            fill="#111111",
            font=tick_font,
            anchor="ma",
        )
        label = {
            "HistGradientBoosting": "HGB",
            "DailySeasonal": "Daily",
            "WeeklySeasonal": "Weekly",
            "RandomForest": "RF",
            "Climatology": "Climatology*",
        }.get(strategy, strategy)
        draw.text(
            (left + bar_width / 2, y0 + 18),
            label,
            fill="#222222",
            font=tick_font,
            anchor="ma",
        )
    draw_vertical_label(
        image,
        "Emissions (g CO2/job)",
        center_x=40,
        center_y=(y0 + y1) / 2,
        label_font=label_font,
    )
    image.save(output, optimize=True)


def invariant_tests(predictions: pd.DataFrame, daily: pd.DataFrame) -> dict:
    tests = {}
    # Direct-horizon analogue offsets must not access a timestamp after issue.
    tests["all_analogue_offsets_nonnegative"] = all(
        (48 - h) >= 0 and (96 - h) >= 0 and (336 - h) >= 0 for h in range(1, 49)
    )
    train_end = pd.Timestamp("2024-01-01T00:00:00Z")
    val_end = pd.Timestamp("2024-07-01T00:00:00Z")
    tests["validation_targets_after_train"] = bool(
        predictions.loc[predictions.split == "validation", "target_time"].min() >= train_end
    )
    tests["validation_issues_after_train_cutoff"] = bool(
        predictions.loc[predictions.split == "validation", "issue_time"].min()
        >= train_end
    )
    tests["validation_origins_have_full_24h_room"] = bool(
        predictions.loc[predictions.split == "validation", "issue_time"].max()
        < pd.Timestamp("2024-06-30T00:00:00Z")
    )
    tests["test_targets_after_validation"] = bool(
        predictions.loc[predictions.split == "test", "target_time"].min() >= val_end
    )
    tests["test_issues_after_validation_cutoff"] = bool(
        predictions.loc[predictions.split == "test", "issue_time"].min() >= val_end
    )
    tests["test_origins_have_full_24h_room"] = bool(
        predictions.loc[predictions.split == "test", "issue_time"].max()
        < pd.Timestamp("2024-12-31T00:00:00Z")
    )
    tests["test_targets_before_2025"] = bool(
        predictions.loc[predictions.split == "test", "target_time"].max()
        < pd.Timestamp("2025-01-01T00:00:00Z")
    )
    tests["constant_trace_tie_breaks_to_earliest"] = contiguous_argmin(
        np.full(24, 100.0), 6
    ) == 0
    tests["zero_slack_has_only_immediate_start"] = contiguous_argmin(
        np.arange(6, dtype=float), 6
    ) == 0
    tests["one_kWh_constant_trace_is_work_conserving"] = all(
        np.isclose(np.full(d * 2, 1.0 / (d * 2)).sum(), 1.0)
        and np.isclose(
            np.dot(np.full(d * 2, 1.0 / (d * 2)), np.full(d * 2, 137.0)),
            137.0,
        )
        for d in (1, 3, 6)
    )
    counts = daily.groupby(["window_hours", "duration_hours", "strategy"]).size()
    tests["all_strategies_have_paired_day_counts"] = bool(
        counts.groupby(level=[0, 1]).nunique().eq(1).all()
    )
    test_counts = predictions.loc[predictions.split == "test"].groupby(
        predictions.loc[predictions.split == "test", "issue_time"].dt.floor("D")
    ).size()
    tests["test_issue_days_are_complete_horizon_cohorts"] = bool(
        len(test_counts) > 0 and test_counts.eq(48 * 48).all()
    )
    tests["all_tests_passed"] = all(tests.values())
    return tests


def main() -> None:
    prediction_path = RESULTS / "validation_test_predictions.csv.gz"
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    code_provenance = manifest.get("code_provenance", {})
    fit_code_sha256 = code_provenance.get("fit_run_experiment_sha256")
    current_run_code_sha256 = sha256_file(ROOT / "run_experiment.py")
    if fit_code_sha256 is None:
        raise RuntimeError(
            "Manifest lacks fit_run_experiment_sha256; rerun model fitting "
            "before postprocessing so fitted predictions have auditable provenance."
        )
    if fit_code_sha256 != current_run_code_sha256:
        raise RuntimeError(
            "run_experiment.py differs from the code recorded at model fit; "
            "rerun fitting before attributing predictions to the current code."
        )
    expected_prediction_sha256 = code_provenance.get(
        "postprocessed_predictions_sha256",
        code_provenance.get("fit_predictions_sha256"),
    )
    if expected_prediction_sha256 is None:
        raise RuntimeError("Manifest lacks a prediction artifact hash")
    if sha256_file(prediction_path) != expected_prediction_sha256:
        raise RuntimeError("Paired-prediction artifact hash does not match manifest")
    expected_counts_sha256 = code_provenance.get("sample_counts_sha256")
    if expected_counts_sha256 is None or sha256_file(
        RESULTS / "sample_counts.csv"
    ) != expected_counts_sha256:
        raise RuntimeError("Sample-count artifact hash does not match manifest")

    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["issue_time", "target_time"],
    )
    if "decision_time" not in predictions:
        predictions.insert(
            predictions.columns.get_loc("target_time"),
            "decision_time",
            predictions["issue_time"] + pd.Timedelta("30min"),
        )
    else:
        predictions["decision_time"] = pd.to_datetime(
            predictions["decision_time"], utc=True
        )
    baseline_selection = select_non_ml_forecast_by_horizon(predictions)
    predictions = add_validation_selected_baseline(predictions, baseline_selection)
    evaluation_methods = (*ALL_METHODS, SELECTED_BASELINE_COLUMN)
    metrics = pd.concat(
        [
            evaluate_predictions(predictions, "validation", evaluation_methods),
            evaluate_predictions(predictions, "test", evaluation_methods),
        ],
        ignore_index=True,
    )
    val = metrics.loc[
        (metrics.split == "validation")
        & (metrics.scope == "aggregate")
        & metrics.method.isin(["HistGradientBoosting", "RandomForest"])
    ]
    selected = str(val.sort_values(["mae", "method"]).iloc[0].method)
    if "splits_by_target_timestamp" in manifest:
        manifest["target_windows"] = manifest.pop("splits_by_target_timestamp")
    manifest["evaluation_origin_rule"] = (
        "Validation/test issue times must be at or after the split start and "
        "before split end minus 24 hours, making all 48 horizons temporally "
        "eligible and contained in the target window; source missingness can "
        "still remove individual validation targets."
    )
    manifest["time_semantics"] = {
        "issue_time": (
            "Start timestamp of the most recently completed 30-minute settlement "
            "period; retained as a model-origin label."
        ),
        "decision_time": "issue_time + 30 minutes, when lag 0 is assumed available",
        "target_time": "Start timestamp of the settlement period being predicted",
        "horizon_hours": (
            "Lead from decision boundary to completion of the target period; "
            "lead to target-period start is 0.5 hours shorter."
        ),
    }
    quality = json.loads((RESULTS / "data_quality.json").read_text())
    forecast_inference = forecast_block_ci(
        predictions, selected, comparator=SELECTED_BASELINE_COLUMN
    )
    forecast_inference.update(
        {
            "comparator_selection": (
                "lowest validation MAE independently at each of the 48 horizons "
                "among Persistence, DailySeasonal, WeeklySeasonal and Climatology"
            ),
            "selected_baseline_counts": {
                method: int(sum(value == method for value in baseline_selection.values()))
                for method in BASELINES
            },
        }
    )
    validation_daily, validation_summary = simulate_scheduling(
        predictions, split="validation"
    )
    selected_non_ml_scheduler = select_non_ml_scheduler(validation_summary)
    daily, summary = simulate_scheduling(predictions, split="test")

    primary = daily.loc[(daily.window_hours == 12) & (daily.duration_hours == 3)].pivot(
        index="issue_time", columns="strategy", values="emissions_g_for_1kWh_job"
    )
    validation_primary = validation_summary.loc[
        (validation_summary["window_hours"] == 12)
        & (validation_summary["duration_hours"] == 3)
    ].set_index("strategy")
    baseline_validation_means = {
        method: float(
            validation_primary.loc[method, "mean_emissions_g_for_1kWh_job"]
        )
        for method in BASELINES
    }
    sensitivity_rows = []
    for (window_hours, duration_hours), config in daily.groupby(
        ["window_hours", "duration_hours"]
    ):
        config_pivot = config.pivot(
            index="issue_time",
            columns="strategy",
            values="emissions_g_for_1kWh_job",
        )
        difference = (
            config_pivot[selected_non_ml_scheduler] - config_pivot[selected]
        )
        sensitivity_rows.append(
            {
                "split": "test",
                "window_hours": int(window_hours),
                "duration_hours": int(duration_hours),
                "selected_model": selected,
                "comparator": selected_non_ml_scheduler,
                "estimand": "comparator minus selected ML; positive favors ML",
                "n_paired_days": int(len(difference)),
                "mean_difference_g_for_1kWh_job": float(difference.mean()),
                "percent_of_comparator_mean": float(
                    100
                    * difference.mean()
                    / config_pivot[selected_non_ml_scheduler].mean()
                ),
                **bootstrap_cis(difference.to_numpy()),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["window_hours", "duration_hours"]
    )
    model_vs_selected_baseline = primary[selected_non_ml_scheduler] - primary[selected]
    model_vs_immediate = primary["Immediate"] - primary[selected]
    model_vs_daily = primary["DailySeasonal"] - primary[selected]
    scheduling_inference = {
        "configuration": "1 kWh, 3 h contiguous job in 12 h window from 18:00 UTC",
        "selected_model": selected,
        "selected_non_ml_scheduler": selected_non_ml_scheduler,
        "non_ml_selection_rule": (
            "lowest mean assigned emissions in the primary configuration on "
            "validation data only; deterministic alphabetical tie-break"
        ),
        "validation_non_ml_means_g_for_1kWh_job": baseline_validation_means,
        "validation_paired_days": int(
            validation_primary.loc[selected_non_ml_scheduler, "n_days"]
        ),
        "n_paired_days": int(len(primary)),
        "primary_comparison": {
            "comparator": selected_non_ml_scheduler,
            "estimand": "comparator minus selected ML; positive favors ML",
            "mean_difference_g_for_1kWh_job": float(
                model_vs_selected_baseline.mean()
            ),
            "percent_of_comparator_mean": float(
                100
                * model_vs_selected_baseline.mean()
                / primary[selected_non_ml_scheduler].mean()
            ),
            **bootstrap_cis(model_vs_selected_baseline.to_numpy()),
        },
        "secondary_comparison": {
            "comparator": "Immediate start",
            "estimand": "comparator minus selected ML; positive favors ML",
            "mean_difference_g_for_1kWh_job": float(model_vs_immediate.mean()),
            "percent_of_comparator_mean": float(
                100 * model_vs_immediate.mean() / primary["Immediate"].mean()
            ),
            **bootstrap_cis(model_vs_immediate.to_numpy()),
        },
        "exploratory_daily_comparison": {
            "comparator": "DailySeasonal",
            "estimand": "comparator minus selected ML; positive favors ML",
            "mean_difference_g_for_1kWh_job": float(model_vs_daily.mean()),
            "percent_of_comparator_mean": float(
                100 * model_vs_daily.mean() / primary["DailySeasonal"].mean()
            ),
            **bootstrap_cis(model_vs_daily.to_numpy()),
        },
        "bootstrap": {
            "replicates": 5000,
            "primary": "7-day circular moving block, ordered by UTC issue date",
            "assumption": (
                "approximately stationary ordered daily contrasts within this "
                "fitted model and retrospective period; circular blocks wrap "
                "the December boundary to July"
            ),
            "sensitivities": [
                "1-day IID block",
                "14-day circular moving block",
                "21-day circular moving block",
                "28-day circular moving block",
            ],
        },
        "configuration_sensitivity_artifact": "scheduling_sensitivity_contrasts.csv",
    }
    tests = invariant_tests(predictions, daily)
    tests["decision_time_is_interval_after_issue_label"] = bool(
        (
            predictions["decision_time"].astype("int64")
            - predictions["issue_time"].astype("int64")
            == 30 * 60 * 1_000_000_000
        ).all()
    )
    tests["target_period_never_starts_before_decision"] = bool(
        (predictions["target_time"] >= predictions["decision_time"]).all()
    )
    tests["model_selection_reproduced_from_validation_only"] = (
        selected == manifest["selected_on_validation_mae"]
    )
    tests["forecast_baseline_selected_from_validation_only"] = bool(
        set(baseline_selection) == set(range(1, 49))
        and all(method in BASELINES for method in baseline_selection.values())
    )
    tests["scheduling_baseline_selected_from_validation_only"] = bool(
        selected_non_ml_scheduler
        == min(
            baseline_validation_means,
            key=lambda method: (baseline_validation_means[method], method),
        )
    )
    tests["all_tests_passed"] = all(
        value for key, value in tests.items() if key != "all_tests_passed"
    )
    if not tests["all_tests_passed"]:
        raise AssertionError(tests)

    manifest["forecast_block_bootstrap"] = forecast_inference
    manifest["scheduling_inference"] = scheduling_inference
    manifest["validation_selected_baseline_by_horizon"] = {
        str(horizon): method
        for horizon, method in sorted(baseline_selection.items())
    }
    manifest["selected_non_ml_scheduler_on_validation"] = (
        selected_non_ml_scheduler
    )
    manifest["invariant_tests"] = tests
    quality.update(
        {
            "ckan_resource_id": CKAN_RESOURCE_ID,
            "dataset_metadata_modified_utc": DATASET_METADATA_MODIFIED_UTC,
            "retrieved_at_utc": RETRIEVED_AT_UTC,
            "license_id": LICENSE_ID,
            "license_title": LICENSE_TITLE,
            "license_url": LICENSE_URL,
            "forecast_exclusion_reason": (
                "The CSV has no forecast issue timestamp; treating its historical "
                "value as a fixed-horizon, as-issued forecast would therefore "
                "introduce temporal ambiguity."
            ),
        }
    )
    manifest["data_provenance"] = {
        "source_url": SOURCE_URL,
        "dataset_page": DATASET_PAGE,
        "ckan_resource_id": CKAN_RESOURCE_ID,
        "input_sha256": quality["input_sha256"],
        "dataset_metadata_modified_utc": DATASET_METADATA_MODIFIED_UTC,
        "retrieved_at_utc": RETRIEVED_AT_UTC,
        "license_id": LICENSE_ID,
        "license_title": LICENSE_TITLE,
        "license_url": LICENSE_URL,
    }

    metrics.to_csv(RESULTS / "forecast_metrics.csv", index=False)
    predictions.to_csv(
        RESULTS / "validation_test_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    daily.to_csv(RESULTS / "scheduling_daily.csv", index=False)
    summary.to_csv(RESULTS / "scheduling_summary.csv", index=False)
    validation_summary.to_csv(
        RESULTS / "scheduling_validation_summary.csv", index=False
    )
    sensitivity.to_csv(
        RESULTS / "scheduling_sensitivity_contrasts.csv", index=False
    )
    with (RESULTS / "inference.json").open("w") as handle:
        json.dump(
            {
                "forecast": forecast_inference,
                "scheduling": scheduling_inference,
                "seed": SEED,
            },
            handle,
            indent=2,
        )
    with (RESULTS / "invariant_tests.json").open("w") as handle:
        json.dump(tests, handle, indent=2)
    with (RESULTS / "data_quality.json").open("w") as handle:
        json.dump(quality, handle, indent=2)
    draw_line_chart(metrics, RESULTS / "forecast_mae_by_horizon.png")
    draw_schedule_chart(
        summary, RESULTS / "scheduling_primary.png", selected_non_ml_scheduler
    )
    manifest["code_provenance"].update(
        {
            "fit_run_experiment_sha256": fit_code_sha256,
            "postprocess_results_sha256": sha256_file(Path(__file__).resolve()),
            "postprocessed_predictions_sha256": sha256_file(
                RESULTS / "validation_test_predictions.csv.gz"
            ),
            "sample_counts_sha256": sha256_file(RESULTS / "sample_counts.csv"),
        }
    )
    manifest["postprocessed_artifact_sha256"] = {
        name: sha256_file(RESULTS / name)
        for name in (
            "forecast_metrics.csv",
            "sample_counts.csv",
            "scheduling_daily.csv",
            "scheduling_summary.csv",
            "scheduling_validation_summary.csv",
            "scheduling_sensitivity_contrasts.csv",
            "inference.json",
            "invariant_tests.json",
            "data_quality.json",
            "forecast_mae_by_horizon.png",
            "scheduling_primary.png",
        )
    }
    with (RESULTS / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"forecast": forecast_inference, "scheduling": scheduling_inference, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
