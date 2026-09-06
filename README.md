# Forecast accuracy and decision value in carbon-aware scheduling

## Reproducibility supplement

This archive supports two alternative, journal-formatted versions of the same empirical study:

1. **Applied Energy:** “Forecast accuracy does not guarantee carbon-aware scheduling value: A Great Britain trace replay.”
2. **IEEE Transactions on Sustainable Computing:** “Forecast Accuracy Versus Decision Value in Carbon-Aware Computing: A Reproducible Trace Replay.”

The versions are alternatives and must not be submitted concurrently. This supplement contains the common evidence and computation; it intentionally excludes obsolete or duplicate manuscript copies.

- **Author:** Aditya Shrivastava
- **Affiliation:** Independent Researcher
- **Contact:** adityashrivastava2003@gmail.com
- **Public repository:** <https://github.com/adityashriv2003/ai-climate-action-research>

A persistent archive DOI should be added only after it exists.

## Central empirical result

On a retrospective July–December 2024 Great Britain trace, validation-selected histogram gradient boosting had lower period-average forecast mean absolute error than a horizon-specific validation-selected non-machine-learning benchmark: 30.66 versus 36.09 g CO2/kWh. The paired point difference was 5.43 g CO2/kWh, with a seven-day moving-block 95% interval of 1.37–9.14; the 21- and 28-day block intervals crossed zero.

Lower forecast error did not provide a reliable advantage in the primary scheduling decision. For a fixed 1 kWh, three-hour job with a 12-hour completion window, training-only time-of-week climatology was assigned 100.94 g CO2/job and the learned policy 102.17 g CO2/job. The signed climatology-minus-learned difference was −1.22 g/job, with a seven-day interval of −2.71 to 0.16. All test-period inference is post hoc.

These are average-intensity-weighted, attributional operational estimates under a historical trace. They are not measurements of generator response, causal avoided emissions, or lifecycle net benefit.

## Contents

- `experiment/`: pinned NESO input data, exact requirements, fit and postprocessing code, methods and results notes, paired predictions, metrics, uncertainty results, invariant tests, figures, and provenance manifest.
- `literature/`: structured review protocol and authoritative source matrix.
- `bibliography/`: verified BibTeX and CSV records.
- `figures/`: the two journal figures at 2800 pixels wide and 400 dpi.
- `manuscripts/`: editable and rendered copies of the two alternative journal manuscripts, plus Applied Energy highlights and declaration files.
- `DATA_AND_LICENCE.md`: source provenance and NESO licence information.
- `SHA256SUMS.txt`: SHA-256 hashes for every other file in this archive.

## Reproduce the experiment

Use Python 3.12 from the archive root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r experiment/requirements.txt
python experiment/run_experiment.py
python experiment/postprocess_results.py
```

The fit takes about 320 seconds on the recorded Apple-silicon environment. The fitting script refuses an input whose SHA-256 differs from the pinned official file. Postprocessing verifies fit-time code, paired-prediction, and sample-count hashes before consuming the outputs. All 19 invariant checks must pass.

## Research and reporting boundaries

The workflow uses chronological forecast-origin cohorts, training-only preprocessing, direct horizon-specific models, validation-only executable model and comparator selection, identical work and feasibility constraints across schedulers, paired evaluation, and moving-block resampling. The protocol and complete comparator set were refined after test inspection, so all test-period inference is expressly post hoc.

The archived NESO finalized estimates are treated as available at the next half-hour boundary. Publication latency and later revisions are not reconstructed. The carbon outcome uses average operational intensity and does not establish a marginal, dispatch, causal, or lifecycle effect.

**Funding:** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Competing interests:** The author declares no competing financial or non-financial interests.

The named human author must personally verify the complete analysis and sources, approve the submitted version, and accept accountability for the work.

This package has not undergone external peer review.
