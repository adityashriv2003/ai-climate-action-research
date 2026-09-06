<!--
IEEE TRANSACTIONS ON SUSTAINABLE COMPUTING PRODUCTION PLAN

Article type: Short Paper.
Target format: Current IEEE journal template, US Letter, double column. The IEEE
Computer Society defines a double-column page as 7.875 x 10.75 in with 9.5-point
type and 11.5-point leading. Format this TSUSC Short Paper at exactly eight pages,
including abstract, figures, tables, disclosures, and references. Do not add an
author biography.

Page budget after typesetting:
  p. 1       title, author, first footnote, 50-word abstract, Index Terms,
             and most of Sec. I
  pp. 1–2    Sec. II (related work and the accuracy/decision-value gap)
  pp. 2–4    Sec. III (data, models, scheduling, estimands, leakage controls)
  pp. 4–5.5  Sec. IV, Table I, Table II, and Figs. 1–2
  pp. 5.5–6.5 Secs. V–VI (interpretation, accounting boundary, limitations)
  p. 6.5     Sec. VII, data/code statement, disclosures, acknowledgment
  pp. 6.5–8  references

Layout controls:
  * Insert both figures at one-column width near their first callouts. If either
    legend becomes unreadable, combine them as two panels across both columns;
    do not enlarge the manuscript past eight pages.
  * Keep Table I to one column. Set Table II across both columns if necessary.
  * Move implementation detail, complete horizon tables, all configuration
    sensitivities, the source matrix, and invariant-test output to a separately
    named supplemental file; do not place an appendix in the eight-page paper.
  * Use IEEE numbered citations in order of first appearance and preserve the
    reference order below.
  * The reproducibility repository is publicly available at the GitHub URL in
    the data statement. Do not add a DOI unless one is actually minted.
  * The author supplied “Independent Researcher” and an e-mail address but no
    geographic address. Obtain the author's country (and city/postal code if
    used) before final submission; do not infer or fabricate those details.
-->

# Forecast Accuracy Versus Decision Value in Carbon-Aware Computing: A Reproducible Trace Replay

**Aditya Shrivastava**

*First footnote—Aditya Shrivastava is an Independent Researcher (e-mail: adityashrivastava2003@gmail.com). The author received no external funding for this work. Corresponding author: Aditya Shrivastava.*

## Abstract

Carbon-aware schedulers may use learned forecasts, but better prediction need not improve decisions. In a Great Britain trace replay, gradient boosting lowered average forecast error, yet did not outperform validation-selected climatology for a fixed flexible workload. Most modeled benefit came from delay tolerance and calendar regularity, not machine learning alone.

**Index Terms—** Carbon-aware computing, carbon intensity forecasting, machine learning, sustainable computing, workload scheduling, time-series evaluation.

## I. INTRODUCTION

Artificial intelligence can help climate action through sensing, prediction, optimization, scientific discovery, and decision support [1], [2]. Its climate value, however, is not an intrinsic property of a model. A prediction must alter an authorized decision, that decision must change a physical system, and the resulting effect must remain favorable after the computational footprint and system response are counted. The distinction is especially important in sustainable computing, where an improvement in model loss is sometimes reported as if it were already an emissions reduction.

Carbon-aware computing provides a bounded test of this distinction. A temporally flexible workload can be moved toward electricity intervals with a lower carbon-intensity signal [3]–[6]. Prior work demonstrates substantial technical potential, but realized benefit varies across grids, workloads, signals, and constraints [7]–[10]. Simple seasonal rules can exploit recurring grid structure; average and marginal carbon signals can recommend different actions [10]–[13]; and synchronized adoption can invalidate the assumption that the trace is unaffected by the scheduler. Consequently, the relevant question is not merely whether a learned forecast is accurate, but whether it improves a functionally identical scheduling decision over the strongest simple policy available with the same information.

This paper reports a reproducible, chronological trace replay of Great Britain electricity carbon intensity. It asks: **Does a learned forecast add scheduling value beyond workload flexibility and validation-selected non-machine-learning rules?** The study makes four contributions.

1. It separates forecast accuracy, workload flexibility, incremental learning value, and unattainable perfect foresight within one common decision problem.
2. It enforces an explicit pseudo-real-time information boundary and validation-only selection of the model family and non-machine-learning comparators.
3. It reports paired moving-block uncertainty and dependence sensitivities instead of treating correlated half-hours as independent observations.
4. It limits the empirical claim to an average-intensity-weighted operational accounting quantity; it does not relabel that quantity as causal or lifecycle emissions reduction.

The main result is deliberately negative. Histogram gradient boosting improved period-average forecast error, but it did not reliably improve the primary scheduling outcome over training-only time-of-week climatology. This finding is useful for sustainable systems design: flexibility and a transparent rule may provide nearly all attainable value in a regime where an additional learned service adds complexity and an unmeasured footprint.

## II. BACKGROUND AND EVALUATION PRINCIPLE

### A. Carbon-Aware Computing

Temporal workload shifting delays flexible computation until a lower-carbon interval; spatial shifting changes execution region; elasticity can change the rate or amount of allocated computing. Trace studies and production-oriented systems show that these mechanisms can reduce an assigned operational footprint under suitable signals [4]–[9]. CarbonCast used learned multi-day grid-carbon forecasts [6], while production datacenter work demonstrated the feasibility of integrating carbon signals into resource management [7]. CarbonScaler considered workload elasticity rather than timing alone [8].

The theoretical opportunity is not the same as incremental machine-learning value. The cleanest decomposition compares: (i) immediate execution with an oracle to quantify available flexibility; (ii) simple feasible schedulers with the oracle to determine how much structure they capture; and (iii) a learned scheduler with the best validation-selected simple scheduler to isolate the added value of learning. Comparing a learned scheduler only with immediate execution credits the model for flexibility that exists without it.

### B. From Accuracy to Climate Outcome

We use the causal chain

**model output → scheduling decision → workload execution → electricity-system response → greenhouse-gas outcome.**

This experiment directly evaluates only the first three elements and assigns an operational carbon value using historical average intensity. The estimate is therefore attributional: it allocates part of a historical grid total to a fixed-energy job. The short-run causal effect of changing load is more closely related to marginal generation, and results can change with the chosen factor [10]–[14]. At scale, dispatch, price, and investment responses also become endogenous.

The same boundary applies to the costs of artificial intelligence. Development, recurrent inference, networking, cooling, and embodied hardware can be material [15]–[18]. Because model energy was not separately instrumented here, this study can test decision utility but cannot establish a net lifecycle benefit.

## III. METHODS

### A. Data, Splits, and Availability

We use the National Energy System Operator (NESO) national Carbon Intensity dataset for Great Britain [19], [20]. It reports half-hourly forecasts and “actual” intensity in grams of carbon dioxide per kilowatt-hour. The latter is an operational estimate derived from metered generation, not a direct stack measurement or a lifecycle carbon-dioxide-equivalent factor [19]. The file was retrieved on 5 September 2026 at 21:42 UTC; its SHA-256 digest is `428476b2bab40d690f38309dd6eebe6e276631d570071785e5311f7ef45b6283`.

The archived forecast column has no issue timestamp from which a fixed as-issued horizon can be reconstructed. We therefore discard it and forecast only the finalized actual series. Training targets span 1 January 2022–31 December 2023, validation targets 1 January–30 June 2024, and test targets 1 July–31 December 2024. All times are normalized to UTC. A complete 30-min grid makes absent records explicit: 33,452 of 35,040 training targets, 8,705 of 8,736 validation targets, and all 8,832 test targets are observed. Of the 1,588 missing training slots, 1,441 form one contiguous gap from 6 March to 5 April 2023.

The model-origin label is the start of the most recently completed settlement interval. Under the stated pseudo-real-time assumption, that interval's finalized value is available at the next boundary, 30 min later. For each learned family, 48 direct horizon-specific estimators predict leads from 0.5 to 24 h measured from that decision boundary to completion of the target interval; the lead to target-interval start is 0–23.5 h. Publication latency and subsequent revisions are not modeled. Evaluation admits only origins for which all horizons fall inside the relevant chronological split.

This origin-cohort design is stricter than retaining every target inside a split and follows the principle that forecasting procedures should be assessed on later, temporally valid observations [21], [22]. Each admitted origin has all 48 targets in the same partition, so short- and long-horizon errors share a common decision cohort. The July–December test period contributes 8,784 complete origins rather than all 8,832 target slots. The final 48 slots are not missing observations; they are excluded because a complete 24-h forecast surface would extend beyond the test boundary. The rule also prevents a training or validation origin from borrowing a target across a split boundary.

The 2023 gap is not independent point missingness. We preserve it on the regular grid and exclude unavailable targets rather than compressing time. Lagged and rolling features therefore retain their calendar meaning: a one-day lag always denotes 48 settlement intervals even around missing runs. Median feature imputation is fitted only on training rows. This avoids turning the month-long outage into a false temporal adjacency, while leaving its possible effect on fitted relations as an explicit validity threat.

### B. Forecasts and Leakage Controls

Four non-machine-learning forecasts use the same available history: last-value persistence, the previous day's slot, the previous week's slot, and a half-hour-of-week mean fitted on training targets. Two fixed learned configurations are considered: histogram gradient boosting and random forest. The former uses squared-error loss, learning rate 0.06, 160 boosting iterations, 31 maximum leaf nodes, 30 minimum samples per leaf, and L2 regularization 1.0. The latter uses 96 trees, maximum depth 18, minimum leaf size 3, and 0.75 feature subsampling. Both use seed 20240906. Aggregate validation mean absolute error selects the learned family: histogram gradient boosting scores 34.10 g/kWh and random forest 34.86 g/kWh.

Features contain the most recently completed intensity and older lags; trailing means, dispersion, extrema, and changes; target-slot analogues from previous days and the previous week; cyclical time-of-day, time-of-week, and time-of-year terms; and a weekend indicator. All analogue offsets point backward from the origin. Median imputation is fitted on training rows only. Each horizon is fitted independently, and no prediction is fed into a later model.

For the forecast comparison, validation mean absolute error selects the lowest-error non-machine-learning rule independently at each horizon. Persistence is selected at 31 horizons and the previous-day rule at 17; an exact validation tie at 24 h is resolved by a fixed alphabetical rule. Model-family and comparator selection use no test labels. Nonetheless, the complete comparator set and dependence analysis were refined during reproducibility audit after initial test inspection. Every test-period interval is therefore post hoc and awaits independent replication.

Selection is intentionally separated by task. Aggregate validation mean absolute error chooses the learned family, horizon-specific validation error chooses the forecast comparator, and mean assigned validation carbon chooses the scheduling comparator. The last choice cannot be inferred from global forecast error: a scheduler needs the relative ordering of feasible blocks inside one release window, whereas mean absolute error scores every origin and horizon. Keeping these targets separate tests whether predictive improvement transfers to the downstream decision.

### C. Scheduling Task and Policies

The primary task releases one non-preemptive job at 18:00 UTC on each eligible day. The job consumes exactly 1 kWh uniformly during three contiguous hours and must complete within a 12-h window. Each policy sees the same feasible starts and completes identical work. For a candidate block, a policy averages its forecast over the occupied slots and chooses the minimum; ties select the earliest start.

The policies are immediate execution; persistence; previous-day and previous-week seasonal rules; training-only time-of-week climatology; histogram gradient boosting; random forest; and an oracle that uses realized future intensity. The oracle is an unattainable lower bound. On 178 paired validation days, climatology has the lowest non-machine-learning mean, 108.62 g/job, and is fixed as the primary test comparator. The test outcome for policy (p) on day (d) is

\[
C_{p,d}=\sum_{t \in B_{p,d}} e_t I_t,
\]

where B(p,d) is the selected contiguous block, e(t) is allocated job energy with a total of 1 kWh, and I(t) is realized NESO average intensity. This is an assigned operational quantity, not observed generator response.

Sensitivity configurations use 6- and 24-h windows and 1-, 3-, and 6-h job durations. All comparisons remain paired by day.

The scheduler is a trace-replay decision rule, not a dispatch optimizer. It cannot change job energy, split the job, violate the deadline, or use future realized intensity except in the labeled oracle. Persistence produces a constant forecast over feasible starts and therefore reduces to immediate execution under the fixed earliest-start tie rule. These restrictions make learned and simple policies functionally equivalent and conserve useful work. They also isolate temporal selection from elasticity, spatial migration, checkpointing, and admission control, which require different estimands.

### D. Estimands and Uncertainty

Forecast evaluation uses mean absolute error, root mean squared error, and bias over 8,784 complete test origins and 48 horizons, yielding 421,632 paired instances. The forecast estimand is daily paired mean absolute error for the validation-selected composite minus that of the selected learned model; positive values favor learning.

The scheduling estimand is climatology minus the learned policy in grams per identical 1 kWh job; positive values favor learning. We also report differences from immediate execution and oracle opportunity captured. The observed 183-day means are deterministic descriptions of the trace.

Uncertainty uses 5,000 seven-day circular moving-block bootstrap replicates over ordered UTC issue days, following the block-resampling rationale for dependent observations [23]. Circular blocks wrap the December boundary to July and assume approximately stationary daily contrasts within this fitted model and period. One-, 14-, 21-, and 28-day blocks probe dependence sensitivity. These conditional intervals neither create an ex-ante confirmatory test nor establish generality to other grids or years. Configuration sensitivities are exploratory and are not multiplicity adjusted.

**TABLE I**  
**Dependence Sensitivity of the Two Primary Paired Contrasts**

| Contrast (positive favors HGB) | Estimate | 1-day interval | 7-day interval | 14-day interval | 21-day interval | 28-day interval |
|---|---:|---:|---:|---:|---:|---:|
| Forecast: composite − HGB (g/kWh) | +5.43 | [3.09, 7.80] | [1.37, 9.14] | [0.41, 9.75] | [−0.24, 10.01] | [−0.42, 10.20] |
| Scheduling: climatology − HGB (g/job) | −1.22 | [−2.44, −0.05] | [−2.71, 0.16] | [−2.90, 0.20] | [−2.97, 0.18] | [−3.00, 0.26] |

*Note: Intervals are percentile intervals from 5,000 paired daily resamples. Seven-day circular moving blocks are primary; other block lengths assess dependence sensitivity. All test-period inference is post hoc.*

### E. Decision-Matched Evaluation

Three nested comparisons prevent attribution errors. Immediate execution versus the oracle measures the trace-specific opportunity created jointly by delay tolerance and future knowledge. Immediate execution versus a feasible policy measures the combined value of flexibility and that policy. The learned policy versus validation-selected climatology isolates the incremental contribution of learning under identical work and constraints. Only the third comparison can support a claim that learning adds scheduling value. Reporting the 32.77% learned-versus-immediate reduction without the climatology comparison would therefore overstate the evidence.

The same logic applies to model overhead. A learned service adds training, tuning, deployment, monitoring, and recurring inference burdens. If its job-level point estimate is already worse than a transparent rule before those burdens are counted, nonnegative overhead cannot reverse the primary ordering at any deployment volume. Conversely, a positive operational difference is not automatically a net benefit; it must first exceed the additional burdens under a common system boundary.

### F. Reproducibility Checks

The executable suite verifies disjoint chronological splits, non-future analogue offsets, complete horizon cohorts, identical paired-day counts, energy conservation, deterministic earliest-slot ties, equivalence of zero slack and immediate execution, oracle dominance under the stated objective, and exact reproduction of all validation selections. All 19 recorded invariants pass.

The supplement preserves the retrieved source file and SHA-256 digest, split and model manifest, exact seed and configurations, paired validation and test predictions, per-horizon metrics, daily scheduling outcomes, bootstrap summaries, and figure data. Hashes bind the executable scripts and generated artifacts. The invariant suite tests properties rather than only headline numbers; for example, it verifies that every analogue points backward from the decision origin and that every policy consumes the same 1 kWh. This makes silent leakage or work mismatch easier to detect in an independent rerun.

## IV. RESULTS

### A. Forecast Accuracy

Table II summarizes aggregate test performance. Histogram gradient boosting has mean absolute error 30.66 g/kWh, compared with 36.09 for the horizon-specific validation-selected composite. The paired period-average reduction is 5.43 g/kWh, or 15.05%. Its seven-day interval is [1.37, 9.14]. The result is dependence-sensitive: the one- and 14-day intervals remain above zero, whereas the 21-day [−0.24, 10.01] and 28-day [−0.42, 10.20] intervals cross zero. Monthly paired reductions also vary, from −10.87 g/kWh in August to +14.54 in October.

**TABLE II**  
**Aggregate Forecast Performance on the July–December 2024 Test Trace**

| Forecast | MAE | RMSE | Bias | Skill vs. composite |
|---|---:|---:|---:|---:|
| Histogram gradient boosting | 30.66 | 40.11 | +11.07 | +15.05% |
| Random forest | 32.34 | 41.96 | +12.04 | +10.39% |
| Validation-selected composite | 36.09 | 48.16 | +0.11 | Reference |
| Persistence | 37.72 | 50.20 | +0.09 | −4.51% |
| Previous day | 40.29 | 52.59 | +0.18 | −11.63% |

*Note: Error and bias units are g CO₂/kWh. Positive bias denotes overprediction. There are 421,632 paired horizon instances. Skill is relative mean absolute error reduction; the composite uses validation selection independently at each horizon. All inference is post hoc.*

The learned advantage diminishes with lead time (Fig. 1). Histogram gradient boosting mean absolute error is 3.57 g/kWh at 0.5 h, 14.78 at 3 h, 24.52 at 6 h, 34.24 at 12 h, and 40.64 at 24 h. At 24 h it is slightly worse than the previous-day rule at 40.25. Aggregate bias is +11.07 g/kWh, consistent with lower test-period intensity than in training. Thus even the selected learned model is neither uniformly superior across horizons nor well calibrated in level.

[FIGURE:../../experiment/results/forecast_mae_by_horizon.png|Fig. 1. Mean absolute forecast error by horizon on the retrospective July–December 2024 test trace. The non-machine-learning composite is selected independently at each horizon using validation only.]

### B. Scheduling Decision Value

Table III gives the primary scheduling outcome. Immediate execution is assigned 151.97 g/job and the perfect-information oracle 96.52 g/job, defining 55.45 g/job of modeled flexibility opportunity. Climatology is assigned 100.94 g/job and captures 92.02% of that opportunity. The learned histogram-gradient-boosting policy is assigned 102.17 g/job and captures 89.81%.

**TABLE III**  
**Primary Scheduling Results: 1 kWh, 3 h Contiguous Job, 12 h Window**

| Policy | Mean g/job | Reduction vs. immediate | Oracle opportunity captured | Mean delay (h) |
|---|---:|---:|---:|---:|
| Immediate | 151.97 | 0.00 (0.00%) | 0.00% | 0.00 |
| Previous day | 104.74 | 47.23 (31.08%) | 85.17% | 5.90 |
| Previous week | 104.60 | 47.37 (31.17%) | 85.43% | 5.90 |
| Training climatology* | **100.94** | **51.03 (33.58%)** | **92.02%** | 6.42 |
| Random forest | 103.12 | 48.86 (32.15%) | 88.10% | 5.12 |
| Histogram gradient boosting | 102.17 | 49.81 (32.77%) | 89.81% | 5.27 |
| Oracle | 96.52 | 55.45 (36.49%) | 100.00% | 5.89 |

*Note: N = 183 paired UTC days. Climatology was selected among non-machine-learning policies on validation only. The oracle has future information and is unattainable. Persistence equals immediate execution because its constant forecast triggers the earliest-start tie rule. All test-period inference is post hoc.*

The primary paired contrast, climatology minus histogram gradient boosting, is −1.22 g/job (−1.21% of the climatology mean); its seven-day interval is [−2.71, 0.16]. Hence this backtest provides no reliable evidence that the learned policy improves the primary outcome over the strongest validation-selected simple rule. By contrast, comparing the learned policy only with immediate execution would attribute a 32.77% reduction to a system whose non-machine-learning comparator performs slightly better. Fig. 2 exposes this attribution error.

[FIGURE:../../experiment/results/scheduling_primary.png|Fig. 2. Average-intensity-weighted operational CO₂ assigned by the primary scheduler. Lower is better. Climatology is the validation-selected non-machine-learning comparator; the oracle uses realized future intensity.]

### C. Sensitivity and Diagnostics

Decision value changes with the feasible set. Signed climatology-minus-learned differences are −0.43 g/job for a 3-h job in a 6-h window, −1.39 for a 1-h job in the 12-h window, −0.93 for a 6-h job in that window, and +7.00 for a 3-h job in a 24-h window. Seven-day intervals are [−1.14, 0.04], [−3.62, 0.65], [−1.89, −0.03], and [1.91, 12.29], respectively. These are post-hoc, unadjusted contrasts. Moreover, the 6-h-duration and 24-h-window intervals cross zero with longer blocks. They show heterogeneity, not confirmatory evidence for choosing one policy by configuration.

The simple comparator's strong scheduling performance despite poor global level accuracy is not contradictory. Scheduling depends chiefly on ranking candidate blocks inside one evening window, whereas aggregate mean absolute error penalizes level errors across all horizons and origins. An objective optimized for the latter can improve without changing—or can even worsen—the former.

Table IV makes the configuration dependence explicit. Four of five point estimates favor climatology; only the 24-h window favors HGB. The six-hour-duration interval is below zero at the seven-day block but crosses zero at longer blocks. Conversely, the positive 24-h-window result crosses zero at 21- and 28-day blocks. These reversals show why selecting a favorable window or duration after inspecting the test would not support a general policy claim.

**TABLE IV**  
**Scheduling Sensitivity: Climatology Minus HGB**

| Window (h) | Duration (h) | Difference (g/job) | 7-day interval | 21-day interval | 28-day interval |
|---:|---:|---:|---:|---:|---:|
| 6 | 3 | −0.43 | [−1.14, 0.04] | [−1.20, 0.06] | [−1.19, 0.05] |
| 12 | 1 | −1.39 | [−3.62, 0.65] | [−3.70, 0.60] | [−3.82, 0.69] |
| 12 | 3 | −1.22 | [−2.71, 0.16] | [−2.97, 0.18] | [−3.00, 0.26] |
| 12 | 6 | −0.93 | [−1.89, −0.03] | [−2.28, 0.21] | [−2.27, 0.25] |
| 24 | 3 | +7.00 | [1.91, 12.29] | [−0.42, 14.61] | [−1.17, 15.36] |

*Note: Positive values favor HGB. Each row contains 183 paired test days. Configuration contrasts are post hoc and unadjusted for multiplicity.*

### D. Delay and Temporal Instability

The carbon-only objective hides a service trade-off. HGB delayed the primary job by a mean of 5.27 h, a median of 5.0 h, and a 95th percentile of 9.0 h. Climatology delayed it by 6.42, 6.5, and 9.0 h, respectively. Thus HGB ran 1.15 h earlier on average but received a 1.22 g/job worse carbon assignment. Neither policy violates the 12-h completion window, yet an operator with a nonzero delay cost could prefer the learned policy for service reasons. That would be a computing-utility decision, not evidence of a carbon advantage. A production trial should therefore define and preregister the joint service-carbon objective rather than choosing it after seeing outcomes.

Forecast differences also vary sharply by month (Table V). HGB is worse than the validation-selected benchmark in August but better in each other test month. The mean advantage in the second half of the test period is materially larger than in the first half, which is consistent with regime variation rather than a stable additive improvement. Aggregate error alone masks this behavior; monitoring should retain horizon- and month-resolved diagnostics, recalibration triggers, and a rule-based fallback.

**TABLE V**  
**Monthly Forecast MAE Difference: Composite Minus HGB**

| Month in 2024 | July | August | September | October | November | December |
|---|---:|---:|---:|---:|---:|---:|
| Difference (g/kWh) | +7.32 | −10.87 | +6.73 | +14.54 | +9.20 | +5.85 |

*Note: Positive values favor HGB. Monthly values are descriptive components of the same post-hoc test period; they are not six independent confirmatory tests.*

## V. DISCUSSION

### A. What the Trace Replay Establishes

Three conclusions follow within the stated period. First, the learned model has a lower period-average forecast-error point estimate than a validation-selected composite, although longer dependence blocks weaken the uncertainty conclusion. Second, forecast improvement does not establish downstream value: the primary interval includes zero, and its point estimate favors climatology. Third, most of the apparent reduction from immediate execution comes from delay tolerance plus regular overnight grid structure, not from machine learning. A transparent training-only rule captures more oracle opportunity than either learned policy.

The null incremental result is operationally consequential. A production model introduces development, monitoring, fallback, security, and recurring inference costs. If a simple rule is functionally equivalent or better, deploying the learned component lacks empirical justification in this regime. The result does not imply that forecasts are never useful. The exploratory 24-h window favors learning in its point estimate, and archived weather, demand, outage, or generation forecasts could add skill. Rather, learned scheduling must demonstrate decision-weighted value under the actual constraints in which it will run.

### B. Carbon and System Boundary

The experiment reaches an attributed operational estimate in trace replay, not a consequential or lifecycle outcome. A causal claim would require a verified marginal or dispatch signal; measured workload power, capacity, and service behavior; measurement of training and inference overhead; and an endogenous grid model when adoption is large enough to alter the signal. Average intensity answers how historical operational emissions are allocated, not which generator changes output because the job moves.

For deployment count \(N\), a minimal break-even expression is

\[
S(N)=N(C_{b}-C_{m,\mathrm{run}})-C_{\mathrm{train+tune}}-\Delta C_{\mathrm{emb}},
\]

where C_baseline is the per-job value for a functionally equivalent baseline, C_model,run includes both the scheduled job and recurring model-service overhead, and the remaining terms cover one-time training/tuning and incremental embodied impacts under the same boundary. In the primary point estimates, the climatology job value is already below the learned-policy job value before model overhead is counted. Nonnegative overhead therefore cannot produce a positive break-even at any deployment volume under this contrast.

### C. Design Implications

Carbon-aware systems should be evaluated in the following order. First, hold useful work, deadlines, and available information constant. Second, quantify the no-flexibility-to-oracle opportunity. Third, compare learned policies with validation-selected simple rules, not only with immediate execution. Fourth, optimize and report the decision outcome alongside forecast loss. Fifth, stress-test dependence, drift, signal choice, and adoption scale. Finally, measure the computing and hardware burden before claiming a net climate benefit.

These principles generalize beyond workload scheduling. In building control, grid operations, and climate-risk decisions, a better proxy is valuable only when it selects a better feasible action, and that action is valuable only if it changes the relevant physical outcome.

### D. An Operational Decision Gate

The primary evidence supports a simple deployment gate. First deploy the transparent climatology as the reference policy, instrument actual workload and facility electricity, and retain a safe immediate-execution fallback. A learned challenger should run in shadow mode on the same release stream. Promotion should require a preregistered decision-weighted improvement over the reference on a future period, stable performance across seasons and dependence assumptions, and a positive margin after measured model-service overhead. A fallback threshold should also cover stale inputs, missing forecasts, distribution shift, and infeasible recommendations.

This gate separates model engineering from climate accounting. Engineering telemetry can establish whether the learned service changes start times, meets deadlines, and consumes equal useful-work energy. Electricity-system analysis must separately establish whether the selected signal represents the intended claim. Average intensity is adequate for allocating a historical operational footprint; marginal or dispatch evidence is needed for short-run causal language, and lifecycle inventory is needed for a net greenhouse-gas claim. Passing the scheduling gate is therefore necessary but not sufficient for calling the system climate beneficial.

## VI. LIMITATIONS AND VALIDITY THREATS

**Internal validity:** The archived actual for the just-completed half-hour is assumed immediately available, and later corrections are not modeled. The source forecast is excluded because issue timestamps are absent. The final comparator set and dependence checks were refined after test inspection; all inference is post hoc despite validation-only executable selection. The one-month training-data gap may affect fitted relations.

**Construct validity:** The assigned outcome uses national average operational carbon dioxide. It excludes marginal dispatch, imports or upstream effects not represented in that factor, lifecycle greenhouse gases, computing overhead, and rebound. Global mean absolute error is not aligned exactly with within-window ranking utility.

**External validity:** The test covers one country, six months, one daily release time, and a synthetic fixed-energy job. It omits server utilization, power-usage effectiveness, queueing, capacity, preemption, network transfer, price, and service penalties. Independent jobs are assumed too small to alter the grid; synchronized or large-scale deployment may create new peaks.

**Statistical conclusion validity:** Moving-block intervals are conditional on an approximately stationary sequence of daily contrasts. Results vary by month and block length. The test period is not an untouched confirmatory sample, and configuration contrasts are not multiplicity adjusted. Independent multi-year and multi-region replication is required.

**Reproducibility validity:** The executable package can recreate this retrospective analysis, but it cannot recreate information that the archival source does not retain. Finalized “actual” values substitute for the last observed operational input under an immediate-availability assumption, and the source forecast lacks an issue timestamp. Exact computational reproduction should therefore not be confused with reconstruction of the historical production information set. A prospective replication should archive every input at its observed publication time.

## VII. CONCLUSION

On a later Great Britain trace, validation-selected histogram gradient boosting reduced aggregate mean absolute forecast error by 15.05% relative to a strong horizon-specific non-machine-learning composite. It nevertheless did not reliably improve the primary scheduling outcome over validation-selected time-of-week climatology: the simple policy was assigned 100.94 g/job and the learned policy 102.17 g/job, with a seven-day paired interval that crossed zero. Delay tolerance and calendar structure created most of the modeled opportunity.

The sustainable-computing lesson is methodological. Forecast accuracy, scheduling utility, and climate impact are distinct claims. A credible carbon-aware system must show incremental decision value over a functionally equivalent simple baseline, maintain an ex-ante information boundary, survive dependence and regime sensitivities, and account for its own footprint. The present results satisfy the first two stages of that evaluation but do not demonstrate causal generator response or net lifecycle savings.

## DATA AND CODE AVAILABILITY

The submission is accompanied by a reproducibility package containing the NESO source file and checksum, data provenance, chronological split manifest, executable forecasting and scheduling code, fixed configurations and seed, paired outputs, all figure data, 19 invariant tests, and regeneration instructions. The reproducibility repository is publicly available at `https://github.com/adityashriv2003/ai-climate-action-research`. The source data are redistributed under the NESO Open Data Licence identified in the package. Complete horizon tables, block-length and configuration sensitivities, and the structured literature-review materials are supplementary rather than part of the eight-page article.

## ACKNOWLEDGMENT AND DISCLOSURES

The study uses public aggregate electricity data and involves no human participants, animals, or personal data. The author declares no competing financial or nonfinancial interests.

An OpenAI GPT-5-based coding assistant supported literature discovery, code authoring, analysis checking, figure generation, and drafting throughout Sections I–VII and the supplemental materials. Numerical results were generated by the archived executable code and programmatically cross-checked against the archived outputs; source metadata and cited claims were checked against DOI records or official publications. The author accepts responsibility for the final content.

## REFERENCES

[1] L. H. Kaack, P. L. Donti, E. Strubell, G. Kamiya, F. Creutzig, and D. Rolnick, “Aligning artificial intelligence with climate change mitigation,” *Nature Climate Change*, vol. 12, no. 6, pp. 518–527, 2022, doi: 10.1038/s41558-022-01377-7.

[2] D. Rolnick *et al*., “Tackling climate change with machine learning,” *ACM Computing Surveys*, vol. 55, no. 2, Art. no. 42, pp. 1–96, 2022, doi: 10.1145/3485128.

[3] P. L. Donti and J. Z. Kolter, “Machine learning for sustainable energy systems,” *Annual Review of Environment and Resources*, vol. 46, pp. 719–747, 2021, doi: 10.1146/annurev-environ-020220-061831.

[4] P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, and L. Thamsen, “Let’s wait awhile: How temporal workload shifting can reduce carbon emissions in the cloud,” in *Proc. 22nd Int. Middleware Conf.*, 2021, pp. 260–272, doi: 10.1145/3464298.3493399.

[5] J. Dodge *et al*., “Measuring the carbon intensity of AI in cloud instances,” in *Proc. 2022 ACM Conf. Fairness, Accountability, and Transparency*, 2022, pp. 1877–1894, doi: 10.1145/3531146.3533234.

[6] D. Maji, P. Shenoy, and R. K. Sitaraman, “CarbonCast: Multi-day forecasting of grid carbon intensity,” in *Proc. 9th ACM Int. Conf. Systems for Energy-Efficient Buildings, Cities, and Transportation*, 2022, pp. 198–207, doi: 10.1145/3563357.3564079.

[7] A. Radovanović *et al*., “Carbon-aware computing for datacenters,” *IEEE Transactions on Power Systems*, vol. 38, no. 2, pp. 1270–1280, Mar. 2023, doi: 10.1109/TPWRS.2022.3173250.

[8] W. A. Hanafy, Q. Liang, N. Bashir, D. Irwin, and P. Shenoy, “CarbonScaler: Leveraging cloud workload elasticity for optimizing carbon-efficiency,” *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, vol. 7, no. 3, Art. no. 57, pp. 1–28, 2023, doi: 10.1145/3626788.

[9] T. Sukprasert, A. Souza, N. Bashir, D. Irwin, and P. Shenoy, “On the limitations of carbon-aware temporal and spatial workload shifting in the cloud,” in *Proc. 19th European Conf. Computer Systems*, 2024, pp. 924–941, doi: 10.1145/3627703.3650079.

[10] T. Sukprasert, N. Bashir, A. Souza, D. Irwin, and P. Shenoy, “On the implications of choosing average versus marginal carbon intensity signals on carbon-aware optimizations,” in *Proc. 15th ACM Int. Conf. Future and Sustainable Energy Systems*, 2024, pp. 422–427, doi: 10.1145/3632775.3661953.

[11] A. D. Hawkes, “Estimating marginal CO2 emissions rates for national electricity systems,” *Energy Policy*, vol. 38, no. 10, pp. 5977–5987, 2010, doi: 10.1016/j.enpol.2010.05.053.

[12] K. Siler-Evans, I. L. Azevedo, and M. G. Morgan, “Marginal emissions factors for the U.S. electricity system,” *Environmental Science & Technology*, vol. 46, no. 9, pp. 4742–4748, 2012, doi: 10.1021/es300145v.

[13] P. L. Donti, J. Z. Kolter, and I. L. Azevedo, “How much are we saving after all? Characterizing the effects of commonly varying assumptions on emissions and damage estimates in PJM,” *Environmental Science & Technology*, vol. 53, no. 16, pp. 9905–9914, 2019, doi: 10.1021/acs.est.8b06586.

[14] B. Tranberg, O. Corradi, B. Lajoie, T. Gibon, I. Staffell, and G. B. Andresen, “Real-time carbon accounting method for the European electricity markets,” *Energy Strategy Reviews*, vol. 26, Art. no. 100367, 2019, doi: 10.1016/j.esr.2019.100367.

[15] P. Henderson, J. Hu, J. Romoff, E. Brunskill, D. Jurafsky, and J. Pineau, “Towards the systematic reporting of the energy and carbon footprints of machine learning,” *Journal of Machine Learning Research*, vol. 21, no. 248, pp. 1–43, 2020.

[16] L. Lannelongue, J. Grealey, and M. Inouye, “Green Algorithms: Quantifying the carbon footprint of computation,” *Advanced Science*, vol. 8, no. 12, Art. no. 2100707, 2021, doi: 10.1002/advs.202100707.

[17] A. S. Luccioni, S. Viguier, and A.-L. Ligozat, “Estimating the carbon footprint of BLOOM, a 176B parameter language model,” *Journal of Machine Learning Research*, vol. 24, no. 253, pp. 1–15, 2023.

[18] A. S. Luccioni, Y. Jernite, and E. Strubell, “Power hungry processing: Watts driving the cost of AI deployment?” in *Proc. 2024 ACM Conf. Fairness, Accountability, and Transparency*, 2024, pp. 85–99, doi: 10.1145/3630106.3658542.

[19] A. R. W. Bruce, L. Ruff, J. Kelloway, F. MacMillan, and A. Rogers, *Carbon Intensity Forecast Methodology*. National Energy System Operator, 2024. [Online]. Available: https://www.neso.energy/data-portal/national-carbon-intensity-forecast/national_carbon_intensity_forecast_methodology

[20] National Energy System Operator, “National Carbon Intensity Forecast,” NESO Data Portal, resource 0e5fde43-2de7-4fb4-833d-c7bca3b658b0, 2026. [Online]. Available: https://www.neso.energy/data-portal/national-carbon-intensity-forecast/national_carbon_intensity_forecast. Accessed: Sep. 6, 2026.

[21] L. J. Tashman, “Out-of-sample tests of forecasting accuracy: An analysis and review,” *International Journal of Forecasting*, vol. 16, no. 4, pp. 437–450, 2000, doi: 10.1016/S0169-2070(00)00065-0.

[22] V. Cerqueira, L. Torgo, and I. Mozetič, “Evaluating time series forecasting models: An empirical study on performance estimation methods,” *Machine Learning*, vol. 109, pp. 1997–2028, 2020, doi: 10.1007/s10994-020-05910-7.

[23] H. R. Künsch, “The jackknife and the bootstrap for general stationary observations,” *The Annals of Statistics*, vol. 17, no. 3, pp. 1217–1241, 1989, doi: 10.1214/aos/1176347265.
