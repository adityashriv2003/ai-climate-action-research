# Forecast accuracy does not guarantee carbon-aware scheduling value: A Great Britain trace replay

Aditya Shrivastava\(^{a,*}\)

\(^{a}\) Independent Researcher  
\(^*\) Corresponding author: adityashrivastava2003@gmail.com

## Abstract

Artificial intelligence can support energy-system decarbonization only when better predictions change feasible decisions and produce net physical benefits. We test that condition in carbon-aware computing using a reproducible pseudo-real-time trace replay of Great Britain's half-hourly electricity carbon intensity. Direct models were trained for each of 48 horizons on 2022–2023 data, selected on January–June 2024 validation data, and evaluated on July–December 2024. Across 421,632 paired forecast instances, validation-selected histogram gradient boosting achieved a mean absolute error of 30.66 g CO₂/kWh versus 36.09 for a horizon-specific non-machine-learning benchmark, a 15.05% period-average improvement (seven-day moving-block 95% confidence interval for the 5.43 g CO₂/kWh difference: [1.37, 9.14]). However, lower error did not improve the primary scheduling outcome. Across 183 daily decisions for a fixed 1 kWh, three-hour job with a 12-hour completion window, a training-only time-of-week climatology was assigned 100.94 g CO₂/job versus 102.17 for the learned policy. The climatology-minus-learned difference was −1.22 g CO₂/job (95% confidence interval: [−2.71, 0.16]), providing no reliable evidence of incremental machine-learning value. The climatology captured 92.02% of the perfect-information opportunity. All test-period inference is post hoc. Results are attributional operational estimates under a historical trace, not measured avoided emissions. Decision-weighted evaluation and competitive simple baselines are therefore necessary before crediting artificial intelligence with energy or climate benefit.

**Keywords:** carbon-aware computing; carbon-intensity forecasting; machine learning; demand flexibility; electricity; emissions accounting

## 1. Introduction

Artificial intelligence (AI) can reduce information and control frictions in electricity systems, buildings, transport, industry, and climate-risk management. It cannot itself replace clean generation, transmission, electrification, efficiency investment, or effective institutions. The relevant scientific question is therefore not whether a model is accurate, but whether its incremental information changes an authorized decision, produces a physical response, and improves a well-defined greenhouse-gas or risk outcome. The Intergovernmental Panel on Climate Change treats digital technologies as enabling tools whose net effects depend on governance and whether demand growth offsets efficiency gains [1]. Reviews of AI and climate mitigation similarly distinguish the footprint of computing, the direct effect of an application, and wider system effects [2,3]. Rapid growth in data-centre electricity demand makes that distinction increasingly material [4].

Carbon-aware computing is a useful test case because the full forecast-to-decision chain can be reproduced. A delay-tolerant workload may be moved to a time with lower grid carbon intensity [5,6], but temporal flexibility is not created by machine learning (ML), and a simple calendar rule may exploit it. The attainable benefit also depends on the feasible window, workload constraints, electricity signal, and synchronized adoption [7]. Most importantly, multiplying electricity use by average carbon intensity yields an attributional footprint; it does not establish the short-run generator response associated with marginal emissions [8–10].

Prior carbon-intensity forecasting studies generally emphasize prediction error, while carbon-aware systems studies often compare optimization policies with immediate execution [5–7]. That comparison conflates the value of flexibility with the incremental value of AI. We instead ask: (i) does a learned forecast outperform strong non-ML forecasts under chronological evaluation; and (ii) does the selected learned forecast improve a fixed downstream scheduling outcome relative to a non-ML scheduler selected on validation data? The contribution is a chronologically controlled, pseudo-real-time trace replay that separates four components: flexibility, simple forecasting, learned forecasting, and perfect foresight. It also reports paired temporal uncertainty, dependence sensitivity, and an explicit carbon-accounting boundary.

## 2. Materials and methods

### 2.1. Design and estimands

The retrospective workflow used chronological training, validation, and test partitions consistent with out-of-sample forecasting guidance [11,12]. Training data fitted features, imputers, climatology, and model parameters. Validation data selected one of two fixed ML families, the forecast comparator independently at each horizon, and the primary non-ML scheduling comparator. Test labels were used to calculate the reported comparisons, while the executable selection rules used training and validation data. Nevertheless, the complete baseline set and protocol were refined after initial inspection of test results; all test-period inference is therefore explicitly post hoc and requires confirmation on an untouched period or region.

The forecast estimand was the paired reduction in absolute error, validation-selected non-ML comparator minus selected ML model, averaged over common issue-time–horizon pairs. The scheduling estimand was the paired difference in average-intensity-weighted operational CO₂ per functionally identical completed job, validation-selected non-ML policy minus selected ML policy; positive values favor ML. Secondary contrasts measured the value of flexibility against immediate execution and regret relative to a perfect-information oracle.

### 2.2. Data and temporal availability

The National Energy System Operator (NESO) Carbon Intensity dataset supplies national half-hourly forecasts and finalized “actual” estimates for Great Britain [13]. The latter are estimates based on metered generation, not direct stack measurements; the national methodology covers operational CO₂ intensity rather than lifecycle CO₂-equivalent emissions [14]. The file was retrieved on 5 September 2026 at 21:42 UTC (SHA-256: `428476b2bab40d690f38309dd6eebe6e276631d570071785e5311f7ef45b6283`).

The archived CSV does not provide the issue timestamp required to reconstruct an as-issued fixed-horizon forecast. We therefore discarded its forecast field and used finalized actual estimates only as targets. Timestamps were normalized to UTC and reindexed to a complete 30-minute grid so that missing records could not silently alter positional lags. The experiment assumes that the finalized estimate for the just-completed interval is available at the next settlement boundary; publication latency and later revisions are not modeled.

**Table 1. Chronological data partitions.**

| Role | Target interval (UTC) | Expected slots | Observed targets | Missing |
|---|---:|---:|---:|---:|
| Training | 1 Jan 2022–31 Dec 2023 | 35,040 | 33,452 | 1,588 |
| Validation | 1 Jan–30 Jun 2024 | 8,736 | 8,705 | 31 |
| Test | 1 Jul–31 Dec 2024 | 8,832 | 8,832 | 0 |

Training missingness was structured: 1,441 of 1,588 absent targets formed one contiguous gap from 6 March to 5 April 2023. Rows with missing targets were excluded from fitting or evaluation; feature missingness was handled by training-fitted median imputation.

### 2.3. Forecast models and leakage control

Forty-eight direct models predicted 0.5–24 h ahead. The model-origin timestamp denotes the start of the most recently completed half-hour; reported horizons measure from the following decision boundary to completion of the target interval. Features comprised lagged carbon intensity; trailing means, standard deviations, extrema, and changes; target-slot analogues from prior days and the prior week; cyclical time encodings; and a weekend indicator. Every analogue offset pointed backward from the origin. Each horizon was fitted independently, so predictions were never recursively supplied as inputs.

Four non-ML methods were declared: last-value persistence, previous-day and previous-week seasonal rules, and a half-hour-of-week climatology fitted on training targets. Two fixed tree-based configurations were compared: histogram gradient boosting (HGB) and random forest. Aggregate validation mean absolute error (MAE) selected HGB (34.10 versus 34.86 g CO₂/kWh). The primary forecast benchmark selected the lowest-validation-MAE non-ML rule separately at each horizon, producing persistence at 31 horizons and the previous-day rule at 17; an exact 24 h tie was resolved alphabetically. CarbonCast provides a relevant multi-day learned-forecast comparison [15], but no external forecast was inserted because comparable issue-time provenance was unavailable.

### 2.4. Scheduling simulation

One independent job was released daily at 18:00 UTC. It consumed exactly 1 kWh uniformly during a contiguous three-hour execution and had to finish within 12 h. Every policy received the same release, feasible starts, deadline, energy, and historical information. Policies were immediate execution; persistence; previous-day and previous-week seasonal forecasts; training-only climatology; HGB; random forest; and an unattainable oracle that minimized realized average intensity. Ties selected the earliest start.

Each feasible block was scored by the corresponding forecast, and its realized accounting outcome was the fixed job energy multiplied by mean NESO average intensity over the selected slots. On 178 paired validation days, climatology was the lowest-emission non-ML scheduler (108.62 g/job), ahead of previous-day (111.79), previous-week (113.72), and persistence (147.82); it was therefore the primary comparator. Sensitivities used 6 h and 24 h windows and one-, three-, and six-hour durations.

### 2.5. Evaluation and statistical analysis

Forecasts were assessed using MAE, root mean squared error (RMSE), and mean bias error on observations shared by every method. Scheduling outcomes included mean g CO₂/job, percentage change from immediate execution, oracle-opportunity capture, and start delay. Because adjacent horizons and decisions share the same electricity trace, observations were paired and individual half-hours were not treated as independent.

Primary 95% confidence intervals used 5,000-replicate, seven-day circular moving-block bootstraps over ordered UTC issue days. Circular blocks wrap the December–July boundary and assume approximately stationary daily contrasts within the fitted model and retrospective period. One-day and 14-, 21-, and 28-day blocks tested dependence sensitivity. Intervals are conditional resampling intervals, not guarantees for another grid or future period; configuration sensitivities are exploratory and unadjusted for multiplicity.

### 2.6. Accounting boundary and reproducibility

The outcome is an attributional, operational estimate for a small, fixed-energy job. It excludes marginal generator response, prices, dispatch, synchronized adoption, rebound, embodied hardware, networking, storage, and separately metered training or inference energy. Computing footprints can vary materially with hardware, location, and utilization [16–19]; consequently, we make no net lifecycle claim. The repository records the raw-data digest, split manifest, environment, seed (20240906), code, paired predictions, bootstrap outputs, and 19 invariant checks. An OpenAI GPT-5-based coding assistant supported code authoring and analysis checking; generated numerical statements were cross-checked programmatically against archived outputs.

## 3. Results

### 3.1. Forecast performance

All 8,832 target half-hours were present in the test window. Evaluation retained 8,784 complete origins at 48 horizons, giving 421,632 paired instances across 183 issue days. HGB attained aggregate MAE 30.66 g CO₂/kWh, compared with 36.09 for the validation-selected composite (Table 2). The paired MAE reduction was 5.43 g CO₂/kWh (seven-day 95% confidence interval (CI): [1.37, 9.14]), or 15.05%. The interval remained above zero for one-day ([3.09, 7.80]) and 14-day blocks ([0.41, 9.75]) but crossed zero for 21-day ([−0.24, 10.01]) and 28-day blocks ([−0.42, 10.20]). Monthly mean advantages varied from −10.87 g CO₂/kWh in August to +14.54 in October, where positive values favor HGB.

**Table 2. Aggregate forecast performance, July–December 2024.**

| Method | MAE | RMSE | Bias | Skill vs composite |
|---|---:|---:|---:|---:|
| HGB | 30.66 | 40.11 | +11.07 | +15.05% |
| Random forest | 32.34 | 41.96 | +12.04 | +10.39% |
| Validation-selected composite | 36.09 | 48.16 | +0.11 | Reference |
| Persistence | 37.72 | 50.20 | +0.09 | −4.51% |
| Previous day | 40.29 | 52.59 | +0.18 | −11.63% |

*Note: Error and bias units are g CO₂/kWh. Positive bias denotes overprediction. N = 421,632 paired instances. Skill is relative to the per-horizon non-ML comparator selected on validation.*

HGB MAE increased from 3.57 g CO₂/kWh at 0.5 h to 14.78 at 3 h, 24.52 at 6 h, 34.24 at 12 h, and 40.64 at 24 h. At 24 h it was slightly worse than the previous-day value of 40.25. Its aggregate bias of +11.07 is consistent with the lower-carbon test period relative to training and cautions against interpreting the average skill estimate as stable calibration.

[FIGURE:experiment/results/forecast_mae_by_horizon.png|Fig. 1. Forecast MAE by horizon on the July–December 2024 test trace. The composite benchmark selects a non-ML rule independently at each horizon using validation only.|Line chart comparing HGB, random forest, the validation-selected composite non-ML benchmark, and the previous-day rule over 0.5–24 h horizons.]

### 3.2. Scheduling performance

Across 183 paired days, immediate execution was assigned 151.97 g CO₂/job; HGB, 102.17; validation-selected climatology, 100.94; and the oracle, 96.52 (Table 3). HGB therefore reduced the accounting estimate by 49.81 g/job (32.77%) relative to immediate execution (seven-day 95% CI: [44.16, 55.31]). That full contrast is not attributable to ML: climatology reduced it by 51.03 g/job and captured 92.02% of the oracle opportunity, compared with 89.81% for HGB.

**Table 3. Primary scheduling outcome for a 1 kWh, three-hour job in a 12 h window.**

| Policy | Mean g CO₂/job | Reduction vs immediate | Oracle captured | Mean delay (h) |
|---|---:|---:|---:|---:|
| Immediate | 151.97 | 0.00 (0.00%) | 0.00% | 0.00 |
| Previous day | 104.74 | 47.23 (31.08%) | 85.17% | 5.90 |
| Training climatologyᵃ | 100.94 | 51.03 (33.58%) | 92.02% | 6.42 |
| Random forest | 103.12 | 48.86 (32.15%) | 88.10% | 5.12 |
| HGB | 102.17 | 49.81 (32.77%) | 89.81% | 5.27 |
| Oracle | 96.52 | 55.45 (36.49%) | 100.00% | 5.89 |

*Note.* N = 183 paired days. ᵃPrimary non-ML comparator selected on validation. The oracle uses future realized intensity and is unattainable.

The primary comparator-minus-HGB difference was −1.22 g CO₂/job, or −1.21% of the climatology mean (seven-day 95% CI: [−2.71, 0.16]). The point estimate favors the simple rule, and the interval provides no reliable evidence of incremental ML benefit. An exploratory contrast with the weaker previous-day policy favored HGB by 2.57 g/job (95% CI: [0.65, 4.45]). Mean, median, and 95th-percentile HGB delays were 5.27, 5.0, and 9.0 h, versus 6.42, 6.5, and 9.0 h for climatology.

[FIGURE:experiment/results/scheduling_primary.png|Fig. 2. Average-intensity-weighted operational CO₂ assigned by policy. Lower is better; the oracle is a perfect-information lower bound.|Bar chart comparing attributed operational grams of CO₂ per fixed 1 kWh job across immediate, non-ML, learned, and oracle scheduling policies.]

### 3.3. Sensitivity and diagnostics

The climatology-minus-HGB contrast was −0.43 g/job for a three-hour job in a 6 h window (seven-day 95% CI: [−1.14, 0.04]), −1.39 for a one-hour job in a 12 h window ([−3.62, 0.65]), −1.22 for the primary case ([−2.71, 0.16]), and −0.93 for a six-hour job ([−1.89, −0.03]); negative values favor climatology. For a three-hour job in a 24 h window, HGB instead improved on climatology by 7.00 g/job ([1.91, 12.29]). The latter interval crossed zero with 21- and 28-day blocks, reinforcing dependence and configuration uncertainty.

All 19 recorded invariants passed, including chronological split separation, non-future analogue offsets, full horizon cohorts, equal paired day counts, energy conservation, earliest-slot tie breaking, zero-slack equivalence to immediate execution, oracle dominance, and reproduction of validation-only model and comparator selection.

## 4. Discussion

### 4.1. Prediction skill and decision value are different quantities

The experiment yields a deliberately asymmetric result. HGB improved period-average forecast MAE against a strong validation-selected benchmark, although the longest-block intervals weaken that conclusion. It did not improve the primary scheduling outcome relative to validation-selected climatology. Global MAE assigns equal weight to errors across the forecast surface, whereas the scheduler depends mainly on the rank and separation of feasible contiguous blocks. A biased climatology can therefore have poor level accuracy yet make a better choice for a repetitive evening workload.

The immediate-versus-HGB contrast is also not an estimate of AI value. It combines the value of a 12 h flexibility window with the scheduler. The appropriate incremental contrast is HGB versus a functionally equivalent simple policy using the same jobs, information boundary, and feasibility set. Here the simple policy captured more of the oracle opportunity. This result accords with the broader principle that ML should be credited only for benefits additional to a competitive non-ML alternative [2,3].

### 4.2. Energy and climate interpretation

Average carbon intensity is useful for allocating an operational footprint but does not identify which generator changes output when demand moves [8–10]. A production claim of avoided emissions would require an as-issued marginal or dispatch-based signal, measured workload electricity including facility overhead, and a causal or validated power-system response. At scale, synchronized shifting could move or create peaks, change prices, and invalidate an exogenous historical trace [7]. The model's training, inference, and hardware footprint would also have to be included.

This distinction generalizes beyond computing. AI-enabled climate claims are strongest when a model changes a bounded intervention and the downstream physical outcome is measured. Machine-learning targeting has, for example, improved allocation in a residential efficiency program [20]. Learned weather and flood systems demonstrate substantial predictive capability [21–23], while satellite analysis can identify major methane sources [24]. In each domain, however, accuracy becomes climate value only through action: retrofit delivery, warnings that prompt protective behavior, or verified leak repair. AI is therefore an enabling layer within energy and climate institutions, not a stand-alone mitigation measure [2,25].

### 4.3. Deployment and evaluation requirements

Before deploying a learned carbon-aware scheduler, five conditions should be tested. First, the energy service, deadline, and reliability constraints must be identical across policies. Second, all features and external forecasts must be demonstrably available at decision time. Third, evaluation should report decision utility as well as prediction loss and include simple seasonal, persistence, and rule-based alternatives. Fourth, the carbon signal must match the claim: average for attributional allocation, marginal or dispatch-based for a short-run causal response, and lifecycle analysis for net benefit. Fifth, model overhead, distribution shift, synchronized adoption, and failure behavior must be measured at the intended scale.

The present evidence does not justify adding HGB for the primary workload. In the observed contrast, climatology already has a lower per-job accounting value; non-negative training and embodied costs cannot reverse that ordering. The 24 h sensitivity indicates that learned forecasts may become useful with a wider action space, but that exploratory result needs untouched evaluation and a production-relevant cost model.

### 4.4. Limitations

The study concerns one national signal, six months, and a synthetic, perfectly delay-tolerant workload. Great Britain's generation mix and market design limit transferability. NESO “actual” values are finalized estimates; the assumed one-interval availability may not match operational publication latency. The models omit archived ex-ante weather, demand, fuel-price, outage, and generation forecasts. The fixed 18:00 UTC release spans daylight-saving changes in local clock time, and a long 2023 missing-data interval may affect training.

The scheduler omits capacity, server utilization, power-usage effectiveness, checkpointing, data transfer, electricity prices, heterogeneous jobs, and service penalties. The small-load assumption precludes dispatch, market, rebound, and synchronized-adoption effects. Development energy was not separately metered. Confidence intervals quantify temporal resampling uncertainty within the trace, not structural uncertainty or external validity. Finally, the protocol was refined after test inspection, so none of the reported intervals should be read as preregistered confirmatory inference.

## 5. Conclusions

On a later Great Britain trace, HGB reduced period-average carbon-intensity forecast MAE by 15.05% relative to a validation-selected non-ML composite. That accuracy gain did not produce a reliable improvement in the primary carbon-aware scheduling outcome: training-only climatology was assigned 100.94 g CO₂/job versus 102.17 for HGB and captured 92.02% of the perfect-information opportunity. Most of the apparent benefit relative to immediate execution came from temporal flexibility and calendar structure, not from ML.

The result is not that forecasting is irrelevant; a wider-window sensitivity favored HGB, and richer ex-ante signals may matter elsewhere. It is that forecast accuracy, scheduling utility, and climate impact are distinct claims. Credible AI-for-energy studies should evaluate the downstream decision against a functionally equivalent simple baseline, enforce deployment-time information boundaries, quantify temporal dependence, and match the emissions claim to its accounting and causal boundary.

## CRediT authorship contribution statement

**Aditya Shrivastava:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Project administration.

## Data availability

The raw NESO dataset is publicly available from the National Carbon Intensity Forecast data portal [13]. The reproducibility package containing acquisition code, the source hash, preprocessing and modeling code, the chronological split manifest, paired predictions, result tables, invariant checks, and environment details is publicly available at `https://github.com/adityashriv2003/ai-climate-action-research`.

## Ethics statement

This study used public aggregate electricity-system data and involved no personal data, human participants, or animals.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of competing interest

The author declares no competing financial or non-financial interests.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the author used an OpenAI GPT-5-based coding assistant to assist with literature discovery, code development, analysis checking, manuscript organization, and language editing. After using this tool/service, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

## References

[1] Intergovernmental Panel on Climate Change. Climate Change 2022: Mitigation of Climate Change. Contribution of Working Group III to the Sixth Assessment Report. Cambridge University Press, Cambridge, 2022. https://doi.org/10.1017/9781009157926.

[2] L.H. Kaack, P.L. Donti, E. Strubell, G. Kamiya, F. Creutzig, D. Rolnick, Aligning artificial intelligence with climate change mitigation, Nat. Clim. Chang. 12 (2022) 518–527. https://doi.org/10.1038/s41558-022-01377-7.

[3] P.L. Donti, J.Z. Kolter, Machine learning for sustainable energy systems, Annu. Rev. Environ. Resour. 46 (2021) 719–747. https://doi.org/10.1146/annurev-environ-020220-061831.

[4] International Energy Agency, Energy and AI, IEA, Paris, 2025. https://www.iea.org/reports/energy-and-ai.

[5] A. Radovanović, R. Koningstein, I. Schneider, et al., Carbon-aware computing for datacenters, IEEE Trans. Power Syst. 38 (2023) 1270–1280. https://doi.org/10.1109/TPWRS.2022.3173250.

[6] P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen, Let's wait awhile: How temporal workload shifting can reduce carbon emissions in the cloud, in: Proc. 22nd Int. Middleware Conf., ACM, 2021, pp. 260–272. https://doi.org/10.1145/3464298.3493399.

[7] T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy, On the limitations of carbon-aware temporal and spatial workload shifting in the cloud, in: Proc. Nineteenth Eur. Conf. Comput. Syst., ACM, 2024, pp. 924–941. https://doi.org/10.1145/3627703.3650079.

[8] A.D. Hawkes, Estimating marginal CO₂ emissions rates for national electricity systems, Energy Policy 38 (2010) 5977–5987. https://doi.org/10.1016/j.enpol.2010.05.053.

[9] P.L. Donti, J.Z. Kolter, I.L. Azevedo, How much are we saving after all? Characterizing the effects of commonly varying assumptions on emissions and damage estimates in PJM, Environ. Sci. Technol. 53 (2019) 9905–9914. https://doi.org/10.1021/acs.est.8b06586.

[10] T. Sukprasert, N. Bashir, A. Souza, D. Irwin, P. Shenoy, On the implications of choosing average versus marginal carbon intensity signals on carbon-aware optimizations, in: Proc. 15th ACM Int. Conf. Future Sustain. Energy Syst., ACM, 2024, pp. 422–427. https://doi.org/10.1145/3632775.3661953.

[11] L.J. Tashman, Out-of-sample tests of forecasting accuracy: An analysis and review, Int. J. Forecast. 16 (2000) 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0.

[12] V. Cerqueira, L. Torgo, I. Mozetič, Evaluating time series forecasting models: An empirical study on performance estimation methods, Mach. Learn. 109 (2020) 1997–2028. https://doi.org/10.1007/s10994-020-05910-7.

[13] National Energy System Operator, National Carbon Intensity Forecast, NESO Data Portal, 2026. https://www.neso.energy/data-portal/national-carbon-intensity-forecast/national_carbon_intensity_forecast (accessed 6 September 2026).

[14] A.R.W. Bruce, L. Ruff, J. Kelloway, F. MacMillan, A. Rogers, Carbon Intensity Forecast Methodology, National Energy System Operator, 2024. https://www.neso.energy/data-portal/national-carbon-intensity-forecast/national_carbon_intensity_forecast_methodology.

[15] D. Maji, P. Shenoy, R.K. Sitaraman, CarbonCast: Multi-day forecasting of grid carbon intensity, in: Proc. 9th ACM Int. Conf. Syst. Energy-Efficient Build. Cities Transp., ACM, 2022, pp. 198–207. https://doi.org/10.1145/3563357.3564079.

[16] L. Lannelongue, J. Grealey, M. Inouye, Green Algorithms: Quantifying the carbon footprint of computation, Adv. Sci. 8 (2021) 2100707. https://doi.org/10.1002/advs.202100707.

[17] A.S. Luccioni, S. Viguier, A.-L. Ligozat, Estimating the carbon footprint of BLOOM, a 176B parameter language model, J. Mach. Learn. Res. 24 (2023) 1–15. https://www.jmlr.org/papers/v24/23-0069.html.

[18] J. Dodge, T. Prewitt, R. Tachet des Combes, et al., Measuring the carbon intensity of AI in cloud instances, in: Proc. 2022 ACM Conf. Fairness Account. Transpar., ACM, 2022, pp. 1877–1894. https://doi.org/10.1145/3531146.3533234.

[19] P. Henderson, J. Hu, J. Romoff, E. Brunskill, D. Jurafsky, J. Pineau, Towards the systematic reporting of the energy and carbon footprints of machine learning, J. Mach. Learn. Res. 21 (2020) 1–43. https://www.jmlr.org/papers/v21/20-312.html.

[20] P. Christensen, P. Francisco, E. Myers, H. Shao, M. Souza, Energy efficiency can deliver for climate policy: Evidence from machine learning-based targeting, J. Public Econ. 234 (2024) 105098. https://doi.org/10.1016/j.jpubeco.2024.105098.

[21] R. Lam, A. Sanchez-Gonzalez, M. Willson, et al., Learning skillful medium-range global weather forecasting, Science 382 (2023) 1416–1421. https://doi.org/10.1126/science.adi2336.

[22] D. Kochkov, J. Yuval, I. Langmore, et al., Neural general circulation models for weather and climate, Nature 632 (2024) 1060–1066. https://doi.org/10.1038/s41586-024-07744-y.

[23] G. Nearing, D. Cohen, V. Dube, et al., Global prediction of extreme floods in ungauged watersheds, Nature 627 (2024) 559–563. https://doi.org/10.1038/s41586-024-07145-1.

[24] T. Lauvaux, C. Giron, M. Mazzolini, et al., Global assessment of oil and gas methane ultra-emitters, Science 375 (2022) 557–561. https://doi.org/10.1126/science.abj4351.

[25] D. Rolnick, P.L. Donti, L.H. Kaack, et al., Tackling climate change with machine learning, ACM Comput. Surv. 55 (2022) 42:1–42:96. https://doi.org/10.1145/3485128.
