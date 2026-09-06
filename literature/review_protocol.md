# Structured critical review protocol

## Purpose

This review maps mechanisms, evidence maturity, and claim boundaries for AI in
climate mitigation and adaptation. It is a structured narrative review, not a
PRISMA systematic review or meta-analysis. It must not be used to estimate the
share of AI applications that succeed.

## Search date and scope

- Final search date: 2026-09-06
- Main publication window: 2015–2026
- Earlier foundational papers retained where necessary for remote sensing,
  marginal-emissions accounting, and evaluation methodology.
- Language: English
- Source priority: peer-reviewed primary studies; field-defining peer-reviewed
  reviews; official intergovernmental, standards, or public-operator sources.
- Preprints and vendor reports were excluded from the main evidentiary basis.

## Query families

Cross-cutting query:

```text
("artificial intelligence" OR "machine learning" OR "deep learning" OR
 "reinforcement learning" OR "neural network*") AND
("climate change" OR decarboni* OR "greenhouse gas*" OR carbon OR resilience
 OR adaptation) AND
(mitigat* OR reduc* OR forecast* OR optim* OR monitor* OR evaluat*)
```

Domain-specific searches appended one or more of:

```text
electricity OR grid OR renewable OR building OR transport OR agriculture OR
forest OR methane OR industry OR materials OR weather OR flood OR disaster OR
"carbon-aware computing" OR "marginal emissions"
```

Known-item and citation-chain searches began from Rolnick et al. (2022), Kaack
et al. (2022), the IPCC AR6 Working Group II and III reports, and the IEA
*Energy and AI* report. DOI landing pages, publisher records, journal pages,
and official institutional pages were used to verify metadata and claims.

## Inclusion criteria

A source was eligible if it satisfied at least one of the following:

1. Quantitatively evaluated an AI component in a climate-relevant sensing,
   prediction, control, discovery, or decision-support chain.
2. Evaluated AI's electricity, operational carbon, embodied carbon, water, or
   system-level effects.
3. Established a necessary emissions-accounting or causal-evaluation
   distinction.
4. Provided authoritative and current system-scale context unavailable in a
   peer-reviewed primary study.

Accuracy-only studies were retained as proxy evidence and explicitly labelled
as such. Field, operational, and causal evidence was distinguished from
hindcast benchmarks, trace simulation, lifecycle modeling, and scenarios.

## Exclusion criteria

- Opportunity lists without an evaluated system or a necessary framework.
- Climate-themed classification benchmarks with no plausible decision link.
- Generic efficient-computing work with no AI or climate linkage.
- Duplicate descriptions of the same experiment when a fuller source existed.
- Unverifiable quantitative claims, marketing pages, and silently
  non-peer-reviewed evidence.
- Studies whose comparator performed a materially different quantity or
  quality of work, unless used only to illustrate that limitation.

## Extraction fields

- Full bibliographic metadata and persistent identifier.
- Domain and mitigation/adaptation objective.
- AI role: sensing, prediction, optimization/control, discovery, or decision
  support.
- Study design and deployment maturity.
- Outcome level: model metric, decision proxy, physical resource, attributed
  emissions, consequential effect, or lifecycle/system effect.
- Comparator, geography, period, data provenance, and uncertainty.
- Supported claim and explicit boundary/caveat.
- Rebound, justice, and governance relevance where reported.

## Synthesis rule

Heterogeneous effects were not pooled. Sources are synthesized by mechanism,
outcome proximity, and deployment maturity. A claim cannot be stronger than
the weakest supported link in the chain from model output to physical response
and climate outcome. Scenario estimates are labeled as scenarios; attributed
operational emissions are not described as causal avoided emissions.

## Limitations

This was not preregistered, database exports and exclusion decisions were not
double-screened by two independent human reviewers, and a complete universe of
records was not enumerated. The review is therefore transparent and
source-verified but not exhaustive. The accompanying source matrix is the
auditable record of included evidence.
