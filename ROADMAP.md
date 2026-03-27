# Roadmap

This roadmap reflects the current project state and the next major build priorities for the multimodal pancreatic cancer detection repository.

## Now: Stabilize The Hybrid Research Repo

Current baseline:

- `uv`-managed environment and lockfile
- modular helper layers under `src/`
- one primary notebook driving the full experimental flow
- tracked reports and figures for the latest results
- local-only raw data, processed data, embeddings, and model checkpoints

Highest-priority remaining work:

- continue refactoring reusable notebook logic into `src/`
- make the notebook more orchestration-focused and less implementation-heavy
- keep documentation aligned with the repo's real state
- preserve a clean boundary between tracked lightweight outputs and ignored heavy assets
- keep the thesis interpretation visible in the codebase: strong biomarker reproducibility, ambiguous CT generalization, and exploratory-only fusion claims

Status:

- Active delivery focus

## Next: Reproducibility And Research-Code Hardening

- add lightweight tests around stable helper modules
- add more explicit run paths for common tasks such as report generation and preprocessing
- reduce notebook-only helper duplication in EDA and diagnostics sections
- tighten artifact naming and report consistency across `reports/` and `figures/`
- keep `uv` dependencies aligned with actual runtime imports
- codify negative-control and multi-seed evaluation patterns so future fusion work stays methodologically honest

Status:

- In progress

## Later: Cleaner Experiment Interfaces

The next major structural improvement after modularization is a cleaner runner surface around the existing notebook and modules.

Targets:

- expose common experiment steps through small scripts or thin CLI entry points
- make key reports reproducible without manually re-running the full notebook
- separate stable experiment interfaces from one-off thesis exploration code
- keep notebook usage focused on analysis, visual inspection, and narrative assembly
- make it easier to rerun the bias-analysis and summary-generation parts of the workflow outside a full notebook session

Status:

- Planned, not started

## Future: Expanded Research Extensions

Potential later work includes:

- stronger uncertainty tooling under `src/uncertainty/`
- cleaner support for repeated ablation runs
- broader experiment packaging for publication or demo purposes
- adaptation to real paired multimodal cohorts if such data becomes available
- external CT validation on additional institutions to resolve whether the current CT signal is pathological, domain-driven, or mixed
- domain-adversarial CT training, likely via a gradient reversal layer, to suppress dataset-of-origin information in learned representations
- matched CT-plus-biomarker cohorts so fusion can be evaluated as a real clinical question instead of a synthetic pairing exercise
- volumetric CT architectures or transformers once the data and validation setup justify moving beyond slice-based modelling

Status:

- Deferred until the current modularization and reproducibility work is stable
