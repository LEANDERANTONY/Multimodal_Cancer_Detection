# Literature novelty check (2026-10-25, before the write-up)

Searched PubMed, Consensus and the web (arXiv / medRxiv / journals) for every claimed contribution. Verdict per
claim, the closest prior work, and how the paper must position itself. Re-run the risky searches (marked ⚠) just
before submission.

## Verdict summary

| # | Our contribution | Verdict | Closest prior work | Positioning |
|---|---|---|---|---|
| 1 | Measured scanner confound in PANORAMA (Cramér's V 0.44, scanner-only AUROC 0.70) and a confound tax with a patient-bootstrap CI, plus LOMO and external | **Novel** ⚠ | No analysis of a vendor/label confound in PANORAMA found; the PANORAMA commentary presents vendor diversity as a strength (Hunsaker). Generic acquisition-bias work: Ong Ly 2024 (npj DM, up to 20 % overestimation from acquisition bias); "Name that manufacturer" (head CT, vendor is learnable); BenchX 2026 (subgroup evaluation incl. contrast phase, 85k CTs, no confound measure, no mitigation) | Lead claim. Cite Ong Ly and BenchX as the general frame; ours is a measured, quantified confound in the flagship public PDAC cohort |
| 2 | Deployment-ROI: oracle vs predicted pancreas crops with real segmenters; TotalSegmentator `roi_subset` fails on 6 % (3x more on PDAC, chest-heavy scans), the documented `robust_crop` recovers half, neighbouring organs recover all; failures raise the tax | **Novel** | No study found that runs one PDAC detector on ground-truth vs predicted pancreas crops. Adjacent: Radiology 2022 nationwide study (9 % tumour mislocalisation by the segmenter); PanDx two-stage design; segmenter benchmarks (TotalSegmentator best of five on 352 CTs) measure Dice, not detection | Lead claim, with the actionable fix (request neighbouring organs) and the failure mechanism |
| 3 | Mitigation panel on a real acquisition confound: tuned ERM not beaten (GBS, DFR, GRL, MAE), planted-shortcut positive control shows the same methods remove 64–91 % of a planted shortcut; MAE raises the tax | **Partly known** | "ERM is hard to beat": DomainBed, MEDFAIR (ICLR 2023: 11 methods, no significant gain over ERM), Yang et al. Nat Med 2024 (debiasing closes in-distribution gaps but does not transfer). Disentanglement benchmark 2026 (Müller & Berens; synthetic OCT confound). Planted/synthetic bias as a tool: Stanley et al. 2024 (eBioMedicine). Foundation models encode centre: de Jong et al. 2025 (pathology) | Do NOT headline "debiasing does not beat ERM". Novel parts: (a) a measured acquisition confound in 3D CT cancer detection with the tax as the target; (b) the planted-shortcut positive control that separates "methods broken" from "nothing removable"; (c) CT masked-image pretraining increases scanner reliance (consistent with de Jong in pathology, new for CT SSL) |
| 4 | 2.5D classifier vs 3D segmentation model: far better in-distribution (+0.138), larger tax, ties externally | **Supporting, moderately novel** | Lesion-level vs image-level supervision and shortcuts in radiographs (arXiv 2104.10553) | Frame as evidence that dense lesion supervision anchors the model; cite the radiograph study |
| 5 | Site confound in the Debernardi urinary biomarker cohort (all controls from one site, V 0.42, site-only AUROC 0.68, tax +0.062) | **Novel observation** | No published critique found. Debernardi 2022 (prediagnostic cohort) reports AUC 0.79, below the 2020 case-control 0.94, consistent with our finding | Secondary result; widely reused public dataset, so the warning is useful |
| 6 | Fusion of unpaired cohorts: the pseudo-paired gain equals the conditional-independence bound; triage simulation over the error correlation | **Novel framing, contested** | Glioma multimodal study 2026 (PMC12996459): label-matched random pairing ≈ true patient pairing on TCGA, where paired data existed to check it; label-agnostic pairing falls to unimodal | Engage directly: where paired data exist, label pairing can be validated (glioma); without them its gain is untestable and equals the independence case. Our ρ-sensitivity makes the assumption explicit |
| 7 | Thesis two-source set: 0.9999 not reproduced (0.68–0.98) by models that never saw either collection | **Case study, not novel alone** | Zech 2018, DeGrave 2021 (COVID shortcuts), Ong Ly 2024 | Use as the motivating example, not a contribution |

## Overall

The paper is not redundant if framed as an **end-to-end audit of measured confounding and deployment in the
flagship public PDAC CT cohort**: contributions 1 and 2 are new, 3 is new in its positive-control design and
setting (but its headline finding is known and must be cited as such), 4–6 support. The main risk is time: the
BenchX / Johns Hopkins group works on the same scans and protocol biases, so a PANORAMA vendor analysis could
appear; re-search before submission.

## Sources

- PanDx (PANORAMA winner): https://arxiv.org/abs/2503.10068
- PANORAMA commentary (Hunsaker, J Med AI): https://jmai.amegroups.org/article/view/11400/html
- PANORAMA study, Lancet Oncology 2025: https://www.sciencedirect.com/science/article/abs/pii/S1470204525005674
- BenchX (2026): https://arxiv.org/abs/2606.24883
- Ong Ly et al., npj Digit Med 2024: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11094145/
- "Name that manufacturer" (head CT): https://www.researchgate.net/publication/343759124
- Risk of shortcutting (knee X-ray, Sci Rep 2024): https://www.nature.com/articles/s41598-024-79838-6
- MEDFAIR (ICLR 2023): https://arxiv.org/abs/2210.01725
- Yang et al., The limits of fair medical imaging AI, Nat Med 2024: https://www.nature.com/articles/s41591-024-03113-4
- Feature-disentanglement shortcut benchmark (2026): https://arxiv.org/abs/2602.18502
- Shortkit-ML (medRxiv 2026): https://www.medrxiv.org/content/10.64898/2026.04.29.26352053v1.full
- Stanley et al., synthetic bias encoding, eBioMedicine 2024: https://consensus.app/papers/details/74e72101519954c294f55f99621b5c3d/
- Banerjee et al., shortcuts in radiology AI, JACR 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC11192466/
- de Jong et al., pathology FMs encode medical centre (2025): https://arxiv.org/abs/2501.18055
- Annotation granularity and shortcuts (radiographs): https://arxiv.org/abs/2104.10553
- Pancreatic cancer detection, nationwide study, Radiology 2022: https://pubs.rsna.org/doi/full/10.1148/radiol.220152
- Pancreas segmenter comparison (Acad Radiol 2024): https://www.sciencedirect.com/science/article/abs/pii/S1076633224003738
- Debernardi et al., PLoS Med 2020: https://consensus.app/papers/details/55e04b76cbfa55d29542e847a23ee90c/
- Debernardi et al., Int J Cancer 2022 (prediagnostic): https://consensus.app/papers/details/32c9b0362c665c59b08c29ff3d179378/
- Glioma pathology + radiology fusion with label pairing (2026): https://pmc.ncbi.nlm.nih.gov/articles/PMC12996459/
