# Project Timeline — Multimodal Pancreatic Cancer Detection
**Window:** Aug 10 – Sept 15, 2025  
**Owner:** Leander Antony A  
**Compute:** Local CPU + Google Colab Pro (GPU)

---

## Assumptions
- CT data and urine CSV are available locally (raw not pushed to GitHub).
- CT heavy training runs on Colab Pro; urine + fusion on local.
- Workload target: ~4–6 hrs on weekdays, ~2–3 hrs weekends (Colab can run longer unattended).

---

## Milestones Overview
1) Unimodal CT (YOLO ROI + CNN) ✅ baseline  
2) Unimodal Urine (MLP) ✅ baseline  
3) Embeddings exported (CT 256D, Urine 32D)  
4) Synthetic pairing (CV-aware, stratified)  
5) Fusion bake-off: Late / Early Concat / Orthogonal  
6) Cross-modal Attention fusion + Attention maps  
7) Contrastive pre-alignment (InfoNCE) ablation  
8) Uncertainty + Calibration (MC Dropout + Temp Scaling)  
9) Federated simulation (FedAvg over folds)  
10) Docs, Figures, Thesis, Slides, Video

---

## Day-by-Day Plan

### Aug 10–12 (Sun–Tue) — Lock Unimodal Baselines
**CT (Colab):** Train YOLOv8 ROI → CT classifier; export per-patient scores + **CT embeddings (256D)**  
**Urine (Local):** Preprocess (impute/scale) → train **MLP (5-fold)** → **Urine embeddings (32D)**  
**Deliverables:**  
- `embeddings/ct_embeddings_fold{k}.parquet`  
- `embeddings/urine_embeddings_fold{k}.parquet`  
- Baseline CT/Urine metrics table

### Aug 13 (Wed) — Synthetic Pairing
**Task:** CV-aware **stratified hot-deck** pairing (by label; add age/sex bins if available), R=3 seeds/fold  
**Deliverables:**  
- `synthetic_pairs/fold_k/{train,val,test}.parquet`, `metadata.json`  
- Pairing sanity report (class balance, KS test)

### Aug 14–16 (Thu–Sat) — Fusion v1 (Late, Early, Orthogonal)
**Tasks:**  
- Late fusion (weight sweep + logistic stacking)  
- Early concat (freeze encoders; light fine-tune last epochs)  
- Orthogonal fusion (λ ∈ {0, 1e-4, 5e-4, 1e-3})  
**Deliverables:** Fusion metrics, bar chart, brief ablation notes

### Aug 17 (Sun) — Cross-Modal Attention (v0)
**Task:** Train **2-token attention head** on embeddings (small grid for d_model/layers)  
**Deliverable:** Initial attention fusion metrics

### Aug 18–19 (Mon–Tue) — Attention Maps
**Tasks:**  
- Extract & average **CT↔Urine attention matrices** over validation  
- (Optional) Spatial cross-attention overlay on ROI for 1–2 cases  
**Deliverables:**  
- `figures/attention/attn_matrix_mean.png` (+ optional spatial overlay)  
- Short interpretation notes

### Aug 20 (Wed) — Contrastive Pre-Alignment
**Task:** **InfoNCE** pretrain with multiple random pairings; re-run Early/Orthogonal/Attention  
**Deliverables:** Table: w/ vs w/o contrastive; stability comment

### Aug 21 (Thu) — Uncertainty & Calibration
**Tasks:**  
- **MC Dropout (T=30)**; **Temperature Scaling** on val  
- ECE, predictive entropy, risk–coverage curves  
**Deliverables:** `figures/uncertainty/ece_curve.png`, `risk_coverage.png`, metrics updated

### Aug 22 (Fri) — Federated (Sim)
**Task:** Treat 5 CV folds as clients; run **FedAvg** on fusion head (embeddings)  
**Deliverable:** Table: centralized vs FedAvg; 2–3 discussion bullets

### Aug 23–24 (Sat–Sun) — Baselines & Ablations Cleanup
**Tasks:**  
- CT **with vs without** ROI (lift)  
- **Modality dropout** at inference (robustness)  
- **Noise** on urine (+5–10%)  
- 95% CI on key metrics  
**Deliverables:** Ablation table + 2 plots

### Aug 25–27 (Mon–Wed) — Figures & Documentation
**Tasks:**  
- **Grad-CAM** panels (CT ROI)  
- **SHAP/perm importance** (urine)  
- README updates: **Fusion Interface**, **Uncertainty**, **Federated**, **Figures**  
- ADR updates: **ADR-001**, **ADR-002 (Attention & Contrastive Additions)**  
**Deliverables:** Polished figures under `figures/` + updated docs

### Aug 28–31 (Thu–Sun) — Thesis Writing Sprint #1
**Tasks:** Methods, Results, Discussion drafts (synthetic justification, transferability, limits, future work)  
**Deliverable:** Draft manuscript with figure placeholders

### Sept 1–3 (Mon–Wed) — Industry Polish (Optional but High ROI)
**Tasks:**  
- **Dockerfile** for inference (embeddings)  
- **FastAPI** endpoint `/predict` (calibrated prob + uncertainty)  
- Optional Streamlit demo screenshot for README  
**Deliverables:** `Dockerfile`, `api/app.py`, demo image

### Sept 4–6 (Thu–Sat) — Thesis Writing Sprint #2
**Tasks:**  
- Strengthen novelty (attention + contrastive) & deployment considerations (size, latency)  
- Incorporate calibration & federated results into discussion  
**Deliverables:** Near-final thesis text

### Sept 7–9 (Sun–Tue) — Final Tables & Figures
**Tasks:**  
- Lock seeds; re-run flaky folds; export **final CSVs**  
- Generate **master results table** + high-res PNG/PDF figures with captions  
**Deliverables:** `results/` finalized

### Sept 10–11 (Wed–Thu) — Slides & Script
**Tasks:**  
- Build presentation: Problem → Data → Pipelines → Fusion bake-off → Attention maps → Uncertainty → FedAvg → Results → Limits → Future  
- Add 2–3 slides on **clinical workflow integration**  
**Deliverables:** `reports/slides.pptx` (or Google Slides)

### Sept 12–13 (Fri–Sat) — Video & Rehearsal
**Tasks:**  
- Record **10–12 min** video (screen + voice)  
- Rehearse Q&A (synthetic justification, novelty, deployment, ethics)  
**Deliverables:** Final video + talking points

### Sept 14 (Sun) — Buffer Day
**Task:** Minor fixes; proofread; reference checks; repo sanity check

### Sept 15 (Mon) — Presentation Day
**Task:** Upload final materials; present 🚀

---

## Progress Checklist
> Tick as you go; add dates for traceability.

### Data & Unimodal
- [ ] CT YOLO ROI trained and inferred on slices
- [ ] CT classifier trained; patient-level scores exported
- [ ] CT embeddings (256D) saved per fold
- [ ] Urine preprocessing complete (impute/scale)
- [ ] Urine MLP (5-fold) trained; embeddings (32D) saved

### Synthetic Pairs
- [ ] CV-aware stratified pairing implemented
- [ ] R=3 resamples per fold saved
- [ ] Pairing sanity report (balance + KS) saved

### Fusion v1
- [ ] Late fusion (weights + logistic stacking) results saved
- [ ] Early concat head results saved
- [ ] Orthogonal fusion (λ sweep) results saved

### Attention & Contrastive
- [ ] Attention fusion head trained
- [ ] Attention maps generated & saved
- [ ] Contrastive pre-alignment trained
- [ ] Fusion heads re-run with contrastive embeddings

### Uncertainty & Federated
- [ ] MC Dropout + Temp Scaling implemented
- [ ] ECE + risk–coverage plots saved
- [ ] Federated (FedAvg) run vs centralized comparison

### Ablations
- [ ] CT with/without ROI comparison
- [ ] Modality dropout robustness
- [ ] Urine noise robustness
- [ ] 95% CI added to key metrics

### Figures & Docs
- [ ] Grad-CAM (CT) figures saved
- [ ] SHAP/Permutation (urine) figures saved
- [ ] README updated (Fusion Interface, Uncertainty, Federated, Figures)
- [ ] ADR-001 and ADR-002 updated/added

### Delivery
- [ ] Final results CSVs & master table exported
- [ ] Thesis manuscript finalized
- [ ] Slides deck completed
- [ ] Presentation video recorded

---

## Risks & Fallbacks
- **YOLO ROI slow:** Use pretrained or heuristic pancreas masks to produce ROI crops; still run ROI vs no-ROI ablation.  
- **Attention head underperforms:** Keep as negative result with analysis; rely on late/early/orthogonal for headline gains.  
- **Contrastive unstable:** Report as ablation; document seeds & stability.  
- **Federated slips:** Keep design description; run small 2-client simulation.

---

## Notes
- All fusion experiments operate on **embeddings**, keeping GPU demand low.  
- Synthetic fusion results are labeled **simulation**; the architecture is designed to be **drop-in** replaceable once real paired datasets are available.  
