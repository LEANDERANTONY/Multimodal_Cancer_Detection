# Journal requirements for paper 1

Paper 1 (scanner confound, deployment, mitigation panel; draft `paper/manuscript_draft_v6.md`, local only) goes to
**Medical Image Analysis (MedIA) first**, at its current length. **Radiology: Artificial Intelligence** is the fallback.
Read from the journals' own pages on 2026-10-27; recheck both before submitting:
- MedIA guide for authors: https://www.sciencedirect.com/journal/medical-image-analysis/publish/guide-for-authors
- Radiology: AI author instructions: https://pubs.rsna.org/page/ai/author-instructions; RSNA editorial policies:
  https://pubs.rsna.org/page/policies

Author decisions (2026-10-27): sole author, no co-authors for now (revisit after a rejection); no funding; no competing
interests. Data licence: the PANORAMA release is CC BY-NC 4.0 (Zenodo record 10999754), so any released model weights
must also be non-commercial.

## 0. Packages (built 2026-10-27, local only, git-ignored)
- `paper/medIA/`: title page, manuscript (245-word abstract, keywords, author–year references, declarations),
  highlights, Statement of Significance (all 18 answers), vitae template, CLAIM table, figures incl. graphical abstract;
  `README.md` lists every requirement and what is left for the author.
- `paper/radiology_ai/`: full title page, cover letter, anonymised 2,254-word manuscript (structured abstract, summary
  statement, key points, 25 numbered references, 3 tables), Appendix E1, CLAIM table, figures; own `README.md`.
- Master draft for both: `paper/manuscript_draft_v7.md` (includes the orientation finding,
  `docs/orientation_investigation.md`). Model weights staged by `tools/build_release.py` for the author's Zenodo upload.
- Left for the author in both: affiliation and contact details, ORCID, AI-use wording, Zenodo DOI, .docx conversion,
  official CLAIM form; MedIA also the vitae and a check of the PANORAMA collective-author name.

## 1. Medical Image Analysis (Elsevier; IF 14.0, CiteScore 25.8 as shown on the guide page)

### Review and submission
- Single-anonymised review, at least two reviewers. A paper rejected by MedIA may not be resubmitted to MedIA.
- Elsevier's Article Transfer Service may offer a transfer to another Elsevier journal after a rejection.
- Source files: Word (.docx, single column) or LaTeX (the Elsevier template is encouraged). A PDF is not accepted as
  the source file.
- Figures are uploaded as separate files (TIFF/JPG/PNG; photographs at least 300 dpi, line art at least 1000 dpi). Tables
  must be editable text, not images.
- Preprints are allowed and do not count as prior publication.

### Manuscript parts
| Item | Requirement | Draft v6 status |
|---|---|---|
| Length | No word limit stated for research papers | ~6,000 words: keep |
| Abstract | At most 250 words, no references, abbreviations defined | ~300 words: **cut ~50** |
| Keywords | 1–7, avoid multi-word phrases joined by "and"/"of" | **to write** |
| Highlights | 3–5 bullets, each at most 85 characters including spaces; separate file named "highlights" | **to write** |
| Graphical abstract | **Required**; at least 1328 px wide, landscape, readable at 13 cm; TIFF/EPS/PDF/Office | **to make** |
| Sections | Numbered (1, 1.1, 1.1.1); a Discussion section is required; abstract not numbered | already numbered |
| Title page | Title, author name, affiliation, corresponding-author email (published) | **to write** |
| References | **Author–year** in text ("Kramer et al. (2023)", "(Jones, 2019; Allan, 2020)"); list alphabetical, then chronological; journal names abbreviated by LTWA; DOIs where available; software and models cited like papers; preprints labelled | numbered Vancouver: **convert** |
| Declaration of competing interests | Elsevier declarations tool; "I have nothing to declare" | to complete at submission |
| Funding | "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors." | **to add** |
| Declaration of generative AI use | **Required**: a section before the references, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", naming the tool, why it was used, and that the author reviewed the content and takes full responsibility | **to draft; the author writes the final wording** |
| CRediT statement | Contributor roles (sole author: all applicable roles) | **to add** |
| Acknowledgements | Separate section directly before the references | optional |
| Vitae | Biography of each author, at most 100 words, editable format | **author to write** |
| Data statement | Required at submission; Option C research-data policy: deposit the data in a repository and cite and link it in the article, or explain why not | **plan: Zenodo record (code, per-case scores, weights) with a DOI, cited in the paper** |
| Figures with generative AI | Not allowed for result images; any AI help with diagrams must be disclosed in the caption | our figures are code-generated: fine |
| Reporting checklists | Not required by MedIA; TRIPOD+AI / CLAIM drafts can go in as supplementary material | drafted in `paper/reporting_checklists.md` |

### Statement of Significance (required file; an irrelevant or incomplete answer means desk rejection with no resubmission)
All questions and answers are copied into one Word or PDF document. Planned answers and risks:

| Q | Asks | Plan / risk |
|---|---|---|
| 1 | Previously submitted to this or another journal? | No (first submission) |
| 2 | Previously published, e.g. at a conference? | No |
| 3 | 3–5 Associate Editors with a reason each; not in the author's country; no prior collaborators; no Executive Committee or Senior Editors | **to draft from the editorial board page** |
| 4 | Methodological innovation in 1–2 sentences (at most 300 words) | **Main risk for an audit paper.** Frame as evaluation methodology: the confound tax with patient-level CIs as a measurable target; the planted-shortcut positive control for debiasing studies; intention-to-treat evaluation of two-stage deployment with predicted masks |
| 5 | Difference from the authors' prior publications (full reference) | The Solar Energy 2020 paper (unrelated field); state that there is no prior imaging publication |
| 6 | 3–5 closest published papers (title, DOI, first and last author, journal, year) and how each differs | From `docs/literature_novelty_check.md`: PANORAMA (Alves et al. 2026), MEDFAIR (Zong et al. 2023), Yang et al. Nat Med 2024, Ong Ly et al. npj Digit Med 2024, Wasserthal et al. (TotalSegmentator) 2023 |
| 7 | Is the method significantly better than the state of the art? | **Honest no**: the contribution is the audit, not a new detector; our 3D model is below the challenge winner (0.787 vs 0.926), which the paper acknowledges |
| 8 | Full resolution and full field of view? If not, justify | The 3D model works at native resolution on pancreas crops (standard two-stage practice); the deployment experiment starts from the full scan; the 2.5D classifier uses slice triplets at 1 × 1 × 2 mm. Justify both in the paper |
| 9 | Figures showing findings across the full spatial extent in at least two orthogonal planes | Done: Fig. 3 (`figures/panorama/fig_failure_case.png`): coronal as stored, coronal anatomical, axial; oracle and predicted crops |
| 10 | Confirm the novelty information (Q4–7) is in the manuscript | Check that the Introduction says it explicitly |
| 11 | Reproducibility: several datasets with at least one public; stratified CV or external tests; **no leakage — was a subject confined to one fold?** | PANORAMA Dutch + MSD + NIH, all public; 5-fold CV, LOMO, external. Answer the leakage item truthfully: folds were drawn by scan; every Dutch metric is computed on the 1,939 single-examination patients, so no test patient contributed to training (`tools/eval_cohort.py`) |
| 12 | Per dataset: public?, sample size, all used?, selection criteria, train/test numbers | Dutch 1,964 (1,939 evaluated), MSD 194, NIH 80; all used |
| 13 | Potential biases in the data and how they are handled | This is the paper's subject: scanner/case-mix confound, the confound tax, LOMO, the planted control; no site identifier; two Dutch centres |
| 14 | Data and code public? | Data: public (PANORAMA, CC BY-NC 4.0). Code, per-case scores, weights: Zenodo DOI + GitHub |
| 15 | Confirm the reproducibility information (Q9–12) is in the manuscript | Check |
| 16 | Written with the help of an LLM? Which one, how, how integrity was ensured (including that citations are correct); confirm this is in the manuscript | Yes: name the tool and its use; citations were checked against PubMed/arXiv. **Author's wording** |
| 17 | Confirm no patient-identifying information | Yes (de-identified public data; image figures must carry no identifiers) |
| 18 | Comments to the publication office | Optional |

## 2. Radiology: Artificial Intelligence (RSNA; fallback)

### Article type: Original Research
| Item | Requirement | Draft v6 gap |
|---|---|---|
| Length | **At most 3,000 words**, Introduction to Discussion | ~6,000: **halve** (move detail to the supplement, 12 pages max) |
| Abstract | Structured, at most 250 words: Purpose / Materials and Methods (retrospective or prospective, date range, patients with age and sex, groups, procedures, one sentence on statistics) / Results (numbers, percentages, significance) / Conclusion (from the results only) | restructure |
| Summary statement | One sentence, at most 255 characters, bold, no abbreviations | to write |
| Key Points | Up to 3, with summary data, no vague language, minimal abbreviations | to write |
| Limits | 35 references, 6 figures (at most 4 parts each), 4 tables | 29 refs fine; figures and tables to cut to fit |
| Abbreviations | **At most 10**, spelled out at first use; none in the summary statement | many now (PDAC, AUROC, ERM, DFR, GRL, MAE, LOMO, MSD, NIH, ECE...): **reduce** |
| Style | No self-evaluation ("novel", "unique"); written for a general radiologist; no first-person self-citation | rephrase |
| Main body | Introduction (ends with hypothesis and purpose) / Materials and Methods / Results / Discussion; limitations in the second-to-last Discussion paragraph; final paragraph summarises; no tables or figures cited in the Discussion | restructure (our numbered conclusions list would go) |
| Methods specifics | First paragraph: IRB approval/waiver and consent; retrospective; date range; subjects with age and sex; inclusion/exclusion with counts per cause; sample-size reasoning; last paragraph: statistical methods, software with versions, P-value threshold | add the ethics/waiver wording for public de-identified data, the P threshold and software versions |
| References | **Numbered** in order of first mention, NLM abbreviations, DOIs | current style already fits |
| Review | **Double-anonymised**: anonymised title page and acknowledgements; no institution or funding in the manuscript | anonymise |
| Files | .docx only (from January 2026); for a new submission, all figures in one document with their legends; cover letter; full title page; anonymised manuscript | prepare |
| Cover letter | Title, authors, subject-overlap statement, conflicts of interest, confirmation of sole submission, AI-use disclosure | prepare |
| Full title page | Author details, funding, manuscript type, word count, **data sharing statement** using one of RSNA's verbatim sentences (e.g. "Data generated by the authors or analyzed during the study are available at: [citation]") | prepare |
| Checklist | **CLAIM required at first submission** (the paper is returned without it) | draft exists: transfer to the official CLAIM Word form |
| Code | Deposit all modelling/analysis code in a public repository on publication; give the link and the revision identifier in Methods; keep it for 5 years | GitHub + tagged release / Zenodo DOI |
| AI use | Disclose in the cover letter **and** the manuscript how AI or AI-assisted tools were used; AI cannot be an author | author's wording |
| Figures | Legends give image type, plane, contrast, what to look at; no identifiers; CT shown "viewed from below"; revisions need TIF/EPS (no PNG) | the image figure would need this |
| Other | Submission is free (open access optional, charged); ORCID required; preprints allowed (add the RSNA notice when posting the submitted version); after rejection, transfer within RSNA journals (Radiology: Imaging Cancer, Radiology Advances) is possible | — |

## 3. Shared to-do (useful for either journal)
1. Image figure in two orthogonal planes (MedIA Q9; also strengthens a radiology paper).
2. Zenodo release: code at a tagged commit, per-case score tables, 3D tight CV final checkpoints (~1.2 GB) and the 15
   ERM 2.5D soups (~0.7 GB), licence CC BY-NC 4.0; cite its DOI. The author decides on and performs the upload.
3. AI-use declaration in the author's words (both journals require it).
4. Ethics wording: retrospective analysis of public, de-identified data; no ethics approval required.
5. Reporting checklist: CLAIM (required by Radiology: AI; supplementary for MedIA).
