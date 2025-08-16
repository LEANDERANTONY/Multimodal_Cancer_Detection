
# 🩺 Multimodal Pancreatic Cancer Detection (CT + Urine Biomarkers)

> **Dual-branch deep learning pipeline** combining CT scan ROI detection and urine biomarker analysis to aid early pancreatic cancer diagnosis.  
> Evaluates **multiple multimodal fusion strategies** (Late, Early, Orthogonal) on synthetic paired datasets to inform future clinical applications.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Data Sources](#-data-sources)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Limitations & Future Work](#-limitations--future-work)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Citations](#-citations)
- [License](#-license)

---

## 🚀 Overview
This project explores **multimodal fusion** of:
1. **CT Imaging** — Processed with **YOLOv8** to detect regions of interest (ROI) and **CNN** for classification.
2. **Urine Biomarkers** — Modeled via a **Multi-Layer Perceptron (MLP)**.

The pipeline tests multiple **fusion strategies** using **synthetic pairing** of urine and CT data due to the unavailability of real paired datasets.

**Why it matters:**  
Pancreatic cancer often goes undetected until late stages. Combining imaging with biomarker data could improve early diagnosis — our results provide insight into which fusion strategies may perform best when paired datasets become available.

---

## 🏗 Architecture
![Pipeline Diagram](docs/thesis_plan_update.png) <!-- Replace with actual pipeline diagram -->

**Steps:**
1. **Data Preprocessing**
   - CT DICOM → PNG → ROI extraction with YOLO
   - Urine biomarker cleaning & normalization
2. **Unimodal Training**
   - CT ROI → CNN classifier
   - Urine biomarkers → MLP classifier
3. **Fusion**
   - Late Fusion (logit averaging)
   - Early Fusion (feature concatenation)
   - Orthogonal Fusion (modality-specific projection layers)
4. **Evaluation**
   - Compare strategies on synthetic paired dataset
5. **Visualization**
   - Grad-CAM for CT slices
   - SHAP plots for urine features

---

## 📊 Data Sources
- **CT Imaging**: NCI Pancreatic CT dataset ([TCIA](https://www.cancerimagingarchive.net/))
- **Urine Biomarkers**: Kaggle — Early Pancreatic Cancer Urinary Biomarker Dataset ([Link](https://www.kaggle.com/))
- Synthetic pairing generated for fusion experiments.

---

## 🔬 Methodology
### **1. CT Imaging Branch**
- ROI detection with **YOLOv8** trained on annotated CT scans.
- Classification with **ResNet50** fine-tuned for pancreatic tumor detection.

### **2. Urine Biomarker Branch**
- Preprocessing: Missing value imputation, z-score normalization.
- Classification with **MLP** (2–3 dense layers + dropout).

### **3. Fusion Strategies**
- **Late Fusion**: Combine final prediction logits from both models.
- **Early Fusion**: Concatenate penultimate layer embeddings.
- **Orthogonal Fusion**: Learn modality-specific projections before fusion.

---

## 📈 Results
| Modality      | Model         | Accuracy | Sensitivity | Specificity |
|---------------|--------------|----------|-------------|-------------|
| CT only       | ResNet50      | `<X.XX>` | `<X.XX>`    | `<X.XX>`    |
| Urine only    | MLP           | `<X.XX>` | `<X.XX>`    | `<X.XX>`    |
| Fusion (Late) | MLP + CNN     | `<X.XX>` | `<X.XX>`    | `<X.XX>`    |
| Fusion (Early)| MLP + CNN     | `<X.XX>` | `<X.XX>`    | `<X.XX>`    |
| Fusion (Orth) | MLP + CNN     | `<X.XX>` | `<X.XX>`    | `<X.XX>`    |

---

## 🎨 Visualizations
### **CT Grad-CAM**
![GradCAM Example](docs/gradcam_example.png)

### **Urine Biomarker SHAP**
![SHAP Example](docs/shap_example.png)

---

## ⚠ Limitations & Future Work
- **Limitations**:
  - No real paired dataset for urine + CT
  - Fusion tested on synthetic pairing
- **Future Work**:
  - Apply on real paired datasets
  - Incorporate blood biomarkers
  - Deploy in a clinical decision-support setting

---

## 📄 Documentation
- [ADR-001: Scope Change & Fusion Strategy Decisions](architectural_decision_records/ADR-001.md)

