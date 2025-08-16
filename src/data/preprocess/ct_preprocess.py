
from pathlib import Path
import os, csv
import numpy as np
import SimpleITK as sitk
import cv2

# Optional pandas/matplotlib (used for summaries & figures)
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from PIL import Image  # used for quick PNG checks

# -------- CONFIG (matches your folder layout) --------
RAW_ROOT = Path("data/raw/ct")

# Study subfolders under RAW_ROOT
CANCER_STUDY_DIR  = "Pancreatic-CT-CBCT-SEG"   # ct/cancer/<this>/
CONTROL_STUDY_DIR = "Pancreas-CT"              # ct/control/<this>/

# Optional NBIA digests (drop the xlsx here; if missing, script falls back)
CANCER_DIGEST_XLSX  = RAW_ROOT / "cancer"  / "cancer_digest.xlsx"   # Pancreatic-CT-CBCT-SEG_v2_20220823-nbia-digest.xlsx
CONTROL_DIGEST_XLSX = RAW_ROOT / "control" / "control_digest.xlsx"  # Pancreas-CT-20200910-nbia-digest.xlsx

OUT_ROOT    = Path("data/processed/ct_images")
INDEX_CSV   = Path("data/processed/ct_index.csv")
INDEX_CLEAN = Path("data/processed/ct_index_clean.csv")
DROP_LOG    = Path("data/processed/dropped_slices_log.csv")
PAT_SUMMARY = Path("data/processed/ct_patient_summary.csv")
FIGURES_ROOT = Path("figures")  # <-- project-root figures

TARGET_SIZE = (224, 224)
WINDOW_LEVEL, WINDOW_WIDTH = 50, 400
MIN_SLICES = 20

# Clean criteria
MIN_FILE_BYTES = 1000   # <1KB likely empty/broken
MIN_STD_PIX    = 1.0    # near-blank slice if std < 1.0
# ----------------------------------------------------

def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

def to_window_uint8(hu_slice, wl=WINDOW_LEVEL, ww=WINDOW_WIDTH):
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    x = np.clip(hu_slice, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8) * 255.0
    return x.astype(np.uint8)

def modality_of(dcm_path: str) -> str:
    img = sitk.ReadImage(dcm_path)
    return img.GetMetaData("0008|0060") if img.HasMetaDataKey("0008|0060") else ""

def list_series(dirpath: Path):
    """
    Return list of (series_uid, files[list]) for all series under dirpath.
    """
    reader = sitk.ImageSeriesReader()
    series = []
    try:
        uids = reader.GetGDCMSeriesIDs(str(dirpath))
    except Exception:
        uids = None
    if not uids:
        return series
    for uid in uids:
        files = reader.GetGDCMSeriesFileNames(str(dirpath), uid)
        if files:
            series.append((uid, files))
    return series

def find_all_series_recursive(root: Path):
    """
    Recursively gather all (series_uid, files[list]) under root.
    """
    all_series = []
    for dirpath, _, _ in os.walk(root):
        dirpath = Path(dirpath)
        all_series.extend(list_series(dirpath))
    return all_series

def pick_series_by_uid(all_series, preferred_uid: str):
    """
    If preferred_uid is present, return its files; else None.
    """
    if not preferred_uid:
        return None
    for uid, files in all_series:
        if uid == preferred_uid and len(files) >= MIN_SLICES:
            # Confirm Modality=CT
            try:
                if modality_of(files[0]) != "CT":
                    continue
            except Exception:
                continue
            return files
    return None

def pick_largest_ct_series(all_series):
    """
    Fallback: choose CT series with most slices (>= MIN_SLICES).
    Returns (files, series_uid) or (None, None).
    """
    best_files = None
    best_uid = None
    best_len = -1
    for uid, files in all_series:
        try:
            if modality_of(files[0]) != "CT":
                continue
        except Exception:
            continue
        if len(files) >= MIN_SLICES and len(files) > best_len:
            best_files = files
            best_uid = uid
            best_len = len(files)
    return best_files, best_uid

def read_series(files):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    vol = reader.Execute()                  # 3D
    arr = sitk.GetArrayFromImage(vol)       # [Z, H, W], HU-scaled by SimpleITK
    return arr

def save_patient_slices(patient_id: str, label: str, vol_zhw, series_uid: str,
                        series_description: str = "", study_date: str = "", from_digest: bool = False):
    out_dir = OUT_ROOT / label / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in range(vol_zhw.shape[0]):
        sl = to_window_uint8(vol_zhw[z])
        sl = cv2.resize(sl, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"slice_{z:03d}.png"
        cv2.imwrite(str(out_path), sl)
        rows.append({
            "patient_id": patient_id,
            "label": label,
            "slice_idx": z,
            "path": str(out_path).replace("\\", "/"),
            "series_uid": series_uid,
            "series_description": series_description,
            "study_date": study_date,
            "from_digest": int(from_digest)
        })
    return rows

def infer_patient_id(p: Path):
    # pick the directory name containing digits (e.g., Pancreas-CT-CB_001, PANCREAS_0001)
    name = p.name
    if any(ch.isdigit() for ch in name):
        return name
    for part in reversed(p.parts):
        if any(ch.isdigit() for ch in part):
            return part
    return name

def load_digest_map(xlsx_path: Path, label: str):
    """
    Build maps using NBIA digest:
      patient_id -> (series_uid, series_description, study_date)
    If digest missing or pandas unavailable, returns {}.
    """
    if pd is None or not xlsx_path.exists():
        print(f"[{label}] digest not used (missing file or pandas). Path: {xlsx_path}")
        return {}

    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"[{label}] failed to read digest {xlsx_path}: {e}")
        return {}

    # Robust column detection
    id_cols = ["Patient ID","Subject ID","PatientID","Patient Id","SubjectId"]
    id_key = next((k for k in id_cols if k in df.columns), None)
    if id_key is None:
        print(f"[{label}] could not find patient id column in digest. Columns: {list(df.columns)[:10]}")
        return {}

    # Optional columns
    if "Image Count" not in df.columns: df["Image Count"] = 0
    if "Series Number" not in df.columns: df["Series Number"] = 0
    if "Series Description" not in df.columns: df["Series Description"] = ""
    if "Study Date" not in df.columns: df["Study Date"] = ""

    # Sort & pick top series per patient
    pick = (df.sort_values([id_key,"Image Count","Series Number"], ascending=[True,False,True])
              .groupby(id_key, as_index=False)
              .head(1)[[id_key,"Series UID","Series Description","Image Count","Series Number","Study Date"]])

    mapping = {}
    for _, row in pick.iterrows():
        pid = str(row[id_key])
        suid = str(row["Series UID"]) if pd.notna(row["Series UID"]) else ""
        sdesc = str(row["Series Description"]) if pd.notna(row["Series Description"]) else ""
        sdate = str(row["Study Date"]) if pd.notna(row["Study Date"]) else ""
        if pid and suid:
            mapping[pid] = (suid, sdesc, sdate)
    print(f"[{label}] digest loaded: {len(mapping)} patient→SeriesUID mappings")
    return mapping

def collect_patient_roots(label: str):
    """
    For 'cancer':  data/raw/ct/cancer/Pancreatic-CT-CBCT-SEG/<PATIENT>/
    For 'control': data/raw/ct/control/Pancreas-CT/<PATIENT>/
    """
    if label == "cancer":
        study_root = RAW_ROOT / "cancer" / CANCER_STUDY_DIR
    else:
        study_root = RAW_ROOT / "control" / CONTROL_STUDY_DIR

    if not study_root.exists():
        print(f"[warn] missing: {study_root}")
        return []

    patients = [p for p in study_root.iterdir() if p.is_dir()]
    return patients

def process_label(label: str, digest_map: dict):
    total_rows = []
    patients = collect_patient_roots(label)
    print(f"[{label}] found {len(patients)} patient folders")

    for p in patients:
        pid = infer_patient_id(p)
        print(f" -> {pid}: scanning CT series…")
        all_series = find_all_series_recursive(p)
        if not all_series:
            print("    (skip) no DICOM series found")
            continue

        # Try digest-preferred series first
        series_uid = ""
        series_desc = ""
        study_date = ""
        used_digest = False

        files = None
        if pid in digest_map:
            pref_uid, pref_desc, pref_date = digest_map[pid]
            files = pick_series_by_uid(all_series, pref_uid)
            if files is not None:
                series_uid = pref_uid
                series_desc = pref_desc
                study_date = pref_date
                used_digest = True

        # Fallback to largest CT series
        if files is None:
            files, fallback_uid = pick_largest_ct_series(all_series)
            series_uid = fallback_uid or ""

        if not files:
            print("    (skip) no suitable CT series found")
            continue

        try:
            vol = read_series(files)
            rows = save_patient_slices(
                patient_id=pid,
                label=label,
                vol_zhw=vol,
                series_uid=series_uid,
                series_description=series_desc,
                study_date=study_date,
                from_digest=used_digest
            )
            print(f"    saved {len(rows)} slices")
            total_rows.extend(rows)
        except Exception as e:
            print(f"    (error) {e}")
    return total_rows

# ---------- NEW: cleaning & summaries/figures ----------
def _is_low_value_png(path: str) -> (bool, str):
    # Criterion A: tiny file
    try:
        if os.path.getsize(path) < MIN_FILE_BYTES:
            return True, "tiny_file"
    except OSError:
        return True, "missing"

    # Criterion B: near-blank by pixel std
    try:
        with Image.open(path) as im:
            arr = np.array(im, dtype=np.uint8)
        if arr.std() < MIN_STD_PIX:
            return True, "low_std"
    except Exception as e:
        return True, f"open_error:{type(e).__name__}"

    return False, ""

def write_clean_index_and_logs(rows):
    """Create ct_index.csv (already done), plus ct_index_clean.csv & drop log."""
    if pd is None:
        print("[clean] pandas not available; skipping clean index & summary.")
        return

    df = pd.DataFrame(rows)
    # Clean
    drops = []
    keep = []
    for _, r in df.iterrows():
        bad, why = _is_low_value_png(r["path"])
        if bad:
            drops.append({**r, "drop_reason": why})
            keep.append(False)
        else:
            keep.append(True)
    df_clean = df[keep].reset_index(drop=True)
    dropped = pd.DataFrame(drops)

    df_clean.to_csv(INDEX_CLEAN, index=False)
    print(f"[clean] wrote clean index: {INDEX_CLEAN} ({len(df_clean)} kept / {len(df)} total)")

    if len(dropped):
        dropped.to_csv(DROP_LOG, index=False)
        print(f"[clean] wrote drop log:   {DROP_LOG} ({len(dropped)} dropped)")
    else:
        print("[clean] no slices dropped")

    # Patient summary
    pat_summary = (df_clean.groupby(["patient_id","label"])
                            .size().reset_index(name="num_slices"))
    pat_summary.to_csv(PAT_SUMMARY, index=False)
    print(f"[clean] wrote patient summary: {PAT_SUMMARY}")

    # Figures (if matplotlib available)
    if plt is not None:
        try:
            # Boxplot of slice counts by label
            plt.figure(figsize=(6,4))
            pat_summary.boxplot(by="label", column="num_slices")
            plt.title("Distribution of slices per patient")
            plt.suptitle("")
            plt.ylabel("Num slices")
            plt.savefig(FIGURES_ROOT / "slice_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[fig] saved {FIGURES_ROOT / 'slice_distribution.png'}")

            # Example grids (first cancer & control if exist)
            def save_grid(pid, title, outpath):
                rows_pid = df_clean[df_clean["patient_id"]==pid].sort_values("slice_idx").head(12)
                if rows_pid.empty:
                    return
                fig, axs = plt.subplots(3,4, figsize=(8,6))
                for ax, (_, rr) in zip(axs.ravel(), rows_pid.iterrows()):
                    with Image.open(rr["path"]) as im:
                        ax.imshow(im, cmap="gray")
                    ax.set_title(f"z={int(rr['slice_idx'])}")
                    ax.axis("off")
                fig.suptitle(title)
                plt.tight_layout()
                plt.savefig(outpath, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"[fig] saved {outpath}")

            cancer_ids = df_clean[df_clean["label"]=="cancer"]["patient_id"].unique()
            control_ids = df_clean[df_clean["label"]=="control"]["patient_id"].unique()
            if len(cancer_ids):
                save_grid(cancer_ids[0], f"Cancer – {cancer_ids[0]}", FIGURES_ROOT / "example_cancer_grid.png")
            if len(control_ids):
                save_grid(control_ids[0], f"Control – {control_ids[0]}", FIGURES_ROOT / "example_control_grid.png")
        except Exception as e:
            print(f"[fig] skipping figures due to error: {e}")
    else:
        print("[fig] matplotlib not available; skipping figure generation.")

# ------------------------------------------------------

def main():
    ensure_dirs()

    # Load digests (optional, safe to miss)
    cancer_map  = load_digest_map(CANCER_DIGEST_XLSX,  "cancer")
    control_map = load_digest_map(CONTROL_DIGEST_XLSX, "control")

    rows = []
    rows += process_label("cancer",  cancer_map)
    rows += process_label("control", control_map)

    # Write full index
    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id","label","slice_idx","path",
            "series_uid","series_description","study_date","from_digest"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {len(rows)} rows to {INDEX_CSV}")

    # NEW: write clean index, logs, and figures
    write_clean_index_and_logs(rows)

if __name__ == "__main__":
    main()

