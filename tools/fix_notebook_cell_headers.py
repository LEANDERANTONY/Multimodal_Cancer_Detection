import json
import re
from pathlib import Path


def infer_phase_prefix(current_phase: str | None) -> int | None:
    if not current_phase:
        return None
    m = re.match(r"^##\s*Phase\s+(\d+)\b", current_phase.strip())
    if not m:
        return None
    return int(m.group(1))


def infer_description(first_line: str) -> str:
    s = first_line.strip()

    if s.startswith("# Cell:"):
        return s[len("# Cell:") :].strip() or "Notebook step"

    # e.g. '# 2.9 Evaluation on Test Set'
    m = re.match(r"^#\s*(\d+\.\d+)\s*(.*)$", s)
    if m:
        rest = m.group(2).strip()
        return rest or "Notebook step"

    if s.startswith("#"):
        return s.lstrip("#").strip() or "Notebook step"

    if s.startswith("!"):
        return "Shell command"

    if s.startswith("import ") or s.startswith("from "):
        return "Imports"

    if s.startswith("df = pd.read_csv") or "read_csv" in s:
        return "Load table"

    if s.startswith("show_random_slices"):
        return "Slice visualization"

    if s.startswith("# ===") or s.startswith("# ====") or s.startswith("# ==="):
        return "Section divider"

    if s.startswith("# =================") or s.startswith("# ==="):
        return "Section divider"

    return "Notebook step"


def main() -> None:
    nb_path = Path("notebooks/01_multimodal_cancer_detection.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    cells = nb.get("cells", [])

    current_phase = None
    edits = 0

    for cell_idx, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type")
        src = cell.get("source") or []

        if ctype == "markdown":
            text = "".join(src)
            for line in text.splitlines():
                if line.strip().startswith("## Phase "):
                    current_phase = line.strip()
                    break

        if ctype != "code":
            continue

        # find first non-empty line
        first_i = None
        first_line = None
        for i, line in enumerate(src):
            if isinstance(line, str) and line.strip():
                first_i = i
                first_line = line.rstrip("\n")
                break

        if first_line is None:
            # empty code cell; insert header
            phase_prefix = infer_phase_prefix(current_phase) or 0
            header = f"# Cell {phase_prefix}.{cell_idx}: Notebook step"
            cell["source"] = [header]
            edits += 1
            continue

        if first_line.lstrip().startswith("# Cell "):
            continue

        phase_prefix = infer_phase_prefix(current_phase)
        if phase_prefix is None:
            # if phase unknown, keep generic 0
            phase_prefix = 0

        desc = infer_description(first_line)
        header = f"# Cell {phase_prefix}.{cell_idx}: {desc}"

        # insert header before the first non-empty line
        new_src = list(src)
        insert_at = first_i if first_i is not None else 0
        new_src.insert(insert_at, header)
        new_src.insert(insert_at + 1, "")
        cell["source"] = new_src
        edits += 1

    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Updated headers in {edits} code cells")


if __name__ == "__main__":
    main()
