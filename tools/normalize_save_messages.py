from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/01_multimodal_cancer_detection.ipynb")


def ensure_fmt_path(cells: list[dict]) -> None:
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        if not any("Cell 1.1: Environment + Setup Paths" in line for line in source):
            continue
        if any("def fmt_path" in line for line in source):
            return
        insert_at = None
        for idx, line in enumerate(source):
            if line.strip().startswith("def resolve_models_dir"):
                insert_at = idx + 2
        if insert_at is None:
            insert_at = len(source)
        source[insert_at:insert_at] = [
            "def fmt_path(p):\n",
            "    try:\n",
            "        p = as_path(p)\n",
            "        return str(p.relative_to(PROJ))\n",
            "    except Exception:\n",
            "        return str(p)\n",
            "\n",
        ]
        return


def normalize_saved_lines(cells: list[dict]) -> int:
    pat_colon = re.compile(r"print\(\".*Saved:\",\s*([^\)]+)\)")
    pat_f = re.compile(r"print\(f\".*Saved: \{([^}]+)\}\"\)")
    changes = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if not isinstance(source, list):
            continue

        new_lines: list[str] = []
        for line in source:
            new_line = line
            if "Saved:" in line and "fmt_path" not in line:
                m = pat_colon.search(line)
                if m:
                    expr = m.group(1).strip()
                    new_line = pat_colon.sub(
                        f'print("✅ Saved:", fmt_path({expr}))',
                        line,
                    )
                else:
                    m2 = pat_f.search(line)
                    if m2:
                        expr = m2.group(1).strip()
                        new_line = pat_f.sub(
                            f'print(f"✅ Saved: {{fmt_path({expr})}}")',
                            line,
                        )
            if new_line != line:
                changes += 1
            new_lines.append(new_line)
        cell["source"] = new_lines

    return changes


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    ensure_fmt_path(cells)
    changes = normalize_saved_lines(cells)
    nb["cells"] = cells
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")
    print(f"Normalized {changes} save messages")


if __name__ == "__main__":
    main()
