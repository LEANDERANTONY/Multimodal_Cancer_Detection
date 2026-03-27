import json
from pathlib import Path


def main() -> None:
    nb_path = Path("notebooks/01_multimodal_cancer_detection.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    cells = nb.get("cells", [])

    missing = []
    current_phase = None

    for idx, cell in enumerate(cells, start=1):
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

        first = None
        for line in src:
            if isinstance(line, str) and line.strip():
                first = line.strip("\n")
                break

        if first is None or not first.lstrip().startswith("# Cell "):
            missing.append(
                {
                    "cell_number": idx,
                    "cell_id": cell.get("id") or cell.get("metadata", {}).get("id"),
                    "phase": current_phase or "(no phase)",
                    "first_line": (first or "<EMPTY>")[:160],
                }
            )

    print("Total cells:", len(cells))
    print("Total code cells:", sum(1 for c in cells if c.get("cell_type") == "code"))
    print("Missing '# Cell ...' header:", len(missing))
    print()

    for item in missing:
        print(
            f"{item['cell_number']:>3} | {item['phase']:<32} | {item['cell_id']} | {item['first_line']}"
        )

    out_path = Path("reports/missing_cell_headers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(missing, indent=2), encoding="utf-8")
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
