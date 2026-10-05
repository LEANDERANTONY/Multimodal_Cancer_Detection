from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    pdf_path = Path(r"F:\My Drive\Multimodal_Cancer_Detection\CT_Project_Consolidated_Progress_Feb2026.pdf")
    out_path = Path(r"F:\My Drive\Multimodal_Cancer_Detection\CT_Project_Consolidated_Progress_Feb2026.txt")

    reader = PdfReader(str(pdf_path))
    lines: list[str] = []
    lines.append(str(pdf_path.name))
    lines.append("=" * 80)
    lines.append(f"Pages: {len(reader.pages)}")
    lines.append("")

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = "\n".join(line.rstrip() for line in text.splitlines())
        lines.append(f"\n--- Page {i} ---\n")
        lines.append(text)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
