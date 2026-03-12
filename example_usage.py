from __future__ import annotations

from pathlib import Path

from research_ai.parsing.paper_builder import parse_paper


def main() -> None:
    sample_pdf = Path("Research Paper Management & Analysis Intelligence System.pdf")
    paper = parse_paper(sample_pdf)

    print(f"Detected title: {paper.title}")
    print("Section names:")
    for section in paper.sections:
        print(f"- {section.section_name}")
    print(f"Number of references extracted: {len(paper.citations)}")

    output_path = Path("data") / "processed" / f"{sample_pdf.stem}.json"
    paper.export_json(output_path)
    print(f"Exported JSON to: {output_path}")


if __name__ == "__main__":
    main()
