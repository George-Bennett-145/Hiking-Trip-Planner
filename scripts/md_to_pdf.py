"""Convert ARCHITECTURE.md to ARCHITECTURE.pdf.

Uses the `markdown` package to render the markdown to HTML, then
xhtml2pdf to print that HTML to PDF. The CSS keeps things readable and
gives code blocks a monospace font with a light background.

Run from the project root:
    .\.venv\Scripts\python.exe scripts\md_to_pdf.py
"""

from pathlib import Path

import markdown
from xhtml2pdf import pisa


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ARCHITECTURE.md"
DST = ROOT / "ARCHITECTURE.pdf"

CSS = """
@page { size: A4; margin: 2cm; }
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #222;
}
h1 { font-size: 22pt; color: #1a3d2e; margin-top: 0; }
h2 { font-size: 15pt; color: #1a3d2e; border-bottom: 1px solid #ccc;
     padding-bottom: 3pt; margin-top: 20pt; }
h3 { font-size: 12pt; color: #1a3d2e; margin-top: 14pt; }
h4 { font-size: 11pt; color: #1a3d2e; }
p  { margin: 6pt 0; }
ul, ol { margin: 6pt 0 6pt 18pt; }
li { margin: 2pt 0; }
code {
    font-family: Consolas, "Courier New", monospace;
    background: #f3f3ee;
    padding: 1pt 3pt;
    font-size: 9.5pt;
}
pre {
    font-family: Consolas, "Courier New", monospace;
    background: #f6f6f1;
    border: 1px solid #e0e0d8;
    padding: 8pt;
    font-size: 9pt;
    line-height: 1.35;
    white-space: pre-wrap;
}
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 4pt 6pt; font-size: 10pt;
         text-align: left; }
th { background: #f0f0eb; }
hr { border: 0; border-top: 1px solid #ccc; margin: 14pt 0; }
blockquote {
    border-left: 3px solid #1a3d2e;
    padding: 4pt 10pt;
    color: #444;
    background: #f7f7f2;
    margin: 8pt 0;
}
a { color: #1a3d2e; text-decoration: none; }
"""


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    with DST.open("wb") as out:
        result = pisa.CreatePDF(html_doc, dest=out, encoding="utf-8")

    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s).")
    print(f"Wrote {DST}  ({DST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
