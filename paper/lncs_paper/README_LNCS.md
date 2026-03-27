# LNCS Submission Scaffold (ICCBR 2026)

This folder contains a Springer LNCS-ready scaffold.

## Files
- `main.tex` — main entry
- `sections/*.tex` — chaptered content
- `references.bib` — BibTeX database
- `figures/*.png` — copied paper figures

## Compile
Use a LNCS environment with `llncs.cls` and `splncs04.bst` available.

Typical build order:
1. `pdflatex main.tex`
2. `bibtex main`
3. `pdflatex main.tex`
4. `pdflatex main.tex`

## ICCBR Notes
- Main paper <= 14 pages.
- Up to 2 extra pages for acknowledgments, generative AI statement, and references.
- If under simultaneous submission elsewhere, add page-1 footnote and notify chairs.

## Next Step
Paste your current paper section-by-section into `sections/*.tex` and send me each section for polishing.
