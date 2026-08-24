# Paper 3 LaTeX package

Branding at the Water's Edge: Perceived Elite Polarization of Ukraine Aid, 2022-2024

## Build

Two independent documents, one shared preamble, one shared bibliography.

Default configuration is biblatex + biber (modern APA):

    Overleaf: compiler pdfLaTeX, bibliography engine Biber. Root document main.tex.

If you prefer classic BibTeX, edit preamble.tex and comment block (A), uncomment
block (B), then in main.tex and supplement.tex comment \printbibliography and
uncomment the two \bibliography lines.

## Do not load apacite in this preamble

apacite (2013) fails here. It errors while loading, which forces LaTeX to ship a
page before \begin{document} and produces a stray near-blank page ahead of the
title, and it disturbs the layout thereafter. Block (C) in preamble.tex records the
lines for reference only. Use biblatex with style=apa instead.

## Contents

    main.tex                 manuscript driver
    supplement.tex           supplementary materials driver
    preamble.tex             shared preamble; bibliography engine selected here
    latexmkrc                pins the root document to main.tex
    bib/references.bib       106 entries, local Zotero fields stripped
    sections/10_introduction.tex
    sections/20_literature_review.tex
    sections/30_method.tex        Tables 1-3
    sections/40_results.tex       Tables 4-8, Figures 1-3
    sections/50_discussion.tex
    sections/90_supplement.tex    Supplementary Methods S1-S8
    figures/                 three PNGs, pre-cropped

## Typography

Times New Roman throughout via mathptmx, with sfdefault and ttdefault remapped to
ptm so tables, math, listings, captions and the reference list all set in Times.
A4, one-inch margins, 11pt. Paragraphs are separated by a blank line with no
first-line indent, via the parskip package.

## Figures

The three PNGs are cropped copies of the images embedded in the Word manuscript.
Word applied a display crop; that crop has been applied to the files themselves,
so no trim option is needed. Original crop values were:

    Figure 1  bottom 13.668%
    Figure 2  top 4.876%, bottom 8.464%
    Figure 3  top 5.231%, bottom 8.118%
