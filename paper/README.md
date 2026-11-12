# Vision Tower Bench paper

`main.pdf` contains an eight-page main paper, followed by references and a supplementary
appendix. It uses 11-point text and one-inch margins. Page limits exclude references and
appendix; the document is not formatted for a particular venue.

From the repository root:

```sh
uv run --group figures python paper/build_assets.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Python needs Matplotlib. The LaTeX build uses `pdflatex`, BibTeX, and `latexmk`.
`latexmk` resolves citations and cross-references through the required passes.
The main-paper endpoint is recorded as `page:main-end` in `main.aux`.

## Evidence and generated assets

- `sections/` contains the main narrative and appendix.
- `build_assets.py` generates all four figures and twelve numerical tables from committed
  result JSONs. The architecture roster is written in `sections/design.tex`.
- `asset-sources.json` records SHA-256 hashes for the 140 result inputs consumed by the
  asset generator. Regenerate it with the figures and tables.
- Paired intervals come from their own bootstrap records. They must not be reconstructed
  by subtracting rounded matrix means. Recognition interval runs and label-budget runs
  differ slightly; matched geometry intervals use deterministic PCA reruns while the full
  geometry sweep retains its original randomized PCA.
- The primary source index is `../results/README.md`; the appendix explains the dataset,
  readout, uncertainty, and correction boundaries.

Regenerating these assets does not train a model. The rewrite does not resolve the
inconclusive attention-pooling control, unequal native geometry grids, or missing Kimi K3
parent-output/projector parity checks; it describes those limitations explicitly.

The paper's related-work discussion was shortened to the methodological comparisons it
uses. No claim of priority or universal projector behavior is made. Bibliography entries
are retained only when cited by the main text or appendix.
