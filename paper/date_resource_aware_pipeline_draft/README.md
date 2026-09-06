# DATE Paper Draft

This directory contains the first paper draft based on the DATE A4 LaTeX
template and the archived ResNet18/YOLOv3-tiny CPU--VTA experiments.

## Files

- `main.tex`: complete first-pass paper draft.
- `references.bib`: preliminary bibliography; verify all metadata before submission.
- `IEEEtran.cls`: copied unchanged from the supplied DATE template.
- `PAPER_NOTES.md`: evidence map, unresolved claims, and required follow-up experiments.

## Build

With a TeX distribution that provides `pdflatex` and `bibtex`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

No LaTeX compiler was available in the current Linux environment when this
draft was created. The source therefore still needs a real compile and page
limit check, either in the Windows TeX installation or after installing a TeX
distribution in WSL.

## Current Positioning

The primary contribution is resource-aware graph-stage selection for an
embedded CPU--FPGA pipeline. The experimental GraphExecutor multi-request path
is supporting runtime infrastructure, not the main novelty. The paper's key
resource rule is:

```text
choose the fewest boundaries that balance the largest CPU stage against
total serialized VTA occupancy
```

## Before Submission

1. Replace anonymous author and affiliation placeholders.
2. Confirm the DATE page limit and compile with the official template.
3. Add exact board clocks, DRAM, software commit, compiler version, and power.
4. Run the executor-matched native all-VTA baseline for both models.
5. Add cost-model ablations and leave-one-model-out Top-K recall.
6. Add a third-model validation, preferably SqueezeNet.
7. Verify every bibliography entry against the publisher record.
