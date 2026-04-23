# Supplementary material

Standalone PDFs containing the appendix content of the MONET paper
(`hatz26monet`), extracted from the paper source so that the code
repository is self-contained.

| File | Contents |
|---|---|
| `tasks_and_environments.pdf` | Per-environment task / solution parameterisations for the four benchmarks (10-DoF Arm, Archery, Cartpole, Hexapod) and their closed-form fitness expressions. |
| `algorithm_and_hyperparameters.pdf` | Full MONET pseudocode (main loop + individual / social learning subroutines), hyperparameter reference table, per-domain SHAP sensitivity analysis, SPOT-tuned configurations, and coupon-collector node-coverage bounds that justify the $10^{6}$-evaluation budget. |

Figures referenced by the PDFs live under `figs/`.

## Rebuilding

Both documents are standalone `article`-class LaTeX sources — no external
class files or bibliography database required. A full TeX Live install is
sufficient:

```bash
cd supplementary
pdflatex tasks_and_environments.tex && pdflatex tasks_and_environments.tex
pdflatex algorithm_and_hyperparameters.tex && pdflatex algorithm_and_hyperparameters.tex
```

(Two passes are needed for the table of contents to resolve.) Or with
`latexmk`:

```bash
latexmk -pdf tasks_and_environments.tex algorithm_and_hyperparameters.tex
latexmk -c   # clean aux files
```
