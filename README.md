# OPTIMAL-DESIGN-PMM — Reproducibility Repository

Code, logs and result files for the paper:

> **Optimal Experimental Design for the Polynomial Maximization Method: a Higher-Order
> Reserve Beyond Second-Order Design Theory**
> Serhii Zabolotnii, 2026.

**Repository:** <https://github.com/SZabolotnii/OPTIMAL-DESIGN-PMM-code-supplement>

**Summary.** For regression with independent errors of constant variance whose standardised
cumulants of order three and higher vary with the regressor, the paper derives optimal designs
for the degree-S polynomial maximization method (PMM) through the g_S-weighted information
matrix M_g. This repository recomputes every number, table and figure of the paper: the
calibrated reduction fields, the optimal designs, the Monte Carlo comparisons, the two-stage
study, and the symbolic and numerical checks of the appendices.

## What this repository reproduces

| Script | Paper artifact |
|---|---|
| `experiments/run_all.R` | Tables 1–4 (calibrated KURT field, design separation, A vs D criterion, N-sweep), Figures 1–4; writes `results/summary.json` |
| `experiments/conditional_sls.R` | Table 6 (global conditional SLSE on the SKEW field) and the covariance check of Proposition 2; writes `results/conditional_sls.json` |
| `experiments/two_stage.R` | Table 5 (two-stage design with an estimated field, one-third pilot); writes `results/two_stage_*.csv`, `results/two_stage_meta.json` |
| `experiments/two_stage_limit.R` | Large-N limit of the two-stage design for pilot fractions 1/3, 0.2, 0.1, 0.05 (Section 6) |
| `experiments/g3_derivation.py` | Appendix A (Proposition 1, Monte Carlo check `prop1_mc.csv`) and Appendix B (closed-form g_3, Table B.1, `g3_beta_closed_form.csv`) |
| `experiments/gS_general_check.py` | Appendix C, Lemma 3: degree-S per-stratum reduction (free shape gives g_S = 1; symmetric law with the symmetry imposed gives g_S = 1/(m_2 b'F^{-1}b)) |
| `experiments/gS_laplace_diag.py` | Appendix C: Laplace errors at S = 5, Monte Carlo noise versus the asymptotic value |
| `experiments/thm2b_six_point.py` | Remark 1: on six support points a non-constant field can give M_g = c M (c = 4.78) |

Each script writes a `.log` file next to itself; the logs of the runs reported in the paper are
included. `tables/` and `figures/` hold the files that the scripts write and the manuscript
inputs.

## How to run

From the repository root:

```bash
Rscript experiments/run_all.R            > experiments/run_all.log
Rscript experiments/conditional_sls.R    > experiments/conditional_sls.log
Rscript experiments/two_stage.R          > experiments/two_stage.log   # reads results/summary.json
Rscript experiments/two_stage_limit.R    > experiments/two_stage_limit.log
python3 experiments/g3_derivation.py
python3 experiments/gS_general_check.py
python3 experiments/gS_laplace_diag.py
python3 experiments/thm2b_six_point.py
```

`two_stage.R` and `two_stage_limit.R` read the calibrated field from `results/summary.json`, so
`run_all.R` runs first. Seeds are fixed inside each script.

## Software

- R 4.5.3 with EstemPMM (CRAN, doi:10.32614/CRAN.package.EstemPMM), jsonlite, ggplot2 and
  parallel. The paper's calibration and Monte Carlo comparisons (`run_all.R`) used EstemPMM 0.4.0
  (CRAN archive) and the two-stage study (`two_stage.R`) used EstemPMM 0.5.0; each log prints the
  version it ran with. `conditional_sls.R` and `two_stage_limit.R` do not call EstemPMM.
- Python 3 with the packages in `requirements.txt` (sympy, numpy, scipy).

To install the EstemPMM version of a given run:

```r
install.packages("remotes")
remotes::install_version("EstemPMM", version = "0.4.0")   # or "0.5.0"
```

## License

MIT, see `LICENSE`.
