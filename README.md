<div align="center">

# SAR-QML

### Task-aligned quantum models for spectral classification

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-numerics-A259FF)](https://github.com/jax-ml/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![EQUALITY](https://img.shields.io/badge/EU%20Horizon-EQUALITY-003399)](https://equality-quantum.eu/)

Code and figure-generation notebooks accompanying our work on inductive bias in data-reuploading quantum circuits and its application to classification from synthetic-aperture-radar data.

</div>

---

## Overview

This repository studies data-reuploading quantum circuits that alternately encode a signal input $\boldsymbol{x}$ and a fixed query input $\boldsymbol{f}$. Their expectation values exhibit a nested Fourier structure: a Fourier expansion over $\boldsymbol{x}$ whose coefficients are modulated by $\boldsymbol{f}$.

We use this structure to investigate how circuit architecture can impose a task-aligned inductive bias for spectral classification. The experiments progress from controlled two-qubit settings to classification from raw synthetic-aperture-radar-like measurements, where the relevant Fourier queries are determined by the acquisition geometry.

The repository contains the Jupyter notebooks that generate every numerical figure in the paper, together with the cached numerical results behind them.

## Repository layout

```
notebooks/   one notebook per figure: configuration, model, experiment flow, figure
src/sar_qml/ supporting code imported by the notebooks
results/     cached paper results (CSV / JSON / NPZ) read by the notebooks
figures/     figures written by the notebooks (PDF and PNG)
```

The notebooks show the configuration, the circuit definitions, the top-level experiment flow and the figure code. Supporting code (linear algebra, samplers, scene generators, training loops, bootstrap statistics, checkpointing and plotting helpers) lives in the `sar_qml` package:

| Module | Used by |
|---|---|
| `sar_qml.reuploading`, `reuploading_encoding`, `reuploading_query`, `reuploading_plotting` | `fig_01`, `fig_02`, `fig_01_02_combined` |
| `sar_qml.alignment` | `fig_03` |
| `sar_qml.teacher_student`, `teacher_student_training`, `teacher_student_curves`, `teacher_student_query_scan` | `fig_05`, `fig_06` |
| `sar_qml.sar` (`geometry`, `gates`, `common`, `checkpoint`, `capacity_matched`, `double_reuploading`) | `fig_07`, `fig_08` |
| `sar_qml.plotting` | shared plotting helpers |

## Installation

The code was tested with Python 3.12 on CPU.

```bash
git clone https://github.com/FelixFrohnertDB/SAR-QML.git
cd SAR-QML
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]"
jupyter lab
```

Figure text is typeset with LaTeX (`text.usetex`). Most notebooks fall back to matplotlib mathtext when no working LaTeX installation is found, or when the environment variable `USE_TEX=0` is set; `fig_03` and `fig_08` require LaTeX.

## Reproducing the figures

Each notebook runs top to bottom and writes its figure to `figures/`. When the cached results in `results/` are present, the notebooks only reload, re-aggregate and re-plot, which takes seconds. Each notebook's first cell explains how to recompute its results from scratch (delete the corresponding cached file); most also offer a quick `PROFILE = "smoke"` check.

| Notebook | Content | Cached results | From scratch (approx.) |
|---|---|---|---|
| [`fig_01_vary_encoding`](notebooks/fig_01_vary_encoding.ipynb) | encoding interpolation at a fixed query (data for the top row of the combined figure) | `results/fig_01_02/` | 2 min |
| [`fig_02_vary_query`](notebooks/fig_02_vary_query.ipynb) | query value along a ray (data for the bottom row of the combined figure) | `results/fig_01_02/` | 3 min |
| [`fig_01_02_combined`](notebooks/fig_01_02_combined.ipynb) | combined figure: query encoding versus query value | `results/fig_01_02/` | — (plots only) |
| [`fig_03_approx_spectrum`](notebooks/fig_03_approx_spectrum.ipynb) | task–spectrum alignment with an MNUDFT task: reachability of the task modes and the spectral-prior distance | `results/task_alignment/` | 3 min |
| [`fig_05_teacher_student_classification`](notebooks/fig_05_teacher_student_classification.ipynb) | query-aligned teacher–student classification and data efficiency | `results/teacher_student/` | 35 min |
| [`fig_06_query_value_sensitivity`](notebooks/fig_06_query_value_sensitivity.ipynb) | classification sensitivity to mismatch between task and model query | `results/query_sensitivity/` | several hours |
| [`fig_07_sar_geometry_classification`](notebooks/fig_07_sar_geometry_classification.ipynb) | geometry-aligned classification of unfocused SAR-like phase histories | `results/sar_geometry/` | 35 min |
| [`fig_08_double_reuploading_geometry`](notebooks/fig_08_double_reuploading_geometry.ipynb) | geometry-selected hypothesis classes in a double data re-uploading classifier | `results/sar_fixed_query/` | several hours |

Run `fig_01_vary_encoding` and `fig_02_vary_query` before `fig_01_02_combined` if their cached results have been removed. To execute all notebooks non-interactively:

```bash
cd notebooks
for nb in fig_01_vary_encoding fig_02_vary_query fig_01_02_combined fig_03_approx_spectrum \
          fig_05_teacher_student_classification fig_06_query_value_sensitivity \
          fig_07_sar_geometry_classification fig_08_double_reuploading_geometry; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
```

Result files of the classification experiments carry a hash of the full experiment configuration (`RUN_ID`) in their name; changing any setting therefore starts a new, separately cached run instead of overwriting the paper results.

## Citation

If you use this code, please cite the accompanying paper. Citation details will be added upon publication.

## License

This project is licensed under the Apache License 2.0; see [LICENSE](LICENSE).

## Funding

This work was funded by the European Union under Grant Agreement No. 101080142 ([EQUALITY — QUAntum ALgorithms for IndusTrY](https://equality-quantum.eu/)). Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the granting authority can be held responsible for them.

## Contact

For questions about the code or experiments, please [open an issue](https://github.com/felixfrohnertdb/sar-qml/issues).
