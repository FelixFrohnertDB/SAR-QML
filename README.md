<div align="center">

# SAR-QML

### Task-aligned quantum models for spectral classification

[![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-quantum%20machine%20learning-00B4D8)](https://pennylane.ai/)
[![EQUALITY](https://img.shields.io/badge/EU%20Horizon-EQUALITY-003399)](https://equality-quantum.eu/)

Code and figure-generation notebooks accompanying our work on inductive bias in data-reuploading quantum circuits and its application to classification from synthetic-aperture-radar data.

</div>

---

## Overview

This repository studies data-reuploading quantum circuits that alternately encode a signal input $\boldsymbol{x}$ and a fixed query input $\boldsymbol{f}$. Their expectation values exhibit a nested Fourier structure: a Fourier expansion over $\boldsymbol{x}$ whose coefficients are modulated by $\boldsymbol{f}$.

We use this structure to investigate how circuit architecture can impose a task-aligned inductive bias for spectral classification. The experiments progress from controlled two-qubit settings to classification from raw synthetic-aperture-radar-like measurements, where the relevant Fourier queries are determined by the acquisition geometry.

The repository contains the six notebooks used to generate the figures in the paper.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/felixfrohnertdb/sar-qml.git
cd sar-qml
python -m pip install -e .
```

Then start Jupyter:

```bash
jupyter lab
```

## Reproducing the figures

The notebooks are located in [`notebooks/`](notebooks/) and are numbered in the order in which the corresponding experiments appear in the paper. Together, they cover:

1. query-induced Fourier-coefficient modulation and its rank criterion;
2. interpolation between query encodings;
3. changes in the represented function class and induced function prior;
4. controlled teacher--student classification and data efficiency;
5. sensitivity to mismatch between the task and model query; and
6. classification directly from raw SAR-like measurements.

Each notebook can be executed from top to bottom after installation and contains the configuration needed to reproduce its corresponding publication figure.

## Citation

If you use this code, please cite the accompanying paper. Citation details will be added upon publication.

## Funding

This work was funded by the European Union under Grant Agreement No. 101080142 ([EQUALITY — QUAntum ALgorithms for IndusTrY](https://equality-quantum.eu/)). Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the granting authority can be held responsible for them.

## Contact

For questions about the code or experiments, please [open an issue](https://github.com/felixfrohnertdb/sar-qml/issues).
