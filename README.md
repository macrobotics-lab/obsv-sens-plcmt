# Observability-Informed Optimal Sensor Placement for Soft Robots

This repository contains the companion code for our IEEE RoboSoft 2026 Conference paper titled ["Observability-Informed Optimal Sensor Placement for Soft Robots"](https://ieeexplore.ieee.org/abstract/document/11522828). 

## Getting Started

The code was developed with Python 3.13.5 and MATLAB 2024b on Windows 11.

Miniconda was used to manage the virtual environment. To create a new environment with the required packages, run the following command from the ```src``` directory:

```conda env create -n new-env-name -f environment.yml```

The virtual environment can then be activated with

```conda activate new-env-name ```

## Reproducing Results From Paper
To reproduce the results from the paper, run the file ```run.bat``` from the ```src``` directory. Running ```run.bat``` will overwrite the current results in the ```src\results``` directory.
