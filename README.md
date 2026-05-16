# hamiltonian_guidance

Small package for Hamiltonian guidance and critically-damped Langevin diffusion
experiments.

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate hamiltonian-guidance
```

If the environment already exists, update it after dependency changes:

```bash
conda env update -f environment.yml --prune
```

The environment installs this repo in editable mode, so local changes under
`src/` are picked up without reinstalling.

PyTorch and TorchVision are installed from the CUDA 12.6 PyTorch wheel index:

```text
https://download.pytorch.org/whl/cu126
```

## Project Layout

- `configs/`: experiment and training configuration files.
- `data/`: local datasets or generated data artifacts.
- `notebooks/`: exploratory notebooks.
- `src/hamiltonian_guidance/`: installable Python package code.

## Run

The current toy CLD diffusion script can still be run from the repo root:

```bash
python cld_diffusion.py
```

You can also run it through the installed console command:

```bash
cld-diffusion
```

or as a package module:

```bash
python -m hamiltonian_guidance.cld_diffusion
```

## Future Training Dependencies

PyTorch is installed by default because the current script needs it. Lightning
and Comet ML are declared as optional experiment dependencies for later:

```bash
pip install -e ".[experiments]"
```
