# ICCBR 2026 Reproducibility Package

This folder contains the minimal materials to reproduce the paper experiments and rebuild the LNCS manuscript.

## Folder Structure

- `code/`  
  Core framework and runnable scripts:
  - `cbr_mas/`
  - `run_experiments.py`
  - `run_full_phased.sh`
  - `train_selector.py`
  - `fuse_guarded.py`
  - `analyze_results.py`
  - `requirements.txt`
  - `env.example`

- `results/`  
  Saved JSON outputs used in the paper.

- `paper_assets/`  
  Plot scripts and exported assets used to generate figures.

- `paper/lncs_paper/`  
  LNCS source files (`main.tex`, section files, bibliography, figures).

## Quick Start

1. Create environment and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r code/requirements.txt`

2. Configure environment variables:
   - Copy `code/env.example` to `.env` and fill API/model settings.

3. Run experiments:
   - `python code/run_experiments.py --help`
   - or staged run: `bash code/run_full_phased.sh`

4. Train/evaluate selector:
   - `python code/train_selector.py --help`

5. Build paper:
   - `cd paper/lncs_paper`
   - `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

