# Machine Teaching for Explainable AI in Industry: A Novel Approach for Time Series Classifiers
#### Machine-Teaching-for-XAI--TimeSeries-Models
by Sebastian Einar Salas Røkholt

## Introduction
This repository contains the implementation for the master's thesis **"Machine Teaching for Explainable AI in Industry: A Novel Approach for Time Series Classifiers"** (University of Bergen, March 2026).

The project involves applying a novel XAI technique for a real-world sequence-level time series anomaly detection (SL-TSAD) problem. The core idea is to use **Machine Teaching** to generate compact, example-based teaching sessions so a learner can better simulate a black-box anomaly detection system's `normal`/`abnormal` decisions.

The repository covers the full pipeline:
- Data wrangling, preprocessing and feature engineering for constructing an EV charging session dataset suitable for model training
- Exploratory Data Analysis
- Sequence-level time series anomaly detection (SL-TSAD) model development with LSTM- and TCN-based forecasting
- Charging curve simplification and curriculum-oriented teaching set construction.
- Automated machine teaching experiment on multimodal LLMs

Primary thesis draft:
- `Docs/INF399___MT4XAI_Master_s_Project-12-03-2026.pdf`

## Repository Structure
```text
.
├── Docs/                         # Thesis drafts and project documents
├── Notebooks/                    # Methodology notebooks (01-06)
│   ├── 01__Data_Wrangling_and_FE.ipynb
│   ├── 02__EDA.ipynb
│   ├── 03__Modelling.ipynb
│   ├── 04__Anomaly_Detection.ipynb
│   ├── 05__Curve_Simplification.ipynb
│   └── 06__MT4XAI.ipynb
├── src/
│   ├── mt4xai/                   # Reusable core package for pipeline logic
│   └── mllm_experiment/          # Script-based MLLM experiment runner
├── Data/
│   ├── etron55-charging-sessions.parquet      # Main cleaned dataset
│   ├── teaching_pool/                         # Teaching pool artifacts
│   ├── exam_teaching_pool/                    # Exam pool artifacts
│   ├── mllm_experiment_metadata/              # teaching_items.csv, exam_items.csv
│   └── mllm_experiment_results/               # Trial outputs/logs
├── Figures/
│   ├── teaching_sets/                          # Teaching set figures
│   └── exam_sets/                              # Exam set figures
├── Models/                      # Ray Tune runs and final model checkpoints
├── config.yaml                  # Project configuration
├── project_config.py            # Config loader
├── linux_requirements.txt       # Pinned Python dependencies
└── pyproject.toml               # Package metadata
```

## Notebook Overview
The notebooks are the step-by-step methodology implementation and documentation. They should be run in order.

| Step | Notebook | Purpose | Main outputs |
|---|---|---|---|
| 1 | `01__Data_Wrangling_and_FE.ipynb` | Clean raw session data and engineer features | `Data/etron55-charging-sessions.parquet` |
| 2 | `02__EDA.ipynb` | Explore distributions, structure, and data quality | EDA figures/tables |
| 3 | `03__Modelling.ipynb` | Train/tune LSTM and TCN forecasting models | `Models/bohb_*`, selected final model |
| 4 | `04__Anomaly_Detection.ipynb` | Build sequence-level anomaly detection setup and thresholding | AD diagnostics and threshold choices |
| 5 | `05__Curve_Simplification.ipynb` | Explore/tune ORS simplifications | Simplification diagnostics and examples |
| 6 | `06__MT4XAI.ipynb` | Build teaching/exam pools and MT4XAI sets | Teaching/exam pool artifacts and figures |

Note: the **MLLM experiment is script-based**, not notebook-based. It should be run last. The script for running experiments lives in `src/mllm_experiment/`.

## Source Code Overview

### `src/mt4xai`
Reusable core package used by notebooks:
- `data.py`: session grouping, splitting, scaling, loaders.
- `model.py`: forecasting model definitions and loading utilities.
- `train.py`: training/tuning helpers.
- `inference.py`: prediction reconstruction and anomaly metrics.
- `ors.py`: ORS simplification logic.
- `teach.py`: teaching pool and teaching set construction/serving.
- `plot.py`: plotting helpers for model outputs and simplifications.

### `src/mllm_experiment`
Script-based MLLM trial pipeline:
- `run_trial.py`: CLI entrypoint for multi-participant experiment runs.
- `trial.py`: participant lifecycle (pre exam → teaching → post exam).
- `data_loading.py`: metadata/image loading and group/phase resolution.
- `prompts.py`: multimodal prompt construction.
- `openai_client.py`: OpenAI API wrapper with retry/rate-limiting.
- `build_teaching_sets.py`: anonymise teaching images + build teaching metadata CSV.
- `build_exam_sets.py`: anonymise exam images + build exam metadata CSV.
- `config.py`, `utils.py`: experiment settings and structured logging helpers.

Current code supports experiments on different MLLM participant groups with one of conditions `A`, `B`, `C`, `D`, `E`, `F`:
 - A: Participant sees both original (raw) and simplified examples with an easy-to-hard curriculum.
 - B: Participant sees the same modality as A but with no curriculum (random order).
 - C: Participant sees only original examples with no curriculum.
 - D: Participant sees simplification-only teaching and post-exam examples, with curriculum.
 - E: Participant sees simplification-only teaching and post-exam examples with sequential rule-of-thumb updates during teaching, then fixed-rule application in post-exam.
 - F: Baseline condition with no teaching phase, only pre and post exam in raw modality.


## Setup

### Prerequisites
- Linux or macOS shell environment.
- Python >=**3.12** recommended (tested with `3.12.10`).
- Optional CUDA GPU for model training/tuning notebooks.
- OpenAI API key for non-dry-run MLLM experiments.
- API keys for Scandinavian metrological services (Frost, DMI) for feature engineering related to temperature readings. 

### Environment setup
```bash
# from repository root
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r linux_requirements.txt
pip install -e .
```

### Environment variables
Create `.env` in repository root:

```bash
OPENAI_API_KEY=your_openai_key
FROST_API_CLIENT_ID=your_frost_client_id
DMI_MET_OBS_API_KEY=your_dmi_api_key
```

Notes:
- `OPENAI_API_KEY` is required for real MLLM runs.
- `FROST_API_CLIENT_ID` and `DMI_MET_OBS_API_KEY` are used in Notebook 01 if re-running external weather enrichment.

### Data access and sharing
- This repository includes data with sharing constraints. See `Data/README.md`.
- If reproducing from raw sources, keep the cleaned dataset path consistent with `config.yaml`:
  - `Data/etron55-charging-sessions.parquet`

## Reproducible Execution (From Scratch)

### 1) Run notebooks in methodology order
```bash
cd Notebooks
../.venv/bin/jupyter lab
```
Then run:
1. `01__Data_Wrangling_and_FE.ipynb`
2. `02__EDA.ipynb`
3. `03__Modelling.ipynb`
4. `04__Anomaly_Detection.ipynb`
5. `05__Curve_Simplification.ipynb`
6. `06__MT4XAI.ipynb`

This pipeline produces the model and teaching/exam artefacts consumed by the MLLM scripts.
At the end of `06__MT4XAI.ipynb`, run the final build cell to generate anonymised
MLLM experiment assets and metadata.

### 2) Build anonymised MLLM teaching/exam metadata
Preferred path: run the final build cell in `06__MT4XAI.ipynb`.

Standalone fallback: run from repository root:

```bash
./.venv/bin/python src/mllm_experiment/build_teaching_sets.py
./.venv/bin/python src/mllm_experiment/build_exam_sets.py
```

Outputs are written to:
- `Figures/teaching_sets/mllm_experiment_sets/`
- `Figures/exam_sets/mllm_experiment_sets/`
- `Data/mllm_experiment_metadata/`

### 3) Validate experiment pipeline in dry-run mode
```bash
PYTHONPATH=src ./.venv/bin/python -m mllm_experiment.run_trial \
  --participants 1 \
  --teaching_set_dir Figures/teaching_sets/mllm_experiment_sets \
  --exam_sets_dir Figures/exam_sets/mllm_experiment_sets \
  --metadata_dir Data/mllm_experiment_metadata \
  --conditions abc \
  --output_dir Data/mllm_experiment_results/dry_run \
  --random_seed 42 \
  --dry_run
```
For reproducible baseline comparisons, start with `--conditions abc`. Include `d` or `e` for simplified-only conditions, and include `f` for the no-teaching baseline.

### 4) Run MLLM experiment (real API)
```bash
PYTHONPATH=src ./.venv/bin/python -m mllm_experiment.run_trial \
  --participants 20 \
  --teaching_set_dir Figures/teaching_sets/mllm_experiment_sets \
  --exam_sets_dir Figures/exam_sets/mllm_experiment_sets \
  --metadata_dir Data/mllm_experiment_metadata \
  --conditions abc \
  --output_dir Data/mllm_experiment_results/pilot_02 \
  --model_name gpt-5-mini \
  --parallel_participants 2 \
  --random_seed 42
```

## Reproducibility Notes
- Keep `config.yaml`, notebook parameters, and CLI flags versioned together with each run.
- Always log and report `--random_seed` for MLLM runs.
- Run dry-run before real API calls to validate metadata/prompt formatting.
- Store each experiment in a unique `--output_dir` to avoid mixed logs.
- Keep immutable snapshots of:
  - `Data/mllm_experiment_metadata/*.csv`
  - `Data/mllm_experiment_results/<run_id>/`
  - `Models/final/final_model.pth`

## Results
