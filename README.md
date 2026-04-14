# Machine Teaching for Explainable AI in Industry: A Novel Approach for Time Series Classifiers
#### Machine-Teaching-for-XAI--TimeSeries-Models
by Sebastian Einar Salas Røkholt

## Project Context
This repository contains the implementation for the University of Bergen MSc thesis **Machine Teaching for Explainable AI in Industry: A Novel Approach for Time Series Classifiers** (March 2026).

The project builds an end to end MT4XAI pipeline for sequence level time series anomaly detection on EV charging sessions.

The implemented system currently covers:
- forecasting based anomaly detection with LSTM and TCN modelling support
- anomaly scoring with session level metrics and threshold based classification
- classifier preserving charging curve simplification with ORS
- machine teaching set construction with facility location selection and curriculum aware serving
- MLLM based proxy learner experiments across conditions A to F

Thesis PDF:
- `Docs/INF399___MT4XAI_Master_s_Project.pdf`

## Architecture and Pipeline
This section documents the current and target pipeline architecture.

### 1. MT4XAI Experimental System Pipeline Diagram (implemented)
![MT4XAI Experimental System Pipeline Diagram](Docs/Diagrams/MT4XAI%20Experimental%20System%20Pipeline%20Diagram.png)

This diagram summarises the implemented offline experimental pipeline in this repository. Prepared charging session batches are used to train the forecasting model, select anomaly scoring configuration, generate ORS simplifications, construct teaching sets, and run the MLLM trial engine with groups A to F.

### 2. MT4XAI Pipeline with Detailed Teaching Set Construction Diagram (implemented)
![MT4XAI Pipeline with Detailed Teaching Set Construction Diagram](Docs/Diagrams/MT4XAI%20Pipeline%20with%20Detailed%20Teaching%20Set%20Construction%20Diagram.png)

This diagram expands the implemented teaching set flow. Simplification candidates are embedded, stratified by simplicity level and budget, then selected with a facility location objective to produce compact and diverse teaching sets used by conditions A to D, where group E reuses D assets and group F is a no teaching baseline.

### 3. MT4XAI Target Production System Pipeline Diagram (future work)
![MT4XAI Target Production System Pipeline Diagram](Docs/Diagrams/MT4XAI%20Target%20Production%20System%20Pipeline%20Diagram.png)

This diagram shows the target production concept for combined offline and online operation. Offline components periodically train and refresh the model and global teaching set, while online components process a single user charging session, classify it, optionally simplify it, and support user facing local or global explanation workflows.

## Repository Structure
```text
.
├── Docs/
│   ├── INF399___MT4XAI_Master_s_Project.pdf
│   └── Diagrams/
├── Notebooks/
│   ├── 00__Data_Anonymisation.ipynb
│   ├── 01__Data_Wrangling_and_FE.ipynb
│   ├── 02__EDA.ipynb
│   ├── 03__Modelling.ipynb
│   ├── 04__Anomaly_Detection.ipynb
│   ├── 05__Curve_Simplification.ipynb
│   ├── 06__MT4XAI.ipynb
│   └── 07__MLLM_Experiment.ipynb
├── src/
│   ├── mt4xai/                  # Core modelling, inference, ORS and teaching logic
│   └── mllm_experiment/         # MLLM experiment runner, prompts and metadata loaders
├── Data/                        # Datasets and generated metadata or results
├── Figures/                     # Teaching and exam set figures
├── Models/                      # Trained model checkpoints and tuning outputs
├── scripts/                     # Utility scripts such as ORS validation
├── config.yaml
├── project_config.py
├── linux_requirements.txt
└── pyproject.toml
```

## Notebook Overview
Notebook execution is split into a reproducibility only step and a recommended main workflow.

**Important**
- `00__Data_Anonymisation.ipynb` depends on the private raw dataset and external weather APIs
- readers without private raw data should start from `01__Data_Wrangling_and_FE.ipynb`

| Step | Notebook | Role in pipeline | Main outputs |
|---|---|---|---|
| 0 | `00__Data_Anonymisation.ipynb` | Reproducibility only preprocessing on private raw source | `Data/etron55-charging-sessions-public.parquet` |
| 1 | `01__Data_Wrangling_and_FE.ipynb` | Public dataset wrangling and feature engineering for modelling | `Data/etron55-charging-sessions.parquet` |
| 2 | `02__EDA.ipynb` | Exploratory analysis used for design choices | EDA tables and figures |
| 3 | `03__Modelling.ipynb` | Forecasting model training and tuning | `Models/` checkpoints and selected final model |
| 4 | `04__Anomaly_Detection.ipynb` | Session level scoring and threshold calibration | Anomaly metric and threshold configuration |
| 5 | `05__Curve_Simplification.ipynb` | ORS diagnostics and parameter selection | Simplification diagnostics and examples |
| 6 | `06__MT4XAI.ipynb` | Teaching pool construction and teaching or exam set export | `Figures/teaching_sets/`, `Figures/exam_sets/`, metadata build inputs |
| 7 | `07__MLLM_Experiment.ipynb` | Trial execution wrapper and effect analysis for groups A to F | `Data/mllm_experiment_results/` analysis outputs |

Recommended order for most users is `01` to `07`.

## Source Code Overview
### `src/mt4xai`
Reusable pipeline package used by notebooks.
- `data.py` data handling, splits, scaling, session containers
- `model.py` LSTM and TCN forecasting models and loaders
- `train.py` Ray based tuning and training loops
- `inference.py` residual prediction reconstruction, anomaly metrics and classification helpers
- `ors.py` merged ORS implementation including DP prefix v3 mode
- `ors_v3.py` compatibility wrapper that maps legacy `ors_v3` imports to `mt4xai.ors`
- `teach.py` teaching pool and teaching set construction logic
- `plot.py` visualisation helpers

### `src/mllm_experiment`
Scriptable MLLM evaluation package.
- `run_trial.py` trial CLI entry point and run metadata handling
- `trial.py` participant lifecycle across pre exam, teaching and post exam
- `data_loading.py` metadata and image loading with group aware resolution
- `prompts.py` multimodal prompt builders per phase and condition
- `openai_client.py` OpenAI Responses API wrapper with retries and shared rate limiting
- `build_teaching_sets.py` anonymised teaching set image export and `teaching_items.csv`
- `build_exam_sets.py` anonymised exam set image export and `exam_items.csv`
- `utils.py` structured logging and result writers

### Experiment Conditions
Current trial support includes conditions `A` to `F`.
- `A` overlay plus simplified with curriculum
- `B` overlay plus simplified without curriculum
- `C` raw modality only
- `D` simplified modality with curriculum
- `E` simplified modality with enforced rule update protocol and locked post exam rule use
- `F` baseline with no teaching phase

## Secrets and Environment Variables
Create `.env` from the committed template:

```bash
cp example.env .env
```

Required secrets by workflow:
- `OPENAI_API_KEY` for real API calls in `07__MLLM_Experiment.ipynb` and `mllm_experiment.run_trial`
- `FROST_API_CLIENT_ID` and `DMI_MET_OBS_API_KEY` for `00__Data_Anonymisation.ipynb`

The template also includes legacy optional placeholders currently present in local environment files.

## Setup
Python 3.12 is the target runtime.

### Option A: CUDA setup on Linux or WSL2 with NVIDIA GPU
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r linux_requirements.txt
pip install -e .
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
PY
```

### Option B: CPU only setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
grep -vE '^(torch==|nvidia-|triton==)' linux_requirements.txt > /tmp/requirements.cpu.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.0
pip install -r /tmp/requirements.cpu.txt
pip install -e .
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
PY
```

## Running the Pipeline
### 1. Run notebooks
```bash
cd Notebooks
../.venv/bin/jupyter lab
```

Run in order:
1. `01__Data_Wrangling_and_FE.ipynb`
2. `02__EDA.ipynb`
3. `03__Modelling.ipynb`
4. `04__Anomaly_Detection.ipynb`
5. `05__Curve_Simplification.ipynb`
6. `06__MT4XAI.ipynb`
7. `07__MLLM_Experiment.ipynb`

Run notebook `00__Data_Anonymisation.ipynb` only when reproducing private data anonymisation.

### 2. Build anonymised teaching and exam metadata with package entry points
```bash
source .venv/bin/activate
exp-build-teaching-sets
exp-build-exam-sets
```

Equivalent module form:
```bash
PYTHONPATH=src python -m mllm_experiment.build_teaching_sets
PYTHONPATH=src python -m mllm_experiment.build_exam_sets
```

### 3. Validate trial runner with dry run
```bash
PYTHONPATH=src python -m mllm_experiment.run_trial \
  --participants 1 \
  --teaching_set_dir Figures/teaching_sets/mllm_experiment_sets \
  --exam_sets_dir Figures/exam_sets/mllm_experiment_sets \
  --metadata_dir Data/mllm_experiment_metadata \
  --conditions all \
  --output_dir Data/mllm_experiment_results/dry_run \
  --random_seed 42 \
  --dry_run
```

### 4. Run real MLLM experiment
```bash
PYTHONPATH=src python -m mllm_experiment.run_trial \
  --participants 30 \
  --teaching_set_dir Figures/teaching_sets/mllm_experiment_sets \
  --exam_sets_dir Figures/exam_sets/mllm_experiment_sets \
  --metadata_dir Data/mllm_experiment_metadata \
  --conditions all \
  --output_dir Data/mllm_experiment_results/gpt-5-nano_experiment_1 \
  --model_name gpt-5-nano \
  --parallel_participants 2 \
  --random_seed 42
```

## Reproducibility Notes
- Keep `config.yaml`, notebook parameters, and CLI flags versioned together for each run
- Use unique output directories under `Data/mllm_experiment_results/`
- Keep immutable snapshots of `Data/mllm_experiment_metadata/*.csv`, run metadata JSON, and selected model checkpoints
- Run dry runs before real API runs to validate metadata and prompt protocol formatting
