# Notebook #3: EV Charging Curve Modelling </br>with LSTM and TCN Architectures
#### by Sebastian Einar Salas Røkholt
----

**Index**  
- [**1 - Introduction and Notebook Setup**](#1---introduction)  
  - [*1.1 Setup*](#11-setup)  
  - [*1.2 Feature selection*](#12-feature-selection) 

- [**2 - Data preparation**](#2---data-preparation)  
  - [*2.1 Splitting the data*](#21---splitting-the-data)  
  - [*2.2 Data normalisation*](#22---data-normalisation)  
  - [*2.3 Building the datasets and data loaders*](#23---building-the-datasets-and-data-loaders)  

- [**3 - Model Architectures**](#3---model-architectures)  
  - [*3.1 Defining the LSTM Architecture*](#31-defining-the-lstm-architecture)  
  - [*3.2 Defining the TCN Architecture*](#32-defining-the-tcn-architecture)  
  - [*3.3 Additional Learnable Weights*](#33-additional-learnable-weights)  
  - [*3.4 Model Builders*](#34-model-builders)  

- [**4 - Model Training**](#4---model-training)  
  - [*4.1 Training Utilities*](#41-training-utilities)  
  - [*4.2 Defining the Ray Trainables*](#42-defining-the-ray-trainables)  
  - [*4.3 Hyperparameter Search Spaces and Result Plotting*](#43-hyperparameter-search-spaces-and-result-plotting)  
  - [*4.4 Training and Tuning an LSTM Model*](#44-training-and-tuning-an-lstm-model)  
  - [*4.5 Training and Tuning the TCN Model*](#45-training-and-tuning-the-tcn-model)  
  - [*4.6 Plot Training Process from Ray Tune Logs*](#46-plot-training-process-from-ray-tune-logs)  

- [**5 - Model Evaluation and Selection**](#5---model-evaluation-and-selection)  
  - [*5.1 Evaluating the models with macro averaging*](#51-evaluating-the-models-with-macro-averaging)  
  - [*5.2 Plotting Predictions*](#52-plotting-predictions)  
    - [*5.2.1 Plotting Setup and Utility Functions*](#521-plotting-setup-and-utility-functions)  
    - [*5.2.2 Plotting input-output pairs at time t for a single sample*](#522-plotting-input-output-pairs-at-time-t-for-a-single-sample)  
    - [*5.2.3 Plotting Multi-Horizon Predictions for a Full Session*](#523-plotting-multi-horizon-predictions-for-a-full-session)  
    - [*5.2.4 Plotting Complete Power Predictions for Multiple Sessions*](#524-plotting-complete-power-predictions-for-multiple-sessions)  
  - [*5.3 Model Selection*](#53-model-selection)  
  - [*5.4 Summary and Next Steps*](#54-summary-and-next-steps)

---

## 1 - Introduction and Notebook Setup
This notebook trains, tunes and evaluates multi-horizon sequence models that forecast residuals of future power (kW) transfer in electric vehicle (EV) charging sessions. Absolute values for predictions are reconstructed by adding the current input power. Models are trained in scaled space and evaluated in original units with macro-averaged mean squared error (Macro-MSE) and robust weigthed squared error (RWSE). The best performin forecasting model is saved/exported for a downstream anomaly detection task.

We compare two sequence modelling approaches: </br>
**Long Short-Term Memory (LSTM) networks**, which learn temporal dependencies through gated recurrent units.</br>
**Temporal Convolutional Networks (TCN)**, which exploit causal convolutions and dilations to capture long-range dependencies efficiently.</br>

We train both models with PyTorch, and tune them with Ray Tune by leveraging **Bayesian optimization with HyperBand (BOHB)**.</br>
The code supports automatic resume, exporting a clean .pth with the best state dict and config.

**Software/API notes**: To keep the notebook focused and organised, the notebook frequently uses imported assets from the project's custom `mt4xai` package, some of which are also reused across the other notebooks in this project. Please refer to the `data.py`, `model.py`, `train.py` and `plot.py` modules for a more in-depth look at each component in the modelling pipeline. 

**Runtime notes**: Prefer a CUDA-enabled GPU with automatic mixed precision (AMP). Set NUM_WORKERS based on the number of available CPU cores. Keep Ray concurrency within GPU memory limits to avoid out-of-memory (OOM) errors on large batches.

 ### 1.1 Setup

This section initializes the computational environment by:
- Importing the required Python libraries, including module from the project's custom `mt4xai` package.
- Defining global constants and configuring runtime settings and device configuration. This includes setting a fixed random seed for reproducability, and initialising `ray` with reduced logging.
- Selecting the features/variables in the dataset we are are going to be using for modelling


```python
import os
import sys 
import random 
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import ray
from ray import tune
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.tuner import Tuner, TuneConfig
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.preprocessing import MinMaxScaler

# project config
sys.path.append(str(Path.cwd().parent))  # Adds additional scripts (e.g. project_config.py) in parent dir to path
from project_config import load_config
cfg = load_config()

# Notebook global constants
RANDOM_SEED = cfg.project.random_seed
TRAIN_LSTM = True
TRAIN_TCN = True
RESUME_TRAIN_IF_CKPT = True
BATCH_SIZE = 256  # default batch size used for test and evaluation
NUM_WORKERS = 4  # used in data loaders for training, validation and test
HORIZON = 5  # model predicts for all horizons h in {1, HORIZON}

# Global paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "etron55-charging-sessions.parquet")
MODEL_FOLDER_PATH = os.path.join(PROJECT_ROOT, "Models")
MODEL_NAME_LSTM = "LSTM_multihorizon_raytuned_model_9.pth"
MODEL_NAME_TCN = "TCN_multihorizon_raytuned_model_4.pth"
MODEL_PATH_LSTM = os.path.join(MODEL_FOLDER_PATH, MODEL_NAME_LSTM)
MODEL_PATH_TCN = os.path.join(MODEL_FOLDER_PATH, MODEL_NAME_TCN)
RAY_TUNE_FOLDER_NAME_LSTM = "bohb_lstm_tuning_run_9"
RAY_TUNE_FOLDER_NAME_TCN = "bohb_tcn_tuning_run_4"
RAY_TUNE_RUN_FOLDER_PATH_LSTM = os.path.join(MODEL_FOLDER_PATH, RAY_TUNE_FOLDER_NAME_LSTM)
RAY_TUNE_RUN_FOLDER_PATH_TCN = os.path.join(MODEL_FOLDER_PATH, RAY_TUNE_FOLDER_NAME_TCN)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Pandas and Seaborn config
pd.options.mode.copy_on_write = True
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:.2f}".format
sns.set_theme(style="whitegrid")

# PyTorch config, Ray config and initialisation
print("[env] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[env] Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
ray.shutdown()  # restart Ray Tune env if running
ray.init(  
    logging_level=logging.ERROR,  # suppresses most Ray logs
    log_to_driver=False,  # disables worker logs streaming into notebook
    ignore_reinit_error=True, 
    _memory=6 * 1024**3,  # limit Ray’s worker memory to 6 GB
    object_store_memory=1 * 1024**3,  # 1 GB for the object store
    # system_reserved_memory=3 * 1024**3,  # for all Ray processes
    _temp_dir="/tmp/ray",   # optional: faster NVMe scratch
    include_dashboard=False,
)

# Jupyter Notebook settings
%load_ext autoreload
%autoreload 2
```

    CONFIG FILE LOADED: 
    {'project': {'random_seed': 42, 'root_dir': None}, 'paths': {'dataset': 'Data/etron55-charging-sessions.parquet', 'teaching_pool': 'Data/teaching_pool', 'models': 'Models', 'final_model': 'Models/final/final_model.pth', 'figures': 'Figures', 'logs': 'Logs'}, 'inference': {'horizon': 5, 'final_model_name': 'final_model.pth', 'horizon_decay_lambda': 0.4, 'power_weight': 0.6522982410461, 't_min_eval': 1, 'ad_rmse_threshold': 8.5962, 'ad_pct_threshold': 0.95, 'metric': 'macro_rmse'}, 'ors': {'soc_stage1_mode': 'rdp', 'soc_rdp_epsilon': 0.75, 'soc_rdp_candidates': 5, 'soc_rdp_eps_min': 1e-06, 'soc_rdp_eps_max': 100.0, 'stage2_err_metric': 'l2', 'epsilon_mode': 'fraction'}, 'teaching': {'teaching_pool_dir': '../Data/teaching_pool', 'teaching_set_size': 60}}
    [env] CUDA available: True
    [env] Device: NVIDIA GeForce RTX 4070 Laptop GPU



```python
# Load the cleaned dataframe
df = pd.read_parquet(DATA_PATH)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>charging_id</th>
      <th>minutes_elapsed</th>
      <th>progress</th>
      <th>timestamp</th>
      <th>power</th>
      <th>rel_power</th>
      <th>d_power</th>
      <th>d_power_ema3</th>
      <th>soc</th>
      <th>d_soc</th>
      <th>d_soc_ema3</th>
      <th>energy</th>
      <th>nominal_power</th>
      <th>charger_cat_low</th>
      <th>charger_cat_mid</th>
      <th>charger_cat_high</th>
      <th>temp</th>
      <th>lat</th>
      <th>lon</th>
      <th>in_taper</th>
      <th>dist_to_taper</th>
      <th>charger_category</th>
      <th>timestamp_H</th>
      <th>timestamp_d</th>
      <th>nearest_weather_station</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>2020-01-11 12:37:00</td>
      <td>89.44</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>40.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.32</td>
      <td>150.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>4</td>
      <td>59.67</td>
      <td>9.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Ultra</td>
      <td>2020-01-11T12</td>
      <td>2020-01-11</td>
      <td>SN28380</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0.14</td>
      <td>2020-01-11 12:38:00</td>
      <td>92.75</td>
      <td>0.52</td>
      <td>3.31</td>
      <td>1.66</td>
      <td>41.00</td>
      <td>1.00</td>
      <td>0.50</td>
      <td>1.84</td>
      <td>150.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>4</td>
      <td>59.67</td>
      <td>9.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Ultra</td>
      <td>2020-01-11T12</td>
      <td>2020-01-11</td>
      <td>SN28380</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.23</td>
      <td>2020-01-11 12:39:00</td>
      <td>94.81</td>
      <td>0.53</td>
      <td>2.06</td>
      <td>1.86</td>
      <td>43.00</td>
      <td>2.00</td>
      <td>1.25</td>
      <td>3.41</td>
      <td>150.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>4</td>
      <td>59.67</td>
      <td>9.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Ultra</td>
      <td>2020-01-11T12</td>
      <td>2020-01-11</td>
      <td>SN28380</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.29</td>
      <td>2020-01-11 12:40:00</td>
      <td>95.68</td>
      <td>0.53</td>
      <td>0.87</td>
      <td>1.36</td>
      <td>45.00</td>
      <td>2.00</td>
      <td>1.62</td>
      <td>5.00</td>
      <td>150.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>4</td>
      <td>59.67</td>
      <td>9.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Ultra</td>
      <td>2020-01-11T12</td>
      <td>2020-01-11</td>
      <td>SN28380</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>0.34</td>
      <td>2020-01-11 12:41:00</td>
      <td>96.88</td>
      <td>0.54</td>
      <td>1.20</td>
      <td>1.28</td>
      <td>47.00</td>
      <td>2.00</td>
      <td>1.81</td>
      <td>6.60</td>
      <td>150.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>4</td>
      <td>59.67</td>
      <td>9.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Ultra</td>
      <td>2020-01-11T12</td>
      <td>2020-01-11</td>
      <td>SN28380</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 Feature selection 
In order to predict power for the next `horizon`=5 timesteps, the model uses previous values of `power` and `soc` together with static context (temperature, nominal_power) and engineered dynamics. We will plot SOC alongside predicted power during qualitative evaluation in order to see if the model is able to capture the taper dynamics when SOC approaches the 70-90% range.

Inputs include static metadata and dynamic time-varying signals engineered in the data wrangling notebook and inspected/analyzed in EDA. The modelling API reads the selected feature lists from the config. The residual formulation of the target variable (`power`) simplifies learning and stabilises multi-horizon training since nearby horizons share structure.

The dataset contains a few other features, but these will not be used for modelling. This is mainly because these features are either derived from other features or were used in the feature engineering step to calculate other derived features. For example, `energy` (in kWh) is simply the `power` (in kW) aggregated to the hour, while `lat` (latitude) and `lon` (longitude) were used to retrieve temperature data. The geographical position of the charging station is unlikely to be a useful predictor for our target variables. 


```python
all_features = df.columns.tolist()
# Define feature sets to be used throughout the notebook
base_features = ["charging_id", "minutes_elapsed"]
static_features = ["temp", "nominal_power"]
ohe_features = ["charger_cat_low", "charger_cat_mid", "charger_cat_high"]
target_features = ["power"]
dynamic_features = ["soc", "progress", "rel_power", "d_power", "d_soc", "d_power_ema3", "d_soc_ema3", "in_taper", "dist_to_taper"]
input_features = static_features + target_features + dynamic_features
selected_features = base_features + input_features + ohe_features

# Selects the relevant features for modelling and analysis/plotting
df = df[selected_features].copy()
print(f"Dropped features: {set(all_features) - set(selected_features)}")
df.head()
```

    Dropped features: {'lon', 'lat', 'timestamp_d', 'nearest_weather_station', 'timestamp_H', 'charger_category', 'timestamp', 'energy'}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>charging_id</th>
      <th>minutes_elapsed</th>
      <th>temp</th>
      <th>nominal_power</th>
      <th>power</th>
      <th>soc</th>
      <th>progress</th>
      <th>rel_power</th>
      <th>d_power</th>
      <th>d_soc</th>
      <th>d_power_ema3</th>
      <th>d_soc_ema3</th>
      <th>in_taper</th>
      <th>dist_to_taper</th>
      <th>charger_cat_low</th>
      <th>charger_cat_mid</th>
      <th>charger_cat_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>150.00</td>
      <td>89.44</td>
      <td>40.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>150.00</td>
      <td>92.75</td>
      <td>41.00</td>
      <td>0.14</td>
      <td>0.52</td>
      <td>3.31</td>
      <td>1.00</td>
      <td>1.66</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>150.00</td>
      <td>94.81</td>
      <td>43.00</td>
      <td>0.23</td>
      <td>0.53</td>
      <td>2.06</td>
      <td>2.00</td>
      <td>1.86</td>
      <td>1.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>150.00</td>
      <td>95.68</td>
      <td>45.00</td>
      <td>0.29</td>
      <td>0.53</td>
      <td>0.87</td>
      <td>2.00</td>
      <td>1.36</td>
      <td>1.62</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>150.00</td>
      <td>96.88</td>
      <td>47.00</td>
      <td>0.34</td>
      <td>0.54</td>
      <td>1.20</td>
      <td>2.00</td>
      <td>1.28</td>
      <td>1.81</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>charging_id</th>
      <th>minutes_elapsed</th>
      <th>temp</th>
      <th>nominal_power</th>
      <th>power</th>
      <th>soc</th>
      <th>progress</th>
      <th>rel_power</th>
      <th>d_power</th>
      <th>d_soc</th>
      <th>d_power_ema3</th>
      <th>d_soc_ema3</th>
      <th>in_taper</th>
      <th>dist_to_taper</th>
      <th>charger_cat_low</th>
      <th>charger_cat_mid</th>
      <th>charger_cat_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1520022</th>
      <td>12657311</td>
      <td>11</td>
      <td>7</td>
      <td>360.00</td>
      <td>126.05</td>
      <td>57.00</td>
      <td>0.52</td>
      <td>0.29</td>
      <td>0.17</td>
      <td>3.00</td>
      <td>-2.27</td>
      <td>2.75</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1520023</th>
      <td>12657311</td>
      <td>12</td>
      <td>7</td>
      <td>360.00</td>
      <td>126.95</td>
      <td>60.00</td>
      <td>0.53</td>
      <td>0.29</td>
      <td>0.90</td>
      <td>3.00</td>
      <td>-0.68</td>
      <td>2.87</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1520024</th>
      <td>12657311</td>
      <td>13</td>
      <td>7</td>
      <td>360.00</td>
      <td>127.97</td>
      <td>62.00</td>
      <td>0.55</td>
      <td>0.30</td>
      <td>1.02</td>
      <td>2.00</td>
      <td>0.17</td>
      <td>2.44</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1520025</th>
      <td>12657311</td>
      <td>14</td>
      <td>7</td>
      <td>360.00</td>
      <td>128.01</td>
      <td>65.00</td>
      <td>0.56</td>
      <td>0.30</td>
      <td>0.04</td>
      <td>3.00</td>
      <td>0.10</td>
      <td>2.72</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1520026</th>
      <td>12657311</td>
      <td>15</td>
      <td>7</td>
      <td>360.00</td>
      <td>129.47</td>
      <td>67.00</td>
      <td>0.58</td>
      <td>0.30</td>
      <td>1.46</td>
      <td>2.00</td>
      <td>0.78</td>
      <td>2.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



## 2 - Data preparation

This section processes raw charging session data into model-ready format. Tasks include splitting into training, validation, and test sets, applying normalisation, and constructing data loaders. This aligns with our self-supervised learning task where the model will be trained to predict the next $1..H$ time steps for `power` from all previous time steps. Preparing temporal data carefully ensures fair evaluation and stable training.


#### 2.1 - Splitting the data 
Splitting the data avoids data leakage and ensures independence between train, validation, and test sets. The dataset is partitioned using grouped sampling (`GroupShuffleSplit`), ensuring that a charging session isn't split across different sets. </br>

We split the allocate 70% of the original dataset for training, 10% for validation and 20% for test. The validation dataset is used to tune the model's hyperparameters, determine anomaly detection thresholds in notebook 4, and determine the optimal parameters for the ORS simplification algorithm in notebook 5. The test dataset is used only once to evaluate the final (selected) model at the end of the modelling notebook, then again in notebook 6 for building teaching sets. 

From `mt4xai/data.py`: 
```python
def split_data(df: pd.DataFrame, test_size: float=0.2, validation_size: float=0.1):
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["charging_id"]))
    train_val_df = df.iloc[train_val_idx]

    adj_val_size = validation_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val_size, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["charging_id"]))

    train_df = train_val_df.iloc[train_idx]
    val_df   = train_val_df.iloc[val_idx]
    test_df  = df.iloc[test_idx]
    return train_df, val_df, test_df
```


```python
from mt4xai.data import split_data

train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1, random_seed=RANDOM_SEED)
print(f"Training set size: {len(train_df)} ({round(100*(len(train_df)/len(df)), 1)}%)\n"
      f"Validation set size: {len(val_df)} ({round(100*(len(val_df)/len(df)), 1)}%)\n"
      f"Test set size: {len(test_df)} ({round(100*(len(test_df)/len(df)), 1)}%)\n"
      f"Total size: {len(df)}")
```

    Training set size: 1064224 (70.0%)
    Validation set size: 152299 (10.0%)
    Test set size: 303504 (20.0%)
    Total size: 1520027



```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>charging_id</th>
      <th>minutes_elapsed</th>
      <th>temp</th>
      <th>nominal_power</th>
      <th>power</th>
      <th>soc</th>
      <th>progress</th>
      <th>rel_power</th>
      <th>d_power</th>
      <th>d_soc</th>
      <th>d_power_ema3</th>
      <th>d_soc_ema3</th>
      <th>in_taper</th>
      <th>dist_to_taper</th>
      <th>charger_cat_low</th>
      <th>charger_cat_mid</th>
      <th>charger_cat_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>150.00</td>
      <td>89.44</td>
      <td>40.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>150.00</td>
      <td>92.75</td>
      <td>41.00</td>
      <td>0.14</td>
      <td>0.52</td>
      <td>3.31</td>
      <td>1.00</td>
      <td>1.66</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>150.00</td>
      <td>94.81</td>
      <td>43.00</td>
      <td>0.23</td>
      <td>0.53</td>
      <td>2.06</td>
      <td>2.00</td>
      <td>1.86</td>
      <td>1.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>150.00</td>
      <td>95.68</td>
      <td>45.00</td>
      <td>0.29</td>
      <td>0.53</td>
      <td>0.87</td>
      <td>2.00</td>
      <td>1.36</td>
      <td>1.62</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>150.00</td>
      <td>96.88</td>
      <td>47.00</td>
      <td>0.34</td>
      <td>0.54</td>
      <td>1.20</td>
      <td>2.00</td>
      <td>1.28</td>
      <td>1.81</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.2 - Data normalisation
Min-Max scalers are fit only on the training set and applied to validation and test sets. Predictions are evaluated in original units by inverse transforming the reconstructed absolute series. Feature values are normalised using Min–Max scaling to the interval \([0, 1]\), which prevents features with large numerical ranges from dominating the loss and stabilises the gradient descent. 


```python
# MinMax scaling
fixed_features_scaler = MinMaxScaler((0, 1))
power_scaler = MinMaxScaler((0, 1))
soc_scaler = MinMaxScaler((0, 1))
delta_features = ["d_power", "d_soc", "d_power_ema3", "d_soc_ema3"]
delta_scaler = MinMaxScaler((0, 1))

fixed_features_scaler.fit(train_df[static_features])
power_scaler.fit(train_df[["power"]])
soc_scaler.fit(train_df[["soc"]])
delta_scaler.fit(train_df[delta_features])

def apply_scaling(df_: pd.DataFrame):
    df_[static_features] = fixed_features_scaler.transform(df_[static_features])
    df_["power"] = power_scaler.transform(df_[["power"]])
    df_["soc"]   = soc_scaler.transform(df_[["soc"]])
    df_[delta_features] = delta_scaler.transform(df_[delta_features])
    return df_

train_df = apply_scaling(train_df)
val_df = apply_scaling(val_df)
test_df = apply_scaling(test_df)

display(train_df.head())

# To be used later for inverse transforms
POWER_MIN = float(power_scaler.data_min_[0])
POWER_MAX = float(power_scaler.data_max_[0])
SOC_MIN = float(soc_scaler.data_min_[0]) 
SOC_MAX = float(soc_scaler.data_max_[0])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>charging_id</th>
      <th>minutes_elapsed</th>
      <th>temp</th>
      <th>nominal_power</th>
      <th>power</th>
      <th>soc</th>
      <th>progress</th>
      <th>rel_power</th>
      <th>d_power</th>
      <th>d_soc</th>
      <th>d_power_ema3</th>
      <th>d_soc_ema3</th>
      <th>in_taper</th>
      <th>dist_to_taper</th>
      <th>charger_cat_low</th>
      <th>charger_cat_mid</th>
      <th>charger_cat_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.56</td>
      <td>0.22</td>
      <td>0.33</td>
      <td>0.39</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.10</td>
      <td>0.48</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0.56</td>
      <td>0.22</td>
      <td>0.35</td>
      <td>0.40</td>
      <td>0.14</td>
      <td>0.52</td>
      <td>0.49</td>
      <td>0.20</td>
      <td>0.49</td>
      <td>0.11</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.56</td>
      <td>0.22</td>
      <td>0.35</td>
      <td>0.42</td>
      <td>0.23</td>
      <td>0.53</td>
      <td>0.49</td>
      <td>0.30</td>
      <td>0.49</td>
      <td>0.24</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.56</td>
      <td>0.22</td>
      <td>0.36</td>
      <td>0.44</td>
      <td>0.29</td>
      <td>0.53</td>
      <td>0.48</td>
      <td>0.30</td>
      <td>0.49</td>
      <td>0.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>0.56</td>
      <td>0.22</td>
      <td>0.36</td>
      <td>0.46</td>
      <td>0.34</td>
      <td>0.54</td>
      <td>0.49</td>
      <td>0.30</td>
      <td>0.49</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


 ### 2.3 - Building the datasets and data loaders

Custom dataset classes and PyTorch DataLoaders are implemented to handle variable-length sequences, which allows for efficient batching and shuffling while preserving sequence integrity. The `ChargingSessionDataset` class yields batches as packed variable-length sequences by using a "length-aware" batch sampler that keeps lengths similar within a batch. The collate function returns padded tensors and length vectors that drive packing, masking, and loss computation. Increase `NUM_WORKERS` for faster I/O operations at the cost of increased RAM usage.

**Why does batching sequences of similar lengths lead to better GPU utilisation?** </br>
Our dataset contains sequences with lengths in the range $[8, 59]$, so in each batch, the shortest sequence must be padded to length of the batch's longest sequence. If very short and very long sessions are mixed together, most of the tensor elements for short sequences become padding, which the GPU still processes even though they carry no information. Grouping sequences of similar length keeps padding minimal, so the GPU spends more FLOPs on real data rather than idle multiplications with zeros. Reduced padding also leads to a higher effective batch size, as more samples can be fit into GPU memory. Additionally, LSTMs require that packed sequences are unrolled to the maximum batch length. If lengths vary greatly, some CUDA threads would finish early and remain idle until the others finish. Grouping by similar lengths keep the threads busy for roughly the same amount of time. 



```python

# in mt4xai/data.py, we define: 

class ChargingSessionDataset(Dataset):
    """
    Yields full sessions as variable-length sequences.
    X: (T, input_size)
    Y: (T, H, C) residual targets: y_{t+h} - y_t, aligned with modelling code.
    """
    def __init__(self, df: pd.DataFrame, input_features: List[str], target_features: List[str],
                 horizon: int):
        self.groups = []
        self.input_features = input_features
        self.target_features = target_features
        self.horizon = horizon

        for sid, g in df.groupby("charging_id"):
            g = g.sort_values("minutes_elapsed").reset_index(drop=True)
            x = g[input_features].to_numpy(dtype=np.float32)            # (T, F)
            y_abs = g[target_features].to_numpy(dtype=np.float32)       # (T, C)
            T = len(g)
            # Build residual target tensor: (T, H, C) with valid region masked later
            y = np.zeros((T, horizon, y_abs.shape[1]), dtype=np.float32)
            for h in range(1, horizon+1):
                y[:-h, h-1, :] = y_abs[h:, :] - y_abs[:-h, :]
            self.groups.append((sid, x, y, T))

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx: int):
        sid, x, y, T = self.groups[idx]
        return sid, x, y, T  # keep session_id for post-hoc grouping

@dataclass
class LengthBucketSampler(Sampler[List[int]]):
    """Batches indices by similar sequence length.
    Sorts indices by length, then slices into contiguous batches. Optionally shuffles
    the list of batches to avoid curriculum effects while keeping within-batch lengths similar.

    Args:
        dataset: A ChargingSessionDataset from which to retrieve sequence lengths
        batch_size: Number of items per batch.
        shuffle: Whether to shuffle the order of batches at iteration time.

    Yields:
        Lists of dataset indices, one per batch.
    """
    dataset: ChargingSessionDataset
    batch_size: int = 8 
    shuffle: bool = True

    def __post_init__(self):
        self.sorted_indices = np.argsort(self.lengths).tolist()
        self.batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self) -> Iterable[List[int]]:
        if self.shuffle:
            random.shuffle(self.batches)
        for b in self.batches:
            yield b

    def __len__(self) -> int:
        return math.ceil(len(self.sorted_indices) / self.batch_size)
    
    @property
    def lengths(self) -> List[int]:
        """returns per-item sequence lengths from a ChargingSessionDataset-like object.
        expects dataset.groups: List[Tuple[sid, x, y, T]]"""
        return [grp[3] for grp in self.dataset.groups]


def session_collate_fn(batch: List[Tuple]) -> Tuple[List[int] | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads variable-length sessions to (B, T_max, ·) and carries IDs + lengths.

    Accepts either items of the form ``(sid, x, y, L)`` or ``(x, y, L)``.
    Returns tensors on CPU; the training step can move them to device.

    Args:
        batch: List of samples; each item is (sid, x, y, L) or (x, y, L).

    Returns:
        session_ids: List of session IDs (or None if not provided).
        X: Tensor of shape (B, T_max, F) float32.
        Y: Tensor of shape (B, T_max, H, C) float32.
        lengths: Tensor of shape (B,) int64.
    """
    first = batch[0]
    if len(first) == 4:
        session_ids, all_x, all_y, lengths = zip(*batch)
        session_ids = list(session_ids)
    else:
        all_x, all_y, lengths = zip(*batch)
        session_ids = None

    B = len(all_x)
    T_max = int(max(lengths))
    F = int(all_x[0].shape[1])
    H = int(all_y[0].shape[1])
    C = int(all_y[0].shape[2])

    X = np.zeros((B, T_max, F), dtype=np.float32)
    Y = np.zeros((B, T_max, H, C), dtype=np.float32)
    for i, (x, y, L) in enumerate(zip(all_x, all_y, lengths)):
        Li = int(L)
        X[i, :Li] = x
        Y[i, :Li] = y

    return (
        session_ids,
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
        torch.tensor(lengths, dtype=torch.long),
    )
...
```


```python
from mt4xai.data import ChargingSessionDataset

# Builds the datasets
train_dataset = ChargingSessionDataset(train_df, input_features, target_features, horizon=HORIZON)
val_dataset = ChargingSessionDataset(val_df, input_features, target_features, horizon=HORIZON)
test_dataset = ChargingSessionDataset(test_df,input_features, target_features, horizon=HORIZON)

print(f"Input features: {input_features}")
print(f"First session in training data: ")
train_dataset.groups[0][1]
```

    Input features: ['temp', 'nominal_power', 'power', 'soc', 'progress', 'rel_power', 'd_power', 'd_soc', 'd_power_ema3', 'd_soc_ema3', 'in_taper', 'dist_to_taper']
    First session in training data: 





    array([[0.5645161 , 0.22222222, 0.33399305, 0.3939394 , 0.        ,
            0.49688888, 0.482091  , 0.1       , 0.48277444, 0.03449224,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.34635347, 0.4040404 , 0.14453241,
            0.5152778 , 0.49277186, 0.2       , 0.4934412 , 0.11494794,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.35404608, 0.42424244, 0.22907846,
            0.5267222 , 0.4887383 , 0.3       , 0.49474633, 0.23563151,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.3572949 , 0.44444445, 0.28906482,
            0.53155553, 0.48489836, 0.3       , 0.49156404, 0.2959733 ,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.36177602, 0.46464646, 0.33559388,
            0.5382222 , 0.48596323, 0.3       , 0.49103633, 0.3261442 ,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.36453938, 0.47474748, 0.37361085,
            0.5423333 , 0.48447886, 0.2       , 0.4892901 , 0.26077393,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.3692072 , 0.4949495 , 0.4057538 ,
            0.5492778 , 0.48612455, 0.3       , 0.4900605 , 0.3085445 ,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37185854, 0.5151515 , 0.43359724,
            0.55322224, 0.48438206, 0.3       , 0.4887055 , 0.33242977,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37238133, 0.53535354, 0.4581569 ,
            0.554     , 0.48254275, 0.3       , 0.48619112, 0.34437242,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.3722693 , 0.54545456, 0.4801263 ,
            0.5538333 , 0.48199418, 0.2       , 0.4843861 , 0.26988804,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37189588, 0.56565654, 0.5       ,
            0.5532778 , 0.4817683 , 0.3       , 0.483258  , 0.31310156,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37152246, 0.5858586 , 0.5181433 ,
            0.5527222 , 0.4817683 , 0.3       , 0.48269394, 0.3347083 ,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37570485, 0.6060606 , 0.5348335 ,
            0.55894446, 0.48570508, 0.3       , 0.48634347, 0.34551167,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.3759289 , 0.61616164, 0.5502862 ,
            0.5592778 , 0.4822846 , 0.2       , 0.4847523 , 0.27045768,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37589157, 0.6363636 , 0.56467235,
            0.5592222 , 0.48205873, 0.3       , 0.48373115, 0.31338638,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37551814, 0.65656567, 0.57812965,
            0.55866665, 0.4817683 , 0.3       , 0.4829305 , 0.33485073,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.37895367, 0.67676765, 0.59077084,
            0.5637778 , 0.4850597 , 0.3       , 0.48581725, 0.3455829 ,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.390978  , 0.6969697 , 0.6026893 ,
            0.58166665, 0.49248144, 0.3       , 0.49467257, 0.350949  ,
            0.        , 0.        ],
           [0.5645161 , 0.22222222, 0.4008738 , 0.7070707 , 0.6139632 ,
            0.5963889 , 0.49064213, 0.2       , 0.49726337, 0.2731763 ,
            1.        , 0.        ],
           [0.5645161 , 0.22222222, 0.40124723, 0.72727275, 0.6246587 ,
            0.59694445, 0.48241368, 0.3       , 0.49034116, 0.3147457 ,
            1.        , 0.06896552],
           [0.5645161 , 0.22222222, 0.4196572 , 0.74747473, 0.63483226,
            0.6243333 , 0.49799934, 0.3       , 0.50244516, 0.33553037,
            1.        , 0.13793103],
           [0.5645161 , 0.22222222, 0.4193958 , 0.7676768 , 0.64453244,
            0.62394446, 0.4818651 , 0.3       , 0.4923842 , 0.3459227 ,
            1.        , 0.20689656],
           [0.5645161 , 0.22222222, 0.4191344 , 0.7878788 , 0.6538013 ,
            0.62355554, 0.4818651 , 0.3       , 0.48735374, 0.3511189 ,
            1.        , 0.27586207]], dtype=float32)



 ## 3 - Model Architectures

Two sequence models are explored: the (recurrent) Long Short-Term Memory (LSTM) network and the (convolutional) Temporal Convolutional Network (TCN). Both architectures are designed to model temporal dependencies but differ in how they capture long-range context. In short, while LSTMs are able to capture long-range and non-stationary dependencies with complex cross-feature dynamics, TCNs are good for recognising short-range patterns and local motifs. Together, the two architectures are complementary and the comparison may provide insight into which inductive bias best capture the dynamics of EV charging curves. 

Each network is designed to produce a tensor of residuals with shape $(B, T, H, C)$, where $B$ is the batch size, $T$ is the sequence length, $H$ is the number of prediction horizons, and $C$ is 1 (the number of target channels, i.e. power).

### 3.1 Defining the LSTM Architecture
The LSTM is a recurrent neural network (RNN) that introduces gating mechanisms (input, forget, output) to mitigate the vanishing gradient problem. It is well-suited for learning long-term dependencies in sequential data. As the longest sequence is only 59 time steps, we could have chosen a simpler Gated Recurrent Unit (GRU) architecture. However, as predicting the abrupt taper dynamics around 80% SOC is very important for achieving a high prediction accuracy, we wanted to utilise the LSTM's separate memory and hidden states that enable it to retain contextual information about earlier charge phases (ramp-up, plateu) while adapting to abrupt power–SOC changes later in the session (taper). Its richer gating structure provides finer control over temporal dependencies, which is beneficial for the non-stationary nature of EV charging curves. The network's dual state memory also supports more accurate multi-horizon residual forecasts by disentangling short-term and long-term effects. Although it is slightly more computationally demanding than an equivalent GRU (~30% more parameters due to the extra gate), this added complexity presumably would lead to smoother training, lower horizon-wise error correlation, and improved stability across diverse session lengths. Given the modest dataset size, feature space and horizon length, the added computational requirement of an LSTM is minor. 

The `MultiHorizonLSTM` class (derived from PyTorch's `Module`) defines the network architecture and the forward pass that packs variable-length sequences, applies the LSTM layer computations. Inputs are packed by length, processed by the LSTM, then unpacked to a padded tensor. A linear (fully connected) head outputs residuals for all horizons and targets at every step. The forward pass also returns output lengths used to construct validity masks for padded sequences. 

From `mt4xai.model.py`:
```python

# LSTM multi-horizon residual model
class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, horizon: int, num_targets: int,
                 num_layers: int, dropout: float=0.0):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, horizon * num_targets)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor):
        packed_x = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, out_lengths = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        out = self.linear(out).view(out.shape[0], out.shape[1], self.horizon, self.num_targets)
        return out, out_lengths
```

### 3.2 Defining the TCN Architecture
The TCN is a convolutional sequence model that uses causal, dilated convolutions to capture temporal dependencies. Its receptive field can be widened efficiently, allowing it to model long-range dependencies without recurrence. The `MultiHorizonTCN` with `SkipTCNBlock` implementation is a causal WaveNet-style TCN with dilated depthwise separable convolutions, residual connections, and global skip aggregation before a $1×1$ head that emits all horizons and targets. This gives a controllable receptive field without recurrence.

TCNs offer several advantages over recurrent architectures like LSTMs if one is willing to trade sequence-wise memory for parallelism and more stable gradients. 
For the EV-charging power prediction problem the main benefits are full parallelism and faster training, more stable gradients and easier optimisation due to convolutional layers (no vanishing/exploding gradient problem), controllable receptive fields and better exploitation of short-term temporal patterns/structure. 

From `mt4xai.model.py`:
```python

class MultiHorizonTCN(nn.Module):
    """
    WaveNet-style TCN with input 1x1 projection, L residual blocks with exponentially increasing dilations, 
    and global skip accumulation --> head.
    """
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int,
                 kernel_size: int, horizon: int, num_targets: int,
                 dropout: float=0.0, dilation_growth: int = 2):
        super().__init__()
        self.horizon, self.num_targets = horizon, num_targets

        self.input_proj = weight_norm(nn.Conv1d(input_size, hidden_dim, kernel_size=1))
        self.blocks = nn.ModuleList([
            SkipTCNBlock(hidden_dim, kernel_size, dilation=(dilation_growth ** i), dropout=dropout)
            for i in range(num_layers)
        ])
        # combines all of the skips, a small mixer, then predict residuals for all horizons.
        self.post = nn.Sequential(
            nn.ReLU(),
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden_dim, horizon * num_targets, kernel_size=1)

    @staticmethod
    def receptive_field(kernel_size, num_layers, dilation_growth=2):
        """RF = 1 + (k-1) * sum_{i=0}^{L-1} (growth^i)"""
        return 1 + (kernel_size - 1) * sum(dilation_growth ** i for i in range(num_layers))

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor):
        # x: (B, T, F) -> (B, C, T)
        B, T, RF = x.shape
        h = self.input_proj(x.transpose(1, 2))
        skip_sum = None
        for blk in self.blocks:
            h, s = blk(h)
            skip_sum = s if skip_sum is None else (skip_sum + s)

        h = self.post(skip_sum)
        y = self.head(h).transpose(1, 2).view(B, T, self.horizon, self.num_targets)
        return y, seq_lengths


class SkipTCNBlock(nn.Module):
    """Residual block that also emits a skip tensor."""
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        c = channels
        self.conv1 = DSConv1d(c, c, kernel_size, dilation, dropout)
        self.conv2 = DSConv1d(c, c, kernel_size, dilation, dropout)
        self.residual = weight_norm(nn.Conv1d(c, c, kernel_size=1))
        self.skip = weight_norm(nn.Conv1d(c, c, kernel_size=1))

    def forward(self, x):  # x: (B, C, T)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return self.residual(h) + x, self.skip(h)


class DSConv1d(nn.Module):
    """Causal depthwise-separable 1D conv with weight norm."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.k, self.d = kernel_size, dilation
        self.dw = weight_norm(nn.Conv1d(in_ch, in_ch, kernel_size,
                                        groups=in_ch, padding=0, dilation=dilation, bias=True))
        self.pw = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True))
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = _causal_pad(x, self.k, self.d)
        x = self.dw(x)
        x = F.relu(x)
        x = self.pw(x)
        return self.do(x)

def _causal_pad(x, k, d):
    # Left-pad so output stays length T and remains causal
    return F.pad(x, ((k-1)*d, 0 ))
```

### 3.3 Prediction Horizons with Horizon Weights
We want to force the models to learn longer-term dependencies.... 
A multi-horizon model predicts the $H$ future time steps in the same forward pass - as opposed to rolling out predictions step-by-step. This approach provides three benefits:
1. Multiple horizons capture the temporal evolution of the charging curve, as each future horizon reflects a different physical regime: ramp-up, plateau, or taper. Predicting them jointly allows the model to learn how near-term and longer-term dynamics relate to each other, which a single-step model cannot capture.
2. Reduced error accumulation. If you were to feed the model’s own predictions back into the next step with recursive forecasting, small residual errors would compound over time. Multi-horizon forecasting avoids this by directly predicting all horizons from the same observed context, providing consistent forecasts without autoregressive drift.
3. Improved efficiency and interpretability. A single shared representation encodes the full short term trajectory of the expected charging behaviour. This is computationally cheaper at inference time and enables horizon-wise error analysis (e.g. Macro-RMSE and RWSE) in downstream inference tasks (e.g. anomaly detection, curve ORS simplification). 

Longer horizons are naturally harder to predict, so noisy and uncertain targets might dominate the loss. In order to control how much each future step contributes to the training loss and evaluation metrics, we weight the prediction horizons with an exponential decay $w_h = \exp!\big(-\alpha(h-1)\big)$ for $h \in {1..H}$. This emphasises short-term accuracy when computing losses or metrics, but the level of decay is tunable as $\alpha > 0$ controls how quickly the weights decrease with horizon. Additionally, as gradients from distant/high-variance horizons could destabilise early training, applying decay keeps gradient magnitudes balanced across horizons, speeds convergence and adds a form of soft regularisation. 

Weights are shaped to broadcast over (B, T, H, C).

```python

def horizon_weights(H: int, alpha: float, device: torch.device) -> Tensor:
    """Computes exponentially decaying weights for each prediction horizon.

    The weights follow the form w[h] = exp(-alpha * (h - 1)) for h in {1, …, H}.
    Higher alpha => increased decay rate, emphasising shorter horizons.

    Args:
        H (int): Number of prediction horizons.
        alpha (float): Exponential decay parameter; higher values produce faster decay.
        device (torch.device): The device on which to allocate the resulting tensor.

    Returns:
        Tensor: Horizon weight tensor of shape (1, 1, H, 1) on the specified device.
    """
    w = torch.exp(-alpha * torch.arange(1, H + 1, device=device, dtype=torch.float32))
    return w.view(1, 1, H, 1)

```

### 3.4 Model Builders
Convenience builders create models from config dictionaries. They capture the model's input size, horizon, target count, depth, and dropout, and place the model on the requested device.

Runtime note. For the TCN, keep the product of channels, layers, and kernel size modest to avoid inflating the receptive field and memory footprint.

from `mt4xai.model.py`:
```python
def build_model_lstm(cfg, horizon, input_features, target_features, device):
    return MultiHorizonLSTM(
        input_size=len(input_features),
        hidden_dim=int(cfg["hidden_dim"]),
        horizon=horizon,
        num_targets=len(target_features),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)


def build_model_tcn(cfg, horizon, input_features, target_features, device):
    return MultiHorizonTCN(
        input_size=len(input_features),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        horizon=horizon,
        num_targets=len(target_features),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
```

## 4 - Model Training and Tuning
This section outlines the training procedures, utility functions, and hyperparameter optimisation (tuning) strategies used to fit both LSTM and TCN models. 

**Model training with Huber loss:**</br>
A key design choice in this project is the use of the Huber loss, also known as SmoothL1, which combines the benefits of Mean Squared Error (MSE) and Mean Absolute Error (MAE). For small residuals, the loss behaves quadratically like MSE, ensuring sensitivity to fine-grained prediction errors. For large residuals, it switches to a linear penalty like MAE, making it more robust to outliers. This is particularly useful in EV charging data, where noisy or atypical charging behaviours may otherwise dominate the training process. In this use case, training with Huber loss stabilises gradients, limits the influence of rare spikes, and provides faster convergence due to the use of MSE around small errors. Additionally, since the scale of residuals remain consistent with RMSE in kW, validation and training magnitudes are comparable. The drawbacks of using the Huber loss are the extra hyperparameter $\delta$ that needs to be tuned, it is slower to compute than simpler alternatives (such as MSE) and it is not exactly aligned with the validation metric (see below). 

For an error $e$, the Huber loss with parameter $\delta$ is
$
L_\delta(e) =
\begin{cases}
\frac{1}{2} e^2, & |e| \le \delta, \\[6pt]
\delta (|e| - \frac{1}{2}\delta), & |e| > \delta.
\end{cases}
$

We compute the loss over valid $(t, h)$ pairs with horizon weights and a 4D-mask derived from sequence lengths. Gradients are clipped, and a cosine or plateau scheduler adapts the learning rate. 



**Model validation with Macro-RMSE:**</br>
At the end of each epoch (after the backwards pass), we validate the model's predictions across horizons on the validation set with Macro-RMSE, an interpretable metric which reflexts errors in kW (original units). It aggregates per-horizon MSE with horizon weights before taking the root at the session level and averaging across sessions/sequences. Macro-RMSE is also used for evaluating and selecting the final model in this notebook's section 5.3 and for the downstream anomaly detection task in notebook #4. Be aware that the Huber loss we use for training differs slightly in curvature and sensitivity, so the two metrics are not exactly aligned - though they stay comparable and use the same units (error in kW). For visualization and debugging purposes, we also calculate and report the Huber loss on the validation set, and we report the training loss and both validation losses separately.

For a single target channel (power) and a given session $i$, let $e_{t,h}^{(i)} = y_{t,h}^{(i)} - \hat{y}_{t,h}^{(i)}$ be the residual at time $t$ and horizon $h \in \{1, \dots, H\}$. With horizon weights $w_h \ge 0$ that sum to 1 (uniform if no decay is desired), the session-level RMSE is defined as:

$$
\mathrm{RMSE}^{(i)} = 
\sqrt{ \sum_{h=1}^{H} w_h 
\left( \frac{1}{T_{i,h}} 
\sum_{t=1}^{T_{i,h}} 
\big( e_{t,h}^{(i)} \big)^2
\right) }.
$$

Macro-averaging across all sessions in the test set gives us:  

$$
\mathrm{MacroRMSE} = 
\frac{1}{N} 
\sum_{i=1}^{N} 
\mathrm{RMSE}^{(i)}.
$$

Macro-RMSE keeps the error in the same physical unit (kW), which makes results interpretable and directly comparable to operational thresholds used in later stages such as anomaly detection. Macro-averaging ensures that each session and each horizon contribute equally (or proportionally to $w_h$) to the overall score, preventing domination by shorter horizons or longer sequences.  

This formulation replaces the earlier Macro-MSE definition to ensure full consistency between the modelling, anomaly-detection, and curve-simplification stages of the pipeline.

**Tuning/hyperparameter optimisation:**</br>
The goal of the hyperparameter optimisation process is to find the hyperparameter configuration that minimises the model's validation loss metric (Macro-RMSE). The Ray library for PyTorch is employed to manage the model configurations and run a search across possible hyperparameters. Ray Tune with the BOHB (Bayesian Optimisation with Hyperband) scheduler explores hyperparameters efficiently, balancing breadth of search with computational efficiency.

**Step-by-step training flow** </br>
Per epoch (complete pass over the training set), we:
1. Pack sequences and run the model to get residuals $(B, T, H, C)$.
2. Reconstruct absolute power $\hat y$ by adding the base input power.
3. Inverse transform to kW.
4. Apply Huber, horizon weights, and the validity mask, then average.
5. Backpropagate with automatic mixed precision (AMP) and clip gradients.
6. Evaluate validation RMSE in kW and step the BOHB w/Hyperband scheduler on the validation metric.


### 4.1 Training Utilities
Utility functions handle tasks such as loss computation, gradient clipping, vectorised masks, horizon weights, gradient clipping, scaler helpers and learning rate scheduling. We also define some utilities for reporting and plotting training results. 


```python
from mt4xai.train import build_adamw, inv_minmax_channel_torch

# Global column indices (used in training/tuning loop and plotting)
IDX_TEMP = input_features.index("temp")
IDX_NOM = input_features.index("nominal_power")
IDX_POWER = input_features.index("power")
IDX_SOC = input_features.index("soc")
```

### 4.2 Defining the Ray Trainables
Ray Tune wraps each model as a trainable. BOHB explores network depth, width, dropout, learning rate, and horizon weighting. Trials report the validation metric continuously and checkpoint the best state dict/trial by validation metric. The notebook can restore a previous run and resume unfinished or errored trials. After tuning, we reload the best checkpoint to export a clean .pth for downstream notebooks.

### 4.3 Hyperparameter Search Spaces
Search spaces for parameters such as the number of hidden layers, the hidden dimension (number of neurons in each hidden layer), learning rate, batch size and dropout are defined. 


```python
def make_bohb_cs_for_lstm(seed=RANDOM_SEED):
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add([
        CSH.CategoricalHyperparameter("hidden_dim", [64, 128, 192, 256, 384, 512]),
        CSH.CategoricalHyperparameter("num_layers", [1, 2, 3, 4, 5]),
        CSH.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.3),
        CSH.UniformFloatHyperparameter("lr", lower=3e-4, upper=3e-2, log=True),
        CSH.UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=3e-3, log=True),
        CSH.CategoricalHyperparameter("batch_size", [32, 64, 96]),
        CSH.CategoricalHyperparameter("grad_clip_norm", [0.0, 1.0, 3.0, 5.0]),
        CSH.UniformFloatHyperparameter("alpha_h", lower=0.20, upper=0.55),
        CSH.Constant("num_epochs", 200),
    ])
    return cs

def make_bohb_cs_for_tcn(seed=RANDOM_SEED):
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add([
        CSH.CategoricalHyperparameter("hidden_dim", [128, 160, 192]),
        CSH.CategoricalHyperparameter("num_layers", [4, 5, 6]),
        CSH.CategoricalHyperparameter("kernel_size", [3, 5, 7]),
        CSH.UniformFloatHyperparameter("dropout", lower=0.18, upper=0.32),
        CSH.UniformFloatHyperparameter("lr", lower=5e-4, upper=2e-3, log=True),
        CSH.UniformFloatHyperparameter("weight_decay",lower=1e-6, upper=1e-4, log=True),
        CSH.CategoricalHyperparameter("batch_size", [32, 64]),
        CSH.CategoricalHyperparameter("grad_clip_norm", [1.0, 3.0]),
        CSH.UniformFloatHyperparameter("alpha_h", lower=0.20, upper=0.40),
        CSH.Constant("num_epochs", 200),
    ])
    return cs

# fixed, non-searched parameters that every trial needs
BASE_CFG = {
    "device": DEVICE,
    "input_features": input_features,
    "target_features": target_features,
    "horizon": HORIZON,
}
```

### 4.4 Training and Tuning an LSTM Model
We launch the LSTM tuner, restore if a prior run exists, then export the best model and config. The validation metric is the macro-averaged RMSE over horizons for power in kW.

**Runtime note**: Limit `max_concurrent_trials` to available CPU cores and GPU VRAM (memory) and scale batch size to fit memory. If runs are preempted, recovery will continue from the last good checkpoint.


```python
from mt4xai.model import build_model_lstm
from mt4xai.train import restore_resultgrid, save_final_model_pth, tune_run_status, tune_train_lstm

results_lstm = None
best_lstm_res = None
cfg_lstm = None
model_lstm = None

# common Tune objects
cs = make_bohb_cs_for_lstm(seed=RANDOM_SEED)
bohb = TuneBOHB(space=cs, metric="val_metric", mode="min")
hb = HyperBandForBOHB(time_attr="epoch", max_t=200, reduction_factor=3)

train_ref = ray.put(train_dataset)
val_ref = ray.put(val_dataset)
trainable_lstm = tune.with_parameters(
    tune_train_lstm, 
    train_dataset_ref=train_ref, 
    val_dataset_ref=val_ref, 
    num_workers=NUM_WORKERS, 
    power_min=POWER_MIN, 
    power_max=POWER_MAX, 
    idx_power=IDX_POWER)
trainable_lstm = tune.with_resources(trainable_lstm, {"cpu": 10, "gpu": 0.5})  # per trial

ckpt_cfg = CheckpointConfig(
    num_to_keep=3,
    checkpoint_score_attribute="val_metric",
    checkpoint_score_order="min",
)
run_cfg = RunConfig(
    storage_path=MODEL_FOLDER_PATH,
    name=RAY_TUNE_FOLDER_NAME_LSTM,
    checkpoint_config=ckpt_cfg,
    verbose=1,
)

run_root = os.path.join(MODEL_FOLDER_PATH, RAY_TUNE_FOLDER_NAME_LSTM)
status = tune_run_status(run_root)

if TRAIN_LSTM:
    if RESUME_TRAIN_IF_CKPT and status["exists"] and not status["finished"]:
        # unfinished run? try minimal restore
        try:
            tuner = Tuner.restore(
                run_root,
                trainable=trainable_lstm,
                resume_unfinished=True,
                resume_errored=True,
                param_space=BASE_CFG,
            )
            print("[lstm] resuming unfinished run (using original searcher/scheduler from disk).")
            results_lstm = tuner.fit()
        except KeyError as e:
            # BOHB resume glitch. trial_to_params missing for a restored trial
            print(f"[lstm][warn] BOHB resume hit KeyError ({e}). "
                  "Your Ray build cannot swap the searcher during restore. "
                  "Either start a fresh run under a new name, or "
                  "load the best trial checkpoint and continue training outside Tune.")
            # pick one path automatically: start a FRESH run under a new folder name
            fresh_name = f"{RAY_TUNE_FOLDER_NAME_LSTM}_fresh"
            run_cfg_fresh = RunConfig(
                storage_path=MODEL_FOLDER_PATH,
                name=fresh_name,
                checkpoint_config=ckpt_cfg,
                verbose=1,
            )
            tuner = Tuner(
                trainable_lstm,
                tune_config=TuneConfig(
                    search_alg=bohb,
                    scheduler=hb,
                    num_samples=64,
                    max_concurrent_trials=2,
                    metric="val_metric", mode="min",
                ),
                run_config=run_cfg_fresh,
                param_space=BASE_CFG,
            )
            print(f"[lstm] launching a fresh run in '{fresh_name}' …")
            results_lstm = tuner.fit()
        except Exception as e:
            # any other restore failure? also start a fresh run
            print(f"[lstm][warn] restore failed ({type(e).__name__}: {e}). Starting a fresh run.")
            tuner = Tuner(
                trainable_lstm,
                tune_config=TuneConfig(
                    search_alg=bohb,
                    scheduler=hb,
                    num_samples=64,
                    max_concurrent_trials=2,
                    metric="val_metric", mode="min",
                ),
                run_config=run_cfg,
                param_space=BASE_CFG,
            )
            results_lstm = tuner.fit()
    elif RESUME_TRAIN_IF_CKPT and status["exists"] and status["finished"]:
        # run already finished, do not resume. simply read results
        print(f"[lstm] previous run is finished (statuses={status['statuses']}); loading results only.")
        results_lstm = restore_resultgrid(RAY_TUNE_RUN_FOLDER_PATH_LSTM)
    else:
        # fresh run
        tuner = Tuner(
            trainable_lstm,
            tune_config=TuneConfig(
                search_alg=bohb,
                scheduler=hb,
                num_samples=64,
                max_concurrent_trials=2,
                metric="val_metric", mode="min",
            ),
            run_config=run_cfg,
            param_space=BASE_CFG,
        )
        results_lstm = tuner.fit()

    # rebuild best & export clean .pth 
    if results_lstm is None:
        raise RuntimeError("No ResultGrid available after run/resume.")

    best_lstm_res = results_lstm.get_best_result(metric="val_metric", mode="min")
    cfg_lstm = dict(best_lstm_res.config)

    ckpt_dir = best_lstm_res.checkpoint.to_directory()
    assert ckpt_dir is not None, "no checkpoint directory found for best_lstm"
    payload = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=DEVICE)
    model_lstm = build_model_lstm(cfg_lstm)
    model_lstm.load_state_dict(payload["model_state_dict"])
    model_lstm.eval()
    save_final_model_pth(model_lstm, best_lstm_res, out_path=MODEL_PATH_LSTM)
    print(f"[LSTM] saved clean final model .pth at {MODEL_PATH_LSTM}")

else:
    # restore most recent completed run
    results_lstm = restore_resultgrid(RAY_TUNE_RUN_FOLDER_PATH_LSTM)
    if results_lstm is not None and len(list(results_lstm)) > 0:
        best_lstm_res = results_lstm.get_best_result(metric="val_metric", mode="min")
        cfg_lstm  = dict(best_lstm_res.config)
        for k, v in BASE_CFG.items(): cfg_lstm.setdefault(k, v)
        try:
            ckpt_dir = best_lstm_res.checkpoint.to_directory()
            payload  = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=DEVICE)
            model_lstm = build_model_lstm(cfg_lstm)
            model_lstm.load_state_dict(payload["model_state_dict"]); model_lstm.eval()
            print("[LSTM] restored model from Tune checkpoint.")
        except AttributeError:
            if os.path.exists(MODEL_PATH_LSTM):
                final = torch.load(MODEL_PATH_LSTM, map_location=DEVICE)
                cfg_lstm = dict(final.get("config", cfg_lstm))
                model_lstm = build_model_lstm(cfg_lstm)
                model_lstm.load_state_dict(final["model_state_dict"]); model_lstm.eval()
                print("[LSTM] loaded model from exported .pth.")
            else:
                print("[LSTM] no Tune checkpoint or .pth found.")
    # or simply load the saved model from path (without checkpoint/config data)
    elif os.path.exists(MODEL_PATH_LSTM):
        final = torch.load(MODEL_PATH_LSTM, map_location=DEVICE)
        cfg_lstm = dict(final.get("config", BASE_CFG))
        for k, v in BASE_CFG.items(): cfg_lstm.setdefault(k, v)
        model_lstm = build_model_lstm(cfg_lstm)
        model_lstm.load_state_dict(final["model_state_dict"]); model_lstm.eval()
        print("[LSTM] loaded model from exported .pth.")
    else:
        print("[LSTM] no previous results or model found.")

```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2025-11-13 00:19:51</td></tr>
<tr><td>Running for: </td><td>01:37:24.68        </td></tr>
<tr><td>Memory:      </td><td>6.7/13.6 GiB       </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      Using HyperBand: num_stopped=62 total_brackets=2<br>Round #0:<br>  Bracket(Max Size (n)=2, Milestone (r)=162, completed=100.0%): {TERMINATED: 64} <br>Logical resource usage: 10.0/24 CPUs, 0.5/1 GPUs (0.0/1.0 accelerator_type:G)
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name              </th><th>status    </th><th>loc               </th><th style="text-align: right;">  alpha_h</th><th style="text-align: right;">  batch_size</th><th style="text-align: right;">   dropout</th><th style="text-align: right;">  grad_clip_norm</th><th style="text-align: right;">  hidden_dim</th><th style="text-align: right;">         lr</th><th style="text-align: right;">  num_epochs</th><th style="text-align: right;">  num_layers</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  train_loss</th><th style="text-align: right;">  val_loss</th><th style="text-align: right;">  val_rmse_power</th></tr>
</thead>
<tbody>
<tr><td>tune_train_lstm_7102e8f0</td><td>TERMINATED</td><td>172.31.29.69:3950 </td><td style="text-align: right;"> 0.331089</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.180335  </td><td style="text-align: right;">               1</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00219286 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.000977501</td><td style="text-align: right;">     2</td><td style="text-align: right;">        10.1464 </td><td style="text-align: right;">    0.611332</td><td style="text-align: right;">  0.569806</td><td style="text-align: right;">         5.03946</td></tr>
<tr><td>tune_train_lstm_03a68a9e</td><td>TERMINATED</td><td>172.31.29.69:4041 </td><td style="text-align: right;"> 0.397151</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.117318  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.0214012  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   0.00227148 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        11.357  </td><td style="text-align: right;">    0.990552</td><td style="text-align: right;">  0.913504</td><td style="text-align: right;">         7.45338</td></tr>
<tr><td>tune_train_lstm_adae3bb5</td><td>TERMINATED</td><td>172.31.29.69:4477 </td><td style="text-align: right;"> 0.205473</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.181788  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.0178462  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   3.36997e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        11.1058 </td><td style="text-align: right;">    1.47625 </td><td style="text-align: right;">  1.52678 </td><td style="text-align: right;">        12.4096 </td></tr>
<tr><td>tune_train_lstm_65a51fc9</td><td>TERMINATED</td><td>172.31.29.69:22058</td><td style="text-align: right;"> 0.443431</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.26414   </td><td style="text-align: right;">               5</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.000497995</td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   8.64529e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        79.6345 </td><td style="text-align: right;">    0.386252</td><td style="text-align: right;">  0.371437</td><td style="text-align: right;">         3.89876</td></tr>
<tr><td>tune_train_lstm_aa435faa</td><td>TERMINATED</td><td>172.31.29.69:4987 </td><td style="text-align: right;"> 0.341343</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.0800343 </td><td style="text-align: right;">               0</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.001646   </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   4.45192e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">        10.9619 </td><td style="text-align: right;">    0.564989</td><td style="text-align: right;">  0.550956</td><td style="text-align: right;">         4.91964</td></tr>
<tr><td>tune_train_lstm_e63f9f80</td><td>TERMINATED</td><td>172.31.29.69:5298 </td><td style="text-align: right;"> 0.379233</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.0340421 </td><td style="text-align: right;">               1</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00392381 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.00179191 </td><td style="text-align: right;">     2</td><td style="text-align: right;">         9.09738</td><td style="text-align: right;">    0.561096</td><td style="text-align: right;">  0.517972</td><td style="text-align: right;">         4.72657</td></tr>
<tr><td>tune_train_lstm_8745d40f</td><td>TERMINATED</td><td>172.31.29.69:5533 </td><td style="text-align: right;"> 0.529047</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.148168  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.0264937  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.000325127</td><td style="text-align: right;">     2</td><td style="text-align: right;">         8.12008</td><td style="text-align: right;">    0.593073</td><td style="text-align: right;">  0.552703</td><td style="text-align: right;">         4.95474</td></tr>
<tr><td>tune_train_lstm_b1eb8be6</td><td>TERMINATED</td><td>172.31.29.69:22149</td><td style="text-align: right;"> 0.529161</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.289086  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.000469326</td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000277091</td><td style="text-align: right;">     6</td><td style="text-align: right;">        72.005  </td><td style="text-align: right;">    0.41333 </td><td style="text-align: right;">  0.403128</td><td style="text-align: right;">         4.05609</td></tr>
<tr><td>tune_train_lstm_03c93d06</td><td>TERMINATED</td><td>172.31.29.69:6060 </td><td style="text-align: right;"> 0.503475</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.00828503</td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.000358749</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.000846869</td><td style="text-align: right;">     2</td><td style="text-align: right;">        25.9844 </td><td style="text-align: right;">    0.506587</td><td style="text-align: right;">  0.504171</td><td style="text-align: right;">         4.65794</td></tr>
<tr><td>tune_train_lstm_b12efce2</td><td>TERMINATED</td><td>172.31.29.69:6391 </td><td style="text-align: right;"> 0.246461</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.0553001 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.00567104 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   2.61714e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">        13.6587 </td><td style="text-align: right;">    0.622252</td><td style="text-align: right;">  0.609818</td><td style="text-align: right;">         5.30391</td></tr>
<tr><td>tune_train_lstm_7d92b739</td><td>TERMINATED</td><td>172.31.29.69:6642 </td><td style="text-align: right;"> 0.542829</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0144177 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         384</td><td style="text-align: right;">0.0162749  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.00187753 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        20.8238 </td><td style="text-align: right;">    0.524606</td><td style="text-align: right;">  0.512351</td><td style="text-align: right;">         4.6674 </td></tr>
<tr><td>tune_train_lstm_73c98c8c</td><td>TERMINATED</td><td>172.31.29.69:6917 </td><td style="text-align: right;"> 0.224276</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.00698158</td><td style="text-align: right;">               0</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.000679204</td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000231155</td><td style="text-align: right;">     2</td><td style="text-align: right;">        20.987  </td><td style="text-align: right;">    0.630179</td><td style="text-align: right;">  0.612081</td><td style="text-align: right;">         5.24347</td></tr>
<tr><td>tune_train_lstm_1e46a91f</td><td>TERMINATED</td><td>172.31.29.69:7229 </td><td style="text-align: right;"> 0.373796</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.28704   </td><td style="text-align: right;">               0</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.000505883</td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.0023341  </td><td style="text-align: right;">     2</td><td style="text-align: right;">        25.2836 </td><td style="text-align: right;">    0.556664</td><td style="text-align: right;">  0.517327</td><td style="text-align: right;">         4.76262</td></tr>
<tr><td>tune_train_lstm_b5856c14</td><td>TERMINATED</td><td>172.31.29.69:559  </td><td style="text-align: right;"> 0.349132</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.111193  </td><td style="text-align: right;">               1</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.00137286 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.000233376</td><td style="text-align: right;">    18</td><td style="text-align: right;">       281.703  </td><td style="text-align: right;">    0.295101</td><td style="text-align: right;">  0.307306</td><td style="text-align: right;">         3.46474</td></tr>
<tr><td>tune_train_lstm_61d00e4a</td><td>TERMINATED</td><td>172.31.29.69:7807 </td><td style="text-align: right;"> 0.38493 </td><td style="text-align: right;">          96</td><td style="text-align: right;">0.275735  </td><td style="text-align: right;">               0</td><td style="text-align: right;">         384</td><td style="text-align: right;">0.0203063  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   2.23901e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        15.8174 </td><td style="text-align: right;">    1.68865 </td><td style="text-align: right;">  2.08374 </td><td style="text-align: right;">        12.946  </td></tr>
<tr><td>tune_train_lstm_b027c3ee</td><td>TERMINATED</td><td>172.31.29.69:8092 </td><td style="text-align: right;"> 0.358634</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.28216   </td><td style="text-align: right;">               1</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00170059 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   1.11568e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">        19.3469 </td><td style="text-align: right;">    0.559079</td><td style="text-align: right;">  0.536742</td><td style="text-align: right;">         4.80188</td></tr>
<tr><td>tune_train_lstm_4e8d7d88</td><td>TERMINATED</td><td>172.31.29.69:22803</td><td style="text-align: right;"> 0.519608</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.286215  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00329191 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.000629851</td><td style="text-align: right;">     6</td><td style="text-align: right;">        62.7452 </td><td style="text-align: right;">    0.387055</td><td style="text-align: right;">  0.399893</td><td style="text-align: right;">         3.98873</td></tr>
<tr><td>tune_train_lstm_34d66f7f</td><td>TERMINATED</td><td>172.31.29.69:7392 </td><td style="text-align: right;"> 0.422826</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.258709  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.00554527 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.00109675 </td><td style="text-align: right;">    54</td><td style="text-align: right;">       956.158  </td><td style="text-align: right;">    0.220576</td><td style="text-align: right;">  0.25192 </td><td style="text-align: right;">         3.01619</td></tr>
<tr><td>tune_train_lstm_6c869795</td><td>TERMINATED</td><td>172.31.29.69:23620</td><td style="text-align: right;"> 0.487627</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0386245 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         384</td><td style="text-align: right;">0.000442747</td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   1.34138e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        88.0056 </td><td style="text-align: right;">    0.380873</td><td style="text-align: right;">  0.387827</td><td style="text-align: right;">         3.98453</td></tr>
<tr><td>tune_train_lstm_06824e8a</td><td>TERMINATED</td><td>172.31.29.69:24062</td><td style="text-align: right;"> 0.38624 </td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0966237 </td><td style="text-align: right;">               5</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00447511 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.00269298 </td><td style="text-align: right;">     6</td><td style="text-align: right;">        85.1602 </td><td style="text-align: right;">    0.346802</td><td style="text-align: right;">  0.34676 </td><td style="text-align: right;">         3.73779</td></tr>
<tr><td>tune_train_lstm_4d4c7ad8</td><td>TERMINATED</td><td>172.31.29.69:9347 </td><td style="text-align: right;"> 0.286085</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.00547274</td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0216707  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000363095</td><td style="text-align: right;">     2</td><td style="text-align: right;">         9.14901</td><td style="text-align: right;">    0.609211</td><td style="text-align: right;">  0.616838</td><td style="text-align: right;">         5.4882 </td></tr>
<tr><td>tune_train_lstm_81ab5ebc</td><td>TERMINATED</td><td>172.31.29.69:9785 </td><td style="text-align: right;"> 0.430332</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0707102 </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.000353784</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   0.00123132 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        28.1801 </td><td style="text-align: right;">    0.572811</td><td style="text-align: right;">  0.539232</td><td style="text-align: right;">         4.8404 </td></tr>
<tr><td>tune_train_lstm_05cdb66b</td><td>TERMINATED</td><td>172.31.29.69:1641 </td><td style="text-align: right;"> 0.352932</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.10828   </td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00490334 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.000851773</td><td style="text-align: right;">    18</td><td style="text-align: right;">       285.468  </td><td style="text-align: right;">    0.303942</td><td style="text-align: right;">  0.309185</td><td style="text-align: right;">         3.43557</td></tr>
<tr><td>tune_train_lstm_1aa61b4b</td><td>TERMINATED</td><td>172.31.29.69:12109</td><td style="text-align: right;"> 0.518759</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.00275754</td><td style="text-align: right;">               5</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.000501552</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.10784e-06</td><td style="text-align: right;">   162</td><td style="text-align: right;">      2509.09   </td><td style="text-align: right;">    0.182767</td><td style="text-align: right;">  0.237634</td><td style="text-align: right;">         2.87889</td></tr>
<tr><td>tune_train_lstm_c0e49171</td><td>TERMINATED</td><td>172.31.29.69:3248 </td><td style="text-align: right;"> 0.535686</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.250214  </td><td style="text-align: right;">               5</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00272109 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000702979</td><td style="text-align: right;">    18</td><td style="text-align: right;">       215.47   </td><td style="text-align: right;">    0.301044</td><td style="text-align: right;">  0.320676</td><td style="text-align: right;">         3.59694</td></tr>
<tr><td>tune_train_lstm_9e2feb0b</td><td>TERMINATED</td><td>172.31.29.69:25403</td><td style="text-align: right;"> 0.539286</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.13332   </td><td style="text-align: right;">               0</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.000794546</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.14523e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        80.9872 </td><td style="text-align: right;">    0.37578 </td><td style="text-align: right;">  0.359586</td><td style="text-align: right;">         3.82286</td></tr>
<tr><td>tune_train_lstm_afc2b740</td><td>TERMINATED</td><td>172.31.29.69:25655</td><td style="text-align: right;"> 0.337785</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.182482  </td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.00121033 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.000146049</td><td style="text-align: right;">     6</td><td style="text-align: right;">        80.4439 </td><td style="text-align: right;">    0.365959</td><td style="text-align: right;">  0.360137</td><td style="text-align: right;">         3.81688</td></tr>
<tr><td>tune_train_lstm_bbc15c7c</td><td>TERMINATED</td><td>172.31.29.69:4048 </td><td style="text-align: right;"> 0.376448</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0733273 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.00129906 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.00235672 </td><td style="text-align: right;">    18</td><td style="text-align: right;">       317.952  </td><td style="text-align: right;">    0.282191</td><td style="text-align: right;">  0.298975</td><td style="text-align: right;">         3.41062</td></tr>
<tr><td>tune_train_lstm_5473fa52</td><td>TERMINATED</td><td>172.31.29.69:26318</td><td style="text-align: right;"> 0.377136</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0967062 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.00931259 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   3.062e-06  </td><td style="text-align: right;">     6</td><td style="text-align: right;">        51.4666 </td><td style="text-align: right;">    0.371635</td><td style="text-align: right;">  0.371689</td><td style="text-align: right;">         3.88283</td></tr>
<tr><td>tune_train_lstm_3c0d101a</td><td>TERMINATED</td><td>172.31.29.69:26693</td><td style="text-align: right;"> 0.435292</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.122647  </td><td style="text-align: right;">               1</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.00982819 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   3.60819e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        42.4734 </td><td style="text-align: right;">    0.386341</td><td style="text-align: right;">  0.360917</td><td style="text-align: right;">         3.85497</td></tr>
<tr><td>tune_train_lstm_4e1831a9</td><td>TERMINATED</td><td>172.31.29.69:27025</td><td style="text-align: right;"> 0.485101</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.18723   </td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.00540179 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.2131e-05 </td><td style="text-align: right;">     6</td><td style="text-align: right;">        84.0688 </td><td style="text-align: right;">    0.332105</td><td style="text-align: right;">  0.355275</td><td style="text-align: right;">         3.857  </td></tr>
<tr><td>tune_train_lstm_63270b5f</td><td>TERMINATED</td><td>172.31.29.69:12630</td><td style="text-align: right;"> 0.348805</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.247513  </td><td style="text-align: right;">               5</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00285615 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.000219803</td><td style="text-align: right;">     2</td><td style="text-align: right;">        20.7696 </td><td style="text-align: right;">    0.549537</td><td style="text-align: right;">  0.532717</td><td style="text-align: right;">         4.77602</td></tr>
<tr><td>tune_train_lstm_c3198343</td><td>TERMINATED</td><td>172.31.29.69:27270</td><td style="text-align: right;"> 0.442594</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.197316  </td><td style="text-align: right;">               0</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.0100698  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.000982846</td><td style="text-align: right;">     6</td><td style="text-align: right;">        62.7884 </td><td style="text-align: right;">    0.419972</td><td style="text-align: right;">  0.451925</td><td style="text-align: right;">         4.28107</td></tr>
<tr><td>tune_train_lstm_a048583e</td><td>TERMINATED</td><td>172.31.29.69:4369 </td><td style="text-align: right;"> 0.493789</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.227441  </td><td style="text-align: right;">               0</td><td style="text-align: right;">         192</td><td style="text-align: right;">0.000932895</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   5.05941e-05</td><td style="text-align: right;">    18</td><td style="text-align: right;">       313.034  </td><td style="text-align: right;">    0.27996 </td><td style="text-align: right;">  0.287834</td><td style="text-align: right;">         3.36902</td></tr>
<tr><td>tune_train_lstm_0d0935e0</td><td>TERMINATED</td><td>172.31.29.69:27922</td><td style="text-align: right;"> 0.466861</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0687754 </td><td style="text-align: right;">               5</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.000854719</td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.00174216 </td><td style="text-align: right;">     6</td><td style="text-align: right;">        76.706  </td><td style="text-align: right;">    0.361139</td><td style="text-align: right;">  0.356383</td><td style="text-align: right;">         3.76263</td></tr>
<tr><td>tune_train_lstm_aa5e1349</td><td>TERMINATED</td><td>172.31.29.69:13778</td><td style="text-align: right;"> 0.253389</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0339714 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         384</td><td style="text-align: right;">0.0121012  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.00156163 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        27.9531 </td><td style="text-align: right;">    0.8976  </td><td style="text-align: right;">  0.953723</td><td style="text-align: right;">         7.27413</td></tr>
<tr><td>tune_train_lstm_838d07a6</td><td>TERMINATED</td><td>172.31.29.69:28353</td><td style="text-align: right;"> 0.343166</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.292823  </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0054403  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   0.0013018  </td><td style="text-align: right;">     6</td><td style="text-align: right;">        90.7446 </td><td style="text-align: right;">    0.403268</td><td style="text-align: right;">  0.376654</td><td style="text-align: right;">         3.99192</td></tr>
<tr><td>tune_train_lstm_42564177</td><td>TERMINATED</td><td>172.31.29.69:14361</td><td style="text-align: right;"> 0.479949</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0664917 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.0102314  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.00172734 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        20.1869 </td><td style="text-align: right;">    0.518898</td><td style="text-align: right;">  0.543623</td><td style="text-align: right;">         4.81353</td></tr>
<tr><td>tune_train_lstm_95c8bdd0</td><td>TERMINATED</td><td>172.31.29.69:12201</td><td style="text-align: right;"> 0.479843</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0321616 </td><td style="text-align: right;">               0</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00208565 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.00246027 </td><td style="text-align: right;">   162</td><td style="text-align: right;">      2177.67   </td><td style="text-align: right;">    0.208534</td><td style="text-align: right;">  0.243467</td><td style="text-align: right;">         2.93628</td></tr>
<tr><td>tune_train_lstm_c5986d86</td><td>TERMINATED</td><td>172.31.29.69:10065</td><td style="text-align: right;"> 0.413966</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0762756 </td><td style="text-align: right;">               5</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.000546781</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   5.61732e-05</td><td style="text-align: right;">    54</td><td style="text-align: right;">       895.31   </td><td style="text-align: right;">    0.222007</td><td style="text-align: right;">  0.25536 </td><td style="text-align: right;">         3.05777</td></tr>
<tr><td>tune_train_lstm_cc83b75b</td><td>TERMINATED</td><td>172.31.29.69:15249</td><td style="text-align: right;"> 0.402171</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.154552  </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0119312  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   6.65838e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        10.3161 </td><td style="text-align: right;">    0.544115</td><td style="text-align: right;">  0.509512</td><td style="text-align: right;">         4.67994</td></tr>
<tr><td>tune_train_lstm_a88f95e6</td><td>TERMINATED</td><td>172.31.29.69:15538</td><td style="text-align: right;"> 0.441769</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.198901  </td><td style="text-align: right;">               0</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.00873329 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   4.4699e-06 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        20.054  </td><td style="text-align: right;">    0.504103</td><td style="text-align: right;">  0.501358</td><td style="text-align: right;">         4.67422</td></tr>
<tr><td>tune_train_lstm_af8797de</td><td>TERMINATED</td><td>172.31.29.69:15628</td><td style="text-align: right;"> 0.38788 </td><td style="text-align: right;">          32</td><td style="text-align: right;">0.155805  </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00120794 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   1.43219e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        22.8132 </td><td style="text-align: right;">    0.533562</td><td style="text-align: right;">  0.50517 </td><td style="text-align: right;">         4.62427</td></tr>
<tr><td>tune_train_lstm_e8e80fd4</td><td>TERMINATED</td><td>172.31.29.69:29269</td><td style="text-align: right;"> 0.38201 </td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0463214 </td><td style="text-align: right;">               5</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00187991 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   0.00178347 </td><td style="text-align: right;">     6</td><td style="text-align: right;">        95.4918 </td><td style="text-align: right;">    0.352037</td><td style="text-align: right;">  0.343631</td><td style="text-align: right;">         3.7712 </td></tr>
<tr><td>tune_train_lstm_42dd86a1</td><td>TERMINATED</td><td>172.31.29.69:29743</td><td style="text-align: right;"> 0.413693</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.22863   </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00367986 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.00107841 </td><td style="text-align: right;">     6</td><td style="text-align: right;">        84.3745 </td><td style="text-align: right;">    0.36652 </td><td style="text-align: right;">  0.355504</td><td style="text-align: right;">         3.81976</td></tr>
<tr><td>tune_train_lstm_928a18cd</td><td>TERMINATED</td><td>172.31.29.69:16678</td><td style="text-align: right;"> 0.263675</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0468458 </td><td style="text-align: right;">               5</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.00606197 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   2.58866e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        19.4786 </td><td style="text-align: right;">    0.566372</td><td style="text-align: right;">  0.491507</td><td style="text-align: right;">         4.6156 </td></tr>
<tr><td>tune_train_lstm_09cd55b5</td><td>TERMINATED</td><td>172.31.29.69:29835</td><td style="text-align: right;"> 0.485305</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.132633  </td><td style="text-align: right;">               1</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00111198 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.000908695</td><td style="text-align: right;">     6</td><td style="text-align: right;">        85.2849 </td><td style="text-align: right;">    0.372919</td><td style="text-align: right;">  0.354012</td><td style="text-align: right;">         3.77751</td></tr>
<tr><td>tune_train_lstm_827bb438</td><td>TERMINATED</td><td>172.31.29.69:30421</td><td style="text-align: right;"> 0.496845</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.247532  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.0108802  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   7.64413e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        61.0811 </td><td style="text-align: right;">    0.400182</td><td style="text-align: right;">  0.385843</td><td style="text-align: right;">         3.93141</td></tr>
<tr><td>tune_train_lstm_117f82f9</td><td>TERMINATED</td><td>172.31.29.69:6201 </td><td style="text-align: right;"> 0.444701</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0406144 </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00137043 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.00215795 </td><td style="text-align: right;">    18</td><td style="text-align: right;">       385.333  </td><td style="text-align: right;">    0.294478</td><td style="text-align: right;">  0.303298</td><td style="text-align: right;">         3.4072 </td></tr>
<tr><td>tune_train_lstm_b0499d23</td><td>TERMINATED</td><td>172.31.29.69:17778</td><td style="text-align: right;"> 0.544541</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0201526 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.00641497 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   1.94044e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">        15.3439 </td><td style="text-align: right;">    0.540609</td><td style="text-align: right;">  0.753184</td><td style="text-align: right;">         6.24865</td></tr>
<tr><td>tune_train_lstm_61380576</td><td>TERMINATED</td><td>172.31.29.69:31048</td><td style="text-align: right;"> 0.467373</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.101897  </td><td style="text-align: right;">               5</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00120471 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000278907</td><td style="text-align: right;">     6</td><td style="text-align: right;">        69.5712 </td><td style="text-align: right;">    0.398713</td><td style="text-align: right;">  0.391249</td><td style="text-align: right;">         4.00785</td></tr>
<tr><td>tune_train_lstm_1d0b7f30</td><td>TERMINATED</td><td>172.31.29.69:31307</td><td style="text-align: right;"> 0.503138</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.264164  </td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00048594 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   3.70364e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        74.7908 </td><td style="text-align: right;">    0.402097</td><td style="text-align: right;">  0.385908</td><td style="text-align: right;">         3.97403</td></tr>
<tr><td>tune_train_lstm_77084bcb</td><td>TERMINATED</td><td>172.31.29.69:31688</td><td style="text-align: right;"> 0.53104 </td><td style="text-align: right;">          32</td><td style="text-align: right;">0.22283   </td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00215463 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000307345</td><td style="text-align: right;">     6</td><td style="text-align: right;">        68.7743 </td><td style="text-align: right;">    0.347394</td><td style="text-align: right;">  0.345554</td><td style="text-align: right;">         3.74714</td></tr>
<tr><td>tune_train_lstm_97b73335</td><td>TERMINATED</td><td>172.31.29.69:18763</td><td style="text-align: right;"> 0.481391</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.116021  </td><td style="text-align: right;">               1</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00101338 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   6.8255e-05 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        19.3231 </td><td style="text-align: right;">    0.550283</td><td style="text-align: right;">  0.52276 </td><td style="text-align: right;">         4.7375 </td></tr>
<tr><td>tune_train_lstm_1393dcef</td><td>TERMINATED</td><td>172.31.29.69:19237</td><td style="text-align: right;"> 0.324485</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.0809734 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00151426 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   0.000260342</td><td style="text-align: right;">     2</td><td style="text-align: right;">         9.73131</td><td style="text-align: right;">    0.58635 </td><td style="text-align: right;">  0.537364</td><td style="text-align: right;">         4.81806</td></tr>
<tr><td>tune_train_lstm_e4d18c69</td><td>TERMINATED</td><td>172.31.29.69:19327</td><td style="text-align: right;"> 0.443841</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.0487451 </td><td style="text-align: right;">               3</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.000515221</td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   1.08729e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        21.4696 </td><td style="text-align: right;">    0.558789</td><td style="text-align: right;">  0.548636</td><td style="text-align: right;">         4.95483</td></tr>
<tr><td>tune_train_lstm_5c1feb95</td><td>TERMINATED</td><td>172.31.29.69:31981</td><td style="text-align: right;"> 0.52106 </td><td style="text-align: right;">          64</td><td style="text-align: right;">0.197635  </td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.0026518  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           2</td><td style="text-align: right;">   0.000864173</td><td style="text-align: right;">     6</td><td style="text-align: right;">        37.4869 </td><td style="text-align: right;">    0.372817</td><td style="text-align: right;">  0.37261 </td><td style="text-align: right;">         3.92667</td></tr>
<tr><td>tune_train_lstm_2a74b6c4</td><td>TERMINATED</td><td>172.31.29.69:20053</td><td style="text-align: right;"> 0.280503</td><td style="text-align: right;">          96</td><td style="text-align: right;">0.16152   </td><td style="text-align: right;">               5</td><td style="text-align: right;">         128</td><td style="text-align: right;">0.00342764 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           1</td><td style="text-align: right;">   0.000104559</td><td style="text-align: right;">     2</td><td style="text-align: right;">         9.19172</td><td style="text-align: right;">    0.597577</td><td style="text-align: right;">  0.582156</td><td style="text-align: right;">         5.03395</td></tr>
<tr><td>tune_train_lstm_f73571f6</td><td>TERMINATED</td><td>172.31.29.69:20144</td><td style="text-align: right;"> 0.382882</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.0831797 </td><td style="text-align: right;">               5</td><td style="text-align: right;">         384</td><td style="text-align: right;">0.0276555  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.00018151 </td><td style="text-align: right;">     2</td><td style="text-align: right;">        18.2214 </td><td style="text-align: right;">    2.10313 </td><td style="text-align: right;">  1.43233 </td><td style="text-align: right;">         9.89226</td></tr>
<tr><td>tune_train_lstm_e7b7af02</td><td>TERMINATED</td><td>172.31.29.69:32314</td><td style="text-align: right;"> 0.329734</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.191723  </td><td style="text-align: right;">               0</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.00938984 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   0.00133581 </td><td style="text-align: right;">     6</td><td style="text-align: right;">        83.5737 </td><td style="text-align: right;">    0.381911</td><td style="text-align: right;">  0.354989</td><td style="text-align: right;">         3.80073</td></tr>
<tr><td>tune_train_lstm_c7127580</td><td>TERMINATED</td><td>172.31.29.69:32561</td><td style="text-align: right;"> 0.460725</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.217863  </td><td style="text-align: right;">               1</td><td style="text-align: right;">         256</td><td style="text-align: right;">0.00812939 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   1.22377e-05</td><td style="text-align: right;">    18</td><td style="text-align: right;">       252.71   </td><td style="text-align: right;">    0.297912</td><td style="text-align: right;">  0.290699</td><td style="text-align: right;">         3.33239</td></tr>
<tr><td>tune_train_lstm_4afb9bca</td><td>TERMINATED</td><td>172.31.29.69:21162</td><td style="text-align: right;"> 0.27717 </td><td style="text-align: right;">          96</td><td style="text-align: right;">0.179182  </td><td style="text-align: right;">               0</td><td style="text-align: right;">         512</td><td style="text-align: right;">0.0184912  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   0.000994769</td><td style="text-align: right;">     2</td><td style="text-align: right;">        22.4196 </td><td style="text-align: right;">    1.40374 </td><td style="text-align: right;">  1.19323 </td><td style="text-align: right;">         8.91229</td></tr>
<tr><td>tune_train_lstm_20288842</td><td>TERMINATED</td><td>172.31.29.69:21422</td><td style="text-align: right;"> 0.342959</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.255362  </td><td style="text-align: right;">               5</td><td style="text-align: right;">         384</td><td style="text-align: right;">0.0163472  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   0.000141575</td><td style="text-align: right;">     2</td><td style="text-align: right;">        24.1275 </td><td style="text-align: right;">    1.10965 </td><td style="text-align: right;">  1.06697 </td><td style="text-align: right;">         8.37242</td></tr>
<tr><td>tune_train_lstm_d2cc2ac3</td><td>TERMINATED</td><td>172.31.29.69:21710</td><td style="text-align: right;"> 0.456318</td><td style="text-align: right;">          32</td><td style="text-align: right;">0.255252  </td><td style="text-align: right;">               1</td><td style="text-align: right;">          64</td><td style="text-align: right;">0.000608117</td><td style="text-align: right;">         200</td><td style="text-align: right;">           3</td><td style="text-align: right;">   1.75477e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">        23.4564 </td><td style="text-align: right;">    0.557201</td><td style="text-align: right;">  0.54054 </td><td style="text-align: right;">         4.83196</td></tr>
</tbody>
</table>
  </div>
</div>
<style>
.tuneStatus {
  color: var(--jp-ui-font-color1);
}
.tuneStatus .systemInfo {
  display: flex;
  flex-direction: column;
}
.tuneStatus td {
  white-space: nowrap;
}
.tuneStatus .trialStatus {
  display: flex;
  flex-direction: column;
}
.tuneStatus h3 {
  font-weight: bold;
}
.tuneStatus .hDivider {
  border-bottom-width: var(--jp-border-width);
  border-bottom-color: var(--jp-border-color0);
  border-bottom-style: solid;
}
.tuneStatus .vDivider {
  border-left-width: var(--jp-border-width);
  border-left-color: var(--jp-border-color0);
  border-left-style: solid;
  margin: 0.5em 1em 0.5em 1em;
}
</style>



    /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/linux_venv/lib64/python3.12/site-packages/statsmodels/nonparametric/kernels.py:62: RuntimeWarning: divide by zero encountered in divide
      kernel_value = np.ones(Xi.size) * h / (num_levels - 1)


    [LSTM] saved clean final model .pth at /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Models/LSTM_multihorizon_raytuned_model_9.pth


    /tmp/ipykernel_1683/3848344726.py:127: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      payload = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=DEVICE)



```python
model_lstm = build_model_lstm(cfg_lstm)
model_lstm.load_state_dict(payload["model_state_dict"])

# exports a clean .pth for downstream tasks
save_final_model_pth(model_lstm, best_lstm_res, out_path=MODEL_PATH_LSTM)
print(f"[LSTM] saved clean final model .pth at {MODEL_PATH_LSTM}")
```

    [LSTM] saved clean final model .pth at /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Models/LSTM_multihorizon_raytuned_model_9.pth


### 4.5 Training and Tuning the TCN Model
We repeat the same process for the TCN. The search explores convolutional kernel size, dilation depth, and network width, focusing on receptive field properties.


```python
from mt4xai.model import build_model_tcn
from mt4xai.train import tune_train_tcn

results_tcn = None
best_tcn_res = None
cfg_tcn = None
model_tcn = None

# restart Ray Tune env if running
ray.shutdown()  
ray.init(  
    num_cpus=20,
    logging_level=logging.ERROR, 
    log_to_driver=False,  
    ignore_reinit_error=True, 
    _memory=8 * 1024**3,  # limit Ray’s worker heap memory to 8 GB
    object_store_memory=1 * 1024**3,  # 1 GB for the object/plasma store
    _temp_dir="/tmp/ray",   # optional: faster NVMe scratch
    include_dashboard=False,
)

# common Tune objects
cs_tcn = make_bohb_cs_for_tcn(seed=RANDOM_SEED)
bohb_tcn = TuneBOHB(space=cs_tcn, metric="val_metric", mode="min")
hb_tcn = HyperBandForBOHB(time_attr="epoch", max_t=200, reduction_factor=3)

train_ref = ray.put(train_dataset)
val_ref = ray.put(val_dataset)
trainable_tcn = tune.with_parameters(
    tune_train_tcn, 
    train_dataset_ref=train_ref, 
    val_dataset_ref=val_ref, 
    num_workers=NUM_WORKERS, 
    power_min=POWER_MIN, 
    power_max=POWER_MAX, 
    idx_power=IDX_POWER)
trainable_tcn = tune.with_resources(trainable_tcn, {"cpu": 10, "gpu": 0.5})  # per trial

ckpt_cfg_tcn = CheckpointConfig(
    num_to_keep=3,
    checkpoint_score_attribute="val_metric",
    checkpoint_score_order="min",
)
run_cfg_tcn = RunConfig(
    storage_path=MODEL_FOLDER_PATH,
    name=RAY_TUNE_FOLDER_NAME_TCN,
    checkpoint_config=ckpt_cfg_tcn,
    verbose=1,
)

run_root_tcn = os.path.join(MODEL_FOLDER_PATH, RAY_TUNE_FOLDER_NAME_TCN)
status_tcn = tune_run_status(run_root_tcn)

if TRAIN_TCN:
    if RESUME_TRAIN_IF_CKPT and status_tcn["exists"] and not status_tcn["finished"]:
        # unfinished run? try minimal restore
        try:
            tuner_tcn = Tuner.restore(
                run_root_tcn,
                trainable=trainable_tcn,
                resume_unfinished=True,
                resume_errored=True,
                param_space=BASE_CFG,
            )
            print("[tcn] resuming unfinished run (using original searcher/scheduler from disk).")
            results_tcn = tuner_tcn.fit()
        except KeyError as e:
            # BOHB resume glitch. trial_to_params missing for a restored trial
            print(f"[tcn][warn] BOHB resume hit KeyError ({e}). "
                  "Your Ray build cannot swap the searcher during restore. "
                  "Options: (a) start a fresh run under a NEW name, or "
                  "(b) load the best trial checkpoint and continue training outside Tune.")
            # pick one path automatically: start a FRESH run under a new folder name
            fresh_name_tcn = f"{RAY_TUNE_FOLDER_NAME_TCN}_fresh"
            run_cfg_tcn_fresh = RunConfig(
                storage_path=MODEL_FOLDER_PATH,
                name=fresh_name_tcn,
                checkpoint_config=ckpt_cfg_tcn,
                verbose=1,
            )
            tuner_tcn = Tuner(
                trainable_tcn,
                tune_config=TuneConfig(
                    search_alg=bohb_tcn,
                    scheduler=hb_tcn,
                    num_samples=64,
                    max_concurrent_trials=2,
                    metric="val_metric", mode="min",
                ),
                run_config=run_cfg_tcn_fresh,
                param_space=BASE_CFG,
            )
            print(f"[tcn] launching a fresh run in '{fresh_name_tcn}' …")
            results_tcn = tuner_tcn.fit()
        except Exception as e:
            # any other restore failure? also start a fresh run
            print(f"[tcn][warn] restore failed ({type(e).__name__}: {e}). Starting a fresh run.")
            tuner_tcn = Tuner(
                trainable_tcn,
                tune_config=TuneConfig(
                    search_alg=bohb_tcn,
                    scheduler=hb_tcn,
                    num_samples=64,
                    max_concurrent_trials=2,
                    metric="val_metric", mode="min",
                ),
                run_config=run_cfg_tcn,
                param_space=BASE_CFG,
            )
            results_tcn = tuner_tcn.fit()
    elif RESUME_TRAIN_IF_CKPT and status_tcn["exists"] and status_tcn["finished"]:
        # run already finished, do not resume. simply read results
        print(f"[tcn] previous run is finished (statuses={status_tcn['statuses']}); loading results only.")
        results_tcn = restore_resultgrid(RAY_TUNE_RUN_FOLDER_PATH_TCN)
    else:
        # fresh run
        tuner_tcn = Tuner(
            trainable_tcn,
            tune_config=TuneConfig(
                search_alg=bohb_tcn,
                scheduler=hb_tcn,
                num_samples=64,
                max_concurrent_trials=2,
                metric="val_metric", mode="min",
            ),
            run_config=run_cfg_tcn,
            param_space=BASE_CFG,  # injects the static keys (device/input/targets/horizon)
        )
        results_tcn = tuner_tcn.fit()

    # rebuild best & export clean .pth
    if results_tcn is None:
        raise RuntimeError("No ResultGrid available after run/resume.")

    best_tcn_res = results_tcn.get_best_result(metric="val_metric", mode="min")
    cfg_tcn = dict(best_tcn_res.config)

    ckpt_dir = best_tcn_res.checkpoint.to_directory()
    assert ckpt_dir is not None, "no checkpoint directory found for best_tcn"
    payload = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=DEVICE)
    model_tcn = build_model_tcn(cfg_tcn)
    state_dict = payload.get("model_state_dict", payload.get("model", None))
    assert state_dict is not None, "checkpoint payload missing weights"
    model_tcn.load_state_dict(state_dict)
    model_tcn.eval()
    save_final_model_pth(model_tcn, best_tcn_res, out_path=MODEL_PATH_TCN)
    print(f"[TCN] saved clean final model .pth at {MODEL_PATH_TCN}")

else:
    # restore most recent completed run
    results_tcn = restore_resultgrid(RAY_TUNE_RUN_FOLDER_PATH_TCN)
    if results_tcn is not None and len(list(results_tcn)) > 0:
        best_tcn_res = results_tcn.get_best_result(metric="val_metric", mode="min")
        cfg_tcn = dict(best_tcn_res.config)
        for k, v in BASE_CFG.items(): cfg_tcn.setdefault(k, v)
        try:
            ckpt_dir = best_tcn_res.checkpoint.to_directory()
            payload  = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=DEVICE)
            model_tcn = build_model_tcn(cfg_tcn)
            sd = payload.get("model_state_dict", payload.get("model", None))
            assert sd is not None
            model_tcn.load_state_dict(sd); model_tcn.eval()
            print("[TCN] restored model from Tune checkpoint.")
        except AttributeError:
            if os.path.exists(MODEL_PATH_TCN):
                final = torch.load(MODEL_PATH_TCN, map_location=DEVICE)
                cfg_tcn = dict(final.get("config", cfg_tcn))
                model_tcn = build_model_tcn(cfg_tcn)
                sd = final.get("model_state_dict", None) or final
                model_tcn.load_state_dict(sd); model_tcn.eval()
                print("[TCN] loaded model from exported .pth.")
            else:
                print("[TCN] no Tune checkpoint or .pth found.")
    # or simply load the saved model from path (without checkpoint/config data)
    elif os.path.exists(MODEL_PATH_TCN):
        final = torch.load(MODEL_PATH_TCN, map_location=DEVICE)
        cfg_tcn = dict(final.get("config", BASE_CFG))
        for k, v in BASE_CFG.items(): cfg_tcn.setdefault(k, v)
        model_tcn = build_model_tcn(cfg_tcn)
        sd = final.get("model_state_dict", None) or final
        model_tcn.load_state_dict(sd); model_tcn.eval()
        print("[TCN] loaded model from exported .pth.")
    else:
        print("[TCN] no previous results or model found.")

```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2025-11-13 03:02:39</td></tr>
<tr><td>Running for: </td><td>02:42:38.97        </td></tr>
<tr><td>Memory:      </td><td>7.2/13.6 GiB       </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      Using HyperBand: num_stopped=62 total_brackets=2<br>Round #0:<br>  Bracket(Max Size (n)=2, Milestone (r)=162, completed=100.0%): {TERMINATED: 64} <br>Logical resource usage: 10.0/20 CPUs, 0.5/1 GPUs (0.0/1.0 accelerator_type:G)
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name             </th><th>status    </th><th>loc               </th><th style="text-align: right;">  alpha_h</th><th style="text-align: right;">  batch_size</th><th style="text-align: right;">  dropout</th><th style="text-align: right;">  grad_clip_norm</th><th style="text-align: right;">  hidden_dim</th><th style="text-align: right;">  kernel_size</th><th style="text-align: right;">         lr</th><th style="text-align: right;">  num_epochs</th><th style="text-align: right;">  num_layers</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  train_loss</th><th style="text-align: right;">  val_loss</th><th style="text-align: right;">  val_rmse_power</th></tr>
</thead>
<tbody>
<tr><td>tune_train_tcn_7bd73371</td><td>TERMINATED</td><td>172.31.29.69:10185</td><td style="text-align: right;"> 0.274908</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.226719</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000606672</td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   1.64093e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        159.979 </td><td style="text-align: right;">    0.425374</td><td style="text-align: right;">  0.41681 </td><td style="text-align: right;">         4.30374</td></tr>
<tr><td>tune_train_tcn_cb54dd27</td><td>TERMINATED</td><td>172.31.29.69:10275</td><td style="text-align: right;"> 0.361679</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.204271</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00065609 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   3.91333e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        124.65  </td><td style="text-align: right;">    0.400788</td><td style="text-align: right;">  0.388606</td><td style="text-align: right;">         4.0426 </td></tr>
<tr><td>tune_train_tcn_b552a41d</td><td>TERMINATED</td><td>172.31.29.69:20163</td><td style="text-align: right;"> 0.258698</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.196222</td><td style="text-align: right;">               1</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00135986 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   6.18767e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         36.1171</td><td style="text-align: right;">    0.53189 </td><td style="text-align: right;">  0.483703</td><td style="text-align: right;">         4.71656</td></tr>
<tr><td>tune_train_tcn_8222b286</td><td>TERMINATED</td><td>172.31.29.69:11050</td><td style="text-align: right;"> 0.320883</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.293137</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00155405 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.17061e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        126.382 </td><td style="text-align: right;">    0.413796</td><td style="text-align: right;">  0.394681</td><td style="text-align: right;">         4.1355 </td></tr>
<tr><td>tune_train_tcn_69bde06b</td><td>TERMINATED</td><td>172.31.29.69:20917</td><td style="text-align: right;"> 0.250783</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.250375</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00120335 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   2.64423e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         20.3896</td><td style="text-align: right;">    0.594471</td><td style="text-align: right;">  0.537119</td><td style="text-align: right;">         4.94728</td></tr>
<tr><td>tune_train_tcn_acadf3ac</td><td>TERMINATED</td><td>172.31.29.69:21008</td><td style="text-align: right;"> 0.279514</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.216112</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00147401 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   8.85784e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         19.3377</td><td style="text-align: right;">    0.573044</td><td style="text-align: right;">  0.510357</td><td style="text-align: right;">         4.82757</td></tr>
<tr><td>tune_train_tcn_7f66fbd1</td><td>TERMINATED</td><td>172.31.29.69:21560</td><td style="text-align: right;"> 0.398891</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.272066</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000655417</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   5.84774e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         18.2345</td><td style="text-align: right;">    0.579792</td><td style="text-align: right;">  0.502281</td><td style="text-align: right;">         4.65818</td></tr>
<tr><td>tune_train_tcn_e06a7d20</td><td>TERMINATED</td><td>172.31.29.69:11445</td><td style="text-align: right;"> 0.314458</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.257952</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00082301 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.44652e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        127.544 </td><td style="text-align: right;">    0.4095  </td><td style="text-align: right;">  0.399392</td><td style="text-align: right;">         4.13381</td></tr>
<tr><td>tune_train_tcn_0a74d483</td><td>TERMINATED</td><td>172.31.29.69:22186</td><td style="text-align: right;"> 0.205523</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.262717</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00102285 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   1.45126e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         21.2666</td><td style="text-align: right;">    0.60455 </td><td style="text-align: right;">  0.550314</td><td style="text-align: right;">         4.98128</td></tr>
<tr><td>tune_train_tcn_0385fab1</td><td>TERMINATED</td><td>172.31.29.69:22562</td><td style="text-align: right;"> 0.28847 </td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.25761 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.000963892</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   7.35991e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         32.1717</td><td style="text-align: right;">    0.547243</td><td style="text-align: right;">  0.498497</td><td style="text-align: right;">         4.67541</td></tr>
<tr><td>tune_train_tcn_9cc779a1</td><td>TERMINATED</td><td>172.31.29.69:11972</td><td style="text-align: right;"> 0.388778</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.244155</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.000778059</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   5.6878e-05 </td><td style="text-align: right;">     6</td><td style="text-align: right;">         76.0944</td><td style="text-align: right;">    0.412524</td><td style="text-align: right;">  0.406726</td><td style="text-align: right;">         4.14252</td></tr>
<tr><td>tune_train_tcn_f45f80b8</td><td>TERMINATED</td><td>172.31.29.69:12368</td><td style="text-align: right;"> 0.350123</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.258661</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000778424</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.66352e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        127.644 </td><td style="text-align: right;">    0.417283</td><td style="text-align: right;">  0.408866</td><td style="text-align: right;">         4.23612</td></tr>
<tr><td>tune_train_tcn_e939d971</td><td>TERMINATED</td><td>172.31.29.69:2022 </td><td style="text-align: right;"> 0.39485 </td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.286611</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00185896 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   6.49694e-05</td><td style="text-align: right;">    54</td><td style="text-align: right;">       1211.58  </td><td style="text-align: right;">    0.294345</td><td style="text-align: right;">  0.289959</td><td style="text-align: right;">         3.35694</td></tr>
<tr><td>tune_train_tcn_457e19a2</td><td>TERMINATED</td><td>172.31.29.69:24020</td><td style="text-align: right;"> 0.211261</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.24688 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00119648 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   8.15293e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         39.2899</td><td style="text-align: right;">    0.557406</td><td style="text-align: right;">  0.534513</td><td style="text-align: right;">         4.92896</td></tr>
<tr><td>tune_train_tcn_f9c6bc14</td><td>TERMINATED</td><td>172.31.29.69:24324</td><td style="text-align: right;"> 0.27067 </td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.237569</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00088424 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   4.83323e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         19.9953</td><td style="text-align: right;">    0.592872</td><td style="text-align: right;">  0.636941</td><td style="text-align: right;">         5.42352</td></tr>
<tr><td>tune_train_tcn_d27865e7</td><td>TERMINATED</td><td>172.31.29.69:24737</td><td style="text-align: right;"> 0.211275</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.285991</td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00102835 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   3.23798e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         31.9203</td><td style="text-align: right;">    0.579086</td><td style="text-align: right;">  0.508279</td><td style="text-align: right;">         4.72975</td></tr>
<tr><td>tune_train_tcn_9e556519</td><td>TERMINATED</td><td>172.31.29.69:25074</td><td style="text-align: right;"> 0.202909</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.243127</td><td style="text-align: right;">               1</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00111675 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   1.07268e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         39.4033</td><td style="text-align: right;">    0.553964</td><td style="text-align: right;">  0.498735</td><td style="text-align: right;">         4.75828</td></tr>
<tr><td>tune_train_tcn_1f753a53</td><td>TERMINATED</td><td>172.31.29.69:25452</td><td style="text-align: right;"> 0.269017</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.250478</td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00180011 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   2.36772e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         19.7356</td><td style="text-align: right;">    0.564465</td><td style="text-align: right;">  0.52403 </td><td style="text-align: right;">         4.89573</td></tr>
<tr><td>tune_train_tcn_00391f34</td><td>TERMINATED</td><td>172.31.29.69:13303</td><td style="text-align: right;"> 0.270829</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.22585 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00174025 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.02171e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        127.044 </td><td style="text-align: right;">    0.41809 </td><td style="text-align: right;">  0.423203</td><td style="text-align: right;">         4.31485</td></tr>
<tr><td>tune_train_tcn_0d05a76a</td><td>TERMINATED</td><td>172.31.29.69:25950</td><td style="text-align: right;"> 0.234675</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.26064 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000659303</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   9.16812e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         21.0454</td><td style="text-align: right;">    0.614931</td><td style="text-align: right;">  0.566074</td><td style="text-align: right;">         5.10928</td></tr>
<tr><td>tune_train_tcn_211b0458</td><td>TERMINATED</td><td>172.31.29.69:26508</td><td style="text-align: right;"> 0.212675</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.246243</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.0014152  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   1.5018e-05 </td><td style="text-align: right;">     2</td><td style="text-align: right;">         35.9098</td><td style="text-align: right;">    0.552174</td><td style="text-align: right;">  0.479278</td><td style="text-align: right;">         4.63788</td></tr>
<tr><td>tune_train_tcn_115d515d</td><td>TERMINATED</td><td>172.31.29.69:13706</td><td style="text-align: right;"> 0.32482 </td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.306906</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00133234 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   4.39051e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        125.939 </td><td style="text-align: right;">    0.396912</td><td style="text-align: right;">  0.383012</td><td style="text-align: right;">         4.04813</td></tr>
<tr><td>tune_train_tcn_e823b416</td><td>TERMINATED</td><td>172.31.29.69:14225</td><td style="text-align: right;"> 0.359459</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.280265</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00182038 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   7.05117e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        142.9   </td><td style="text-align: right;">    0.394882</td><td style="text-align: right;">  0.381183</td><td style="text-align: right;">         4.0243 </td></tr>
<tr><td>tune_train_tcn_9454a66c</td><td>TERMINATED</td><td>172.31.29.69:27575</td><td style="text-align: right;"> 0.378637</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.313934</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000517214</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.16105e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         31.267 </td><td style="text-align: right;">    0.548458</td><td style="text-align: right;">  0.505924</td><td style="text-align: right;">         4.64019</td></tr>
<tr><td>tune_train_tcn_09b3ea80</td><td>TERMINATED</td><td>172.31.29.69:26186</td><td style="text-align: right;"> 0.383319</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.3148  </td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00163062 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.74961e-06</td><td style="text-align: right;">    18</td><td style="text-align: right;">        396.688 </td><td style="text-align: right;">    0.33777 </td><td style="text-align: right;">  0.325814</td><td style="text-align: right;">         3.66415</td></tr>
<tr><td>tune_train_tcn_fabe4781</td><td>TERMINATED</td><td>172.31.29.69:15191</td><td style="text-align: right;"> 0.393421</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.228135</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000560982</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   5.16148e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        138.458 </td><td style="text-align: right;">    0.407203</td><td style="text-align: right;">  0.389286</td><td style="text-align: right;">         4.05748</td></tr>
<tr><td>tune_train_tcn_ddc04f47</td><td>TERMINATED</td><td>172.31.29.69:28747</td><td style="text-align: right;"> 0.222648</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.265105</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.000935906</td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   3.63908e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         38.6406</td><td style="text-align: right;">    0.528958</td><td style="text-align: right;">  0.497249</td><td style="text-align: right;">         4.7344 </td></tr>
<tr><td>tune_train_tcn_36e64d40</td><td>TERMINATED</td><td>172.31.29.69:27489</td><td style="text-align: right;"> 0.375746</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.293852</td><td style="text-align: right;">               1</td><td style="text-align: right;">         160</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00144614 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   3.29192e-06</td><td style="text-align: right;">    18</td><td style="text-align: right;">        404.95  </td><td style="text-align: right;">    0.350311</td><td style="text-align: right;">  0.341808</td><td style="text-align: right;">         3.76075</td></tr>
<tr><td>tune_train_tcn_86f7bfe9</td><td>TERMINATED</td><td>172.31.29.69:16146</td><td style="text-align: right;"> 0.395108</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.189391</td><td style="text-align: right;">               1</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.000871472</td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   1.94088e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">         87.428 </td><td style="text-align: right;">    0.419008</td><td style="text-align: right;">  0.394198</td><td style="text-align: right;">         4.08023</td></tr>
<tr><td>tune_train_tcn_8ff1c526</td><td>TERMINATED</td><td>172.31.29.69:29600</td><td style="text-align: right;"> 0.331785</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.259722</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.000562926</td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   4.82653e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         37.7669</td><td style="text-align: right;">    0.540446</td><td style="text-align: right;">  0.522035</td><td style="text-align: right;">         4.94804</td></tr>
<tr><td>tune_train_tcn_241ac3a8</td><td>TERMINATED</td><td>172.31.29.69:30160</td><td style="text-align: right;"> 0.332216</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.185486</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.000635809</td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   1.92372e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         39.5815</td><td style="text-align: right;">    0.542115</td><td style="text-align: right;">  0.519404</td><td style="text-align: right;">         4.853  </td></tr>
<tr><td>tune_train_tcn_ba21f40f</td><td>TERMINATED</td><td>172.31.29.69:16453</td><td style="text-align: right;"> 0.380157</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.196908</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000576379</td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   8.13692e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        161.056 </td><td style="text-align: right;">    0.416517</td><td style="text-align: right;">  0.415578</td><td style="text-align: right;">         4.24956</td></tr>
<tr><td>tune_train_tcn_f5e831de</td><td>TERMINATED</td><td>172.31.29.69:30920</td><td style="text-align: right;"> 0.399351</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.311279</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00137623 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   4.53562e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         35.5845</td><td style="text-align: right;">    0.508859</td><td style="text-align: right;">  0.501604</td><td style="text-align: right;">         4.76391</td></tr>
<tr><td>tune_train_tcn_0b85c6ee</td><td>TERMINATED</td><td>172.31.29.69:16982</td><td style="text-align: right;"> 0.385497</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.219084</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00158239 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   3.45389e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        126.689 </td><td style="text-align: right;">    0.375787</td><td style="text-align: right;">  0.373868</td><td style="text-align: right;">         4.05058</td></tr>
<tr><td>tune_train_tcn_4afe4be4</td><td>TERMINATED</td><td>172.31.29.69:31668</td><td style="text-align: right;"> 0.32483 </td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.195993</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00103626 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.28338e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         17.5424</td><td style="text-align: right;">    0.55204 </td><td style="text-align: right;">  0.559591</td><td style="text-align: right;">         4.9062 </td></tr>
<tr><td>tune_train_tcn_34125638</td><td>TERMINATED</td><td>172.31.29.69:17484</td><td style="text-align: right;"> 0.227377</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.297335</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00186427 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   4.92965e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        144.894 </td><td style="text-align: right;">    0.432067</td><td style="text-align: right;">  0.421868</td><td style="text-align: right;">         4.297  </td></tr>
<tr><td>tune_train_tcn_4238345c</td><td>TERMINATED</td><td>172.31.29.69:4851 </td><td style="text-align: right;"> 0.391162</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.220073</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00056592 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   6.00299e-06</td><td style="text-align: right;">   162</td><td style="text-align: right;">       4933.27  </td><td style="text-align: right;">    0.25781 </td><td style="text-align: right;">  0.262431</td><td style="text-align: right;">         3.12856</td></tr>
<tr><td>tune_train_tcn_459a70ec</td><td>TERMINATED</td><td>172.31.29.69:307  </td><td style="text-align: right;"> 0.379549</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.206906</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.000528001</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   7.40899e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         31.9862</td><td style="text-align: right;">    0.526362</td><td style="text-align: right;">  0.505429</td><td style="text-align: right;">         4.77438</td></tr>
<tr><td>tune_train_tcn_4de0a18f</td><td>TERMINATED</td><td>172.31.29.69:662  </td><td style="text-align: right;"> 0.268877</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.291918</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00143204 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   2.18693e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         38.5053</td><td style="text-align: right;">    0.521487</td><td style="text-align: right;">  0.495807</td><td style="text-align: right;">         4.71302</td></tr>
<tr><td>tune_train_tcn_445ec0c5</td><td>TERMINATED</td><td>172.31.29.69:28974</td><td style="text-align: right;"> 0.388106</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.300876</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00173844 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   6.57841e-05</td><td style="text-align: right;">    18</td><td style="text-align: right;">        406.425 </td><td style="text-align: right;">    0.342496</td><td style="text-align: right;">  0.329441</td><td style="text-align: right;">         3.66528</td></tr>
<tr><td>tune_train_tcn_16eab31a</td><td>TERMINATED</td><td>172.31.29.69:1416 </td><td style="text-align: right;"> 0.294642</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.24639 </td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.0015241  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   8.39239e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         18.2622</td><td style="text-align: right;">    0.544771</td><td style="text-align: right;">  0.508033</td><td style="text-align: right;">         4.76759</td></tr>
<tr><td>tune_train_tcn_1c11e536</td><td>TERMINATED</td><td>172.31.29.69:19258</td><td style="text-align: right;"> 0.386518</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.302315</td><td style="text-align: right;">               1</td><td style="text-align: right;">         128</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00196625 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   1.50352e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">         75.3622</td><td style="text-align: right;">    0.432176</td><td style="text-align: right;">  0.413527</td><td style="text-align: right;">         4.23964</td></tr>
<tr><td>tune_train_tcn_4d1ab968</td><td>TERMINATED</td><td>172.31.29.69:1947 </td><td style="text-align: right;"> 0.395385</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.205309</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00118608 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.44547e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         31.8227</td><td style="text-align: right;">    0.48188 </td><td style="text-align: right;">  0.500629</td><td style="text-align: right;">         4.6972 </td></tr>
<tr><td>tune_train_tcn_b043a07e</td><td>TERMINATED</td><td>172.31.29.69:4942 </td><td style="text-align: right;"> 0.374653</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.248138</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00163461 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   8.32993e-06</td><td style="text-align: right;">    54</td><td style="text-align: right;">       1405.16  </td><td style="text-align: right;">    0.294566</td><td style="text-align: right;">  0.292981</td><td style="text-align: right;">         3.38067</td></tr>
<tr><td>tune_train_tcn_f371d9fb</td><td>TERMINATED</td><td>172.31.29.69:20049</td><td style="text-align: right;"> 0.372531</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.286881</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00195349 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   5.81093e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        125.245 </td><td style="text-align: right;">    0.395558</td><td style="text-align: right;">  0.383697</td><td style="text-align: right;">         4.04774</td></tr>
<tr><td>tune_train_tcn_b41382df</td><td>TERMINATED</td><td>172.31.29.69:30460</td><td style="text-align: right;"> 0.397837</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.212294</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00143046 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   5.25114e-05</td><td style="text-align: right;">    18</td><td style="text-align: right;">        211.215 </td><td style="text-align: right;">    0.335907</td><td style="text-align: right;">  0.339591</td><td style="text-align: right;">         3.71513</td></tr>
<tr><td>tune_train_tcn_9cba0e7f</td><td>TERMINATED</td><td>172.31.29.69:3594 </td><td style="text-align: right;"> 0.208875</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.232176</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00106257 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   4.72092e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         32.0496</td><td style="text-align: right;">    0.562019</td><td style="text-align: right;">  0.500581</td><td style="text-align: right;">         4.68148</td></tr>
<tr><td>tune_train_tcn_eab99680</td><td>TERMINATED</td><td>172.31.29.69:3954 </td><td style="text-align: right;"> 0.231721</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.307218</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000512938</td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   2.15157e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         18.7745</td><td style="text-align: right;">    0.643154</td><td style="text-align: right;">  0.591071</td><td style="text-align: right;">         5.16151</td></tr>
<tr><td>tune_train_tcn_397d7aab</td><td>TERMINATED</td><td>172.31.29.69:20771</td><td style="text-align: right;"> 0.362964</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.249818</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00142478 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   4.31035e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        161.121 </td><td style="text-align: right;">    0.391066</td><td style="text-align: right;">  0.403495</td><td style="text-align: right;">         4.18103</td></tr>
<tr><td>tune_train_tcn_e458f0fd</td><td>TERMINATED</td><td>172.31.29.69:31111</td><td style="text-align: right;"> 0.375572</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.240132</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00162025 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   2.62904e-06</td><td style="text-align: right;">    18</td><td style="text-align: right;">        519.529 </td><td style="text-align: right;">    0.348377</td><td style="text-align: right;">  0.340882</td><td style="text-align: right;">         3.75196</td></tr>
<tr><td>tune_train_tcn_309f4ed2</td><td>TERMINATED</td><td>172.31.29.69:5126 </td><td style="text-align: right;"> 0.378875</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.229176</td><td style="text-align: right;">               3</td><td style="text-align: right;">         128</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00157169 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   4.14928e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         38.7169</td><td style="text-align: right;">    0.524094</td><td style="text-align: right;">  0.498495</td><td style="text-align: right;">         4.78125</td></tr>
<tr><td>tune_train_tcn_38e32b6c</td><td>TERMINATED</td><td>172.31.29.69:21948</td><td style="text-align: right;"> 0.379886</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.280275</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00182463 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   1.38965e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        144.962 </td><td style="text-align: right;">    0.395734</td><td style="text-align: right;">  0.387936</td><td style="text-align: right;">         4.126  </td></tr>
<tr><td>tune_train_tcn_c1c44748</td><td>TERMINATED</td><td>172.31.29.69:31538</td><td style="text-align: right;"> 0.387722</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.21399 </td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.0019493  </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   2.562e-06  </td><td style="text-align: right;">    18</td><td style="text-align: right;">        526.082 </td><td style="text-align: right;">    0.332512</td><td style="text-align: right;">  0.325008</td><td style="text-align: right;">         3.63311</td></tr>
<tr><td>tune_train_tcn_ccb3376a</td><td>TERMINATED</td><td>172.31.29.69:5917 </td><td style="text-align: right;"> 0.214939</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.183442</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.001308   </td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   7.71315e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         39.2774</td><td style="text-align: right;">    0.551873</td><td style="text-align: right;">  0.496892</td><td style="text-align: right;">         4.74144</td></tr>
<tr><td>tune_train_tcn_1259d063</td><td>TERMINATED</td><td>172.31.29.69:6527 </td><td style="text-align: right;"> 0.396796</td><td style="text-align: right;">          64</td><td style="text-align: right;"> 0.258785</td><td style="text-align: right;">               1</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00097471 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   4.48591e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         20.3817</td><td style="text-align: right;">    0.563292</td><td style="text-align: right;">  0.544937</td><td style="text-align: right;">         4.89067</td></tr>
<tr><td>tune_train_tcn_4ad25caa</td><td>TERMINATED</td><td>172.31.29.69:385  </td><td style="text-align: right;"> 0.375262</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.283791</td><td style="text-align: right;">               1</td><td style="text-align: right;">         160</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00197174 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   2.49933e-06</td><td style="text-align: right;">    18</td><td style="text-align: right;">        457.554 </td><td style="text-align: right;">    0.355547</td><td style="text-align: right;">  0.344453</td><td style="text-align: right;">         3.74896</td></tr>
<tr><td>tune_train_tcn_d9786fd8</td><td>TERMINATED</td><td>172.31.29.69:23421</td><td style="text-align: right;"> 0.362374</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.314984</td><td style="text-align: right;">               1</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00129555 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   2.75014e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        142.425 </td><td style="text-align: right;">    0.395115</td><td style="text-align: right;">  0.394668</td><td style="text-align: right;">         4.07031</td></tr>
<tr><td>tune_train_tcn_9e87dc47</td><td>TERMINATED</td><td>172.31.29.69:7513 </td><td style="text-align: right;"> 0.328219</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.315354</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00086894 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   5.2986e-05 </td><td style="text-align: right;">     2</td><td style="text-align: right;">         32.8862</td><td style="text-align: right;">    0.530818</td><td style="text-align: right;">  0.486747</td><td style="text-align: right;">         4.62236</td></tr>
<tr><td>tune_train_tcn_f039208e</td><td>TERMINATED</td><td>172.31.29.69:7918 </td><td style="text-align: right;"> 0.29292 </td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.289179</td><td style="text-align: right;">               1</td><td style="text-align: right;">         160</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.000812959</td><td style="text-align: right;">         200</td><td style="text-align: right;">           6</td><td style="text-align: right;">   4.39422e-05</td><td style="text-align: right;">     2</td><td style="text-align: right;">         39.3013</td><td style="text-align: right;">    0.558391</td><td style="text-align: right;">  0.526393</td><td style="text-align: right;">         4.80263</td></tr>
<tr><td>tune_train_tcn_dbafc0c5</td><td>TERMINATED</td><td>172.31.29.69:23884</td><td style="text-align: right;"> 0.324231</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.295168</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00156283 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   7.86085e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        125.978 </td><td style="text-align: right;">    0.397439</td><td style="text-align: right;">  0.382087</td><td style="text-align: right;">         4.04599</td></tr>
<tr><td>tune_train_tcn_060475b4</td><td>TERMINATED</td><td>172.31.29.69:8624 </td><td style="text-align: right;"> 0.382472</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.261712</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00152099 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.12519e-06</td><td style="text-align: right;">   162</td><td style="text-align: right;">       3704.1   </td><td style="text-align: right;">    0.273939</td><td style="text-align: right;">  0.27444 </td><td style="text-align: right;">         3.22226</td></tr>
<tr><td>tune_train_tcn_44822b28</td><td>TERMINATED</td><td>172.31.29.69:24806</td><td style="text-align: right;"> 0.365858</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.270078</td><td style="text-align: right;">               3</td><td style="text-align: right;">         160</td><td style="text-align: right;">            5</td><td style="text-align: right;">0.00139801 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.63798e-06</td><td style="text-align: right;">     6</td><td style="text-align: right;">        124.27  </td><td style="text-align: right;">    0.395591</td><td style="text-align: right;">  0.405388</td><td style="text-align: right;">         4.18876</td></tr>
<tr><td>tune_train_tcn_b7fe6e7f</td><td>TERMINATED</td><td>172.31.29.69:9416 </td><td style="text-align: right;"> 0.35826 </td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.307302</td><td style="text-align: right;">               1</td><td style="text-align: right;">         192</td><td style="text-align: right;">            3</td><td style="text-align: right;">0.00168844 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           4</td><td style="text-align: right;">   1.26697e-06</td><td style="text-align: right;">     2</td><td style="text-align: right;">         32.1298</td><td style="text-align: right;">    0.492075</td><td style="text-align: right;">  0.53974 </td><td style="text-align: right;">         5.10944</td></tr>
<tr><td>tune_train_tcn_3a9060f5</td><td>TERMINATED</td><td>172.31.29.69:25307</td><td style="text-align: right;"> 0.316221</td><td style="text-align: right;">          32</td><td style="text-align: right;"> 0.281569</td><td style="text-align: right;">               3</td><td style="text-align: right;">         192</td><td style="text-align: right;">            7</td><td style="text-align: right;">0.00153829 </td><td style="text-align: right;">         200</td><td style="text-align: right;">           5</td><td style="text-align: right;">   6.73474e-05</td><td style="text-align: right;">     6</td><td style="text-align: right;">        187.855 </td><td style="text-align: right;">    0.393787</td><td style="text-align: right;">  0.387803</td><td style="text-align: right;">         4.0236 </td></tr>
</tbody>
</table>
  </div>
</div>
<style>
.tuneStatus {
  color: var(--jp-ui-font-color1);
}
.tuneStatus .systemInfo {
  display: flex;
  flex-direction: column;
}
.tuneStatus td {
  white-space: nowrap;
}
.tuneStatus .trialStatus {
  display: flex;
  flex-direction: column;
}
.tuneStatus h3 {
  font-weight: bold;
}
.tuneStatus .hDivider {
  border-bottom-width: var(--jp-border-width);
  border-bottom-color: var(--jp-border-color0);
  border-bottom-style: solid;
}
.tuneStatus .vDivider {
  border-left-width: var(--jp-border-width);
  border-left-color: var(--jp-border-color0);
  border-left-style: solid;
  margin: 0.5em 1em 0.5em 1em;
}
</style>



    [TCN] saved clean final model .pth at /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Models/TCN_multihorizon_raytuned_model_4.pth


    /tmp/ipykernel_1683/2757692989.py:140: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      payload = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=DEVICE)
    /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/linux_venv/lib64/python3.12/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
      WeightNorm.apply(module, name, dim)


### 4.6 Plot Training Process from Ray Tune Logs
We plot per-epoch metrics for the best trials to inspect convergence. Validation curves are computed without dropout and reflect masked, horizon-weighted errors in original units. These plots aim to provide some insight into convergence behaviour and potential under-/overfitting.



```python
from mt4xai.plot import plot_losses_from_result
# plots the per-epoch metrics from best_result.metrics_dataframe for each model archetype
if best_lstm_res is not None: plot_losses_from_result(best_lstm_res, title_prefix="LSTM")
if best_tcn_res  is not None: plot_losses_from_result(best_tcn_res,  title_prefix="TCN")
```


    
![png](03__Modelling_files/03__Modelling_35_0.png)
    



    
![png](03__Modelling_files/03__Modelling_35_1.png)
    



    
![png](03__Modelling_files/03__Modelling_35_2.png)
    



    
![png](03__Modelling_files/03__Modelling_35_3.png)
    


The training curves show that both LSTM and TCN models converge steadily, with validation loss consistently below training loss. This happens because the use of dropout and gradient clipping during training injects noise that inflates the training loss, whereas validation is evaluated without these regularisers, leading to lower reported loss. Furthermore, the training loss is computed over longer and more diverse sequences, often including noisy or harder-to-predict segments, while the validation set may contain somewhat shorter or cleaner sequences. The use of the Huber loss also contributes: since it is more robust to outliers, validation loss is less penalised by extreme errors, while the training loss reflects the influence of a larger number of atypical or noisy samples. Across epochs, validation RMSE decreases and stabilises, with SOC predictions achieving the lowest error (≈0.5–0.6%) while power predictions remain more challenging (≈6–6.5 kW RMSE). 


 ## 5 - Model Evaluation and Selection
This section plots the best models' predictions, then evaluates the best LSTM and TCN models with Macro-RMSE before the performing model is selected and exported/saved for downstream applications. 

 ### 5.1 Plotting Predictions

 #### 5.1.1 Step-by-step predictions
In this section, we will inspect the models' predictions on individual charging sessions in order to better understand the models' behaviour. 
The utility functions below help prepare and align predictions with ground truth for visualisation.

The `plot_inputs_to_single_output_grid` was used to debug the models' predictions in the modelling process. Here, we only want to exemplify it's usage for error inspection and to show that the models' aren't doing sequence-level reconstruction, but step-by-step predictions across all horizons. Thus, we only show the LSTM model's predictions for a particular segment of a particular sessions. These plots do not provide the information we need to gain generalisable insights about the models' behaviour. 


```python
from mt4xai.plot import plot_inputs_to_single_output_grid
from mt4xai.data import LengthBucketSampler, fetch_charging_session, session_collate_fn

# test loader for plotting
test_sampler = LengthBucketSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    collate_fn=session_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,
)

# Pick one session via (batch_index, sample_index)
BATCH_INDEX, SAMPLE_INDEX = 31, 0

# get a session with predictions attached
session_lstm = fetch_charging_session(
    model=model_lstm, loader=test_loader, device=torch.device(DEVICE),
    power_scaler=power_scaler, soc_scaler=soc_scaler,
    idx_power_inp=IDX_POWER, idx_soc_inp=IDX_SOC,
    batch_index=BATCH_INDEX, sample_index=SAMPLE_INDEX,
)

# plot a small grid
plot_inputs_to_single_output_grid(
    session_lstm, i_list=[9,10,11,12,13,14], horizon=1,
    features_to_show=["power"], window_len=30, ncols=3,
    batch_index=BATCH_INDEX, sample_index=SAMPLE_INDEX,
);
```


    
![png](03__Modelling_files/03__Modelling_39_0.png)
    


#### 5.1.2 Plotting Multi-Horizon Predictions for a Complete Session
Plots of entire sessions show model predictions across horizons, illustrating temporal coherence of forecasts.


```python
session_lstm.plot_power_predictions();
```


    
![png](03__Modelling_files/03__Modelling_41_0.png)
    



```python
# get a session with TCN predictions attached
session_tcn = fetch_charging_session(
    model=model_tcn, loader=test_loader, device=torch.device(DEVICE),
    power_scaler=power_scaler, soc_scaler=soc_scaler,
    idx_power_inp=IDX_POWER, idx_soc_inp=IDX_SOC,
    batch_index=BATCH_INDEX, sample_index=SAMPLE_INDEX,
)

session_tcn.plot_power_predictions();
```


    
![png](03__Modelling_files/03__Modelling_42_0.png)
    


#### 5.1.3 Plotting Complete Power Predictions for Multiple Sample Sessions
Predicted and true power curves are compared across multiple sessions to evaluate generalisation.


```python
from mt4xai.plot import plot_grid_power_predictions

# choose six samples for each model
pairs = [(12, 0), (18, 1), (36, 0), (19, 0)]

def build_sessions(model, pairs):
    out = []
    for b, s in pairs:
        sess = fetch_charging_session(
            model=model, loader=test_loader, device=torch.device(DEVICE),
            power_scaler=power_scaler, soc_scaler=soc_scaler,
            idx_power_inp=IDX_POWER, idx_soc_inp=IDX_SOC,
            batch_index=b, sample_index=s,
        )
        out.append(sess)
    return out

# LSTM grid plot across horizons
if model_lstm is not None:
    sessions_lstm = build_sessions(model_lstm, pairs)
    plot_grid_power_predictions(sessions_lstm, t_min_eval=1, show_points=True, show_soc=True)
```


    
![png](03__Modelling_files/03__Modelling_44_0.png)
    



```python
# TCN predictions grid plot across horizons
if model_tcn is not None:
    sessions_tcn = build_sessions(model_tcn, pairs)
    plot_grid_power_predictions(sessions_tcn, t_min_eval=1, show_points=True, show_soc=True)
```


    
![png](03__Modelling_files/03__Modelling_45_0.png)
    


### 5.2 Model Evaluation and Selection
We select the final model by evaluating the best LSTM and TCN models with Macro-Averaged Root Mean Squared Error (Macro-RMSE) on the held-out test set, complemented by the qualitative inspection of the multi-horizon prediction plots. For a detailed explanation of Macro-RMSE, see section 4.


```python
from mt4xai.inference import evaluate_model

# Evaluates whichever models are available
if model_lstm is not None:
    stats_lstm = evaluate_model(model_lstm, test_dataset, BATCH_SIZE, DEVICE, power_scaler, 
                                horizon=HORIZON, weight_decay=float(cfg_lstm["weight_decay"]))
    lstm_score = stats_lstm["MacroRMSE"]
    print(f"[LSTM] Test Macro-Averaged RMSE: {lstm_score:.4f} "
          f"(across {stats_lstm['NumSequencesEvaluated']} sequences)")

if model_tcn is not None:
    stats_tcn = evaluate_model(model_tcn, test_dataset, BATCH_SIZE, DEVICE, 
                               power_scaler, horizon=HORIZON, weight_decay=float(cfg_tcn["weight_decay"]))
    tcn_score = stats_tcn["MacroRMSE"]
    print(f"[TCN]  Test Macro-Averaged RMSE: {tcn_score:.4f} "
          f"(across {stats_tcn['NumSequencesEvaluated']} sequences)")
    
best_model_name = "LSTM" if lstm_score <= tcn_score else "TCN"
print(f"Test Macro-RMSE — LSTM: {lstm_score:.3f} kW | TCN: {tcn_score:.3f} kW. Selected: {best_model_name}")
```

    [LSTM] Test Macro-Averaged RMSE: 3.5138 (across 12183 sequences)
    [TCN]  Test Macro-Averaged RMSE: 3.6100 (across 12183 sequences)
    Test Macro-RMSE — LSTM: 3.514 kW | TCN: 3.610 kW. Selected: LSTM



```python
# Save the selected model with its state dict and Ray Tune config 
cfg_lstm["horizon"] = HORIZON
torch.save({
    "model_state_dict": model_lstm.state_dict(),
    "config": cfg_lstm,  # Ray Tune best config
    "input_features": input_features,  # to reconstruct input size
    "target_features": target_features
}, os.path.join(MODEL_FOLDER_PATH, "final/final_model.pth"))

```

### 5.4 Summary and Next Steps

We trained and tuned residual multi-horizon forecasters, evaluated them with Macro-RMSE in kW, and selected the LSTM for deployment. Tuned checkpoints were restored and saved to versioned `.pth` files to ensure reproducibility. The saved LSTM model is used in the downstream tasks of anomaly detection and charging curve simplification. 

**Next steps:**
1. Notebook 4: [Anomaly Detection](04__Anomaly_Detection.ipynb)
2. Notebook 5: [Curve Simplification](05__Curve_Simplification.ipynb)
3. Notebook 6: [MT4XAI](06__MT4XAI.ipynb)
