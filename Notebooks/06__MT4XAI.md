# Notebook \#6: Machine Teaching for XAI
#### by Sebastian Einar Salas Røkholt

---

**Summary**  
This notebook operationalises Machine Teaching for Explainable AI (MT4XAI) on EV-charging time series. The goal of our simplified-example-based approach to MT4XAI is to help participants efficiently form an accurate mental model of the time series classifier from a *small* set of automatically selected and simplified examples. In this notebook, we load the trained classifier and session data, compute an **ORS** pool of robust piece-wise linear simplifications (k-segment curves that preserve the model’s label), bin the pool by class and k to control curriculum difficulty, select a compact, diverse **teaching set** $S$ using a facility-location objective with fidelity/robustness terms, and serve grouped teaching sessions with the original power curve + the simplified power curve and simplified SOC overlays for the upcoming user study (two experimental groups + control group). 

---

**Index**  
- [**1 - Introduction and Setup**](#1---introduction-and-setup)  
  - [*1.1 Setup*](#11-setup)  
  - [*1.2 Load the Forecasting Model*](#12-load-the-forecasting-model)  
  - [*1.3 Data Preparation*](#13-data-preparation)  
- [**2 - Computing the Teaching Pool**](#2---computing-the-teaching-pool)  
  - [*2.1 - Running the pool constructor*](#21---running-the-pool-constructor)  
  - [*2.2 - Pool summary and diagnostics*](#22---pool-summary-and-diagnostics)  
  - [*2.3 - Binning the pool by class and k*](#23---binning-the-pool-by-class-and-k)  
- [**3 - Constructing the teaching set S**](#3---constructing-the-teaching-set-s)  
  - [*3.1 - Selection budget*](#31---selection-budget)  
  - [*3.2 - Teaching set selection*](#32---teaching-set-selection)  
- [**4 - The Teaching Session: Serving examples from the teaching set**](#teaching)  
  - [*Serving examples from the teaching set*](#4---serving-examples-from-the-teaching-set)  

---

**Introduction: Machine Teaching for XAI (MT4XAI)**  
Traditional post-hoc XAI for time series often struggles with human comprehensibility, and doubly so for time series classifiers as time series are notoriously difficult for humans to interpret. MT4XAI reframes example-based XAI techniques as *teaching*: the machine teaching system (teacher) selects a *small* set of examples (a.k.a. "witnesses") s.t. a human (learner) can reconstruct an accurate mental model of the black-box AI's behaviour.

In this project, example complexity is controlled by **Optimal Robust Simplifications (ORS)**: for each session we compute a piece-wise linear simplification with k segments that (i) stays close to the original curve, (ii) keeps the model's decision, and (iii) is robust to local perturbations. ORS admits a principled balance between error, simplicity (k), and robustness, with a polynomial-time algorithm under mild conditions. These simplifications intend to make the salient shape cues apparent to non-experts while preserving decision-relevant shapes/structure. 

We then assemble a teaching set S from a class-balanced "pool" of pre-computed simplifications using machine teaching principles. Rather than showing arbitrary cases, we optimise for a compact subset that best "covers" the model’s behaviours, enabling simulate-the-model tasks in the user study. Prior MT4XAI studies show that machine-selected witness sets can teach target behaviours more effectively than random sampling, and that accounting for human priors and representation choices matters. This notebook instantiates those ideas for multivariate charging sessions (power + SOC overlays) and prepares trial groups A/B/C for the user study: 
 - Group A receives original `power`, simplified `power` and simplified `soc` ordered by simplicity/difficulty
 - Group B receives the same teaching set as group A but in random order
 - Group C receives the same teaching set as B but without simplifications (original `power` and `soc`).

**References & Background Literature:**
- [Optimal Robust Simplifications for Explaining Time Series Classifications](https://xai.w.uib.no/files/2024/07/ORS.pdf) (2024) by Telle, Ferri & Håvardstun.
- [XAI with Machine Teaching when Humans Are (Not) Informed about the Irrelevant Features](https://doi.org/10.1007/978-3-031-43418-1_23) (2023) by Håvardstun, Ferri, Hernández-Orallo, Parviainen & Telle. 
- [When Redundancy Matters: Machine Teaching of Representations](https://arxiv.org/pdf/2401.12711.pdf) (2024) by Ferri, Garigliotti, Håvardstun, Hernandez-Orallo & Telle.
- [XAI with Machine Teaching when Humans Are (Not) Informed about the Irrelevant Features](https://doi.org/10.1007/978-3-031-43418-1_23) (2023) by Håvardstun, Ferri, Hernandez-Orallo, Parviainen & Telle. 


## 1 - Setup
This notebook uses a class-based API defined in the project's custom Python package named `mt4xai`. In its modules named `teach` and `data`, we find these classes:
- `ChargingSession` contains the original, dense, unscaled arrays for the `power` series (and optionally for other channels/features as well). 
- `ChargingSessionSimplification` with knots\* only (indices + values). We cnvert these to dense simplifications and fetch original raw series on demand.
- `TeachingPool` owns the pool (`pool.parquet`), binning, and paths.
- `TeachingSet` performs selection with greedy facility-location (+ lazy pruning), then exposes A/B/C samplers.

Note: 
A *knot* is an endpoint of a straight line segment in the simplification. We define **k = number of straight line segments = knots − 1**.


```python
import sys
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# project config
sys.path.append(str(Path.cwd().parent))  # Adds additional scripts (e.g. project_config.py) in parent dir to path
from project_config import load_config
cfg = load_config()

# Modules from the project's MT4XAI package
%load_ext autoreload
%autoreload 2
from mt4xai.model import load_lstm_model
from mt4xai.data import split_data, apply_scalers, fit_scalers_on_train, build_loader
from mt4xai.ors import ORSParams
from mt4xai.teach import TeachingPool, TeachingPoolConfig, TeachingSet, selection_vs_pool_report

# Pandas config
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", 120)

# paths
DATA_FP = cfg.paths.dataset
MODEL_FP = cfg.paths.final_model
TEACHING_DIR = Path(cfg.paths.teaching_pool) 
POOL_PARQUET = TEACHING_DIR / "pool.parquet"
SAMPLE_PLAN = TEACHING_DIR / "sampled_normals.json"

# constants / tunable knobs
DEVICE = torch.device(cfg.project.device)
print("Device: ", DEVICE)
RANDOM_SEED = cfg.project.random_seed
AD_THRESHOLD = cfg.anomaly_detection.rmse_threshold
```

    CONFIG FILE LOADED: 
    {'project': {'random_seed': 42, 'root_dir': None}, 'paths': {'dataset': 'Data/etron55-charging-sessions.parquet', 'teaching_pool': 'Data/teaching_pool', 'models': 'Models', 'final_model': 'Models/final/final_model.pth', 'figures': 'Figures', 'logs': 'Logs'}, 'inference': {'horizon': 5, 'final_model_name': 'final_model.pth', 'power_weight': 0.6522982410461, 'horizon_decay_lambda': 0.4}, 'anomaly_detection': {'t_min_eval': 1, 'rmse_threshold': 8.5962, 'ad_pct_threshold': 0.95, 'metric': 'macro_rmse'}, 'ors': {'soc_stage1_mode': 'rdp', 'soc_rdp_epsilon': 0.75, 'soc_rdp_candidates': 5, 'soc_rdp_eps_min': 1e-06, 'soc_rdp_eps_max': 100.0, 'stage2_err_metric': 'l2', 'epsilon_mode': 'fraction'}, 'teaching': {'teaching_pool_dir': '../Data/teaching_pool', 'teaching_set_size': 200}}
    Device:  cuda


### 1.2 Load the Forecasting Model
Here we load the LSTM forecasting model trained in the `03__Modelling.ipynb` notebook along with hyperparamters and some constants required to run inference on new data. 


```python
# load model and hyperparams from Ray Tune checkpoint
model, ckpt = load_lstm_model(MODEL_FP, device=DEVICE)
input_features  = ckpt["input_features"]
target_features = ckpt["target_features"]
H = int(ckpt["config"]["horizon"])  # forecast horizon
idx_power_inp = input_features.index("power")
idx_soc_inp = input_features.index("soc")
print(f"Loaded pretrained forecasting model {model}")
```

    /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/src/mt4xai/model.py:45: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      checkpoint = torch.load(path, map_location=device)


    Loaded pretrained forecasting model MultiHorizonLSTM(
      (lstm): LSTM(12, 512, num_layers=5, batch_first=True, dropout=0.2754636459029)
      (linear): Linear(in_features=512, out_features=10, bias=True)
    )


### 1.3 Data Preparation
In this section, we will load, transform, and split the data in exactly the same way as we did previously in order to extract the test set, which we will use as a starting point for building the teaching set $S$. 


```python
# load and scale data (in the same way as in previous notebooks)
df = pd.read_parquet(DATA_FP)
drop_cols = [c for c in [
    "energy","charger_category","timestamp","nearest_weather_station",
    "timestamp_d","lat","lon","timestamp_H"
] if c in df.columns]
df = df.drop(columns=drop_cols).copy()

train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1, random_seed=RANDOM_SEED)  # exact same split as before
scalers = fit_scalers_on_train(train_df, list(set(input_features) | set(target_features)))
power_scaler, soc_scaler = scalers["power"], scalers["soc"]
test_s = apply_scalers(test_df, scalers)

test_loader = build_loader(test_s, input_features, target_features, H,
                           batch_size=16, shuffle=False, num_workers=0)
print(f"There are {len(test_loader.dataset.groups)} charging sessions in the test set")
```

    There are 12183 charging sessions in the test set


### 2 - Computing the Teaching Pool
The Teaching Pool is a relatively large collection of examples that have been sampled (charging sessions / examples) from the test set. By definition, 95% of the charging sessions in the test set are normal while only 5% are abnormal. However, we want the teaching set to have an equal class distribution, so as an intermediate step we will compute a "teaching pool" that contains all abnormals and a random sampling of an equal number of normals. This is strategy is reasonable because there is less variation among normals than for abnormals. We then compute ORS for all sessions in the teaching pool. 

#### 2.1 - Running the pool constructor
Constructing the pool of simplifications + raw sessions is very computationally intensive, since to calculate the ORS simplifications for `power` we need to run model inference thousands of times per simplification. The teaching set contains 586 abnormals, so we will compute over a thousand ORS simplifications for `power` in our teaching pool. Additionally, we calculate ORS simplifications for `SOC`. We drastically reduce the computational effort for `SOC` simplifications by employing the Ramer-Douglas-Pecker simplification algorithm for generating stage-1 candidates for ORS, as opposed to the `power` simplifications which use dynamic programming with prefix sums. This is acceptable because the `SOC` feature has low importance towards determining the anomaly detection system's classification of the charging session. In contrary to `power`, `SOC` is a monotonically increasing and linear curve so we expect `k` ∈ [1, 3] even with RDP for ORS stage 1. </br>

The `mt4xai` API let's us load an existing pool (from the file `pool.parquet`) or construct it from scratch. Construction runs inference + ORS + embedding and writes the SQLite cache and the Parquet snapshot.


```python
# loads an existing pool if available, otherwise build from scratch
pool_parquet = TEACHING_DIR / "pool.parquet"
construct_pool = False

# configures the teaching pool and ORS parameters
tpconfig = TeachingPoolConfig(
    model_path=MODEL_FP,  # a pre-trained "black box" (multivariate LSTM) model
    output_dir=TEACHING_DIR,
    threshold=AD_THRESHOLD,  # 8.5962, i.e. the 95th percentile
    random_seed=RANDOM_SEED,
    device=DEVICE,
    export_every=10,
    L=128, P=4,   # for embeddings 
    ors_params=ORSParams(
        stage1_mode="dp_prefix", stage2_err_metric="rmse",
        dp_q=300, rdp_stage1_candidates=50,
        dp_alpha=0.0075, beta=4.0, gamma=0.05,
        R=3000, epsilon_mode="fraction", epsilon_value=0.2,
        t_min_eval=1, anchor_endpoints="last", 
        min_k=1, max_k=12, random_seed=RANDOM_SEED,
        soc_stage1_mode="rdp", soc_rdp_epsilon=0.75,
        model_id="final_model.pth"
    ),
    power_weight=cfg.inference.power_weight, 
    decay_lambda=cfg.inference.horizon_decay_lambda
)

if pool_parquet.exists():
    # loads the pool snapshot from disk
    print("Loaded previously computed teaching pool from disk")
    pool = TeachingPool.load_from_parquet(pool_parquet, config=tpconfig)  # or pass config=None if tpconfig is not correct
    print(len(pool.pool_df))
    if len(pool.pool_df) < (586*2):  # 586 normals + 586 abnormals in test set
        construct_pool = True
else: 
    construct_pool = True

if construct_pool:
    # builds the pool end-to-end, then loads it
    # builds the pool of simplifications and raw sessions
    pool = TeachingPool.construct_from_cfg(
        model=model,
        config=tpconfig,
        test_loader=test_loader,
        power_scaler=power_scaler,
        soc_scaler=soc_scaler,
        idx_power_inp=idx_power_inp,
        idx_soc_inp=idx_soc_inp,
    )
```

    Loaded previously computed teaching pool from disk
    200
    [teach] loaded existing sampling plan: abnormal=586, normal=586
    [teach] sampling plan: abnormal=586, normal=586, total=1172
    [teach] found 206 already cached; skipping those.
    [teach] ORS    1/966 ( 0.1%) sid=2722660
    [teach] ORS    2/966 ( 0.2%) sid=2725884
    [teach] ORS    3/966 ( 0.3%) sid=2731164
    [teach] ORS    4/966 ( 0.4%) sid=2734400
    [teach] ORS    5/966 ( 0.5%) sid=2749519
    [teach] ORS    6/966 ( 0.6%) sid=2750783
    [teach] ORS    7/966 ( 0.7%) sid=2770866
    [teach] ORS    8/966 ( 0.8%) sid=2812620
    [teach] ORS    9/966 ( 0.9%) sid=2819003
    [teach] ORS   10/966 ( 1.0%) sid=2900786
    [teach] exporting parquet snapshot ...
    [teach] ORS   11/966 ( 1.1%) sid=2901435
    [teach] ORS   12/966 ( 1.2%) sid=2902359
    [teach] ORS   13/966 ( 1.3%) sid=2919176
    [teach] ORS   14/966 ( 1.4%) sid=2922361
    [teach] ORS   15/966 ( 1.6%) sid=2925485
    [teach] ORS   16/966 ( 1.7%) sid=2954210
    [teach] ORS   17/966 ( 1.8%) sid=2955335
    [teach] ORS   18/966 ( 1.9%) sid=2970205
    [teach] ORS   19/966 ( 2.0%) sid=2983799
    [teach] ORS   20/966 ( 2.1%) sid=2989700
    [teach] exporting parquet snapshot ...
    [teach] ORS   21/966 ( 2.2%) sid=2989716
    [teach] ORS   22/966 ( 2.3%) sid=2991114
    [teach] ORS   23/966 ( 2.4%) sid=2994731
    [teach] ORS   24/966 ( 2.5%) sid=3000345
    [teach] ORS   25/966 ( 2.6%) sid=3002001
    [teach] ORS   26/966 ( 2.7%) sid=3078594
    [teach] ORS   27/966 ( 2.8%) sid=3093479
    [teach] ORS   28/966 ( 2.9%) sid=3134843
    [teach] ORS   29/966 ( 3.0%) sid=3188925
    [teach] ORS   30/966 ( 3.1%) sid=3225910
    [teach] exporting parquet snapshot ...
    [teach] ORS   31/966 ( 3.2%) sid=3259587
    [teach] ORS   32/966 ( 3.3%) sid=3259637
    [teach] ORS   33/966 ( 3.4%) sid=3271344
    [teach] ORS   34/966 ( 3.5%) sid=3304183
    [teach] ORS   35/966 ( 3.6%) sid=3306275
    [teach] ORS   36/966 ( 3.7%) sid=3313275
    [teach] ORS   37/966 ( 3.8%) sid=3323361
    [teach] ORS   38/966 ( 3.9%) sid=3342983
    [teach] ORS   39/966 ( 4.0%) sid=3343555
    [teach] ORS   40/966 ( 4.1%) sid=3359178
    [teach] exporting parquet snapshot ...
    [teach] ORS   41/966 ( 4.2%) sid=3370407
    [teach] ORS   42/966 ( 4.3%) sid=3403789
    [teach] ORS   43/966 ( 4.5%) sid=3421948
    [teach] ORS   44/966 ( 4.6%) sid=3436995
    [teach] ORS   45/966 ( 4.7%) sid=3437471
    [teach] ORS   46/966 ( 4.8%) sid=3449138
    [teach] ORS   47/966 ( 4.9%) sid=3453672
    [teach] ORS   48/966 ( 5.0%) sid=3474163
    [teach] ORS   49/966 ( 5.1%) sid=3492091
    [teach] ORS   50/966 ( 5.2%) sid=3504298
    [teach] exporting parquet snapshot ...
    [teach] ORS   51/966 ( 5.3%) sid=3511159
    [teach] ORS   52/966 ( 5.4%) sid=3512443
    [teach] ORS   53/966 ( 5.5%) sid=3530734
    [teach] ORS   54/966 ( 5.6%) sid=3534379
    [teach] ORS   55/966 ( 5.7%) sid=3544732
    [teach] ORS   56/966 ( 5.8%) sid=3550778
    [teach] ORS   57/966 ( 5.9%) sid=3553088
    [teach] ORS   58/966 ( 6.0%) sid=3579639
    [teach] ORS   59/966 ( 6.1%) sid=3587506
    [teach] ORS   60/966 ( 6.2%) sid=3590256
    [teach] exporting parquet snapshot ...
    [teach] ORS   61/966 ( 6.3%) sid=3604358
    [teach] ORS   62/966 ( 6.4%) sid=3613915
    [teach] ORS   63/966 ( 6.5%) sid=3619928
    [teach] ORS   64/966 ( 6.6%) sid=3626036
    [teach] ORS   65/966 ( 6.7%) sid=3634244
    [teach] ORS   66/966 ( 6.8%) sid=3638776
    [teach] ORS   67/966 ( 6.9%) sid=3642460
    [teach] ORS   68/966 ( 7.0%) sid=3649192
    [teach] ORS   69/966 ( 7.1%) sid=3677675
    [teach] ORS   70/966 ( 7.2%) sid=3690034
    [teach] exporting parquet snapshot ...
    [teach] ORS   71/966 ( 7.3%) sid=3709559
    [teach] ORS   72/966 ( 7.5%) sid=3718802
    [teach] ORS   73/966 ( 7.6%) sid=3719978
    [teach] ORS   74/966 ( 7.7%) sid=3724473
    [teach] ORS   75/966 ( 7.8%) sid=3742697
    [teach] ORS   76/966 ( 7.9%) sid=3757373
    [teach] ORS   77/966 ( 8.0%) sid=3767738
    [teach] ORS   78/966 ( 8.1%) sid=3768137
    [teach] ORS   79/966 ( 8.2%) sid=3771367
    [teach] ORS   80/966 ( 8.3%) sid=3782291
    [teach] exporting parquet snapshot ...
    [teach] ORS   81/966 ( 8.4%) sid=3800089
    [teach] ORS   82/966 ( 8.5%) sid=3806040
    [teach] ORS   83/966 ( 8.6%) sid=3808136
    [teach] ORS   84/966 ( 8.7%) sid=3808890
    [teach] ORS   85/966 ( 8.8%) sid=3810107
    [teach] ORS   86/966 ( 8.9%) sid=3811672
    [teach] ORS   87/966 ( 9.0%) sid=3815079
    [teach] ORS   88/966 ( 9.1%) sid=3818120
    [teach] ORS   89/966 ( 9.2%) sid=3819802
    [teach] ORS   90/966 ( 9.3%) sid=3828746
    [teach] exporting parquet snapshot ...
    [teach] ORS   91/966 ( 9.4%) sid=3840974
    [teach] ORS   92/966 ( 9.5%) sid=3860883
    [teach] ORS   93/966 ( 9.6%) sid=3871088
    [teach] ORS   94/966 ( 9.7%) sid=3875026
    [teach] ORS   95/966 ( 9.8%) sid=3891229
    [teach] ORS   96/966 ( 9.9%) sid=3895264
    [teach] ORS   97/966 (10.0%) sid=3907158
    [teach] ORS   98/966 (10.1%) sid=3914671
    [teach] ORS   99/966 (10.2%) sid=3916390
    [teach] ORS  100/966 (10.4%) sid=3925410
    [teach] exporting parquet snapshot ...
    [teach] ORS  101/966 (10.5%) sid=3930687
    [teach] ORS  102/966 (10.6%) sid=3932553
    [teach] ORS  103/966 (10.7%) sid=3933670
    [teach] ORS  104/966 (10.8%) sid=3937885
    [teach] ORS  105/966 (10.9%) sid=3945122
    [teach] ORS  106/966 (11.0%) sid=3945661
    [teach] ORS  107/966 (11.1%) sid=3958428
    [teach] ORS  108/966 (11.2%) sid=3991560
    [teach] ORS  109/966 (11.3%) sid=3992532
    [teach] ORS  110/966 (11.4%) sid=3994469
    [teach] exporting parquet snapshot ...
    [teach] ORS  111/966 (11.5%) sid=3994821
    [teach] ORS  112/966 (11.6%) sid=3999834
    [teach] ORS  113/966 (11.7%) sid=4000845
    [teach] ORS  114/966 (11.8%) sid=4001524
    [teach] ORS  115/966 (11.9%) sid=4001762
    [teach] ORS  116/966 (12.0%) sid=4003228
    [teach] ORS  117/966 (12.1%) sid=4007889
    [teach] ORS  118/966 (12.2%) sid=4007926
    [teach] ORS  119/966 (12.3%) sid=4008161
    [teach] ORS  120/966 (12.4%) sid=4016763
    [teach] exporting parquet snapshot ...
    [teach] ORS  121/966 (12.5%) sid=4029988
    [teach] ORS  122/966 (12.6%) sid=4036553
    [teach] ORS  123/966 (12.7%) sid=4043093
    [teach] ORS  124/966 (12.8%) sid=4043459
    [teach] ORS  125/966 (12.9%) sid=4046121
    [teach] ORS  126/966 (13.0%) sid=4046654
    [teach] ORS  127/966 (13.1%) sid=4067186
    [teach] ORS  128/966 (13.3%) sid=4068608
    [teach] ORS  129/966 (13.4%) sid=4077109
    [teach] ORS  130/966 (13.5%) sid=4081735
    [teach] exporting parquet snapshot ...
    [teach] ORS  131/966 (13.6%) sid=4085328
    [teach] ORS  132/966 (13.7%) sid=4088088
    [teach] ORS  133/966 (13.8%) sid=4098786
    [teach] ORS  134/966 (13.9%) sid=4100694
    [teach] ORS  135/966 (14.0%) sid=4112976
    [teach] ORS  136/966 (14.1%) sid=4130877
    [teach] ORS  137/966 (14.2%) sid=4139946
    [teach] ORS  138/966 (14.3%) sid=4144539
    [teach] ORS  139/966 (14.4%) sid=4151827
    [teach] ORS  140/966 (14.5%) sid=4154088
    [teach] exporting parquet snapshot ...
    [teach] ORS  141/966 (14.6%) sid=4154794
    [teach] ORS  142/966 (14.7%) sid=4156880
    [teach] ORS  143/966 (14.8%) sid=4171422
    [teach] ORS  144/966 (14.9%) sid=4188829
    [teach] ORS  145/966 (15.0%) sid=4197803
    [teach] ORS  146/966 (15.1%) sid=4224342
    [teach] ORS  147/966 (15.2%) sid=4228275
    [teach] ORS  148/966 (15.3%) sid=4241046
    [teach] ORS  149/966 (15.4%) sid=4241761
    [teach] ORS  150/966 (15.5%) sid=4243345
    [teach] exporting parquet snapshot ...
    [teach] ORS  151/966 (15.6%) sid=4247247
    [teach] ORS  152/966 (15.7%) sid=4256887
    [teach] ORS  153/966 (15.8%) sid=4262897
    [teach] ORS  154/966 (15.9%) sid=4270746
    [teach] ORS  155/966 (16.0%) sid=4275664
    [teach] ORS  156/966 (16.1%) sid=4275867
    [teach] ORS  157/966 (16.3%) sid=4279664
    [teach] ORS  158/966 (16.4%) sid=4280374
    [teach] ORS  159/966 (16.5%) sid=4285339
    [teach] ORS  160/966 (16.6%) sid=4289132
    [teach] exporting parquet snapshot ...
    [teach] ORS  161/966 (16.7%) sid=4289812
    [teach] ORS  162/966 (16.8%) sid=4300458
    [teach] ORS  163/966 (16.9%) sid=4325298
    [teach] ORS  164/966 (17.0%) sid=4325999
    [teach] ORS  165/966 (17.1%) sid=4350735
    [teach] ORS  166/966 (17.2%) sid=4353507
    [teach] ORS  167/966 (17.3%) sid=4358631
    [teach] ORS  168/966 (17.4%) sid=4360282
    [teach] ORS  169/966 (17.5%) sid=4362098
    [teach] ORS  170/966 (17.6%) sid=4377168
    [teach] exporting parquet snapshot ...
    [teach] ORS  171/966 (17.7%) sid=4388201
    [teach] ORS  172/966 (17.8%) sid=4394254
    [teach] ORS  173/966 (17.9%) sid=4395980
    [teach] ORS  174/966 (18.0%) sid=4403200
    [teach] ORS  175/966 (18.1%) sid=4404387
    [teach] ORS  176/966 (18.2%) sid=4413569
    [teach] ORS  177/966 (18.3%) sid=4420531
    [teach] ORS  178/966 (18.4%) sid=4422992
    [teach] ORS  179/966 (18.5%) sid=4426407
    [teach] ORS  180/966 (18.6%) sid=4438887
    [teach] exporting parquet snapshot ...
    [teach] ORS  181/966 (18.7%) sid=4440104
    [teach] ORS  182/966 (18.8%) sid=4440923
    [teach] ORS  183/966 (18.9%) sid=4446449
    [teach] ORS  184/966 (19.0%) sid=4452472
    [teach] ORS  185/966 (19.2%) sid=4454587
    [teach] ORS  186/966 (19.3%) sid=4454858
    [teach] ORS  187/966 (19.4%) sid=4457960
    [teach] ORS  188/966 (19.5%) sid=4461493
    [teach] ORS  189/966 (19.6%) sid=4468530
    [teach] ORS  190/966 (19.7%) sid=4470802
    [teach] exporting parquet snapshot ...
    [teach] ORS  191/966 (19.8%) sid=4471020
    [teach] ORS  192/966 (19.9%) sid=4476408
    [teach] ORS  193/966 (20.0%) sid=4481151
    [teach] ORS  194/966 (20.1%) sid=4481684
    [teach] ORS  195/966 (20.2%) sid=4482078
    [teach] ORS  196/966 (20.3%) sid=4482161
    [teach] ORS  197/966 (20.4%) sid=4482547
    [teach] ORS  198/966 (20.5%) sid=4486923
    [teach] ORS  199/966 (20.6%) sid=4486995
    [teach] ORS  200/966 (20.7%) sid=4495090
    [teach] exporting parquet snapshot ...
    [teach] ORS  201/966 (20.8%) sid=4497661
    [teach] ORS  202/966 (20.9%) sid=4497749
    [teach] ORS  203/966 (21.0%) sid=4498925
    [teach] ORS  204/966 (21.1%) sid=4503129
    [teach] ORS  205/966 (21.2%) sid=4504151
    [teach] ORS  206/966 (21.3%) sid=4505585
    [teach] ORS  207/966 (21.4%) sid=4508258
    [teach] ORS  208/966 (21.5%) sid=4513706
    [teach] ORS  209/966 (21.6%) sid=4516031
    [teach] ORS  210/966 (21.7%) sid=4518003
    [teach] exporting parquet snapshot ...
    [teach] ORS  211/966 (21.8%) sid=4522442
    [teach] ORS  212/966 (21.9%) sid=4524989
    [teach] ORS  213/966 (22.0%) sid=4530951
    [teach] ORS  214/966 (22.2%) sid=4531450
    [teach] ORS  215/966 (22.3%) sid=4531928
    [teach] ORS  216/966 (22.4%) sid=4533384
    [teach] ORS  217/966 (22.5%) sid=4533746
    [teach] ORS  218/966 (22.6%) sid=4535899
    [teach] ORS  219/966 (22.7%) sid=4539249
    [teach] ORS  220/966 (22.8%) sid=4541745
    [teach] exporting parquet snapshot ...
    [teach] ORS  221/966 (22.9%) sid=4545597
    [teach] ORS  222/966 (23.0%) sid=4552659
    [teach] ORS  223/966 (23.1%) sid=4554937
    [teach] ORS  224/966 (23.2%) sid=4557615
    [teach] ORS  225/966 (23.3%) sid=4561725
    [teach] ORS  226/966 (23.4%) sid=4565492
    [teach] ORS  227/966 (23.5%) sid=4585741
    [teach] ORS  228/966 (23.6%) sid=4587269
    [teach] ORS  229/966 (23.7%) sid=4600308
    [teach] ORS  230/966 (23.8%) sid=4613546
    [teach] exporting parquet snapshot ...
    [teach] ORS  231/966 (23.9%) sid=4613664
    [teach] ORS  232/966 (24.0%) sid=4613944
    [teach] ORS  233/966 (24.1%) sid=4618983
    [teach] ORS  234/966 (24.2%) sid=4619077
    [teach] ORS  235/966 (24.3%) sid=4620608
    [teach] ORS  236/966 (24.4%) sid=4620610
    [teach] ORS  237/966 (24.5%) sid=4628084
    [teach] ORS  238/966 (24.6%) sid=4635833
    [teach] ORS  239/966 (24.7%) sid=4636930
    [teach] ORS  240/966 (24.8%) sid=4640148
    [teach] exporting parquet snapshot ...
    [teach] ORS  241/966 (24.9%) sid=4640676
    [teach] ORS  242/966 (25.1%) sid=4644643
    [teach] ORS  243/966 (25.2%) sid=4663089
    [teach] ORS  244/966 (25.3%) sid=4663106
    [teach] ORS  245/966 (25.4%) sid=4665440
    [teach] ORS  246/966 (25.5%) sid=4668060
    [teach] ORS  247/966 (25.6%) sid=4674827
    [teach] ORS  248/966 (25.7%) sid=4681174
    [teach] ORS  249/966 (25.8%) sid=4688211
    [teach] ORS  250/966 (25.9%) sid=4705951
    [teach] exporting parquet snapshot ...
    [teach] ORS  251/966 (26.0%) sid=4711742
    [teach] ORS  252/966 (26.1%) sid=4712266
    [teach] ORS  253/966 (26.2%) sid=4722413
    [teach] ORS  254/966 (26.3%) sid=4734739
    [teach] ORS  255/966 (26.4%) sid=4752081
    [teach] ORS  256/966 (26.5%) sid=4756211
    [teach] ORS  257/966 (26.6%) sid=4757273
    [teach] ORS  258/966 (26.7%) sid=4776302
    [teach] ORS  259/966 (26.8%) sid=4783052
    [teach] ORS  260/966 (26.9%) sid=4783725
    [teach] exporting parquet snapshot ...
    [teach] ORS  261/966 (27.0%) sid=4804101
    [teach] ORS  262/966 (27.1%) sid=4811212
    [teach] ORS  263/966 (27.2%) sid=4812176
    [teach] ORS  264/966 (27.3%) sid=4826027
    [teach] ORS  265/966 (27.4%) sid=4829450
    [teach] ORS  266/966 (27.5%) sid=4837199
    [teach] ORS  267/966 (27.6%) sid=4838570
    [teach] ORS  268/966 (27.7%) sid=4850872
    [teach] ORS  269/966 (27.8%) sid=4859257
    [teach] ORS  270/966 (28.0%) sid=4863374
    [teach] exporting parquet snapshot ...
    [teach] ORS  271/966 (28.1%) sid=4869884
    [teach] ORS  272/966 (28.2%) sid=4871000
    [teach] ORS  273/966 (28.3%) sid=4871411
    [teach] ORS  274/966 (28.4%) sid=4874500
    [teach] ORS  275/966 (28.5%) sid=4885929
    [teach] ORS  276/966 (28.6%) sid=4891046
    [teach] ORS  277/966 (28.7%) sid=4898707
    [teach] ORS  278/966 (28.8%) sid=4898888
    [teach] ORS  279/966 (28.9%) sid=4915092
    [teach] ORS  280/966 (29.0%) sid=4922448
    [teach] exporting parquet snapshot ...
    [ORS][warn] sid=4931199 no valid candidates after constraints (mode=dp_prefix, k_span=2..4, dp_q=36, beta=4.0). Trying fallback #1.
    [teach] ORS  281/966 (29.1%) sid=4931199
    [teach] ORS  282/966 (29.2%) sid=4973057
    [teach] ORS  283/966 (29.3%) sid=4974198
    [teach] ORS  284/966 (29.4%) sid=4980920
    [teach] ORS  285/966 (29.5%) sid=4982038
    [teach] ORS  286/966 (29.6%) sid=4983209
    [teach] ORS  287/966 (29.7%) sid=4986773
    [teach] ORS  288/966 (29.8%) sid=4988002
    [teach] ORS  289/966 (29.9%) sid=5019531
    [teach] ORS  290/966 (30.0%) sid=5030895
    [teach] exporting parquet snapshot ...
    [teach] ORS  291/966 (30.1%) sid=5039418
    [teach] ORS  292/966 (30.2%) sid=5040100
    [teach] ORS  293/966 (30.3%) sid=5057855
    [teach] ORS  294/966 (30.4%) sid=5058732
    [teach] ORS  295/966 (30.5%) sid=5079383
    [teach] ORS  296/966 (30.6%) sid=5080852
    [teach] ORS  297/966 (30.7%) sid=5086286
    [teach] ORS  298/966 (30.8%) sid=5088859
    [teach] ORS  299/966 (31.0%) sid=5097344
    [teach] ORS  300/966 (31.1%) sid=5097725
    [teach] exporting parquet snapshot ...
    [teach] ORS  301/966 (31.2%) sid=5097950
    [teach] ORS  302/966 (31.3%) sid=5110948
    [teach] ORS  303/966 (31.4%) sid=5126949
    [teach] ORS  304/966 (31.5%) sid=5127458
    [teach] ORS  305/966 (31.6%) sid=5137866
    [teach] ORS  306/966 (31.7%) sid=5139231
    [teach] ORS  307/966 (31.8%) sid=5143122
    [teach] ORS  308/966 (31.9%) sid=5148163
    [teach] ORS  309/966 (32.0%) sid=5148444
    [teach] ORS  310/966 (32.1%) sid=5153434
    [teach] exporting parquet snapshot ...
    [teach] ORS  311/966 (32.2%) sid=5153456
    [teach] ORS  312/966 (32.3%) sid=5154405
    [teach] ORS  313/966 (32.4%) sid=5169567
    [teach] ORS  314/966 (32.5%) sid=5184041
    [teach] ORS  315/966 (32.6%) sid=5187284
    [teach] ORS  316/966 (32.7%) sid=5190567
    [teach] ORS  317/966 (32.8%) sid=5212624
    [teach] ORS  318/966 (32.9%) sid=5251716
    [teach] ORS  319/966 (33.0%) sid=5258586
    [teach] ORS  320/966 (33.1%) sid=5265338
    [teach] exporting parquet snapshot ...
    [teach] ORS  321/966 (33.2%) sid=5289551
    [teach] ORS  322/966 (33.3%) sid=5339227
    [teach] ORS  323/966 (33.4%) sid=5360278
    [teach] ORS  324/966 (33.5%) sid=5368472
    [teach] ORS  325/966 (33.6%) sid=5376267
    [teach] ORS  326/966 (33.7%) sid=5378202
    [teach] ORS  327/966 (33.9%) sid=5407003
    [teach] ORS  328/966 (34.0%) sid=5415039
    [teach] ORS  329/966 (34.1%) sid=5425765
    [teach] ORS  330/966 (34.2%) sid=5432218
    [teach] exporting parquet snapshot ...
    [teach] ORS  331/966 (34.3%) sid=5466427
    [teach] ORS  332/966 (34.4%) sid=5473797
    [teach] ORS  333/966 (34.5%) sid=5478032
    [teach] ORS  334/966 (34.6%) sid=5492203
    [teach] ORS  335/966 (34.7%) sid=5493525
    [teach] ORS  336/966 (34.8%) sid=5496650
    [teach] ORS  337/966 (34.9%) sid=5521638
    [teach] ORS  338/966 (35.0%) sid=5554428
    [teach] ORS  339/966 (35.1%) sid=5577997
    [teach] ORS  340/966 (35.2%) sid=5597272
    [teach] exporting parquet snapshot ...
    [teach] ORS  341/966 (35.3%) sid=5603003
    [teach] ORS  342/966 (35.4%) sid=5609197
    [teach] ORS  343/966 (35.5%) sid=5634561
    [ORS][warn] sid=5649864 no valid candidates after constraints (mode=dp_prefix, k_span=13..15, dp_q=44, beta=4.0). Trying fallback #1.
    [teach] ORS  344/966 (35.6%) sid=5649864
    [teach] ORS  345/966 (35.7%) sid=5660315
    [teach] ORS  346/966 (35.8%) sid=5673503
    [teach] ORS  347/966 (35.9%) sid=5686796
    [teach] ORS  348/966 (36.0%) sid=5687389
    [teach] ORS  349/966 (36.1%) sid=5698404
    [teach] ORS  350/966 (36.2%) sid=5711492
    [teach] exporting parquet snapshot ...
    [teach] ORS  351/966 (36.3%) sid=5732482
    [teach] ORS  352/966 (36.4%) sid=5734266
    [teach] ORS  353/966 (36.5%) sid=5778107
    [teach] ORS  354/966 (36.6%) sid=5791679
    [teach] ORS  355/966 (36.7%) sid=5795333
    [teach] ORS  356/966 (36.9%) sid=5813334
    [teach] ORS  357/966 (37.0%) sid=5814989
    [teach] ORS  358/966 (37.1%) sid=5823501
    [teach] ORS  359/966 (37.2%) sid=5824034
    [teach] ORS  360/966 (37.3%) sid=5829487
    [teach] exporting parquet snapshot ...
    [teach] ORS  361/966 (37.4%) sid=5845412
    [teach] ORS  362/966 (37.5%) sid=5850371
    [teach] ORS  363/966 (37.6%) sid=5852330
    [teach] ORS  364/966 (37.7%) sid=5858938
    [teach] ORS  365/966 (37.8%) sid=5871769
    [teach] ORS  366/966 (37.9%) sid=5880521
    [teach] ORS  367/966 (38.0%) sid=5907037
    [teach] ORS  368/966 (38.1%) sid=5919254
    [teach] ORS  369/966 (38.2%) sid=5924054
    [teach] ORS  370/966 (38.3%) sid=5933092
    [teach] exporting parquet snapshot ...
    [teach] ORS  371/966 (38.4%) sid=5937754
    [teach] ORS  372/966 (38.5%) sid=5945706
    [teach] ORS  373/966 (38.6%) sid=5987703
    [teach] ORS  374/966 (38.7%) sid=6031843
    [teach] ORS  375/966 (38.8%) sid=6061631
    [teach] ORS  376/966 (38.9%) sid=6074769
    [teach] ORS  377/966 (39.0%) sid=6097211
    [teach] ORS  378/966 (39.1%) sid=6106069
    [teach] ORS  379/966 (39.2%) sid=6106377
    [teach] ORS  380/966 (39.3%) sid=6123781
    [teach] exporting parquet snapshot ...
    [teach] ORS  381/966 (39.4%) sid=6129779
    [teach] ORS  382/966 (39.5%) sid=6174344
    [teach] ORS  383/966 (39.6%) sid=6220336
    [teach] ORS  384/966 (39.8%) sid=6236061
    [teach] ORS  385/966 (39.9%) sid=6257154
    [teach] ORS  386/966 (40.0%) sid=6288636
    [teach] ORS  387/966 (40.1%) sid=6311549
    [teach] ORS  388/966 (40.2%) sid=6322698
    [teach] ORS  389/966 (40.3%) sid=6351033
    [teach] ORS  390/966 (40.4%) sid=6356776
    [teach] exporting parquet snapshot ...
    [teach] ORS  391/966 (40.5%) sid=6362388
    [teach] ORS  392/966 (40.6%) sid=6363419
    [teach] ORS  393/966 (40.7%) sid=6366435
    [teach] ORS  394/966 (40.8%) sid=6373507
    [teach] ORS  395/966 (40.9%) sid=6386378
    [teach] ORS  396/966 (41.0%) sid=6406761
    [teach] ORS  397/966 (41.1%) sid=6408528
    [teach] ORS  398/966 (41.2%) sid=6409686
    [teach] ORS  399/966 (41.3%) sid=6413152
    [teach] ORS  400/966 (41.4%) sid=6413520
    [teach] exporting parquet snapshot ...
    [teach] ORS  401/966 (41.5%) sid=6418610
    [teach] ORS  402/966 (41.6%) sid=6431322
    [teach] ORS  403/966 (41.7%) sid=6437298
    [teach] ORS  404/966 (41.8%) sid=6455445
    [teach] ORS  405/966 (41.9%) sid=6492394
    [teach] ORS  406/966 (42.0%) sid=6495681
    [teach] ORS  407/966 (42.1%) sid=6496954
    [teach] ORS  408/966 (42.2%) sid=6505800
    [teach] ORS  409/966 (42.3%) sid=6508485
    [teach] ORS  410/966 (42.4%) sid=6512761
    [teach] exporting parquet snapshot ...
    [teach] ORS  411/966 (42.5%) sid=6527736
    [teach] ORS  412/966 (42.7%) sid=6534650
    [teach] ORS  413/966 (42.8%) sid=6537802
    [teach] ORS  414/966 (42.9%) sid=6571778
    [teach] ORS  415/966 (43.0%) sid=6573661
    [teach] ORS  416/966 (43.1%) sid=6580285
    [teach] ORS  417/966 (43.2%) sid=6594054
    [teach] ORS  418/966 (43.3%) sid=6596088
    [teach] ORS  419/966 (43.4%) sid=6613180
    [teach] ORS  420/966 (43.5%) sid=6648051
    [teach] exporting parquet snapshot ...
    [teach] ORS  421/966 (43.6%) sid=6676101
    [teach] ORS  422/966 (43.7%) sid=6677278
    [teach] ORS  423/966 (43.8%) sid=6679791
    [teach] ORS  424/966 (43.9%) sid=6681643
    [teach] ORS  425/966 (44.0%) sid=6684959
    [teach] ORS  426/966 (44.1%) sid=6711834
    [teach] ORS  427/966 (44.2%) sid=6721641
    [teach] ORS  428/966 (44.3%) sid=6735439
    [teach] ORS  429/966 (44.4%) sid=6736547
    [teach] ORS  430/966 (44.5%) sid=6738359
    [teach] exporting parquet snapshot ...
    [teach] ORS  431/966 (44.6%) sid=6747437
    [teach] ORS  432/966 (44.7%) sid=6748348
    [ORS][warn] sid=6749919 no valid candidates after constraints (mode=dp_prefix, k_span=1..3, dp_q=9, beta=4.0). Trying fallback #1.
    [ORS][warn] sid=6749919 still no valid candidates (fb#1 k_span=1..3, dp_q=18, beta=16.0). Trying fallback #2 (rdp).
    [ORS][warn] sid=6749919 fallbacks exhausted (rdp k_span=1..8); skipping session.
    [teach][warn] ORS invalid for sid=6749919 (res_not_dict); skipping.
    [teach] ORS  434/966 (44.9%) sid=6776966
    [teach] ORS  435/966 (45.0%) sid=6780248
    [teach] ORS  436/966 (45.1%) sid=6789138
    [teach] ORS  437/966 (45.2%) sid=6811612
    [teach] ORS  438/966 (45.3%) sid=6813569
    [teach] ORS  439/966 (45.4%) sid=6823722
    [teach] ORS  440/966 (45.5%) sid=6824351
    [teach] exporting parquet snapshot ...
    [teach] ORS  441/966 (45.7%) sid=6825354
    [teach] ORS  442/966 (45.8%) sid=6830580
    [teach] ORS  443/966 (45.9%) sid=6833914
    [teach] ORS  444/966 (46.0%) sid=6869457
    [teach] ORS  445/966 (46.1%) sid=6877871
    [teach] ORS  446/966 (46.2%) sid=6878964
    [teach] ORS  447/966 (46.3%) sid=6893908
    [teach] ORS  448/966 (46.4%) sid=6935315
    [teach] ORS  449/966 (46.5%) sid=6945950
    [teach] ORS  450/966 (46.6%) sid=6988499
    [teach] exporting parquet snapshot ...
    [teach] ORS  451/966 (46.7%) sid=7004749
    [teach] ORS  452/966 (46.8%) sid=7014997
    [teach] ORS  453/966 (46.9%) sid=7037725
    [teach] ORS  454/966 (47.0%) sid=7044746
    [teach] ORS  455/966 (47.1%) sid=7046147
    [teach] ORS  456/966 (47.2%) sid=7057639
    [teach] ORS  457/966 (47.3%) sid=7066922
    [teach] ORS  458/966 (47.4%) sid=7080921
    [teach] ORS  459/966 (47.5%) sid=7109843
    [teach] ORS  460/966 (47.6%) sid=7112357
    [teach] exporting parquet snapshot ...
    [teach] ORS  461/966 (47.7%) sid=7122389
    [teach] ORS  462/966 (47.8%) sid=7124465
    [teach] ORS  463/966 (47.9%) sid=7131229
    [teach] ORS  464/966 (48.0%) sid=7134894
    [teach] ORS  465/966 (48.1%) sid=7169280
    [teach] ORS  466/966 (48.2%) sid=7174096
    [teach] ORS  467/966 (48.3%) sid=7209446
    [teach] ORS  468/966 (48.4%) sid=7211821
    [teach] ORS  469/966 (48.6%) sid=7211838
    [teach] ORS  470/966 (48.7%) sid=7213262
    [teach] exporting parquet snapshot ...
    [teach] ORS  471/966 (48.8%) sid=7216489
    [teach] ORS  472/966 (48.9%) sid=7220453
    [teach] ORS  473/966 (49.0%) sid=7225009
    [teach] ORS  474/966 (49.1%) sid=7230626
    [teach] ORS  475/966 (49.2%) sid=7241589
    [teach] ORS  476/966 (49.3%) sid=7246534
    [teach] ORS  477/966 (49.4%) sid=7258277
    [teach] ORS  478/966 (49.5%) sid=7312689
    [teach] ORS  479/966 (49.6%) sid=7312898
    [teach] ORS  480/966 (49.7%) sid=7330343
    [teach] exporting parquet snapshot ...
    [teach] ORS  481/966 (49.8%) sid=7330749
    [teach] ORS  482/966 (49.9%) sid=7360374
    [teach] ORS  483/966 (50.0%) sid=7364030
    [teach] ORS  484/966 (50.1%) sid=7367260
    [teach] ORS  485/966 (50.2%) sid=7372716
    [teach] ORS  486/966 (50.3%) sid=7374959
    [teach] ORS  487/966 (50.4%) sid=7378822
    [teach] ORS  488/966 (50.5%) sid=7381648
    [teach] ORS  489/966 (50.6%) sid=7393270
    [teach] ORS  490/966 (50.7%) sid=7403290
    [teach] exporting parquet snapshot ...
    [teach] ORS  491/966 (50.8%) sid=7408373
    [teach] ORS  492/966 (50.9%) sid=7414065
    [teach] ORS  493/966 (51.0%) sid=7420252
    [teach] ORS  494/966 (51.1%) sid=7482633
    [teach] ORS  495/966 (51.2%) sid=7518853
    [teach] ORS  496/966 (51.3%) sid=7525040
    [teach] ORS  497/966 (51.4%) sid=7540722
    [teach] ORS  498/966 (51.6%) sid=7547343
    [teach] ORS  499/966 (51.7%) sid=7548077
    [ORS][warn] sid=7551082 no valid candidates after constraints (mode=dp_prefix, k_span=4..7, dp_q=47, beta=4.0). Trying fallback #1.
    [teach] ORS  500/966 (51.8%) sid=7551082
    [teach] exporting parquet snapshot ...
    [teach] ORS  501/966 (51.9%) sid=7573935
    [teach] ORS  502/966 (52.0%) sid=7580799
    [teach] ORS  503/966 (52.1%) sid=7583288
    [teach] ORS  504/966 (52.2%) sid=7587787
    [teach] ORS  505/966 (52.3%) sid=7604237
    [teach] ORS  506/966 (52.4%) sid=7609062
    [teach] ORS  507/966 (52.5%) sid=7638861
    [teach] ORS  508/966 (52.6%) sid=7693025
    [teach] ORS  509/966 (52.7%) sid=7700810
    [teach] ORS  510/966 (52.8%) sid=7701819
    [teach] exporting parquet snapshot ...
    [teach] ORS  511/966 (52.9%) sid=7704522
    [teach] ORS  512/966 (53.0%) sid=7714277
    [teach] ORS  513/966 (53.1%) sid=7715555
    [teach] ORS  514/966 (53.2%) sid=7720133
    [teach] ORS  515/966 (53.3%) sid=7722025
    [teach] ORS  516/966 (53.4%) sid=7723314
    [teach] ORS  517/966 (53.5%) sid=7727209
    [teach] ORS  518/966 (53.6%) sid=7727681
    [teach] ORS  519/966 (53.7%) sid=7728026
    [teach] ORS  520/966 (53.8%) sid=7729981
    [teach] exporting parquet snapshot ...
    [teach] ORS  521/966 (53.9%) sid=7736569
    [teach] ORS  522/966 (54.0%) sid=7741850
    [teach] ORS  523/966 (54.1%) sid=7753775
    [teach] ORS  524/966 (54.2%) sid=7766446
    [teach] ORS  525/966 (54.3%) sid=7776575
    [teach] ORS  526/966 (54.5%) sid=7778309
    [teach] ORS  527/966 (54.6%) sid=7786552
    [teach] ORS  528/966 (54.7%) sid=7786991
    [teach] ORS  529/966 (54.8%) sid=7787953
    [teach] ORS  530/966 (54.9%) sid=7789717
    [teach] exporting parquet snapshot ...
    [teach] ORS  531/966 (55.0%) sid=7795787
    [teach] ORS  532/966 (55.1%) sid=7811536
    [teach] ORS  533/966 (55.2%) sid=7812714
    [teach] ORS  534/966 (55.3%) sid=7813535
    [teach] ORS  535/966 (55.4%) sid=7816707
    [teach] ORS  536/966 (55.5%) sid=7820673
    [teach] ORS  537/966 (55.6%) sid=7821800
    [teach] ORS  538/966 (55.7%) sid=7829172
    [teach] ORS  539/966 (55.8%) sid=7833577
    [teach] ORS  540/966 (55.9%) sid=7838923
    [teach] exporting parquet snapshot ...
    [teach] ORS  541/966 (56.0%) sid=7839408
    [teach] ORS  542/966 (56.1%) sid=7839751
    [teach] ORS  543/966 (56.2%) sid=7844933
    [teach] ORS  544/966 (56.3%) sid=7870600
    [teach] ORS  545/966 (56.4%) sid=7878196
    [teach] ORS  546/966 (56.5%) sid=7881836
    [teach] ORS  547/966 (56.6%) sid=7885527
    [teach] ORS  548/966 (56.7%) sid=7893168
    [teach] ORS  549/966 (56.8%) sid=7901433
    [teach] ORS  550/966 (56.9%) sid=7918634
    [teach] exporting parquet snapshot ...
    [teach] ORS  551/966 (57.0%) sid=7923282
    [teach] ORS  552/966 (57.1%) sid=7929243
    [teach] ORS  553/966 (57.2%) sid=7930828
    [teach] ORS  554/966 (57.3%) sid=7935375
    [teach] ORS  555/966 (57.5%) sid=7935455
    [teach] ORS  556/966 (57.6%) sid=7938356
    [teach] ORS  557/966 (57.7%) sid=7948362
    [teach] ORS  558/966 (57.8%) sid=7949006
    [teach] ORS  559/966 (57.9%) sid=7955030
    [teach] ORS  560/966 (58.0%) sid=7956593
    [teach] exporting parquet snapshot ...
    [teach] ORS  561/966 (58.1%) sid=7965585
    [teach] ORS  562/966 (58.2%) sid=7970522
    [teach] ORS  563/966 (58.3%) sid=7973585
    [teach] ORS  564/966 (58.4%) sid=7980429
    [teach] ORS  565/966 (58.5%) sid=7985148
    [teach] ORS  566/966 (58.6%) sid=7987503
    [teach] ORS  567/966 (58.7%) sid=7990289
    [teach] ORS  568/966 (58.8%) sid=8003254
    [teach] ORS  569/966 (58.9%) sid=8004031
    [teach] ORS  570/966 (59.0%) sid=8013698
    [teach] exporting parquet snapshot ...
    [teach] ORS  571/966 (59.1%) sid=8015343
    [teach] ORS  572/966 (59.2%) sid=8026139
    [teach] ORS  573/966 (59.3%) sid=8031653
    [teach] ORS  574/966 (59.4%) sid=8051526
    [teach] ORS  575/966 (59.5%) sid=8055151
    [teach] ORS  576/966 (59.6%) sid=8061977
    [teach] ORS  577/966 (59.7%) sid=8066498
    [teach] ORS  578/966 (59.8%) sid=8077204
    [teach] ORS  579/966 (59.9%) sid=8105362
    [teach] ORS  580/966 (60.0%) sid=8107395
    [teach] exporting parquet snapshot ...
    [teach] ORS  581/966 (60.1%) sid=8110415
    [teach] ORS  582/966 (60.2%) sid=8116330
    [teach] ORS  583/966 (60.4%) sid=8138518
    [teach] ORS  584/966 (60.5%) sid=8138707
    [teach] ORS  585/966 (60.6%) sid=8141456
    [teach] ORS  586/966 (60.7%) sid=8144234
    [teach] ORS  587/966 (60.8%) sid=8192039
    [teach] ORS  588/966 (60.9%) sid=8203265
    [teach] ORS  589/966 (61.0%) sid=8207664
    [teach] ORS  590/966 (61.1%) sid=8209840
    [teach] exporting parquet snapshot ...
    [teach] ORS  591/966 (61.2%) sid=8218338
    [teach] ORS  592/966 (61.3%) sid=8219769
    [teach] ORS  593/966 (61.4%) sid=8247874
    [teach] ORS  594/966 (61.5%) sid=8250419
    [teach] ORS  595/966 (61.6%) sid=8253823
    [teach] ORS  596/966 (61.7%) sid=8269301
    [teach] ORS  597/966 (61.8%) sid=8276321
    [teach] ORS  598/966 (61.9%) sid=8279809
    [teach] ORS  599/966 (62.0%) sid=8280108
    [teach] ORS  600/966 (62.1%) sid=8290277
    [teach] exporting parquet snapshot ...
    [teach] ORS  601/966 (62.2%) sid=8292678
    [teach] ORS  602/966 (62.3%) sid=8325089
    [teach] ORS  603/966 (62.4%) sid=8356609
    [teach] ORS  604/966 (62.5%) sid=8359613
    [teach] ORS  605/966 (62.6%) sid=8360139
    [teach] ORS  606/966 (62.7%) sid=8367305
    [teach] ORS  607/966 (62.8%) sid=8367379
    [teach] ORS  608/966 (62.9%) sid=8375951
    [teach] ORS  609/966 (63.0%) sid=8376958
    [teach] ORS  610/966 (63.1%) sid=8384400
    [teach] exporting parquet snapshot ...
    [teach] ORS  611/966 (63.3%) sid=8389908
    [teach] ORS  612/966 (63.4%) sid=8430580
    [teach] ORS  613/966 (63.5%) sid=8439412
    [teach] ORS  614/966 (63.6%) sid=8458199
    [teach] ORS  615/966 (63.7%) sid=8458460
    [teach] ORS  616/966 (63.8%) sid=8477590
    [teach] ORS  617/966 (63.9%) sid=8484901
    [teach] ORS  618/966 (64.0%) sid=8493177
    [teach] ORS  619/966 (64.1%) sid=8496286
    [teach] ORS  620/966 (64.2%) sid=8507916
    [teach] exporting parquet snapshot ...
    [teach] ORS  621/966 (64.3%) sid=8534928
    [teach] ORS  622/966 (64.4%) sid=8537433
    [teach] ORS  623/966 (64.5%) sid=8553459
    [teach] ORS  624/966 (64.6%) sid=8568294
    [teach] ORS  625/966 (64.7%) sid=8584670
    [teach] ORS  626/966 (64.8%) sid=8593917
    [teach] ORS  627/966 (64.9%) sid=8630437
    [teach] ORS  628/966 (65.0%) sid=8640247
    [teach] ORS  629/966 (65.1%) sid=8647429
    [teach] ORS  630/966 (65.2%) sid=8647914
    [teach] exporting parquet snapshot ...
    [teach] ORS  631/966 (65.3%) sid=8659962
    [teach] ORS  632/966 (65.4%) sid=8661444
    [teach] ORS  633/966 (65.5%) sid=8661544
    [teach] ORS  634/966 (65.6%) sid=8701440
    [teach] ORS  635/966 (65.7%) sid=8732902
    [teach] ORS  636/966 (65.8%) sid=8736448
    [teach] ORS  637/966 (65.9%) sid=8738931
    [teach] ORS  638/966 (66.0%) sid=8743262
    [teach] ORS  639/966 (66.1%) sid=8745967
    [teach] ORS  640/966 (66.3%) sid=8753555
    [teach] exporting parquet snapshot ...
    [teach] ORS  641/966 (66.4%) sid=8764504
    [teach] ORS  642/966 (66.5%) sid=8798239
    [teach] ORS  643/966 (66.6%) sid=8803593
    [teach] ORS  644/966 (66.7%) sid=8812531
    [teach] ORS  645/966 (66.8%) sid=8845628
    [teach] ORS  646/966 (66.9%) sid=8858057
    [teach] ORS  647/966 (67.0%) sid=8886299
    [teach] ORS  648/966 (67.1%) sid=8899670
    [teach] ORS  649/966 (67.2%) sid=8904990
    [teach] ORS  650/966 (67.3%) sid=8979512
    [teach] exporting parquet snapshot ...
    [teach] ORS  651/966 (67.4%) sid=8986981
    [teach] ORS  652/966 (67.5%) sid=9002133
    [teach] ORS  653/966 (67.6%) sid=9022843
    [teach] ORS  654/966 (67.7%) sid=9054685
    [teach] ORS  655/966 (67.8%) sid=9054757
    [teach] ORS  656/966 (67.9%) sid=9061412
    [teach] ORS  657/966 (68.0%) sid=9074084
    [teach] ORS  658/966 (68.1%) sid=9080913
    [teach] ORS  659/966 (68.2%) sid=9081225
    [teach] ORS  660/966 (68.3%) sid=9144525
    [teach] exporting parquet snapshot ...
    [teach] ORS  661/966 (68.4%) sid=9166867
    [teach] ORS  662/966 (68.5%) sid=9189528
    [teach] ORS  663/966 (68.6%) sid=9198410
    [teach] ORS  664/966 (68.7%) sid=9216134
    [teach] ORS  665/966 (68.8%) sid=9245905
    [teach] ORS  666/966 (68.9%) sid=9260929
    [teach] ORS  667/966 (69.0%) sid=9276203
    [teach] ORS  668/966 (69.2%) sid=9276930
    [teach] ORS  669/966 (69.3%) sid=9278188
    [teach] ORS  670/966 (69.4%) sid=9280905
    [teach] exporting parquet snapshot ...
    [teach] ORS  671/966 (69.5%) sid=9298757
    [teach] ORS  672/966 (69.6%) sid=9312099
    [teach] ORS  673/966 (69.7%) sid=9332240
    [teach] ORS  674/966 (69.8%) sid=9352936
    [teach] ORS  675/966 (69.9%) sid=9385410
    [teach] ORS  676/966 (70.0%) sid=9441730
    [teach] ORS  677/966 (70.1%) sid=9453967
    [teach] ORS  678/966 (70.2%) sid=9456943
    [teach] ORS  679/966 (70.3%) sid=9460546
    [teach] ORS  680/966 (70.4%) sid=9474787
    [teach] exporting parquet snapshot ...
    [teach] ORS  681/966 (70.5%) sid=9489647
    [teach] ORS  682/966 (70.6%) sid=9552137
    [teach] ORS  683/966 (70.7%) sid=9558889
    [teach] ORS  684/966 (70.8%) sid=9584973
    [teach] ORS  685/966 (70.9%) sid=9631049
    [teach] ORS  686/966 (71.0%) sid=9651230
    [teach] ORS  687/966 (71.1%) sid=9687533
    [teach] ORS  688/966 (71.2%) sid=9719898
    [teach] ORS  689/966 (71.3%) sid=9760492
    [teach] ORS  690/966 (71.4%) sid=9764861
    [teach] exporting parquet snapshot ...
    [teach] ORS  691/966 (71.5%) sid=9780281
    [teach] ORS  692/966 (71.6%) sid=9784944
    [teach] ORS  693/966 (71.7%) sid=9790972
    [teach] ORS  694/966 (71.8%) sid=9792022
    [teach] ORS  695/966 (71.9%) sid=9812424
    [teach] ORS  696/966 (72.0%) sid=9827139
    [teach] ORS  697/966 (72.2%) sid=9834543
    [teach] ORS  698/966 (72.3%) sid=9834959
    [teach] ORS  699/966 (72.4%) sid=9838619
    [teach] ORS  700/966 (72.5%) sid=9892756
    [teach] exporting parquet snapshot ...
    [teach] ORS  701/966 (72.6%) sid=9923355
    [teach] ORS  702/966 (72.7%) sid=9926590
    [teach] ORS  703/966 (72.8%) sid=9926849
    [teach] ORS  704/966 (72.9%) sid=9941191
    [teach] ORS  705/966 (73.0%) sid=9942273
    [teach] ORS  706/966 (73.1%) sid=9959922
    [teach] ORS  707/966 (73.2%) sid=9997665
    [teach] ORS  708/966 (73.3%) sid=10003312
    [teach] ORS  709/966 (73.4%) sid=10119292
    [teach] ORS  710/966 (73.5%) sid=10125147
    [teach] exporting parquet snapshot ...
    [teach] ORS  711/966 (73.6%) sid=10130352
    [teach] ORS  712/966 (73.7%) sid=10139844
    [teach] ORS  713/966 (73.8%) sid=10204705
    [teach] ORS  714/966 (73.9%) sid=10229995
    [teach] ORS  715/966 (74.0%) sid=10232151
    [teach] ORS  716/966 (74.1%) sid=10237658
    [teach] ORS  717/966 (74.2%) sid=10239076
    [teach] ORS  718/966 (74.3%) sid=10241481
    [ORS][warn] sid=10244968 no valid candidates after constraints (mode=dp_prefix, k_span=4..5, dp_q=33, beta=4.0). Trying fallback #1.
    [teach] ORS  719/966 (74.4%) sid=10244968
    [teach] ORS  720/966 (74.5%) sid=10248840
    [teach] exporting parquet snapshot ...
    [teach] ORS  721/966 (74.6%) sid=10257721
    [teach] ORS  722/966 (74.7%) sid=10261424
    [teach] ORS  723/966 (74.8%) sid=10266047
    [teach] ORS  724/966 (74.9%) sid=10285817
    [teach] ORS  725/966 (75.1%) sid=10287456
    [teach] ORS  726/966 (75.2%) sid=10298910
    [teach] ORS  727/966 (75.3%) sid=10312952
    [teach] ORS  728/966 (75.4%) sid=10313357
    [teach] ORS  729/966 (75.5%) sid=10334040
    [teach] ORS  730/966 (75.6%) sid=10350999
    [teach] exporting parquet snapshot ...
    [teach] ORS  731/966 (75.7%) sid=10352883
    [teach] ORS  732/966 (75.8%) sid=10353659
    [teach] ORS  733/966 (75.9%) sid=10362279
    [teach] ORS  734/966 (76.0%) sid=10367206
    [teach] ORS  735/966 (76.1%) sid=10376385
    [teach] ORS  736/966 (76.2%) sid=10381532
    [teach] ORS  737/966 (76.3%) sid=10388728
    [teach] ORS  738/966 (76.4%) sid=10393598
    [teach] ORS  739/966 (76.5%) sid=10393637
    [teach] ORS  740/966 (76.6%) sid=10399456
    [teach] exporting parquet snapshot ...
    [teach] ORS  741/966 (76.7%) sid=10415762
    [teach] ORS  742/966 (76.8%) sid=10419548
    [teach] ORS  743/966 (76.9%) sid=10424948
    [teach] ORS  744/966 (77.0%) sid=10491369
    [teach] ORS  745/966 (77.1%) sid=10500957
    [teach] ORS  746/966 (77.2%) sid=10509709
    [teach] ORS  747/966 (77.3%) sid=10515875
    [teach] ORS  748/966 (77.4%) sid=10525015
    [teach] ORS  749/966 (77.5%) sid=10536320
    [teach] ORS  750/966 (77.6%) sid=10544732
    [teach] exporting parquet snapshot ...
    [teach] ORS  751/966 (77.7%) sid=10566178
    [teach] ORS  752/966 (77.8%) sid=10566424
    [teach] ORS  753/966 (78.0%) sid=10569669
    [teach] ORS  754/966 (78.1%) sid=10570040
    [teach] ORS  755/966 (78.2%) sid=10599100
    [teach] ORS  756/966 (78.3%) sid=10629196
    [teach] ORS  757/966 (78.4%) sid=10636801
    [teach] ORS  758/966 (78.5%) sid=10638109
    [teach] ORS  759/966 (78.6%) sid=10644767
    [teach] ORS  760/966 (78.7%) sid=10649682
    [teach] exporting parquet snapshot ...
    [teach] ORS  761/966 (78.8%) sid=10651719
    [teach] ORS  762/966 (78.9%) sid=10671959
    [teach] ORS  763/966 (79.0%) sid=10677511
    [teach] ORS  764/966 (79.1%) sid=10680807
    [teach] ORS  765/966 (79.2%) sid=10685489
    [teach] ORS  766/966 (79.3%) sid=10687428
    [teach] ORS  767/966 (79.4%) sid=10704592
    [teach] ORS  768/966 (79.5%) sid=10706141
    [teach] ORS  769/966 (79.6%) sid=10706991
    [teach] ORS  770/966 (79.7%) sid=10707331
    [teach] exporting parquet snapshot ...
    [teach] ORS  771/966 (79.8%) sid=10708831
    [teach] ORS  772/966 (79.9%) sid=10710896
    [teach] ORS  773/966 (80.0%) sid=10712327
    [teach] ORS  774/966 (80.1%) sid=10745510
    [teach] ORS  775/966 (80.2%) sid=10746543
    [teach] ORS  776/966 (80.3%) sid=10752455
    [teach] ORS  777/966 (80.4%) sid=10769551
    [teach] ORS  778/966 (80.5%) sid=10784762
    [teach] ORS  779/966 (80.6%) sid=10794496
    [teach] ORS  780/966 (80.7%) sid=10818869
    [teach] exporting parquet snapshot ...
    [teach] ORS  781/966 (80.8%) sid=10831918
    [teach] ORS  782/966 (81.0%) sid=10836192
    [teach] ORS  783/966 (81.1%) sid=10846276
    [teach] ORS  784/966 (81.2%) sid=10850895
    [teach] ORS  785/966 (81.3%) sid=10855309
    [teach] ORS  786/966 (81.4%) sid=10923529
    [teach] ORS  787/966 (81.5%) sid=10927300
    [teach] ORS  788/966 (81.6%) sid=10927496
    [teach] ORS  789/966 (81.7%) sid=10943656
    [teach] ORS  790/966 (81.8%) sid=10944077
    [teach] exporting parquet snapshot ...
    [teach] ORS  791/966 (81.9%) sid=10967308
    [teach] ORS  792/966 (82.0%) sid=10975698
    [teach] ORS  793/966 (82.1%) sid=10989111
    [teach] ORS  794/966 (82.2%) sid=11002629
    [teach] ORS  795/966 (82.3%) sid=11012229
    [teach] ORS  796/966 (82.4%) sid=11018700
    [teach] ORS  797/966 (82.5%) sid=11025567
    [teach] ORS  798/966 (82.6%) sid=11030010
    [teach] ORS  799/966 (82.7%) sid=11035355
    [teach] ORS  800/966 (82.8%) sid=11045844
    [teach] exporting parquet snapshot ...
    [teach] ORS  801/966 (82.9%) sid=11058367
    [teach] ORS  802/966 (83.0%) sid=11078849
    [teach] ORS  803/966 (83.1%) sid=11081105
    [teach] ORS  804/966 (83.2%) sid=11112614
    [teach] ORS  805/966 (83.3%) sid=11155024
    [teach] ORS  806/966 (83.4%) sid=11156542
    [teach] ORS  807/966 (83.5%) sid=11221009
    [teach] ORS  808/966 (83.6%) sid=11229586
    [teach] ORS  809/966 (83.7%) sid=11234740
    [teach] ORS  810/966 (83.9%) sid=11284135
    [teach] exporting parquet snapshot ...
    [teach] ORS  811/966 (84.0%) sid=11294478
    [teach] ORS  812/966 (84.1%) sid=11341687
    [teach] ORS  813/966 (84.2%) sid=11346859
    [teach] ORS  814/966 (84.3%) sid=11352856
    [teach] ORS  815/966 (84.4%) sid=11371670
    [teach] ORS  816/966 (84.5%) sid=11376100
    [teach] ORS  817/966 (84.6%) sid=11388952
    [teach] ORS  818/966 (84.7%) sid=11396070
    [teach] ORS  819/966 (84.8%) sid=11396701
    [teach] ORS  820/966 (84.9%) sid=11397514
    [teach] exporting parquet snapshot ...
    [teach] ORS  821/966 (85.0%) sid=11404889
    [teach] ORS  822/966 (85.1%) sid=11408024
    [teach] ORS  823/966 (85.2%) sid=11417451
    [teach] ORS  824/966 (85.3%) sid=11440124
    [teach] ORS  825/966 (85.4%) sid=11445515
    [teach] ORS  826/966 (85.5%) sid=11447384
    [teach] ORS  827/966 (85.6%) sid=11462386
    [teach] ORS  828/966 (85.7%) sid=11469756
    [teach] ORS  829/966 (85.8%) sid=11470568
    [teach] ORS  830/966 (85.9%) sid=11482298
    [teach] exporting parquet snapshot ...
    [teach] ORS  831/966 (86.0%) sid=11483697
    [teach] ORS  832/966 (86.1%) sid=11487147
    [teach] ORS  833/966 (86.2%) sid=11497324
    [teach] ORS  834/966 (86.3%) sid=11513099
    [teach] ORS  835/966 (86.4%) sid=11519799
    [teach] ORS  836/966 (86.5%) sid=11523452
    [teach] ORS  837/966 (86.6%) sid=11526118
    [teach] ORS  838/966 (86.7%) sid=11527264
    [teach] ORS  839/966 (86.9%) sid=11528426
    [teach] ORS  840/966 (87.0%) sid=11530012
    [teach] exporting parquet snapshot ...
    [teach] ORS  841/966 (87.1%) sid=11534915
    [teach] ORS  842/966 (87.2%) sid=11535753
    [teach] ORS  843/966 (87.3%) sid=11540931
    [teach] ORS  844/966 (87.4%) sid=11546798
    [teach] ORS  845/966 (87.5%) sid=11547678
    [teach] ORS  846/966 (87.6%) sid=11548856
    [teach] ORS  847/966 (87.7%) sid=11550990
    [teach] ORS  848/966 (87.8%) sid=11559051
    [teach] ORS  849/966 (87.9%) sid=11566365
    [teach] ORS  850/966 (88.0%) sid=11574745
    [teach] exporting parquet snapshot ...
    [teach] ORS  851/966 (88.1%) sid=11577696
    [teach] ORS  852/966 (88.2%) sid=11589474
    [teach] ORS  853/966 (88.3%) sid=11590286
    [teach] ORS  854/966 (88.4%) sid=11591128
    [teach] ORS  855/966 (88.5%) sid=11592638
    [teach] ORS  856/966 (88.6%) sid=11595059
    [teach] ORS  857/966 (88.7%) sid=11601924
    [teach] ORS  858/966 (88.8%) sid=11607358
    [teach] ORS  859/966 (88.9%) sid=11610725
    [teach] ORS  860/966 (89.0%) sid=11613979
    [teach] exporting parquet snapshot ...
    [teach] ORS  861/966 (89.1%) sid=11621880
    [teach] ORS  862/966 (89.2%) sid=11628423
    [teach] ORS  863/966 (89.3%) sid=11635547
    [teach] ORS  864/966 (89.4%) sid=11640731
    [teach] ORS  865/966 (89.5%) sid=11644138
    [teach] ORS  866/966 (89.6%) sid=11657374
    [teach] ORS  867/966 (89.8%) sid=11660574
    [teach] ORS  868/966 (89.9%) sid=11665555
    [teach] ORS  869/966 (90.0%) sid=11668248
    [teach] ORS  870/966 (90.1%) sid=11687255
    [teach] exporting parquet snapshot ...
    [teach] ORS  871/966 (90.2%) sid=11689627
    [teach] ORS  872/966 (90.3%) sid=11689761
    [teach] ORS  873/966 (90.4%) sid=11690687
    [teach] ORS  874/966 (90.5%) sid=11691589
    [teach] ORS  875/966 (90.6%) sid=11700738
    [teach] ORS  876/966 (90.7%) sid=11703904
    [teach] ORS  877/966 (90.8%) sid=11705806
    [teach] ORS  878/966 (90.9%) sid=11713457
    [teach] ORS  879/966 (91.0%) sid=11725251
    [teach] ORS  880/966 (91.1%) sid=11728077
    [teach] exporting parquet snapshot ...
    [teach] ORS  881/966 (91.2%) sid=11730271
    [teach] ORS  882/966 (91.3%) sid=11732342
    [teach] ORS  883/966 (91.4%) sid=11741817
    [teach] ORS  884/966 (91.5%) sid=11742248
    [teach] ORS  885/966 (91.6%) sid=11745153
    [teach] ORS  886/966 (91.7%) sid=11758004
    [teach] ORS  887/966 (91.8%) sid=11766550
    [teach] ORS  888/966 (91.9%) sid=11770967
    [teach] ORS  889/966 (92.0%) sid=11781698
    [teach] ORS  890/966 (92.1%) sid=11789188
    [teach] exporting parquet snapshot ...
    [teach] ORS  891/966 (92.2%) sid=11796634
    [teach] ORS  892/966 (92.3%) sid=11807865
    [teach] ORS  893/966 (92.4%) sid=11820631
    [teach] ORS  894/966 (92.5%) sid=11843156
    [teach] ORS  895/966 (92.7%) sid=11844329
    [teach] ORS  896/966 (92.8%) sid=11870677
    [teach] ORS  897/966 (92.9%) sid=11885295
    [teach] ORS  898/966 (93.0%) sid=11887443
    [teach] ORS  899/966 (93.1%) sid=11895715
    [teach] ORS  900/966 (93.2%) sid=11896790
    [teach] exporting parquet snapshot ...
    [teach] ORS  901/966 (93.3%) sid=11907569
    [teach] ORS  902/966 (93.4%) sid=11909667
    [teach] ORS  903/966 (93.5%) sid=11934921
    [teach] ORS  904/966 (93.6%) sid=11937188
    [teach] ORS  905/966 (93.7%) sid=11942751
    [teach] ORS  906/966 (93.8%) sid=11951835
    [teach] ORS  907/966 (93.9%) sid=11953071
    [teach] ORS  908/966 (94.0%) sid=11956008
    [teach] ORS  909/966 (94.1%) sid=11975700
    [teach] ORS  910/966 (94.2%) sid=11997964
    [teach] exporting parquet snapshot ...
    [teach] ORS  911/966 (94.3%) sid=12034414
    [teach] ORS  912/966 (94.4%) sid=12046893
    [teach] ORS  913/966 (94.5%) sid=12050371
    [teach] ORS  914/966 (94.6%) sid=12051063
    [teach] ORS  915/966 (94.7%) sid=12058611
    [teach] ORS  916/966 (94.8%) sid=12059264
    [teach] ORS  917/966 (94.9%) sid=12063051
    [teach] ORS  918/966 (95.0%) sid=12079027
    [teach] ORS  919/966 (95.1%) sid=12097473
    [teach] ORS  920/966 (95.2%) sid=12101609
    [teach] exporting parquet snapshot ...
    [teach] ORS  921/966 (95.3%) sid=12109803
    [teach] ORS  922/966 (95.4%) sid=12117600
    [teach] ORS  923/966 (95.5%) sid=12134403
    [teach] ORS  924/966 (95.7%) sid=12145955
    [teach] ORS  925/966 (95.8%) sid=12174756
    [teach] ORS  926/966 (95.9%) sid=12178898
    [teach] ORS  927/966 (96.0%) sid=12187922
    [teach] ORS  928/966 (96.1%) sid=12193183
    [teach] ORS  929/966 (96.2%) sid=12195263
    [teach] ORS  930/966 (96.3%) sid=12195589
    [teach] exporting parquet snapshot ...
    [teach] ORS  931/966 (96.4%) sid=12211585
    [teach] ORS  932/966 (96.5%) sid=12211905
    [teach] ORS  933/966 (96.6%) sid=12215667
    [teach] ORS  934/966 (96.7%) sid=12278141
    [teach] ORS  935/966 (96.8%) sid=12279812
    [teach] ORS  936/966 (96.9%) sid=12289381
    [teach] ORS  937/966 (97.0%) sid=12300625
    [teach] ORS  938/966 (97.1%) sid=12302930
    [teach] ORS  939/966 (97.2%) sid=12303176
    [teach] ORS  940/966 (97.3%) sid=12303502
    [teach] exporting parquet snapshot ...
    [teach] ORS  941/966 (97.4%) sid=12321524
    [teach] ORS  942/966 (97.5%) sid=12330040
    [teach] ORS  943/966 (97.6%) sid=12331745
    [teach] ORS  944/966 (97.7%) sid=12332308
    [teach] ORS  945/966 (97.8%) sid=12380386
    [teach] ORS  946/966 (97.9%) sid=12401718
    [teach] ORS  947/966 (98.0%) sid=12423834
    [teach] ORS  948/966 (98.1%) sid=12427725
    [teach] ORS  949/966 (98.2%) sid=12427932
    [teach] ORS  950/966 (98.3%) sid=12438831
    [teach] exporting parquet snapshot ...
    [teach] ORS  951/966 (98.4%) sid=12441787
    [teach] ORS  952/966 (98.6%) sid=12465020
    [teach] ORS  953/966 (98.7%) sid=12473317
    [teach] ORS  954/966 (98.8%) sid=12479101
    [teach] ORS  955/966 (98.9%) sid=12505559
    [teach] ORS  956/966 (99.0%) sid=12527142
    [teach] ORS  957/966 (99.1%) sid=12543963
    [teach] ORS  958/966 (99.2%) sid=12556655
    [teach] ORS  959/966 (99.3%) sid=12566912
    [teach] ORS  960/966 (99.4%) sid=12584175
    [teach] exporting parquet snapshot ...
    [teach] ORS  961/966 (99.5%) sid=12590071
    [teach] ORS  962/966 (99.6%) sid=12608409
    [teach] ORS  963/966 (99.7%) sid=12612355
    [teach] ORS  964/966 (99.8%) sid=12637759
    [teach] ORS  965/966 (99.9%) sid=12645343
    [teach] ORS  966/966 (100.0%) sid=12645758
    [teach] exporting parquet snapshot ...
    [teach] done. processed 966 sessions in 11 h 25 m.
    parquet: /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/pool.parquet


#### 2.2 - Teaching pool summary statistics


```python
# Summary statistics
pool.describe()
display(pool.pool_df.head())
print("Frequencies of k for normal sessions in the teaching pool:")
print(pool.pool_df[pool.pool_df["label_text"] == "normal"]["k"].value_counts())
print("Frequencies of k for abnormal sessions in the teaching pool:")
print(pool.pool_df[pool.pool_df["label_text"] == "abnormal"]["k"].value_counts())
```

    [teaching pool] rows=1171, classes:
    label_int
    0    586
    1    585
    
    [k] stats:
     count    1171.000000
    mean        3.121264
    std         1.530538
    min         1.000000
    25%         2.000000
    50%         3.000000
    75%         4.000000
    max        12.000000



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
      <th>session_id</th>
      <th>label_text</th>
      <th>label_int</th>
      <th>k</th>
      <th>err</th>
      <th>frag</th>
      <th>robust_prob</th>
      <th>margin</th>
      <th>threshold</th>
      <th>model_id</th>
      <th>ts_unix</th>
      <th>sts_full_path</th>
      <th>piv_path</th>
      <th>emb_dim</th>
      <th>emb</th>
      <th>raw_power_path</th>
      <th>raw_soc_path</th>
      <th>piv_soc_path</th>
      <th>sts_soc_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>137598</td>
      <td>normal</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.265889</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>8.330311</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.762027e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/137...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/137598.npy</td>
      <td>264</td>
      <td>[1.0127782821655273, 0.9989215731620789, 0.9850648045539856, 0.9712080359458923, 0.9573512673377991, 0.9434944987297...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/13...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/1375...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/1375...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/1375...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>167874</td>
      <td>abnormal</td>
      <td>1</td>
      <td>3.0</td>
      <td>13.232937</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.636737</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.762027e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/167...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/167874.npy</td>
      <td>264</td>
      <td>[0.417507529258728, 0.41868191957473755, 0.41985630989074707, 0.4210307002067566, 0.4222050905227661, 0.423379451036...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/16...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/1678...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/1678...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/1678...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>231570</td>
      <td>abnormal</td>
      <td>1</td>
      <td>2.0</td>
      <td>9.314174</td>
      <td>0.133000</td>
      <td>0.867000</td>
      <td>0.717974</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.762027e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/231...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/231570.npy</td>
      <td>264</td>
      <td>[-0.22393187880516052, -0.21197888255119324, -0.20002590119838715, -0.18807290494441986, -0.17611990869045258, -0.16...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/23...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/2315...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/2315...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/2315...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240316</td>
      <td>normal</td>
      <td>0</td>
      <td>1.0</td>
      <td>5.543470</td>
      <td>0.212667</td>
      <td>0.787333</td>
      <td>3.052730</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.762027e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/240...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/240316.npy</td>
      <td>264</td>
      <td>[1.7185717821121216, 1.6915076971054077, 1.6644434928894043, 1.6373794078826904, 1.610315203666687, 1.58325111865997...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/24...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/2403...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/2403...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/2403...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>248477</td>
      <td>abnormal</td>
      <td>1</td>
      <td>4.0</td>
      <td>8.690758</td>
      <td>0.278000</td>
      <td>0.722000</td>
      <td>0.094558</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.762027e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/248...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/248477.npy</td>
      <td>264</td>
      <td>[2.676745891571045, 2.6957359313964844, 2.7147257328033447, 2.733715772628784, 2.7527055740356445, 2.771695613861084...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/24...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/2484...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/2484...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/2484...</td>
    </tr>
  </tbody>
</table>
</div>


    Frequencies of k for normal sessions in the teaching pool:
    k
    2.0    455
    3.0     69
    1.0     31
    4.0     30
    5.0      1
    Name: count, dtype: int64
    Frequencies of k for abnormal sessions in the teaching pool:
    k
    3.0     220
    5.0     115
    4.0     109
    6.0      58
    2.0      45
    7.0      12
    8.0       9
    1.0       6
    9.0       5
    10.0      2
    12.0      2
    11.0      2
    Name: count, dtype: int64


#### 2.3 - Binning the pool by class and $k$
We bin by `(class_label, k)` to form strata used by selection.  
Binning may be **quantile** (robust) or **fixed** (stable edges). We keep the earlier defaults.

Outputs:
- `pool.bins_df` with `k_bin_idx` and `k_bin_label`
- a JSON meta on disk with edges and counts (for provenance)


```python
# bins the pool by k and class
bins_df, bins_meta = pool.bin_pool(
    label_source="base", # use base labels from the sampling plan
    binning="fixed",   # "fixed" is stable across runs, while "quantile" adapts to the distribution
    target_bins=5, min_bins=4, max_bins=6,
    fixed_edges_per_class=None,  # None => auto edges from observed [k_min, k_max]
    ensure_extrema=True, save_outputs=True,
    verbose=True
)
```

    rows in pool: 1171
    class 0:
      unique k: 5  range: [1,5]
      k counts: {1: 31, 2: 455, 3: 69, 4: 30, 5: 1}
      bins (4): labels=[1, 2, 3, 4, 5]  counts=[486, 69, 30, 1]
    class 1:
      unique k: 12  range: [1,12]
      k counts: {1: 6, 2: 45, 3: 220, 4: 109, 5: 115, 6: 58, 7: 12, 8: 9, 9: 5, 10: 2, 11: 2, 12: 2}
      bins (5): labels=[1, 4, 6, 8, 10, 12]  counts=[271, 224, 70, 14, 6]


### 3 - Constructing the teaching set $S$
In this section, we select diverse examples from the teaching pool to construct our teaching set $S$. </br>
We employ a teaching set construction strategy proposed by ...

#### 3.1 - Selection budget
We now derive per-bin budgets for the selector.  
Two policies are supported:
- even: split the per-class target evenly across that class’s bins; remainders go to densest bins.
- proportional: split by per-bin availability; fix rounding drift.

We stick with **even** for clarity.


```python
EXAMPLES_PER_CLASS = 50

# derive per-bin budgets from the pool
per_bin_budget = pool.derive_per_bin_budget(per_class_target=EXAMPLES_PER_CLASS, bin_allocation="even")
print(per_bin_budget)
```

    {'0': {'[1, 2]': 13, '[2, 3]': 13, '[3, 4]': 12, '[4, 5]': 12}, '1': {'[1, 4]': 10, '[4, 6]': 10, '[6, 8]': 10, '[8, 10]': 10, '[10, 12]': 10}}


#### 3.2 - Teaching set selection

We construct a `TeachingSet` from the $k$ strata and budgets. The objective is:

$$
F(S) = \sum_{i \in U} \max_{s \in S} \mathrm{sim}(i,s) \\
\qquad\mathrm{sim} = \text{cosine}(\text{L2-normalised embeddings})
$$

At each step, we take the argmax of the **marginal gain** plus small linear terms:

$$
\mathrm{score}(x) = \Delta F(x \mid S) + \lambda_m\,\mathrm{margin}(x) + \lambda_r\,\mathrm{robust\_prob}(x)\\
\quad (\lambda_m=0.10, \lambda_r=0.05)
$$

We also add the *deterministic* lever: `min_per_k=1` inside each bin to ensure rare `k` values are represented.



```python
# build the teaching set with even class distribution and a length range
s = TeachingSet(
    pool,
    per_bin_budget=per_bin_budget,
    per_class_target=None,         # ignored because per_bin_budget is provided
    bin_allocation="even",  # enforce even selection across k bins
    enforce_even_class_dist=True, # stops both classes when one hits its achievable limit
    length_range=(11, None),  # constrains minimum charging duration to >=11 minutes
    lambda_margin=0.10,
    lambda_robust=0.05,
    lazy_prune=True,
    random_seed=RANDOM_SEED,
    min_per_k=2,  # Enforce selection >= 2 of each k value so whole dist of k is represented
    output_dir=TEACHING_DIR,
)

s.save(output_dir=TEACHING_DIR)
```

    [teach] length filter [11, 60] kept 1141 / 1171 rows.
    [teach] selection complete.
      class 0: selected=28 | F(S)=554.4176
      class 1: selected=28 | F(S)=444.4229
      wrote → /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/selection.parquet
      wrote → /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/selection_config.json


#### 3.3 - Teaching set analytics


```python
s.describe()
display(s.teaching_set_df.head())
```

    [teaching set] rows=56, classes:
    class_label
    0    28
    1    28
    
    [coverage] facility-location by class:
      class 0: 554.4176
      class 1: 444.4229
    
    [per-bin selected] counts by class:
      class 0: {'[1, 2]': 13, '[2, 3]': 13, '[3, 4]': 1, '[4, 5]': 1}
      class 1: {'[1, 4]': 7, '[4, 6]': 7, '[6, 8]': 6, '[8, 10]': 4, '[10, 12]': 4}



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
      <th>session_id</th>
      <th>class_label</th>
      <th>k</th>
      <th>k_bin_idx</th>
      <th>k_bin_label</th>
      <th>label_int</th>
      <th>label_text</th>
      <th>robust_prob</th>
      <th>margin</th>
      <th>threshold</th>
      <th>...</th>
      <th>sts_soc_path</th>
      <th>piv_soc_path</th>
      <th>model_id</th>
      <th>length</th>
      <th>gain_coverage</th>
      <th>score_total</th>
      <th>rank_in_bin</th>
      <th>rank_in_class</th>
      <th>k_bin</th>
      <th>chosen_neighbors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6174344</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>5.758543</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/6174...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/6174...</td>
      <td>final_model.pth</td>
      <td>18</td>
      <td>206.167267</td>
      <td>206.793121</td>
      <td>1</td>
      <td>1</td>
      <td>[1, 2]</td>
      <td>[6174344, 4973057, 11640731, 11975700, 2263716]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4973057</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>7.151542</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/4973...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/4973...</td>
      <td>final_model.pth</td>
      <td>47</td>
      <td>2.204341</td>
      <td>2.969495</td>
      <td>2</td>
      <td>2</td>
      <td>[1, 2]</td>
      <td>[4973057, 2263716, 6174344, 11640731, 12195263]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9189528</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.391536</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/9189...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/9189...</td>
      <td>final_model.pth</td>
      <td>35</td>
      <td>194.846405</td>
      <td>195.735559</td>
      <td>3</td>
      <td>3</td>
      <td>[1, 2]</td>
      <td>[5634561, 1221566, 2045529, 10229995, 2676147]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9489647</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.351512</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/9489...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/9489...</td>
      <td>final_model.pth</td>
      <td>13</td>
      <td>0.000265</td>
      <td>0.885416</td>
      <td>4</td>
      <td>4</td>
      <td>[1, 2]</td>
      <td>[1221566, 10229995, 2045529, 10362279, 2676147]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2970205</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>7.422955</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_soc/2970...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv_soc/2970...</td>
      <td>final_model.pth</td>
      <td>30</td>
      <td>20.870968</td>
      <td>21.663263</td>
      <td>5</td>
      <td>5</td>
      <td>[1, 2]</td>
      <td>[2970205, 10566424, 12505559, 3819802, 5079383]</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>


What are the k values for selected examples vs the examples in the pool? </br>

We compare `(class, k)` and `(class, k-bin)` frequencies to inspect representation, especially at the tails. </br>
We report **selection rate** and **representation lift**:

$$
\text{selection\_rate} = \frac{n_{\text{sel}}}{n_{\text{pool}}},\quad
\text{lift} = \frac{\text{share}_{\text{sel}}}{\text{share}_{\text{pool}}}
$$


```python
# builds summary tables
rep = selection_vs_pool_report(pool.bins_df, s.teaching_set_df)

print("\nTeaching Pool share vs Teaching Set share")
from matplotlib.figure import Figure, Axes
import seaborn as sns 


def plot_teaching_share_by_k(
    by_k_df: pd.DataFrame,
    *,
    class_names: dict[int, str] | None = None,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (10.0, 3.2),
    bar_height: float = 0.6,
    legend_outside: bool = True,
) -> list[tuple[Figure, Axes]]:
    """Draws horizontal bar charts of share-by-k for pool vs selection, one figure per class.

    Converts the wide summary table (with columns like 'share_in_class_pool' and
    'share_in_class_sel') to long form and plots a grouped horizontal bar chart
    per class. Bars are ordered by ascending k and use Seaborn's categorical
    plotting for clear comparison of discrete segment counts.

    Args:
        by_k_df: Per-class summary with columns ['class_label', 'k',
            'share_in_class_pool', 'share_in_class_sel'].
        class_names: Optional mapping from class id (e.g. 0/1) to display name
            (e.g. {"0": "Normal", "1": "Abnormal"} or {0: "Normal", 1: "Abnormal"}).
        palette: Optional colour mapping for the two sources. Keys must match
            {'Teaching Pool Share', 'Teaching Set Share'}.
        figsize: Matplotlib figure size (width, height) for each class figure.
        bar_height: Height of each bar for a single hue; adjust for spacing.
        legend_outside: If True, places the legend to the right of the axes.

    Returns:
        list of (Figure, Axes) tuples, one per class label in ascending order.
    """
    # tidy names for the two series
    rename_cols = {
        "share_in_class_pool": "Teaching Pool Share",
        "share_in_class_sel": "Teaching Set Share",
    }
    if palette is None:
        # default to a simple, readable palette
        palette = {
            "Teaching Pool Share": sns.color_palette()[0],
            "Teaching Set Share": sns.color_palette()[1],
        }

    figs_axes: list[tuple[Figure, Axes]] = []
    classes = sorted(by_k_df["class_label"].unique())

    for c in classes:
        d = by_k_df[by_k_df["class_label"] == c].copy()
        d = d.rename(columns=rename_cols)

        # melt to long form for seaborn (source vs share)
        long = d.melt(
            id_vars=["k"],
            value_vars=list(rename_cols.values()),
            var_name="Source",
            value_name="Share",
        )

        # ensure discrete ordering by k
        k_order = sorted(d["k"].unique())
        long["k"] = pd.Categorical(long["k"], categories=k_order, ordered=True)

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=long,
            y="k",
            x="Share",
            hue="Source",
            orient="h",
            dodge=True,
            ax=ax,
            palette=palette,
            errorbar=None,
            linewidth=0.5,
            edgecolor="white",
        )

        # style
        disp_name = (
            class_names.get(c, str(c)) if isinstance(class_names, dict) else f"class {c}"
        )
        ax.set_title(f"{disp_name} — share by k")
        ax.set_xlabel("share within class")
        ax.set_ylabel("k")
        ax.set_ylim(-0.5, len(k_order) - 0.5)

        # consistent bar thickness
        for ccoll in ax.containers:
            try:
                ccoll.set_height(bar_height)
            except Exception:
                pass

        # legend placement
        if legend_outside:
            leg = ax.legend(
                title=None,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
            )
            fig.tight_layout()
        else:
            ax.legend(frameon=True)
            fig.tight_layout()

        figs_axes.append((fig, ax))

    return figs_axes


# from mt4xai.plot import plot_teaching_share_by_k

_ = plot_teaching_share_by_k(
    rep["by_k"],
    class_names={0: "Normals", 1: "Abnormals"},
    figsize=(10, 3.2),
    legend_outside=True,
)

print("Per-class, per-k comparison:")
display(
    rep["by_k"]
      .assign(selection_rate=lambda d: d["selection_rate"].round(3),
              lift=lambda d: d["lift"].round(3))
      .sort_values(["class_label", "k"])
      .reset_index(drop=True)
)

print("\nPer-class, per-k-bin comparison:")
display(
    rep["by_k_bin"]
      .assign(selection_rate=lambda d: d["selection_rate"].round(3),
              lift=lambda d: d["lift"].round(3))
      .sort_values(["class_label", "k_bin_label"])
      .reset_index(drop=True)
)

print("\nExtremes summary: (min/max k and rarest k per class in the pool)")
display(rep["extremes_summary"])
```

    
    Teaching Pool share vs Teaching Set share
    Per-class, per-k comparison:



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
      <th>class_label</th>
      <th>k</th>
      <th>n_pool</th>
      <th>share_in_class_pool</th>
      <th>n_sel</th>
      <th>share_in_class_sel</th>
      <th>selection_rate</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>0.052901</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>0.065</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>455</td>
      <td>0.776451</td>
      <td>11.0</td>
      <td>0.392857</td>
      <td>0.024</td>
      <td>0.506</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>69</td>
      <td>0.117747</td>
      <td>13.0</td>
      <td>0.464286</td>
      <td>0.188</td>
      <td>3.943</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4</td>
      <td>30</td>
      <td>0.051195</td>
      <td>1.0</td>
      <td>0.035714</td>
      <td>0.033</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0.001706</td>
      <td>1.0</td>
      <td>0.035714</td>
      <td>1.0</td>
      <td>20.929</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0.010256</td>
      <td>3.0</td>
      <td>0.107143</td>
      <td>0.5</td>
      <td>10.446</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>2</td>
      <td>45</td>
      <td>0.076923</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>0.044</td>
      <td>0.929</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>3</td>
      <td>220</td>
      <td>0.376068</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>0.009</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>4</td>
      <td>109</td>
      <td>0.186325</td>
      <td>4.0</td>
      <td>0.142857</td>
      <td>0.037</td>
      <td>0.767</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>5</td>
      <td>115</td>
      <td>0.196581</td>
      <td>3.0</td>
      <td>0.107143</td>
      <td>0.026</td>
      <td>0.545</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>6</td>
      <td>58</td>
      <td>0.099145</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>0.034</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>0.020513</td>
      <td>4.0</td>
      <td>0.142857</td>
      <td>0.333</td>
      <td>6.964</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>8</td>
      <td>9</td>
      <td>0.015385</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>0.222</td>
      <td>4.643</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>0.008547</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>0.4</td>
      <td>8.357</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>0.003419</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>1.0</td>
      <td>20.893</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>11</td>
      <td>2</td>
      <td>0.003419</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>0.003419</td>
      <td>2.0</td>
      <td>0.071429</td>
      <td>1.0</td>
      <td>20.893</td>
    </tr>
  </tbody>
</table>
</div>


    
    Per-class, per-k-bin comparison:



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
      <th>class_label</th>
      <th>k_bin_label</th>
      <th>n_pool</th>
      <th>share_in_class_pool</th>
      <th>n_sel</th>
      <th>share_in_class_sel</th>
      <th>selection_rate</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[1, 2]</td>
      <td>486</td>
      <td>0.829352</td>
      <td>13</td>
      <td>0.464286</td>
      <td>0.027</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[2, 3]</td>
      <td>69</td>
      <td>0.117747</td>
      <td>13</td>
      <td>0.464286</td>
      <td>0.188</td>
      <td>3.943</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[3, 4]</td>
      <td>30</td>
      <td>0.051195</td>
      <td>1</td>
      <td>0.035714</td>
      <td>0.033</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[4, 5]</td>
      <td>1</td>
      <td>0.001706</td>
      <td>1</td>
      <td>0.035714</td>
      <td>1.0</td>
      <td>20.929</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>[1, 4]</td>
      <td>271</td>
      <td>0.463248</td>
      <td>7</td>
      <td>0.250000</td>
      <td>0.026</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>[10, 12]</td>
      <td>6</td>
      <td>0.010256</td>
      <td>4</td>
      <td>0.142857</td>
      <td>0.667</td>
      <td>13.929</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>[4, 6]</td>
      <td>224</td>
      <td>0.382906</td>
      <td>7</td>
      <td>0.250000</td>
      <td>0.031</td>
      <td>0.653</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>[6, 8]</td>
      <td>70</td>
      <td>0.119658</td>
      <td>6</td>
      <td>0.214286</td>
      <td>0.086</td>
      <td>1.791</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>[8, 10]</td>
      <td>14</td>
      <td>0.023932</td>
      <td>4</td>
      <td>0.142857</td>
      <td>0.286</td>
      <td>5.969</td>
    </tr>
  </tbody>
</table>
</div>


    
    Extremes summary: (min/max k and rarest k per class in the pool)



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
      <th>class_label</th>
      <th>min_k</th>
      <th>max_k</th>
      <th>rare_k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](06__MT4XAI_files/06__MT4XAI_21_6.png)
    



    
![png](06__MT4XAI_files/06__MT4XAI_21_7.png)
    


### 4 - The Teaching Session: Serving examples from the teaching set <a id="teaching"></a>
The user study has three trial groups. All groups sample from the same teaching set, i.e. the same charging session IDs. The nature and ordering of examples differ:

- **Group A** (explanations with curriculum): order by `k_power↑` then `margin_power↓`, overlay raw power with simplification and show simplified SOC. 
- **Group B** (explanations, no curriculum): random order, overlay raw power with simplification and show simplified SOC. 
- **Group C** (control, no explanations): random order, raw power and SOC only (no simplifications).

Constructing the teaching set is computationally expensive and the optimal teaching set for explaining the time series classifier (TSC) is large. 
We therefore cap the teaching set at most 100 per class by default to limit the computational effort required to build it. 


```python
# serve without pre-capping; apply post-order cap when exporting
SAVE_DIR = Path(cfg.paths.figures) / "Teaching Sets"
s.build_group_iterators()  # no cap here anymore

# Create the Teaching Set and examples for trial groups A, B, C
examples = s.serve_examples(
    group="All",
    plot_examples=False,
    n=50,  # num examples to serve
    save_dir=SAVE_DIR,
    show_meta=False,
)
# The teaching sessions alterate between normal and abnormal examples. 
print(f"{len(examples["A"])} examples saved in {SAVE_DIR}/A")
print(f"{len(examples["B"])} examples saved in {SAVE_DIR}/B")
print(f"{len(examples["C"])} examples saved in {SAVE_DIR}/C")
```

    50 examples saved in /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Figures/Teaching Sets/A
    50 examples saved in /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Figures/Teaching Sets/B
    50 examples saved in /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Figures/Teaching Sets/C



```python
# Plot a few examples from the Teaching Set
s.build_group_iterators()  # resets the iterators (same order but restart from first example per set)
_ = s.serve_examples(
    group="A",
    plot_examples=True,
    n=30,
    save_dir=None,
    show_meta=False,
)
```

    ----------------  Teaching Set A  ----------------
    Classification label: normal, k: 1, session ID: 9489647



    
![png](06__MT4XAI_files/06__MT4XAI_24_1.png)
    


    Classification label: abnormal, k: 12, session ID: 5649864



    
![png](06__MT4XAI_files/06__MT4XAI_24_3.png)
    


    Classification label: normal, k: 1, session ID: 9189528



    
![png](06__MT4XAI_files/06__MT4XAI_24_5.png)
    


    Classification label: abnormal, k: 12, session ID: 11691589



    
![png](06__MT4XAI_files/06__MT4XAI_24_7.png)
    


    Classification label: normal, k: 2, session ID: 6174344



    
![png](06__MT4XAI_files/06__MT4XAI_24_9.png)
    


    Classification label: abnormal, k: 10, session ID: 4565492



    
![png](06__MT4XAI_files/06__MT4XAI_24_11.png)
    


    Classification label: normal, k: 2, session ID: 6823722



    
![png](06__MT4XAI_files/06__MT4XAI_24_13.png)
    


    Classification label: abnormal, k: 10, session ID: 10350999



    
![png](06__MT4XAI_files/06__MT4XAI_24_15.png)
    


    Classification label: normal, k: 2, session ID: 4711742



    
![png](06__MT4XAI_files/06__MT4XAI_24_17.png)
    


    Classification label: abnormal, k: 9, session ID: 1238697



    
![png](06__MT4XAI_files/06__MT4XAI_24_19.png)
    


    Classification label: normal, k: 2, session ID: 4973057



    
![png](06__MT4XAI_files/06__MT4XAI_24_21.png)
    


    Classification label: abnormal, k: 9, session ID: 6437298



    
![png](06__MT4XAI_files/06__MT4XAI_24_23.png)
    


    Classification label: normal, k: 2, session ID: 2970205



    
![png](06__MT4XAI_files/06__MT4XAI_24_25.png)
    


    Classification label: abnormal, k: 8, session ID: 3323361



    
![png](06__MT4XAI_files/06__MT4XAI_24_27.png)
    


    Classification label: normal, k: 2, session ID: 2617274



    
![png](06__MT4XAI_files/06__MT4XAI_24_29.png)
    


    Classification label: abnormal, k: 8, session ID: 10680807



    
![png](06__MT4XAI_files/06__MT4XAI_24_31.png)
    


    Classification label: normal, k: 2, session ID: 11482298



    
![png](06__MT4XAI_files/06__MT4XAI_24_33.png)
    


    Classification label: abnormal, k: 7, session ID: 1967784



    
![png](06__MT4XAI_files/06__MT4XAI_24_35.png)
    


    Classification label: normal, k: 2, session ID: 258594



    
![png](06__MT4XAI_files/06__MT4XAI_24_37.png)
    


    Classification label: abnormal, k: 7, session ID: 7930828



    
![png](06__MT4XAI_files/06__MT4XAI_24_39.png)
    


    Classification label: normal, k: 2, session ID: 2413381



    
![png](06__MT4XAI_files/06__MT4XAI_24_41.png)
    


    Classification label: abnormal, k: 7, session ID: 4241761



    
![png](06__MT4XAI_files/06__MT4XAI_24_43.png)
    


    Classification label: normal, k: 2, session ID: 4043459



    
![png](06__MT4XAI_files/06__MT4XAI_24_45.png)
    


    Classification label: abnormal, k: 7, session ID: 11820631



    
![png](06__MT4XAI_files/06__MT4XAI_24_47.png)
    


    Classification label: normal, k: 2, session ID: 9002133



    
![png](06__MT4XAI_files/06__MT4XAI_24_49.png)
    


    Classification label: abnormal, k: 6, session ID: 7372716



    
![png](06__MT4XAI_files/06__MT4XAI_24_51.png)
    


    Classification label: normal, k: 3, session ID: 10419548



    
![png](06__MT4XAI_files/06__MT4XAI_24_53.png)
    


    Classification label: abnormal, k: 6, session ID: 8568294



    
![png](06__MT4XAI_files/06__MT4XAI_24_55.png)
    


    Classification label: normal, k: 3, session ID: 6747437



    
![png](06__MT4XAI_files/06__MT4XAI_24_57.png)
    


    Classification label: abnormal, k: 5, session ID: 11156542



    
![png](06__MT4XAI_files/06__MT4XAI_24_59.png)
    



```python

```
