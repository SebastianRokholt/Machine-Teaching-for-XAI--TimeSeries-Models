# Notebook \#6: Machine Teaching for XAI
#### by Sebastian Einar Salas Røkholt

---


**Index**  
- [**1 - Introduction and Setup**](#1---introduction-and-setup)  
  - [*1.1 Setup*](#11-setup)  
  - [*1.2 Load the Anomaly Detection Model*](#12-load-the-model)  
  - [*1.3 Data Preparation*](#13-data-preparation)  
- [**2 - Chapter 2**](#2---chapter-2)  
  - [*2.1 ....*](#21-....)  

---

**References & Background Literature:**
- [Optimal Robust Simplifications for Explaining Time Series Classifications](https://xai.w.uib.no/files/2024/07/ORS.pdf) (2024) by Telle, Ferri & Håvardstun.
- [XAI with Machine Teaching when Humans Are (Not) Informed about the Irrelevant Features](https://doi.org/10.1007/978-3-031-43418-1_23) (2023) by Håvardstun, Ferri, Hernández-Orallo, Parviainen & Telle. 
- [When Redundancy Matters: Machine Teaching of Representations](https://arxiv.org/pdf/2401.12711.pdf) (2024) by Ferri, Garigliotti, Håvardstun, Hernandez-Orallo & Telle.
- [XAI with Machine Teaching when Humans Are (Not) Informed about the Irrelevant Features](https://doi.org/10.1007/978-3-031-43418-1_23) (2023) by Håvardstun, Ferri, Hernandez-Orallo, Parviainen & Telle. 


## 1 - Introduction and Setup
The aim of this notebook is to build a suitable teaching set which can be used to serve example-based explanations of the anomaly detection system (the black box AI) to human users. 

This notebook uses a class-based API defined in the `mt4xai` package's `teach` and `data` modules: 
- `TeachingPool` owns the pool (`pool.parquet`), binning, and paths.
- `TeachingSet` performs selection with greedy facility-location (+ lazy pruning), then exposes A/B/C samplers.
- `ChargingSession` contains the original, dense, unscaled arrays for the `power` series (and optionally for other channels/features as well). 
- `ChargingSessionSimplification` with knots\* only (indices + values). We cnvert these to dense simplifications and fetch original raw series on demand.
A *knot* is an endpoint of a straight line segment in the simplification. We define **k = number of straight line segments = knots − 1**.


```python
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# project config
sys.path.append(str(Path.cwd().parent))  # Add scripts (e.g. project_config.py) in parent dir to path
from project_config import load_config
cfg = load_config()

# Modules from the project's MT4XAI package
%load_ext autoreload
%autoreload 2
from mt4xai.model import load_lstm_model
from mt4xai.data import split_data, apply_scalers, fit_scalers_on_train, build_loader, ChargingSession, ChargingSessionSimplification
from mt4xai.ors import ORSParams
from mt4xai.teach import TeachingPool, TeachingPoolConfig, TeachingSet, selection_vs_pool_report, build_group_iterators

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
DEVICE = cfg.project.device
print("Device: ", DEVICE)
RANDOM_SEED = cfg.project.random_seed
AD_THRESHOLD = cfg.anomaly_detection.rmse_threshold
```

    CONFIG FILE LOADED: 
    {'project': {'random_seed': 42, 'root_dir': None}, 'paths': {'dataset': 'Data/etron55-charging-sessions.parquet', 'teaching_pool': 'Data/teaching_pool', 'models': 'Models', 'final_model': 'Models/final/final_model.pth', 'figures': 'Figures', 'logs': 'Logs'}, 'modelling': {'horizon': 5, 'final_model_name': 'final_model.pth', 'power_weight': 0.6522982410461, 'decay_lambda': 0.4}, 'anomaly_detection': {'t_min_eval': 1, 'rmse_threshold': 8.5962, 'ad_pct_threshold': 0.95, 'metric': 'macro_rmse'}, 'ors': None, 'teaching': {'teaching_pool_dir': '../Data/teaching_pool', 'teaching_set_size': 200}}
    Device:  cuda:0


### 1.2 Load the Forecasting Model


```python
# load model and hyperparams from Ray Tune checkpoint
model, ckpt = load_lstm_model(MODEL_FP, device=DEVICE)
input_features  = ckpt["input_features"]
target_features = ckpt["target_features"]
H = int(ckpt["config"]["horizon"])  # forecast horizon
power_weight = float(ckpt["config"].get("power_weight", 0.5))
idx_power_inp = input_features.index("power")
idx_soc_inp = input_features.index("soc")
```

    /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/src/mt4xai/model.py:45: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      checkpoint = torch.load(path, map_location=device)


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

train_df, val_df, test_df = split_data(df, test_size=0.2, validation_size=0.1)
scalers = fit_scalers_on_train(train_df, list(set(input_features) | set(target_features)))
power_scaler, soc_scaler = scalers["power"], scalers["soc"]
test_s = apply_scalers(test_df, scalers)

test_loader = build_loader(test_s, input_features, target_features, H,
                           batch_size=16, shuffle=False, num_workers=0)
print(f"There are {len(test_loader.dataset.groups)} charging sessions in the test set")
```

    There are 12183 charging sessions in the test set


### 2 - Computing the Teaching Pool
The Teaching Pool is a relatively large collection of samples (charging sessions / examples) from the test set. 

#### 2.1 - Running the pool constructor
Constructing the pool of simplifications + raw sessions is very computationally intensive, since ORS runs model inference thousands of times per simplification, and we want over a thousand simplifications in our teaching pool. The `mt4xai` API let's us load an existing pool (from the file `pool.parquet`) or construct it from scratch. Construction runs inference + ORS + embedding and writes the SQLite cache and the Parquet snapshot.



```python
# choose: load an existing pool if available; otherwise show how to construct
pool_parquet = TEACHING_DIR / "pool.parquet"

if pool_parquet.exists():
    # loads the pool snapshot from disk
    print("Loaded previously computed teaching pool from disk")
    pool = TeachingPool.load_from_parquet(pool_parquet)
else:
    # builds the pool end-to-end, then loads it
    # note: this path requires model + test_loader + scalers to be in scope
    # builds the pool of simplifications and raw sessions
    tpconfig = TeachingPoolConfig(
        model_path=MODEL_FP,  # our pre-trained "black box" (multivariate LSTM) model
        output_dir=TEACHING_DIR,
        threshold=AD_THRESHOLD,  # 8.5962, e.g the 95th percentile
        seed=RANDOM_SEED,
        device_str=DEVICE,
        export_every=25,
        L=128, P=4,   # for embeddings 
        ors_params=ORSParams(
            stage1_mode="dp_prefix", stage2_err_metric="l2",
            dp_q=150, rdp_stage1_candidates=30,
            dp_alpha=0.01, beta=3.0, gamma=0.05,
            R=2000, epsilon_mode="fraction", epsilon_value=0.2,
            t_min_eval=1, min_k=1, max_k=12, seed=cfg.project.random_seed,
            model_id="final_model.pth"
        ),
        decay_lambda=cfg.modelling.decay_lambda  # 0.4
    )

    pool = TeachingPool.construct(
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

    [teaching pool] rows=1172, classes:
    label_int
    0    586
    1    586
    
    [k] stats:
     count    1172.000000
    mean        3.537543
    std         1.750771
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>137598</td>
      <td>normal</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.300746</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>8.295454</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.760392e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/137...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/137598.npy</td>
      <td>264</td>
      <td>[2.240215301513672, 2.1924808025360107, 2.1447460651397705, 2.0970146656036377, 2.0492897033691406, 2.00156450271606...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>167874</td>
      <td>abnormal</td>
      <td>1</td>
      <td>4.0</td>
      <td>13.030439</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.434239</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.760392e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/167...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/167874.npy</td>
      <td>264</td>
      <td>[-2.2508716583251953, -1.8199647665023804, -1.3890577554702759, -0.9581508040428162, -0.5272438526153564, -0.0963368...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>231570</td>
      <td>abnormal</td>
      <td>1</td>
      <td>4.0</td>
      <td>9.624545</td>
      <td>0.037500</td>
      <td>0.962500</td>
      <td>1.028345</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.760392e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/231...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/231570.npy</td>
      <td>264</td>
      <td>[-2.075390100479126, -1.734521746635437, -1.3936532735824585, -1.0527849197387695, -0.711916446685791, -0.3710480034...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240316</td>
      <td>normal</td>
      <td>0</td>
      <td>3.0</td>
      <td>1.925004</td>
      <td>0.037500</td>
      <td>0.962500</td>
      <td>6.671196</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.760392e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/240...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/240316.npy</td>
      <td>264</td>
      <td>[1.6217665672302246, 1.6354061365127563, 1.6490458250045776, 1.6626853942871094, 1.6763250827789307, 1.6899646520614...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>248477</td>
      <td>abnormal</td>
      <td>1</td>
      <td>4.0</td>
      <td>8.747483</td>
      <td>0.259167</td>
      <td>0.740833</td>
      <td>0.151283</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.760392e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/248...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/248477.npy</td>
      <td>264</td>
      <td>[2.564788341522217, 2.6007373332977295, 2.636686325073242, 2.672635078430176, 2.7085840702056885, 2.744532823562622,...</td>
    </tr>
  </tbody>
</table>
</div>


    Frequencies of k for normal sessions in the teaching pool:
    k
    2.0    381
    3.0     99
    4.0     60
    1.0     24
    5.0     20
    6.0      2
    Name: count, dtype: int64
    Frequencies of k for abnormal sessions in the teaching pool:
    k
    3.0     175
    4.0     131
    5.0     115
    6.0      85
    7.0      30
    8.0      19
    2.0      12
    9.0       9
    11.0      4
    12.0      3
    10.0      3
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

    rows in pool: 1172
    class 0:
      unique k: 6  range: [1,6]
      k counts: {1: 24, 2: 381, 3: 99, 4: 60, 5: 20, 6: 2}
      bins (5): labels=[1, 2, 3, 4, 5, 6]  counts=[405, 99, 60, 20, 2]
    class 1:
      unique k: 11  range: [2,12]
      k counts: {2: 12, 3: 175, 4: 131, 5: 115, 6: 85, 7: 30, 8: 19, 9: 9, 10: 3, 11: 4, 12: 3}
      bins (5): labels=[2, 4, 6, 8, 10, 12]  counts=[318, 200, 49, 12, 7]


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
# derives a per-bin budget (target ≤100 per class)
per_bin_budget = pool.derive_per_bin_budget(per_class_target=100, bin_allocation="even")

print("budgets (class 0):", per_bin_budget.get("0", {}))
print("budgets (class 1):", per_bin_budget.get("1", {}))
```

    budgets (class 0): {'[1, 2]': 20, '[2, 3]': 20, '[3, 4]': 20, '[4, 5]': 20, '[5, 6]': 20}
    budgets (class 1): {'[2, 4]': 20, '[4, 6]': 20, '[6, 8]': 20, '[8, 10]': 20, '[10, 12]': 20}


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
# runs the selection process
s = TeachingSet(
    pool,
    per_bin_budget=per_bin_budget,
    per_class_target=None,   # used only if per_bin_budget=None
    bin_allocation="even",
    lambda_margin=0.10,
    lambda_robust=0.05,
    normalize_embeddings=True,
    lazy_prune=True,
    dtw_tie_refine=False, # keep off unless near ties are common
    sim_clip_min=0.0,  # clip negatives to 0 for safer gains
    seed=RANDOM_SEED,
    min_per_k=2,  # easy lever: guarantee ≥2 per k inside each bin (if present)
    output_dir=TEACHING_DIR,
)

s.save(output_dir=TEACHING_DIR)
```

    [teach] selection complete.
      class 0: selected=55 | F(S)=573.0247
      class 1: selected=54 | F(S)=501.9070
      wrote → /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/selection.parquet
      wrote → /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/selection_config.json


#### 3.3 - Teaching set analytics


```python
s.describe()
display(s.selected_df.head())
```

    [teaching set] rows=109, classes:
    class_label
    0    55
    1    54
    
    [coverage] facility-location by class:
      class 0: 573.0247
      class 1: 501.9070
    
    [per-bin selected] counts by class:
      class 0: {'[1, 2]': 20, '[2, 3]': 20, '[3, 4]': 2, '[4, 5]': 11, '[5, 6]': 2}
      class 1: {'[2, 4]': 20, '[4, 6]': 20, '[6, 8]': 0, '[8, 10]': 8, '[10, 12]': 6}



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
      <th>emb</th>
      <th>sts_full_path</th>
      <th>piv_path</th>
      <th>model_id</th>
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
      <td>8661544</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.363071</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-2.000138998031616, -1.964316725730896, -1.9284944534301758, -1.8926723003387451, -1.856868028640747, -1.8210674524...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/866...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/8661544.npy</td>
      <td>final_model.pth</td>
      <td>191.079895</td>
      <td>191.966202</td>
      <td>1</td>
      <td>1</td>
      <td>[1, 2]</td>
      <td>[8661544, 6356776, 1517248, 2617274, 2045529]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10706141</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.221979</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-2.061953544616699, -2.0238306522369385, -1.9857079982757568, -1.9475852251052856, -1.9094624519348145, -1.87133967...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/107...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/10706141...</td>
      <td>final_model.pth</td>
      <td>1.218426</td>
      <td>2.090624</td>
      <td>2</td>
      <td>2</td>
      <td>[1, 2]</td>
      <td>[10706141, 6413520, 5813334, 2617274, 6356776]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12145955</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.407463</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-1.7185766696929932, -1.6915091276168823, -1.6644415855407715, -1.6373741626739502, -1.6103066205978394, -1.5832390...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/121...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/12145955...</td>
      <td>final_model.pth</td>
      <td>6.536997</td>
      <td>7.427743</td>
      <td>3</td>
      <td>3</td>
      <td>[1, 2]</td>
      <td>[12145955, 10229995, 2560674, 2361727, 9489647]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9189528</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.387282</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-1.7185765504837036, -1.6915032863616943, -1.664430022239685, -1.6373568773269653, -1.610289454460144, -1.583238124...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/918...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/9189528.npy</td>
      <td>final_model.pth</td>
      <td>0.000012</td>
      <td>0.888740</td>
      <td>4</td>
      <td>4</td>
      <td>[1, 2]</td>
      <td>[12145955, 10229995, 2560674, 2361727, 9489647]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4275867</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>5.803730</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[1.0843602418899536, 1.0736451148986816, 1.0629298686981201, 1.0522147417068481, 1.0414994955062866, 1.0307843685150...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/427...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/4275867.npy</td>
      <td>final_model.pth</td>
      <td>157.401123</td>
      <td>158.031496</td>
      <td>5</td>
      <td>5</td>
      <td>[1, 2]</td>
      <td>[4275867, 2368429, 2537218, 6174344, 8743262]</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
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
rep = selection_vs_pool_report(pool.bins_df, s.selected_df)

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

print("\nTeaching Pool share vs Teaching Set share")
for c in (0, 1):
    plot_df = rep["by_k"][rep["by_k"]["class_label"] == c]
    fig = plt.figure(figsize=(10, 3.2))
    ax = plt.gca()
    ax.plot(plot_df["k"], plot_df["share_in_class_pool"], label="pool share")
    ax.plot(plot_df["k"], plot_df["share_in_class_sel"], label="selection share")
    ax.set_title(f"class {c} — share by k")
    ax.set_xlabel("k")
    ax.set_ylabel("share within class")
    ax.legend()
    plt.show()

```

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
      <td>24</td>
      <td>0.040956</td>
      <td>2.0</td>
      <td>0.036364</td>
      <td>0.083</td>
      <td>0.888</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>381</td>
      <td>0.650171</td>
      <td>18.0</td>
      <td>0.327273</td>
      <td>0.047</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>99</td>
      <td>0.168942</td>
      <td>20.0</td>
      <td>0.363636</td>
      <td>0.202</td>
      <td>2.152</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4</td>
      <td>60</td>
      <td>0.102389</td>
      <td>2.0</td>
      <td>0.036364</td>
      <td>0.033</td>
      <td>0.355</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>20</td>
      <td>0.03413</td>
      <td>11.0</td>
      <td>0.200000</td>
      <td>0.55</td>
      <td>5.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>0.003413</td>
      <td>2.0</td>
      <td>0.036364</td>
      <td>1.0</td>
      <td>10.655</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>0.020478</td>
      <td>3.0</td>
      <td>0.055556</td>
      <td>0.25</td>
      <td>2.713</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>3</td>
      <td>175</td>
      <td>0.298635</td>
      <td>13.0</td>
      <td>0.240741</td>
      <td>0.074</td>
      <td>0.806</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>4</td>
      <td>131</td>
      <td>0.223549</td>
      <td>4.0</td>
      <td>0.074074</td>
      <td>0.031</td>
      <td>0.331</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>5</td>
      <td>115</td>
      <td>0.196246</td>
      <td>15.0</td>
      <td>0.277778</td>
      <td>0.13</td>
      <td>1.415</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>6</td>
      <td>85</td>
      <td>0.145051</td>
      <td>5.0</td>
      <td>0.092593</td>
      <td>0.059</td>
      <td>0.638</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>7</td>
      <td>30</td>
      <td>0.051195</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>8</td>
      <td>19</td>
      <td>0.032423</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>0.015358</td>
      <td>6.0</td>
      <td>0.111111</td>
      <td>0.667</td>
      <td>7.235</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>0.005119</td>
      <td>2.0</td>
      <td>0.037037</td>
      <td>0.667</td>
      <td>7.235</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>11</td>
      <td>4</td>
      <td>0.006826</td>
      <td>3.0</td>
      <td>0.055556</td>
      <td>0.75</td>
      <td>8.139</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>12</td>
      <td>3</td>
      <td>0.005119</td>
      <td>3.0</td>
      <td>0.055556</td>
      <td>1.0</td>
      <td>10.852</td>
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
      <td>405</td>
      <td>0.691126</td>
      <td>20.0</td>
      <td>0.363636</td>
      <td>0.049</td>
      <td>0.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[2, 3]</td>
      <td>99</td>
      <td>0.168942</td>
      <td>20.0</td>
      <td>0.363636</td>
      <td>0.202</td>
      <td>2.152</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[3, 4]</td>
      <td>60</td>
      <td>0.102389</td>
      <td>2.0</td>
      <td>0.036364</td>
      <td>0.033</td>
      <td>0.355</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[4, 5]</td>
      <td>20</td>
      <td>0.03413</td>
      <td>11.0</td>
      <td>0.200000</td>
      <td>0.55</td>
      <td>5.86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[5, 6]</td>
      <td>2</td>
      <td>0.003413</td>
      <td>2.0</td>
      <td>0.036364</td>
      <td>1.0</td>
      <td>10.655</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>[10, 12]</td>
      <td>7</td>
      <td>0.011945</td>
      <td>6.0</td>
      <td>0.111111</td>
      <td>0.857</td>
      <td>9.302</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>[2, 4]</td>
      <td>318</td>
      <td>0.542662</td>
      <td>20.0</td>
      <td>0.370370</td>
      <td>0.063</td>
      <td>0.683</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>[4, 6]</td>
      <td>200</td>
      <td>0.341297</td>
      <td>20.0</td>
      <td>0.370370</td>
      <td>0.1</td>
      <td>1.085</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>[6, 8]</td>
      <td>49</td>
      <td>0.083618</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>[8, 10]</td>
      <td>12</td>
      <td>0.020478</td>
      <td>8.0</td>
      <td>0.148148</td>
      <td>0.667</td>
      <td>7.235</td>
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
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>12.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Teaching Pool share vs Teaching Set share



    
![png](06__Teach_files/06__Teach_21_7.png)
    



    
![png](06__Teach_files/06__Teach_21_8.png)
    


#### 4 - Serving examples from the teaching set
The user study has three trial groups. All groups sample from the same teaching set, i.e. the same charging session IDs. The nature and ordering of examples differ:

- **Group A** (explanations with curriculum): order by `k↑` then `margin↓`, overlay simplification.
- **Group B** (explanations, no curriculum): random order, overlay simplification.
- **Group C** (control, no explanations): random order, raw only (no overlay).

We keep at most 100 per class by default.



```python
# builds group views from the selected set
df_A = s.sample_group_A(max_per_class=100)  # curriculum: k↑, margin↓
df_B = s.sample_group_B(max_per_class=100, seed=RANDOM_SEED)
df_C = s.sample_group_C(max_per_class=100, seed=RANDOM_SEED)

# verify same IDs across groups
ids_A = set(df_A["session_id"].unique().tolist())
ids_B = set(df_B["session_id"].unique().tolist())
ids_C = set(df_C["session_id"].unique().tolist())
print("ID sets identical across groups:", (ids_A == ids_B == ids_C))

# peek
display(df_A.head(1))
display(df_B.head(1))
display(df_C.head(1))
```

    ID sets identical across groups: True



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
      <th>piv_path</th>
      <th>model_id</th>
      <th>gain_coverage</th>
      <th>score_total</th>
      <th>rank_in_bin</th>
      <th>rank_in_class</th>
      <th>k_bin</th>
      <th>chosen_neighbors</th>
      <th>group</th>
      <th>show_simpl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>12145955</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>[1, 2]</td>
      <td>0</td>
      <td>normal</td>
      <td>1.0</td>
      <td>8.407463</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/12145955...</td>
      <td>final_model.pth</td>
      <td>6.536997</td>
      <td>7.427743</td>
      <td>3</td>
      <td>3</td>
      <td>[1, 2]</td>
      <td>[12145955, 10229995, 2560674, 2361727, 9489647]</td>
      <td>A</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>



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
      <th>piv_path</th>
      <th>model_id</th>
      <th>gain_coverage</th>
      <th>score_total</th>
      <th>rank_in_bin</th>
      <th>rank_in_class</th>
      <th>k_bin</th>
      <th>chosen_neighbors</th>
      <th>group</th>
      <th>show_simpl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7813535</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>[4, 6]</td>
      <td>1</td>
      <td>abnormal</td>
      <td>0.9255</td>
      <td>1.706221</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/7813535.npy</td>
      <td>final_model.pth</td>
      <td>0.294781</td>
      <td>0.511678</td>
      <td>4</td>
      <td>24</td>
      <td>[4, 6]</td>
      <td>[7813535, 4481684, 7741850, 5824034, 4007889]</td>
      <td>B</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>



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
      <th>piv_path</th>
      <th>model_id</th>
      <th>gain_coverage</th>
      <th>score_total</th>
      <th>rank_in_bin</th>
      <th>rank_in_class</th>
      <th>k_bin</th>
      <th>chosen_neighbors</th>
      <th>group</th>
      <th>show_simpl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7813535</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>[4, 6]</td>
      <td>1</td>
      <td>abnormal</td>
      <td>0.9255</td>
      <td>1.706221</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/7813535.npy</td>
      <td>final_model.pth</td>
      <td>0.294781</td>
      <td>0.511678</td>
      <td>4</td>
      <td>24</td>
      <td>[4, 6]</td>
      <td>[7813535, 4481684, 7741850, 5824034, 4007889]</td>
      <td>C</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>



```python
# build and attach iterators to the TeachingSet
iters = s.build_group_iterators(max_per_class=100, seed=RANDOM_SEED)

# demo: serve 6 examples per group and log metadata
logs = {"A": [], "B": [], "C": []}
# for g in ("A", "B", "C"):
for _ in range(20):
    try:
        meta = s.serve_sessions(group="A")  # plots immediately
        logs["A"].append(meta)
    except StopIteration:
        print(f"[{"A"}] no more examples.")
        break

pd.DataFrame(logs["A"]).head()

```

    k=1 segments
    k=2 segments
    k=1 segments
    k=2 segments
    k=2 segments
    k=2 segments
    k=2 segments
    k=3 segments
    k=2 segments
    k=3 segments
    k=2 segments
    k=3 segments
    k=2 segments
    k=3 segments
    k=2 segments
    k=3 segments
    k=2 segments
    k=3 segments
    k=2 segments
    k=3 segments





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
      <th>group</th>
      <th>k</th>
      <th>label</th>
      <th>sts_full_path</th>
      <th>piv_path</th>
      <th>raw_power_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12145955</td>
      <td>A</td>
      <td>1</td>
      <td>normal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/121...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/12145955...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/12...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2086667</td>
      <td>A</td>
      <td>2</td>
      <td>abnormal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/208...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/2086667.npy</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/20...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9189528</td>
      <td>A</td>
      <td>1</td>
      <td>normal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/918...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/9189528.npy</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/91...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3907158</td>
      <td>A</td>
      <td>2</td>
      <td>abnormal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/390...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/3907158.npy</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/39...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9002133</td>
      <td>A</td>
      <td>2</td>
      <td>normal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/900...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/9002133.npy</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/90...</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](06__Teach_files/06__Teach_24_2.png)
    



    
![png](06__Teach_files/06__Teach_24_3.png)
    



    
![png](06__Teach_files/06__Teach_24_4.png)
    



    
![png](06__Teach_files/06__Teach_24_5.png)
    



    
![png](06__Teach_files/06__Teach_24_6.png)
    



    
![png](06__Teach_files/06__Teach_24_7.png)
    



    
![png](06__Teach_files/06__Teach_24_8.png)
    



    
![png](06__Teach_files/06__Teach_24_9.png)
    



    
![png](06__Teach_files/06__Teach_24_10.png)
    



    
![png](06__Teach_files/06__Teach_24_11.png)
    



    
![png](06__Teach_files/06__Teach_24_12.png)
    



    
![png](06__Teach_files/06__Teach_24_13.png)
    



    
![png](06__Teach_files/06__Teach_24_14.png)
    



    
![png](06__Teach_files/06__Teach_24_15.png)
    



    
![png](06__Teach_files/06__Teach_24_16.png)
    



    
![png](06__Teach_files/06__Teach_24_17.png)
    



    
![png](06__Teach_files/06__Teach_24_18.png)
    



    
![png](06__Teach_files/06__Teach_24_19.png)
    



    
![png](06__Teach_files/06__Teach_24_20.png)
    



    
![png](06__Teach_files/06__Teach_24_21.png)
    



```python

```
