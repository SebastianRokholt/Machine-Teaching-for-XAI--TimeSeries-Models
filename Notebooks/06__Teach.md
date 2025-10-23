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
import torch
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
DEVICE = torch.device(cfg.project.device)
print("Device: ", DEVICE)
RANDOM_SEED = cfg.project.random_seed
AD_THRESHOLD = cfg.anomaly_detection.rmse_threshold
```

    CONFIG FILE LOADED: 
    {'project': {'random_seed': 42, 'root_dir': None}, 'paths': {'dataset': 'Data/etron55-charging-sessions.parquet', 'teaching_pool': 'Data/teaching_pool', 'models': 'Models', 'final_model': 'Models/final/final_model.pth', 'figures': 'Figures', 'logs': 'Logs'}, 'inference': {'horizon': 5, 'final_model_name': 'final_model.pth', 'power_weight': 0.6522982410461, 'horizon_decay_lambda': 0.4}, 'anomaly_detection': {'t_min_eval': 1, 'rmse_threshold': 8.5962, 'ad_pct_threshold': 0.95, 'metric': 'macro_rmse'}, 'ors': {'soc_stage1_mode': 'rdp', 'soc_rdp_epsilon': 0.75, 'soc_rdp_candidates': 5, 'soc_rdp_eps_min': 1e-06, 'soc_rdp_eps_max': 100.0, 'stage2_err_metric': 'l2', 'epsilon_mode': 'fraction'}, 'teaching': {'teaching_pool_dir': '../Data/teaching_pool', 'teaching_set_size': 200}}
    Device:  cuda


### 1.2 Load the Forecasting Model


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
    # builds the pool of simplifications and raw sessions
    tpconfig = TeachingPoolConfig(
        model_path=MODEL_FP,  # our pre-trained "black box" (multivariate LSTM) model
        output_dir=TEACHING_DIR,
        threshold=AD_THRESHOLD,  # 8.5962, e.g the 95th percentile
        seed=RANDOM_SEED,
        device=DEVICE,
        export_every=25,
        L=128, P=4,   # for embeddings 
        ors_params=ORSParams(
            stage1_mode="dp_prefix", stage2_err_metric="l2",
            dp_q=200, rdp_stage1_candidates=30,
            dp_alpha=0.01, beta=3.0, gamma=0.05,
            R=3000, epsilon_mode="fraction", epsilon_value=0.2,
            t_min_eval=1, min_k=1, max_k=12, seed=RANDOM_SEED,
            soc_stage1_mode="rdp", soc_rdp_epsilon=0.75,
            model_id="final_model.pth"
        ),
        power_weight=cfg.inference.power_weight, 
        decay_lambda=cfg.inference.horizon_decay_lambda
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

    [teach] computing base labels on test set ...
    [teach] base labels: abnormal=586, normal=11597
    [teach] wrote sampling plan → /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sampled_normals.json
    [teach] sampling plan: abnormal=586, normal=586, total=1172
    [teach] no cached rows yet.
    [teach] ORS    1/1172 ( 0.1%) sid=137598
    [teach] ORS    2/1172 ( 0.2%) sid=167874
    [teach] ORS    3/1172 ( 0.3%) sid=231570
    [teach] ORS    4/1172 ( 0.3%) sid=240316
    [teach] ORS    5/1172 ( 0.4%) sid=248477
    [teach] ORS    6/1172 ( 0.5%) sid=258594
    [teach] ORS    7/1172 ( 0.6%) sid=273939
    [teach] ORS    8/1172 ( 0.7%) sid=275694
    [teach] ORS    9/1172 ( 0.8%) sid=277881
    [teach] ORS   10/1172 ( 0.9%) sid=278776
    [teach] ORS   11/1172 ( 0.9%) sid=287384
    [teach] ORS   12/1172 ( 1.0%) sid=292366
    [teach] ORS   13/1172 ( 1.1%) sid=312605
    [teach] ORS   14/1172 ( 1.2%) sid=316422
    [teach] ORS   15/1172 ( 1.3%) sid=326062
    [teach] ORS   16/1172 ( 1.4%) sid=364069
    [teach] ORS   17/1172 ( 1.5%) sid=380783
    [teach] ORS   18/1172 ( 1.5%) sid=383034
    [teach] ORS   19/1172 ( 1.6%) sid=415916
    [teach] ORS   20/1172 ( 1.7%) sid=423717
    [teach] ORS   21/1172 ( 1.8%) sid=438679
    [teach] ORS   22/1172 ( 1.9%) sid=449444
    [teach] ORS   23/1172 ( 2.0%) sid=486627
    [teach] ORS   24/1172 ( 2.0%) sid=503580
    [teach] ORS   25/1172 ( 2.1%) sid=528568
    [teach] exporting parquet snapshot ...
    [teach] ORS   26/1172 ( 2.2%) sid=539527
    [teach] ORS   27/1172 ( 2.3%) sid=542556
    [teach] ORS   28/1172 ( 2.4%) sid=565736
    [teach] ORS   29/1172 ( 2.5%) sid=572469
    [teach] ORS   30/1172 ( 2.6%) sid=696300
    [teach] ORS   31/1172 ( 2.6%) sid=733834
    [teach] ORS   32/1172 ( 2.7%) sid=753198
    [teach] ORS   33/1172 ( 2.8%) sid=773853
    [teach] ORS   34/1172 ( 2.9%) sid=838323
    [teach] ORS   35/1172 ( 3.0%) sid=867026
    [teach] ORS   36/1172 ( 3.1%) sid=901853
    [teach] ORS   37/1172 ( 3.2%) sid=949170
    [teach] ORS   38/1172 ( 3.2%) sid=1004178
    [teach] ORS   39/1172 ( 3.3%) sid=1005704
    [teach] ORS   40/1172 ( 3.4%) sid=1046397
    [teach] ORS   41/1172 ( 3.5%) sid=1051257
    [teach] ORS   42/1172 ( 3.6%) sid=1084286
    [teach] ORS   43/1172 ( 3.7%) sid=1185591
    [teach] ORS   44/1172 ( 3.8%) sid=1188288
    [teach] ORS   45/1172 ( 3.8%) sid=1194228
    [teach] ORS   46/1172 ( 3.9%) sid=1221566
    [teach] ORS   47/1172 ( 4.0%) sid=1233409
    [ORS][warn] sid=1238697 no valid candidates after constraints (mode=dp_prefix, k_span=15..19, dp_q=51, beta=3.0). Trying fallback #1.
    [teach] ORS   48/1172 ( 4.1%) sid=1238697
    [teach] ORS   49/1172 ( 4.2%) sid=1248426
    [teach] ORS   50/1172 ( 4.3%) sid=1288122
    [teach] exporting parquet snapshot ...
    [teach] ORS   51/1172 ( 4.4%) sid=1290634
    [teach] ORS   52/1172 ( 4.4%) sid=1295642
    [teach] ORS   53/1172 ( 4.5%) sid=1306102
    [teach] ORS   54/1172 ( 4.6%) sid=1386029
    [teach] ORS   55/1172 ( 4.7%) sid=1398919
    [teach] ORS   56/1172 ( 4.8%) sid=1420420
    [teach] ORS   57/1172 ( 4.9%) sid=1420968
    [teach] ORS   58/1172 ( 4.9%) sid=1424428
    [teach] ORS   59/1172 ( 5.0%) sid=1431374
    [teach] ORS   60/1172 ( 5.1%) sid=1455320
    [teach] ORS   61/1172 ( 5.2%) sid=1463902
    [teach] ORS   62/1172 ( 5.3%) sid=1496220
    [teach] ORS   63/1172 ( 5.4%) sid=1498205
    [teach] ORS   64/1172 ( 5.5%) sid=1517248
    [teach] ORS   65/1172 ( 5.5%) sid=1528901
    [teach] ORS   66/1172 ( 5.6%) sid=1547198
    [teach] ORS   67/1172 ( 5.7%) sid=1573095
    [teach] ORS   68/1172 ( 5.8%) sid=1617603
    [teach] ORS   69/1172 ( 5.9%) sid=1627938
    [teach] ORS   70/1172 ( 6.0%) sid=1631444
    [teach] ORS   71/1172 ( 6.1%) sid=1634350
    [teach] ORS   72/1172 ( 6.1%) sid=1635164
    [teach] ORS   73/1172 ( 6.2%) sid=1653776
    [teach] ORS   74/1172 ( 6.3%) sid=1660002
    [teach] ORS   75/1172 ( 6.4%) sid=1667459
    [teach] exporting parquet snapshot ...
    [teach] ORS   76/1172 ( 6.5%) sid=1682607
    [teach] ORS   77/1172 ( 6.6%) sid=1752437
    [teach] ORS   78/1172 ( 6.7%) sid=1754470
    [teach] ORS   79/1172 ( 6.7%) sid=1763422
    [teach] ORS   80/1172 ( 6.8%) sid=1781615
    [teach] ORS   81/1172 ( 6.9%) sid=1783874
    [teach] ORS   82/1172 ( 7.0%) sid=1794783
    [teach] ORS   83/1172 ( 7.1%) sid=1799330
    [teach] ORS   84/1172 ( 7.2%) sid=1804119
    [teach] ORS   85/1172 ( 7.3%) sid=1805356
    [teach] ORS   86/1172 ( 7.3%) sid=1806086
    [teach] ORS   87/1172 ( 7.4%) sid=1807025
    [teach] ORS   88/1172 ( 7.5%) sid=1821681
    [teach] ORS   89/1172 ( 7.6%) sid=1826297
    [teach] ORS   90/1172 ( 7.7%) sid=1827733
    [teach] ORS   91/1172 ( 7.8%) sid=1831204
    [teach] ORS   92/1172 ( 7.8%) sid=1836322
    [teach] ORS   93/1172 ( 7.9%) sid=1836998
    [teach] ORS   94/1172 ( 8.0%) sid=1837644
    [teach] ORS   95/1172 ( 8.1%) sid=1839877
    [teach] ORS   96/1172 ( 8.2%) sid=1842109
    [teach] ORS   97/1172 ( 8.3%) sid=1843280
    [teach] ORS   98/1172 ( 8.4%) sid=1843943
    [teach] ORS   99/1172 ( 8.4%) sid=1844337
    [teach] ORS  100/1172 ( 8.5%) sid=1845854
    [teach] exporting parquet snapshot ...
    [teach] ORS  101/1172 ( 8.6%) sid=1852766
    [teach] ORS  102/1172 ( 8.7%) sid=1853663
    [teach] ORS  103/1172 ( 8.8%) sid=1855864
    [teach] ORS  104/1172 ( 8.9%) sid=1857646
    [teach] ORS  105/1172 ( 9.0%) sid=1858743
    [teach] ORS  106/1172 ( 9.0%) sid=1859002
    [teach] ORS  107/1172 ( 9.1%) sid=1879383
    [teach] ORS  108/1172 ( 9.2%) sid=1879932
    [teach] ORS  109/1172 ( 9.3%) sid=1880107
    [teach] ORS  110/1172 ( 9.4%) sid=1881046
    [teach] ORS  111/1172 ( 9.5%) sid=1883166
    [teach] ORS  112/1172 ( 9.6%) sid=1885084
    [teach] ORS  113/1172 ( 9.6%) sid=1885793
    [teach] ORS  114/1172 ( 9.7%) sid=1886560
    [teach] ORS  115/1172 ( 9.8%) sid=1886738
    [teach] ORS  116/1172 ( 9.9%) sid=1895103
    [teach] ORS  117/1172 (10.0%) sid=1896280
    [teach] ORS  118/1172 (10.1%) sid=1900968
    [teach] ORS  119/1172 (10.2%) sid=1901903
    [teach] ORS  120/1172 (10.2%) sid=1902519
    [teach] ORS  121/1172 (10.3%) sid=1907788
    [teach] ORS  122/1172 (10.4%) sid=1915153
    [teach] ORS  123/1172 (10.5%) sid=1915189
    [teach] ORS  124/1172 (10.6%) sid=1923351
    [teach] ORS  125/1172 (10.7%) sid=1924835
    [teach] exporting parquet snapshot ...
    [teach] ORS  126/1172 (10.8%) sid=1926606
    [teach] ORS  127/1172 (10.8%) sid=1928437
    [teach] ORS  128/1172 (10.9%) sid=1928729
    [teach] ORS  129/1172 (11.0%) sid=1929758
    [teach] ORS  130/1172 (11.1%) sid=1934377
    [teach] ORS  131/1172 (11.2%) sid=1946174
    [teach] ORS  132/1172 (11.3%) sid=1951855
    [teach] ORS  133/1172 (11.3%) sid=1952904
    [teach] ORS  134/1172 (11.4%) sid=1954684
    [teach] ORS  135/1172 (11.5%) sid=1956682
    [teach] ORS  136/1172 (11.6%) sid=1967784
    [teach] ORS  137/1172 (11.7%) sid=1972873
    [teach] ORS  138/1172 (11.8%) sid=1973082
    [teach] ORS  139/1172 (11.9%) sid=1982321
    [teach] ORS  140/1172 (11.9%) sid=1997055
    [teach] ORS  141/1172 (12.0%) sid=1998542
    [teach] ORS  142/1172 (12.1%) sid=2002103
    [teach] ORS  143/1172 (12.2%) sid=2003304
    [teach] ORS  144/1172 (12.3%) sid=2007118
    [teach] ORS  145/1172 (12.4%) sid=2012647
    [teach] ORS  146/1172 (12.5%) sid=2020491
    [teach] ORS  147/1172 (12.5%) sid=2024860
    [teach] ORS  148/1172 (12.6%) sid=2036781
    [teach] ORS  149/1172 (12.7%) sid=2044082
    [teach] ORS  150/1172 (12.8%) sid=2045529
    [teach] exporting parquet snapshot ...
    [teach] ORS  151/1172 (12.9%) sid=2066782
    [teach] ORS  152/1172 (13.0%) sid=2069601
    [teach] ORS  153/1172 (13.1%) sid=2072453
    [teach] ORS  154/1172 (13.1%) sid=2086667
    [teach] ORS  155/1172 (13.2%) sid=2091567
    [teach] ORS  156/1172 (13.3%) sid=2111465
    [teach] ORS  157/1172 (13.4%) sid=2131690
    [teach] ORS  158/1172 (13.5%) sid=2146379
    [teach] ORS  159/1172 (13.6%) sid=2152524
    [teach] ORS  160/1172 (13.7%) sid=2152598
    [teach] ORS  161/1172 (13.7%) sid=2177340
    [teach] ORS  162/1172 (13.8%) sid=2189515
    [teach] ORS  163/1172 (13.9%) sid=2216509
    [teach] ORS  164/1172 (14.0%) sid=2221775
    [teach] ORS  165/1172 (14.1%) sid=2224575
    [teach] ORS  166/1172 (14.2%) sid=2250857
    [teach] ORS  167/1172 (14.2%) sid=2263716
    [teach] ORS  168/1172 (14.3%) sid=2267943
    [teach] ORS  169/1172 (14.4%) sid=2269666
    [teach] ORS  170/1172 (14.5%) sid=2279788
    [teach] ORS  171/1172 (14.6%) sid=2288554
    [teach] ORS  172/1172 (14.7%) sid=2288789
    [teach] ORS  173/1172 (14.8%) sid=2289044
    [teach] ORS  174/1172 (14.8%) sid=2289841
    [teach] ORS  175/1172 (14.9%) sid=2291926
    [teach] exporting parquet snapshot ...
    [teach] ORS  176/1172 (15.0%) sid=2319343
    [teach] ORS  177/1172 (15.1%) sid=2356098
    [teach] ORS  178/1172 (15.2%) sid=2358377
    [teach] ORS  179/1172 (15.3%) sid=2361727
    [teach] ORS  180/1172 (15.4%) sid=2364323
    [teach] ORS  181/1172 (15.4%) sid=2368429
    [teach] ORS  182/1172 (15.5%) sid=2370471
    [teach] ORS  183/1172 (15.6%) sid=2389145
    [teach] ORS  184/1172 (15.7%) sid=2394329
    [teach] ORS  185/1172 (15.8%) sid=2413381
    [teach] ORS  186/1172 (15.9%) sid=2496628
    [teach] ORS  187/1172 (16.0%) sid=2504817
    [teach] ORS  188/1172 (16.0%) sid=2505491
    [teach] ORS  189/1172 (16.1%) sid=2505507
    [teach] ORS  190/1172 (16.2%) sid=2516532
    [teach] ORS  191/1172 (16.3%) sid=2537218
    [teach] ORS  192/1172 (16.4%) sid=2541535
    [teach] ORS  193/1172 (16.5%) sid=2542835
    [teach] ORS  194/1172 (16.6%) sid=2550561
    [teach] ORS  195/1172 (16.6%) sid=2560674
    [teach] ORS  196/1172 (16.7%) sid=2571326
    [teach] ORS  197/1172 (16.8%) sid=2595874
    [teach] ORS  198/1172 (16.9%) sid=2617274
    [teach] ORS  199/1172 (17.0%) sid=2621294
    [teach] ORS  200/1172 (17.1%) sid=2630198
    [teach] exporting parquet snapshot ...
    [teach] ORS  201/1172 (17.2%) sid=2639031
    [teach] ORS  202/1172 (17.2%) sid=2658090
    [teach] ORS  203/1172 (17.3%) sid=2666474
    [teach] ORS  204/1172 (17.4%) sid=2676147
    [teach] ORS  205/1172 (17.5%) sid=2706225
    [teach] ORS  206/1172 (17.6%) sid=2710185
    [teach] ORS  207/1172 (17.7%) sid=2722660
    [teach] ORS  208/1172 (17.7%) sid=2725884
    [teach] ORS  209/1172 (17.8%) sid=2731164
    [teach] ORS  210/1172 (17.9%) sid=2734400
    [teach] ORS  211/1172 (18.0%) sid=2749519
    [teach] ORS  212/1172 (18.1%) sid=2750783
    [teach] ORS  213/1172 (18.2%) sid=2770866
    [teach] ORS  214/1172 (18.3%) sid=2812620
    [teach] ORS  215/1172 (18.3%) sid=2819003
    [teach] ORS  216/1172 (18.4%) sid=2900786
    [teach] ORS  217/1172 (18.5%) sid=2901435
    [teach] ORS  218/1172 (18.6%) sid=2902359
    [teach] ORS  219/1172 (18.7%) sid=2919176
    [teach] ORS  220/1172 (18.8%) sid=2922361
    [teach] ORS  221/1172 (18.9%) sid=2925485
    [teach] ORS  222/1172 (18.9%) sid=2954210
    [teach] ORS  223/1172 (19.0%) sid=2955335
    [teach] ORS  224/1172 (19.1%) sid=2970205
    [teach] ORS  225/1172 (19.2%) sid=2983799
    [teach] exporting parquet snapshot ...
    [teach] ORS  226/1172 (19.3%) sid=2989700
    [teach] ORS  227/1172 (19.4%) sid=2989716
    [teach] ORS  228/1172 (19.5%) sid=2991114
    [teach] ORS  229/1172 (19.5%) sid=2994731
    [teach] ORS  230/1172 (19.6%) sid=3000345
    [teach] ORS  231/1172 (19.7%) sid=3002001
    [teach] ORS  232/1172 (19.8%) sid=3078594
    [teach] ORS  233/1172 (19.9%) sid=3093479
    [teach] ORS  234/1172 (20.0%) sid=3134843
    [teach] ORS  235/1172 (20.1%) sid=3188925
    [teach] ORS  236/1172 (20.1%) sid=3225910
    [teach] ORS  237/1172 (20.2%) sid=3259587
    [teach] ORS  238/1172 (20.3%) sid=3259637
    [teach] ORS  239/1172 (20.4%) sid=3271344
    [teach] ORS  240/1172 (20.5%) sid=3304183
    [teach] ORS  241/1172 (20.6%) sid=3306275
    [teach] ORS  242/1172 (20.6%) sid=3313275
    [teach] ORS  243/1172 (20.7%) sid=3323361
    [teach] ORS  244/1172 (20.8%) sid=3342983
    [teach] ORS  245/1172 (20.9%) sid=3343555
    [teach] ORS  246/1172 (21.0%) sid=3359178
    [teach] ORS  247/1172 (21.1%) sid=3370407
    [teach] ORS  248/1172 (21.2%) sid=3403789
    [teach] ORS  249/1172 (21.2%) sid=3421948
    [teach] ORS  250/1172 (21.3%) sid=3436995
    [teach] exporting parquet snapshot ...
    [teach] ORS  251/1172 (21.4%) sid=3437471
    [teach] ORS  252/1172 (21.5%) sid=3449138
    [teach] ORS  253/1172 (21.6%) sid=3453672
    [teach] ORS  254/1172 (21.7%) sid=3474163
    [teach] ORS  255/1172 (21.8%) sid=3492091
    [teach] ORS  256/1172 (21.8%) sid=3504298
    [teach] ORS  257/1172 (21.9%) sid=3511159
    [teach] ORS  258/1172 (22.0%) sid=3512443
    [teach] ORS  259/1172 (22.1%) sid=3530734
    [teach] ORS  260/1172 (22.2%) sid=3534379
    [teach] ORS  261/1172 (22.3%) sid=3544732
    [teach] ORS  262/1172 (22.4%) sid=3550778
    [teach] ORS  263/1172 (22.4%) sid=3553088
    [teach] ORS  264/1172 (22.5%) sid=3579639
    [teach] ORS  265/1172 (22.6%) sid=3587506
    [teach] ORS  266/1172 (22.7%) sid=3590256
    [teach] ORS  267/1172 (22.8%) sid=3604358
    [teach] ORS  268/1172 (22.9%) sid=3613915
    [teach] ORS  269/1172 (23.0%) sid=3619928
    [teach] ORS  270/1172 (23.0%) sid=3626036
    [teach] ORS  271/1172 (23.1%) sid=3634244
    [teach] ORS  272/1172 (23.2%) sid=3638776
    [teach] ORS  273/1172 (23.3%) sid=3642460
    [teach] ORS  274/1172 (23.4%) sid=3649192
    [teach] ORS  275/1172 (23.5%) sid=3677675
    [teach] exporting parquet snapshot ...
    [teach] ORS  276/1172 (23.5%) sid=3690034
    [teach] ORS  277/1172 (23.6%) sid=3709559
    [teach] ORS  278/1172 (23.7%) sid=3718802
    [teach] ORS  279/1172 (23.8%) sid=3719978
    [teach] ORS  280/1172 (23.9%) sid=3724473
    [teach] ORS  281/1172 (24.0%) sid=3742697
    [teach] ORS  282/1172 (24.1%) sid=3757373
    [teach] ORS  283/1172 (24.1%) sid=3767738
    [teach] ORS  284/1172 (24.2%) sid=3768137
    [teach] ORS  285/1172 (24.3%) sid=3771367
    [teach] ORS  286/1172 (24.4%) sid=3782291
    [teach] ORS  287/1172 (24.5%) sid=3800089
    [teach] ORS  288/1172 (24.6%) sid=3806040
    [teach] ORS  289/1172 (24.7%) sid=3808136
    [teach] ORS  290/1172 (24.7%) sid=3808890
    [teach] ORS  291/1172 (24.8%) sid=3810107
    [teach] ORS  292/1172 (24.9%) sid=3811672
    [teach] ORS  293/1172 (25.0%) sid=3815079
    [teach] ORS  294/1172 (25.1%) sid=3818120
    [teach] ORS  295/1172 (25.2%) sid=3819802
    [teach] ORS  296/1172 (25.3%) sid=3828746
    [teach] ORS  297/1172 (25.3%) sid=3840974
    [teach] ORS  298/1172 (25.4%) sid=3860883
    [teach] ORS  299/1172 (25.5%) sid=3871088
    [teach] ORS  300/1172 (25.6%) sid=3875026
    [teach] exporting parquet snapshot ...
    [teach] ORS  301/1172 (25.7%) sid=3891229
    [teach] ORS  302/1172 (25.8%) sid=3895264
    [teach] ORS  303/1172 (25.9%) sid=3907158
    [teach] ORS  304/1172 (25.9%) sid=3914671
    [teach] ORS  305/1172 (26.0%) sid=3916390
    [teach] ORS  306/1172 (26.1%) sid=3925410
    [teach] ORS  307/1172 (26.2%) sid=3930687
    [teach] ORS  308/1172 (26.3%) sid=3932553
    [teach] ORS  309/1172 (26.4%) sid=3933670
    [teach] ORS  310/1172 (26.5%) sid=3937885
    [teach] ORS  311/1172 (26.5%) sid=3945122
    [teach] ORS  312/1172 (26.6%) sid=3945661
    [teach] ORS  313/1172 (26.7%) sid=3958428
    [teach] ORS  314/1172 (26.8%) sid=3991560
    [teach] ORS  315/1172 (26.9%) sid=3992532
    [teach] ORS  316/1172 (27.0%) sid=3994469
    [teach] ORS  317/1172 (27.0%) sid=3994821
    [teach] ORS  318/1172 (27.1%) sid=3999834
    [teach] ORS  319/1172 (27.2%) sid=4000845
    [teach] ORS  320/1172 (27.3%) sid=4001524
    [teach] ORS  321/1172 (27.4%) sid=4001762
    [teach] ORS  322/1172 (27.5%) sid=4003228
    [teach] ORS  323/1172 (27.6%) sid=4007889
    [teach] ORS  324/1172 (27.6%) sid=4007926
    [teach] ORS  325/1172 (27.7%) sid=4008161
    [teach] exporting parquet snapshot ...
    [teach] ORS  326/1172 (27.8%) sid=4016763
    [teach] ORS  327/1172 (27.9%) sid=4029988
    [teach] ORS  328/1172 (28.0%) sid=4036553
    [teach] ORS  329/1172 (28.1%) sid=4043093
    [teach] ORS  330/1172 (28.2%) sid=4043459
    [teach] ORS  331/1172 (28.2%) sid=4046121
    [teach] ORS  332/1172 (28.3%) sid=4046654
    [teach] ORS  333/1172 (28.4%) sid=4067186
    [teach] ORS  334/1172 (28.5%) sid=4068608
    [teach] ORS  335/1172 (28.6%) sid=4077109
    [teach] ORS  336/1172 (28.7%) sid=4081735
    [teach] ORS  337/1172 (28.8%) sid=4085328
    [teach] ORS  338/1172 (28.8%) sid=4088088
    [teach] ORS  339/1172 (28.9%) sid=4098786
    [teach] ORS  340/1172 (29.0%) sid=4100694
    [teach] ORS  341/1172 (29.1%) sid=4112976
    [teach] ORS  342/1172 (29.2%) sid=4130877
    [teach] ORS  343/1172 (29.3%) sid=4139946
    [teach] ORS  344/1172 (29.4%) sid=4144539
    [teach] ORS  345/1172 (29.4%) sid=4151827
    [teach] ORS  346/1172 (29.5%) sid=4154088
    [teach] ORS  347/1172 (29.6%) sid=4154794
    [teach] ORS  348/1172 (29.7%) sid=4156880
    [teach] ORS  349/1172 (29.8%) sid=4171422
    [teach] ORS  350/1172 (29.9%) sid=4188829
    [teach] exporting parquet snapshot ...
    [teach] ORS  351/1172 (29.9%) sid=4197803
    [teach] ORS  352/1172 (30.0%) sid=4224342
    [teach] ORS  353/1172 (30.1%) sid=4228275
    [teach] ORS  354/1172 (30.2%) sid=4241046
    [teach] ORS  355/1172 (30.3%) sid=4241761
    [teach] ORS  356/1172 (30.4%) sid=4243345
    [teach] ORS  357/1172 (30.5%) sid=4247247
    [teach] ORS  358/1172 (30.5%) sid=4256887
    [teach] ORS  359/1172 (30.6%) sid=4262897
    [teach] ORS  360/1172 (30.7%) sid=4270746
    [teach] ORS  361/1172 (30.8%) sid=4275664
    [teach] ORS  362/1172 (30.9%) sid=4275867
    [teach] ORS  363/1172 (31.0%) sid=4279664
    [teach] ORS  364/1172 (31.1%) sid=4280374
    [teach] ORS  365/1172 (31.1%) sid=4285339
    [teach] ORS  366/1172 (31.2%) sid=4289132
    [teach] ORS  367/1172 (31.3%) sid=4289812
    [teach] ORS  368/1172 (31.4%) sid=4300458
    [teach] ORS  369/1172 (31.5%) sid=4325298
    [teach] ORS  370/1172 (31.6%) sid=4325999
    [teach] ORS  371/1172 (31.7%) sid=4350735
    [teach] ORS  372/1172 (31.7%) sid=4353507
    [teach] ORS  373/1172 (31.8%) sid=4358631
    [teach] ORS  374/1172 (31.9%) sid=4360282
    [teach] ORS  375/1172 (32.0%) sid=4362098
    [teach] exporting parquet snapshot ...
    [teach] ORS  376/1172 (32.1%) sid=4377168
    [teach] ORS  377/1172 (32.2%) sid=4388201
    [teach] ORS  378/1172 (32.3%) sid=4394254
    [teach] ORS  379/1172 (32.3%) sid=4395980
    [teach] ORS  380/1172 (32.4%) sid=4403200
    [teach] ORS  381/1172 (32.5%) sid=4404387
    [teach] ORS  382/1172 (32.6%) sid=4413569
    [teach] ORS  383/1172 (32.7%) sid=4420531
    [teach] ORS  384/1172 (32.8%) sid=4422992
    [teach] ORS  385/1172 (32.8%) sid=4426407
    [teach] ORS  386/1172 (32.9%) sid=4438887
    [teach] ORS  387/1172 (33.0%) sid=4440104
    [teach] ORS  388/1172 (33.1%) sid=4440923
    [teach] ORS  389/1172 (33.2%) sid=4446449
    [teach] ORS  390/1172 (33.3%) sid=4452472
    [teach] ORS  391/1172 (33.4%) sid=4454587
    [teach] ORS  392/1172 (33.4%) sid=4454858
    [teach] ORS  393/1172 (33.5%) sid=4457960
    [teach] ORS  394/1172 (33.6%) sid=4461493
    [teach] ORS  395/1172 (33.7%) sid=4468530
    [teach] ORS  396/1172 (33.8%) sid=4470802
    [teach] ORS  397/1172 (33.9%) sid=4471020
    [teach] ORS  398/1172 (34.0%) sid=4476408
    [teach] ORS  399/1172 (34.0%) sid=4481151
    [teach] ORS  400/1172 (34.1%) sid=4481684
    [teach] exporting parquet snapshot ...
    [teach] ORS  401/1172 (34.2%) sid=4482078
    [teach] ORS  402/1172 (34.3%) sid=4482161
    [teach] ORS  403/1172 (34.4%) sid=4482547
    [teach] ORS  404/1172 (34.5%) sid=4486923
    [teach] ORS  405/1172 (34.6%) sid=4486995
    [teach] ORS  406/1172 (34.6%) sid=4495090
    [teach] ORS  407/1172 (34.7%) sid=4497661
    [teach] ORS  408/1172 (34.8%) sid=4497749
    [teach] ORS  409/1172 (34.9%) sid=4498925
    [teach] ORS  410/1172 (35.0%) sid=4503129
    [teach] ORS  411/1172 (35.1%) sid=4504151
    [teach] ORS  412/1172 (35.2%) sid=4505585
    [teach] ORS  413/1172 (35.2%) sid=4508258
    [teach] ORS  414/1172 (35.3%) sid=4513706
    [teach] ORS  415/1172 (35.4%) sid=4516031
    [teach] ORS  416/1172 (35.5%) sid=4518003
    [teach] ORS  417/1172 (35.6%) sid=4522442
    [teach] ORS  418/1172 (35.7%) sid=4524989
    [teach] ORS  419/1172 (35.8%) sid=4530951
    [teach] ORS  420/1172 (35.8%) sid=4531450
    [teach] ORS  421/1172 (35.9%) sid=4531928
    [teach] ORS  422/1172 (36.0%) sid=4533384
    [teach] ORS  423/1172 (36.1%) sid=4533746
    [teach] ORS  424/1172 (36.2%) sid=4535899
    [teach] ORS  425/1172 (36.3%) sid=4539249
    [teach] exporting parquet snapshot ...
    [teach] ORS  426/1172 (36.3%) sid=4541745
    [teach] ORS  427/1172 (36.4%) sid=4545597
    [teach] ORS  428/1172 (36.5%) sid=4552659
    [teach] ORS  429/1172 (36.6%) sid=4554937
    [teach] ORS  430/1172 (36.7%) sid=4557615
    [teach] ORS  431/1172 (36.8%) sid=4561725
    [teach] ORS  432/1172 (36.9%) sid=4565492
    [teach] ORS  433/1172 (36.9%) sid=4585741
    [teach] ORS  434/1172 (37.0%) sid=4587269
    [teach] ORS  435/1172 (37.1%) sid=4600308
    [teach] ORS  436/1172 (37.2%) sid=4613546
    [teach] ORS  437/1172 (37.3%) sid=4613664
    [teach] ORS  438/1172 (37.4%) sid=4613944
    [teach] ORS  439/1172 (37.5%) sid=4618983
    [teach] ORS  440/1172 (37.5%) sid=4619077
    [teach] ORS  441/1172 (37.6%) sid=4620608
    [teach] ORS  442/1172 (37.7%) sid=4620610
    [teach] ORS  443/1172 (37.8%) sid=4628084
    [teach] ORS  444/1172 (37.9%) sid=4635833
    [teach] ORS  445/1172 (38.0%) sid=4636930
    [teach] ORS  446/1172 (38.1%) sid=4640148
    [teach] ORS  447/1172 (38.1%) sid=4640676
    [teach] ORS  448/1172 (38.2%) sid=4644643
    [teach] ORS  449/1172 (38.3%) sid=4663089
    [teach] ORS  450/1172 (38.4%) sid=4663106
    [teach] exporting parquet snapshot ...
    [teach] ORS  451/1172 (38.5%) sid=4665440
    [teach] ORS  452/1172 (38.6%) sid=4668060
    [teach] ORS  453/1172 (38.7%) sid=4674827
    [teach] ORS  454/1172 (38.7%) sid=4681174
    [teach] ORS  455/1172 (38.8%) sid=4688211
    [teach] ORS  456/1172 (38.9%) sid=4705951
    [teach] ORS  457/1172 (39.0%) sid=4711742
    [teach] ORS  458/1172 (39.1%) sid=4712266
    [teach] ORS  459/1172 (39.2%) sid=4722413
    [teach] ORS  460/1172 (39.2%) sid=4734739
    [teach] ORS  461/1172 (39.3%) sid=4752081
    [teach] ORS  462/1172 (39.4%) sid=4756211
    [teach] ORS  463/1172 (39.5%) sid=4757273
    [teach] ORS  464/1172 (39.6%) sid=4776302
    [teach] ORS  465/1172 (39.7%) sid=4783052
    [teach] ORS  466/1172 (39.8%) sid=4783725
    [teach] ORS  467/1172 (39.8%) sid=4804101
    [teach] ORS  468/1172 (39.9%) sid=4811212
    [teach] ORS  469/1172 (40.0%) sid=4812176
    [teach] ORS  470/1172 (40.1%) sid=4826027
    [teach] ORS  471/1172 (40.2%) sid=4829450
    [teach] ORS  472/1172 (40.3%) sid=4837199
    [teach] ORS  473/1172 (40.4%) sid=4838570
    [teach] ORS  474/1172 (40.4%) sid=4850872
    [teach] ORS  475/1172 (40.5%) sid=4859257
    [teach] exporting parquet snapshot ...
    [teach] ORS  476/1172 (40.6%) sid=4863374
    [teach] ORS  477/1172 (40.7%) sid=4869884
    [teach] ORS  478/1172 (40.8%) sid=4871000
    [teach] ORS  479/1172 (40.9%) sid=4871411
    [teach] ORS  480/1172 (41.0%) sid=4874500
    [teach] ORS  481/1172 (41.0%) sid=4885929
    [teach] ORS  482/1172 (41.1%) sid=4891046
    [teach] ORS  483/1172 (41.2%) sid=4898707
    [teach] ORS  484/1172 (41.3%) sid=4898888
    [teach] ORS  485/1172 (41.4%) sid=4915092
    [teach] ORS  486/1172 (41.5%) sid=4922448
    [teach] ORS  487/1172 (41.6%) sid=4931199
    [teach] ORS  488/1172 (41.6%) sid=4973057
    [teach] ORS  489/1172 (41.7%) sid=4974198
    [teach] ORS  490/1172 (41.8%) sid=4980920
    [teach] ORS  491/1172 (41.9%) sid=4982038
    [teach] ORS  492/1172 (42.0%) sid=4983209
    [teach] ORS  493/1172 (42.1%) sid=4986773
    [teach] ORS  494/1172 (42.2%) sid=4988002
    [teach] ORS  495/1172 (42.2%) sid=5019531
    [teach] ORS  496/1172 (42.3%) sid=5030895
    [teach] ORS  497/1172 (42.4%) sid=5039418
    [teach] ORS  498/1172 (42.5%) sid=5040100
    [teach] ORS  499/1172 (42.6%) sid=5057855
    [teach] ORS  500/1172 (42.7%) sid=5058732
    [teach] exporting parquet snapshot ...
    [teach] ORS  501/1172 (42.7%) sid=5079383
    [teach] ORS  502/1172 (42.8%) sid=5080852
    [teach] ORS  503/1172 (42.9%) sid=5086286
    [teach] ORS  504/1172 (43.0%) sid=5088859
    [teach] ORS  505/1172 (43.1%) sid=5097344
    [teach] ORS  506/1172 (43.2%) sid=5097725
    [teach] ORS  507/1172 (43.3%) sid=5097950
    [teach] ORS  508/1172 (43.3%) sid=5110948
    [teach] ORS  509/1172 (43.4%) sid=5126949
    [teach] ORS  510/1172 (43.5%) sid=5127458
    [teach] ORS  511/1172 (43.6%) sid=5137866
    [teach] ORS  512/1172 (43.7%) sid=5139231
    [teach] ORS  513/1172 (43.8%) sid=5143122
    [teach] ORS  514/1172 (43.9%) sid=5148163
    [teach] ORS  515/1172 (43.9%) sid=5148444
    [teach] ORS  516/1172 (44.0%) sid=5153434
    [teach] ORS  517/1172 (44.1%) sid=5153456
    [teach] ORS  518/1172 (44.2%) sid=5154405
    [teach] ORS  519/1172 (44.3%) sid=5169567
    [teach] ORS  520/1172 (44.4%) sid=5184041
    [teach] ORS  521/1172 (44.5%) sid=5187284
    [teach] ORS  522/1172 (44.5%) sid=5190567
    [teach] ORS  523/1172 (44.6%) sid=5212624
    [teach] ORS  524/1172 (44.7%) sid=5251716
    [teach] ORS  525/1172 (44.8%) sid=5258586
    [teach] exporting parquet snapshot ...
    [teach] ORS  526/1172 (44.9%) sid=5265338
    [teach] ORS  527/1172 (45.0%) sid=5289551
    [teach] ORS  528/1172 (45.1%) sid=5339227
    [teach] ORS  529/1172 (45.1%) sid=5360278
    [teach] ORS  530/1172 (45.2%) sid=5368472
    [teach] ORS  531/1172 (45.3%) sid=5376267
    [teach] ORS  532/1172 (45.4%) sid=5378202
    [teach] ORS  533/1172 (45.5%) sid=5407003
    [teach] ORS  534/1172 (45.6%) sid=5415039
    [teach] ORS  535/1172 (45.6%) sid=5425765
    [teach] ORS  536/1172 (45.7%) sid=5432218
    [teach] ORS  537/1172 (45.8%) sid=5466427
    [teach] ORS  538/1172 (45.9%) sid=5473797
    [teach] ORS  539/1172 (46.0%) sid=5478032
    [teach] ORS  540/1172 (46.1%) sid=5492203
    [teach] ORS  541/1172 (46.2%) sid=5493525
    [teach] ORS  542/1172 (46.2%) sid=5496650
    [teach] ORS  543/1172 (46.3%) sid=5521638
    [teach] ORS  544/1172 (46.4%) sid=5554428
    [teach] ORS  545/1172 (46.5%) sid=5577997
    [teach] ORS  546/1172 (46.6%) sid=5597272
    [teach] ORS  547/1172 (46.7%) sid=5603003
    [teach] ORS  548/1172 (46.8%) sid=5609197
    [teach] ORS  549/1172 (46.8%) sid=5634561
    [ORS][warn] sid=5649864 no valid candidates after constraints (mode=dp_prefix, k_span=13..15, dp_q=44, beta=3.0). Trying fallback #1.
    [teach] ORS  550/1172 (46.9%) sid=5649864
    [teach] exporting parquet snapshot ...
    [teach] ORS  551/1172 (47.0%) sid=5660315
    [teach] ORS  552/1172 (47.1%) sid=5673503
    [teach] ORS  553/1172 (47.2%) sid=5686796
    [teach] ORS  554/1172 (47.3%) sid=5687389
    [teach] ORS  555/1172 (47.4%) sid=5698404
    [teach] ORS  556/1172 (47.4%) sid=5711492
    [teach] ORS  557/1172 (47.5%) sid=5732482
    [teach] ORS  558/1172 (47.6%) sid=5734266
    [teach] ORS  559/1172 (47.7%) sid=5778107
    [teach] ORS  560/1172 (47.8%) sid=5791679
    [teach] ORS  561/1172 (47.9%) sid=5795333
    [teach] ORS  562/1172 (48.0%) sid=5813334
    [teach] ORS  563/1172 (48.0%) sid=5814989
    [teach] ORS  564/1172 (48.1%) sid=5823501
    [teach] ORS  565/1172 (48.2%) sid=5824034
    [teach] ORS  566/1172 (48.3%) sid=5829487
    [teach] ORS  567/1172 (48.4%) sid=5845412
    [teach] ORS  568/1172 (48.5%) sid=5850371
    [teach] ORS  569/1172 (48.5%) sid=5852330
    [teach] ORS  570/1172 (48.6%) sid=5858938
    [teach] ORS  571/1172 (48.7%) sid=5871769
    [teach] ORS  572/1172 (48.8%) sid=5880521
    [teach] ORS  573/1172 (48.9%) sid=5907037
    [teach] ORS  574/1172 (49.0%) sid=5919254
    [teach] ORS  575/1172 (49.1%) sid=5924054
    [teach] exporting parquet snapshot ...
    [teach] ORS  576/1172 (49.1%) sid=5933092
    [teach] ORS  577/1172 (49.2%) sid=5937754
    [teach] ORS  578/1172 (49.3%) sid=5945706
    [teach] ORS  579/1172 (49.4%) sid=5987703
    [teach] ORS  580/1172 (49.5%) sid=6031843
    [teach] ORS  581/1172 (49.6%) sid=6061631
    [teach] ORS  582/1172 (49.7%) sid=6074769
    [teach] ORS  583/1172 (49.7%) sid=6097211
    [teach] ORS  584/1172 (49.8%) sid=6106069
    [teach] ORS  585/1172 (49.9%) sid=6106377
    [teach] ORS  586/1172 (50.0%) sid=6123781
    [teach] ORS  587/1172 (50.1%) sid=6129779
    [teach] ORS  588/1172 (50.2%) sid=6174344
    [teach] ORS  589/1172 (50.3%) sid=6220336
    [teach] ORS  590/1172 (50.3%) sid=6236061
    [teach] ORS  591/1172 (50.4%) sid=6257154
    [teach] ORS  592/1172 (50.5%) sid=6288636
    [teach] ORS  593/1172 (50.6%) sid=6311549
    [teach] ORS  594/1172 (50.7%) sid=6322698
    [teach] ORS  595/1172 (50.8%) sid=6351033
    [teach] ORS  596/1172 (50.9%) sid=6356776
    [teach] ORS  597/1172 (50.9%) sid=6362388
    [teach] ORS  598/1172 (51.0%) sid=6363419
    [teach] ORS  599/1172 (51.1%) sid=6366435
    [teach] ORS  600/1172 (51.2%) sid=6373507
    [teach] exporting parquet snapshot ...
    [teach] ORS  601/1172 (51.3%) sid=6386378
    [teach] ORS  602/1172 (51.4%) sid=6406761
    [teach] ORS  603/1172 (51.5%) sid=6408528
    [teach] ORS  604/1172 (51.5%) sid=6409686
    [teach] ORS  605/1172 (51.6%) sid=6413152
    [teach] ORS  606/1172 (51.7%) sid=6413520
    [teach] ORS  607/1172 (51.8%) sid=6418610
    [teach] ORS  608/1172 (51.9%) sid=6431322
    [teach] ORS  609/1172 (52.0%) sid=6437298
    [teach] ORS  610/1172 (52.0%) sid=6455445
    [teach] ORS  611/1172 (52.1%) sid=6492394
    [teach] ORS  612/1172 (52.2%) sid=6495681
    [teach] ORS  613/1172 (52.3%) sid=6496954
    [teach] ORS  614/1172 (52.4%) sid=6505800
    [teach] ORS  615/1172 (52.5%) sid=6508485
    [teach] ORS  616/1172 (52.6%) sid=6512761
    [teach] ORS  617/1172 (52.6%) sid=6527736
    [teach] ORS  618/1172 (52.7%) sid=6534650
    [teach] ORS  619/1172 (52.8%) sid=6537802
    [teach] ORS  620/1172 (52.9%) sid=6571778
    [teach] ORS  621/1172 (53.0%) sid=6573661
    [teach] ORS  622/1172 (53.1%) sid=6580285
    [teach] ORS  623/1172 (53.2%) sid=6594054
    [teach] ORS  624/1172 (53.2%) sid=6596088
    [teach] ORS  625/1172 (53.3%) sid=6613180
    [teach] exporting parquet snapshot ...
    [teach] ORS  626/1172 (53.4%) sid=6648051
    [teach] ORS  627/1172 (53.5%) sid=6676101
    [teach] ORS  628/1172 (53.6%) sid=6677278
    [teach] ORS  629/1172 (53.7%) sid=6679791
    [teach] ORS  630/1172 (53.8%) sid=6681643
    [teach] ORS  631/1172 (53.8%) sid=6684959
    [teach] ORS  632/1172 (53.9%) sid=6711834
    [teach] ORS  633/1172 (54.0%) sid=6721641
    [teach] ORS  634/1172 (54.1%) sid=6735439
    [teach] ORS  635/1172 (54.2%) sid=6736547
    [teach] ORS  636/1172 (54.3%) sid=6738359
    [teach] ORS  637/1172 (54.4%) sid=6747437
    [teach] ORS  638/1172 (54.4%) sid=6748348
    [teach] ORS  639/1172 (54.5%) sid=6749919
    [teach] ORS  640/1172 (54.6%) sid=6776966
    [teach] ORS  641/1172 (54.7%) sid=6780248
    [teach] ORS  642/1172 (54.8%) sid=6789138
    [teach] ORS  643/1172 (54.9%) sid=6811612
    [teach] ORS  644/1172 (54.9%) sid=6813569
    [teach] ORS  645/1172 (55.0%) sid=6823722
    [teach] ORS  646/1172 (55.1%) sid=6824351
    [teach] ORS  647/1172 (55.2%) sid=6825354
    [teach] ORS  648/1172 (55.3%) sid=6830580
    [teach] ORS  649/1172 (55.4%) sid=6833914
    [teach] ORS  650/1172 (55.5%) sid=6869457
    [teach] exporting parquet snapshot ...
    [teach] ORS  651/1172 (55.5%) sid=6877871
    [teach] ORS  652/1172 (55.6%) sid=6878964
    [teach] ORS  653/1172 (55.7%) sid=6893908
    [teach] ORS  654/1172 (55.8%) sid=6935315
    [teach] ORS  655/1172 (55.9%) sid=6945950
    [teach] ORS  656/1172 (56.0%) sid=6988499
    [teach] ORS  657/1172 (56.1%) sid=7004749
    [teach] ORS  658/1172 (56.1%) sid=7014997
    [teach] ORS  659/1172 (56.2%) sid=7037725
    [teach] ORS  660/1172 (56.3%) sid=7044746
    [teach] ORS  661/1172 (56.4%) sid=7046147
    [teach] ORS  662/1172 (56.5%) sid=7057639
    [teach] ORS  663/1172 (56.6%) sid=7066922
    [teach] ORS  664/1172 (56.7%) sid=7080921
    [teach] ORS  665/1172 (56.7%) sid=7109843
    [teach] ORS  666/1172 (56.8%) sid=7112357
    [teach] ORS  667/1172 (56.9%) sid=7122389
    [teach] ORS  668/1172 (57.0%) sid=7124465
    [teach] ORS  669/1172 (57.1%) sid=7131229
    [teach] ORS  670/1172 (57.2%) sid=7134894
    [teach] ORS  671/1172 (57.3%) sid=7169280
    [teach] ORS  672/1172 (57.3%) sid=7174096
    [teach] ORS  673/1172 (57.4%) sid=7209446
    [teach] ORS  674/1172 (57.5%) sid=7211821
    [teach] ORS  675/1172 (57.6%) sid=7211838
    [teach] exporting parquet snapshot ...
    [teach] ORS  676/1172 (57.7%) sid=7213262
    [teach] ORS  677/1172 (57.8%) sid=7216489
    [teach] ORS  678/1172 (57.8%) sid=7220453
    [teach] ORS  679/1172 (57.9%) sid=7225009
    [teach] ORS  680/1172 (58.0%) sid=7230626
    [teach] ORS  681/1172 (58.1%) sid=7241589
    [teach] ORS  682/1172 (58.2%) sid=7246534
    [teach] ORS  683/1172 (58.3%) sid=7258277
    [teach] ORS  684/1172 (58.4%) sid=7312689
    [teach] ORS  685/1172 (58.4%) sid=7312898
    [teach] ORS  686/1172 (58.5%) sid=7330343
    [teach] ORS  687/1172 (58.6%) sid=7330749
    [teach] ORS  688/1172 (58.7%) sid=7360374
    [teach] ORS  689/1172 (58.8%) sid=7364030
    [teach] ORS  690/1172 (58.9%) sid=7367260
    [teach] ORS  691/1172 (59.0%) sid=7372716
    [teach] ORS  692/1172 (59.0%) sid=7374959
    [teach] ORS  693/1172 (59.1%) sid=7378822
    [teach] ORS  694/1172 (59.2%) sid=7381648
    [teach] ORS  695/1172 (59.3%) sid=7393270
    [teach] ORS  696/1172 (59.4%) sid=7403290
    [teach] ORS  697/1172 (59.5%) sid=7408373
    [teach] ORS  698/1172 (59.6%) sid=7414065
    [teach] ORS  699/1172 (59.6%) sid=7420252
    [teach] ORS  700/1172 (59.7%) sid=7482633
    [teach] exporting parquet snapshot ...
    [teach] ORS  701/1172 (59.8%) sid=7518853
    [teach] ORS  702/1172 (59.9%) sid=7525040
    [teach] ORS  703/1172 (60.0%) sid=7540722
    [teach] ORS  704/1172 (60.1%) sid=7547343
    [teach] ORS  705/1172 (60.2%) sid=7548077
    [teach] ORS  706/1172 (60.2%) sid=7551082
    [teach] ORS  707/1172 (60.3%) sid=7573935
    [teach] ORS  708/1172 (60.4%) sid=7580799
    [teach] ORS  709/1172 (60.5%) sid=7583288
    [teach] ORS  710/1172 (60.6%) sid=7587787
    [teach] ORS  711/1172 (60.7%) sid=7604237
    [teach] ORS  712/1172 (60.8%) sid=7609062
    [teach] ORS  713/1172 (60.8%) sid=7638861
    [teach] ORS  714/1172 (60.9%) sid=7693025
    [teach] ORS  715/1172 (61.0%) sid=7700810
    [teach] ORS  716/1172 (61.1%) sid=7701819
    [teach] ORS  717/1172 (61.2%) sid=7704522
    [teach] ORS  718/1172 (61.3%) sid=7714277
    [teach] ORS  719/1172 (61.3%) sid=7715555
    [teach] ORS  720/1172 (61.4%) sid=7720133
    [teach] ORS  721/1172 (61.5%) sid=7722025
    [teach] ORS  722/1172 (61.6%) sid=7723314
    [teach] ORS  723/1172 (61.7%) sid=7727209
    [teach] ORS  724/1172 (61.8%) sid=7727681
    [teach] ORS  725/1172 (61.9%) sid=7728026
    [teach] exporting parquet snapshot ...
    [teach] ORS  726/1172 (61.9%) sid=7729981
    [teach] ORS  727/1172 (62.0%) sid=7736569
    [teach] ORS  728/1172 (62.1%) sid=7741850
    [teach] ORS  729/1172 (62.2%) sid=7753775
    [teach] ORS  730/1172 (62.3%) sid=7766446
    [teach] ORS  731/1172 (62.4%) sid=7776575
    [teach] ORS  732/1172 (62.5%) sid=7778309
    [teach] ORS  733/1172 (62.5%) sid=7786552
    [teach] ORS  734/1172 (62.6%) sid=7786991
    [teach] ORS  735/1172 (62.7%) sid=7787953
    [teach] ORS  736/1172 (62.8%) sid=7789717
    [teach] ORS  737/1172 (62.9%) sid=7795787
    [teach] ORS  738/1172 (63.0%) sid=7811536
    [teach] ORS  739/1172 (63.1%) sid=7812714
    [teach] ORS  740/1172 (63.1%) sid=7813535
    [teach] ORS  741/1172 (63.2%) sid=7816707
    [teach] ORS  742/1172 (63.3%) sid=7820673
    [teach] ORS  743/1172 (63.4%) sid=7821800
    [teach] ORS  744/1172 (63.5%) sid=7829172
    [teach] ORS  745/1172 (63.6%) sid=7833577
    [teach] ORS  746/1172 (63.7%) sid=7838923
    [teach] ORS  747/1172 (63.7%) sid=7839408
    [teach] ORS  748/1172 (63.8%) sid=7839751
    [teach] ORS  749/1172 (63.9%) sid=7844933
    [teach] ORS  750/1172 (64.0%) sid=7870600
    [teach] exporting parquet snapshot ...
    [teach] ORS  751/1172 (64.1%) sid=7878196
    [teach] ORS  752/1172 (64.2%) sid=7881836
    [teach] ORS  753/1172 (64.2%) sid=7885527
    [teach] ORS  754/1172 (64.3%) sid=7893168
    [teach] ORS  755/1172 (64.4%) sid=7901433
    [teach] ORS  756/1172 (64.5%) sid=7918634
    [teach] ORS  757/1172 (64.6%) sid=7923282
    [teach] ORS  758/1172 (64.7%) sid=7929243
    [teach] ORS  759/1172 (64.8%) sid=7930828
    [teach] ORS  760/1172 (64.8%) sid=7935375
    [teach] ORS  761/1172 (64.9%) sid=7935455
    [teach] ORS  762/1172 (65.0%) sid=7938356
    [teach] ORS  763/1172 (65.1%) sid=7948362
    [teach] ORS  764/1172 (65.2%) sid=7949006
    [teach] ORS  765/1172 (65.3%) sid=7955030
    [teach] ORS  766/1172 (65.4%) sid=7956593
    [teach] ORS  767/1172 (65.4%) sid=7965585
    [teach] ORS  768/1172 (65.5%) sid=7970522
    [teach] ORS  769/1172 (65.6%) sid=7973585
    [teach] ORS  770/1172 (65.7%) sid=7980429
    [teach] ORS  771/1172 (65.8%) sid=7985148
    [teach] ORS  772/1172 (65.9%) sid=7987503
    [teach] ORS  773/1172 (66.0%) sid=7990289
    [teach] ORS  774/1172 (66.0%) sid=8003254
    [teach] ORS  775/1172 (66.1%) sid=8004031
    [teach] exporting parquet snapshot ...
    [teach] ORS  776/1172 (66.2%) sid=8013698
    [teach] ORS  777/1172 (66.3%) sid=8015343
    [teach] ORS  778/1172 (66.4%) sid=8026139
    [teach] ORS  779/1172 (66.5%) sid=8031653
    [teach] ORS  780/1172 (66.6%) sid=8051526
    [teach] ORS  781/1172 (66.6%) sid=8055151
    [ORS][warn] sid=8061977 no valid candidates after constraints (mode=dp_prefix, k_span=13..17, dp_q=32, beta=3.0). Trying fallback #1.
    [teach] ORS  782/1172 (66.7%) sid=8061977
    [teach] ORS  783/1172 (66.8%) sid=8066498
    [teach] ORS  784/1172 (66.9%) sid=8077204
    [teach] ORS  785/1172 (67.0%) sid=8105362
    [teach] ORS  786/1172 (67.1%) sid=8107395
    [teach] ORS  787/1172 (67.2%) sid=8110415
    [teach] ORS  788/1172 (67.2%) sid=8116330
    [teach] ORS  789/1172 (67.3%) sid=8138518
    [teach] ORS  790/1172 (67.4%) sid=8138707
    [teach] ORS  791/1172 (67.5%) sid=8141456
    [teach] ORS  792/1172 (67.6%) sid=8144234
    [teach] ORS  793/1172 (67.7%) sid=8192039
    [teach] ORS  794/1172 (67.7%) sid=8203265
    [teach] ORS  795/1172 (67.8%) sid=8207664
    [teach] ORS  796/1172 (67.9%) sid=8209840
    [teach] ORS  797/1172 (68.0%) sid=8218338
    [teach] ORS  798/1172 (68.1%) sid=8219769
    [teach] ORS  799/1172 (68.2%) sid=8247874
    [teach] ORS  800/1172 (68.3%) sid=8250419
    [teach] exporting parquet snapshot ...
    [teach] ORS  801/1172 (68.3%) sid=8253823
    [teach] ORS  802/1172 (68.4%) sid=8269301
    [teach] ORS  803/1172 (68.5%) sid=8276321
    [teach] ORS  804/1172 (68.6%) sid=8279809
    [teach] ORS  805/1172 (68.7%) sid=8280108
    [teach] ORS  806/1172 (68.8%) sid=8290277
    [teach] ORS  807/1172 (68.9%) sid=8292678
    [teach] ORS  808/1172 (68.9%) sid=8325089
    [teach] ORS  809/1172 (69.0%) sid=8356609
    [teach] ORS  810/1172 (69.1%) sid=8359613
    [teach] ORS  811/1172 (69.2%) sid=8360139
    [teach] ORS  812/1172 (69.3%) sid=8367305
    [teach] ORS  813/1172 (69.4%) sid=8367379
    [teach] ORS  814/1172 (69.5%) sid=8375951
    [teach] ORS  815/1172 (69.5%) sid=8376958
    [teach] ORS  816/1172 (69.6%) sid=8384400
    [teach] ORS  817/1172 (69.7%) sid=8389908
    [teach] ORS  818/1172 (69.8%) sid=8430580
    [teach] ORS  819/1172 (69.9%) sid=8439412
    [teach] ORS  820/1172 (70.0%) sid=8458199
    [teach] ORS  821/1172 (70.1%) sid=8458460
    [teach] ORS  822/1172 (70.1%) sid=8477590
    [teach] ORS  823/1172 (70.2%) sid=8484901
    [teach] ORS  824/1172 (70.3%) sid=8493177
    [teach] ORS  825/1172 (70.4%) sid=8496286
    [teach] exporting parquet snapshot ...
    [teach] ORS  826/1172 (70.5%) sid=8507916
    [teach] ORS  827/1172 (70.6%) sid=8534928
    [teach] ORS  828/1172 (70.6%) sid=8537433
    [teach] ORS  829/1172 (70.7%) sid=8553459
    [teach] ORS  830/1172 (70.8%) sid=8568294
    [teach] ORS  831/1172 (70.9%) sid=8584670
    [teach] ORS  832/1172 (71.0%) sid=8593917
    [teach] ORS  833/1172 (71.1%) sid=8630437
    [teach] ORS  834/1172 (71.2%) sid=8640247
    [teach] ORS  835/1172 (71.2%) sid=8647429
    [teach] ORS  836/1172 (71.3%) sid=8647914
    [teach] ORS  837/1172 (71.4%) sid=8659962
    [teach] ORS  838/1172 (71.5%) sid=8661444
    [teach] ORS  839/1172 (71.6%) sid=8661544
    [teach] ORS  840/1172 (71.7%) sid=8701440
    [teach] ORS  841/1172 (71.8%) sid=8732902
    [teach] ORS  842/1172 (71.8%) sid=8736448
    [teach] ORS  843/1172 (71.9%) sid=8738931
    [teach] ORS  844/1172 (72.0%) sid=8743262
    [teach] ORS  845/1172 (72.1%) sid=8745967
    [teach] ORS  846/1172 (72.2%) sid=8753555
    [teach] ORS  847/1172 (72.3%) sid=8764504
    [teach] ORS  848/1172 (72.4%) sid=8798239
    [teach] ORS  849/1172 (72.4%) sid=8803593
    [teach] ORS  850/1172 (72.5%) sid=8812531
    [teach] exporting parquet snapshot ...
    [teach] ORS  851/1172 (72.6%) sid=8845628
    [teach] ORS  852/1172 (72.7%) sid=8858057
    [teach] ORS  853/1172 (72.8%) sid=8886299
    [teach] ORS  854/1172 (72.9%) sid=8899670
    [teach] ORS  855/1172 (73.0%) sid=8904990
    [teach] ORS  856/1172 (73.0%) sid=8979512
    [teach] ORS  857/1172 (73.1%) sid=8986981
    [teach] ORS  858/1172 (73.2%) sid=9002133
    [teach] ORS  859/1172 (73.3%) sid=9022843
    [teach] ORS  860/1172 (73.4%) sid=9054685
    [teach] ORS  861/1172 (73.5%) sid=9054757
    [teach] ORS  862/1172 (73.5%) sid=9061412
    [teach] ORS  863/1172 (73.6%) sid=9074084
    [teach] ORS  864/1172 (73.7%) sid=9080913
    [teach] ORS  865/1172 (73.8%) sid=9081225
    [teach] ORS  866/1172 (73.9%) sid=9144525
    [teach] ORS  867/1172 (74.0%) sid=9166867
    [teach] ORS  868/1172 (74.1%) sid=9189528
    [teach] ORS  869/1172 (74.1%) sid=9198410
    [teach] ORS  870/1172 (74.2%) sid=9216134
    [teach] ORS  871/1172 (74.3%) sid=9245905
    [teach] ORS  872/1172 (74.4%) sid=9260929
    [teach] ORS  873/1172 (74.5%) sid=9276203
    [teach] ORS  874/1172 (74.6%) sid=9276930
    [teach] ORS  875/1172 (74.7%) sid=9278188
    [teach] exporting parquet snapshot ...
    [teach] ORS  876/1172 (74.7%) sid=9280905
    [teach] ORS  877/1172 (74.8%) sid=9298757
    [teach] ORS  878/1172 (74.9%) sid=9312099
    [teach] ORS  879/1172 (75.0%) sid=9332240
    [teach] ORS  880/1172 (75.1%) sid=9352936
    [teach] ORS  881/1172 (75.2%) sid=9385410
    [teach] ORS  882/1172 (75.3%) sid=9441730
    [teach] ORS  883/1172 (75.3%) sid=9453967
    [teach] ORS  884/1172 (75.4%) sid=9456943
    [teach] ORS  885/1172 (75.5%) sid=9460546
    [teach] ORS  886/1172 (75.6%) sid=9474787
    [teach] ORS  887/1172 (75.7%) sid=9489647
    [teach] ORS  888/1172 (75.8%) sid=9552137
    [teach] ORS  889/1172 (75.9%) sid=9558889
    [teach] ORS  890/1172 (75.9%) sid=9584973
    [teach] ORS  891/1172 (76.0%) sid=9631049
    [teach] ORS  892/1172 (76.1%) sid=9651230
    [teach] ORS  893/1172 (76.2%) sid=9687533
    [teach] ORS  894/1172 (76.3%) sid=9719898
    [teach] ORS  895/1172 (76.4%) sid=9760492
    [teach] ORS  896/1172 (76.5%) sid=9764861
    [teach] ORS  897/1172 (76.5%) sid=9780281
    [teach] ORS  898/1172 (76.6%) sid=9784944
    [teach] ORS  899/1172 (76.7%) sid=9790972
    [teach] ORS  900/1172 (76.8%) sid=9792022
    [teach] exporting parquet snapshot ...
    [teach] ORS  901/1172 (76.9%) sid=9812424
    [teach] ORS  902/1172 (77.0%) sid=9827139
    [teach] ORS  903/1172 (77.0%) sid=9834543
    [teach] ORS  904/1172 (77.1%) sid=9834959
    [teach] ORS  905/1172 (77.2%) sid=9838619
    [teach] ORS  906/1172 (77.3%) sid=9892756
    [teach] ORS  907/1172 (77.4%) sid=9923355
    [teach] ORS  908/1172 (77.5%) sid=9926590
    [teach] ORS  909/1172 (77.6%) sid=9926849
    [teach] ORS  910/1172 (77.6%) sid=9941191
    [teach] ORS  911/1172 (77.7%) sid=9942273
    [teach] ORS  912/1172 (77.8%) sid=9959922
    [teach] ORS  913/1172 (77.9%) sid=9997665
    [teach] ORS  914/1172 (78.0%) sid=10003312
    [teach] ORS  915/1172 (78.1%) sid=10119292
    [teach] ORS  916/1172 (78.2%) sid=10125147
    [teach] ORS  917/1172 (78.2%) sid=10130352
    [teach] ORS  918/1172 (78.3%) sid=10139844
    [teach] ORS  919/1172 (78.4%) sid=10204705
    [teach] ORS  920/1172 (78.5%) sid=10229995
    [teach] ORS  921/1172 (78.6%) sid=10232151
    [teach] ORS  922/1172 (78.7%) sid=10237658
    [teach] ORS  923/1172 (78.8%) sid=10239076
    [teach] ORS  924/1172 (78.8%) sid=10241481
    [ORS][warn] sid=10244968 no valid candidates after constraints (mode=dp_prefix, k_span=5..6, dp_q=33, beta=3.0). Trying fallback #1.
    [teach] ORS  925/1172 (78.9%) sid=10244968
    [teach] exporting parquet snapshot ...
    [teach] ORS  926/1172 (79.0%) sid=10248840
    [teach] ORS  927/1172 (79.1%) sid=10257721
    [teach] ORS  928/1172 (79.2%) sid=10261424
    [teach] ORS  929/1172 (79.3%) sid=10266047
    [teach] ORS  930/1172 (79.4%) sid=10285817
    [teach] ORS  931/1172 (79.4%) sid=10287456
    [teach] ORS  932/1172 (79.5%) sid=10298910
    [teach] ORS  933/1172 (79.6%) sid=10312952
    [teach] ORS  934/1172 (79.7%) sid=10313357
    [teach] ORS  935/1172 (79.8%) sid=10334040
    [teach] ORS  936/1172 (79.9%) sid=10350999
    [teach] ORS  937/1172 (79.9%) sid=10352883
    [teach] ORS  938/1172 (80.0%) sid=10353659
    [teach] ORS  939/1172 (80.1%) sid=10362279
    [teach] ORS  940/1172 (80.2%) sid=10367206
    [teach] ORS  941/1172 (80.3%) sid=10376385
    [teach] ORS  942/1172 (80.4%) sid=10381532
    [teach] ORS  943/1172 (80.5%) sid=10388728
    [teach] ORS  944/1172 (80.5%) sid=10393598
    [teach] ORS  945/1172 (80.6%) sid=10393637
    [teach] ORS  946/1172 (80.7%) sid=10399456
    [teach] ORS  947/1172 (80.8%) sid=10415762
    [teach] ORS  948/1172 (80.9%) sid=10419548
    [teach] ORS  949/1172 (81.0%) sid=10424948
    [teach] ORS  950/1172 (81.1%) sid=10491369
    [teach] exporting parquet snapshot ...
    [teach] ORS  951/1172 (81.1%) sid=10500957
    [teach] ORS  952/1172 (81.2%) sid=10509709
    [teach] ORS  953/1172 (81.3%) sid=10515875
    [teach] ORS  954/1172 (81.4%) sid=10525015
    [teach] ORS  955/1172 (81.5%) sid=10536320
    [teach] ORS  956/1172 (81.6%) sid=10544732
    [teach] ORS  957/1172 (81.7%) sid=10566178
    [teach] ORS  958/1172 (81.7%) sid=10566424
    [teach] ORS  959/1172 (81.8%) sid=10569669
    [teach] ORS  960/1172 (81.9%) sid=10570040
    [teach] ORS  961/1172 (82.0%) sid=10599100
    [teach] ORS  962/1172 (82.1%) sid=10629196
    [teach] ORS  963/1172 (82.2%) sid=10636801
    [teach] ORS  964/1172 (82.3%) sid=10638109
    [teach] ORS  965/1172 (82.3%) sid=10644767
    [teach] ORS  966/1172 (82.4%) sid=10649682
    [teach] ORS  967/1172 (82.5%) sid=10651719
    [teach] ORS  968/1172 (82.6%) sid=10671959
    [teach] ORS  969/1172 (82.7%) sid=10677511
    [teach] ORS  970/1172 (82.8%) sid=10680807
    [teach] ORS  971/1172 (82.8%) sid=10685489
    [teach] ORS  972/1172 (82.9%) sid=10687428
    [teach] ORS  973/1172 (83.0%) sid=10704592
    [teach] ORS  974/1172 (83.1%) sid=10706141
    [teach] ORS  975/1172 (83.2%) sid=10706991
    [teach] exporting parquet snapshot ...
    [teach] ORS  976/1172 (83.3%) sid=10707331
    [teach] ORS  977/1172 (83.4%) sid=10708831
    [teach] ORS  978/1172 (83.4%) sid=10710896
    [teach] ORS  979/1172 (83.5%) sid=10712327
    [teach] ORS  980/1172 (83.6%) sid=10745510
    [teach] ORS  981/1172 (83.7%) sid=10746543
    [teach] ORS  982/1172 (83.8%) sid=10752455
    [teach] ORS  983/1172 (83.9%) sid=10769551
    [teach] ORS  984/1172 (84.0%) sid=10784762
    [teach] ORS  985/1172 (84.0%) sid=10794496
    [teach] ORS  986/1172 (84.1%) sid=10818869
    [teach] ORS  987/1172 (84.2%) sid=10831918
    [teach] ORS  988/1172 (84.3%) sid=10836192
    [teach] ORS  989/1172 (84.4%) sid=10846276
    [teach] ORS  990/1172 (84.5%) sid=10850895
    [teach] ORS  991/1172 (84.6%) sid=10855309
    [teach] ORS  992/1172 (84.6%) sid=10923529
    [teach] ORS  993/1172 (84.7%) sid=10927300
    [teach] ORS  994/1172 (84.8%) sid=10927496
    [teach] ORS  995/1172 (84.9%) sid=10943656
    [teach] ORS  996/1172 (85.0%) sid=10944077
    [teach] ORS  997/1172 (85.1%) sid=10967308
    [teach] ORS  998/1172 (85.2%) sid=10975698
    [teach] ORS  999/1172 (85.2%) sid=10989111
    [teach] ORS 1000/1172 (85.3%) sid=11002629
    [teach] exporting parquet snapshot ...
    [teach] ORS 1001/1172 (85.4%) sid=11012229
    [teach] ORS 1002/1172 (85.5%) sid=11018700
    [teach] ORS 1003/1172 (85.6%) sid=11025567
    [teach] ORS 1004/1172 (85.7%) sid=11030010
    [teach] ORS 1005/1172 (85.8%) sid=11035355
    [teach] ORS 1006/1172 (85.8%) sid=11045844
    [teach] ORS 1007/1172 (85.9%) sid=11058367
    [teach] ORS 1008/1172 (86.0%) sid=11078849
    [teach] ORS 1009/1172 (86.1%) sid=11081105
    [teach] ORS 1010/1172 (86.2%) sid=11112614
    [teach] ORS 1011/1172 (86.3%) sid=11155024
    [teach] ORS 1012/1172 (86.3%) sid=11156542
    [teach] ORS 1013/1172 (86.4%) sid=11221009
    [teach] ORS 1014/1172 (86.5%) sid=11229586
    [teach] ORS 1015/1172 (86.6%) sid=11234740
    [teach] ORS 1016/1172 (86.7%) sid=11284135
    [teach] ORS 1017/1172 (86.8%) sid=11294478
    [teach] ORS 1018/1172 (86.9%) sid=11341687
    [teach] ORS 1019/1172 (86.9%) sid=11346859
    [teach] ORS 1020/1172 (87.0%) sid=11352856
    [teach] ORS 1021/1172 (87.1%) sid=11371670
    [teach] ORS 1022/1172 (87.2%) sid=11376100
    [teach] ORS 1023/1172 (87.3%) sid=11388952
    [teach] ORS 1024/1172 (87.4%) sid=11396070
    [teach] ORS 1025/1172 (87.5%) sid=11396701
    [teach] exporting parquet snapshot ...
    [teach] ORS 1026/1172 (87.5%) sid=11397514
    [teach] ORS 1027/1172 (87.6%) sid=11404889
    [teach] ORS 1028/1172 (87.7%) sid=11408024
    [teach] ORS 1029/1172 (87.8%) sid=11417451
    [teach] ORS 1030/1172 (87.9%) sid=11440124
    [teach] ORS 1031/1172 (88.0%) sid=11445515
    [teach] ORS 1032/1172 (88.1%) sid=11447384
    [teach] ORS 1033/1172 (88.1%) sid=11462386
    [teach] ORS 1034/1172 (88.2%) sid=11469756
    [teach] ORS 1035/1172 (88.3%) sid=11470568
    [teach] ORS 1036/1172 (88.4%) sid=11482298
    [teach] ORS 1037/1172 (88.5%) sid=11483697
    [teach] ORS 1038/1172 (88.6%) sid=11487147
    [teach] ORS 1039/1172 (88.7%) sid=11497324
    [teach] ORS 1040/1172 (88.7%) sid=11513099
    [teach] ORS 1041/1172 (88.8%) sid=11519799
    [teach] ORS 1042/1172 (88.9%) sid=11523452
    [teach] ORS 1043/1172 (89.0%) sid=11526118
    [teach] ORS 1044/1172 (89.1%) sid=11527264
    [teach] ORS 1045/1172 (89.2%) sid=11528426
    [teach] ORS 1046/1172 (89.2%) sid=11530012
    [teach] ORS 1047/1172 (89.3%) sid=11534915
    [teach] ORS 1048/1172 (89.4%) sid=11535753
    [teach] ORS 1049/1172 (89.5%) sid=11540931
    [teach] ORS 1050/1172 (89.6%) sid=11546798
    [teach] exporting parquet snapshot ...
    [teach] ORS 1051/1172 (89.7%) sid=11547678
    [teach] ORS 1052/1172 (89.8%) sid=11548856
    [teach] ORS 1053/1172 (89.8%) sid=11550990
    [teach] ORS 1054/1172 (89.9%) sid=11559051
    [teach] ORS 1055/1172 (90.0%) sid=11566365
    [teach] ORS 1056/1172 (90.1%) sid=11574745
    [teach] ORS 1057/1172 (90.2%) sid=11577696
    [teach] ORS 1058/1172 (90.3%) sid=11589474
    [teach] ORS 1059/1172 (90.4%) sid=11590286
    [teach] ORS 1060/1172 (90.4%) sid=11591128
    [teach] ORS 1061/1172 (90.5%) sid=11592638
    [teach] ORS 1062/1172 (90.6%) sid=11595059
    [teach] ORS 1063/1172 (90.7%) sid=11601924
    [teach] ORS 1064/1172 (90.8%) sid=11607358
    [teach] ORS 1065/1172 (90.9%) sid=11610725
    [teach] ORS 1066/1172 (91.0%) sid=11613979
    [teach] ORS 1067/1172 (91.0%) sid=11621880
    [teach] ORS 1068/1172 (91.1%) sid=11628423
    [teach] ORS 1069/1172 (91.2%) sid=11635547
    [teach] ORS 1070/1172 (91.3%) sid=11640731
    [teach] ORS 1071/1172 (91.4%) sid=11644138
    [teach] ORS 1072/1172 (91.5%) sid=11657374
    [teach] ORS 1073/1172 (91.6%) sid=11660574
    [teach] ORS 1074/1172 (91.6%) sid=11665555
    [teach] ORS 1075/1172 (91.7%) sid=11668248
    [teach] exporting parquet snapshot ...
    [teach] ORS 1076/1172 (91.8%) sid=11687255
    [teach] ORS 1077/1172 (91.9%) sid=11689627
    [teach] ORS 1078/1172 (92.0%) sid=11689761
    [teach] ORS 1079/1172 (92.1%) sid=11690687
    [ORS][warn] sid=11691589 no valid candidates after constraints (mode=dp_prefix, k_span=14..16, dp_q=30, beta=3.0). Trying fallback #1.
    [teach] ORS 1080/1172 (92.2%) sid=11691589
    [teach] ORS 1081/1172 (92.2%) sid=11700738
    [teach] ORS 1082/1172 (92.3%) sid=11703904
    [teach] ORS 1083/1172 (92.4%) sid=11705806
    [teach] ORS 1084/1172 (92.5%) sid=11713457
    [teach] ORS 1085/1172 (92.6%) sid=11725251
    [teach] ORS 1086/1172 (92.7%) sid=11728077
    [teach] ORS 1087/1172 (92.7%) sid=11730271
    [teach] ORS 1088/1172 (92.8%) sid=11732342
    [teach] ORS 1089/1172 (92.9%) sid=11741817
    [teach] ORS 1090/1172 (93.0%) sid=11742248
    [teach] ORS 1091/1172 (93.1%) sid=11745153
    [teach] ORS 1092/1172 (93.2%) sid=11758004
    [teach] ORS 1093/1172 (93.3%) sid=11766550
    [teach] ORS 1094/1172 (93.3%) sid=11770967
    [teach] ORS 1095/1172 (93.4%) sid=11781698
    [teach] ORS 1096/1172 (93.5%) sid=11789188
    [teach] ORS 1097/1172 (93.6%) sid=11796634
    [teach] ORS 1098/1172 (93.7%) sid=11807865
    [teach] ORS 1099/1172 (93.8%) sid=11820631
    [teach] ORS 1100/1172 (93.9%) sid=11843156
    [teach] exporting parquet snapshot ...
    [teach] ORS 1101/1172 (93.9%) sid=11844329
    [teach] ORS 1102/1172 (94.0%) sid=11870677
    [teach] ORS 1103/1172 (94.1%) sid=11885295
    [teach] ORS 1104/1172 (94.2%) sid=11887443
    [teach] ORS 1105/1172 (94.3%) sid=11895715
    [teach] ORS 1106/1172 (94.4%) sid=11896790
    [teach] ORS 1107/1172 (94.5%) sid=11907569
    [teach] ORS 1108/1172 (94.5%) sid=11909667
    [teach] ORS 1109/1172 (94.6%) sid=11934921
    [teach] ORS 1110/1172 (94.7%) sid=11937188
    [teach] ORS 1111/1172 (94.8%) sid=11942751
    [teach] ORS 1112/1172 (94.9%) sid=11951835
    [teach] ORS 1113/1172 (95.0%) sid=11953071
    [teach] ORS 1114/1172 (95.1%) sid=11956008
    [teach] ORS 1115/1172 (95.1%) sid=11975700
    [teach] ORS 1116/1172 (95.2%) sid=11997964
    [teach] ORS 1117/1172 (95.3%) sid=12034414
    [teach] ORS 1118/1172 (95.4%) sid=12046893
    [teach] ORS 1119/1172 (95.5%) sid=12050371
    [teach] ORS 1120/1172 (95.6%) sid=12051063
    [teach] ORS 1121/1172 (95.6%) sid=12058611
    [teach] ORS 1122/1172 (95.7%) sid=12059264
    [teach] ORS 1123/1172 (95.8%) sid=12063051
    [teach] ORS 1124/1172 (95.9%) sid=12079027
    [teach] ORS 1125/1172 (96.0%) sid=12097473
    [teach] exporting parquet snapshot ...
    [teach] ORS 1126/1172 (96.1%) sid=12101609
    [teach] ORS 1127/1172 (96.2%) sid=12109803
    [teach] ORS 1128/1172 (96.2%) sid=12117600
    [teach] ORS 1129/1172 (96.3%) sid=12134403
    [teach] ORS 1130/1172 (96.4%) sid=12145955
    [teach] ORS 1131/1172 (96.5%) sid=12174756
    [teach] ORS 1132/1172 (96.6%) sid=12178898
    [teach] ORS 1133/1172 (96.7%) sid=12187922
    [teach] ORS 1134/1172 (96.8%) sid=12193183
    [teach] ORS 1135/1172 (96.8%) sid=12195263
    [teach] ORS 1136/1172 (96.9%) sid=12195589
    [teach] ORS 1137/1172 (97.0%) sid=12211585
    [teach] ORS 1138/1172 (97.1%) sid=12211905
    [teach] ORS 1139/1172 (97.2%) sid=12215667
    [teach] ORS 1140/1172 (97.3%) sid=12278141
    [teach] ORS 1141/1172 (97.4%) sid=12279812
    [teach] ORS 1142/1172 (97.4%) sid=12289381
    [teach] ORS 1143/1172 (97.5%) sid=12300625
    [teach] ORS 1144/1172 (97.6%) sid=12302930
    [teach] ORS 1145/1172 (97.7%) sid=12303176
    [teach] ORS 1146/1172 (97.8%) sid=12303502
    [teach] ORS 1147/1172 (97.9%) sid=12321524
    [teach] ORS 1148/1172 (98.0%) sid=12330040
    [teach] ORS 1149/1172 (98.0%) sid=12331745
    [teach] ORS 1150/1172 (98.1%) sid=12332308
    [teach] exporting parquet snapshot ...
    [teach] ORS 1151/1172 (98.2%) sid=12380386
    [teach] ORS 1152/1172 (98.3%) sid=12401718
    [teach] ORS 1153/1172 (98.4%) sid=12423834
    [teach] ORS 1154/1172 (98.5%) sid=12427725
    [teach] ORS 1155/1172 (98.5%) sid=12427932
    [teach] ORS 1156/1172 (98.6%) sid=12438831
    [teach] ORS 1157/1172 (98.7%) sid=12441787
    [teach] ORS 1158/1172 (98.8%) sid=12465020
    [teach] ORS 1159/1172 (98.9%) sid=12473317
    [teach] ORS 1160/1172 (99.0%) sid=12479101
    [teach] ORS 1161/1172 (99.1%) sid=12505559
    [teach] ORS 1162/1172 (99.1%) sid=12527142
    [teach] ORS 1163/1172 (99.2%) sid=12543963
    [teach] ORS 1164/1172 (99.3%) sid=12556655
    [teach] ORS 1165/1172 (99.4%) sid=12566912
    [teach] ORS 1166/1172 (99.5%) sid=12584175
    [teach] ORS 1167/1172 (99.6%) sid=12590071
    [teach] ORS 1168/1172 (99.7%) sid=12608409
    [teach] ORS 1169/1172 (99.7%) sid=12612355
    [teach] ORS 1170/1172 (99.8%) sid=12637759
    [teach] ORS 1171/1172 (99.9%) sid=12645343
    [teach] ORS 1172/1172 (100.0%) sid=12645758
    [teach] exporting parquet snapshot ...
    [teach] done. processed 1172 sessions in 3 h 6 m.
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

    [teaching pool] rows=1172, classes:
    label_int
    0    586
    1    586
    
    [k] stats:
     count    1172.000000
    mean        3.537543
    std         1.751259
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
      <td>0.300746</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>8.295454</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.761166e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/137...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/137598.npy</td>
      <td>264</td>
      <td>[2.240215301513672, 2.1924808025360107, 2.1447460651397705, 2.0970146656036377, 2.0492897033691406, 2.00156450271606...</td>
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
      <td>4.0</td>
      <td>13.030439</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.434239</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.761166e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/167...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/167874.npy</td>
      <td>264</td>
      <td>[-2.2508716583251953, -1.8199647665023804, -1.3890577554702759, -0.9581508040428162, -0.5272438526153564, -0.0963368...</td>
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
      <td>4.0</td>
      <td>9.624545</td>
      <td>0.041333</td>
      <td>0.958667</td>
      <td>1.028345</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.761166e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/231...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/231570.npy</td>
      <td>264</td>
      <td>[-2.075390100479126, -1.734521746635437, -1.3936532735824585, -1.0527849197387695, -0.711916446685791, -0.3710480034...</td>
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
      <td>3.0</td>
      <td>1.925004</td>
      <td>0.035667</td>
      <td>0.964333</td>
      <td>6.671196</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.761167e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/240...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/240316.npy</td>
      <td>264</td>
      <td>[1.6217665672302246, 1.6354061365127563, 1.6490458250045776, 1.6626853942871094, 1.6763250827789307, 1.6899646520614...</td>
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
      <td>8.747483</td>
      <td>0.250333</td>
      <td>0.749667</td>
      <td>0.151283</td>
      <td>8.5962</td>
      <td>final_model.pth</td>
      <td>1.761167e+09</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/248...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/248477.npy</td>
      <td>264</td>
      <td>[2.564788341522217, 2.6007373332977295, 2.636686325073242, 2.672635078430176, 2.7085840702056885, 2.744532823562622,...</td>
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
    2.0    381
    3.0     99
    4.0     60
    1.0     24
    5.0     20
    6.0      2
    Name: count, dtype: int64
    Frequencies of k for abnormal sessions in the teaching pool:
    k
    3.0     176
    4.0     129
    5.0     116
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
      k counts: {2: 12, 3: 176, 4: 129, 5: 116, 6: 85, 7: 30, 8: 19, 9: 9, 10: 3, 11: 4, 12: 3}
      bins (5): labels=[2, 4, 6, 8, 10, 12]  counts=[317, 201, 49, 12, 7]


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
      class 0: selected=55 | F(S)=573.0051
      class 1: selected=54 | F(S)=501.4734
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
      class 0: 573.0051
      class 1: 501.4734
    
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
      <td>1.000</td>
      <td>8.363038</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-2.000138998031616, -1.964316725730896, -1.9284944534301758, -1.8926723003387451, -1.856868028640747, -1.8210674524...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/866...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/8661544.npy</td>
      <td>final_model.pth</td>
      <td>191.079895</td>
      <td>191.966199</td>
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
      <td>1.000</td>
      <td>8.221996</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-2.061953544616699, -2.0238306522369385, -1.9857079982757568, -1.9475852251052856, -1.9094624519348145, -1.87133967...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/107...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/10706141...</td>
      <td>final_model.pth</td>
      <td>1.218426</td>
      <td>2.090626</td>
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
      <td>1.000</td>
      <td>8.407482</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-1.7185766696929932, -1.6915091276168823, -1.6644415855407715, -1.6373741626739502, -1.6103066205978394, -1.5832390...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/121...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/12145955...</td>
      <td>final_model.pth</td>
      <td>6.536997</td>
      <td>7.427745</td>
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
      <td>1.000</td>
      <td>8.387293</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[-1.7185765504837036, -1.6915032863616943, -1.664430022239685, -1.6373568773269653, -1.610289454460144, -1.583238124...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/918...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/9189528.npy</td>
      <td>final_model.pth</td>
      <td>0.000012</td>
      <td>0.888741</td>
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
      <td>0.989</td>
      <td>5.802258</td>
      <td>8.5962</td>
      <td>...</td>
      <td>[1.0843602418899536, 1.0736451148986816, 1.0629298686981201, 1.0522147417068481, 1.0414994955062866, 1.0307843685150...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/427...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/4275867.npy</td>
      <td>final_model.pth</td>
      <td>157.401123</td>
      <td>158.030799</td>
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
      <td>176</td>
      <td>0.300341</td>
      <td>13.0</td>
      <td>0.240741</td>
      <td>0.074</td>
      <td>0.802</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>4</td>
      <td>129</td>
      <td>0.220137</td>
      <td>4.0</td>
      <td>0.074074</td>
      <td>0.031</td>
      <td>0.336</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>5</td>
      <td>116</td>
      <td>0.197952</td>
      <td>15.0</td>
      <td>0.277778</td>
      <td>0.129</td>
      <td>1.403</td>
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
      <td>317</td>
      <td>0.540956</td>
      <td>20.0</td>
      <td>0.370370</td>
      <td>0.063</td>
      <td>0.685</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>[4, 6]</td>
      <td>201</td>
      <td>0.343003</td>
      <td>20.0</td>
      <td>0.370370</td>
      <td>0.1</td>
      <td>1.08</td>
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

- **Group A** (explanations with curriculum): order by `k_power↑` then `margin_power↓`, overlay raw power with simplification and show simplified SOC. 
- **Group B** (explanations, no curriculum): random order, overlay raw power with simplification and show simplified SOC. 
- **Group C** (control, no explanations): random order, raw power and SOC only (no simplifications).

Constructing the teaching set is computationally expensive and the optimal teaching set for explaining the time series classifier (TSC) is large. 
We therefore cap the teaching set at most 100 per class by default to limit the computational effort required to build it. 


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
      <td>8.407482</td>
      <td>8.5962</td>
      <td>...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/12145955...</td>
      <td>final_model.pth</td>
      <td>6.536997</td>
      <td>7.427745</td>
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



```python
# build and attach iterators to the TeachingSet
iters = s.build_group_iterators(max_per_class=100, seed=RANDOM_SEED)

# demo: serve 6 examples per group and log metadata
logs = {"A": [], "B": [], "C": []}
# for g in ("A", "B", "C"):
for _ in range(20):
    try:
        meta = s.serve_sessions(group="C")  # plots immediately
        logs["A"].append(meta)
    except StopIteration:
        print(f"[{"A"}] no more examples.")
        break

pd.DataFrame(logs["A"]).head()

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
      <th>session_id</th>
      <th>group</th>
      <th>k</th>
      <th>label</th>
      <th>sts_full_path</th>
      <th>sts_soc_path</th>
      <th>piv_path</th>
      <th>piv_soc_path</th>
      <th>raw_power_path</th>
      <th>raw_soc_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11975700</td>
      <td>C</td>
      <td>None</td>
      <td>normal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/119...</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/11975700...</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/11...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/1197...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>528568</td>
      <td>C</td>
      <td>None</td>
      <td>abnormal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/528...</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/528568.npy</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/52...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/5285...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9189528</td>
      <td>C</td>
      <td>None</td>
      <td>normal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/918...</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/9189528.npy</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/91...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/9189...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5778107</td>
      <td>C</td>
      <td>None</td>
      <td>abnormal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/577...</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/5778107.npy</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/57...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/5778...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2621294</td>
      <td>C</td>
      <td>None</td>
      <td>normal</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/sts_full/262...</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/piv/2621294.npy</td>
      <td>None</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_power/26...</td>
      <td>/home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/Data/teaching_pool/raw_soc/2621...</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](06__Teach_files/06__Teach_24_1.png)
    



    
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
    



```python

```
