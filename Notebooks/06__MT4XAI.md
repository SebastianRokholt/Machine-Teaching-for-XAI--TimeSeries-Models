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
AD_THRESHOLD = cfg.inference.ad_rmse_threshold
```

    CONFIG FILE LOADED: 
    {'project': {'random_seed': 42, 'root_dir': None}, 'paths': {'dataset': 'Data/etron55-charging-sessions.parquet', 'teaching_pool': 'Data/teaching_pool', 'models': 'Models', 'final_model': 'Models/final/final_model.pth', 'figures': 'Figures', 'logs': 'Logs'}, 'inference': {'horizon': 5, 'final_model_name': 'final_model.pth', 'horizon_decay_lambda': 0.4, 'power_weight': 0.6522982410461, 't_min_eval': 1, 'ad_rmse_threshold': 8.5962, 'ad_pct_threshold': 0.95, 'metric': 'macro_rmse'}, 'ors': {'soc_stage1_mode': 'rdp', 'soc_rdp_epsilon': 0.75, 'soc_rdp_candidates': 5, 'soc_rdp_eps_min': 1e-06, 'soc_rdp_eps_max': 100.0, 'stage2_err_metric': 'l2', 'epsilon_mode': 'fraction'}, 'teaching': {'teaching_pool_dir': '../Data/teaching_pool', 'teaching_set_size': 60}}
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

    /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/src/mt4xai/model.py:43: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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
    ad_threshold=AD_THRESHOLD,  # 8.5962, i.e. the 95th percentile
    random_seed=RANDOM_SEED,
    length_range=(11, 60),  # only samples sequences in range (min_T, max_T)
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

    [teach] computing base labels on test set ...
    [teach] base labels: abnormal=12183, normal=0



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[6], line 42
         37     construct_pool = True
         39 if construct_pool:
         40     # builds the pool end-to-end, then loads it
         41     # builds the pool of simplifications and raw sessions
    ---> 42     pool = TeachingPool.construct_from_cfg(
         43         model=model,
         44         config=tpconfig,
         45         test_loader=test_loader,
         46         power_scaler=power_scaler,
         47         soc_scaler=soc_scaler,
         48         idx_power_inp=idx_power_inp,
         49         idx_soc_inp=idx_soc_inp,
         50     )


    File /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/src/mt4xai/teach.py:85, in TeachingPool.construct_from_cfg(cls, model, config, test_loader, power_scaler, soc_scaler, idx_power_inp, idx_soc_inp)
         66 """Constructs the teaching pool on disk and returns a loaded `TeachingPool`.
         67 
         68 This wraps `TeachingPool.build(...)`: it runs the pipeline that
       (...)
         82     TeachingPool: a ready-to-use pool object with `pool_df` loaded and standard `paths` initialised.
         83 """
         84 # build artefacts to disk
    ---> 85 cls.build(
         86     model=model,
         87     config=config,
         88     test_loader=test_loader,
         89     power_scaler=power_scaler,
         90     soc_scaler=soc_scaler,
         91     idx_power_inp=idx_power_inp,
         92     idx_soc_inp=idx_soc_inp,
         93 )
         95 # load snapshot
         96 root = Path(config.output_dir)


    File /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/src/mt4xai/teach.py:195, in TeachingPool.build(cls, model, config, test_loader, power_scaler, soc_scaler, idx_power_inp, idx_soc_inp)
        181 abn_ids, norm_ids, err_by_id = cls.compute_base_labels(
        182     test_loader,
        183     model,
       (...)
        192     threshold=config.ad_threshold,
        193 )
        194 print(f"[teach] base labels: abnormal={len(abn_ids)}, normal={len(norm_ids)}")
    --> 195 plan = cls.sample_sessions(abn_ids, norm_ids, random_seed=config.random_seed)
        196 (dirs["root"] / "sampled_normals.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        197 print(f"[teach] wrote sampling plan → {sp_json}")


    File /home/srokholt/Masters_Project_Linux_Env/Machine-Teaching-for-XAI--TimeSeries-Models/src/mt4xai/teach.py:748, in TeachingPool.sample_sessions(abnormal_ids, normal_ids, random_seed)
        746 n = len(abnormal_ids)
        747 if len(normal_ids) < n:
    --> 748     raise ValueError(f"not enough normal sessions to sample {n} (have {len(normal_ids)} normal, need {n}).")
        749 sel_normals = sorted(rng.choice(normal_ids, size=n, replace=False).tolist())
        750 return {"abnormal": sorted(list(abnormal_ids)), "normal": sel_normals}


    ValueError: not enough normal sessions to sample 12183 (have 0 normal, need 12183).


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
    lambda_margin=0.10,
    lambda_robust=0.05,
    lazy_prune=True,
    random_seed=RANDOM_SEED,
    min_per_k=2,  # Enforce selection >= 2 of each k value so whole dist of k is represented
    output_dir=TEACHING_DIR,
)

s.save(output_dir=TEACHING_DIR)
```

#### 3.3 - Teaching set analytics


```python
s.describe()
display(s.teaching_set_df.head())
```

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


```python

```
