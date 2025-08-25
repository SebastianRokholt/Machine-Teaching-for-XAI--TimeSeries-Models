# %% [markdown]
# # Notebook \#2:</br> Exploratory Data Analysis
# #### by Sebastian Einar Salas Røkholt
# 
# ---
# 
# **Notebook Index**  

# %% [markdown]
# - [**1 - Introduction and Notebook Setup**](#1---introduction-and-notebook-setup)  
#   - [*1.1 Setup*](#11-setup)  
#   - [*1.2  Load cleaned dataset*](#12-load-cleaned-dataset)  
#   - [*1.3  Explanation of variables in the dataset*](#13-explanation-of-variables-in-the-dataset) 
# - [**2 - Summary Statistics**](#2---summary-statistics)  
# - [**3 - Mapping Charging Locations**](#3---mapping-charging-locations)  

# %% [markdown]
# ## 1 - Introduction and Notebook Setup

# %% [markdown]
# ### 1.1 Setup

# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


# Notebook settings
%matplotlib inline  
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:.2f}'.format  # By default, display all floats with two decimals

# %% [markdown]
# ### 1.2 Load cleaned dataset

# %%
df = pd.read_parquet("../Data/etron55-charging-sessions.parquet")
df.head()

# %% [markdown]
# ### 1.3 Explanation of variables in the dataset
# The dataset contains 1590144 measurements divided into 62,422 distinct charging sessions </br>for the Audi E-tron 55 EV. Each charging session was recorded at one of <a href="https://www.eviny.no/">Eviny</a>'s 286 charging stations. </br>
# 
# ---
#  - **`charging_id`, categorical, static (per session):**  The identifier for the entire charging session. A charging session is a single car, charging once at a single charging station. </br>
#  ---
#  - **`minutes_elapsed`, numerical, monotonic, time-dependent:** How many minutes have elapsed since the charging session began. This feature is calculated directly from the `timestamp` feature. </br>
#  - **`progress`** — *float ∈ [0,1], time-dependent*: Log-scaled timeline of `minutes_elapsed` (compressed long tails) capped at 120 min.
#  - **`timestamp`, DateTime, piecewise continuous:** The date and time of each measurement (YYYY-mm-dd HH:MM:SS). The time-dependent variables are measured at one minute intervals. </br>
#  - **`timestamp_H`** — *string, time-derived*: Hour bucket (`YYYY-MM-DDTHH`) for hourly grouping.
#  - **`timestamp_d`** — *string, time-derived*: Calendar day (`YYYY-MM-DD`) for daily grouping.
# ---
#  - **`power`, numerical, piecewise continuous, time-dependent:** The current power output in kW from the charging station to the car. </br>
#  - **`rel_power`** — *unitless ∈ [0,1], time-dependent*: `power` normalized by `nominal_power`, clipped at 120% then rescaled to [0,1].
#  - **`d_power`** — *kW/min, time-dependent*: First difference of `power` within a session.
#  - **`d_power_ema3`** — *kW/min, time-dependent*: Exponential moving average (span=3) of `d_power` to reduce high-frequency noise.
# ---
#  - **`soc`, numerical, piecewise continuous, time-dependent**: The State of Charge (SOC) of the car\'s battery as a percentage </br>
#  - **`d_soc`** — *pp/min, time-dependent*: First difference of `soc` within a session (percentage points per minute).
#  - **`d_soc_ema3`** — *pp/min, time-dependent*: EMA (span=3) of `d_soc`.
# ---
#  - **`energy`, numerical, piecewise continuous, time-dependent:** Cumulative delivered energy over the session, in kWh. 
# ---
#  - **`nominal_power`** — *kW, ordinal, static*: Nameplate capacity of the charging stall.
#  - **`charger_category`** — *categorical, static*: Provider’s category label (e.g., *Ultra*, *Rapid*).
#  - **`charger_cat_low`**, **`charger_cat_mid`**, **`charger_cat_high`** — *binary dummies, static*: One-hot encoding derived from `nominal_power` bins (`low` ≤ 75 kW, `mid` (75,200], `high` > 200 kW).
# ---
#  - **`temp`, numerical, discrete, static (per session):** The approximate ambient temperature in Celcius in the area (corresponding to the nearest weather station). Even though the actual temperature may have fluctuated slightly over the course of the charging session, we took the temperature rounded to the nearest integer at the beginning of each session, and held it constant throughout the session. </br>
#  - **`nearest_weather_station`** — *categorical, static*: Identifier of the station used for `temp` (e.g., MET Norway code).
#  - **`lat`, numerical, continuous, static (per session):** The latitude of the charging station. </br>
#  - **`lon`, numerical, continuous, static (per session):** The longitude of the charging station.</br>
# ---

# %% [markdown]
# ## 2 - Summary Statistics

# %%
# Calculate some basic descriptive metrics
print("Shape: ", df.shape)
print("Number of charging sessions: ", len(pd.unique(df["charging_id"])))
print("Time interval: ", df["timestamp"].min(), " to ", df["timestamp"].max())
unique_locations = df[["lat", "lon"]].drop_duplicates()
print(f"There are {len(unique_locations)} different charging locations in the dataset.")
print("\nPandas .info(): ")
display(df.info())
print("\nPandas .describe(): ")
display(df.describe())

# %%
print(df["charger_category"].unique())
print(df["nominal_power"].unique())
print(len(df["nominal_power"].unique()))

# %% [markdown]
# ## 3 - Mapping Charging Locations

# %%
# Plot charging sessions on the world map
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"  # WGS84 Latitude/Longitude
)
# load world map downloaded from Natural Earth: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
world = gpd.read_file("../Data/map_data_for_plotting/ne_10m_admin_0_countries.shp")  
fig, ax = plt.subplots()
world.plot(ax=ax, color='lightgrey', edgecolor='black')
gdf.plot(ax=ax, marker='x', color='red', markersize=30)
plt.savefig("../Images/EDA Plots/Charging_sessions_on_world_map.png", dpi=300, bbox_inches='tight')  # Save as PNG file
plt.show()

# %%
# Re-plot cropped map to see more detailed view
bbox = box(-10, 54, 30, 72)
clipped_world = gpd.clip(world, bbox)
fig, ax = plt.subplots(figsize=(15, 10))
clipped_world.plot(ax=ax, color='lightgrey', edgecolor='black')
gdf.plot(ax=ax, marker='x', color='darkgreen', markersize=10)
plt.savefig("../Images/EDA Plots/Charging_sessions_on_scandinavia_map.png")  # Save as PNG file
plt.show()


# %% [markdown]
# ## 4 - Correlation Metrics

# %% [markdown]
# ### 4.1 Correlation of persistance model per horizon

# %%
MAX_H = 15  # Maximum prediction horizon
df = df.sort_values(["charging_id","minutes_elapsed"]).copy() 

def persistence_per_horizon(df: pd.DataFrame, max_H: int=1):
    H = range(1, max_H)  # Prediction horizons from 1 to H minutes
    rows = []
    for h in H:
        # Future targets
        yP = df.groupby("charging_id")["power"].shift(-h)
        yS = df.groupby("charging_id")["soc"].shift(-h)
        # Persistence baseline: ŷ_(t+h) = y_(t)
        rmse_pwr = np.sqrt(((yP - df["power"])**2).dropna().mean())
        rmse_soc = np.sqrt(((yS - df["soc"])**2).dropna().mean())
        rows.append({"h":h, "RMSE_persist_power":rmse_pwr, "RMSE_persist_soc":rmse_soc})
    result = pd.DataFrame(rows)
    return result.round(4)


# Calculate persistence baseline RMSE and ACF
persistance_baseline = persistence_per_horizon(df, MAX_H)
print("\n\nPersistence baseline RMSE:\n", persistance_baseline)


# %% [markdown]
# ### 4.2 Autocorrelation per horizon

# %%
def autocorr_per_horizon(df: pd.DataFrame, max_H: int=1):
    H = range(1, max_H)  # Autocorr from 1 to H minutes
    rows = []
    for h in H:
        # Autocorr at lag h (average across sessions)
        autocorr_pwr = df.groupby("charging_id")["power"].apply(lambda s: s.autocorr(lag=h)).mean()
        autocorr_soc = df.groupby("charging_id")["soc"].apply(lambda s: s.autocorr(lag=h)).mean()
        rows.append({"h":h, "ACF_power(h)":autocorr_pwr, "ACF_soc(h)":autocorr_soc})
    result = pd.DataFrame(rows)
    return result.round(4)

with np.errstate(divide="ignore", invalid="ignore"):  # Ignore warnings for division by zero due to constant series
    autocorr = autocorr_per_horizon(df, MAX_H)
    print("\n\nAutocorrelation per horizon:\n", autocorr)

# %%
from sklearn.feature_selection import mutual_info_regression

features = ["minutes_elapsed","soc","progress","temp","nominal_power","rel_power",
            "d_power","d_soc","d_power_ema3","d_soc_ema3",
            "charger_cat_low","charger_cat_mid","charger_cat_high"]

def horizon_table(target_col):
    out = []
    for h in range(1,6):
        y = df.groupby("charging_id")[target_col].shift(-h)
        X = df[features].copy()
        mask = y.notna()
        y_ = y[mask].values
        X_ = X[mask].fillna(0).values
        # Pearson/Spearman (monotone)
        corr = pd.Series({f: df.loc[mask, f].corr(y) for f in features}, name=f"corr@h{h}")
        # Mutual Information (nonlinear)
        mi = pd.Series(mutual_info_regression(X_, y_), index=features, name=f"MI@h{h}")
        out.append(pd.concat([corr, mi], axis=1))
    return pd.concat(out, axis=1)

rel_power_tbl = horizon_table("power")
rel_soc_tbl = horizon_table("soc")

# Top drivers per horizon (example for power)
for h in range(1,6):
    display(rel_power_tbl[[f"MI@h{h}"]].sort_values(f"MI@h{h}", ascending=False).head(8))


# %%

def taper_soc_of_session(s, drop_ratio=0.95):
    p = s["power"].to_numpy()
    soc = s["soc"].to_numpy()
    if len(p) < 5: return np.nan
    runmax = np.maximum.accumulate(p)
    mask = p < drop_ratio * runmax  # first time power falls 5% below running max
    return soc[np.argmax(mask)] if mask.any() else np.nan

taper = (df.sort_values(["charging_id","minutes_elapsed"])
           .groupby("charging_id", group_keys=False)
           .apply(taper_soc_of_session)
           .rename("taper_soc")
           .to_frame())

print(taper.describe())

# Stratify by temperature quartile to see shifts in taper onset
temp_q = pd.qcut(df.groupby("charging_id")["temp"].median(), 4, labels=False)
taper["temp_q"] = temp_q
print(taper.groupby("temp_q")["taper_soc"].median())


# %%
# Conditional std of future power given SOC bin at time t, across horizons
std_list = []
for h in range(1,6):
    y = df.groupby("charging_id")["power"].shift(-h)
    tmp = pd.DataFrame({"soc": df["soc"], "y": y}).dropna()
    tmp["soc_bin"] = pd.cut(tmp["soc"], bins=20)
    std = tmp.groupby("soc_bin")["y"].std()
    std.name = f"h{h}"
    std_list.append(std)

cond_std = pd.concat(std_list, axis=1)
print(cond_std.head())  # heatmap in your plotting stack

# Optional: quantile width (P90 - P10) for robustness
qw_list = []
for h in range(1,6):
    y = df.groupby("charging_id")["power"].shift(-h)
    tmp = pd.DataFrame({"soc": df["soc"], "y": y}).dropna()
    tmp["soc_bin"] = pd.cut(tmp["soc"], bins=20)
    q = tmp.groupby("soc_bin")["y"].quantile([.1,.9]).unstack()
    q["width"] = q[0.9] - q[0.1]
    qw_list.append(q["width"].rename(f"h{h}"))
cond_qwidth = pd.concat(qw_list, axis=1)
print(cond_qwidth.head())


# %%
# Label availability per horizon
cov = []
for h in range(1,6):
    yP = df.groupby("charging_id")["power"].shift(-h)
    yS = df.groupby("charging_id")["soc"].shift(-h)
    cov.append({"h":h,
                "N_power_labels": int(yP.notna().sum()),
                "N_soc_labels":   int(yS.notna().sum())})
coverage = pd.DataFrame(cov)
print(coverage)

# Session length distribution (helps choose min length, padding strategy)
lengths = (df.groupby("charging_id")["minutes_elapsed"].max() + 1).rename("length")
print(lengths.describe())


# %%



