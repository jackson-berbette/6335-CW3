import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


# read relevant input files
outbound_df = pd.read_csv("Data/outbound.csv")
clients_df = pd.read_csv("Data/clients.csv")
client_sites_df = pd.read_csv("Data/client_sites.csv")
skus_df = pd.read_csv("Data/skus.csv")

WEEKS = 52 

# -------- Part a --------
# pallets shipped per client over the year
client_pallets = (
    outbound_df.groupby("client_id")
    .size()
    .reset_index(name="pallets_yr")
)

clients_table_df = (
    clients_df.rename(columns={"sites": "stores"})
    .merge(client_pallets, on="client_id", how="left")
)

clients_table_df["client"] = clients_table_df["client_name"].str.replace(" Restaurants", "")
clients_table_df["pallets_wk"] = clients_table_df["pallets_yr"] / WEEKS
clients_table_df["share"] = clients_table_df["pallets_wk"] / clients_table_df["pallets_wk"].sum()
clients_table_df["pallets_per_store_wk"] = clients_table_df["pallets_wk"] / clients_table_df["stores"]


# where the restaurants are: sattes holding at least 5% of the client's stores, with store counts
def main_states(states): 
    counts = states.value_counts()
    return ", ".join(
        f"{state} {n}" for state, n in counts.items() if n >= 0.05 * len(states)
    )

client_states = (
    client_sites_df.groupby("client_id")["state"]
    .agg(main_states)
    .reset_index(name="states")
)

clients_table_df = (
    clients_table_df.merge(client_states, on="client_id", how="left")
    .sort_values("pallets_wk", ascending=False)
    .reset_index(drop=True)
)

clients_table_df = clients_table_df[
    ["client_id", "client", "stores", "skus", "pallets_wk", "share", "pallets_per_store_wk", "states"]
]

# total row
total_row = pd.DataFrame([{
    "client_id": "Total",
    "client": "Total",
    "stores": clients_table_df["stores"].sum(),
    "skus": clients_table_df["skus"].sum(),
    "pallets_wk": clients_table_df["pallets_wk"].sum(),
    "share": 1.0,
    "pallets_per_store_wk": clients_table_df["pallets_wk"].sum() / clients_table_df["stores"].sum(),
    "states": f"{client_sites_df['state'].nunique()} states"
}])

clients_table_df = pd.concat([clients_table_df, total_row], ignore_index=True)

# validate
assert len(clients_df) == 9
assert np.isclose(clients_table_df["pallets_wk"].iloc[:-1].sum() * WEEKS, len(outbound_df))

display(clients_table_df.round(3))

top4_share = clients_table_df["share"].iloc[:4].sum()
bottom3_share = clients_table_df["share"].iloc[6:9].sum()
bottom3_skus = clients_table_df["skus"].iloc[6:9].sum()

print(f"Top 4 clients ship {top4_share:.1%} of pallets")
print(f"Bottom 3 clients ship {bottom3_share:.1%} with {bottom3_skus} SKUs")

clients_table_df.to_csv("Output/Task1a_Clients.csv", index=False)

# -------- Part b --------
# pallets shipped per SKU over the year
sku_pallets = (
    outbound_df.groupby("sku_id")
    .size()
    .reset_index(name="pallets_yr")
)

sku_velocity_df = (
    skus_df[["sku_id", "client_id", "pallet_height_in", "pallet_weight_lb"]]
    .merge(sku_pallets, on="sku_id", how="left")
    .fillna({"pallets_yr": 0})
    .sort_values(["pallets_yr", "sku_id"], ascending=[False, True])
    .reset_index(drop=True)
)

sku_velocity_df["pallets_yr"] = sku_velocity_df["pallets_yr"].astype(int)
sku_velocity_df["pallets_wk"] = sku_velocity_df["pallets_yr"] / WEEKS
sku_velocity_df["rank"] = np.arange(1, len(sku_velocity_df) + 1)
sku_velocity_df["cum_share"] = sku_velocity_df["pallets_yr"].cumsum() / sku_velocity_df["pallets_yr"].sum()

# velocity classes, cut-offs in pallets shipped per year
# A: >= 260 (>= 5 / week) ships almost every shipping day
# B: 52 - 259 (1 - 5 / week) ships most weeks
# C: 13 - 51 (1 / month - 1 / week)
# D: <= 12 (<= 1 / month) ships a few times a year, or never
def velocity_class(pallets_yr):
    if pallets_yr >= 260:
        return "A"
    if pallets_yr >= 52:
        return "B"
    if pallets_yr >= 13:
        return "C"
    return "D"

sku_velocity_df["velocity_class"] = sku_velocity_df["pallets_yr"].apply(velocity_class)

# concentration of demand
def skus_for_share(share):
    return int((sku_velocity_df["cum_share"] < share).sum() + 1)

print(f"SKUs shipping 50% of pallets: {skus_for_share(0.50)}")
print(f"SKUs shipping 80% of pallets: {skus_for_share(0.80)}")
print(f"SKUs that never shipped: {(sku_velocity_df['pallets_yr'] == 0).sum()}")

# class summary: size, flow, and pallet profile weighted by pallets shipped
def class_profile(df):
    w = df["pallets_yr"]
    return pd.Series({
        "skus": len(df),
        "pallets_yr": w.sum(),
        "pallets_wk": w.sum() / WEEKS,
        "mean_height_in": np.average(df["pallet_height_in"], weights=w) if w.sum() else np.nan,
        "share_over_72in": np.average(df["pallet_height_in"] > 72, weights=w) if w.sum() else np.nan,
        "mean_weight_lb": np.average(df["pallet_weight_lb"], weights=w) if w.sum() else np.nan,
        "share_over_2000lb": np.average(df["pallet_weight_lb"] > 2000, weights=w) if w.sum() else np.nan,
    })

class_summary_df = (
    sku_velocity_df.groupby("velocity_class")
    .apply(class_profile, include_groups=False)
    .reset_index()
)

class_summary_df["share_skus"] = class_summary_df["skus"] / class_summary_df["skus"].sum()
class_summary_df["share_flow"] = class_summary_df["pallets_yr"] / class_summary_df["pallets_yr"].sum()

class_summary_df = class_summary_df[
    ["velocity_class", "skus", "share_skus", "pallets_wk", "share_flow",
     "mean_height_in", "share_over_72in", "mean_weight_lb", "share_over_2000lb"]
]

# shipping rhythm per class: how often each SKU ships (evidence for the cut-offs)
ship_dates = pd.to_datetime(outbound_df["ship_date"])
shipping_days = ship_dates.nunique()  # 312 shipping days (Sun-Fri)

sku_rhythm = (
    outbound_df.assign(
        ship_date=ship_dates,
        week=(ship_dates - ship_dates.min()).dt.days // 7
    )
    .groupby("sku_id")
    .agg(days_shipped=("ship_date", "nunique"), weeks_shipped=("week", "nunique"))
    .reset_index()
)

sku_velocity_df = (
    sku_velocity_df.merge(sku_rhythm, on="sku_id", how="left")
    .fillna({"days_shipped": 0, "weeks_shipped": 0})
)
sku_velocity_df["share_of_days_shipped"] = sku_velocity_df["days_shipped"] / shipping_days

class_rhythm_df = (
    sku_velocity_df.groupby("velocity_class")
    .agg(median_share_of_days_shipped=("share_of_days_shipped", "median"),
         median_weeks_shipped=("weeks_shipped", "median"))
    .reset_index()
)

display(class_rhythm_df.round(2))

# pallet profile of all shipped pallets (pallet-weighted medians)
shipped_pallets = outbound_df[["sku_id"]].merge(
    skus_df[["sku_id", "pallet_height_in", "pallet_weight_lb"]], on="sku_id", how="left"
)
print(f"Shipped pallets: height {shipped_pallets['pallet_height_in'].min()}-"
      f"{shipped_pallets['pallet_height_in'].max()} in (median {shipped_pallets['pallet_height_in'].median():.0f}), "
      f"weight {shipped_pallets['pallet_weight_lb'].min()}-{shipped_pallets['pallet_weight_lb'].max()} lb "
      f"(median {shipped_pallets['pallet_weight_lb'].median():.0f})")

# validate
assert len(sku_velocity_df) == len(skus_df)
assert sku_velocity_df["pallets_yr"].sum() == len(outbound_df)
assert class_summary_df["skus"].sum() == len(skus_df)

display(class_summary_df.round(3))

# cumulative curve with the class cut-offs
cutoffs = (
    sku_velocity_df.groupby("velocity_class")["rank"]
    .max()
)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(sku_velocity_df["rank"], sku_velocity_df["cum_share"] * 100, color="#2a78d6", lw=2)

for cls in ["A", "B", "C"]:
    k = cutoffs[cls]
    share = sku_velocity_df.loc[k - 1, "cum_share"]
    ax.axvline(k, color="grey", lw=0.8, ls="--")
    ax.text(k + 10, 8, f"end of {cls}\n{k} SKUs\n{share:.0%} of pallets", fontsize=8)


ax.set_xlabel("SKUs ranked by pallets shipped")
ax.set_ylabel("Cumulative % of pallets shipped")
ax.set_ylim(0, 102)
ax.set_xlim(0, len(sku_velocity_df) + 10)
ax.set_title(f"{cutoffs['A']} SKUs ship {sku_velocity_df.loc[cutoffs['A'] - 1, 'cum_share']:.0%} of pallets",
             loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Output/Task1b_Cumulative_Curve.png", dpi=150)
plt.show()

# save classes for Part f and Tasks 2-3
sku_velocity_df.to_csv("Output/Task1b_SKU_Velocity.csv", index=False)
class_summary_df.to_csv("Output/Task1b_Class_Summary.csv", index=False)