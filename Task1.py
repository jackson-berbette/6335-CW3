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

 # -------- Part c --------
demand_df = outbound_df[["pallet_id", "client_id", "ship_date", "picked_at"]].copy()
demand_df["ship_date"] = pd.to_datetime(demand_df["ship_date"])
demand_df["picked_at"] = pd.to_datetime(demand_df["picked_at"])
demand_df["week"] = (demand_df["ship_date"] - demand_df["ship_date"].min()).dt.days // 7 + 1
demand_df["day_of_week"] = demand_df["ship_date"].dt.day_name()
demand_df["hour"] = demand_df["picked_at"].dt.hour

shipping_days = demand_df["ship_date"].nunique()

# pallets per week across the year
weekly_df = (
    demand_df.groupby("week")
    .agg(week_start=("ship_date", "min"), pallets=("pallet_id", "size"))
    .reset_index()
)

weekly_mean = weekly_df["pallets"].mean()
weekly_cv = weekly_df["pallets"].std() / weekly_mean
trend_per_week = np.polyfit(weekly_df["week"], weekly_df["pallets"], 1)[0] # slope of a linear fit

print(f"Pallets per week: mean {weekly_mean:.0f}, min {weekly_df['pallets'].min()}, "
      f"max {weekly_df['pallets'].max()}, CV {weekly_cv:.1%}")
print(f"Trend: {trend_per_week:+.2f} pallets per week, per week ({trend_per_week * 52:+.0f} over the year)")


# seasonality: average pallets per shipping day, by month
monthly_df = (
    demand_df.groupby(demand_df["ship_date"].dt.to_period("M"))
    .agg(pallets=("pallet_id", "size"), days=("ship_date", "nunique"))
    .reset_index()
)
monthly_df["pallets_per_day"] = monthly_df["pallets"] / monthly_df["days"]
full_months = monthly_df[monthly_df["days"] >= 20] # july 2026 has only 4 shipping days
print(f"Monthly average per shipping day (full months): {full_months['pallets_per_day'].min():.0f}"
      f"-{full_months['pallets_per_day'].max():.0f} pallets")

# week-to-week variability by client
client_weekly = (
    demand_df.groupby(["client_id", "week"])
    .size()
    .unstack("client_id", fill_value=0)
)
client_cv_df = (
    (client_weekly.std() / client_weekly.mean())
    .reset_index(name="weekly_cv")
    .merge(clients_df[["client_id", "client_name"]], on="client_id")
    .sort_values("weekly_cv")
)
display(client_cv_df.round(3))

# pallets per day of week
dow_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
daily_df = (
    demand_df.groupby(["ship_date", "day_of_week"])
    .size()
    .reset_index(name="pallets")
)

dow_df = (
    daily_df.groupby("day_of_week")["pallets"]
    .agg(["mean", "min", "max"])
    .reindex(dow_order)
    .reset_index()
)
display(dow_df.round(0))

print(f"Shipping days: {shipping_days}; pallets per day: mean {daily_df['pallets'].mean():.0f}, "
      f"min {daily_df['pallets'].min()}, max {daily_df['pallets'].max()}")
 
peak_days_df = daily_df.sort_values("pallets", ascending=False).head(10)
display(peak_days_df)

# pallets picked per hour of the day
hourly_df = (
    demand_df.groupby(["ship_date", "hour"])
    .size()
    .reset_index(name="picks")
)
avg_hour_df = (
    hourly_df.groupby("hour")["picks"]
    .sum()
    .div(shipping_days)
    .reset_index(name="avg_picks")
)
display(avg_hour_df.round(1))

print(f"Busiest hour of the year: {hourly_df['picks'].max()} picks; "
      f"95th percentile hour: {hourly_df['picks'].quantile(0.95):.0f} picks")

 # validate
assert len(weekly_df) == WEEKS
assert weekly_df["pallets"].sum() == len(outbound_df)
assert daily_df["pallets"].sum() == len(outbound_df)


# chart: pallets per week across the year, and average pallets by day of week
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})

ax1.plot(weekly_df["week_start"], weekly_df["pallets"], color="#2a78d6", lw=2)
ax1.axhline(weekly_mean, color="grey", lw=0.8, ls="--")
ax1.set_ylim(0, weekly_df["pallets"].max() * 1.15)
ax1.set_ylabel("Pallets shipped per week")
ax1.set_title(f"Flat year: weekly CV {weekly_cv:.1%}, no trend", loc="left", fontweight="bold", fontsize=10)


ax2.bar([d[:3] for d in dow_df["day_of_week"]], dow_df["mean"], color="#2a78d6", width=0.6)
for i, v in enumerate(dow_df["mean"]):
    ax2.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=8)
ax2.set_ylabel("Average pallets per day")
ax2.set_title("Wednesday peak", loc="left", fontweight="bold", fontsize=10)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("Output/Task1c_Demand_Over_Time.png", dpi=150)
plt.show()

weekly_df.to_csv("Output/Task1c_Weekly.csv", index=False)
dow_df.to_csv("Output/Task1c_Day_Of_Week.csv", index=False)
avg_hour_df.to_csv("Output/Task1c_Hourly.csv", index=False)


# ***** part d: characterize restaurants, routes, and trucks *****
restaurants_df = pd.read_csv("Data/client_sites.csv")
outbound_df = pd.read_csv("Data/outbound.csv")
trucks_df = pd.read_csv("Data/trucks.csv")

# convert to a date time
outbound_df["ship_date"] = pd.to_datetime(outbound_df["ship_date"])
trucks_df["ship_date"] = pd.to_datetime(trucks_df["ship_date"])

# Monday of the week containing each shipment
outbound_df["week_start"] = (
    outbound_df["ship_date"]
    - pd.to_timedelta(outbound_df["ship_date"].dt.day_of_week, unit="D")
)

weekly_pallets_df = (
    outbound_df
    .groupby(["site_id", "week_start"])
    .agg(pallets=("pallet_id", "nunique"))
    .reset_index()
)

# include weeks with zero deliveries
# All restaurants and all weeks
all_sites = restaurants_df["site_id"].unique()
all_weeks = pd.date_range(
    outbound_df["week_start"].min(),
    outbound_df["week_start"].max(),
    freq="7D"
)

# Create all restaurant-week combinations
full_index = pd.MultiIndex.from_product(
    [all_sites, all_weeks],
    names=["site_id", "week_start"]
)

weekly_pallets_df = (
    weekly_pallets_df
    .set_index(["site_id", "week_start"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

restaurant_summary_df = (
    weekly_pallets_df
    .groupby("site_id")
    .agg(
        avg_pallets_per_week=("pallets", "mean"),
        std_pallets_per_week=("pallets", "std"),
        min_pallets_per_week=("pallets", "min"),
        max_pallets_per_week=("pallets", "max"),
        total_pallets=("pallets", "sum")
    )
    .reset_index()
)

# Add client and location information
restaurant_summary_df = restaurant_summary_df.merge(
    restaurants_df[["site_id", "client_id", "city", "state"]],
    on="site_id",
    how="left"
)

restaurant_summary_df.to_csv("Output/restaurant_pallets_per_week.csv")
print(
    restaurant_summary_df["avg_pallets_per_week"]
    .describe()
)

plt.figure(figsize=(9, 5))

plt.hist(
    restaurant_summary_df["avg_pallets_per_week"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Average Pallets per Restaurant per Week")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Weekly Restaurant Pallet Demand")
plt.grid(axis="y", alpha=0.3)

plt.show()


# *** delivery days
day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

trucks_df["day_of_week"] = trucks_df["ship_date"].dt.day_name()
outbound_df["day_of_week"] = outbound_df["ship_date"].dt.day_name()

# Aggregate to calendar day
daily_summary_df = (
    trucks_df.groupby("ship_date")
    .agg(
        trucks=("truck_id", "nunique"),
        pallets=("pallets", "sum"),
        stops=("stops", "sum")
    )
    .reset_index()
)

# Include days without deliveries
all_dates = pd.date_range(
    trucks_df["ship_date"].min(),
    trucks_df["ship_date"].max()
)

daily_summary_df = (
    daily_summary_df.set_index("ship_date")
    .reindex(all_dates, fill_value=0)
    .rename_axis("ship_date")
    .reset_index()
)

daily_summary_df["day_of_week"] = (
    daily_summary_df["ship_date"].dt.day_name()
)

weekday_summary_df = (
    daily_summary_df.groupby("day_of_week")
    [["trucks", "pallets", "stops"]]
    .mean()
    .reindex(day_order)
)

print(weekday_summary_df.round(2))


# *** routes
route_summary_df = (
    trucks_df
    .groupby("route_id")
    .agg(
        operating_day=("day_of_week", "first"),
        trips=("truck_id", "nunique"),
        avg_pallets=("pallets", "mean"),
        avg_stops=("stops", "mean"),
        min_pallets=("pallets", "min"),
        max_pallets=("pallets", "max")
    )
    .reset_index()
)

print(route_summary_df.head())
print(route_summary_df.describe())


# *** pallets and stops per truck
truck_load_summary = (
    trucks_df[["pallets", "stops"]].describe()
)

print(truck_load_summary)



# ***** part e: characterize supply *****
suppliers_df = pd.read_csv("Data/suppliers.csv")
inbound_df = pd.read_csv("Data/inbound.csv")

# date time adjustment 
inbound_df["receipt_date"] = pd.to_datetime(
    inbound_df["receipt_date"]
)
inbound_df["arrived_at"] = pd.to_datetime(
    inbound_df["arrived_at"]
)
inbound_df["putaway_at"] = pd.to_datetime(
    inbound_df["putaway_at"]
)

# Assign Monday-start weeks
inbound_df["week_start"] = (
    inbound_df["receipt_date"]
    - pd.to_timedelta(
        inbound_df["receipt_date"].dt.day_of_week,
        unit="D"
    )
)

# *** purchase orders per week
weekly_order_df = (
    inbound_df.groupby("week_start")
    .agg(
        purchase_orders=("po_id", "nunique"),
        pallets=("pallet_id", "nunique")
    )
    .reset_index()
)

print("\nPurchase orders per week:")
print(weekly_order_df["purchase_orders"].describe())

# *** SKUs and pallets per order
order_summary_df = (
    inbound_df.groupby("po_id")
    .agg(
        supplier_id=("supplier_id", "first"),
        receipt_date=("receipt_date", "first"),
        skus=("sku_id", "nunique"),
        pallets=("pallet_id", "nunique")
    )
    .reset_index()
)

print("\nSKUs and Pallets per order:")
print(order_summary_df[["skus", "pallets"]].describe())

# *** Supplier characterization
supplier_summary_df = (
    inbound_df.groupby("supplier_id")
    .agg(
        total_pallets=("pallet_id", "nunique"),
        total_pos=("po_id", "nunique"),
        distinct_skus=("sku_id", "nunique")
    )
    .reset_index()
)

supplier_summary_df = supplier_summary_df.merge(
    suppliers_df,
    on="supplier_id",
    how="left"
)

supplier_summary_df["pallet_share"] = (
    supplier_summary_df["total_pallets"]
    / supplier_summary_df["total_pallets"].sum()
)

supplier_summary_df = supplier_summary_df.sort_values(
    "total_pallets",
    ascending=False
)

print(f"number of suppliers: {suppliers_df.shape[0]}")
print(supplier_summary_df.head(20))
supplier_summary_df.to_csv("Output/supplier_pallet_share.csv")

print("\nSupplier lead times:")
print(supplier_summary_df["lead_time_days"].describe())

# *** receiving nights
inbound_df["day_of_week"] = (
    inbound_df["receipt_date"].dt.day_name()
)

receiving_daily_df = (
    inbound_df.groupby("receipt_date")
    .agg(
        pallets=("pallet_id", "nunique"),
        purchase_orders=("po_id", "nunique"),
        skus=("sku_id", "nunique")
    )
    .reset_index()
)

receiving_daily_df["day_of_week"] = (
    receiving_daily_df["receipt_date"].dt.day_name()
)

receiving_weekday_df = (
    receiving_daily_df.groupby("day_of_week")
    .agg(
        avg_pallets=("pallets", "mean"),
        avg_pos=("purchase_orders", "mean"),
        avg_skus=("skus", "mean"),
        max_pallets=("pallets", "max")
    )
    .reindex([
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday"
    ])
)

print("\nReceiving Nights summary:")
print(receiving_weekday_df.round(2))



# ***** part f: daily inventory *****
opening_df = pd.read_csv("Data/opening_stock.csv")

# *** daily inventory by SKU
# opening pallets per sku
opening_by_sku = (
    opening_df.groupby("sku_id")
    .agg(opening_stock=("pallet_id", "nunique"))
    .reset_index()
)

# daily receipts per SKU
daily_receipts = (
    inbound_df.groupby(["receipt_date", "sku_id"])
    .agg(receipts=("pallet_id", "nunique"))
    .reset_index()
    .rename(columns={"receipt_date": "date"})
)

# daily shipments per SKU
daily_shipments = (
    outbound_df.groupby(["ship_date", "sku_id"])
    .agg(shipments=("pallet_id", "nunique"))
    .reset_index()
    .rename(columns={"ship_date": "date"})
)

# merge dfs
all_skus = sorted(
    set(opening_df["sku_id"])
    | set(inbound_df["sku_id"])
    | set(outbound_df["sku_id"])
)

all_dates = pd.date_range(
    outbound_df["ship_date"].min(),
    outbound_df["ship_date"].max(),
    freq="D"
)

full_index = pd.MultiIndex.from_product(
    [all_dates, all_skus],
    names=["date", "sku_id"]
)

sku_daily_inventory_df = (
    pd.DataFrame(index=full_index)
    .reset_index()
)

sku_daily_inventory_df = sku_daily_inventory_df.merge(
    daily_receipts,
    on=["date", "sku_id"],
    how="left"
)

sku_daily_inventory_df = sku_daily_inventory_df.merge(
    daily_shipments,
    on=["date", "sku_id"],
    how="left"
)

sku_daily_inventory_df = sku_daily_inventory_df.merge(
    opening_by_sku,
    on="sku_id",
    how="left"
)

sku_daily_inventory_df[
    ["receipts", "shipments", "opening_stock"]
] = sku_daily_inventory_df[
    ["receipts", "shipments", "opening_stock"]
].fillna(0)

# calculate cumulative inventory
sku_daily_inventory_df = (
    sku_daily_inventory_df
    .sort_values(["sku_id", "date"])
)

sku_daily_inventory_df["net_change"] = (
    sku_daily_inventory_df["receipts"]
    - sku_daily_inventory_df["shipments"]
)

sku_daily_inventory_df["ending_stock"] = (
    sku_daily_inventory_df["opening_stock"]
    + sku_daily_inventory_df.groupby("sku_id")[
        "net_change"
    ].cumsum()
)



sku_stock_summary_df = (
    sku_daily_inventory_df.groupby("sku_id")
    .agg(
        avg_stock=("ending_stock", "mean"),
        peak_stock=("ending_stock", "max"),
        min_stock=("ending_stock", "min"),
        total_receipts=("receipts", "sum"),
        total_shipments=("shipments", "sum")
    )
    .reset_index()
)

print(sku_stock_summary_df.head())

# Largest SKUs by average inventory
print(
    sku_stock_summary_df.sort_values(
        "avg_stock",
        ascending=False
    ).head(20)
)

sku_stock_summary_df.to_csv("Output/sku_stock_summary.csv")

# *** total daily inventory metrics
# by pallet
daily_inventory_df = (
    sku_daily_inventory_df.groupby("date")
    .agg(
        receipts=("receipts", "sum"),
        shipments=("shipments", "sum"),
        ending_stock=("ending_stock", "sum")
    )
    .reset_index()
)

avg_stock = daily_inventory_df["ending_stock"].mean()
peak_stock = daily_inventory_df["ending_stock"].max()

# by slot utilization
slots_df = pd.read_csv("Data/slots.csv")
total_slots = slots_df["slot_id"].nunique()

daily_inventory_df["slot_utilization"] = (
    daily_inventory_df["ending_stock"] / total_slots
)

avg_slot_util = round((avg_stock / total_slots) * 100, 2)
peak_slot_util = round((peak_stock / total_slots) * 100, 2)

# by days of shipments
total_shipments = daily_inventory_df["shipments"].sum()
num_days = len(daily_inventory_df)

avg_daily_shipments = total_shipments / num_days

avg_days = avg_stock / avg_daily_shipments
peak_days = peak_stock / avg_daily_shipments

# report results
print(f"Average stock: {avg_stock:,.1f} pallets")
print(f"Peak stock: {peak_stock:,.0f} pallets")
print(f"Average utilization: {avg_slot_util}")
print(f"Peak utilization: {peak_slot_util}")
print(f"Average stock: {avg_days:.2f} days of shipments")
print(f"Peak stock: {peak_days:.2f} days of shipments")