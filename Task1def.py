import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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