import pandas as pd
import numpy as np
from IPython.display import display
 
TEAM_NAME = "WembyFanClub" 
 
# read relevant input files
inbound_df = pd.read_csv("Data/inbound.csv", parse_dates=["receipt_date"])
outbound_df = pd.read_csv("Data/outbound.csv", parse_dates=["ship_date"])
opening_stock_df = pd.read_csv("Data/opening_stock.csv")
current_stock_df = pd.read_csv("Data/current_stock.csv")
skus_df = pd.read_csv("Data/skus.csv")
sku_velocity_df = pd.read_csv("Output/Task1b_SKU_Velocity.csv")  # pallets shipped and velocity class from Task 1b
 
# every day of the historical year
days = pd.date_range(
    min(inbound_df["receipt_date"].min(), outbound_df["ship_date"].min()),
    max(inbound_df["receipt_date"].max(), outbound_df["ship_date"].max())
)
sku_ids = skus_df["sku_id"]
 
# pallets received and shipped per SKU per day
received_df = (
    inbound_df.groupby(["receipt_date", "sku_id"]).size()
    .unstack(fill_value=0)
    .reindex(index=days, columns=sku_ids, fill_value=0)
)
 
shipped_df = (
    outbound_df.groupby(["ship_date", "sku_id"]).size()
    .unstack(fill_value=0)
    .reindex(index=days, columns=sku_ids, fill_value=0)
)
 
opening_stock = opening_stock_df["sku_id"].value_counts().reindex(sku_ids, fill_value=0)
current_stock = current_stock_df["sku_id"].value_counts().reindex(sku_ids, fill_value=0)
 
# daily stock per SKU at midnight: opening stock + receipts to date - shipments to date
stock_df = opening_stock + received_df.cumsum() - shipped_df.cumsum()
 
# one row per SKU: pallets shipped, velocity class, average and peak stock
velocity_stock_df = (
    sku_velocity_df[["sku_id", "pallets_yr", "velocity_class"]]
    .rename(columns={"pallets_yr": "pallets_shipped"})
    .merge(
        pd.DataFrame({
            "sku_id": sku_ids.values,
            "avg_stock_pallets": stock_df.mean().round(2).values,
            "peak_stock_pallets": stock_df.max().values
        }),
        on="sku_id", how="left"
    )
    .sort_values(["pallets_shipped", "sku_id"], ascending=[False, True])
)
 
 
# validate
assert len(velocity_stock_df) == len(skus_df)
assert velocity_stock_df["pallets_shipped"].sum() == len(outbound_df)
assert (stock_df - received_df >= 0).all().all()               # never ships more than is in stock
assert (stock_df.iloc[-1] == current_stock).all()              # year-end stock matches current_stock.csv
assert velocity_stock_df.notna().all().all()
 
display(velocity_stock_df.head(10))
display(
    velocity_stock_df.groupby("velocity_class")
    .agg(skus=("sku_id", "size"),
         pallets_shipped=("pallets_shipped", "sum"),
         avg_stock_pallets=("avg_stock_pallets", "sum"))
    .round(1)
)
 
print(f"Rows: {len(velocity_stock_df)}; days: {len(days)}")
print(f"Average stock across all SKUs: {stock_df.sum(axis=1).mean():,.0f} pallets; "
      f"peak: {stock_df.sum(axis=1).max():,} pallets")
 
velocity_stock_df.to_csv(f"Output/CW3P1T1_VelocityStock_{TEAM_NAME}.csv", index=False)