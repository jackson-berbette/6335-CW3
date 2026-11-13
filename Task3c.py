import pandas as pd
import numpy as np
from IPython.display import display

# read relevant input files
inbound_df = pd.read_csv("Data/inbound.csv")
outbound_df = pd.read_csv("Data/outbound.csv")
slots_df = pd.read_csv("Data/slots.csv")
skus_df = pd.read_csv("Data/skus.csv")
clients_df = pd.read_csv("Data/clients.csv")
sku_velocity_df = pd.read_csv("Output/Task1b_SKU_Velocity.csv") # velocity classes and ranking from Tasb 1b

TRAVEL_SPEED_FT_PER_MIN = 264
LIFT_SPEED_FT_PER_MIN = 60
HANDLING_MIN = 1.5

# one table with every slot move: put-aways and picks
putaways_df = (
    inbound_df[["sku_id", "putaway_slot", "travel_ft", "lift_ft"]]
    .rename(columns={"putaway_slot": "slot_id"})
    .assign(move="putaway")
)

picks_df = (
    outbound_df[["sku_id", "picked_from_slot", "travel_ft", "lift_ft"]]
    .rename(columns={"picked_from_slot": "slot_id"})
    .assign(move="pick")
)

moves_df = (
    pd.concat([putaways_df, picks_df], ignore_index=True)
    .merge(slots_df[["slot_id", "aisle", "level"]], on="slot_id", how="left")
    .merge(skus_df[["sku_id", "client_id"]], on="sku_id", how="left")
    .merge(sku_velocity_df[["sku_id", "velocity_class", "rank"]], on="sku_id", how="left")
    .merge(clients_df[["client_id", "client_name"]], on="client_id", how="left")
)

moves_df["client"] = moves_df["client_name"].str.replace(" Restaurants", "")

# rule 3: slot move time in minutes; round-trip travel in miles
moves_df["move_minutes"] = (
    2 * moves_df["travel_ft"] / TRAVEL_SPEED_FT_PER_MIN
    + 2 * moves_df["lift_ft"] / LIFT_SPEED_FT_PER_MIN
    + HANDLING_MIN
)
moves_df["round_trip_miles"] = 2 * moves_df["travel_ft"] / 5280


# travel and labour for any grouping
def travel_labour_breakdown(group_column):
    table = (
        moves_df.groupby(group_column)
        .agg(
            moves=("move_minutes", "size"),
            travel_miles=("round_trip_miles", "sum"),
            avg_travel_ft=("travel_ft", "mean"),
            avg_lift_ft=("lift_ft", "mean"),
            labour_minutes=("move_minutes", "sum")
        )
        .reset_index()
    )

    table["labour_hours"] = table["labour_minutes"] / 60
    table["share_moves"] = table["moves"] / table["moves"].sum()
    table["share_travel"] = table["travel_miles"] / table["travel_miles"].sum()
    table["share_labour"] = table["labour_hours"] / table["labour_hours"].sum()
    table["min_per_move"] = table["labour_minutes"] / table["moves"]

    return table[
        [group_column, "moves", "share_moves", "travel_miles", "share_travel",
         "avg_travel_ft", "avg_lift_ft", "labour_hours", "share_labour", "min_per_move"]
    ]


class_breakdown_df = travel_labour_breakdown("velocity_class")
client_breakdown_df = travel_labour_breakdown("client").sort_values("moves", ascending=False)
level_breakdown_df = travel_labour_breakdown("level")
aisle_breakdown_df = travel_labour_breakdown("aisle")


# fastest 5% of SKUs, by pallets shipped (Task 1b ranking)
n_fastest = int(round(0.05 * len(skus_df)))  # 5% of 1,189 SKUs = 59
moves_df["fastest_5pct"] = moves_df["rank"] <= n_fastest

fastest_df = travel_labour_breakdown("fastest_5pct")

# validate
assert len(moves_df) == len(inbound_df) + len(outbound_df)
assert moves_df[["aisle", "velocity_class", "client"]].notna().all().all()
assert np.isclose(class_breakdown_df["share_travel"].sum(), 1)

display(class_breakdown_df.round(3))
display(client_breakdown_df.round(3))
display(level_breakdown_df.round(3))
display(aisle_breakdown_df.round(3))

fastest = fastest_df[fastest_df["fastest_5pct"]].iloc[0]
others = fastest_df[~fastest_df["fastest_5pct"]].iloc[0]

print(f"Fastest 5% of SKUs: top {n_fastest} of {len(skus_df)} by pallets shipped")
print(f"  slot moves: {fastest['moves']:,} (all other SKUs: {others['moves']:,})")
print(f"  share of slot moves: {fastest['share_moves']:.1%}")
print(f"  share of travel:     {fastest['share_travel']:.1%}")
print(f"  share of labour:     {fastest['share_labour']:.1%}")
print(f"  average travel per move: {fastest['avg_travel_ft']:.1f} ft "
      f"(all other SKUs: {others['avg_travel_ft']:.1f} ft)")


class_breakdown_df.to_csv("Output/Task3c_By_Class.csv", index=False)
client_breakdown_df.to_csv("Output/Task3c_By_Client.csv", index=False)
level_breakdown_df.to_csv("Output/Task3c_By_Level.csv", index=False)
aisle_breakdown_df.to_csv("Output/Task3c_By_Aisle.csv", index=False)
fastest_df.to_csv("Output/Task3c_Fastest_5pct.csv", index=False)
