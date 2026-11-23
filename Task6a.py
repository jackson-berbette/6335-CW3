import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

# design the moves are build for (Task 5 final design, run 12)
DESIGN = Path("Output/Evaluator_InputsLatest/Round2/P5_DoS_Target_v2_TallProtected_NEAREST")
OUT = Path("Output/Task6")
OUT.mkdir(parents=True, exist_ok=True)


# read relevant input files
slots_df = pd.read_csv("Data/slots.csv")
skus_df = pd.read_csv("Data/skus.csv")
current_stock_df = pd.read_csv("Data/current_stock.csv", parse_dates=["received_date"])
velocity_stock_df = pd.read_csv("Output/sku_velocity_and_stock.csv", index_col=0)
# Task 1f
slot_zones_df = pd.read_csv(DESIGN / "slot_zones.csv")
sku_zones_df = pd.read_csv(DESIGN / "sku_zones.csv")

TRAVEL_SPEED_FT_PER_MIN = 264 # rule 3
LIFT_SPEED_FT_PER_MIN = 60 # rule 3
HANDLING_MIN = 1.5 # rule 3
DAYS = 364
TIER = {"T1": 1, "T2": 2, "T3": 3, "T4": 4} # stay targets, nearest the dock first


# -------- Part a --------
# every pallet in the racks on 5 july 2026, with its slot, its zone under the design and its SKU's preferred zones
slots_df = slots_df.merge(slot_zones_df[["slot_id", "zone"]], on="slot_id", how="left")

pallets_df = (
    current_stock_df
    .merge(slots_df[["slot_id", "zone", "aisle_x_ft", "y_ft", "dist_dock_ft", "elev_ft"]], on="slot_id", how="left")
    .merge(sku_zones_df[["sku_id", "zone_1", "zone_2"]], on="sku_id", how="left")
    .merge(skus_df[["sku_id", "pallet_height_in"]], on="sku_id", how="left")
    .merge(velocity_stock_df[["sku_id", "pallets_yr"]], on="sku_id", how="left")
)

# remaining stay under FIFO: a pallet ships after the older pallets of its SKU, 
# so remaining days = its position in the SKU's queue / pallets shipped per day
pallets_df = pallets_df.sort_values(["sku_id", "received_date", "pallet_id"])
pallets_df["queue_position"] = pallets_df.groupby("sku_id").cumcount() + 1
pallets_df["remaining_days"] = pallets_df["queue_position"] / (pallets_df["pallets_yr"] / DAYS).replace(0, np.nan)

pallets_df["slot_tier"] = pallets_df["zone"].str[:2].map(TIER)
pallets_df["target_tier"] = pallets_df["zone_1"].str[:2].map(TIER)
pallets_df["placement"] = np.select(
    [pallets_df["zone"] == pallets_df["zone_1"],
     pallets_df["zone"] == pallets_df["zone_2"],
     pallets_df["slot_tier"] < pallets_df["target_tier"],
     pallets_df["slot_tier"] > pallets_df["target_tier"]],
    ["in first choice", "in second choice", "long stay sitting nearer the dock", "short stay sitting farther back"],
    default="right tier, wrong height part",
)

# zone fill on 5 July: how much of each zone is already taken, and by its own SKUs
zone_fill_df = (
    pd.DataFrame({
        "slots": slot_zones_df["zone"].value_counts(),
        "occupied": pallets_df["zone"].value_counts(),
        "occupied_by_own_skus": pallets_df.loc[pallets_df["zone"] == pallets_df["zone_1"], "zone"].value_counts(),
    })
    .fillna(0).astype(int)
    .sort_index()
)
zone_fill_df["fill"] = zone_fill_df["occupied"] / zone_fill_df["slots"]


# destination for each candidate: the nearest free slot that fits in its first-choice zone, else its second
occupied = set(current_stock_df["slot_id"])
free_slots_df = (
    slots_df[~slots_df["slot_id"].isin(occupied) & slots_df["zone"].notna()]
    .sort_values(["dist_dock_ft", "level", "slot_id"])
)


def plan_moves(candidates_df):
    free = {z: g[["slot_id", "max_pallet_in"]].values.tolist() for z, g in free_slots_df.groupby("zone")}
    rows = []
    for p in candidates_df.sort_values(["remaining_days", "pallet_id"], ascending=[False, True]).itertuples():
        for z in [p.zone_1, p.zone_2]:
            fits = [i for i, (_, h) in enumerate(free.get(z, [])) if h >= p.pallet_height_in]
            if fits:
                to_slot = free[z].pop(fits[0])[0]
                rows.append({"pallet_id": p.pallet_id, "to_slot": to_slot})
                break
    return pd.DataFrame(rows, columns=["pallet_id", "to_slot"])


def move_estimate(moves_df):
    # rule 2: slot to slot via the front cross-aisle, both lifts, one handling, at the rule 3 rates
    m = (moves_df
         .merge(pallets_df[["pallet_id", "slot_id", "aisle_x_ft", "y_ft", "dist_dock_ft", "elev_ft"]], on="pallet_id")
         .merge(slots_df[["slot_id", "aisle_x_ft", "y_ft", "dist_dock_ft", "elev_ft"]]
                .rename(columns=lambda c: c + "_to"), left_on="to_slot", right_on="slot_id_to"))
    travel = m["y_ft"] + (m["aisle_x_ft"] - m["aisle_x_ft_to"]).abs() + m["y_ft_to"]
    m["move_min"] = 2 * travel / TRAVEL_SPEED_FT_PER_MIN + 2 * (m["elev_ft"] + m["elev_ft_to"]) / LIFT_SPEED_FT_PER_MIN + HANDLING_MIN
    # the moved pallet's own pick from its new slot instead of its old one (negative = longer pick)
    m["own_pick_change_min"] = (2 * (m["dist_dock_ft"] - m["dist_dock_ft_to"]) / TRAVEL_SPEED_FT_PER_MIN
                                + 2 * (m["elev_ft"] - m["elev_ft_to"]) / LIFT_SPEED_FT_PER_MIN)
    return m


# candidate move sets, tested one by one in the evaluator
long_stay_front = pallets_df["placement"] == "long stay sitting nearer the dock"
short_stay_back = pallets_df["placement"] == "short stay sitting farther back"
variants = {
    "V1_long_stay_out_of_front_30d": pallets_df[long_stay_front & (pallets_df["remaining_days"] >= 30)],
    "V2_long_stay_out_of_front_60d": pallets_df[long_stay_front & (pallets_df["remaining_days"] >= 60)],
    "V3_short_stay_to_front_7d": pallets_df[short_stay_back & (pallets_df["remaining_days"] >= 7)],
}

summary = []
for name, candidates in variants.items():
    moves_df = plan_moves(candidates)
    est = move_estimate(moves_df)
    moves_df.to_csv(OUT / f"moves_{name}.csv", index=False)
    summary.append({"variant": name, "candidates": len(candidates), "moves": len(moves_df),
                    "est_one_off_hours": est["move_min"].sum() / 60,
                    "avg_move_min": est["move_min"].mean(),
                    "own_pick_change_hours": est["own_pick_change_min"].sum() / 60})
summary_df = pd.DataFrame(summary)


# validate
assert len(pallets_df) == len(current_stock_df)
assert pallets_df["zone_1"].notna().all()
for name in variants:
    m = pd.read_csv(OUT / f"moves_{name}.csv")
    assert m["pallet_id"].is_unique and m["to_slot"].is_unique
    assert not m["to_slot"].isin(occupied).any()

display(pallets_df["placement"].value_counts())
display(zone_fill_df.round(2))
display(pallets_df.groupby("placement")["remaining_days"].median().round(1))
display(summary_df.round(2))

summary_df.to_csv(OUT / "Task6a_Variants.csv", index=False)
zone_fill_df.to_csv(OUT / "Task6a_Zone_Fill_5July.csv")


