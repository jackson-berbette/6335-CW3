import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

# design the re-beaming is built for (Task 5 final design, Run 12)
DESIGN = Path("Output/Evaluator_InputsLatest/Round2/P5_DoS_Target_v2_TallProtected_NEAREST")
OUT = Path("Output/Task6")
OUT.mkdir(parents=True, exist_ok=True)

# read relevant input files
slots_df = pd.read_csv("Data/slots.csv")
skus_df = pd.read_csv("Data/skus.csv")
current_stock_df = pd.read_csv("Data/current_stock.csv")
slot_zones_df = pd.read_csv(DESIGN / "slot_zones.csv")

TRAVEL_SPEED_FT_PER_MIN = 264  # rule 3
LIFT_SPEED_FT_PER_MIN = 60     # rule 3
HANDLING_MIN = 1.5             # rule 3
CREW_MIN_PER_BEAM_LEVEL = 20   # rule 3 (re-beaming)
CLEARANCE_IN = 4               # rule 5
PITCH_IN = 4                   # rule 3: openings change in steps of 4 in
TALL_OPENING_IN = skus_df["pallet_height_in"].max() + CLEARANCE_IN  # 88 in + 4 in = 92 in: any tall pallet still fits
BEAM_IN = 5                    # beam depth implied by slots.csv elevations (e.g. 116 in + 5 in = 10.1 ft)


# -------- Part b --------
slots_df = slots_df.merge(slot_zones_df[["slot_id", "zone"]], on="slot_id", how="left")
slots_df["tier"] = slots_df["zone"].str[:2]
slots_df["tall_zone"] = slots_df["zone"].str.endswith("TALL")
slots_df["bay_key"] = slots_df["aisle"] + "-" + slots_df["side"] + "-" + slots_df["bay"].astype(str)
slots_df["top_level"] = slots_df["level"] == slots_df.groupby("bay_key")["level"].transform("max")


# new opening: lower every TALL-zone opening above 92 in as far as the 4 in pitch allows, but not below 92 in,
# so every tall pallet still fits and every TALL slot stays above 72 in (the design's zones stay valid).
# Lowering only ever reduces a bay's total height, so the "no more than today" rule always holds.
excess = (slots_df["opening_height_in"] - TALL_OPENING_IN).clip(lower=0)
slots_df["candidate_opening_in"] = np.where(
    slots_df["tall_zone"], slots_df["opening_height_in"] - PITCH_IN * (excess // PITCH_IN), slots_df["opening_height_in"]
)


def rebeam_plan(bays, below_top_only=False):
    """New openings and elevations for the chosen bays, with the one-off work it causes."""
    s = slots_df.copy()
    chosen = s["bay_key"].isin(bays) & (~s["top_level"] if below_top_only else True)
    s["new_opening_in"] = np.where(chosen, s["candidate_opening_in"], s["opening_height_in"])
    s["changed"] = s["new_opening_in"] != s["opening_height_in"]
    s = s.sort_values(["bay_key", "level"])

    # elevation of a level = openings + beams of the levels below it
    s["new_elev_ft"] = (s.groupby("bay_key")["new_opening_in"].cumsum() - s["new_opening_in"]
                        + BEAM_IN * (s["level"] - 1)) / 12
    old_elev = (s.groupby("bay_key")["opening_height_in"].cumsum() - s["opening_height_in"] + BEAM_IN * (s["level"] - 1)) / 12
    s["elev_drop_ft"] = old_elev - s["new_elev_ft"]


   # beams that move: from the top of the lowest changed level to the top of the bay
    lowest = s[s["changed"]].groupby("bay_key")["level"].min()
    top = s.groupby("bay_key")["level"].max()
    beam_levels = (top[lowest.index] - lowest + 1).sum()

    # every pallet in a re-beamed bay is taken down and put back (here: back into its own slot, which still fits)
    bays_changed = lowest.index
    taken_down = (current_stock_df
                  .merge(s[["slot_id", "bay_key", "y_ft", "elev_ft", "new_elev_ft", "new_opening_in"]], on="slot_id")
                  .merge(skus_df[["sku_id", "pallet_height_in"]], on="sku_id"))
    taken_down = taken_down[taken_down["bay_key"].isin(bays_changed)]
    assert (taken_down["new_opening_in"] - CLEARANCE_IN >= taken_down["pallet_height_in"]).all()

    down_min = 2 * taken_down["y_ft"] / TRAVEL_SPEED_FT_PER_MIN + 2 * taken_down["elev_ft"] / LIFT_SPEED_FT_PER_MIN + HANDLING_MIN
    back_min = 2 * taken_down["y_ft"] / TRAVEL_SPEED_FT_PER_MIN + 2 * taken_down["new_elev_ft"] / LIFT_SPEED_FT_PER_MIN + HANDLING_MIN

    rebeam_df = s.loc[chosen & s["changed"], ["slot_id", "new_opening_in"]]
    moves_df = taken_down[["pallet_id", "slot_id"]].rename(columns={"slot_id": "to_slot"})
    summary = {
        "bays_rebeamed": len(bays_changed),
        "slots_changed": int(s["changed"].sum()),
        "beam_levels_moved_est": int(beam_levels),
        "crew_hours_est": beam_levels * CREW_MIN_PER_BEAM_LEVEL / 60,
        "pallets_taken_down": len(taken_down),
        "forklift_hours_est": (down_min.sum() + back_min.sum()) / 60,
        "avg_opening_cut_in": (s.loc[s["changed"], "opening_height_in"] - s.loc[s["changed"], "new_opening_in"]).mean(),
        "slots_lowered": int((s["elev_drop_ft"] > 0.01).sum()),
        "avg_elev_drop_ft_of_lowered_slots": s.loc[s["elev_drop_ft"] > 0.01, "elev_drop_ft"].mean(),
    }
    summary["total_one_off_hours_est"] = summary["crew_hours_est"] + summary["forklift_hours_est"]
    return rebeam_df, moves_df, summary


# candidate versions: re-beam the bays that hold the nearest stay targets, where most moves happen
def bays_with(tiers):
    return slots_df.loc[slots_df["tier"].isin(tiers), "bay_key"].unique()

variants = {
    "R1_T1_bays": bays_with(["T1"]),
    "R1b_T1_bays_below_top_only": bays_with(["T1"]),
    "R2_T1_T2_bays": bays_with(["T1", "T2"]),
    "R3_all_bays": slots_df["bay_key"].unique(),
}

# the full rebeam template: every slot with today's opening, new_opening_in filled only where it changes
template = slots_df[["slot_id", "aisle", "side", "bay", "level", "opening_height_in"]].rename(
    columns={"opening_height_in": "current_opening_in"})

summary = []
for name, bays in variants.items():
    rebeam_df, moves_df, s = rebeam_plan(bays, below_top_only="below_top" in name)
    (template.merge(rebeam_df, on="slot_id", how="left")
     .to_csv(OUT / f"rebeam_{name}.csv", index=False))
    moves_df.to_csv(OUT / f"moves_{name}.csv", index=False)
    summary.append({"variant": name, **s})
summary_df = pd.DataFrame(summary)


# validate: every change is on the 4 in pitch, at least 24 in, and no bay grows
check = slots_df.merge(pd.read_csv(OUT / "rebeam_R3_all_bays.csv")[["slot_id", "new_opening_in"]], on="slot_id")
changed = check["new_opening_in"].notna()
assert ((check.loc[changed, "opening_height_in"] - check.loc[changed, "new_opening_in"]) % PITCH_IN == 0).all()
assert (check.loc[changed, "new_opening_in"] >= max(24, TALL_OPENING_IN)).all()
new_total = check["new_opening_in"].fillna(check["opening_height_in"]).groupby(check["bay_key"]).sum()
assert (new_total <= check.groupby("bay_key")["opening_height_in"].sum()).all()


oversized = slots_df["tall_zone"] & (slots_df["opening_height_in"] > TALL_OPENING_IN)
print(f"TALL-zone openings above {TALL_OPENING_IN} in: {oversized.sum():,}, of which at the top of their bay: "
      f"{(oversized & slots_df['top_level']).sum():,} (lowering these cuts unused height but lowers no slot above)")
display(summary_df.round(2))
summary_df.to_csv(OUT / "Task6b_Variants.csv", index=False)

