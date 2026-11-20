from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "Output/Evaluator_Inputs"
OUTPUT.mkdir(exist_ok=True)
TALL_THRESHOLD_IN = 72
STAY_ZONES = ["T1 (up to 14 d)", "T2 (14-30 d)",
              "T3 (30-60 d)", "T4 (over 60 d)"]

# Read each input once. The same data is used for every policy.
slots = pd.read_csv("Data/slots.csv").sort_values(
    ["dist_dock_ft", "level", "slot_id"]
).reset_index(drop=True)
skus = pd.read_csv("Data/skus.csv")
velocity = pd.read_csv("Output/CW3P1T1_VelocityStock_WembyFanClub.csv")
zone_sizes = pd.read_csv("Output/Task4b_Slots_By_Zone_Height.csv")

sku_info = velocity[["sku_id", "pallets_shipped", "avg_stock_pallets", "peak_stock_pallets"]].merge(
    skus[["sku_id", "pallet_height_in"]], on="sku_id", validate="one_to_one"
)
sku_info = sku_info.sort_values(
    ["pallets_shipped", "sku_id"], ascending=[False, True]
).reset_index(drop=True)

# Match Task 4b's velocity class and duration-of-stay definitions.
share_before = sku_info.pallets_shipped.cumsum().shift(fill_value=0) / sku_info.pallets_shipped.sum()
sku_info["class"] = np.select([share_before < 0.5, share_before < 0.8], ["A", "B"], default="C")
sku_info["expected_stay"] = sku_info.avg_stock_pallets / (
    sku_info.pallets_shipped / 364
).replace(0, np.nan)
sku_info["stay"] = pd.cut(
    sku_info.expected_stay.fillna(np.inf), [0, 14, 30, 60, np.inf], labels=STAY_ZONES
).astype(str)


def save_policy(folder_name, slot_assignments, sku_assignments, shortfalls=None):
    folder = OUTPUT / folder_name
    folder.mkdir(exist_ok=True)
    assert len(slot_assignments) == len(slots)
    assert len(sku_assignments) == len(sku_info)
    assert slot_assignments.slot_id.is_unique and sku_assignments.sku_id.is_unique
    slot_assignments.to_csv(folder / "slot_zones.csv", index=False)
    sku_assignments.to_csv(folder / "sku_zones.csv", index=False)
    if shortfalls is not None:
        shortfalls.to_csv(folder / "allocation_shortfalls.csv", index=False)
    print(f"{folder_name}: {len(slot_assignments):,} slots, {len(sku_assignments):,} SKUs")


def sku_preferences(preferences):
    result = sku_info[["sku_id"]].copy()
    for index in range(3):
        result[f"zone_{index + 1}"] = [
            choices[index] if index < len(choices) else "" for choices in preferences
        ]
    return result


def generate_shared_or_grouped(folder_name, policy_name, zones, preferences):
    assignments = slots[["slot_id"]].copy()
    assignments["zone"] = ""
    if len(zones) == 1:
        assignments["zone"] = zones[0]
    else:
        policy_sizes = zone_sizes.loc[
            (zone_sizes.policy == policy_name)
            & (zone_sizes.tall_protected.astype(str).str.lower() == "false")
        ].set_index("zone")
        cursor = 0
        for zone in zones[:-1]:
            count = int(policy_sizes.loc[zone, "zone_slots_nearest_first"])
            assignments.loc[cursor:cursor + count - 1, "zone"] = zone
            cursor += count
        # Any remaining slots join the last zone, including spare capacity.
        assignments.loc[cursor:, "zone"] = zones[-1]
    assignments["fill_order"] = assignments.groupby("zone").cumcount() + 1
    save_policy(folder_name, assignments, sku_preferences(preferences))


def generate_dedicated(folder_name, protect_tall):
    assignments = slots[["slot_id"]].copy()
    assignments["zone"] = ""  # Blank means the slot is not assigned to any zone.
    assignments["fill_order"] = pd.Series(pd.NA, index=assignments.index, dtype="Int64")
    free = np.ones(len(slots), dtype=bool)
    max_heights = slots.max_pallet_in.to_numpy()

    # Task 4b allocates dedicated SKU capacity in this priority order.
    priority = sku_info.copy()
    priority["slots_per_move"] = priority.peak_stock_pallets / (
        2 * priority.pallets_shipped.replace(0, np.nan)
    )
    priority = priority.sort_values(["slots_per_move", "sku_id"], na_position="last")

    shortfall_rows = []
    for sku in priority.itertuples(index=False):
        eligible = free & (max_heights >= sku.pallet_height_in)
        if protect_tall:
            if float(sku.pallet_height_in) > TALL_THRESHOLD_IN:
                eligible &= max_heights > TALL_THRESHOLD_IN
            else:
                eligible &= max_heights <= TALL_THRESHOLD_IN

        required = int(sku.peak_stock_pallets)
        chosen = np.flatnonzero(eligible)[:required]
        free[chosen] = False
        assignments.loc[chosen, "zone"] = sku.sku_id
        assignments.loc[chosen, "fill_order"] = np.arange(1, len(chosen) + 1)
        shortfall_rows.append({
            "sku_id": sku.sku_id, "required_slots": required,
            "allocated_slots": len(chosen), "shortfall": required - len(chosen)
        })

    # A dedicated SKU has no alternative zones.
    preferences = sku_preferences([[sku_id] for sku_id in sku_info.sku_id])
    shortfalls = pd.DataFrame(shortfall_rows)
    save_policy(folder_name, assignments, preferences, shortfalls)
    print(f"  Unmet dedicated slots: {shortfalls.shortfall.sum():,}")


# Policy 1: Dedicated, no tall-opening protection.
generate_dedicated("P1_Dedicated_COL", protect_tall=False)

# Policy 2: A/B/C class zones, closest open location within each zone.
class_fallback = {
    "A": ["A", "B", "C"], "B": ["B", "A", "C"], "C": ["C", "B", "A"]
}
generate_shared_or_grouped(
    "P2_Class_COL", "Product-Class-COL", ["A", "B", "C"],
    [class_fallback[cls] for cls in sku_info["class"]]
)

# Policy 3: Dedicated, with tall openings reserved for tall pallets.
generate_dedicated("P3_Dedicated_COL_TallProtected", protect_tall=True)

# Policy 4: All SKUs share one zone.
generate_shared_or_grouped(
    "P4_Shared_COL", "Product-Shared-COL", ["All SKUs"],
    [["All SKUs"] for _ in range(len(sku_info))]
)

# Policy 5: Expected duration of stay determines the preferred zone.
stay_fallback = {
    STAY_ZONES[0]: [STAY_ZONES[0], STAY_ZONES[1], STAY_ZONES[2]],
    STAY_ZONES[1]: [STAY_ZONES[1], STAY_ZONES[0], STAY_ZONES[2]],
    STAY_ZONES[2]: [STAY_ZONES[2], STAY_ZONES[1], STAY_ZONES[3]],
    STAY_ZONES[3]: [STAY_ZONES[3], STAY_ZONES[2], STAY_ZONES[1]],
}
generate_shared_or_grouped(
    "P5_DoS_Target", "Duration-of-Stay-Target", STAY_ZONES,
    [stay_fallback[stay] for stay in sku_info["stay"]]
)
print(f"All evaluator files saved to: {OUTPUT}")
