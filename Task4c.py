from pathlib import Path
import heapq

import numpy as np
import pandas as pd


# 1. ----- SETTINGS AND INPUT FILES -----

# Speeds and handling time are taken from Task 3d.
HORIZONTAL_SPEED_FT_PER_MIN = 264
VERTICAL_SPEED_FT_PER_MIN = 60
HANDLING_MIN_PER_MOVE = 1.5
DAYS_IN_STOCK_HISTORY = 364

POLICIES = {
    "Product-Class-COL": "P2_Class_COL",
    "Product-Shared-COL": "P4_Shared_COL",
    "Duration-of-Stay-Target": "P5_DoS_Target",
}
STAY_ZONES = [
    "T1 (up to 14 d)", "T2 (14-30 d)",
    "T3 (30-60 d)", "T4 (over 60 d)",
]

# read relevant data
slots = pd.read_csv("Data/slots.csv").sort_values(
    ["dist_dock_ft", "level", "slot_id"]
).reset_index(drop=True)
skus = pd.read_csv("Data/skus.csv")
velocity_stock = pd.read_csv("Output/CW3P1T1_VelocityStock_WembyFanClub.csv")
inbound = pd.read_csv("Data/inbound.csv")
outbound = pd.read_csv("Data/outbound.csv")
opening_stock = pd.read_csv("Data/opening_stock.csv")
zone_sizes = pd.read_csv("Output/Task4b_Slots_By_Zone_Height.csv")
feasibility = pd.read_csv("Output/Task4b_Feasibility.csv")

assert len(slots) == 8365
assert len(skus) == 1189

# Lookup tables: the array index is the position of a slot in dock-distance order.
slot_ids = slots["slot_id"].to_numpy()
slot_index = {slot_id: index for index, slot_id in enumerate(slot_ids)}
slot_distance = slots["dist_dock_ft"].to_numpy()
slot_elevation = slots["elev_ft"].to_numpy()
slot_max_height = slots["max_pallet_in"].to_numpy()
slot_opening_height = slots["opening_height_in"].to_numpy()


# 2. ----- CLASSIFY SKUS BY VELOCITY AND EXPECTED DURATION OF STAY -----

sku_info = velocity_stock[[
    "sku_id", "pallets_shipped", "avg_stock_pallets", "peak_stock_pallets"
]].merge(skus[["sku_id", "pallet_height_in"]], on="sku_id")

sku_info = sku_info.sort_values(
    ["pallets_shipped", "sku_id"], ascending=[False, True]
).reset_index(drop=True)

# A SKUs account for the first 50% of shipments; B for the next 30%; C the rest.
share_before_sku = (
    sku_info["pallets_shipped"].cumsum().shift(fill_value=0)
    / sku_info["pallets_shipped"].sum()
)
sku_info["velocity_class"] = np.select(
    [share_before_sku < 0.50, share_before_sku < 0.80],
    ["A", "B"], default="C"
)

# Little's Law estimate: expected days in storage = average inventory / daily shipments.
daily_shipments = sku_info["pallets_shipped"] / DAYS_IN_STOCK_HISTORY
sku_info["expected_stay_days"] = (
    sku_info["avg_stock_pallets"] / daily_shipments.replace(0, np.nan)
)
sku_info["stay_zone"] = pd.cut(
    sku_info["expected_stay_days"].fillna(np.inf),
    bins=[0, 14, 30, 60, np.inf],
    labels=STAY_ZONES,
).astype(str)

sku_height = sku_info.set_index("sku_id")["pallet_height_in"].to_dict()
sku_class = sku_info.set_index("sku_id")["velocity_class"].to_dict()
sku_stay_zone = sku_info.set_index("sku_id")["stay_zone"].to_dict()
sku_velocity = sku_info.set_index("sku_id")["pallets_shipped"].to_dict()

# Only slots tall enough for the shortest pallet can be used.
usable_slot_indices = np.flatnonzero(
    slot_max_height >= sku_info["pallet_height_in"].min()
)


# 3. RECREATE THE RESERVED ZONES FROM TASK 4B -------------------------------

def create_storage_zones(policy):
    if policy == "Product-Class-COL":
        preferred_zones = ["A", "B", "C"]
    elif policy == "Product-Shared-COL":
        preferred_zones = ["All SKUs"]
    else:
        preferred_zones = STAY_ZONES

    policy_sizes = zone_sizes.loc[
        (zone_sizes["policy"] == policy)
        & (zone_sizes["tall_protected"] == False)
    ].set_index("zone")

    remaining_slots = usable_slot_indices.copy()
    zone_slots = {}
    for zone in preferred_zones:
        count = int(policy_sizes.loc[zone, "zone_slots_nearest_first"])
        zone_slots[zone] = remaining_slots[:count].copy()
        remaining_slots = remaining_slots[count:]

    zone_slots["UNRESERVED"] = remaining_slots
    return preferred_zones, zone_slots


def preferred_zone_for_sku(policy, sku_id):
    if policy == "Product-Shared-COL":
        return "All SKUs"
    if policy == "Product-Class-COL":
        return sku_class[sku_id]
    return sku_stay_zone[sku_id]


# 4. SIMULATE PUT-AWAY AND PICKING -------------------------------------------

# function to simulate the history under the given storage policy
def simulate_policy(policy):
    preferred_zones, zone_slots = create_storage_zones(policy)
    slot_to_zone = {
        int(index): zone
        for zone, indices in zone_slots.items()
        for index in indices
    }

    # Each zone has one min-heap per opening-height type. The smallest slot
    # index is the closest available slot, because slots were sorted by dock distance.
    free_slots = {zone: {} for zone in zone_slots}
    for zone, indices in zone_slots.items():
        for index in indices:
            opening_type = slot_max_height[index]
            free_slots[zone].setdefault(opening_type, []).append(int(index))
        for heap in free_slots[zone].values():
            heapq.heapify(heap)

    pallet_location = {}       # pallet ID -> assigned slot index
    occupied = np.zeros(len(slots), dtype=bool)
    move_records = []
    daily_records = []
    counters = {
        "failed_opening": 0, "failed_putaway": 0,
        "cross_zone": 0, "unreserved": 0,
        "opening_overflow": 0, "opening_unreserved": 0,
        "peak_occupied": 0, "min_free": len(slots),
        "missing_picks": 0,
    }

    # try preferred zone, then other zones, then unreserved slots
    def place_pallet(pallet_id, sku_id, kind, original_slot=None):
        required_height = sku_height[sku_id]
        preferred = preferred_zone_for_sku(policy, sku_id)
        search_order = (
            [preferred]
            + [zone for zone in preferred_zones if zone != preferred]
            + ["UNRESERVED"]
        )

        for zone in search_order:
            # Consider the closest free slot in every compatible height type.
            candidates = [
                (heap[0], opening_type)
                for opening_type, heap in free_slots[zone].items()
                if opening_type >= required_height and heap
            ]
            if not candidates:
                continue

            slot_idx, opening_type = min(candidates)
            heapq.heappop(free_slots[zone][opening_type])
            occupied[slot_idx] = True
            pallet_location[pallet_id] = slot_idx

            if zone != preferred:
                key = "cross_zone" if kind == "putaway" else "opening_overflow"
                counters[key] += 1
            if zone == "UNRESERVED":
                key = "unreserved" if kind == "putaway" else "opening_unreserved"
                counters[key] += 1

            if kind == "putaway":
                move_records.append((
                    kind, pallet_id, sku_id, original_slot,
                    slot_idx, zone, zone != preferred
                ))
            return

        # If no compatible free slot exists anywhere, record a failure.
        key = "failed_opening" if kind == "opening" else "failed_putaway"
        counters[key] += 1
        pallet_location[pallet_id] = None
        if kind == "putaway":
            move_records.append((
                kind, pallet_id, sku_id, original_slot, None, "FAILED", True
            ))

    def pick_pallet(row):
        pallet_id = row.pallet_id
        if pallet_id not in pallet_location:
            counters["missing_picks"] += 1
            return

        slot_idx = pallet_location.pop(pallet_id)
        if slot_idx is not None:
            occupied[slot_idx] = False
            zone = slot_to_zone[slot_idx]
            opening_type = slot_max_height[slot_idx]
            heapq.heappush(free_slots[zone][opening_type], slot_idx)

        move_records.append((
            "pick", pallet_id, row.sku_id,
            row.picked_from_slot, slot_idx, "", False
        ))

    # Assign locations to opening inventory, fastest-moving SKUs first.
    # This initial redistribution is assumed to have occurred before the year.
    initial = opening_stock.copy()
    initial["velocity"] = initial["sku_id"].map(sku_velocity)
    initial = initial.sort_values(
        ["velocity", "sku_id", "pallet_id"],
        ascending=[False, True, True]
    )
    for row in initial.itertuples(index=False):
        place_pallet(row.pallet_id, row.sku_id, "opening")

    counters["peak_occupied"] = int(occupied.sum())

    receipts = inbound.copy()
    shipments = outbound.copy()
    receipts["day"] = pd.to_datetime(receipts["receipt_date"])
    shipments["day"] = pd.to_datetime(shipments["ship_date"])
    receipts["velocity"] = receipts["sku_id"].map(sku_velocity)
    receipts = receipts.sort_values(
        ["day", "velocity", "arrived_at", "pallet_id"],
        ascending=[True, False, True, True]
    )
    shipments = shipments.sort_values(["day", "picked_at", "pallet_id"])

    receipts_by_day = dict(tuple(receipts.groupby("day", sort=False)))
    shipments_by_day = dict(tuple(shipments.groupby("day", sort=False)))
    dates = pd.date_range(
        min(receipts["day"].min(), shipments["day"].min()),
        max(receipts["day"].max(), shipments["day"].max())
    )

    for day in dates:
        # Day shift: retrieve and ship pallets.
        if day in shipments_by_day:
            for row in shipments_by_day[day].itertuples(index=False):
                pick_pallet(row)

        # Night shift: put away arrivals, with fast SKUs processed first.
        if day in receipts_by_day:
            for row in receipts_by_day[day].itertuples(index=False):
                place_pallet(row.pallet_id, row.sku_id, "putaway", row.putaway_slot)

        number_occupied = int(occupied.sum())
        counters["peak_occupied"] = max(counters["peak_occupied"], number_occupied)
        counters["min_free"] = min(counters["min_free"], len(slots) - number_occupied)
        daily_records.append((
            day.date(), number_occupied,
            counters["cross_zone"], counters["failed_putaway"]
        ))

    moves = pd.DataFrame(move_records, columns=[
        "move", "pallet_id", "sku_id", "old_slot",
        "new_idx", "zone", "overflow"
    ])
    daily = pd.DataFrame(daily_records, columns=[
        "day", "occupied", "cumulative_overflow", "cumulative_failed_putaway"
    ])
    return moves, daily, counters, zone_slots


# 5. ESTIMATE TRAVEL, VERTICAL LIFT, AND LABOR -------------------------------

def add_movement_costs(moves):
    historical_moves = pd.concat([
        inbound[["pallet_id", "travel_ft", "lift_ft", "putaway_slot"]]
        .rename(columns={"putaway_slot": "old_slot"})
        .assign(move="putaway"),
        outbound[["pallet_id", "travel_ft", "lift_ft", "picked_from_slot"]]
        .rename(columns={"picked_from_slot": "old_slot"})
        .assign(move="pick"),
    ], ignore_index=True)

    moves = moves.merge(
        historical_moves,
        on=["pallet_id", "move", "old_slot"],
        how="left", validate="one_to_one"
    )
    successful = moves["new_idx"].notna()
    moves["new_slot"] = None
    moves["new_travel_ft"] = np.nan
    moves["new_lift_ft"] = np.nan
    moves["unused_height_in"] = np.nan

    new_indices = moves.loc[successful, "new_idx"].astype(int).to_numpy()
    old_indices = moves.loc[successful, "old_slot"].map(slot_index).astype(int).to_numpy()

    moves.loc[successful, "new_slot"] = slot_ids[new_indices]
    moves.loc[successful, "new_travel_ft"] = np.maximum(
        0,
        moves.loc[successful, "travel_ft"].to_numpy()
        + slot_distance[new_indices] - slot_distance[old_indices]
    )
    moves.loc[successful, "new_lift_ft"] = np.maximum(
        0,
        moves.loc[successful, "lift_ft"].to_numpy()
        + slot_elevation[new_indices] - slot_elevation[old_indices]
    )
    moves.loc[successful, "unused_height_in"] = (
        slot_opening_height[new_indices]
        - moves.loc[successful, "sku_id"].map(sku_height).to_numpy()
    )

    moves["minutes"] = (
        2 * moves["new_travel_ft"] / HORIZONTAL_SPEED_FT_PER_MIN
        + 2 * moves["new_lift_ft"] / VERTICAL_SPEED_FT_PER_MIN
        + HANDLING_MIN_PER_MOVE
    )
    return moves


# baseline for comparison
def historical_reference():
    history = pd.concat([
        inbound[["travel_ft", "lift_ft", "putaway_slot", "sku_id"]]
        .rename(columns={"putaway_slot": "slot_id"}),
        outbound[["travel_ft", "lift_ft", "picked_from_slot", "sku_id"]]
        .rename(columns={"picked_from_slot": "slot_id"}),
    ], ignore_index=True)
    indices = history["slot_id"].map(slot_index).astype(int).to_numpy()
    minutes = (
        2 * history["travel_ft"] / HORIZONTAL_SPEED_FT_PER_MIN
        + 2 * history["lift_ft"] / VERTICAL_SPEED_FT_PER_MIN
        + HANDLING_MIN_PER_MOVE
    )
    return {
        "policy": "Historical reference",
        "slots_required": 6260,
        "binding_margin_slots": 1398,
        "zone_footprint_slots": np.nan,
        "travel_roundtrip_miles": 2 * history["travel_ft"].sum() / 5280,
        "lift_roundtrip_ft": 2 * history["lift_ft"].sum(),
        "labor_hours": minutes.sum() / 60,
        "avg_minutes_per_move": minutes.mean(),
        "avg_unused_height_in": (
            slot_opening_height[indices]
            - history["sku_id"].map(sku_height).to_numpy()
        ).mean(),
        "pct_short_in_tall": 100 * (
            (history["sku_id"].map(sku_height) <= 72)
            & (slot_max_height[indices] > 72)
        ).mean(),
        "overflow_putaways": np.nan,
        "unreserved_putaways": np.nan,
        "opening_overflow": np.nan,
        "failed_putaways": 0,
        "failed_opening": 0,
        "missing_picks": 0,
        "peak_occupied": 6260,
    }


# 6. RUN POLICIES AND SAVE COMPARISON ----------------------------------------

def main():
    results = [historical_reference()]

    for policy, filename_prefix in POLICIES.items():
        print(f"Simulating {policy} ...", flush=True)
        moves, daily, counters, zones = simulate_policy(policy)
        moves = add_movement_costs(moves)

        moves.to_csv(f"Output/Task4c/{filename_prefix}_moves.csv", index=False)
        daily.to_csv(f"Output/Task4c/{filename_prefix}_daily.csv", index=False)

        completed = moves[moves["new_idx"].notna()]
        indices = completed["new_idx"].astype(int).to_numpy()
        capacity = feasibility.loc[
            (feasibility["policy"] == policy)
            & (feasibility["tall_protected"] == False)
        ].iloc[0]

        results.append({
            "policy": policy,
            "slots_required": int(capacity["slots_required"]),
            "binding_margin_slots": int(capacity["margin_slots"]),
            "zone_footprint_slots": sum(
                len(indices) for zone, indices in zones.items()
                if zone != "UNRESERVED"
            ),
            "travel_roundtrip_miles": 2 * completed["new_travel_ft"].sum() / 5280,
            "lift_roundtrip_ft": 2 * completed["new_lift_ft"].sum(),
            "labor_hours": completed["minutes"].sum() / 60,
            "avg_minutes_per_move": completed["minutes"].mean(),
            "avg_unused_height_in": completed["unused_height_in"].mean(),
            "pct_short_in_tall": 100 * (
                (completed["sku_id"].map(sku_height) <= 72)
                & (slot_max_height[indices] > 72)
            ).mean(),
            "overflow_putaways": counters["cross_zone"],
            "unreserved_putaways": counters["unreserved"],
            "opening_overflow": counters["opening_overflow"],
            "failed_putaways": counters["failed_putaway"],
            "failed_opening": counters["failed_opening"],
            "missing_picks": counters["missing_picks"],
            "peak_occupied": counters["peak_occupied"],
        })
        print(f"  Done. Failed put-aways: {counters['failed_putaway']}")

    comparison = pd.DataFrame(results)
    baseline_hours = comparison.loc[0, "labor_hours"]
    comparison["labor_savings_vs_historical_hours"] = (
        baseline_hours - comparison["labor_hours"]
    )
    comparison["labor_savings_pct"] = (
        100 * comparison["labor_savings_vs_historical_hours"] / baseline_hours
    )
    comparison.to_csv("Output/Task4c/Task4c_Comparison.csv", index=False)
    print("\n", comparison.round(2).to_string(index=False))


if __name__ == "__main__":
     main()
