

"""
STACKBAY CASEWORK 3 - TASK 2
========================================


Sections:
    Task 2(a): Warehouse layout
    Task 2(b): Putaway and storage behavior
    Task 2(c): FIFO and pallet residence time
    Task 2(d): Forklift workload and Digital Twin template

Required:
    pip install pandas numpy matplotlib

Folder structure:
    StackBay/
        StackBay_Task2.py
        Task1b_SKU_Velocity.csv  (optional)
        Data/
            slots.csv
            docks.csv
            skus.csv
            clients.csv
            opening_stock.csv
            inbound.csv
            outbound.csv
            current_stock.csv

Run:
    python StackBay_Task2.py

Results are saved in:
    Task2_Results/

"""

# ============================================================
# STEP 0: IMPORT LIBRARIES AND DEFINE SETTINGS
# ============================================================

from pathlib import Path
from collections import Counter, defaultdict
import heapq
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Folder locations
DATA_FOLDER = Path("Data")
OUTPUT_FOLDER = Path("Task2_Results")
TASK1_VELOCITY_FILE = Path("Task1b_SKU_Velocity.csv")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Historical period
FIRST_DAY = pd.Timestamp("2025-07-07")
LAST_DAY = pd.Timestamp("2026-07-05")

ALL_DAYS = pd.date_range(FIRST_DAY, LAST_DAY)


# Forklift modeling assumptions
TRAVEL_SPEED_FT_PER_MIN = 264
LIFT_SPEED_FT_PER_MIN = 60

SLOT_HANDLING_MIN = 1.5
DOCK_HANDLING_MIN = 1.8

DAY_SHIFT_HOURS = 12
NIGHT_SHIFT_HOURS = 7.5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_table(table, filename):
    """Save a DataFrame as a CSV file."""
    table.to_csv(
        OUTPUT_FOLDER / filename,
        index=False
    )


def save_chart(filename):
    """Save the current matplotlib chart."""
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / filename,
        dpi=160
    )

    plt.close()


def require(condition, explanation):
    """Stop the program if a validation fails."""
    if not condition:
        raise ValueError(explanation)


# ============================================================
# STEP 1: LOAD ALL DATA
# ============================================================

print("Reading StackBay data...")

files = [
    "slots",
    "docks",
    "skus",
    "clients",
    "opening_stock",
    "inbound",
    "outbound",
    "current_stock"
]

data = {}

for filename in files:

    file_path = DATA_FOLDER / (filename + ".csv")

    data[filename] = pd.read_csv(file_path)


slots = data["slots"]
docks = data["docks"]
skus = data["skus"]
clients = data["clients"]

opening = data["opening_stock"]
inbound = data["inbound"]
outbound = data["outbound"]
closing = data["current_stock"]


print("All data files loaded successfully.")


# ============================================================
# STEP 2: USE TASK 1(b) SKU VELOCITY CLASSES
# ============================================================

print("Preparing SKU velocity classes...")


# Count annual outbound pallets for each SKU
annual_shipments = (
    outbound.groupby("sku_id")
    .size()
    .rename("pallets_yr")
)


# Merge annual shipments with SKU information
velocity = (
    skus[
        ["sku_id", "client_id", "pallet_height_in"]
    ]
    .merge(
        annual_shipments,
        left_on="sku_id",
        right_index=True,
        how="left"
    )
)


velocity["pallets_yr"] = (
    velocity["pallets_yr"]
    .fillna(0)
    .astype(int)
)


# Same classification as Task 1(b)
def classify(pallets_per_year):

    if pallets_per_year >= 260:
        return "A"

    elif pallets_per_year >= 52:
        return "B"

    elif pallets_per_year >= 13:
        return "C"

    else:
        return "D"


velocity["velocity_class"] = (
    velocity["pallets_yr"].apply(classify)
)


# Compare with teammate's Task 1(b) file if available
if TASK1_VELOCITY_FILE.exists():

    teammate = pd.read_csv(TASK1_VELOCITY_FILE)

    compare = velocity[
        ["sku_id", "velocity_class"]
    ].merge(
        teammate[
            ["sku_id", "velocity_class"]
        ],
        on="sku_id",
        how="outer",
        suffixes=("_calculated", "_task1"),
        validate="one_to_one"
    )

    require(
        compare.notna().all().all(),
        "Task 1(b) file contains missing or extra SKUs."
    )

    require(
        (
            compare["velocity_class_calculated"]
            == compare["velocity_class_task1"]
        ).all(),
        "Task 1(b) velocity classes do not match."
    )

    print("Velocity classes match Task 1(b).")

else:

    print(
        "Task 1(b) file not found. "
        "Using the same A/B/C/D thresholds."
    )


save_table(
    velocity,
    "Task2_Velocity_Classes.csv"
)


# ============================================================
# STEP 3: VALIDATE HISTORICAL INVENTORY
# ============================================================

print("Validating inventory records...")


# Opening stock pallets
opening_pallets = opening[
    ["pallet_id", "sku_id", "slot_id"]
]


# Inbound pallets
inbound_pallets = (
    inbound[
        ["pallet_id", "sku_id", "putaway_slot"]
    ]
    .rename(
        columns={"putaway_slot": "slot_id"}
    )
)


# Combine all pallets that entered or existed in the warehouse
origin = pd.concat(
    [opening_pallets, inbound_pallets],
    ignore_index=True
)


require(
    origin["pallet_id"].is_unique,
    "Duplicate pallet IDs found."
)


require(
    set(outbound["pallet_id"]).issubset(
        set(origin["pallet_id"])
    ),
    "Some outbound pallets have no inventory origin."
)


# Expected closing inventory
expected_closing_ids = (
    set(origin["pallet_id"])
    - set(outbound["pallet_id"])
)


require(
    expected_closing_ids == set(closing["pallet_id"]),
    "Inventory does not reconcile with current stock."
)


# Verify shipped SKU and slot
shipped_check = outbound[
    ["pallet_id", "sku_id", "picked_from_slot"]
].merge(
    origin,
    on="pallet_id",
    suffixes=("_shipped", "_original"),
    validate="one_to_one"
)


require(
    (
        shipped_check["sku_id_shipped"]
        == shipped_check["sku_id_original"]
    ).all(),
    "Shipped SKU does not match original SKU."
)


require(
    (
        shipped_check["picked_from_slot"]
        == shipped_check["slot_id"]
    ).all(),
    "Shipped pallet slot does not match original slot."
)


# Check historical dates
for frame, date_column in [
    (inbound, "receipt_date"),
    (outbound, "ship_date")
]:

    parsed_dates = pd.to_datetime(
        frame[date_column]
    )

    require(
        parsed_dates.between(
            FIRST_DAY,
            LAST_DAY
        ).all(),
        f"{date_column} contains dates outside the period."
    )


print("Historical inventory validation passed.")


summary = {}


# ============================================================
# TASK 2(a): WAREHOUSE LAYOUT
# ============================================================

print("\nTASK 2(a): Analyzing warehouse layout...")


# ------------------------------------------------------------
# 2(a).1: Analyze rack levels
# ------------------------------------------------------------

levels = (
    slots.groupby("level")
    .agg(
        slots=("slot_id", "size"),
        average_opening_in=(
            "opening_height_in",
            "mean"
        ),
        average_distance_ft=(
            "dist_dock_ft",
            "mean"
        )
    )
    .reset_index()
)


save_table(
    levels,
    "2a_Slots_By_Level.csv"
)


# ------------------------------------------------------------
# 2(a).2: Analyze aisles
# ------------------------------------------------------------

aisles = (
    slots.groupby("aisle")
    .agg(
        slots=("slot_id", "size"),
        bays=("bay", "nunique"),
        levels=("level", "nunique"),
        average_distance_ft=(
            "dist_dock_ft",
            "mean"
        ),
        tall_openings=(
            "opening_height_in",
            lambda heights: (heights >= 90).sum()
        )
    )
    .reset_index()
)


save_table(
    aisles,
    "2a_Aisles.csv"
)


# ------------------------------------------------------------
# 2(a).3: Count opening heights
# ------------------------------------------------------------

opening_heights = (
    slots.groupby("opening_height_in")
    .size()
    .reset_index(name="slots")
)


save_table(
    opening_heights,
    "2a_Opening_Heights.csv"
)


# ------------------------------------------------------------
# 2(a).4: Verify dock-to-slot distances
# ------------------------------------------------------------

require(
    np.allclose(docks["y_ft"], -40),
    "Dock coordinates do not match the assumed apron."
)


computed_distances = []


for slot in slots.itertuples(index=False):

    # Manhattan distance from each dock door
    distance_to_each_door = (
        abs(docks["x_ft"] - slot.aisle_x_ft)
        + 40
        + slot.y_ft
    )

    # Average across dock doors
    average_distance = distance_to_each_door.mean()

    computed_distances.append(
        average_distance
    )


distance_check = slots[
    ["slot_id", "dist_dock_ft"]
].copy()


distance_check["calculated_distance_ft"] = (
    computed_distances
)


distance_check["difference_ft"] = (
    distance_check["dist_dock_ft"]
    - distance_check["calculated_distance_ft"]
)


save_table(
    distance_check,
    "2a_Distance_Check.csv"
)


# ------------------------------------------------------------
# 2(a).5: Graph slots by level
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

plt.bar(
    levels["level"].astype(str),
    levels["slots"]
)

plt.xlabel("Rack level")
plt.ylabel("Number of pallet slots")
plt.title("Task 2(a): Slots by rack level")

save_chart(
    "2a_Slots_By_Level.png"
)


# ------------------------------------------------------------
# 2(a).6: Warehouse opening-height map
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    slots["x_ft"],
    slots["y_ft"],
    c=slots["opening_height_in"],
    s=4,
    cmap="viridis"
)

plt.colorbar(
    label="Opening height (inches)"
)

plt.xlabel("Warehouse x-coordinate (ft)")
plt.ylabel("Warehouse depth (ft)")

plt.title(
    "Task 2(a): Rack opening heights"
)

save_chart(
    "2a_Opening_Height_Map.png"
)


summary["2a"] = {
    "aisles": int(slots["aisle"].nunique()),
    "slots": int(len(slots)),
    "dock_doors": int(len(docks)),
    "max_distance_difference_ft": float(
        distance_check["difference_ft"].abs().max()
    )
}


print("Task 2(a) completed.")


# ============================================================
# TASK 2(b): ACTUAL PUTAWAY BEHAVIOR
# ============================================================

print("\nTASK 2(b): Analyzing putaway behavior...")


# ------------------------------------------------------------
# 2(b).1: Merge putaway pallets with slot information
# ------------------------------------------------------------

putaway = inbound.merge(
    slots[
        [
            "slot_id",
            "aisle",
            "level",
            "opening_height_in",
            "max_pallet_in",
            "dist_dock_ft"
        ]
    ],
    left_on="putaway_slot",
    right_on="slot_id",
    validate="many_to_one"
)


# Add SKU velocity and pallet height
putaway = putaway.merge(
    velocity[
        [
            "sku_id",
            "client_id",
            "velocity_class",
            "pallet_height_in",
            "pallets_yr"
        ]
    ],
    on="sku_id",
    validate="many_to_one"
)


# ------------------------------------------------------------
# 2(b).2: Calculate unused vertical space
# ------------------------------------------------------------

putaway["unused_height_in"] = (
    putaway["opening_height_in"]
    - putaway["pallet_height_in"]
)


putaway["extra_clearance_in"] = (
    putaway["max_pallet_in"]
    - putaway["pallet_height_in"]
)


require(
    (putaway["extra_clearance_in"] >= 0).all(),
    "Some pallets do not fit their assigned slots."
)


# ------------------------------------------------------------
# 2(b).3: Putaway by velocity class
# ------------------------------------------------------------

by_velocity = (
    putaway.groupby("velocity_class")
    .agg(
        putaways=("pallet_id", "size"),
        average_travel_ft=("travel_ft", "mean"),
        average_level=("level", "mean"),
        average_unused_height_in=(
            "unused_height_in",
            "mean"
        ),
        average_extra_clearance_in=(
            "extra_clearance_in",
            "mean"
        )
    )
    .reindex(["A", "B", "C", "D"])
    .reset_index()
)


save_table(
    by_velocity,
    "2b_Putaway_By_Velocity.csv"
)


# ------------------------------------------------------------
# 2(b).4: Putaway by client
# ------------------------------------------------------------

by_client = (
    putaway.groupby("client_id")
    .agg(
        putaways=("pallet_id", "size"),
        average_travel_ft=("travel_ft", "mean"),
        average_level=("level", "mean"),
        average_unused_height_in=(
            "unused_height_in",
            "mean"
        )
    )
    .reset_index()
)


save_table(
    by_client,
    "2b_Putaway_By_Client.csv"
)


# ------------------------------------------------------------
# 2(b).5: Putaway by rack level
# ------------------------------------------------------------

by_level = (
    putaway.groupby("level")
    .agg(
        putaways=("pallet_id", "size"),
        average_travel_ft=("travel_ft", "mean"),
        average_unused_height_in=(
            "unused_height_in",
            "mean"
        )
    )
    .reset_index()
)


save_table(
    by_level,
    "2b_Putaway_By_Level.csv"
)


# ------------------------------------------------------------
# 2(b).6: Client distribution across aisles
# ------------------------------------------------------------

client_aisle = pd.crosstab(
    putaway["client_id"],
    putaway["aisle"]
)


client_aisle.to_csv(
    OUTPUT_FOLDER / "2b_Client_Aisle_Counts.csv"
)


client_aisle_shares = client_aisle.div(
    client_aisle.sum(axis=1),
    axis=0
)


client_aisle_shares.to_csv(
    OUTPUT_FOLDER / "2b_Client_Aisle_Shares.csv"
)


# ------------------------------------------------------------
# 2(b).7: Graph average travel by velocity
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

plt.bar(
    by_velocity["velocity_class"],
    by_velocity["average_travel_ft"]
)

plt.xlabel("SKU velocity class")
plt.ylabel("Average putaway travel (ft)")

plt.title(
    "Task 2(b): Putaway travel by velocity"
)

save_chart(
    "2b_Travel_By_Velocity.png"
)


# ------------------------------------------------------------
# 2(b).8: Graph travel by client
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))

plt.bar(
    by_client["client_id"],
    by_client["average_travel_ft"]
)

plt.xlabel("Client")
plt.ylabel("Average putaway travel (ft)")

plt.title(
    "Task 2(b): Putaway travel by client"
)

save_chart(
    "2b_Travel_By_Client.png"
)


# ------------------------------------------------------------
# 2(b).9: Graph excess clearance
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

plt.hist(
    putaway["extra_clearance_in"],
    bins=35
)

plt.xlabel(
    "Unused height beyond required clearance (in)"
)

plt.ylabel("Number of putaways")

plt.title(
    "Task 2(b): Excess vertical clearance"
)

save_chart(
    "2b_Excess_Height.png"
)


# ------------------------------------------------------------
# 2(b).10: Reconstruct occupied slots through the year
# ------------------------------------------------------------

# Initially occupied slots
occupied_slots = set(
    opening["slot_id"]
)


# Slot lookup table
slot_lookup = slots.set_index(
    "slot_id"
)


# SKU height lookup
sku_height = (
    skus.set_index("sku_id")["pallet_height_in"]
    .to_dict()
)


# Group shipments and receipts by date
shipments_by_day = dict(
    tuple(outbound.groupby("ship_date"))
)

receipts_by_day = dict(
    tuple(inbound.groupby("receipt_date"))
)


comparisons = []
occupancy_history = []

putaway_number = 0


# ------------------------------------------------------------
# Process every historical day
# ------------------------------------------------------------

for day in ALL_DAYS:

    date = day.strftime("%Y-%m-%d")

    # DAY SHIFT: REMOVE SHIPPED PALLETS
    if date in shipments_by_day:

        todays_shipments = shipments_by_day[date]

        for pallet in todays_shipments.itertuples(index=False):

            require(
                pallet.picked_from_slot in occupied_slots,
                f"Picking from empty slot on {date}: "
                f"{pallet.pallet_id}"
            )

            occupied_slots.remove(
                pallet.picked_from_slot
            )

    # NIGHT SHIFT: ADD RECEIVED PALLETS
    if date in receipts_by_day:

        todays_receipts = (
            receipts_by_day[date]
            .sort_values(
                ["putaway_at", "pallet_id"]
            )
        )

        for pallet in todays_receipts.itertuples(index=False):

            chosen_slot = pallet.putaway_slot

            require(
                chosen_slot not in occupied_slots,
                f"Putting into occupied slot on {date}: "
                f"{chosen_slot}"
            )

            # Sample every 50th putaway
            if putaway_number % 50 == 0:

                # Identify currently empty slots
                free_slots = slot_lookup.loc[
                    ~slot_lookup.index.isin(
                        occupied_slots
                    )
                ]

                # Keep slots tall enough for the pallet
                feasible = free_slots.loc[
                    free_slots["max_pallet_in"]
                    >= sku_height[pallet.sku_id]
                ]

                require(
                    chosen_slot in feasible.index,
                    f"Chosen slot is not feasible: "
                    f"{chosen_slot}"
                )

                chosen_distance = float(
                    slot_lookup.loc[
                        chosen_slot,
                        "dist_dock_ft"
                    ]
                )

                closest_distance = float(
                    feasible["dist_dock_ft"].min()
                )

                comparisons.append({
                    "pallet_id": pallet.pallet_id,
                    "receipt_date": date,
                    "sku_id": pallet.sku_id,
                    "chosen_slot": chosen_slot,
                    "number_of_feasible_free_slots": len(feasible),
                    "chosen_avg_dock_distance_ft": chosen_distance,
                    "nearest_feasible_avg_dock_distance_ft": closest_distance,
                    "extra_distance_ft": (
                        chosen_distance - closest_distance
                    )
                })

            # Mark chosen slot as occupied
            occupied_slots.add(
                chosen_slot
            )

            putaway_number += 1

    # Record occupancy at the end of the day
    occupancy_history.append({
        "date": date,
        "occupied_slots": len(occupied_slots)
    })


# Verify final occupied slots
require(
    occupied_slots == set(closing["slot_id"]),
    "Final occupied slots do not match closing stock."
)


save_table(
    pd.DataFrame(occupancy_history),
    "2b_Daily_Occupied_Slots.csv"
)


comparison_table = pd.DataFrame(
    comparisons
)


save_table(
    comparison_table,
    "2b_Sampled_Free_Slot_Comparison.csv"
)


summary["2b"] = {
    "putaways": int(len(putaway)),
    "average_unused_height_in": float(
        putaway["unused_height_in"].mean()
    ),
    "average_extra_clearance_in": float(
        putaway["extra_clearance_in"].mean()
    ),
    "sampled_putaways": int(
        len(comparison_table)
    ),
    "average_extra_distance_vs_nearest_ft": float(
        comparison_table["extra_distance_ft"].mean()
    ),
    "final_occupancy_matches_current_stock": True
}


print("Task 2(b) completed.")


# ============================================================
# TASK 2(c): FIFO AND PALLET RESIDENCE TIME
# ============================================================

print("\nTASK 2(c): Analyzing FIFO and residence time...")


# ------------------------------------------------------------
# 2(c).1: Build initial inventory by SKU
# ------------------------------------------------------------

available_by_sku = defaultdict(list)


for pallet in opening.itertuples(index=False):

    heapq.heappush(
        available_by_sku[pallet.sku_id],
        (
            pallet.received_date,
            pallet.pallet_id
        )
    )


fifo_rows = []


# ------------------------------------------------------------
# 2(c).2: Process shipments and receipts chronologically
# ------------------------------------------------------------

for day in ALL_DAYS:

    date = day.strftime("%Y-%m-%d")

    # DAY SHIFT: CHECK OUTBOUND FIFO
    if date in shipments_by_day:

        todays_shipments = (
            shipments_by_day[date]
            .sort_values(
                ["picked_at", "pallet_id"]
            )
        )

        for pallet in todays_shipments.itertuples(index=False):

            available = available_by_sku[
                pallet.sku_id
            ]

            require(
                len(available) > 0,
                f"No available stock for "
                f"{pallet.sku_id} on {date}"
            )

            # Oldest receipt date currently available
            oldest_date = available[0][0]

            # Compare actual picked pallet's receipt date
            compliant = (
                pallet.received_date == oldest_date
            )

            fifo_rows.append({
                "pallet_id": pallet.pallet_id,
                "sku_id": pallet.sku_id,
                "client_id": pallet.client_id,
                "ship_date": date,
                "received_date": pallet.received_date,
                "oldest_available_date": oldest_date,
                "fifo_compliant": compliant,
                "days_in_stock": pallet.days_in_stock
            })

            # Remove the ACTUAL pallet that was shipped
            actual_pallet = (
                pallet.received_date,
                pallet.pallet_id
            )

            require(
                actual_pallet in available,
                f"Actual pallet not available: "
                f"{pallet.pallet_id}"
            )

            available.remove(
                actual_pallet
            )

            heapq.heapify(
                available
            )

    # NIGHT SHIFT: ADD INBOUND PALLETS
    if date in receipts_by_day:

        for pallet in receipts_by_day[date].itertuples(index=False):

            heapq.heappush(
                available_by_sku[pallet.sku_id],
                (
                    pallet.receipt_date,
                    pallet.pallet_id
                )
            )


# ------------------------------------------------------------
# 2(c).3: Verify remaining inventory
# ------------------------------------------------------------

remaining_pallets = {
    pallet_id
    for stock in available_by_sku.values()
    for _, pallet_id in stock
}


require(
    remaining_pallets == set(closing["pallet_id"]),
    "FIFO reconstruction does not match closing inventory."
)


# ------------------------------------------------------------
# 2(c).4: Build FIFO audit table
# ------------------------------------------------------------

fifo = pd.DataFrame(
    fifo_rows
)


fifo = fifo.merge(
    velocity[
        ["sku_id", "velocity_class"]
    ],
    on="sku_id",
    validate="many_to_one"
)


# Calculate residence time from dates
calculated_age = (
    pd.to_datetime(fifo["ship_date"])
    - pd.to_datetime(fifo["received_date"])
).dt.days


require(
    (
        calculated_age == fifo["days_in_stock"]
    ).all(),
    "Recorded residence time does not match dates."
)


save_table(
    fifo,
    "2c_FIFO_Pallet_Audit.csv"
)


# Save strict FIFO exceptions
save_table(
    fifo.loc[
        ~fifo["fifo_compliant"]
    ],
    "2c_FIFO_Exceptions.csv"
)


# Save longest-stored pallets
save_table(
    fifo.nlargest(
        25,
        "days_in_stock"
    ),
    "2c_Longest_Stored_Pallets.csv"
)


# ------------------------------------------------------------
# 2(c).5: Daily-batch FIFO check
# ------------------------------------------------------------

# This check ignores the order of picks within one day.
# It asks whether the correct receipt-date batches
# were shipped during that day.

stock_by_sku_and_date = defaultdict(
    Counter
)


for pallet in opening.itertuples(index=False):

    stock_by_sku_and_date[
        pallet.sku_id
    ][pallet.received_date] += 1


batch_rows = []


for day in ALL_DAYS:

    date = day.strftime("%Y-%m-%d")

    # Check shipments by SKU
    if date in shipments_by_day:

        todays_shipments = shipments_by_day[date]

        for sku_id, group in todays_shipments.groupby("sku_id"):

            available = stock_by_sku_and_date[
                sku_id
            ]

            # Actual receipt-date batches shipped
            actual_dates = Counter(
                group["received_date"]
            )

            number_to_ship = len(group)

            # Ideal FIFO receipt-date batches
            ideal_dates = Counter()

            for received_date in sorted(available):

                quantity = min(
                    number_to_ship,
                    available[received_date]
                )

                ideal_dates[received_date] = quantity

                number_to_ship -= quantity

                if number_to_ship == 0:
                    break

            require(
                number_to_ship == 0,
                f"Insufficient stock for "
                f"{sku_id} on {date}"
            )

            # Count pallets shipped from nonideal batches
            exceptions = sum(
                (
                    actual_dates - ideal_dates
                ).values()
            )

            batch_rows.append({
                "ship_date": date,
                "sku_id": sku_id,
                "shipments": len(group),
                "daily_batch_fifo_exceptions": exceptions
            })

            # Remove actual shipped batches
            for received_date, quantity in actual_dates.items():

                require(
                    available[received_date] >= quantity,
                    f"Insufficient stock for "
                    f"{sku_id} on {date}"
                )

                available[received_date] -= quantity

                if available[received_date] == 0:

                    del available[received_date]

    # Add night receipts
    if date in receipts_by_day:

        for pallet in receipts_by_day[date].itertuples(index=False):

            stock_by_sku_and_date[
                pallet.sku_id
            ][pallet.receipt_date] += 1


batch = pd.DataFrame(
    batch_rows
)


save_table(
    batch,
    "2c_FIFO_Daily_Batch_Audit.csv"
)


# ------------------------------------------------------------
# 2(c).6: Residence time by velocity
# ------------------------------------------------------------

by_velocity_age = (
    fifo.groupby("velocity_class")
    .agg(
        shipped_pallets=("pallet_id", "size"),
        fifo_compliance_pct=(
            "fifo_compliant",
            lambda values: 100 * values.mean()
        ),
        mean_days=("days_in_stock", "mean"),
        median_days=("days_in_stock", "median"),
        p95_days=(
            "days_in_stock",
            lambda values: values.quantile(0.95)
        ),
        max_days=("days_in_stock", "max")
    )
    .reindex(["A", "B", "C", "D"])
    .reset_index()
)


save_table(
    by_velocity_age,
    "2c_Residence_By_Velocity.csv"
)


# ------------------------------------------------------------
# 2(c).7: Residence time by client
# ------------------------------------------------------------

by_client_age = (
    fifo.groupby("client_id")
    .agg(
        shipped_pallets=("pallet_id", "size"),
        fifo_compliance_pct=(
            "fifo_compliant",
            lambda values: 100 * values.mean()
        ),
        mean_days=("days_in_stock", "mean"),
        median_days=("days_in_stock", "median"),
        p95_days=(
            "days_in_stock",
            lambda values: values.quantile(0.95)
        ),
        max_days=("days_in_stock", "max")
    )
    .reset_index()
)


save_table(
    by_client_age,
    "2c_Residence_By_Client.csv"
)


# ------------------------------------------------------------
# 2(c).8: Residence-time histogram
# ------------------------------------------------------------

plt.figure(figsize=(9, 4))

plt.hist(
    fifo["days_in_stock"],
    bins=55
)

plt.xlabel("Pallet residence time (days)")
plt.ylabel("Number of shipped pallets")

plt.title(
    "Task 2(c): Pallet residence time"
)

save_chart(
    "2c_Residence_Histogram.png"
)


# ------------------------------------------------------------
# 2(c).9: Residence time by velocity graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

plt.bar(
    by_velocity_age["velocity_class"],
    by_velocity_age["mean_days"]
)

plt.xlabel("SKU velocity class")
plt.ylabel("Mean residence time (days)")

plt.title(
    "Task 2(c): Residence time by velocity"
)

save_chart(
    "2c_Residence_By_Velocity.png"
)


summary["2c"] = {
    "shipped_pallets_audited": int(
        len(fifo)
    ),
    "strict_fifo_exceptions": int(
        (~fifo["fifo_compliant"]).sum()
    ),
    "daily_batch_fifo_exceptions": int(
        batch["daily_batch_fifo_exceptions"].sum()
    ),
    "mean_residence_days": float(
        fifo["days_in_stock"].mean()
    ),
    "maximum_residence_days": int(
        fifo["days_in_stock"].max()
    )
}


print("Task 2(c) completed.")


# ============================================================
# TASK 2(d): FORKLIFT WORKLOAD
# ============================================================

print("\nTASK 2(d): Calculating forklift workload...")


# ------------------------------------------------------------
# 2(d).1: Function to calculate shift workload
# ------------------------------------------------------------

def calculate_shift_workload(
    pallets,
    date_column,
    shift_name
):

    movements = pallets.copy()

    movements["date"] = pd.to_datetime(
        movements[date_column]
    )

    # Round-trip horizontal travel time
    movements["travel_minutes"] = (
        2 * movements["travel_ft"]
        / TRAVEL_SPEED_FT_PER_MIN
    )

    # Round-trip lifting time
    movements["lifting_minutes"] = (
        2 * movements["lift_ft"]
        / LIFT_SPEED_FT_PER_MIN
    )

    # Rack handling time
    movements["handling_minutes"] = (
        SLOT_HANDLING_MIN
    )

    # Total rack movement time
    movements["rack_move_minutes"] = (
        movements["travel_minutes"]
        + movements["lifting_minutes"]
        + movements["handling_minutes"]
    )

    # Dock movement time
    movements["dock_move_minutes"] = (
        DOCK_HANDLING_MIN
    )

    # Total movement time per pallet
    movements["total_minutes"] = (
        movements["rack_move_minutes"]
        + movements["dock_move_minutes"]
    )

    # Summarize daily movements
    daily = (
        movements.groupby("date")
        .agg(
            pallets=("pallet_id", "size"),
            rack_minutes=("rack_move_minutes", "sum"),
            dock_minutes=("dock_move_minutes", "sum"),
            total_minutes=("total_minutes", "sum")
        )
        .reset_index()
    )

    # Convert minutes into hours
    daily["rack_hours"] = (
        daily["rack_minutes"] / 60
    )

    daily["dock_hours"] = (
        daily["dock_minutes"] / 60
    )

    daily["total_hours"] = (
        daily["total_minutes"] / 60
    )

    daily["shift"] = shift_name

    return daily[
        [
            "date",
            "shift",
            "pallets",
            "rack_hours",
            "dock_hours",
            "total_hours"
        ]
    ]


# ------------------------------------------------------------
# 2(d).2: Calculate day and night workload
# ------------------------------------------------------------

# Outbound pallets handled during day shift
day_work = calculate_shift_workload(
    outbound,
    "ship_date",
    "day"
)


# Inbound pallets handled during night shift
night_work = calculate_shift_workload(
    inbound,
    "receipt_date",
    "night"
)


# ------------------------------------------------------------
# 2(d).3: Include dates with zero movements
# ------------------------------------------------------------

calendar = pd.DataFrame({
    "date": ALL_DAYS
})


day_work = calendar.merge(
    day_work,
    on="date",
    how="left"
)


night_work = calendar.merge(
    night_work,
    on="date",
    how="left"
)


day_work["shift"] = "day"
night_work["shift"] = "night"


workload = pd.concat(
    [day_work, night_work],
    ignore_index=True
)


for column in [
    "pallets",
    "rack_hours",
    "dock_hours",
    "total_hours"
]:

    workload[column] = (
        workload[column].fillna(0)
    )


# ------------------------------------------------------------
# 2(d).4: Estimate ideal forklift equivalents
# ------------------------------------------------------------

workload["shift_hours"] = np.where(
    workload["shift"] == "day",
    DAY_SHIFT_HOURS,
    NIGHT_SHIFT_HOURS
)


# Theoretical forklifts assuming 100% utilization
workload["ideal_forklift_equivalents"] = (
    workload["total_hours"]
    / workload["shift_hours"]
)


# Round up to whole forklifts
workload["ideal_forklifts_rounded_up"] = (
    np.ceil(
        workload["ideal_forklift_equivalents"]
    )
    .astype(int)
)


save_table(
    workload,
    "2d_Daily_Forklift_Workload.csv"
)


# ------------------------------------------------------------
# 2(d).5: Summarize day and night workload
# ------------------------------------------------------------

shift_summary = (
    workload.groupby("shift")
    .agg(
        annual_pallets=("pallets", "sum"),
        annual_rack_hours=("rack_hours", "sum"),
        annual_dock_hours=("dock_hours", "sum"),
        annual_total_hours=("total_hours", "sum"),
        average_daily_hours=("total_hours", "mean"),
        peak_daily_hours=("total_hours", "max"),
        p95_daily_hours=(
            "total_hours",
            lambda values: values.quantile(0.95)
        )
    )
    .reset_index()
)


save_table(
    shift_summary,
    "2d_Shift_Summary.csv"
)


# Identify busiest shifts
peak_shifts = workload.nlargest(
    30,
    "total_hours"
)


save_table(
    peak_shifts,
    "2d_Peak_Shifts.csv"
)


# ------------------------------------------------------------
# 2(d).6: Graph daily forklift workload
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))


for shift in ["day", "night"]:

    series = workload[
        workload["shift"] == shift
    ].sort_values("date")

    plt.plot(
        series["date"],
        series["total_hours"],
        label=shift,
        linewidth=0.8
    )


plt.xlabel("Historical date")
plt.ylabel("Estimated forklift work-hours")

plt.title(
    "Task 2(d): Forklift workload by shift"
)

plt.legend()


save_chart(
    "2d_Daily_Forklift_Hours.png"
)


# ------------------------------------------------------------
# 2(d).7: Prepare Digital Twin experiment template
# ------------------------------------------------------------

experiment_columns = [
    "historical_date",
    "forklifts",
    "outbound_trucks_late",
    "inbound_trucks_late",
    "average_forklift_wait_min",
    "night_finished_by_midnight",
    "night_end_time",
    "truck_wait_count",
    "mean_truck_delay_min",
    "max_truck_delay_min",
    "queue_observations",
    "bottleneck_explanation",
    "notes"
]


experiment_rows = []


# Initial nine experiments
experiment_dates = [
    "2026-03-04",
    "2026-06-08",
    "2025-07-08"
]


forklift_options = [
    4,
    6,
    8
]


for date in experiment_dates:

    for forklifts in forklift_options:

        row = {
            column: ""
            for column in experiment_columns
        }

        row["historical_date"] = date

        row["forklifts"] = forklifts

        experiment_rows.append(
            row
        )


experiment_template = pd.DataFrame(
    experiment_rows
)


save_table(
    experiment_template,
    "2d_Digital_Twin_Experiments_TO_FILL.csv"
)


# ------------------------------------------------------------
# 2(d).8: Save forklift workload summary
# ------------------------------------------------------------

summary["2d"] = {
    "annual_rack_movement_hours": float(
        workload["rack_hours"].sum()
    ),
    "annual_dock_movement_hours": float(
        workload["dock_hours"].sum()
    ),
    "annual_total_movement_hours": float(
        workload["total_hours"].sum()
    ),
    "peak_day_shift_hours": float(
        day_work["total_hours"].max()
    ),
    "peak_night_shift_hours": float(
        night_work["total_hours"].max()
    ),
    "digital_twin_experiments": (
        "Not run; fill template manually"
    )
}


print("Task 2(d) completed.")


# ============================================================
# FINAL STEP: SAVE SUMMARY AND FINISH
# ============================================================

summary_file = (
    OUTPUT_FOLDER / "Task2_Summary.json"
)


summary_file.write_text(
    json.dumps(
        summary,
        indent=2
    )
)


print("\n========================================")
print("ALL FOUR TASK 2 SECTIONS COMPLETED")
print("========================================")


print(
    json.dumps(
        summary,
        indent=2
    )
)


print(
    "\nResults saved in:",
    OUTPUT_FOLDER.resolve()
)


print(
    "\nRemaining work:"
)

print(
    "1. Run the Digital Twin experiments manually."
)

print(
    "2. Record actual simulator results."
)

print(
    "3. Inspect the facility map."
)

print(
    "4. Write the putaway-rule explanation."
)

print(
    "5. Identify three evidence-based operational concerns."
)
