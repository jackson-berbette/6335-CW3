"""
Task 5a - Express the retained storage policies as evaluator design files
=========================================================================
Writes one folder per design in Output/Evaluator_Inputs/, each holding
slot_zones.csv and sku_zones.csv for the StackBay Storage Design Evaluator,
plus a manifest describing every design and how it approximates its policy.

Inputs : Data/slots.csv, Data/skus.csv
         Output/CW3P1T1_VelocityStock_WembyFanClub.csv   (Task 1)
         Output/Task4b_Slots_By_Zone_Height.csv          (Task 4b zone sizes)
Outputs: Output/Evaluator_Inputs/<design>/slot_zones.csv, sku_zones.csv
         Output/Evaluator_Inputs/P1*/P3*: allocation_shortfalls.csv
         Output/Evaluator_Inputs/Task5a_Design_Manifest.csv

Designs
  P1   Product-Dedicated-COL (mandatory)       each SKU owns slots for its average stock; no fallback
  P1b  Product-Dedicated-COL + overflow        as P1, leftover slots form one shared OVERFLOW zone
  P2   Product-Class-COL                       A / B / C zones nearest first, sized at class peak (4b)
  P3   Product-Dedicated-COL, tall protected   as P1, tall pallets only in tall openings and vice versa
  P4   Product-Shared-COL                      one zone, closest open location
  P5   Duration-of-Stay-Target                 4 stay targets nearest first; zone_1 = target, zone_2 = NEAREST
  P5b  Duration-of-Stay zones (variant)        same zones, neighbouring stay zones as fallback

Slots kept out of every zone (written with a BLANK zone, as the evaluator guide asks)
  UNUSABLE    slots too low for any pallet (kept out of every zone, as in Task 4b)
  UNASSIGNED  usable slots that no SKU lists (P1, P3): reached only by the evaluator's
              last-resort fall-back on today's habit
  These two names are used inside this script and in the manifest only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "Output" / "Evaluator_Inputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

TALL_THRESHOLD_IN = 72                      # tall pallet / tall opening: over 72 in (as in 4a/4b)
DAYS = 364                                  # days of history used for expected stay (as in 4b)
STAY_ZONES = ["T1 (up to 14 d)", "T2 (14-30 d)", "T3 (30-60 d)", "T4 (over 60 d)"]
NEAREST = "NEAREST"
UNUSABLE = "UNUSABLE"
UNASSIGNED = "UNASSIGNED"
OVERFLOW = "OVERFLOW"


# ---------------------------------------------------------------------------
# 1. Inputs (read once, shared by every design)
# ---------------------------------------------------------------------------
# Slots ranked nearest the dock first, then lowest level: the same order as Task 4b.
slots = pd.read_csv(BASE / "Data/slots.csv").sort_values(
    ["dist_dock_ft", "level", "slot_id"]
).reset_index(drop=True)
skus = pd.read_csv(BASE / "Data/skus.csv")
velocity = pd.read_csv(BASE / "Output/CW3P1T1_VelocityStock_WembyFanClub.csv")
zone_sizes = pd.read_csv(BASE / "Output/Task4b_Slots_By_Zone_Height.csv")

sku_info = velocity[["sku_id", "pallets_shipped", "avg_stock_pallets", "peak_stock_pallets"]].merge(
    skus[["sku_id", "pallet_height_in"]], on="sku_id", validate="one_to_one"
)
sku_info = sku_info.sort_values(
    ["pallets_shipped", "sku_id"], ascending=[False, True]
).reset_index(drop=True)

# Phase 2 reporting classes (same definition as Task 4a/4b):
# A = first 50% of pallets shipped, B = next 30%, C = the rest.
share_before = sku_info.pallets_shipped.cumsum().shift(fill_value=0) / sku_info.pallets_shipped.sum()
sku_info["class"] = np.select([share_before < 0.5, share_before < 0.8], ["A", "B"], default="C")

# Expected stay of a pallet (Little's law, as in 4b): average stock / pallets shipped per day.
# SKUs never shipped get the longest stay target.
sku_info["expected_stay"] = sku_info.avg_stock_pallets / (
    sku_info.pallets_shipped / DAYS
).replace(0, np.nan)
sku_info["stay"] = pd.cut(
    sku_info.expected_stay.fillna(np.inf), [0, 14, 30, 60, np.inf], labels=STAY_ZONES
).astype(str)

# Slots too low for the shortest pallet can never be used. Task 4b sized its zones on
# the usable slots only, so the zones here are cut from the usable slots only as well.
usable = (slots.max_pallet_in >= sku_info.pallet_height_in.min()).to_numpy()
usable_idx = slots.index[usable]
sku_info_class = sku_info.set_index("sku_id")["class"].to_dict()
print(f"Slots: {len(slots):,}; usable: {usable.sum():,}; too low for any pallet: {(~usable).sum():,}")

manifest = []


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def sku_preferences(preferences):
    """One row per SKU with up to three zones in order of preference ('' = none)."""
    result = sku_info[["sku_id"]].copy()
    for k in range(3):
        result[f"zone_{k + 1}"] = [p[k] if k < len(p) else "" for p in preferences]
    return result


def add_fill_order(assignments):
    """Within each zone, fill nearest the dock first (slots are already in that order)."""
    assignments["fill_order"] = assignments.groupby("zone").cumcount() + 1
    return assignments


def check_design(slot_assignments, sku_assignments):
    """Stop before writing a file the evaluator would reject or misread."""
    assert len(slot_assignments) == len(slots) and slot_assignments.slot_id.is_unique
    assert len(sku_assignments) == len(sku_info) and sku_assignments.sku_id.is_unique
    assert (slot_assignments.zone != "").all(), "every slot needs a zone label"
    zone_cols = ["zone_1", "zone_2", "zone_3"]
    assert (sku_assignments.zone_1 != "").all(), "every SKU needs a first-choice zone"
    for row in sku_assignments[zone_cols].itertuples(index=False):
        filled = [z for z in row if z != ""]
        assert filled == list(row[:len(filled)]), "zones must be listed without gaps"
        if NEAREST in filled:
            assert filled[-1] == NEAREST and filled.index(NEAREST) > 0, \
                "NEAREST must come last and after at least one zone"
    named = set(sku_assignments[zone_cols].stack()) - {"", NEAREST}
    missing = named - set(slot_assignments.zone)
    assert not missing, f"SKUs name zones that hold no slots: {sorted(missing)[:5]}"
    unusable_named = named & {UNUSABLE, UNASSIGNED}
    assert not unusable_named, "no SKU may list UNUSABLE or UNASSIGNED"


def save_design(folder_name, policy, slot_assignments, sku_assignments, approximation, shortfalls=None):
    check_design(slot_assignments, sku_assignments)
    folder = OUTPUT / folder_name
    folder.mkdir(exist_ok=True)
    # The evaluator guide: "leave zone blank to keep a slot out of every zone".
    # UNUSABLE / UNASSIGNED are internal labels only; they are written as blanks.
    to_write = slot_assignments.copy()
    outside = to_write.zone.isin([UNUSABLE, UNASSIGNED])
    to_write["fill_order"] = to_write["fill_order"].astype("Int64")
    to_write.loc[outside, "fill_order"] = pd.NA
    to_write.loc[outside, "zone"] = ""
    to_write.to_csv(folder / "slot_zones.csv", index=False)
    sku_assignments.to_csv(folder / "sku_zones.csv", index=False)
    if shortfalls is not None:
        shortfalls.to_csv(folder / "allocation_shortfalls.csv", index=False)

    design_zones = slot_assignments.zone[~slot_assignments.zone.isin([UNUSABLE, UNASSIGNED])]
    manifest.append({
        "design": folder_name, "policy": policy,
        "zones": design_zones.nunique(),
        "slots_in_design_zones": len(design_zones),
        "slots_unassigned": int((slot_assignments.zone == UNASSIGNED).sum()),
        "slots_unusable": int((slot_assignments.zone == UNUSABLE).sum()),
        "skus_with_fallback": int((sku_assignments.zone_2 != "").sum()),
        "skus_with_nearest": int((sku_assignments[["zone_2", "zone_3"]] == NEAREST).any(axis=1).sum()),
        "approximation": approximation,
    })
    sizes = design_zones.value_counts()
    shown = sizes.to_dict() if len(sizes) <= 6 else f"{len(sizes)} zones, {sizes.min()}-{sizes.max()} slots each"
    print(f"{folder_name}: {len(design_zones):,} slots in design zones -> {shown}")


# ---------------------------------------------------------------------------
# 3. Zoned designs (shared, class, duration of stay)
# ---------------------------------------------------------------------------
def zoned_slots(policy_name, zones):
    """Cut consecutive blocks of usable slots, nearest first, at the Task 4b zone sizes
    (heights mixed). The last zone also takes every usable slot left over."""
    assignments = slots[["slot_id"]].copy()
    assignments["zone"] = UNUSABLE
    if len(zones) == 1:
        assignments.loc[usable_idx, "zone"] = zones[0]
        return add_fill_order(assignments)

    sizes = zone_sizes.loc[
        (zone_sizes.policy == policy_name)
        & (zone_sizes.tall_protected.astype(str).str.lower() == "false")
    ].set_index("zone")["zone_slots_nearest_first"]
    counts = [int(sizes[z]) for z in zones[:-1]]
    assert sum(counts) <= len(usable_idx), f"{policy_name}: Task 4b zones exceed the usable slots"
    cursor = 0
    for zone, count in zip(zones[:-1], counts):
        assignments.loc[usable_idx[cursor:cursor + count], "zone"] = zone
        cursor += count
    assignments.loc[usable_idx[cursor:], "zone"] = zones[-1]
    return add_fill_order(assignments)


# ---------------------------------------------------------------------------
# 4. Dedicated designs
# ---------------------------------------------------------------------------
def dedicated_slots(protect_tall, overflow):
    """Each SKU owns the nearest free slots it fits, for its AVERAGE stock rounded up.
    Priority: fewest reserved slots per move first (average stock / (2 x pallets shipped)),
    i.e. the cube-per-order index of the quantity actually reserved.
    Task 4b ranked by PEAK stock per move because it sized at peak; peak sizing needs
    9,747 slots > 8,365, so here the reservation and the ranking both use average stock."""
    assignments = slots[["slot_id"]].copy()
    assignments["zone"] = np.where(usable, UNASSIGNED, UNUSABLE)
    free = usable.copy()
    max_heights = slots.max_pallet_in.to_numpy()

    priority = sku_info.assign(
        reserved=lambda d: np.maximum(1, np.ceil(d.avg_stock_pallets)).astype(int),
        slots_per_move=lambda d: d.reserved / (2 * d.pallets_shipped.replace(0, np.nan)),
    ).sort_values(["slots_per_move", "sku_id"], na_position="last")

    rows = []
    for sku in priority.itertuples(index=False):
        eligible = free & (max_heights >= sku.pallet_height_in)
        if protect_tall:
            tall_pallet = float(sku.pallet_height_in) > TALL_THRESHOLD_IN
            eligible &= (max_heights > TALL_THRESHOLD_IN) if tall_pallet else (max_heights <= TALL_THRESHOLD_IN)
        chosen = np.flatnonzero(eligible)[:sku.reserved]
        relaxed = 0
        if protect_tall and len(chosen) < sku.reserved:
            # Its own height group is full: top up from any free slot it fits, and record it.
            extra = np.flatnonzero(free & (max_heights >= sku.pallet_height_in) & ~np.isin(np.arange(len(free)), chosen))
            extra = extra[:sku.reserved - len(chosen)]
            relaxed = len(extra)
            chosen = np.concatenate([chosen, extra])
        free[chosen] = False
        assignments.loc[chosen, "zone"] = sku.sku_id
        rows.append({"sku_id": sku.sku_id, "class": sku_info_class[sku.sku_id],
                     "required_slots": sku.reserved, "allocated_slots": len(chosen),
                     "shortfall": sku.reserved - len(chosen), "slots_outside_height_group": relaxed})
    shortfalls = pd.DataFrame(rows)
    if protect_tall:
        print(f"  tall protection relaxed for {(shortfalls.slots_outside_height_group > 0).sum()} SKUs "
              f"({shortfalls.slots_outside_height_group.sum():,} slots outside their height group)")

    # Every SKU must own at least one slot, otherwise its zone would not exist.
    no_slot = shortfalls.loc[shortfalls.allocated_slots == 0, "sku_id"]
    if not no_slot.empty:
        raise ValueError(f"{len(no_slot)} SKUs received no dedicated slot, e.g. {list(no_slot[:5])}")

    if overflow:
        assignments.loc[assignments.zone == UNASSIGNED, "zone"] = OVERFLOW
        preferences = [[s, OVERFLOW] for s in sku_info.sku_id]
    else:
        preferences = [[s] for s in sku_info.sku_id]   # pure dedicated: no alternative zone

    print(f"  reserved {shortfalls.required_slots.sum():,} slots, allocated {shortfalls.allocated_slots.sum():,}, "
          f"short {shortfalls.shortfall.sum():,} (by class: "
          f"{shortfalls.groupby('class').shortfall.sum().to_dict()})")
    return add_fill_order(assignments), sku_preferences(preferences), shortfalls


DEDICATED_NOTE = ("Reserving each SKU's peak needs 9,747 slots > 8,365, so each SKU owns its average "
                  "stock rounded up (min 1), nearest slots first, ranked by reserved slots per move. ")

# ---------------------------------------------------------------------------
# 5. Build every design
# ---------------------------------------------------------------------------
# P1: Product-Dedicated-COL (mandatory), heights mixed, no fallback
a, p, s = dedicated_slots(protect_tall=False, overflow=False)
save_design("P1_Dedicated_COL", "Product-Dedicated-COL", a, p,
            DEDICATED_NOTE + "Pallets beyond the reservation fall back on today's habit (UNASSIGNED slots).", s)

# P1b: as P1, the slots left over form one shared OVERFLOW zone (closest open location)
a, p, s = dedicated_slots(protect_tall=False, overflow=True)
save_design("P1b_Dedicated_COL_Overflow", "Product-Dedicated-COL + shared overflow", a, p,
            DEDICATED_NOTE + "Pallets beyond the reservation go to a shared OVERFLOW zone of all leftover slots.", s)

# P2: Product-Class-COL. A full zone spills to the next zone AWAY from the dock;
# slow C pallets never spill into the A zone, so the best slots stay free for fast SKUs.
class_fallback = {"A": ["A", "B", "C"], "B": ["B", "C", "A"], "C": ["C", "B"]}
save_design("P2_Class_COL", "Product-Class-COL",
            zoned_slots("Product-Class-COL", ["A", "B", "C"]),
            sku_preferences([class_fallback[c] for c in sku_info["class"]]),
            "Zones A and B sized at their class peak stock (Task 4b, heights mixed), nearest first; "
            "zone C takes every remaining usable slot. Closest open location within a zone is "
            "approximated by the zone's fill order (nearest first).")

# P3: Product-Dedicated-COL with tall openings protected, no fallback
a, p, s = dedicated_slots(protect_tall=True, overflow=False)
save_design("P3_Dedicated_COL_TallProtected", "Product-Dedicated-COL, tall protected", a, p,
            DEDICATED_NOTE + f"Pallets over {TALL_THRESHOLD_IN} in only in openings over {TALL_THRESHOLD_IN} in "
            "and shorter pallets only in the others; when a height group runs out, the SKU's remaining "
            "reservation is topped up from any slot it fits (see allocation_shortfalls.csv, "
            "slots_outside_height_group). Overflow falls back on today's habit.", s)

# P4: Product-Shared-COL: one zone of every usable slot
save_design("P4_Shared_COL", "Product-Shared-COL",
            zoned_slots("Product-Shared-COL", ["All SKUs"]),
            sku_preferences([["All SKUs"]] * len(sku_info)),
            "One zone holding every usable slot, filled nearest the dock first; exact expression of COL.")

# P5: Duration-of-Stay-Target as the brief defines a target policy:
# zone_1 = the SKU's stay target, zone_2 = NEAREST.
save_design("P5_DoS_Target", "Duration-of-Stay-Target",
            zoned_slots("Duration-of-Stay-Target", STAY_ZONES),
            sku_preferences([[z, NEAREST] for z in sku_info["stay"]]),
            "Duration of stay estimated per SKU (average stock / pallets shipped per day), because zones "
            "are set per SKU, not per pallet. Four targets (<=14, 14-30, 30-60, >60 days) sized at their "
            "peak (Task 4b, heights mixed), nearest first; T4 also takes the remaining usable slots. "
            "A full target sends the pallet to the free slot whose distance is closest to the target (NEAREST).")

# P5b: variant of P5 with neighbouring stay zones as fallback instead of NEAREST
stay_fallback = {
    STAY_ZONES[0]: [STAY_ZONES[0], STAY_ZONES[1], STAY_ZONES[2]],
    STAY_ZONES[1]: [STAY_ZONES[1], STAY_ZONES[2], STAY_ZONES[0]],
    STAY_ZONES[2]: [STAY_ZONES[2], STAY_ZONES[3], STAY_ZONES[1]],
    STAY_ZONES[3]: [STAY_ZONES[3], STAY_ZONES[2], STAY_ZONES[1]],
}
save_design("P5b_DoS_Zones", "Duration-of-Stay zones (variant of P5)",
            zoned_slots("Duration-of-Stay-Target", STAY_ZONES),
            sku_preferences([stay_fallback[z] for z in sku_info["stay"]]),
            "Same zones as P5; a full zone spills to the neighbouring zone farther from the dock first, "
            "then the nearer one, then today's habit. Kept to compare with NEAREST.")

pd.DataFrame(manifest).to_csv(OUTPUT / "Task5a_Design_Manifest.csv", index=False)
print(f"\nAll evaluator files and the design manifest saved to: {OUTPUT}")
