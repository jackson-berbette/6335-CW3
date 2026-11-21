"""
Task 5c - Round 2: improved versions of the round-1 leaders
===========================================================
Round 1 (evaluator, 21/11/2026) showed, in the worst practice future:
  P2 Class     : A 93.6%, B 99.5% (too small), C 62.6% (too big)          -> 231.2 h/week
  P5b DoS zones: T1 92.8%, T2 99.5% (too small), T3 94.0%, T4 61.8%,
                 0.4% last resort                                         -> 230.6 h/week
  P3 (tall protected) was the only design that improved height fit (15.4 vs 17.0 in),
  while every zoned design worsened it (about 19.8-19.9 in).

Round-2 designs written by this script (same inputs and rules as Task5a.py):
  P2_Class_COL_v2                 A +100, B +250 slots, taken from C (C takes the rest)
  P5b_DoS_Zones_v2                T2 +300, T3 +100 slots, taken from T4 (T4 takes the rest);
                                  T2 overflow now goes T2 -> T3 -> T4 (T4 has spare room)
  P5b_DoS_Zones_v2_TallProtected  P5b_v2 zones, each split into a TALL part (openings > 72 in)
                                  and a SHORT part; tall pallets use TALL parts only, short
                                  pallets use SHORT parts first and a TALL part only as last choice

Inputs : Data/slots.csv, Data/skus.csv, Output/CW3P1T1_VelocityStock_WembyFanClub.csv
Outputs: Output/Evaluator_Inputs/Round2/<design>/slot_zones.csv, sku_zones.csv
         Output/Evaluator_Inputs/Round2/Task5c_Round2_Manifest.csv
Run    : python Task5c_Round2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "Output" / "Evaluator_Inputs" / "Round2"
OUTPUT.mkdir(parents=True, exist_ok=True)

TALL_THRESHOLD_IN = 72
DAYS = 364
STAY_ZONES = ["T1 (up to 14 d)", "T2 (14-30 d)", "T3 (30-60 d)", "T4 (over 60 d)"]

# Round-1 zone sizes (from Task 4b, as written by Task5a.py) and the round-2 changes.
P2_V1 = {"A": 2045, "B": 1797}                     # C = all remaining usable slots
P2_V2 = {"A": P2_V1["A"] + 100, "B": P2_V1["B"] + 250}
P5_V1 = {STAY_ZONES[0]: 1761, STAY_ZONES[1]: 2504, STAY_ZONES[2]: 1477}   # T4 = remaining
P5_V2 = {STAY_ZONES[0]: P5_V1[STAY_ZONES[0]],
         STAY_ZONES[1]: P5_V1[STAY_ZONES[1]] + 300,
         STAY_ZONES[2]: P5_V1[STAY_ZONES[2]] + 100}

# ---------------------------------------------------------------------------
# Inputs: identical preparation to Task5a.py
# ---------------------------------------------------------------------------
slots = pd.read_csv(BASE / "Data/slots.csv").sort_values(
    ["dist_dock_ft", "level", "slot_id"]).reset_index(drop=True)
skus = pd.read_csv(BASE / "Data/skus.csv")
velocity = pd.read_csv(BASE / "Output/CW3P1T1_VelocityStock_WembyFanClub.csv")

sku_info = velocity[["sku_id", "pallets_shipped", "avg_stock_pallets"]].merge(
    skus[["sku_id", "pallet_height_in"]], on="sku_id", validate="one_to_one")
sku_info = sku_info.sort_values(["pallets_shipped", "sku_id"], ascending=[False, True]).reset_index(drop=True)
share_before = sku_info.pallets_shipped.cumsum().shift(fill_value=0) / sku_info.pallets_shipped.sum()
sku_info["class"] = np.select([share_before < 0.5, share_before < 0.8], ["A", "B"], default="C")
sku_info["expected_stay"] = sku_info.avg_stock_pallets / (sku_info.pallets_shipped / DAYS).replace(0, np.nan)
sku_info["stay"] = pd.cut(sku_info.expected_stay.fillna(np.inf), [0, 14, 30, 60, np.inf],
                          labels=STAY_ZONES).astype(str)
sku_info["tall"] = sku_info.pallet_height_in > TALL_THRESHOLD_IN

usable = (slots.max_pallet_in >= sku_info.pallet_height_in.min()).to_numpy()
usable_idx = slots.index[usable]
tall_slot = (slots.max_pallet_in > TALL_THRESHOLD_IN).to_numpy()
print(f"Slots: {len(slots):,}; usable: {usable.sum():,}; tall openings (> {TALL_THRESHOLD_IN} in): "
      f"{(tall_slot & usable).sum():,}")

manifest = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cut_zones(sizes, last_zone):
    """Consecutive blocks of usable slots, nearest the dock first; the last zone takes the rest.
    Too-low slots keep a blank zone (kept out of every zone, as the evaluator guide asks)."""
    zone = pd.Series("", index=slots.index, dtype=object)
    counts = list(sizes.values())
    assert sum(counts) <= len(usable_idx), "zones exceed the usable slots"
    cursor = 0
    for z, n in sizes.items():
        zone.loc[usable_idx[cursor:cursor + n]] = z
        cursor += n
    zone.loc[usable_idx[cursor:]] = last_zone
    return zone


def preferences(lists):
    out = sku_info[["sku_id"]].copy()
    for k in range(3):
        out[f"zone_{k + 1}"] = [p[k] if k < len(p) else "" for p in lists]
    return out


def save(name, zone, prefs, note):
    sz = slots[["slot_id"]].copy()
    sz["zone"] = zone.values
    sz["fill_order"] = pd.Series(pd.NA, index=sz.index, dtype="Int64")
    inside = sz.zone != ""
    sz.loc[inside, "fill_order"] = sz[inside].groupby("zone").cumcount() + 1   # nearest first
    # checks: every SKU has a first choice, no gaps, every named zone holds slots
    named = set(prefs[["zone_1", "zone_2", "zone_3"]].stack()) - {""}
    missing = named - set(sz.zone[inside])
    assert not missing, f"{name}: zones with no slots: {missing}"
    assert (prefs.zone_1 != "").all() and len(prefs) == len(sku_info) and len(sz) == len(slots)
    folder = OUTPUT / name
    folder.mkdir(exist_ok=True)
    sz.to_csv(folder / "slot_zones.csv", index=False)
    prefs.to_csv(folder / "sku_zones.csv", index=False)
    sizes = sz.zone[inside].value_counts()
    manifest.append({"design": name, "zones": len(sizes), "slots_in_zones": int(inside.sum()),
                     "slots_blank": int((~inside).sum()),
                     "zone_sizes": "; ".join(f"{z}: {n}" for z, n in sizes.sort_index().items()),
                     "change": note})
    print(f"{name}: " + ", ".join(f"{z} {n}" for z, n in sizes.sort_index().items()))


# ---------------------------------------------------------------------------
# P2_Class_COL_v2
# ---------------------------------------------------------------------------
class_fallback = {"A": ["A", "B", "C"], "B": ["B", "C", "A"], "C": ["C", "B"]}   # same as round 1
save("P2_Class_COL_v2", cut_zones(P2_V2, "C"),
     preferences([class_fallback[c] for c in sku_info["class"]]),
     "Round 1: B peaked at 99.5%, C at 62.6%. A +100 and B +250 slots, taken from C.")

# ---------------------------------------------------------------------------
# P5b_DoS_Zones_v2
# ---------------------------------------------------------------------------
T1, T2, T3, T4 = STAY_ZONES
stay_fallback_v2 = {T1: [T1, T2, T3], T2: [T2, T3, T4], T3: [T3, T4, T2], T4: [T4, T3, T2]}
zone_p5 = cut_zones(P5_V2, T4)
save("P5b_DoS_Zones_v2", zone_p5,
     preferences([stay_fallback_v2[s] for s in sku_info["stay"]]),
     "Round 1: T2 99.5%, T4 61.8%, 0.4% last resort. T2 +300 and T3 +100 slots, taken from T4; "
     "T2 overflow now T2 -> T3 -> T4 (was T2 -> T3 -> T1).")

# ---------------------------------------------------------------------------
# P5b_DoS_Zones_v2_TallProtected: split each stay zone into TALL and SHORT parts
# ---------------------------------------------------------------------------
zone_hp = zone_p5.copy()
inside = zone_hp != ""
zone_hp[inside & tall_slot] = zone_hp[inside & tall_slot] + " TALL"
zone_hp[inside & ~tall_slot] = zone_hp[inside & ~tall_slot] + " SHORT"
existing = set(zone_hp[zone_hp != ""])

prefs_hp = []
for s, is_tall in zip(sku_info["stay"], sku_info["tall"]):
    order = stay_fallback_v2[s]
    if is_tall:     # tall pallets only fit tall openings
        p = [f"{z} TALL" for z in order]
    else:           # short pallets: own SHORT part, next SHORT part, then own TALL part as last choice
        p = [f"{order[0]} SHORT", f"{order[1]} SHORT", f"{order[0]} TALL"]
    prefs_hp.append([z for z in p if z in existing])
save("P5b_DoS_Zones_v2_TallProtected", zone_hp, preferences(prefs_hp),
     "P5b_v2 zones split into TALL (openings > 72 in) and SHORT parts: tall pallets use TALL parts only; "
     "short pallets use their SHORT part, the next zone's SHORT part, and their own TALL part only as "
     "third choice. Aims to recover height fit, as P3 did (15.4 in).")

pd.DataFrame(manifest).to_csv(OUTPUT / "Task5c_Round2_Manifest.csv", index=False)
print(f"\nRound-2 design files and manifest saved to: {OUTPUT}")

