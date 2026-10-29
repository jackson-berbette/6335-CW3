"""
Casework 3 - StackBay Warehousing - Phase 1, Task 3: Performance baseline
==========================================================================
Run from the project folder:   python task3.py              (data in ./data)
                         or:   python task3.py path/to/data
Needs outputs_task1/velocity_class_and_stock.csv and outputs_task1/1f_daily_stock_total.csv
(run task1.py first). Outputs go to ./outputs_task3/  (about 30 seconds).

3a  measure definitions        -> 3a_measure_definitions.csv
3b  baseline values            -> 3b_baseline_scorecard.csv   (+ 3_scorecard.md, ready to paste)
3c  travel/labour breakdown    -> 3c_by_class / client / level / aisle .csv + charts, 3c_top5pct.csv
3d  opportunity estimate       -> 3d_opportunity.csv, 3d_slot_plan_*.csv, chart
3e  judgement evidence         -> 3e_inherent_vs_chosen.csv

Modelling rules used (same as task2.py):
  one-way travel  = 40 ft + |door x - aisle x| + y of the bay           (outbound/inbound travel_ft)
  slot move (min) = 2*travel/264 + 2*lift/60 + 1.5                       (pick or put-away)
  dock move (min) = 1.8                                                  (loading or unloading)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = "outputs_task3"
os.makedirs(OUT, exist_ok=True)
VEL_FILE = "outputs_task1/velocity_class_and_stock.csv"
STOCK_FILE = "outputs_task1/1f_daily_stock_total.csv"
SPEED_FT_MIN, LIFT_FT_MIN, HANDLE_MIN, DOCK_MOVE_MIN = 264, 60, 1.5, 1.8
DAY_SHIFT_H, NIGHT_SHIFT_H, DRIVER_H_WEEK = 12.0, 7.5, 36.0
WEEKS = 52
TALL_IN = 90                       # "tall" opening (same threshold as task2.py)
SHORT_IN = 60                      # "short" pallet
HIGH_LEVEL = 4                     # rack level counted as a "high lift" for drivers
FT_PER_MILE = 5280
CLS_COL = {"A": "#dc2626", "B": "#f59e0b", "C": "#10b981", "D": "#64748b"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string())


def move_min(travel_ft, lift_ft):
    return 2 * travel_ft / SPEED_FT_MIN + 2 * lift_ft / LIFT_FT_MIN + HANDLE_MIN


# ---------------------------------------------------------------------------
# Load and build one table of all slot moves (picks + put-aways)
# ---------------------------------------------------------------------------
slots = pd.read_csv(f"{DATA}/slots.csv")
docks = pd.read_csv(f"{DATA}/docks.csv")
skus = pd.read_csv(f"{DATA}/skus.csv").set_index("sku_id")
clients = pd.read_csv(f"{DATA}/clients.csv").set_index("client_id")
trucks = pd.read_csv(f"{DATA}/trucks.csv", parse_dates=["ship_date", "departs_at"])
inb = pd.read_csv(f"{DATA}/inbound.csv", parse_dates=["receipt_date", "arrived_at", "putaway_at"])
out = pd.read_csv(f"{DATA}/outbound.csv", parse_dates=["ship_date", "picked_at", "departs_at", "received_date"])
opening = pd.read_csv(f"{DATA}/opening_stock.csv", parse_dates=["received_date"])
current = pd.read_csv(f"{DATA}/current_stock.csv", parse_dates=["received_date"])
vel = pd.read_csv(VEL_FILE).set_index("sku_id")
stock_day = pd.read_csv(STOCK_FILE)
stock_col = [c for c in stock_day.columns if "stock" in c.lower() or "pallets" in c.lower()][-1]
END = pd.Timestamp("2026-07-05")
S = slots.set_index("slot_id")
N_SLOTS = len(slots)
cname = {c: clients.client_name[c].replace(" Restaurants", "") for c in clients.index}

moves = pd.concat([
    pd.DataFrame({"type": "pick", "date": out.ship_date, "sku_id": out.sku_id, "door_id": out.door_id,
                  "slot_id": out.picked_from_slot, "travel_ft": out.travel_ft, "lift_ft": out.lift_ft}),
    pd.DataFrame({"type": "put-away", "date": inb.receipt_date, "sku_id": inb.sku_id, "door_id": inb.door_id,
                  "slot_id": inb.putaway_slot, "travel_ft": inb.travel_ft, "lift_ft": inb.lift_ft})],
    ignore_index=True)
moves["client"] = moves.sku_id.map(skus.client_id).map(cname)
moves["velocity_class"] = moves.sku_id.map(vel.velocity_class)
moves["top5"] = moves.sku_id.map(vel.top5pct_fastest).astype(bool)
moves["level"] = moves.slot_id.map(S.level)
moves["aisle"] = moves.slot_id.map(S.aisle)
moves["rt_ft"] = 2 * moves.travel_ft                                     # round trip
moves["travel_min"] = moves.rt_ft / SPEED_FT_MIN
moves["lift_min"] = 2 * moves.lift_ft / LIFT_FT_MIN
moves["move_min"] = moves.travel_min + moves.lift_min + HANDLE_MIN
moves["dock_min"] = DOCK_MOVE_MIN                                          # load or unload of that pallet
moves["labour_min"] = moves.move_min + moves.dock_min

# ---------------------------------------------------------------------------
# 3a. Measure definitions
# ---------------------------------------------------------------------------
defs = [
    # id, area, measure, unit, perspective, formula, storage design affects?, why
    ("T1", "Travel", "Forklift travel per year", "miles/yr", "StackBay, driver",
     "sum over picks + put-aways of 2 x travel_ft / 5280 (outbound.travel_ft, inbound.travel_ft)", "Yes",
     "Slot location sets the distance of every move"),
    ("T2", "Travel", "Average round trip per slot move", "ft", "StackBay, driver",
     "mean of 2 x travel_ft over all slot moves", "Yes", "Same as T1, per move"),
    ("T3", "Travel", "Share of travel caused by fastest 5% of SKUs", "%", "StackBay",
     "travel of top-5% SKUs (Task 1b flag) / total travel", "Yes", "Depends on where fast SKUs sit"),
    ("L1", "Labour", "Forklift hours per week (slot + dock moves)", "h/week", "StackBay, driver",
     "sum(2*travel/264 + 2*lift/60 + 1.5) + 1.8 per pallet loaded/unloaded, / 52", "Yes (slot-move part)",
     "Travel and lift depend on slots; the 1.8 min dock move and the 1.5 min handling do not"),
    ("L2", "Labour", "Forklift minutes per pallet handled", "min/pallet", "StackBay",
     "L1 total minutes / (pallets picked + pallets put away)", "Yes (partly)", "As L1"),
    ("L3", "Labour", "Full-time drivers needed", "drivers", "StackBay, driver",
     "L1 / 36 h per driver-week (no allowance for breaks or waiting)", "Yes", "Follows L1"),
    ("L4", "Labour", "Peak day-shift / night-shift forklift hours", "h/shift", "StackBay, driver",
     "max over shifts of the L1 hours of that shift (day: picks + loading; night: unloading + put-away)",
     "Yes", "Peaks set the fleet size; slot choice changes hours per move"),
    ("S1", "Space", "Slot occupancy (average, peak)", "% of 8,365 slots", "StackBay",
     "daily stock in pallets (Task 1f) / number of slots", "Partly",
     "Pallet count is set by flows; a design can only change how many slots are made available or reserved"),
    ("S2", "Space", "Slots a dedicated (fixed-slot) design would need", "slots",
     "StackBay", "sum over SKUs of peak stock (Task 1f)", "Yes (choice of shared vs dedicated)",
     "Shows the space cost of dedicating slots"),
    ("F1", "Fit", "Unused height above stored pallets", "in, % of opening", "StackBay",
     "opening_height_in - pallet_height_in, averaged over pallet-days in stock", "Yes",
     "Matching pallet height to opening height is a storage choice"),
    ("F2", "Fit", "Tall slots holding short pallets", "% of pallet-days in tall slots",
     "StackBay", f"pallet-days with opening >= {TALL_IN} in and pallet < {SHORT_IN} in / pallet-days in tall slots",
     "Yes", "Tall openings are scarce; filling them with short pallets wastes them"),
    ("F3", "Fit", "Cube utilisation of occupied slots", "%", "StackBay",
     "40 x 48 x pallet height / 1728 / slot cube_ft3, averaged over pallet-days", "Yes", "As F1"),
    ("I1", "Inventory", "Average / peak stock", "pallets", "StackBay, client",
     "Task 1f daily stock", "No", "Set by receipts and shipments (client orders, supplier deliveries)"),
    ("I2", "Inventory", "Days of supply", "days", "StackBay, client",
     "average stock / average pallets shipped per operating day", "No", "As I1"),
    ("I3", "Inventory", "Dwell time of shipped pallets (median, mean, 90th pct)", "days", "Client",
     "outbound.days_in_stock", "No", "Set by stock levels and FIFO, not by where pallets sit"),
    ("I4", "Inventory", "Pallets in stock > 180 days at year end", "pallets", "Client, StackBay",
     "current_stock age on 5 Jul 2026", "No", "Slow/dead stock is a client/replenishment issue"),
    ("I5", "Inventory", "FIFO compliance", "% of picks", "Client",
     "share of picks that took the oldest pallet of the SKU then in stock", "Partly",
     "Mostly a picking rule; scattered slots make FIFO harder"),
    ("C1", "Client service", "Pick-to-departure lead", "min", "Client",
     "departs_at - picked_at, mean and 5th percentile", "Yes (via labour)",
     "Faster picking gives more slack before the truck leaves"),
    ("C2", "Client service", "Truck fill", "% of 26 pallets", "Client, StackBay",
     "trucks.pallets / 26", "No", "Set by routes and orders"),
    ("C3", "Receiving", "Dock-to-stock time", "min", "StackBay, supplier",
     "putaway_at - arrived_at, mean and 95th pct", "Yes (via labour)", "Longer put-away trips delay put-away"),
    ("C4", "Receiving", "Nights whose last put-away is after 23:00", "nights", "StackBay, driver",
     "count of receipt nights with max(putaway_at) >= 23:00", "Yes (via labour)", "As C3"),
    ("D1", "Driver", "High lifts", f"% of slot moves at level >= {HIGH_LEVEL}", "Driver",
     f"share of picks + put-aways with slot level >= {HIGH_LEVEL}", "Yes", "Level of the slot is a design choice"),
    ("D2", "Driver", "Average lift per slot move", "ft", "Driver", "mean lift_ft", "Yes", "As D1"),
    ("D3", "Driver", "Miles driven per driver-shift", "miles", "Driver",
     "travel per shift / forklifts needed in that shift (L4 / shift length)", "Yes", "As T1"),
]
D = pd.DataFrame(defs, columns=["id", "area", "measure", "unit", "perspective", "how_computed_from_files",
                                "storage_design_affects", "why"])
D.to_csv(f"{OUT}/3a_measure_definitions.csv", index=False)

cannot = pd.DataFrame([
    ("Demand (pallets shipped per SKU, client, day)", "Set by the clients' restaurants"),
    ("Receipts, purchase orders, supplier arrival times", "Set by suppliers and the replenishment policy"),
    ("Truck routes, stops, departure times and truck fill", "Set by the clients' delivery plans"),
    ("Stock levels and dwell time", "Equal to receipts minus shipments - storage location does not change them"),
    ("Dock moves (1.8 min per pallet loaded/unloaded) and 1.5 min handling", "Same wherever the pallet is stored"),
    ("Number of slots, rack heights, building shape, door positions", "Fixed building (unless re-racked)"),
], columns=["measure a storage design cannot affect", "why"])
cannot.to_csv(f"{OUT}/3a_not_affected_by_storage.csv", index=False)
show("3a Measures a storage design cannot affect", cannot)

# ---------------------------------------------------------------------------
# 3b. Baseline values
# ---------------------------------------------------------------------------
val = {}
val["T1"] = moves.rt_ft.sum() / FT_PER_MILE
val["T2"] = moves.rt_ft.mean()
val["T3"] = 100 * moves.loc[moves.top5, "rt_ft"].sum() / moves.rt_ft.sum()
val["L1"] = moves.labour_min.sum() / 60 / WEEKS
val["L2"] = moves.labour_min.sum() / len(moves)
val["L3"] = val["L1"] / DRIVER_H_WEEK

picks, puts = moves[moves.type == "pick"], moves[moves.type == "put-away"]
day_h = picks.groupby("date").labour_min.sum() / 60
night_h = puts.groupby("date").labour_min.sum() / 60
day_mi = picks.groupby("date").rt_ft.sum() / FT_PER_MILE
night_mi = puts.groupby("date").rt_ft.sum() / FT_PER_MILE
val["L4"] = f"day {day_h.max():.1f} h ({day_h.idxmax().date()}); night {night_h.max():.1f} h ({night_h.idxmax().date()})"

st = stock_day[stock_col]
val["S1"] = f"average {100 * st.mean() / N_SLOTS:.0f}% ({st.mean():.0f} pallets); peak {100 * st.max() / N_SLOTS:.0f}% ({st.max():.0f})"
val["S2"] = f"{vel.peak_stock_pallets.sum():.0f} slots = {100 * vel.peak_stock_pallets.sum() / N_SLOTS:.0f}% of the building"

# Fit: every stored pallet weighted by the days it was in stock
dwell = pd.concat([out.set_index("pallet_id").days_in_stock,
                   (END - current.set_index("pallet_id").received_date).dt.days])
stored = pd.concat([opening[["pallet_id", "sku_id", "slot_id"]],
                    inb[["pallet_id", "sku_id", "putaway_slot"]].rename(columns={"putaway_slot": "slot_id"})])
stored = stored.merge(slots[["slot_id", "opening_height_in", "cube_ft3", "level"]], on="slot_id")
stored["h"] = stored.sku_id.map(skus.pallet_height_in)
stored["days"] = stored.pallet_id.map(dwell).fillna(0) + 1e-9
stored["unused_in"] = stored.opening_height_in - stored.h
stored["cube_util"] = 40 * 48 * stored.h / 1728 / stored.cube_ft3
w = stored.days
tall = stored.opening_height_in >= TALL_IN
val["F1"] = (f"{np.average(stored.unused_in, weights=w):.1f} in = "
             f"{100 * np.average(stored.unused_in / stored.opening_height_in, weights=w):.0f}% of opening")
val["F2"] = f"{100 * w[tall & (stored.h < SHORT_IN)].sum() / w[tall].sum():.0f}%"
val["F3"] = f"{100 * np.average(stored.cube_util, weights=w):.0f}%"

ship_days = out.ship_date.nunique()
val["I1"] = f"average {st.mean():.0f}; peak {st.max():.0f} pallets"
val["I2"] = st.mean() / (len(out) / ship_days)
dd = out.days_in_stock
val["I3"] = f"median {dd.median():.0f}; mean {dd.mean():.1f}; p90 {dd.quantile(0.9):.0f}"
age = (END - current.received_date).dt.days
val["I4"] = f"{(age > 180).sum()} (of which {(age > 365).sum()} > 1 year)"

# FIFO (replay of stock by SKU in time order; no slots needed)
stock_dates = {}
for sk, rd in zip(opening.sku_id, opening.received_date):
    stock_dates.setdefault(sk, []).append(rd)
ev = pd.concat([pd.DataFrame({"t": out.picked_at, "k": 0, "sku": out.sku_id, "rd": out.received_date}),
                pd.DataFrame({"t": inb.putaway_at, "k": 1, "sku": inb.sku_id, "rd": inb.receipt_date})]) \
       .sort_values(["t", "k"], kind="stable")
ok = n = 0
for _, k, sk, rd in ev.itertuples(index=False):
    lst = stock_dates.setdefault(sk, [])
    if k == 0:
        ok += rd <= min(lst); n += 1
        lst.remove(rd)
    else:
        lst.append(rd)
val["I5"] = f"{100 * ok / n:.1f}%"

lead = (out.departs_at - out.picked_at).dt.total_seconds() / 60
val["C1"] = f"mean {lead.mean():.0f} min; 5th pct {lead.quantile(0.05):.0f} min; min {lead.min():.0f} min"
val["C2"] = f"{100 * trucks.pallets.mean() / 26:.0f}% (avg {trucks.pallets.mean():.1f} pallets)"
d2s = (inb.putaway_at - inb.arrived_at).dt.total_seconds() / 60
val["C3"] = f"mean {d2s.mean():.0f} min; p95 {d2s.quantile(0.95):.0f} min; max {d2s.max():.0f} min"
last = inb.groupby("receipt_date").putaway_at.max()
val["C4"] = f"{(last.dt.hour >= 23).sum()} of {len(last)} nights"
val["D1"] = f"{100 * (moves.level >= HIGH_LEVEL).mean():.0f}%"
val["D2"] = moves.lift_ft.mean()
day_drivers = day_h / DAY_SHIFT_H
night_drivers = night_h / NIGHT_SHIFT_H
val["D3"] = (f"day {np.mean(day_mi / day_drivers):.1f}; night {np.mean(night_mi / night_drivers):.1f} "
             f"miles per driver-shift")


def fmt(v):
    return f"{v:,.1f}" if isinstance(v, (float, np.floating)) else str(v)


score = D[["id", "area", "measure", "unit", "perspective", "storage_design_affects"]].copy()
score["baseline_value"] = score.id.map(lambda i: fmt(val[i]))
score.to_csv(f"{OUT}/3b_baseline_scorecard.csv", index=False)
show("3b Baseline scorecard (historical year as operated)", score[["id", "measure", "baseline_value", "unit"]])

time_split = pd.Series({"travel": moves.travel_min.sum(), "lift": moves.lift_min.sum(),
                        "handling (1.5 min)": HANDLE_MIN * len(moves), "dock moves (1.8 min)": moves.dock_min.sum()})
time_split = (100 * time_split / time_split.sum()).round(1)
show("3b Where forklift time goes (% of all labour)", time_split.to_frame("%"))
time_split.to_csv(f"{OUT}/3b_labour_split.csv", header=["pct"])

# ---------------------------------------------------------------------------
# 3c. Breakdown
# ---------------------------------------------------------------------------
def breakdown(key):
    g = moves.groupby(key).agg(moves=("rt_ft", "size"), travel_miles=("rt_ft", "sum"),
                               labour_h=("labour_min", "sum"), slot_move_h=("move_min", "sum"),
                               avg_rt_ft=("rt_ft", "mean"), avg_lift_ft=("lift_ft", "mean"))
    g["travel_miles"] /= FT_PER_MILE
    g["labour_h"] /= 60
    g["slot_move_h"] /= 60
    g["pct_moves"] = 100 * g.moves / g.moves.sum()
    g["pct_travel"] = 100 * g.travel_miles / g.travel_miles.sum()
    g["pct_labour"] = 100 * g.labour_h / g.labour_h.sum()
    g["travel_vs_moves"] = g.pct_travel / g.pct_moves          # >1: travels more than its share
    return g.round(2)


bks = {}
for key in ["velocity_class", "client", "level", "aisle"]:
    bks[key] = breakdown(key)
    bks[key].to_csv(f"{OUT}/3c_by_{key}.csv")
    if key != "aisle":
        show(f"3c Travel and labour by {key}", bks[key])
show("3c Travel and labour by aisle (first 10 rows; full table in csv)", bks["aisle"].head(10))

top = breakdown("top5")
top.index = top.index.map({True: "fastest 5% of SKUs", False: "other 95%"})
top.to_csv(f"{OUT}/3c_top5pct.csv")
show("3c Fastest 5% of SKUs", top)
n_top = int(vel.top5pct_fastest.sum())
print(f"The {n_top} fastest SKUs make {top.loc['fastest 5% of SKUs', 'pct_moves']:.1f}% of slot moves and cause "
      f"{top.loc['fastest 5% of SKUs', 'pct_travel']:.1f}% of travel "
      f"(average round trip {top.loc['fastest 5% of SKUs', 'avg_rt_ft']:.0f} ft vs "
      f"{top.loc['other 95%', 'avg_rt_ft']:.0f} ft for the rest).")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, key in zip(axes.flat, ["velocity_class", "client", "level", "aisle"]):
    b = bks[key]
    x = np.arange(len(b)); wd = 0.4
    ax.bar(x - wd / 2, b.pct_moves, wd, label="% of moves", color="#94a3b8")
    ax.bar(x + wd / 2, b.pct_travel, wd, label="% of travel", color="#2563eb")
    ax.set_xticks(x); ax.set_xticklabels(b.index.astype(str), rotation=90 if key == "aisle" else 0)
    ax.set_title(f"by {key.replace('_', ' ')}"); ax.set_ylabel("%")
axes[0, 0].legend()
fig.suptitle("3c. Share of slot moves vs share of forklift travel")
save(fig, "3c_breakdown")

# ---------------------------------------------------------------------------
# 3d. Opportunity: fastest SKUs in the slots closest to the dock
# ---------------------------------------------------------------------------
# door x slot one-way distance (rule 1) so each move keeps its historical door
Dm = 40 + np.abs(slots.aisle_x_ft.values[None, :] - docks.x_ft.values[:, None]) + slots.y_ft.values[None, :]
didx = {d: k for k, d in enumerate(docks.door_id)}
moves["door_i"] = moves.door_id.map(didx)
order_slots = np.argsort(slots.dist_dock_ft.values, kind="stable")       # closest first (mean over doors)
maxp = slots.max_pallet_in.values
elev = slots.elev_ft.values
hist_move_min = moves.move_min.sum()
hist_rt = moves.rt_ft.sum()


def plan(sku_order, need, respect_fit):
    """Greedy: give each SKU, in order, its `need` closest free slots (that fit its height if respect_fit)."""
    free = np.ones(N_SLOTS, bool)
    assign, short = {}, []
    for sk in sku_order:
        k = int(need[sk])
        if k <= 0:
            continue
        cand = order_slots[free[order_slots]]
        if respect_fit:
            cand = cand[maxp[cand] >= skus.pallet_height_in[sk]]
        if len(cand) < k:
            short.append(sk)
            continue
        pick = cand[:k]
        free[pick] = False
        assign[sk] = pick
    return assign, short


def evaluate(assign):
    """Moves of assigned SKUs use the average distance/lift of their slots from the move's own door;
    every other move keeps its historical distance and lift."""
    rt = moves.rt_ft.values.copy()
    lift = moves.lift_ft.values.copy()
    for sk, sl in assign.items():
        m = np.flatnonzero(moves.sku_id.values == sk)
        rt[m] = 2 * Dm[moves.door_i.values[m]][:, sl].mean(axis=1)
        lift[m] = elev[sl].mean()
    mm = rt / SPEED_FT_MIN + 2 * lift / LIFT_FT_MIN + HANDLE_MIN
    return rt, mm


ranked = vel.sort_values("rank").index
peak_need = vel.peak_stock_pallets.round().astype(int)
avg_need = np.ceil(vel.avg_stock_pallets).astype(int)
moves_per_sku = moves.groupby("sku_id").size()
coi_order = (moves_per_sku.reindex(vel.index).fillna(0) / avg_need.clip(lower=1)).sort_values(ascending=False).index
top_set = vel.index[vel.top5pct_fastest.astype(bool)]
top_ranked = [s for s in ranked if s in set(top_set)]

scen = [
    ("S1", "Fastest 5% only, dedicated slots = peak stock, height fit respected", top_ranked, peak_need, True),
    ("S2", "Fastest 5% only, dedicated slots = peak stock, height fit ignored", top_ranked, peak_need, False),
    ("S3", "All SKUs by pallets shipped, slots = average stock (rounded up), height fit respected",
     list(ranked), avg_need, True),
    ("S4", "All SKUs by moves per slot needed (COI), slots = average stock, height fit respected",
     list(coi_order), avg_need, True),
]
rows = []
for sid, desc, order, need, fit in scen:
    assign, short = plan(order, need, fit)
    rt, mm = evaluate(assign)
    sel = moves.top5.values
    used = sum(len(v) for v in assign.values())
    rows.append({
        "scenario": sid, "description": desc, "slots_used": used, "skus_not_placed": len(short),
        "travel_miles_baseline": hist_rt / FT_PER_MILE, "travel_miles_new": rt.sum() / FT_PER_MILE,
        "travel_saving_pct": 100 * (1 - rt.sum() / hist_rt),
        "top5_travel_saving_pct": 100 * (1 - rt[sel].sum() / moves.rt_ft.values[sel].sum()),
        "slot_move_h_per_week_baseline": hist_move_min / 60 / WEEKS,
        "slot_move_h_per_week_new": mm.sum() / 60 / WEEKS,
        "labour_saving_h_per_week": (hist_move_min - mm.sum()) / 60 / WEEKS,
        "labour_saving_pct_of_all_forklift_h": 100 * (hist_move_min - mm.sum()) / moves.labour_min.sum(),
        "drivers_saved": (hist_move_min - mm.sum()) / 60 / WEEKS / DRIVER_H_WEEK,
    })
    plan_df = pd.DataFrame([(sk, slots.slot_id.values[s]) for sk, sl in assign.items() for s in sl],
                           columns=["sku_id", "slot_id"])
    plan_df.to_csv(f"{OUT}/3d_slot_plan_{sid}.csv", index=False)
opp = pd.DataFrame(rows).set_index("scenario").round(2)
opp.to_csv(f"{OUT}/3d_opportunity.csv")
show("3d Opportunity: same moves, fast SKUs moved to the slots closest to the dock", opp.T)

assumptions = [
    "Same pallets, same days, same doors as the historical year; only the slot of each pallet changes.",
    "'Closest to the dock' = smallest dist_dock_ft (mean distance over the 32 doors); the saving is then computed "
    "with each move's real door.",
    "Dedicated slots: a SKU's moves are spread evenly over its slots (distance = average over its slots).",
    "S1/S2 reserve each top-5% SKU's PEAK stock (never runs out of its slots); S3/S4 use AVERAGE stock rounded up "
    "(sum of peaks is ~117% of the building, so peak-based dedication for all SKUs is infeasible).",
    "S1/S2: SKUs outside the top 5% keep their historical distances (displacing them is ignored, small bias).",
    "Lift: the average elevation of the assigned slots (low levels are close and tall racks are deeper, so lift "
    "can rise or fall).",
    "Height fit respected = slot max_pallet_in >= pallet height; S2 shows the effect of ignoring it.",
    "No congestion, no re-slotting moves, no honeycombing; dock moves (1.8 min) and handling (1.5 min) unchanged.",
]
pd.Series(assumptions, name="assumption").to_csv(f"{OUT}/3d_assumptions.csv", index=False)
print("\nAssumptions:\n- " + "\n- ".join(assumptions))

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(opp.index[::-1], opp.travel_saving_pct[::-1], color="#2563eb", label="travel saving (%)")
for y_, (v, hsav) in enumerate(zip(opp.travel_saving_pct[::-1], opp.labour_saving_h_per_week[::-1])):
    ax.text(v + 0.3, y_, f"{v:.1f}%  ({hsav:.0f} h/week)", va="center")
ax.set_xlabel("Saving vs historical year (% of forklift travel)")
ax.set_title("3d. Opportunity: fast SKUs stored closest to the dock")
ax.set_xlim(0, max(opp.travel_saving_pct) * 1.4)
save(fig, "3d_opportunity")

# ---------------------------------------------------------------------------
# 3e. Inherent vs chosen (numbers for the judgement table)
# ---------------------------------------------------------------------------
day_counts = out.groupby(out.ship_date.dt.day_name()).size() / WEEKS
night_counts = inb.groupby(inb.receipt_date.dt.day_name()).size() / WEEKS
cls_dist = moves.groupby("velocity_class").rt_ft.mean()
ev3 = pd.DataFrame([
    ("Inherent", "Demand concentration", f"{n_top} SKUs (5%) = {top.loc['fastest 5% of SKUs', 'pct_moves']:.0f}% of slot moves"),
    ("Inherent", "Receiving peaks Monday/Tuesday night",
     f"Mon night avg {night_counts.get('Monday', 0):.0f} pallets vs Thu {night_counts.get('Thursday', 0):.0f}"),
    ("Inherent", "Space is tight", f"average occupancy {100 * st.mean() / N_SLOTS:.0f}%, peak {100 * st.max() / N_SLOTS:.0f}%"),
    ("Inherent", "Slow/dead stock held for clients", val["I4"] + " pallets > 180 days at year end"),
    ("Inherent", "Building: 32 doors on one wall, deep aisles", f"closest slot {slots.dist_dock_ft.min():.0f} ft, "
     f"farthest {slots.dist_dock_ft.max():.0f} ft (mean over doors)"),
    ("Chosen", "Put-away ignores velocity",
     f"avg round trip class A {cls_dist.get('A', np.nan):.0f} ft vs class D {cls_dist.get('D', np.nan):.0f} ft"),
    ("Chosen", "Height fit", f"unused height {val['F1']}; tall slots holding short pallets {val['F2']}"),
    ("Chosen", "Travel share of forklift time", f"{time_split['travel']:.0f}% of all forklift time is travel"),
    ("Chosen", "Opportunity left on the table", f"S1 saves {opp.loc['S1', 'travel_saving_pct']:.1f}% of travel; "
     f"S4 saves {opp.loc['S4', 'travel_saving_pct']:.1f}% ({opp.loc['S4', 'labour_saving_h_per_week']:.0f} h/week)"),
    ("Chosen (good)", "FIFO", f"{val['I5']} of picks took the oldest pallet"),
], columns=["inherent_or_chosen", "aspect", "evidence"])
ev3.to_csv(f"{OUT}/3e_inherent_vs_chosen.csv", index=False)
show("3e Evidence for the judgement", ev3)

# ---------------------------------------------------------------------------
# Scorecard as markdown (paste into the report)
# ---------------------------------------------------------------------------
with open(f"{OUT}/3_scorecard.md", "w") as f:
    f.write("# Task 3 baseline scorecard\n\n## Measures and baseline values\n\n")
    f.write("| ID | Area | Measure | Baseline | Unit | Perspective | Storage design affects? |\n|---|---|---|---|---|---|---|\n")
    for r in score.itertuples():
        f.write(f"| {r.id} | {r.area} | {r.measure} | {r.baseline_value} | {r.unit} | {r.perspective} | "
                f"{r.storage_design_affects} |\n")
    f.write("\n## Where forklift time goes\n\n| Component | % |\n|---|---|\n")
    for k, v in time_split.items():
        f.write(f"| {k} | {v} |\n")
    f.write("\n## Travel breakdown by velocity class\n\n| Class | % moves | % travel | avg round trip (ft) |\n|---|---|---|---|\n")
    for k, r in bks["velocity_class"].iterrows():
        f.write(f"| {k} | {r.pct_moves:.1f} | {r.pct_travel:.1f} | {r.avg_rt_ft:.0f} |\n")
    f.write("\n## Opportunity (3d)\n\n| Scenario | Travel saving % | Top-5% travel saving % | Forklift h/week saved | Drivers |\n|---|---|---|---|---|\n")
    for k, r in opp.iterrows():
        f.write(f"| {k}: {r.description} | {r.travel_saving_pct:.1f} | {r.top5_travel_saving_pct:.1f} | "
                f"{r.labour_saving_h_per_week:.1f} | {r.drivers_saved:.2f} |\n")
print(f"\nScorecard written to {OUT}/3_scorecard.md")
print("Done. All tables and charts are in", OUT)
