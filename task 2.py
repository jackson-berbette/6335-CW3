"""
Casework 3 - StackBay Warehousing - Phase 1, Task 2: Diagnose current operation
================================================================================
Run from the project folder:   python task2.py              (data in ./data)
                         or:   python task2.py path/to/data
Needs outputs_task1/velocity_class_and_stock.csv from task1.py (run task1.py first).
Outputs go to ./outputs_task2/   (about 1 minute: the year is replayed event by event)

Method of 2b/2c: the whole year is replayed in time order. Opening stock fills the racks;
every pick (outbound.picked_at) empties its slot, every put-away (inbound.putaway_at) fills one.
At each put-away we know exactly which slots were free and tall enough, so we can compare the
slot the driver chose with (i) all the alternatives and (ii) a random free fitting slot.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = "outputs_task2"
os.makedirs(OUT, exist_ok=True)
VEL_FILE = "outputs_task1/velocity_class_and_stock.csv"
TALL_IN = 90                      # "tall opening" threshold (inches) used in 2a/2b
SPEED_FT_MIN, LIFT_FT_MIN, HANDLE_MIN, DOCK_MOVE_MIN = 264, 60, 1.5, 1.8
DAY_SHIFT_H, NIGHT_SHIFT_H, DRIVER_H_WEEK = 12.0, 7.5, 36.0
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLS_COL = {"A": "#dc2626", "B": "#f59e0b", "C": "#10b981", "D": "#64748b"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string())


def move_min(travel_ft, lift_ft):
    """Modeling rule 3: one slot move (round trip + lift up/down + handling), minutes."""
    return 2 * travel_ft / SPEED_FT_MIN + 2 * lift_ft / LIFT_FT_MIN + HANDLE_MIN


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
slots = pd.read_csv(f"{DATA}/slots.csv")
docks = pd.read_csv(f"{DATA}/docks.csv")
skus = pd.read_csv(f"{DATA}/skus.csv").set_index("sku_id")
clients = pd.read_csv(f"{DATA}/clients.csv").set_index("client_id")
inb = pd.read_csv(f"{DATA}/inbound.csv", parse_dates=["receipt_date", "arrived_at", "putaway_at"])
out = pd.read_csv(f"{DATA}/outbound.csv", parse_dates=["ship_date", "picked_at", "departs_at", "received_date"])
opening = pd.read_csv(f"{DATA}/opening_stock.csv", parse_dates=["received_date"])
current = pd.read_csv(f"{DATA}/current_stock.csv", parse_dates=["received_date"])
vel = pd.read_csv(VEL_FILE).set_index("sku_id")
END = pd.Timestamp("2026-07-05")

for df, col in [(inb, "sku_id"), (out, "sku_id")]:
    df["velocity_class"] = df[col].map(vel.velocity_class)
    df["client_id"] = df[col].map(skus.client_id)
    df["pallet_height_in"] = df[col].map(skus.pallet_height_in)
S = slots.set_index("slot_id")

# ---------------------------------------------------------------------------
# 2a. Layout
# ---------------------------------------------------------------------------
lay = slots.groupby("aisle").agg(bays=("bay", "nunique"), slots=("slot_id", "size"),
                                 max_level=("level", "max"), depth_ft=("y_ft", "max"),
                                 mean_dist_dock_ft=("dist_dock_ft", "mean"),
                                 tall_openings=("opening_height_in", lambda h: (h >= TALL_IN).sum()))
lay.round(1).to_csv(f"{OUT}/2a_layout_by_aisle.csv")
show("2a Layout by aisle", lay.round(1))
lvl = slots.groupby("level").agg(slots=("slot_id", "size"), elev_ft=("elev_ft", "mean"),
                                 opening_mean_in=("opening_height_in", "mean"),
                                 opening_min_in=("opening_height_in", "min"),
                                 opening_max_in=("opening_height_in", "max"),
                                 tall_openings=("opening_height_in", lambda h: (h >= TALL_IN).sum()))
lvl.round(1).to_csv(f"{OUT}/2a_layout_by_level.csv")
show("2a Layout by level", lvl.round(1))
print(f"Total: {len(slots)} slots, {slots.aisle.nunique()} aisles, {slots.groupby('aisle').bay.nunique().sum()} bays "
      f"(sides L/R), up to {slots.level.max()} levels; openings {slots.opening_height_in.min()}-"
      f"{slots.opening_height_in.max()} in; slots >= {TALL_IN} in: {(slots.opening_height_in >= TALL_IN).sum()}")

# Distance verification. Rule 1: travel = 40 ft apron + |door x - aisle x| + depth into aisle (y of bay)
D = 40 + np.abs(slots.aisle_x_ft.values[None, :] - docks.x_ft.values[:, None]) + slots.y_ft.values[None, :]
chk = inb.merge(slots[["slot_id", "aisle_x_ft", "y_ft", "elev_ft"]], left_on="putaway_slot", right_on="slot_id") \
         .merge(docks.rename(columns={"x_ft": "door_x_ft", "y_ft": "door_y_ft"}), on="door_id")
err_travel = (40 + (chk.aisle_x_ft - chk.door_x_ft).abs() + chk.y_ft - chk.travel_ft).abs().max()
err_lift = (chk.elev_ft - chk.lift_ft).abs().max()
err_avg = np.abs(D.mean(axis=0) - slots.dist_dock_ft.values).max()
ex = 40 + abs(S.loc["A06-003-C", "aisle_x_ft"] - docks.set_index("door_id").loc["D01", "x_ft"]) + S.loc["A06-003-C", "y_ft"]
print(f"\n=== 2a Distance check ===\nFormula vs travel_ft (all {len(chk)} put-aways): max error {err_travel:.2e} ft; "
      f"lift_ft = elev_ft: max error {err_lift}; mean over 32 doors vs dist_dock_ft: max error {err_avg:.2f} ft\n"
      f"Facility-map example D01 -> A06-003-C: {ex:.1f} ft (map: 40 + 103.4 + 15.5 = 158.9 ft)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(slots.opening_height_in, bins=range(20, 136, 4), color="#2563eb")
axes[0].axvline(TALL_IN, color="r", ls="--", label=f"tall >= {TALL_IN} in")
axes[0].set_xlabel("Opening height (in)"); axes[0].set_ylabel("Slots"); axes[0].legend()
axes[0].set_title("Distribution of opening heights")
sc = axes[1].scatter(slots.x_ft, slots.y_ft, c=slots.opening_height_in, s=4, cmap="viridis")
tall = slots[slots.opening_height_in >= TALL_IN]
axes[1].scatter(tall.x_ft, tall.y_ft, s=10, facecolors="none", edgecolors="r", lw=0.6, label=f"opening >= {TALL_IN} in")
plt.colorbar(sc, ax=axes[1], label="opening height (in), any level")
axes[1].set_xlabel("x (ft)"); axes[1].set_ylabel("y (ft) into the building"); axes[1].legend(loc="upper right")
axes[1].set_title("Where the tall openings are (plan view)")
fig.suptitle("2a. Rack openings")
save(fig, "2a_openings")

# ---------------------------------------------------------------------------
# 2b + 2c. Replay the year
# ---------------------------------------------------------------------------
sidx = {k: n for n, k in enumerate(slots.slot_id)}
didx = {k: n for n, k in enumerate(docks.door_id)}
maxp = slots.max_pallet_in.values
lvl_arr = slots.level.values
aisle_arr = pd.factorize(slots.aisle)[0]
occ = np.zeros(len(slots), bool)
who = np.full(len(slots), None, dtype=object)
stock_dates = {}                      # sku -> list of receipt dates of pallets in stock (for FIFO)
for sl_, sk_, rd in zip(opening.slot_id, opening.sku_id, opening.received_date):
    occ[sidx[sl_]] = True
    who[sidx[sl_]] = sk_
    stock_dates.setdefault(sk_, []).append(rd)

ev = pd.concat([
    pd.DataFrame({"t": out.picked_at, "kind": 0, "slot": out.picked_from_slot, "sku": out.sku_id,
                  "door": out.door_id, "rdate": out.received_date, "pid": out.pallet_id}),
    pd.DataFrame({"t": inb.putaway_at, "kind": 1, "slot": inb.putaway_slot, "sku": inb.sku_id,
                  "door": inb.door_id, "rdate": inb.receipt_date, "pid": inb.pallet_id})
]).sort_values(["t", "kind"], kind="stable")

rng = np.random.default_rng(2026)
put_rows, fifo_rows, conflicts = [], [], 0
for t, kind, slot, sku, door, rdate, pid in ev.itertuples(index=False):
    j = sidx[slot]
    if kind == 0:                                           # pick
        lst = stock_dates[sku]
        fifo_rows.append((pid, sku, rdate, min(lst), len(lst)))
        lst.remove(rdate)
        conflicts += not occ[j]
        occ[j] = False
        who[j] = None
        continue
    conflicts += occ[j]                                     # put-away
    h = skus.pallet_height_in[sku]
    d = D[didx[door]]
    free = np.flatnonzero((~occ) & (maxp >= h))
    r = rng.choice(free)
    same = set(aisle_arr[who == sku])
    slack_free = maxp[free] - h

    def stats(x):
        return ((d[free] < d[x]).mean(), (slack_free < maxp[x] - h).mean(), (lvl_arr[free] < lvl_arr[x]).mean(),
                aisle_arr[x] in same, d[x], maxp[x] - h, lvl_arr[x])
    put_rows.append((pid, sku, len(free), len(same) > 0, d[free].min()) + stats(j) + stats(r))
    occ[j] = True
    who[j] = sku
    stock_dates.setdefault(sku, []).append(rdate)
print(f"\nReplay done: slot conflicts = {conflicts} (must be 0)")

names = ["pct_free_closer", "pct_free_tighter", "pct_free_lower_level", "same_aisle_as_sku", "travel_ft",
         "slack_in", "level"]
P = pd.DataFrame(put_rows, columns=["pallet_id", "sku_id", "free_fitting_slots", "sku_already_in_stock",
                                    "closest_free_ft"] + [f"actual_{n}" for n in names] + [f"random_{n}" for n in names])
P["velocity_class"] = P.sku_id.map(vel.velocity_class)
P.to_csv(f"{OUT}/2b_putaway_replay.csv", index=False)

# --- 2b tests -----------------------------------------------------------------
tests = pd.DataFrame({
    "actual choice": [P.actual_pct_free_closer.mean(), P.actual_pct_free_tighter.mean(),
                      P.actual_pct_free_lower_level.mean(), P.actual_same_aisle_as_sku[P.sku_already_in_stock].mean(),
                      P.actual_travel_ft.mean(), P.actual_slack_in.mean(), P.actual_level.mean()],
    "random free fitting slot": [P.random_pct_free_closer.mean(), P.random_pct_free_tighter.mean(),
                                 P.random_pct_free_lower_level.mean(),
                                 P.random_same_aisle_as_sku[P.sku_already_in_stock].mean(),
                                 P.random_travel_ft.mean(), P.random_slack_in.mean(), P.random_level.mean()]},
    index=["share of free fitting slots closer to the door", "share of free fitting slots with a tighter fit",
           "share of free fitting slots on a lower level", "put in an aisle already holding the SKU",
           "travel door->slot (ft)", "unused clearance above pallet (in)", "rack level"])
tests.round(3).to_csv(f"{OUT}/2b_putaway_tests.csv")
show("2b Put-away choice vs a random free slot the pallet fits in", tests.round(3))
print(f"Closest free fitting slot was on average {P.closest_free_ft.mean():.0f} ft from the door; "
      f"drivers drove {P.actual_travel_ft.mean():.0f} ft.")

# velocity
by_cls = inb.merge(slots[["slot_id", "dist_dock_ft", "level"]], left_on="putaway_slot", right_on="slot_id") \
            .groupby("velocity_class").agg(putaways=("pallet_id", "size"), mean_dist_dock_ft=("dist_dock_ft", "mean"),
                                           mean_travel_ft=("travel_ft", "mean"), mean_level=("level", "mean"))
show("2b Velocity: where each class is put away (building average dist_dock_ft = "
     f"{slots.dist_dock_ft.mean():.1f})", by_cls.round(1))
g = inb.merge(slots[["slot_id", "dist_dock_ft"]], left_on="putaway_slot", right_on="slot_id") \
       .groupby("sku_id").dist_dock_ft.mean()
rho = spearmanr(vel.pallets_per_week.reindex(g.index), g)
print(f"Spearman(pallets/week, mean slot distance) over SKUs = {rho.statistic:.3f} (p = {rho.pvalue:.2f})")

# client
inb["aisle"] = inb.putaway_slot.map(S.aisle)
ct = pd.crosstab(inb.client_id, inb.aisle)
chi2 = chi2_contingency(ct)[0]
cramer = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
share = (100 * ct.div(ct.sum(axis=1), axis=0)).round(0).astype(int)
share.index = [clients.client_name[c].replace(" Restaurants", "") for c in share.index]
share.to_csv(f"{OUT}/2b_client_by_aisle_pct.csv")
show(f"2b Client vs aisle: % of each client's put-aways per aisle (Cramer's V = {cramer:.2f})", share)
top_aisle = share.max(axis=1)
print("Largest single-aisle share per client:", top_aisle.to_dict())

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(share.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(share.shape[1])); ax.set_xticklabels(share.columns, rotation=90)
ax.set_yticks(range(share.shape[0])); ax.set_yticklabels(share.index)
plt.colorbar(im, label="% of client's put-aways")
ax.set_title(f"2b. Clients are partly zoned by aisle (Cramer's V = {cramer:.2f})")
save(fig, "2b_client_aisle_heatmap")

# height fit and unused height (every stored pallet: opening stock + put-aways)
stored = pd.concat([opening[["pallet_id", "sku_id", "slot_id"]],
                    inb[["pallet_id", "sku_id", "putaway_slot"]].rename(columns={"putaway_slot": "slot_id"})])
stored = stored.merge(slots[["slot_id", "opening_height_in", "max_pallet_in", "level", "aisle"]], on="slot_id")
stored["pallet_height_in"] = stored.sku_id.map(skus.pallet_height_in)
stored["unused_in"] = stored.opening_height_in - stored.pallet_height_in            # empty height in the opening
stored["unused_beyond_clearance_in"] = stored.max_pallet_in - stored.pallet_height_in  # beyond the 4 in rule
stored["unused_pct"] = 100 * stored.unused_in / stored.opening_height_in
stored["unused_cube_ft3"] = 40 * 48 * stored.unused_beyond_clearance_in / 1728
# weight by days in stock -> unused height actually "occupied" over the year
dwell = pd.concat([out.set_index("pallet_id").days_in_stock,
                   (END - current.set_index("pallet_id").received_date).dt.days])
stored["days"] = stored.pallet_id.map(dwell).fillna(0)
fit = pd.DataFrame({
    "per stored pallet": [stored.unused_in.mean(), stored.unused_beyond_clearance_in.mean(), stored.unused_pct.mean(),
                          stored.unused_cube_ft3.mean()],
    "weighted by days in stock": [np.average(stored.unused_in, weights=stored.days + 1e-9),
                                  np.average(stored.unused_beyond_clearance_in, weights=stored.days + 1e-9),
                                  np.average(stored.unused_pct, weights=stored.days + 1e-9),
                                  np.average(stored.unused_cube_ft3, weights=stored.days + 1e-9)]},
    index=["unused height in opening (in)", "unused height beyond 4 in clearance (in)",
           "unused share of opening height (%)", "unused cube beyond clearance (ft3)"])
show("2b Height left unused above stored pallets", fit.round(1))
tall_st = stored[stored.opening_height_in >= TALL_IN]
print(f"Pallets stored in tall openings (>= {TALL_IN} in): {len(tall_st)}; of these "
      f"{100 * (tall_st.pallet_height_in < 60).mean():.0f}% are short (< 60 in). "
      f"Pallets > 80 in: {(stored.pallet_height_in > 80).sum()} ({100 * (stored.pallet_height_in > 80).mean():.0f}%)")
show("2b Unused height by level", stored.groupby("level")[["unused_in", "unused_pct"]].mean().round(1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(stored.pallet_height_in + rng.normal(0, 0.6, len(stored)), stored.opening_height_in, s=1, alpha=0.05)
axes[0].plot([26, 88], [30, 92], "r--", label="pallet + 4 in clearance")
axes[0].set_xlabel("Pallet height (in)"); axes[0].set_ylabel("Opening height of its slot (in)"); axes[0].legend()
axes[1].hist(stored.unused_beyond_clearance_in, bins=range(0, 104, 4), color="#f59e0b")
axes[1].set_xlabel("Unused height beyond the 4 in clearance (in)"); axes[1].set_ylabel("Stored pallets")
fig.suptitle("2b. Height fit of stored pallets")
save(fig, "2b_height_fit")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, tests.iloc[:3, 0] * 100, w, label="drivers' actual choice")
ax.bar(x + w / 2, tests.iloc[:3, 1] * 100, w, label="random free fitting slot")
ax.set_xticks(x); ax.set_xticklabels(["closer to door", "tighter fit", "lower level"])
ax.set_ylabel("% of free fitting slots that were better"); ax.legend()
ax.set_title("2b. What drivers optimise: fit and level, not distance")
save(fig, "2b_putaway_tests")

# ---------------------------------------------------------------------------
# 2c. FIFO and dwell time
# ---------------------------------------------------------------------------
F = pd.DataFrame(fifo_rows, columns=["pallet_id", "sku_id", "received_date", "oldest_in_stock", "pallets_of_sku"])
F["fifo_ok"] = F.received_date <= F.oldest_in_stock
F["days_older_skipped"] = (F.received_date - F.oldest_in_stock).dt.days
print(f"\n=== 2c FIFO ===\nShipments that took the oldest pallet of their SKU: {100 * F.fifo_ok.mean():.2f}% "
      f"({(~F.fifo_ok).sum()} exceptions; days_in_stock = ship - receipt check: "
      f"{((out.ship_date - out.received_date).dt.days == out.days_in_stock).mean() * 100:.1f}% consistent)")
F.loc[~F.fifo_ok].to_csv(f"{OUT}/2c_fifo_exceptions.csv", index=False)
print("FIFO exceptions - days the shipped pallet was younger than the oldest one in stock:")
print(F.loc[~F.fifo_ok, "days_older_skipped"].describe().round(1).to_string())

dw = out.days_in_stock
dsum = pd.concat({
    "all": dw.describe(percentiles=[0.5, 0.9, 0.99]),
    **{f"class {c}": g.days_in_stock.describe(percentiles=[0.5, 0.9, 0.99]) for c, g in out.groupby("velocity_class")}},
    axis=1).T
show("2c Dwell time of shipped pallets (days)", dsum.round(1))
dcl = out.groupby("client_id").days_in_stock.describe(percentiles=[0.5, 0.9])
dcl.index = [clients.client_name[c] for c in dcl.index]
show("2c Dwell time by client (days)", dcl.round(1))
dsum.round(2).to_csv(f"{OUT}/2c_dwell_by_class.csv"); dcl.round(2).to_csv(f"{OUT}/2c_dwell_by_client.csv")

cur = current.copy()
cur["age_days_on_5jul2026"] = (END - cur.received_date).dt.days
longest = pd.concat([
    out.nlargest(15, "days_in_stock")[["pallet_id", "sku_id", "client_id", "velocity_class", "received_date",
                                       "ship_date", "days_in_stock"]].assign(status="shipped"),
    cur.nlargest(15, "age_days_on_5jul2026").assign(
        client_id=lambda d: d.sku_id.map(skus.client_id), velocity_class=lambda d: d.sku_id.map(vel.velocity_class),
        days_in_stock=lambda d: d.age_days_on_5jul2026, status="still in stock")
    [["pallet_id", "sku_id", "client_id", "velocity_class", "received_date", "days_in_stock", "status"]]])
longest.to_csv(f"{OUT}/2c_longest_stays.csv", index=False)
show("2c Pallets that stayed longest", longest.head(30))
print(f"Pallets still in stock on 5 Jul 2026 older than 180 days: {(cur.age_days_on_5jul2026 > 180).sum()}, "
      f"older than 365 days: {(cur.age_days_on_5jul2026 > 365).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data = [out.days_in_stock[out.velocity_class == c] for c in "ABCD"]
axes[0].boxplot(data, showfliers=False)
axes[0].set_xticks(range(1, 5)); axes[0].set_xticklabels(list("ABCD"))
axes[0].set_xlabel("Velocity class"); axes[0].set_ylabel("Days in stock")
cl_order = out.groupby("client_id").days_in_stock.median().sort_values().index
axes[1].boxplot([out.days_in_stock[out.client_id == c] for c in cl_order], showfliers=False)
axes[1].set_xticks(range(1, len(cl_order) + 1))
axes[1].set_xticklabels([clients.client_name[c].split()[0] for c in cl_order])
axes[1].set_ylabel("Days in stock"); axes[1].tick_params(axis="x", rotation=45)
fig.suptitle("2c. Dwell time of shipped pallets")
save(fig, "2c_dwell")

# ---------------------------------------------------------------------------
# 2d. Forklift hours per shift (modeling rules 3 and 4)
# ---------------------------------------------------------------------------
out["slot_move_min"] = move_min(out.travel_ft, out.lift_ft)
inb["slot_move_min"] = move_min(inb.travel_ft, inb.lift_ft)
day = out.groupby("ship_date").agg(pallets=("pallet_id", "size"), pick_h=("slot_move_min", "sum"))
day["pick_h"] /= 60
day["load_h"] = day.pallets * DOCK_MOVE_MIN / 60
day["total_h"] = day.pick_h + day.load_h
night = inb.groupby("receipt_date").agg(pallets=("pallet_id", "size"), putaway_h=("slot_move_min", "sum"))
night["putaway_h"] /= 60
night["unload_h"] = night.pallets * DOCK_MOVE_MIN / 60
night["total_h"] = night.putaway_h + night.unload_h
day["forklifts_needed"] = day.total_h / DAY_SHIFT_H
night["forklifts_needed"] = night.total_h / NIGHT_SHIFT_H

wk_day = day.groupby(day.index.day_name()).agg(["mean", "max"]).reindex([d for d in WEEKDAYS if d != "Saturday"])
wk_night = night.groupby(night.index.day_name()).agg(["mean", "max"]).reindex(WEEKDAYS[:5])
show("2d Day shift (picks = slot moves, loading = 1.8 min dock moves), hours", wk_day.round(1))
show("2d Night shift (put-aways = slot moves, unloading = 1.8 min dock moves), hours", wk_night.round(1))
weekly_rule4 = (out.slot_move_min.sum() + inb.slot_move_min.sum()) / 60 / 52
weekly_all = (day.total_h.sum() + night.total_h.sum()) / 52
print(f"\nRule 4 labour (picks + put-aways only): {weekly_rule4:.0f} h/week; including dock moves: {weekly_all:.0f} h/week"
      f" -> {weekly_all / DRIVER_H_WEEK:.1f} full-time drivers at {DRIVER_H_WEEK:.0f} h/week (no allowance for breaks/waiting)")
print(f"Average minutes per slot move: pick {out.slot_move_min.mean():.2f}, put-away {inb.slot_move_min.mean():.2f}; "
      f"travel share of slot-move time: {100 * (2 * out.travel_ft / SPEED_FT_MIN).sum() / out.slot_move_min.sum():.0f}%")
day.round(2).to_csv(f"{OUT}/2d_day_shift_hours.csv"); night.round(2).to_csv(f"{OUT}/2d_night_shift_hours.csv")
pd.DataFrame({"busiest day shift": [day.total_h.idxmax().date(), day.total_h.max()],
              "busiest night shift": [night.total_h.idxmax().date(), night.total_h.max()]},
             index=["date", "forklift hours"]).to_csv(f"{OUT}/2d_busiest_days.csv")
print(f"Busiest day shift: {day.total_h.idxmax().date()} ({day.total_h.max():.1f} h); "
      f"busiest night: {night.total_h.idxmax().date()} ({night.total_h.max():.1f} h) -> use these in the Digital Twin")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(7); w = 0.38
dmean = wk_day[("total_h", "mean")].reindex(WEEKDAYS).fillna(0)
nmean = wk_night[("total_h", "mean")].reindex(WEEKDAYS).fillna(0)
ax.bar(x - w / 2, dmean, w, label="day shift (pick + load)", color="#2563eb")
ax.bar(x + w / 2, nmean, w, label="night shift (unload + put away)", color="#f59e0b")
ax.set_xticks(x); ax.set_xticklabels([d[:3] for d in WEEKDAYS]); ax.set_ylabel("Forklift hours")
ax.set_title("2d. Average forklift hours per shift (modeling rules)"); ax.legend()
save(fig, "2d_forklift_hours")

# Digital Twin experiment log (fill in by hand while running the Twin)
pd.DataFrame({"date": [day.total_h.idxmax().date()] * 6 + [night.total_h.idxmax().date()] * 6,
              "shift": ["day"] * 6 + ["night"] * 6,
              "forklifts": list(range(4, 10)) + list(range(6, 12)),
              "first_truck_waiting_at": "", "max_truck_wait_min": "", "night_shift_finish": "",
              "forklift_utilisation_%": "", "notes": ""}).to_csv(f"{OUT}/2d_digital_twin_log_TEMPLATE.csv", index=False)
print(f"\nDigital Twin log template written to {OUT}/2d_digital_twin_log_TEMPLATE.csv")
print("Done. All tables and charts are in", OUT)
