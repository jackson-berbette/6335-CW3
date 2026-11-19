import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display

# read relevant input files
inbound_df = pd.read_csv("Data/inbound.csv", usecols=["receipt_date", "sku_id", "pallet_id", "putaway_slot"],
                         parse_dates=["receipt_date"])
outbound_df = pd.read_csv("Data/outbound.csv", usecols=["ship_date", "sku_id", "pallet_id"], parse_dates=["ship_date"])
opening_stock_df = pd.read_csv("Data/opening_stock.csv")
slots_df = pd.read_csv("Data/slots.csv")
skus_df = pd.read_csv("Data/skus.csv")
velocity_stock_df = pd.read_csv("Output/sku_velocity_and_stock.csv", index_col=0)  # Task 1f: pallets_yr, avg_stock, peak_stock

TALL_IN = 72  # tall pallet / tall opening: over 72 in (as in 3b and 4a)
DAYS = 364
HEIGHT_BINS = [0, 48, 72, 200]
HEIGHT_LABELS = ["up to 48 in", "49-72 in", "over 72 in"]
STAY_BINS = [0, 14, 30, 60, np.inf]  # duration-of-stay targets, expected days in stock
STAY_LABELS = ["T1 (up to 14 d)", "T2 (14-30 d)", "T3 (30-60 d)", "T4 (over 60 d)"]
N_SLOTS = len(slots_df)


# SKUs: classes and expected stay
sku_df = (
    velocity_stock_df[["sku_id", "pallets_yr", "avg_stock", "peak_stock"]]
    .merge(skus_df[["sku_id", "pallet_height_in"]], on="sku_id", how="left")
    .astype({"pallets_yr": int, "peak_stock": int})
    .sort_values(["pallets_yr", "sku_id"], ascending=[False, True])
    .reset_index(drop=True)
)
share_before = sku_df["pallets_yr"].cumsum().shift(fill_value=0) / sku_df["pallets_yr"].sum()
sku_df["p2_class"] = np.select([share_before < 0.5, share_before < 0.8], ["A", "B"], default="C")
sku_df["height_group"] = pd.cut(sku_df["pallet_height_in"], HEIGHT_BINS, labels=HEIGHT_LABELS)
# expected stay of a pallet (Little's law): average stock / pallets shipped per day; never shipped -> longest target
sku_df["expected_stay_days"] = sku_df["avg_stock"] / (sku_df["pallets_yr"] / DAYS).replace(0, np.nan)
sku_df["dos_target"] = pd.cut(sku_df["expected_stay_days"].fillna(np.inf), STAY_BINS, labels=STAY_LABELS)
sku = sku_df.set_index("sku_id")


# every pallet of the year: in stock at the end of day d when in_day <= d < out_day
days = pd.date_range(outbound_df["ship_date"].min(), outbound_df["ship_date"].max())
n_days = len(days)
day_index = pd.Series(np.arange(n_days), index=days)

pallet_df = pd.concat([
    opening_stock_df.rename(columns={"slot_id": "slot"}).assign(in_day=0)[["pallet_id", "sku_id", "slot", "in_day"]],
    inbound_df.rename(columns={"putaway_slot": "slot"})
    .assign(in_day=lambda d: d["receipt_date"].map(day_index))[["pallet_id", "sku_id", "slot", "in_day"]],
], ignore_index=True)
ship_day = outbound_df.set_index("pallet_id")["ship_date"].map(day_index)
pallet_df["out_day"] = pallet_df["pallet_id"].map(ship_day).fillna(n_days).astype(int)
pallet_df = pallet_df.join(sku[["pallet_height_in", "p2_class", "dos_target"]], on="sku_id")

# pallets in stock at the end of each day
# per: code (rows) and day (columns)
def daily_stock(codes, n_codes):
    m = np.zeros((n_codes, n_days + 1), dtype=int)
    np.add.at(m, (codes, pallet_df["in_day"].to_numpy()), 1)
    np.add.at(m, (codes, pallet_df["out_day"].to_numpy()), -1)
    return m.cumsum(axis=1)[:, :n_days]


total_stock = daily_stock(np.zeros(len(pallet_df), dtype=int), 1)[0]
peak_day = int(total_stock.argmax())

# stock per (zone, exact pallet height): the basis of every sizing below
heights = np.sort(sku_df["pallet_height_in"].unique())
height_code = np.searchsorted(heights, pallet_df["pallet_height_in"].to_numpy())


def stock_by_zone_height(zone_of_pallet, zones):
    zone_code = pd.Categorical(zone_of_pallet, categories=zones).codes
    m = daily_stock(zone_code * len(heights) + height_code, len(zones) * len(heights))
    return {z: m[i * len(heights):(i + 1) * len(heights)] for i, z in enumerate(zones)}


# slots ranked by distance from the dock (nearest, then lowest level first)
usable = slots_df["max_pallet_in"] >= heights.min()
slots_ranked = (slots_df[usable].sort_values(["dist_dock_ft", "level", "slot_id"]).reset_index(drop=True))
slots_ranked["height_group"] = pd.cut(slots_ranked["max_pallet_in"], HEIGHT_BINS, labels=HEIGHT_LABELS)
slot_max = slots_ranked["max_pallet_in"].to_numpy()
slot_tall = slot_max > TALL_IN

# peak pallets held at once that need a slot taking height h, for every pallet height h
def need_vector(stock_h, protect):
    return np.array([need for _, _, need in conditions(stock_h, protect)])


def supply_vector(protect):
    return np.array([eligible.sum() for _, eligible, _ in conditions(np.zeros((len(heights), 1)), protect)])

# nested height conditions of a zone
def conditions(stock_h, protect):
    conds = []
    for i, h in enumerate(heights):
        if protect and h <= TALL_IN:
            need = stock_h[i:np.searchsorted(heights, TALL_IN, side="right")].sum(axis=0).max()
            eligible = (slot_max >= h) & ~slot_tall
        else:
            need = stock_h[i:].sum(axis=0).max()
            eligible = slot_max >= h
        conds.append((h, eligible, need))
    return conds

# Smallest set of nearest free slots that meets all conditions of a zone: slot mask and slots short
def footprint(stock_h, free, protect):
    k_needed, short = 0, 0
    for h, eligible, need in conditions(stock_h, protect):
        cum = np.cumsum(eligible & free)
        if need > cum[-1]:
            short = max(short, need - cum[-1])
            k_needed = len(free)
        elif need > 0:
            k_needed = max(k_needed, int(np.searchsorted(cum, need)) + 1)
    zone = free.copy()
    zone[k_needed:] = False
    return zone, short


def peak_by_height_group(stock_h):
    group = pd.cut(heights, HEIGHT_BINS, labels=HEIGHT_LABELS)
    return {g: stock_h[group == g].sum(axis=0).max() for g in HEIGHT_LABELS}


rows = []
needs = {}  # (policy, tall_protected) -> summed need for every pallet height

# zones filled in order from the nearest free slots it needs
def size_zones(policy, zone_of_pallet, zones, protect):
    stock = stock_by_zone_height(zone_of_pallet, zones)
    free = np.ones(len(slots_ranked), dtype=bool)
    for z in zones:
        zone_mask, short = footprint(stock[z], free, protect)
        free &= ~zone_mask
        slot_groups = slots_ranked.loc[zone_mask, "height_group"].value_counts()
        rows.append({"policy": policy, "tall_protected": protect, "zone": z,
                     "peak_pallets": stock[z].sum(axis=0).max(),
                     **{f"peak_pallets {g}": n for g, n in peak_by_height_group(stock[z]).items()},
                     "slots": zone_mask.sum(),
                     **{f"slots {g}": slot_groups.get(g, 0) for g in HEIGHT_LABELS},
                     "slots_short": short,
                     "dist_from_ft": slots_ranked.loc[zone_mask, "dist_dock_ft"].min(),
                     "dist_to_ft": slots_ranked.loc[zone_mask, "dist_dock_ft"].max()})
        needs[(policy, protect)] = needs.get((policy, protect), 0) + need_vector(stock[z], protect)


# how current slots are occupied on the peak day
on_peak = pallet_df[(pallet_df["in_day"] <= peak_day) & (pallet_df["out_day"] > peak_day)]
on_peak = on_peak.join(slots_df.set_index("slot_id")["max_pallet_in"], on="slot")
on_peak["slot_group"] = pd.cut(on_peak["max_pallet_in"], HEIGHT_BINS, labels=HEIGHT_LABELS)
on_peak["pallet_group"] = pd.cut(on_peak["pallet_height_in"], HEIGHT_BINS, labels=HEIGHT_LABELS)
today_fit_df = pd.crosstab(on_peak["pallet_group"], on_peak["slot_group"], margins=True)
slots_used_year = pallet_df["slot"].nunique()
short_in_tall_today = ((on_peak["max_pallet_in"] > TALL_IN) & (on_peak["pallet_height_in"] <= TALL_IN)).sum()

for protect in [True, False]:
    # Product-Shared-COL: one zone, any pallet takes the nearest free slot it fits
    size_zones("Product-Shared-COL", np.repeat("All SKUs", len(pallet_df)), ["All SKUs"], protect)
    # Product-Class-COL: A nearest, then B, then C, each zone sized at its class peak
    size_zones("Product-Class-COL", pallet_df["p2_class"], ["A", "B", "C"], protect)
    # Duration-of-Stay-Target: shortest expected stay in the nearest target
    size_zones("Duration-of-Stay-Target", pallet_df["dos_target"].astype(str), STAY_LABELS, protect)


# product-Dedicated-COL:
dedicated_order = (sku_df.assign(slots_per_move=lambda d: d["peak_stock"] / (2 * d["pallets_yr"]).replace(0, np.nan))
                   .sort_values("slots_per_move", na_position="last"))
sku_daily = daily_stock(pd.Categorical(pallet_df["sku_id"], categories=sku_df["sku_id"]).codes, len(sku_df))
sku_daily = pd.DataFrame(sku_daily.T, columns=sku_df["sku_id"])

# nearest eligible free slots to each SKU
def assign_dedicated(slots_per_sku, protect):
    free = np.ones(len(slots_ranked), dtype=bool)
    short = {}
    for _, s in dedicated_order.iterrows():
        need = slots_per_sku[s["sku_id"]]
        tall = s["pallet_height_in"] > TALL_IN
        eligible = free & (slot_max >= s["pallet_height_in"]) & (slot_tall if (protect and tall) else
                                                                  (~slot_tall if protect else True))
        idx = np.flatnonzero(eligible)[:need]
        free[idx] = False
        short[s["sku_id"]] = need - len(idx)
    return pd.Series(short), free


sku_heights_code = np.searchsorted(heights, sku_df["pallet_height_in"].to_numpy())
for protect in [True, False]:
    peak_h = np.zeros((len(heights), 1))
    np.add.at(peak_h[:, 0], sku_heights_code, sku_df["peak_stock"].to_numpy())
    needs[("Product-Dedicated-COL", protect)] = need_vector(peak_h, protect)
    short, free = assign_dedicated(sku_df.set_index("sku_id")["peak_stock"], protect)
    for cls in ["A", "B", "C"]:
        members = sku_df[sku_df["p2_class"] == cls]
        rows.append({"policy": "Product-Dedicated-COL", "tall_protected": protect, "zone": f"{cls} SKUs ({len(members)})",
                     "peak_pallets": members["peak_stock"].sum(),
                     **{f"peak_pallets {g}": members.loc[members["height_group"] == g, "peak_stock"].sum()
                        for g in HEIGHT_LABELS},
                     "slots": members["peak_stock"].sum() - short[members["sku_id"]].sum(),
                     **{f"slots {g}": np.nan for g in HEIGHT_LABELS},
                     "slots_short": short[members["sku_id"]].sum(),
                     "dist_from_ft": np.nan, "dist_to_ft": np.nan})

# how far below its own peak could each SKU be sized to fit, and what overflows then
dedicated_sensitivity = []
for q in [1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75]:
    alloc = np.ceil(sku_daily.quantile(q)).astype(int)
    overflow = (sku_daily - alloc).clip(lower=0).sum(axis=1)
    short, _ = assign_dedicated(alloc, True)
    dedicated_sensitivity.append({"sized_at_stock_percentile": q, "dedicated_slots": alloc.sum(),
                                  "slots_short_tall_protected": short.sum(),
                                  "sku_days_over_allocation": (sku_daily > alloc).to_numpy().mean(),
                                  "peak_overflow_pallets": overflow.max()})
dedicated_sensitivity_df = pd.DataFrame(dedicated_sensitivity)


# summary: feasibility and margin in 8,365 slots
# slots required = sum of the zones' own peaks (a zone cannot borrow another zone's empty slots);
# feasibility = for every pallet height h, slots that take h >= pallets needing them at the zones' peaks;
# margin = the smallest spare over all heights, and the pallet height where it binds
zone_df = pd.DataFrame(rows).rename(columns={"slots": "zone_slots_nearest_first"})
summary = []
needs[("Today (reference)", False)] = need_vector(stock_by_zone_height(np.repeat("All", len(pallet_df)), ["All"])["All"], False)
for (policy, protect), need in needs.items():
    spare = supply_vector(protect) - need
    z = zone_df[(zone_df["policy"] == policy) & (zone_df["tall_protected"] == protect)]
    group_need = {g: z[f"peak_pallets {g}"].sum() for g in HEIGHT_LABELS} if len(z) else \
        peak_by_height_group(stock_by_zone_height(np.repeat("All", len(pallet_df)), ["All"])["All"])
    summary.append({"policy": policy, "tall_protected": protect, "zones": max(len(z), 1),
                    "slots_required": z["peak_pallets"].sum() if len(z) else total_stock.max(),
                    **{f"required {g}": n for g, n in group_need.items()},
                    "tall_slots_spare": int(spare[heights > TALL_IN].min()),
                    "short_slots_spare": int(spare[heights <= TALL_IN].min()),
                    "margin_slots": int(spare.min()), "binding_pallet_height_in": heights[spare.argmin()],
                    "feasible": spare.min() >= 0,
                    "zone_slots_nearest_first": z["zone_slots_nearest_first"].sum() if len(z) and z["dist_to_ft"].notna().all() else np.nan,
                    "farthest_zone_slot_ft": z["dist_to_ft"].max()})
summary_df = pd.DataFrame(summary)
summary_df["margin_pct_of_required"] = summary_df["margin_slots"] / summary_df["slots_required"]
order = ["Today (reference)", "Product-Dedicated-COL", "Product-Class-COL", "Product-Shared-COL", "Duration-of-Stay-Target"]
summary_df = summary_df.sort_values(["policy", "tall_protected"], key=lambda c: c.map(order.index) if c.name == "policy" else ~c)
summary_df = summary_df.reset_index(drop=True)


# validate
assert len(pallet_df) == len(inbound_df) + len(opening_stock_df)
assert total_stock.max() == 6260 and np.isclose(total_stock.mean(), 5540, atol=1)
assert today_fit_df.loc["All", "All"] == total_stock.max()
assert (pallet_df["out_day"] >= pallet_df["in_day"]).all()
assert sku_daily.max().sum() == sku_df["peak_stock"].sum() == 9747
shared = zone_df[(zone_df["policy"] == "Product-Shared-COL") & ~zone_df["tall_protected"]].iloc[0]
assert shared["peak_pallets"] == total_stock.max() and shared["zone_slots_nearest_first"] >= shared["peak_pallets"]
assert (needs[("Product-Shared-COL", False)] == needs[("Today (reference)", False)]).all()
assert summary_df.loc[summary_df["policy"] == "Product-Dedicated-COL", "slots_required"].eq(9747).all()

pd.set_option("display.width", 200)
display(summary_df.round(3))
display(zone_df.drop(columns=[c for c in zone_df if c.startswith("slots ") and zone_df[c].isna().all()]).round(0))
display(today_fit_df)
display(dedicated_sensitivity_df.round(3))

print(f"Usable slots: {len(slots_ranked):,} of {N_SLOTS:,} ({N_SLOTS - len(slots_ranked)} too low for any pallet); "
      f"tall openings: {slot_tall.sum():,}")
print(f"Peak stock {total_stock.max():,} pallets on {days[peak_day].date()}; average {total_stock.mean():,.0f}")
print(f"Today on the peak day: {short_in_tall_today:,} pallets up to 72 in sit in tall openings; "
      f"{slots_used_year:,} distinct slots used during the year")
for _, r in summary_df.iterrows():
    tag = "tall protected" if r["tall_protected"] else "heights mixed"
    state = (f"margin {r['margin_slots']:,} slots ({r['margin_pct_of_required']:.1%} of need)" if r["feasible"]
             else f"INFEASIBLE, {-r['margin_slots']:,} slots short") + f", binding at {r['binding_pallet_height_in']} in pallets"
    print(f"{r['policy']:<24} {tag:<15} requires {r['slots_required']:>6,} slots -> {state}")


zone_colors = {"All SKUs": "#5b5b5b", "A": "#2a78d6", "B": "#f0a33a", "C": "#c9c9c9",
               STAY_LABELS[0]: "#08306b", STAY_LABELS[1]: "#2171b5", STAY_LABELS[2]: "#6baed6", STAY_LABELS[3]: "#c6dbef"}
n_usable = len(slots_ranked)
fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
for r, policy in enumerate(order):
    y = len(order) - 1 - r
    protect = policy != "Today (reference)"
    s = summary_df[(summary_df["policy"] == policy) & (summary_df["tall_protected"] == protect)].iloc[0]
    z = zone_df[(zone_df["policy"] == policy) & (zone_df["tall_protected"] == protect)]
    segments = [(n, zone_colors.get(str(k)[0] if policy == "Product-Dedicated-COL" else k, "#5b5b5b"))
                for k, n in zip(z["zone"], z["peak_pallets"])] or [(total_stock.max(), "#5b5b5b")]
    left = 0
    for n, color in segments:
        ax.barh(y, n, left=left, color=color, height=0.6, edgecolor="white", linewidth=0.8)
        left += n
    if s["margin_slots"] < 0:
        ax.barh(y, -s["margin_slots"], left=left + s["margin_slots"], height=0.6, color="none",
                hatch="///", edgecolor="#d64545", linewidth=0)
    ax.text(max(left, n_usable) + 80, y, f"{s['margin_slots']:+,}", va="center", fontsize=8,
            color="#d64545" if s["margin_slots"] < 0 else "black")
ax.axvline(n_usable, color="black", lw=1)
ax.text(n_usable, len(order) - 0.45, f"{n_usable:,} usable slots", ha="center", fontsize=8)
ax.set_yticks(range(len(order)), order[::-1], fontsize=9)
ax.set_xlim(0, summary_df["slots_required"].max() + 1000)
ax.set_ylim(-0.6, len(order) - 0.3)
ax.set_xlabel("Slots required (sum of zone peaks)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Patch(color=zone_colors[k], label=f"Class {k}") for k in ["A", "B", "C"]]
          + [Patch(color=zone_colors[k], label=k.split(" ")[0]) for k in STAY_LABELS]
          + [Patch(facecolor="none", hatch="///", edgecolor="#d64545", label="No room")],
          loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=8, fontsize=8, frameon=False)
ax.set_title("A. Slots required and margin (Tall openings protected)", loc="left", fontweight="bold", fontsize=10)

groups = ["Up to 72 in", "Over 72 in"]
avail = [(~slot_tall).sum(), slot_tall.sum()]
bars = [("Available", avail, "#bdbdbd")]
for policy, color in [("Product-Dedicated-COL", "#d64545"), ("Product-Class-COL", "#2a78d6"),
                      ("Product-Shared-COL", "#5b5b5b"), ("Duration-of-Stay-Target", "#6baed6")]:
    s = summary_df[(summary_df["policy"] == policy) & summary_df["tall_protected"]].iloc[0]
    bars.append((policy.replace("Product-", "").replace("Duration-of-Stay", "DoS"),
                 [s[f"required {HEIGHT_LABELS[0]}"] + s[f"required {HEIGHT_LABELS[1]}"], s[f"required {HEIGHT_LABELS[2]}"]],
                 color))
x = np.arange(len(groups))
w = 0.8 / len(bars)
for i, (label, values, color) in enumerate(bars):
    axh.bar(x - 0.4 + w * (i + 0.5), values, width=w, color=color, label=label)
axh.set_xticks(x, groups, fontsize=8)
axh.set_xlabel("Pallet height / opening")
axh.set_ylabel("Slots")
axh.spines[["top", "right"]].set_visible(False)
axh.legend(fontsize=7, frameon=False)
axh.set_title("B. Required vs available by height", loc="left", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig("Output/Task4b_Slots_Required.png", dpi=150, bbox_inches="tight")
plt.show()

summary_df.to_csv("Output/Task4b_Feasibility.csv", index=False)
zone_df.to_csv("Output/Task4b_Slots_By_Zone_Height.csv", index=False)
today_fit_df.to_csv("Output/Task4b_Today_Peak_Day_Fit.csv")
dedicated_sensitivity_df.to_csv("Output/Task4b_Dedicated_Sensitivity.csv", index=False)
