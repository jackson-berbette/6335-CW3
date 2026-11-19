import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from IPython.display import display

# read relevant input files
inbound_df = pd.read_csv("Data/inbound.csv", parse_dates=["receipt_date"])
outbound_df = pd.read_csv("Data/outbound.csv", parse_dates=["ship_date"])
opening_stock_df = pd.read_csv("Data/opening_stock.csv")
slots_df = pd.read_csv("Data/slots.csv")
skus_df = pd.read_csv("Data/skus.csv")
velocity_stock_df = pd.read_csv("Output/sku_velocity_and_stock.csv", index_col=0) # Task 1f: pallets_yr, avg_stock, peak_stock

TALL_IN = 72 # tall pallet: taller than 72 in (as in 3b)
DAYS = 364


# -------- Part a --------
# SKUs with the Phase 2 reporting classes: A = first 50% of pallets shipped, B = next 30%, C = rest
sku_df = (
    velocity_stock_df[["sku_id", "pallets_yr", "avg_stock", "peak_stock"]]
    .merge(skus_df[["sku_id", "pallet_height_in"]], on="sku_id", how="left")
    .astype({"pallets_yr": int, "peak_stock": int})  # Task 1f saves these as decimals (e.g. 111.0)
    .sort_values(["pallets_yr", "sku_id"], ascending=[False, True])
    .reset_index(drop=True)
)
share_before = sku_df["pallets_yr"].cumsum().shift(fill_value=0) / sku_df["pallets_yr"].sum()
sku_df["p2_class"] = np.select([share_before < 0.5, share_before < 0.8], ["A", "B"], default="C")
sku_df["tall"] = sku_df["pallet_height_in"] > TALL_IN

# expected stay of a pallet of each SKU (days) = average stock / pallets shipped per day
sku_df["expected_stay_days"] = sku_df["avg_stock"] / (sku_df["pallets_yr"] / DAYS)

class_df = (
    sku_df.groupby("p2_class")
    .agg(skus=("sku_id", "size"),
         pallets_yr=("pallets_yr", "sum"),
         avg_stock=("avg_stock", "sum"),
         tall_skus=("tall", "sum"),
         median_stay_days=("expected_stay_days", "median"))
    .reset_index()
)
class_df["share_pallets"] = class_df["pallets_yr"] / class_df["pallets_yr"].sum()


# fact 1: is duration of stay different information from velocity?
shipped = sku_df["pallets_yr"] > 0
stay_vs_velocity = spearmanr(sku_df.loc[shipped, "pallets_yr"], sku_df.loc[shipped, "expected_stay_days"]).statistic

days_in_stock = outbound_df["days_in_stock"]
sku_mean_stay = outbound_df.groupby("sku_id")["days_in_stock"].transform("mean")
within_sku_share = ((days_in_stock - sku_mean_stay) ** 2).sum() / ((days_in_stock - days_in_stock.mean()) ** 2).sum()

# fact 2: heights and distances
tall_slots = (slots_df["max_pallet_in"] > TALL_IN).sum()
unusable_slots = (slots_df["max_pallet_in"] < sku_df["pallet_height_in"].min()).sum()
tall_flow_share = (sku_df["pallets_yr"] * sku_df["tall"]).sum() / sku_df["pallets_yr"].sum()

# fact 3: dedicated storage needs every SKU's own peak
dedicated_slots = sku_df["peak_stock"].sum()

# daily stock per SKU (as in Task 1f), to size the illustrative zones by class peak
days = pd.date_range(outbound_df["ship_date"].min(), outbound_df["ship_date"].max())
sku_ids = sku_df["sku_id"]
received = (inbound_df.groupby(["receipt_date", "sku_id"]).size().unstack(fill_value=0)
            .reindex(index=days, columns=sku_ids, fill_value=0))
shipped_daily = (outbound_df.groupby(["ship_date", "sku_id"]).size().unstack(fill_value=0)
                 .reindex(index=days, columns=sku_ids, fill_value=0))
opening = opening_stock_df["sku_id"].value_counts().reindex(sku_ids, fill_value=0)
stock_df = opening + received.cumsum() - shipped_daily.cumsum()

class_of_sku = sku_df.set_index("sku_id")["p2_class"]
class_peak = stock_df.T.groupby(class_of_sku).sum().T.max()
total_avg = stock_df.sum(axis=1).mean()
total_peak = stock_df.sum(axis=1).max()

# illustration: which pallets each policy puts in each slot, slots ranked by distance from the dock
slots_ranked = slots_df.sort_values(["dist_dock_ft", "level", "slot_id"]).reset_index(drop=True)
n_slots = len(slots_ranked)
rank_of_slot = pd.Series(slots_ranked.index, index=slots_ranked["slot_id"])


# today (reference): mix of classes historically put away, per block of 250 slots ranked by distance
BLOCK = 250
today_mix_df = (
    inbound_df[["putaway_slot", "sku_id"]]
    .assign(p2_class=lambda d: d["sku_id"].map(class_of_sku),
            block=lambda d: d["putaway_slot"].map(rank_of_slot) // BLOCK)
    .groupby(["block", "p2_class"]).size().unstack(fill_value=0)
)
today_mix_df = today_mix_df.div(today_mix_df.sum(axis=1), axis=0)


# product-dedicated-COL: each SKU owns its peak in slots; lowest peak per move goes nearest
dedicated_order = (
    sku_df.assign(moves=lambda d: d["pallets_yr"] * 2)
    .assign(slots_per_move=lambda d: d["peak_stock"] / d["moves"].replace(0, np.nan))
    .sort_values("slots_per_move", na_position="last")
)
dedicated_strip = np.repeat(dedicated_order["p2_class"].to_numpy(), dedicated_order["peak_stock"].to_numpy())

# product-class-COL: A nearest, then B, then C, each zone sized at its class peak
class_strip = np.repeat(["A", "B", "C"], [class_peak["A"], class_peak["B"], n_slots - class_peak["A"] - class_peak["B"]])

# product-shared-COL: any pallet takes the nearest free slot; average stock fills the nearest slots
shared_strip = np.where(np.arange(n_slots) < total_avg, "used", np.where(np.arange(n_slots) < total_peak, "peak", "empty"))

# duration-of-stay-target: pallets expected to leave soonest nearest, up to the peak
stay_strip = np.where(np.arange(n_slots) < total_peak, np.arange(n_slots) / total_peak, np.nan)


# validate
assert len(sku_df) == len(skus_df) == 1189
assert class_df["skus"].sum() == len(skus_df)
assert np.isclose(total_avg, 5540, atol=1) and total_peak == 6260
assert len(dedicated_strip) == dedicated_slots

display(class_df.round(3))
print(f"Phase 2 classes: A {class_df.loc[0, 'skus']} SKUs, B {class_df.loc[1, 'skus']}, C {class_df.loc[2, 'skus']}")
print(f"Class peak stock: A {class_peak['A']}, B {class_peak['B']}, C {class_peak['C']}; all SKUs {total_peak} (average {total_avg:,.0f})")
print(f"Dedicated storage needs {dedicated_slots:,} slots (sum of SKU peaks) for {n_slots:,} slots")
sum_of_peaks = sku_df.groupby("p2_class")["peak_stock"].sum()
print(f"Class zones (class peak vs sum of its SKUs' peaks): A {class_peak['A']} vs {sum_of_peaks['A']}, "
      f"B {class_peak['B']} vs {sum_of_peaks['B']}, C zone = remaining {n_slots - class_peak['A'] - class_peak['B']} slots")
top_sku = sku_df.iloc[0]
print(f"Fastest SKU {top_sku['sku_id']}: {top_sku['pallets_yr']:,} pallets/yr, peak {top_sku['peak_stock']:.0f} pallets")
print(f"Median expected stay: A {class_df.loc[0, 'median_stay_days']:.1f}, B {class_df.loc[1, 'median_stay_days']:.1f}, "
      f"C {class_df.loc[2, 'median_stay_days']:.1f} days")
print(f"Tall pallets (> {TALL_IN} in): {sku_df['tall'].sum()} SKUs, {tall_flow_share:.1%} of pallets shipped; "
      f"slots that take them: {tall_slots:,}; slots too low for any pallet: {unusable_slots}")
print(f"Slot distance from the dock: {slots_df['dist_dock_ft'].min():.0f}-{slots_df['dist_dock_ft'].max():.0f} ft "
      f"(average {slots_df['dist_dock_ft'].mean():.0f} ft)")
print(f"Expected stay vs velocity, rank correlation across SKUs: {stay_vs_velocity:.2f}")
print(f"Share of pallet stay variation within the same SKU: {within_sku_share:.1%}")


# height groups: slots by the tallest pallet they take, against peak stock of pallets of that height
HEIGHT_BINS = [0, 48, 72, 200]
HEIGHT_LABELS = ["up to 48 in", "49-72 in", "over 72 in"]
slots_df["height_group"] = pd.cut(slots_df["max_pallet_in"], HEIGHT_BINS, labels=HEIGHT_LABELS)
sku_df["height_group"] = pd.cut(sku_df["pallet_height_in"], HEIGHT_BINS, labels=HEIGHT_LABELS)
height_of_sku = sku_df.set_index("sku_id")["height_group"]

height_df = pd.DataFrame({
    "slots": slots_df.loc[slots_df["max_pallet_in"] >= sku_df["pallet_height_in"].min(), "height_group"]
             .value_counts().reindex(HEIGHT_LABELS),
    "peak_pallets": stock_df.T.groupby(height_of_sku, observed=False).sum().T.max().reindex(HEIGHT_LABELS),
})
display(height_df)


# figure: where each policy puts pallets along the slots ranked by distance
colors = {"A": "#2a78d6", "B": "#f0a33a", "C": "#c9c9c9", "empty": "#ffffff", "used": "#5b5b5b", "peak": "#a8a8a8"}
rows = [("Product-Dedicated-COL", dedicated_strip),
        ("Product-Class-COL", class_strip), ("Product-Shared-COL", shared_strip)]

fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [3.2, 1]})

# today: each block's bar split by the share of A, B and C pallets put away there
y_today = len(rows) + 1
for block, mix in today_mix_df.iterrows():
    bottom = y_today - 0.35
    for cls in ["A", "B", "C"]:
        ax.bar(block * BLOCK, 0.7 * mix.get(cls, 0), width=min(BLOCK, n_slots - block * BLOCK), bottom=bottom, align="edge",
               color=colors[cls], linewidth=0)
        bottom += 0.7 * mix.get(cls, 0)
ax.text(-80, y_today, "Today (reference)", ha="right", va="center", fontsize=9)

for r, (label, strip) in enumerate(rows):
    y = len(rows) - r
    runs = pd.Series(strip).ne(pd.Series(strip).shift()).cumsum()
    for _, run in pd.Series(strip).groupby(runs):
        ax.barh(y, len(run), left=run.index[0], height=0.7, color=colors[run.iloc[0]], linewidth=0)
    ax.text(-80, y, label, ha="right", va="center", fontsize=9)
    if len(strip) > n_slots:
        ax.barh(y, len(strip) - n_slots, left=n_slots, height=0.7, color="#d64545", hatch="///", edgecolor="white", linewidth=0)
        ax.text(len(strip) + 40, y, f"{len(strip) - n_slots:,} short", va="center", fontsize=8, color="#d64545")

stay_values = np.nan_to_num(stay_strip, nan=1.0)
ax.imshow(np.where(np.isnan(stay_strip), np.nan, stay_values)[None, :], extent=(0, n_slots, -0.35, 0.35),
          aspect="auto", cmap="Blues_r", vmin=-0.3, vmax=1.3)
ax.text(-80, 0, "Duration-of-Stay-Target", ha="right", va="center", fontsize=9)
ax.text(total_peak + 60, 0, "dark = shortest stay", va="center", fontsize=8)

ax.axvline(n_slots, color="black", lw=1)
ax.text(n_slots, len(rows) + 1.55, "8,365 slots", ha="center", fontsize=8)
ax.set_xlim(0, max(len(dedicated_strip), n_slots) + 700)
ax.set_ylim(-0.6, len(rows) + 1.8)
ax.set_yticks([])
ax.set_xlabel("Slots ranked by distance from the dock (nearest on the left)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(handles=[Patch(color=colors["A"], label="Class A"), Patch(color=colors["B"], label="Class B"),
                   Patch(color=colors["C"], label="Class C"), Patch(color=colors["used"], label="Any SKU, average stock"),
                   Patch(color=colors["peak"], label="Any SKU, peak only"),
                   Patch(facecolor="#d64545", hatch="///", edgecolor="white", label="No room")],
          loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=6, fontsize=8, frameon=False)
ax.set_title("A. Where each policy puts pallets", loc="left", fontweight="bold", fontsize=10)


# panel B: slots by the tallest pallet they take vs peak stock of pallets of that height
x = np.arange(len(HEIGHT_LABELS))
axh.bar(x - 0.2, height_df["slots"], width=0.4, color="#5b5b5b", label="Slots")
axh.bar(x + 0.2, height_df["peak_pallets"], width=0.4, color="#2a78d6", label="Peak pallets in stock")
for xi, (n_s, n_p) in enumerate(zip(height_df["slots"], height_df["peak_pallets"])):
    axh.text(xi - 0.2, n_s + 60, f"{n_s:,}", ha="center", fontsize=8)
    axh.text(xi + 0.2, n_p + 60, f"{n_p:,}", ha="center", fontsize=8)
axh.set_xticks(x, HEIGHT_LABELS, fontsize=8)
axh.set_xlabel("Pallet height a slot takes / pallet height")
axh.set_ylim(0, height_df.max().max() * 1.35)
axh.spines[["top", "right"]].set_visible(False)
axh.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
axh.set_title("B. Openings vs pallets by height", loc="left", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig("Output/Task4a_Policies.png", dpi=150, bbox_inches="tight")
plt.show()

sku_df.to_csv("Output/Task4a_SKU_Classes.csv", index=False)
class_df.to_csv("Output/Task4a_Class_Summary.csv", index=False)

